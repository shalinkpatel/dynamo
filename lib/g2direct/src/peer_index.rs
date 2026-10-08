// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend-neutral, eventually-consistent physical KV replica directory.
//!
//! The directory records where bytes were last advertised, but never proves
//! residency. Consumers must acquire a backend-specific source lease before
//! reading a location. It deliberately has no ownership or placement policy.

use dashmap::DashMap;
use dynamo_tokens::{SequenceHash, compute_hash_v2};
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::net::Ipv4Addr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use uuid::Uuid;

use anyhow::{Context, Result};
use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::component::{Component, Namespace};
use dynamo_runtime::pipeline::RouterMode;
use dynamo_runtime::pipeline::network::egress::push_router::PushRouter;
use dynamo_runtime::pipeline::{
    AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream, SingleIn, async_trait,
    network::Ingress,
};
use dynamo_runtime::stream::{self, StreamExt};
use dynamo_runtime::transports::event_plane::{EventPublisher, EventSubscriber};

pub const SERVING_G2_REPLICA_TOPIC: &str = "kvbm.serving_g2.replicas.v1";
pub const SERVING_G2_INDEX_NAMESPACE: &str = "kvbm-serving-g2";
pub const SERVING_G2_MODEL_NAMESPACE_ENV: &str = "DYNAMO_MODEL_NAMESPACE";
pub const SERVING_G2_DEFAULT_MODEL_NAMESPACE: &str = "dynamo";
pub const SERVING_G2_INDEX_COMPONENT: &str = "replica-index";
pub const SERVING_G2_INDEX_QUERY_ENDPOINT: &str = "query";
pub const SERVING_G2_PEER_COMPONENT: &str = "replica-peer";
pub const SERVING_G2_PEER_ENDPOINT: &str = "fetch";
const REPLICA_EVENT_WORKERS: usize = 8;
const REPLICA_BATCH_DEADLINE: Duration = Duration::from_millis(50);
const REPLICA_BATCH_MAX_BLOCKS: usize = 4_096;
const REPLICA_PUBLISH_QUEUE_CAPACITY: usize = 1_024;

/// Scope direct G2G discovery beneath the model deployment's namespace.
pub fn serving_g2_namespace(
    runtime: &DistributedRuntime,
    model_namespace: &str,
) -> Result<Namespace> {
    let model_namespace = model_namespace.trim();
    anyhow::ensure!(
        !model_namespace.is_empty(),
        "serving-G2 model namespace must not be empty"
    );
    runtime
        .namespace(model_namespace)?
        .namespace(SERVING_G2_INDEX_NAMESPACE)
}

/// Resolve the model namespace used by Baseten workers and connector clients.
const BASETEN_ROW_ENV: &str = "BASETEN_ROW";
const BASETEN_HOST_IP_ENV: &str = "BASETEN_HOST_IP";
const POD_IP_ENV: &str = "POD_IP";

fn optional_utf8_env(name: &str) -> Result<Option<String>> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn resolve_baseten_row(
    explicit_row: Option<&str>,
    host_ip: Option<&str>,
) -> Result<Option<String>> {
    if let Some(row) = explicit_row.map(str::trim).filter(|row| !row.is_empty()) {
        return Ok(Some(row.to_string()));
    }

    let Some(host_ip) = host_ip.map(str::trim).filter(|host_ip| !host_ip.is_empty()) else {
        return Ok(None);
    };
    let host_ip = host_ip
        .parse::<Ipv4Addr>()
        .with_context(|| format!("{BASETEN_HOST_IP_ENV} must contain an IPv4 address"))?;
    // Baseten rows are encoded in the third octet's high nibble of the host IP.
    let row_id = host_ip.octets()[2] >> 4;
    Ok(Some(format!("row{row_id:02}")))
}

/// Physical row of this host, from `BASETEN_ROW` or derived from `BASETEN_HOST_IP`.
pub fn baseten_row_from_env() -> Result<Option<String>> {
    let explicit_row = optional_utf8_env(BASETEN_ROW_ENV)?;
    let host_ip = optional_utf8_env(BASETEN_HOST_IP_ENV)?;
    resolve_baseten_row(explicit_row.as_deref(), host_ip.as_deref())
}

fn resolve_g2pb_replica_id(
    explicit_replica_id: Option<&str>,
    host_ip: Option<&str>,
    pod_ip: Option<&str>,
    instance_id: u64,
    tensor_parallel_world_size: u32,
) -> Result<u64> {
    let replica_id = explicit_replica_id
        .map(str::trim)
        .filter(|replica_id| !replica_id.is_empty())
        .or_else(|| host_ip.map(str::trim).filter(|host_ip| !host_ip.is_empty()))
        .or_else(|| pod_ip.map(str::trim).filter(|pod_ip| !pod_ip.is_empty()));
    let Some(replica_id) = replica_id else {
        anyhow::ensure!(
            tensor_parallel_world_size == 1,
            "TP{tensor_parallel_world_size} direct G2-to-G2 serving requires {}, {BASETEN_HOST_IP_ENV}, or {POD_IP_ENV}",
            dynamo_runtime::config::environment_names::kvbm::DYN_KVBM_G2PB_REPLICA_ID,
        );
        return Ok(instance_id);
    };
    Ok(compute_hash_v2(replica_id.as_bytes(), 0))
}

/// Stable identity shared by every tensor-parallel rank of one serving replica.
pub fn g2pb_replica_id_from_env(instance_id: u64, tensor_parallel_world_size: u32) -> Result<u64> {
    let explicit_replica_id = optional_utf8_env(
        dynamo_runtime::config::environment_names::kvbm::DYN_KVBM_G2PB_REPLICA_ID,
    )?;
    let host_ip = optional_utf8_env(BASETEN_HOST_IP_ENV)?;
    let pod_ip = optional_utf8_env(POD_IP_ENV)?;
    resolve_g2pb_replica_id(
        explicit_replica_id.as_deref(),
        host_ip.as_deref(),
        pod_ip.as_deref(),
        instance_id,
        tensor_parallel_world_size,
    )
}

fn parse_g2pb_allow_cross_row(value: Option<&str>) -> Result<bool> {
    match value.map(str::trim).map(str::to_ascii_lowercase) {
        None => Ok(true),
        Some(value) if matches!(value.as_str(), "true" | "1" | "yes" | "on") => Ok(true),
        Some(value) if matches!(value.as_str(), "false" | "0" | "no" | "off") => Ok(false),
        Some(value) => anyhow::bail!(
            "{} must be a boolean; got {value}",
            dynamo_runtime::config::environment_names::kvbm::DYN_KVBM_G2PB_ALLOW_CROSS_ROW
        ),
    }
}

/// Whether direct G2-to-G2 onboarding may read from cache owners in another row.
pub fn g2pb_allow_cross_row_from_env() -> Result<bool> {
    parse_g2pb_allow_cross_row(
        std::env::var(
            dynamo_runtime::config::environment_names::kvbm::DYN_KVBM_G2PB_ALLOW_CROSS_ROW,
        )
        .ok()
        .as_deref(),
    )
}

pub fn serving_g2_namespace_from_env(runtime: &DistributedRuntime) -> Result<Namespace> {
    let model_namespace = std::env::var(SERVING_G2_MODEL_NAMESPACE_ENV)
        .unwrap_or_else(|_| SERVING_G2_DEFAULT_MODEL_NAMESPACE.to_string());
    serving_g2_namespace(runtime, &model_namespace)
}

/// Stable identity used for membership and sticky event routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReplicaSourceKey {
    pub instance_id: u64,
    pub tp_rank: u32,
}

/// One physical cache producer. This identity is intentionally unrelated to
/// inference-router ownership.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReplicaSource {
    pub key: ReplicaSourceKey,
    /// Stable identity shared by every TP rank in one model replica. Physical
    /// endpoint IDs remain rank-specific so fetches reach the owning process.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replica_id: Option<u64>,
    pub tp_world_size: u32,
    pub incarnation: Uuid,
    pub block_size: u32,
    /// Physical topology row advertised by the source worker. Older publishers
    /// omit this field and remain visible to clients that do not require a row.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub row: Option<String>,
}

impl ReplicaSource {
    pub fn replica_id(&self) -> Option<u64> {
        match self.replica_id {
            Some(replica_id) => Some(replica_id),
            None if self.tp_world_size == 1 => Some(self.key.instance_id),
            None => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplicaBlock {
    pub key: SequenceHash,
    pub size_bytes: u64,
    pub checksum: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicaEventKind {
    Upsert(Vec<ReplicaBlock>),
    Remove(Vec<SequenceHash>),
    RemoveSource,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplicaEvent {
    pub source: ReplicaSource,
    pub revision: u64,
    pub kind: ReplicaEventKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplicaLocation {
    pub source: ReplicaSource,
    pub block: ReplicaBlock,
    pub revision: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplicaQueryRequest {
    pub keys: Vec<SequenceHash>,
    pub block_size: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicaQueryResponse {
    Locations(Vec<Vec<ReplicaLocation>>),
    Error(String),
}

#[derive(Clone)]
pub struct ReplicaIndexClient {
    query: PushRouter<ReplicaQueryRequest, ReplicaQueryResponse>,
}

impl ReplicaIndexClient {
    pub async fn new(component: &Component) -> Result<Self> {
        let client = component
            .endpoint(SERVING_G2_INDEX_QUERY_ENDPOINT)
            .client()
            .await?;
        client.wait_for_instances().await?;
        let query =
            PushRouter::from_client_no_fault_detection(client, RouterMode::RoundRobin).await?;
        Ok(Self { query })
    }

    pub async fn query(&self, request: ReplicaQueryRequest) -> Result<Vec<Vec<ReplicaLocation>>> {
        let mut response = self.query.round_robin(SingleIn::new(request)).await?;
        match response.next().await {
            Some(ReplicaQueryResponse::Locations(locations)) => Ok(locations),
            Some(ReplicaQueryResponse::Error(message)) => anyhow::bail!(message),
            None => anyhow::bail!("serving-G2 replica index returned no response"),
        }
    }
}

impl dynamo_runtime::protocols::maybe_error::MaybeError for ReplicaQueryResponse {
    fn from_err(error: impl std::error::Error + 'static) -> Self {
        Self::Error(error.to_string())
    }

    fn err(&self) -> Option<dynamo_runtime::error::DynamoError> {
        match self {
            Self::Locations(_) => None,
            Self::Error(message) => Some(dynamo_runtime::error::DynamoError::msg(message.clone())),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectedReplicaBlock {
    pub key: SequenceHash,
    pub source: ReplicaSource,
}

/// Assign every remotely available logical block to one or more peers.
///
/// `logical_blocks` contains one physical key per TP rank for each logical
/// block. Selection starts at the tail and prefers a single peer with the
/// longest backward run. When that peer cannot serve a block, another peer is
/// selected for that block. Incomplete blocks are skipped rather than ending
/// the plan, leaving prefix/usability policy to the caller.
pub fn select_replica_sources(
    logical_blocks: &[Vec<SequenceHash>],
    locations: &HashMap<SequenceHash, Vec<ReplicaLocation>>,
) -> Vec<Vec<SelectedReplicaBlock>> {
    select_replica_sources_inner(logical_blocks, locations)
}

/// Assign every TP-complete same-row block first, then fill unavailable blocks
/// from any row. The resulting source groups can execute concurrently, so a
/// local multi-rail transfer and a cross-row transfer do not serialize.
pub fn select_replica_sources_prefer_row(
    logical_blocks: &[Vec<SequenceHash>],
    locations: &HashMap<SequenceHash, Vec<ReplicaLocation>>,
    preferred_row: Option<&str>,
) -> Vec<Vec<SelectedReplicaBlock>> {
    let Some(preferred_row) = preferred_row else {
        return select_replica_sources_inner(logical_blocks, locations);
    };
    let same_row_locations = locations
        .iter()
        .filter_map(|(key, candidates)| {
            let candidates = candidates
                .iter()
                .filter(|candidate| candidate.source.row.as_deref() == Some(preferred_row))
                .cloned()
                .collect::<Vec<_>>();
            (!candidates.is_empty()).then_some((*key, candidates))
        })
        .collect::<HashMap<_, _>>();
    let mut selected = select_replica_sources_inner(logical_blocks, &same_row_locations);
    let fallback = select_replica_sources_inner(logical_blocks, locations);
    for (selected_block, fallback_block) in selected.iter_mut().zip(fallback) {
        if selected_block.is_empty() {
            *selected_block = fallback_block;
        }
    }
    selected
}

fn select_replica_sources_inner(
    logical_blocks: &[Vec<SequenceHash>],
    locations: &HashMap<SequenceHash, Vec<ReplicaLocation>>,
) -> Vec<Vec<SelectedReplicaBlock>> {
    let complete_by_block = logical_blocks
        .iter()
        .map(|logical_block| {
            if logical_block.is_empty() {
                return HashMap::new();
            }
            let mut by_replica = HashMap::<u64, Vec<SelectedReplicaBlock>>::new();
            for (rank, key) in logical_block.iter().copied().enumerate() {
                let Some(candidates) = locations.get(&key) else {
                    return HashMap::new();
                };
                for candidate in candidates
                    .iter()
                    .filter(|candidate| candidate.source.key.tp_rank == rank as u32)
                {
                    let Some(replica_id) = candidate.source.replica_id() else {
                        // Legacy TP events lack a shared replica identity and cannot be
                        // assembled safely across rank-specific endpoints.
                        continue;
                    };
                    by_replica
                        .entry(replica_id)
                        .or_default()
                        .push(SelectedReplicaBlock {
                            key,
                            source: candidate.source.clone(),
                        });
                }
            }
            by_replica.retain(|_, blocks| {
                blocks.len() == logical_block.len()
                    && blocks
                        .iter()
                        .enumerate()
                        .all(|(rank, block)| block.source.key.tp_rank == rank as u32)
            });
            by_replica
        })
        .collect::<Vec<_>>();

    let coverage_by_replica =
        complete_by_block
            .iter()
            .fold(HashMap::<u64, usize>::new(), |mut coverage, candidates| {
                for replica_id in candidates.keys() {
                    *coverage.entry(*replica_id).or_default() += 1;
                }
                coverage
            });
    let mut preferred_replica = None;
    let mut selected = vec![Vec::new(); logical_blocks.len()];

    for logical_index in (0..logical_blocks.len()).rev() {
        let candidates = &complete_by_block[logical_index];
        if candidates.is_empty() {
            preferred_replica = None;
            continue;
        }
        let replica_id = preferred_replica
            .filter(|replica_id| candidates.contains_key(replica_id))
            .unwrap_or_else(|| {
                *candidates
                    .keys()
                    .max_by_key(|replica_id| {
                        let backward_run = (0..=logical_index)
                            .rev()
                            .take_while(|index| complete_by_block[*index].contains_key(replica_id))
                            .count();
                        (
                            backward_run,
                            coverage_by_replica
                                .get(replica_id)
                                .copied()
                                .unwrap_or_default(),
                            Reverse(**replica_id),
                        )
                    })
                    .expect("non-empty candidates must select a replica")
            });
        preferred_replica = Some(replica_id);
        selected[logical_index] = candidates[&replica_id].clone();
    }
    selected
}

struct SourceIndexState {
    incarnation: Uuid,
    revision: u64,
    keys: HashSet<SequenceHash>,
}

impl SourceIndexState {
    fn new(source: &ReplicaSource) -> Self {
        Self {
            incarnation: source.incarnation,
            revision: 0,
            keys: HashSet::new(),
        }
    }
}

/// Sharded exact-replica directory with reverse state per physical producer.
#[derive(Default)]
pub struct ReplicaIndex {
    by_key: DashMap<SequenceHash, Vec<ReplicaLocation>>,
    by_source: DashMap<ReplicaSourceKey, SourceIndexState>,
    /// Instances purged after leaving discovery. Instance ids are discovery
    /// lease ids and are never reused, so a late delta from one is dropped
    /// rather than re-creating entries nothing would purge again.
    departed: DashMap<u64, ()>,
}

impl ReplicaIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn subscribe(component: &Component) -> Result<Arc<Self>> {
        let index = Arc::new(Self::new());
        let subscriber = EventSubscriber::for_component(component, SERVING_G2_REPLICA_TOPIC)
            .await?
            .typed::<ReplicaEvent>();
        spawn_replica_subscriber(subscriber, Arc::downgrade(&index));
        Ok(index)
    }

    pub fn apply(&self, event: ReplicaEvent) -> bool {
        let source_key = event.source.key;
        if self.departed.contains_key(&source_key.instance_id) {
            return false;
        }
        let mut source = self
            .by_source
            .entry(source_key)
            .or_insert_with(|| SourceIndexState::new(&event.source));
        if source.incarnation != event.source.incarnation {
            for key in std::mem::take(&mut source.keys) {
                self.remove_replica(key, source_key);
            }
            source.incarnation = event.source.incarnation;
            source.revision = 0;
        }
        if source.revision >= event.revision {
            return false;
        }
        match event.kind {
            ReplicaEventKind::Upsert(blocks) => {
                for block in blocks {
                    source.keys.insert(block.key);
                    self.upsert_replica(event.source.clone(), event.revision, block);
                }
            }
            ReplicaEventKind::Remove(keys) => {
                for key in keys {
                    source.keys.remove(&key);
                    self.remove_replica(key, source_key);
                }
            }
            ReplicaEventKind::RemoveSource => {
                for key in std::mem::take(&mut source.keys) {
                    self.remove_replica(key, source_key);
                }
            }
        }
        source.revision = event.revision;
        drop(source);
        // The instance may have been purged while this event was applied.
        if self.departed.contains_key(&source_key.instance_id) {
            self.remove_source(source_key);
            return false;
        }
        true
    }

    fn upsert_replica(&self, source: ReplicaSource, revision: u64, block: ReplicaBlock) {
        let key = block.key;
        let location = ReplicaLocation {
            source,
            block,
            revision,
        };
        let mut locations = self.by_key.entry(key).or_default();
        if let Some(existing) = locations
            .iter_mut()
            .find(|existing| existing.source.key == location.source.key)
        {
            *existing = location;
        } else {
            locations.push(location);
        }
    }

    fn remove_replica(&self, key: SequenceHash, source: ReplicaSourceKey) {
        if let Some(mut locations) = self.by_key.get_mut(&key) {
            locations.retain(|location| location.source.key != source);
            let empty = locations.is_empty();
            drop(locations);
            if empty {
                self.by_key.remove(&key);
            }
        }
    }

    /// Drop every entry one physical producer advertised. Returns how many.
    fn remove_source(&self, source: ReplicaSourceKey) -> usize {
        let Some((_source, state)) = self.by_source.remove(&source) else {
            return 0;
        };
        let removed = state.keys.len();
        for key in state.keys {
            self.remove_replica(key, source);
        }
        removed
    }

    /// Forget an instance that has left discovery: drop every source (TP rank)
    /// it published and ignore anything it still has in flight. Returns
    /// (sources, entries) removed.
    pub fn purge_instance(&self, instance_id: u64) -> (usize, usize) {
        // Mark first: an event applied after this is dropped, and one already
        // past the check is caught by the re-check at the end of `apply`.
        self.departed.insert(instance_id, ());
        let sources = self
            .by_source
            .iter()
            .filter(|source| source.key().instance_id == instance_id)
            .map(|source| *source.key())
            .collect::<Vec<_>>();
        let entries = sources
            .iter()
            .map(|source| self.remove_source(*source))
            .sum();
        (sources.len(), entries)
    }

    fn locations(&self, key: SequenceHash, block_size: Option<u32>) -> Vec<ReplicaLocation> {
        let mut result = self
            .by_key
            .get(&key)
            .map(|locations| {
                locations
                    .iter()
                    .filter(|location| {
                        block_size.is_none_or(|expected| location.source.block_size == expected)
                    })
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        result.sort_unstable_by_key(|location| {
            (location.source.key.instance_id, location.source.key.tp_rank)
        });
        result
    }

    /// One sharded map lookup per physical key. Output order matches input.
    pub fn batch_locations(
        &self,
        keys: &[SequenceHash],
        block_size: Option<u32>,
    ) -> Vec<Vec<ReplicaLocation>> {
        keys.iter()
            .map(|key| self.locations(*key, block_size))
            .collect()
    }

    #[cfg(test)]
    fn entry_count(&self) -> usize {
        self.by_key.len()
    }
}

/// Start a copy of the serving-G2 directory and serve it on the index endpoint.
/// Returns the copy: a process that runs one (the HiCache G2 transport) reads it
/// directly, so a lookup never depends on another replica being reachable.
pub async fn start_replica_index_service(component: &Component) -> Result<Arc<ReplicaIndex>> {
    let index = ReplicaIndex::subscribe(component).await?;
    let ingress =
        Ingress::<SingleIn<ReplicaQueryRequest>, ManyOut<ReplicaQueryResponse>>::for_engine(
            Arc::new(ReplicaQueryEngine {
                index: Arc::clone(&index),
            }),
        )?;
    let endpoint = component.endpoint(SERVING_G2_INDEX_QUERY_ENDPOINT);
    tokio::spawn(async move {
        if let Err(error) = endpoint
            .endpoint_builder()
            .handler(ingress)
            .graceful_shutdown(true)
            .start()
            .await
        {
            tracing::error!(%error, "serving-G2 replica index endpoint failed");
        }
    });
    Ok(index)
}

struct ReplicaQueryEngine {
    index: Arc<ReplicaIndex>,
}

#[async_trait]
impl AsyncEngine<SingleIn<ReplicaQueryRequest>, ManyOut<ReplicaQueryResponse>, anyhow::Error>
    for ReplicaQueryEngine
{
    async fn generate(
        &self,
        request: SingleIn<ReplicaQueryRequest>,
    ) -> Result<ManyOut<ReplicaQueryResponse>> {
        let (request, context) = request.into_parts();
        let response = ReplicaQueryResponse::Locations(
            self.index
                .batch_locations(&request.keys, Some(request.block_size)),
        );
        Ok(ResponseStream::new(
            Box::pin(stream::iter(vec![response])),
            context.context(),
        ))
    }
}

pub struct ReplicaPublisher {
    tx: mpsc::Sender<ReplicaEventKind>,
}

impl ReplicaPublisher {
    pub async fn new(component: &Component, source: ReplicaSource) -> Result<Arc<Self>> {
        let publisher = Arc::new(
            EventPublisher::for_component(component, SERVING_G2_REPLICA_TOPIC)
                .await
                .context("failed to create replica event publisher")?,
        );
        let (tx, rx) = mpsc::channel(REPLICA_PUBLISH_QUEUE_CAPACITY);
        tokio::spawn(run_replica_publisher(source, publisher, rx));
        Ok(Arc::new(Self { tx }))
    }

    pub fn upsert(&self, blocks: Vec<ReplicaBlock>) -> Result<()> {
        self.publish(ReplicaEventKind::Upsert(blocks))
    }

    pub fn remove(&self, keys: Vec<SequenceHash>) -> Result<()> {
        self.publish(ReplicaEventKind::Remove(keys))
    }

    fn publish(&self, kind: ReplicaEventKind) -> Result<()> {
        self.tx
            .try_send(kind)
            .map_err(|error| anyhow::anyhow!("replica publish queue rejected an event: {error}"))
    }
}

#[derive(Default)]
struct PendingReplicaDelta {
    changes: HashMap<SequenceHash, Option<ReplicaBlock>>,
}

impl PendingReplicaDelta {
    fn absorb(&mut self, kind: ReplicaEventKind) {
        match kind {
            ReplicaEventKind::Upsert(blocks) => {
                for block in blocks {
                    self.changes.insert(block.key, Some(block));
                }
            }
            ReplicaEventKind::Remove(keys) => {
                for key in keys {
                    self.changes.insert(key, None);
                }
            }
            ReplicaEventKind::RemoveSource => unreachable!("source removal is not batched"),
        }
    }

    fn into_kinds(self, max_blocks: usize) -> Vec<ReplicaEventKind> {
        let mut upserts = Vec::new();
        let mut removals = Vec::new();
        for (key, block) in self.changes {
            match block {
                Some(block) => upserts.push(block),
                None => removals.push(key),
            }
        }
        let mut kinds = Vec::new();
        for chunk in removals.chunks(max_blocks) {
            kinds.push(ReplicaEventKind::Remove(chunk.to_vec()));
        }
        for chunk in upserts.chunks(max_blocks) {
            kinds.push(ReplicaEventKind::Upsert(chunk.to_vec()));
        }
        kinds
    }
}

async fn publish_replica_event(
    publisher: &EventPublisher,
    source: &ReplicaSource,
    revision: &mut u64,
    kind: ReplicaEventKind,
) {
    *revision += 1;
    if let Err(error) = publisher
        .publish(&ReplicaEvent {
            source: source.clone(),
            revision: *revision,
            kind,
        })
        .await
    {
        tracing::warn!(error = %error, "failed to publish replica event batch");
    }
}

async fn run_replica_publisher(
    source: ReplicaSource,
    publisher: Arc<EventPublisher>,
    mut rx: mpsc::Receiver<ReplicaEventKind>,
) {
    let mut revision = 0;
    while let Some(first) = rx.recv().await {
        let mut pending = PendingReplicaDelta::default();
        pending.absorb(first);

        let deadline = tokio::time::sleep(REPLICA_BATCH_DEADLINE);
        tokio::pin!(deadline);
        while pending.changes.len() < REPLICA_BATCH_MAX_BLOCKS {
            tokio::select! {
                _ = &mut deadline => break,
                next = rx.recv() => match next {
                    Some(delta) => pending.absorb(delta),
                    None => break,
                }
            }
        }

        for kind in pending.into_kinds(REPLICA_BATCH_MAX_BLOCKS) {
            publish_replica_event(&publisher, &source.clone(), &mut revision, kind).await;
        }
    }
    publish_replica_event(
        &publisher,
        &source.clone(),
        &mut revision,
        ReplicaEventKind::RemoveSource,
    )
    .await;
}

fn spawn_replica_subscriber(
    mut subscriber: dynamo_runtime::transports::event_plane::TypedEventSubscriber<ReplicaEvent>,
    index: std::sync::Weak<ReplicaIndex>,
) {
    let mut workers = Vec::with_capacity(REPLICA_EVENT_WORKERS);
    for _ in 0..REPLICA_EVENT_WORKERS {
        let (tx, mut rx) = mpsc::unbounded_channel::<ReplicaEvent>();
        workers.push(tx);
        let index = index.clone();
        tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                let Some(index) = index.upgrade() else {
                    break;
                };
                index.apply(event);
            }
        });
    }

    tokio::spawn(async move {
        while let Some(event) = subscriber.next().await {
            match event {
                Ok((_envelope, event)) => {
                    let sticky =
                        event.source.key.instance_id as usize ^ event.source.key.tp_rank as usize;
                    if workers[sticky % workers.len()].send(event).is_err() {
                        break;
                    }
                }
                Err(error) => tracing::warn!(error = %error, "failed to consume replica event"),
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn resolves_baseten_row_from_annotation_or_host_ip() {
        assert_eq!(
            resolve_baseten_row(Some(" row07 "), Some("10.4.74.30")).unwrap(),
            Some("row07".to_string())
        );
        assert_eq!(
            resolve_baseten_row(None, Some("10.4.74.30")).unwrap(),
            Some("row04".to_string())
        );
        assert_eq!(resolve_baseten_row(None, None).unwrap(), None);
        assert!(resolve_baseten_row(None, Some("not-an-ip")).is_err());
    }

    #[test]
    fn resolves_replica_identity_shared_by_tensor_parallel_ranks() {
        let rank_0 = resolve_g2pb_replica_id(Some(" replica-a "), None, None, 70, 4).unwrap();
        let rank_1 = resolve_g2pb_replica_id(Some("replica-a"), None, None, 71, 4).unwrap();
        assert_eq!(rank_0, rank_1);

        let host_rank_0 =
            resolve_g2pb_replica_id(None, Some("10.1.2.3"), Some("10.4.5.6"), 70, 4).unwrap();
        let host_rank_1 =
            resolve_g2pb_replica_id(Some(" "), Some("10.1.2.3"), None, 71, 4).unwrap();
        assert_eq!(host_rank_0, host_rank_1);

        let pod_rank_0 = resolve_g2pb_replica_id(None, None, Some("10.4.5.6"), 70, 4).unwrap();
        let pod_rank_1 = resolve_g2pb_replica_id(None, None, Some("10.4.5.6"), 71, 4).unwrap();
        assert_eq!(pod_rank_0, pod_rank_1);
        assert_ne!(host_rank_0, pod_rank_0);

        assert_eq!(
            resolve_g2pb_replica_id(None, None, None, 70, 1).unwrap(),
            70
        );
        assert!(resolve_g2pb_replica_id(None, None, None, 70, 4).is_err());
    }

    #[test]
    fn cross_row_policy_defaults_to_allowed_and_parses_booleans() {
        assert!(parse_g2pb_allow_cross_row(None).unwrap());
        assert!(parse_g2pb_allow_cross_row(Some("true")).unwrap());
        assert!(parse_g2pb_allow_cross_row(Some("1")).unwrap());
        assert!(!parse_g2pb_allow_cross_row(Some("false")).unwrap());
        assert!(!parse_g2pb_allow_cross_row(Some(" off ")).unwrap());
        assert!(parse_g2pb_allow_cross_row(Some("sometimes")).is_err());
    }

    fn source(instance_id: u64, rank: u32, incarnation: u128) -> ReplicaSource {
        ReplicaSource {
            key: ReplicaSourceKey {
                instance_id,
                tp_rank: rank,
            },
            replica_id: Some(instance_id),
            tp_world_size: 2,
            incarnation: Uuid::from_u128(incarnation),
            block_size: 64,
            row: None,
        }
    }

    fn block(key: u64) -> ReplicaBlock {
        ReplicaBlock {
            key,
            size_bytes: 4096,
            checksum: Some(key.wrapping_mul(17)),
        }
    }

    #[test]
    fn restart_revision_batch_and_compatibility_filters_work() {
        let index = ReplicaIndex::new();
        let old = source(7, 0, 1);
        assert!(index.apply(ReplicaEvent {
            source: old.clone(),
            revision: 1,
            kind: ReplicaEventKind::Upsert(vec![block(11), block(12)]),
        }));
        assert!(!index.apply(ReplicaEvent {
            source: old,
            revision: 1,
            kind: ReplicaEventKind::Remove(vec![11]),
        }));
        let new = source(7, 0, 2);
        assert!(index.apply(ReplicaEvent {
            source: new.clone(),
            revision: 1,
            kind: ReplicaEventKind::Upsert(vec![block(13)]),
        }));
        assert!(index.locations(11, None).is_empty());
        assert_eq!(index.batch_locations(&[12, 13], Some(64))[1].len(), 1);
        assert!(index.locations(13, Some(32)).is_empty());
    }

    #[test]
    fn source_removal_preserves_other_replicas() {
        let index = ReplicaIndex::new();
        for instance in [7, 9] {
            let source = source(instance, 0, instance as u128);
            index.apply(ReplicaEvent {
                source,
                revision: 1,
                kind: ReplicaEventKind::Upsert(vec![block(11)]),
            });
        }
        assert_eq!(index.locations(11, None).len(), 2);
        index.remove_source(source(7, 0, 7).key);
        assert_eq!(index.locations(11, None)[0].source.key.instance_id, 9);
    }

    fn upsert(index: &ReplicaIndex, instance: u64, rank: u32, keys: &[u64]) {
        index.apply(ReplicaEvent {
            source: source(instance, rank, instance as u128),
            revision: 1,
            kind: ReplicaEventKind::Upsert(keys.iter().copied().map(block).collect()),
        });
    }

    #[test]
    fn purge_instance_drops_every_rank_and_keeps_other_sources() {
        let index = ReplicaIndex::new();
        // 7 departs (both TP ranks); 9 shares key 11 and stays.
        upsert(&index, 7, 0, &[11, 12]);
        upsert(&index, 7, 1, &[11, 13]);
        upsert(&index, 9, 0, &[11]);
        assert_eq!(index.locations(11, None).len(), 3);

        assert_eq!(index.purge_instance(7), (2, 4));
        let at_11 = index.locations(11, None);
        assert_eq!(at_11.len(), 1);
        assert_eq!(at_11[0].source.key.instance_id, 9);
        assert!(index.locations(12, None).is_empty());
        assert!(index.locations(13, None).is_empty());
        assert_eq!(index.entry_count(), 1);
        // Purging again, or an instance never seen, is a no-op.
        assert_eq!(index.purge_instance(7), (0, 0));
        assert_eq!(index.purge_instance(42), (0, 0));
    }

    #[test]
    fn purged_instance_late_events_are_dropped() {
        let index = ReplicaIndex::new();
        upsert(&index, 7, 0, &[11]);
        index.purge_instance(7);
        // A delta still in flight from the departed instance must not re-create
        // entries nothing would purge again.
        assert!(!index.apply(ReplicaEvent {
            source: source(7, 0, 7),
            revision: 2,
            kind: ReplicaEventKind::Upsert(vec![block(12)]),
        }));
        assert!(index.locations(12, None).is_empty());
        assert_eq!(index.entry_count(), 0);
        // Other instances are unaffected.
        upsert(&index, 9, 0, &[12]);
        assert_eq!(index.locations(12, None).len(), 1);
    }

    #[test]
    fn replica_query_preserves_same_and_cross_row_sources() {
        let index = ReplicaIndex::new();
        for (instance_id, row) in [(7, Some("row04")), (9, Some("row05")), (11, None)] {
            let mut source = source(instance_id, 0, instance_id as u128);
            source.row = row.map(str::to_string);
            index.apply(ReplicaEvent {
                source,
                revision: 1,
                kind: ReplicaEventKind::Upsert(vec![block(11)]),
            });
        }

        let locations = index.batch_locations(&[11], None);
        assert_eq!(locations[0].len(), 3);
        assert_eq!(
            locations[0]
                .iter()
                .map(|location| location.source.key.instance_id)
                .collect::<Vec<_>>(),
            vec![7, 9, 11]
        );
    }

    #[test]
    fn row_fields_are_wire_compatible_with_legacy_messages() {
        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct LegacyReplicaSource {
            key: ReplicaSourceKey,
            tp_world_size: u32,
            incarnation: Uuid,
            block_size: u32,
        }
        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct LegacyReplicaQueryRequest {
            keys: Vec<SequenceHash>,
            block_size: u32,
        }

        let legacy_source = LegacyReplicaSource {
            key: ReplicaSourceKey {
                instance_id: 7,
                tp_rank: 0,
            },
            tp_world_size: 2,
            incarnation: Uuid::from_u128(1),
            block_size: 64,
        };
        let decoded_source: ReplicaSource = rmp_serde::from_slice(
            &rmp_serde::to_vec_named(&legacy_source).expect("serialize legacy source"),
        )
        .expect("decode legacy source");
        assert_eq!(decoded_source.replica_id, None);
        assert_eq!(decoded_source.row, None);

        let legacy_query = LegacyReplicaQueryRequest {
            keys: vec![11, 12],
            block_size: 64,
        };
        let decoded_query: ReplicaQueryRequest = rmp_serde::from_slice(
            &rmp_serde::to_vec_named(&legacy_query).expect("serialize legacy query"),
        )
        .expect("decode legacy query");
        assert_eq!(decoded_query.keys, legacy_query.keys);
        assert_eq!(decoded_query.block_size, legacy_query.block_size);

        let mut new_source = source(7, 0, 1);
        new_source.row = Some("row04".to_string());
        let decoded_legacy_source: LegacyReplicaSource = rmp_serde::from_slice(
            &rmp_serde::to_vec_named(&new_source).expect("serialize new source"),
        )
        .expect("legacy source ignores row");
        assert_eq!(decoded_legacy_source, legacy_source);
    }

    #[test]
    fn revision_gap_applies_the_latest_delta_under_eventual_consistency() {
        let index = ReplicaIndex::new();
        let source = source(7, 0, 1);
        assert!(index.apply(ReplicaEvent {
            source: source.clone(),
            revision: 3,
            kind: ReplicaEventKind::Upsert(vec![block(11)]),
        }));
        assert!(index.apply(ReplicaEvent {
            source: source.clone(),
            revision: 4,
            kind: ReplicaEventKind::Upsert(vec![block(12)]),
        }));
        assert!(index.apply(ReplicaEvent {
            source: source.clone(),
            revision: 5,
            kind: ReplicaEventKind::Upsert(vec![block(13)]),
        }));
        assert_eq!(index.locations(11, None).len(), 1);
        assert_eq!(index.locations(13, None).len(), 1);
    }

    #[test]
    fn micro_batch_is_last_change_wins_and_bounded() {
        let mut pending = PendingReplicaDelta::default();
        pending.absorb(ReplicaEventKind::Upsert(vec![block(11), block(12)]));
        pending.absorb(ReplicaEventKind::Remove(vec![11, 12]));
        pending.absorb(ReplicaEventKind::Upsert(vec![block(11)]));
        let kinds = pending.into_kinds(1);
        assert_eq!(kinds.len(), 2);
        assert!(matches!(&kinds[0], ReplicaEventKind::Remove(keys) if keys == &[12]));
        assert!(matches!(&kinds[1], ReplicaEventKind::Upsert(blocks) if blocks[0].key == 11));
    }

    #[test]
    fn selection_prefers_tail_peer_and_skips_incomplete_tp_blocks() {
        let source_a0 = source(7, 0, 1);
        let source_a1 = source(7, 1, 1);
        let source_b0 = source(9, 0, 2);
        let source_b1 = source(9, 1, 2);
        let location = |key, source: &ReplicaSource| ReplicaLocation {
            source: source.clone(),
            block: block(key),
            revision: 1,
        };
        let locations = HashMap::from([
            (11, vec![location(11, &source_a0), location(11, &source_b0)]),
            (12, vec![location(12, &source_a1), location(12, &source_b1)]),
            (21, vec![location(21, &source_a0), location(21, &source_b0)]),
            (22, vec![location(22, &source_a1), location(22, &source_b1)]),
            (31, vec![location(31, &source_a0)]),
        ]);

        let selected =
            select_replica_sources(&[vec![11, 12], vec![31, 32], vec![21, 22]], &locations);
        assert_eq!(
            selected
                .iter()
                .enumerate()
                .filter_map(|(index, blocks)| (!blocks.is_empty()).then_some(index))
                .collect::<Vec<_>>(),
            vec![0, 2]
        );
        assert!(
            selected
                .iter()
                .flatten()
                .all(|block| block.source.key.instance_id == 7)
        );
    }

    #[test]
    fn selection_groups_rank_specific_endpoints_by_replica_and_rejects_unidentified_sources() {
        let mut source_rank_0 = source(70, 0, 1);
        let mut source_rank_1 = source(71, 1, 2);
        source_rank_0.replica_id = Some(7);
        source_rank_1.replica_id = Some(7);
        let locations = HashMap::from([
            (
                11,
                vec![ReplicaLocation {
                    source: source_rank_0,
                    block: block(11),
                    revision: 1,
                }],
            ),
            (
                12,
                vec![ReplicaLocation {
                    source: source_rank_1,
                    block: block(12),
                    revision: 1,
                }],
            ),
        ]);

        let selected = select_replica_sources(&[vec![11, 12]], &locations);
        assert_eq!(
            selected[0]
                .iter()
                .map(|block| block.source.key.instance_id)
                .collect::<Vec<_>>(),
            vec![70, 71]
        );

        let locations = locations
            .into_iter()
            .map(|(key, locations)| {
                let locations = locations
                    .into_iter()
                    .map(|mut location| {
                        location.source.replica_id = None;
                        location
                    })
                    .collect();
                (key, locations)
            })
            .collect();
        assert!(select_replica_sources(&[vec![11, 12]], &locations)[0].is_empty());
    }

    #[test]
    fn selection_walks_backward_and_uses_a_fallback_peer_for_a_gap() {
        let source_a0 = source(7, 0, 1);
        let source_a1 = source(7, 1, 1);
        let source_b0 = source(9, 0, 2);
        let source_b1 = source(9, 1, 2);
        let location = |key, source: &ReplicaSource| ReplicaLocation {
            source: source.clone(),
            block: block(key),
            revision: 1,
        };
        let locations = HashMap::from([
            (11, vec![location(11, &source_a0)]),
            (12, vec![location(12, &source_a1)]),
            (21, vec![location(21, &source_a0)]),
            (22, vec![location(22, &source_a1)]),
            (31, vec![location(31, &source_b0)]),
            (32, vec![location(32, &source_b1)]),
            (41, vec![location(41, &source_a0)]),
            (42, vec![location(42, &source_a1)]),
        ]);

        let selected = select_replica_sources(
            &[vec![11, 12], vec![21, 22], vec![31, 32], vec![41, 42]],
            &locations,
        );
        assert_eq!(selected.len(), 4);
        assert_eq!(
            selected
                .iter()
                .map(|blocks| blocks[0].source.key.instance_id)
                .collect::<Vec<_>>(),
            vec![7, 7, 9, 7]
        );
    }

    #[test]
    fn selection_chooses_the_longest_peer_when_the_tail_is_replicated() {
        let source_a0 = source(7, 0, 1);
        let source_a1 = source(7, 1, 1);
        let source_b0 = source(9, 0, 2);
        let source_b1 = source(9, 1, 2);
        let location = |key, source: &ReplicaSource| ReplicaLocation {
            source: source.clone(),
            block: block(key),
            revision: 1,
        };
        let locations = HashMap::from([
            (11, vec![location(11, &source_a0)]),
            (12, vec![location(12, &source_a1)]),
            (21, vec![location(21, &source_a0)]),
            (22, vec![location(22, &source_a1)]),
            (31, vec![location(31, &source_a0), location(31, &source_b0)]),
            (32, vec![location(32, &source_a1), location(32, &source_b1)]),
        ]);

        let selected =
            select_replica_sources(&[vec![11, 12], vec![21, 22], vec![31, 32]], &locations);
        assert!(
            selected
                .iter()
                .flatten()
                .all(|block| block.source.key.instance_id == 7)
        );
    }

    #[test]
    fn selection_prefers_same_row_when_coverage_is_equal() {
        let mut same0 = source(9, 0, 1);
        let mut same1 = source(9, 1, 1);
        same0.row = Some("row04".to_string());
        same1.row = Some("row04".to_string());
        let mut cross0 = source(7, 0, 2);
        let mut cross1 = source(7, 1, 2);
        cross0.row = Some("row03".to_string());
        cross1.row = Some("row03".to_string());
        let location = |key, source: &ReplicaSource| ReplicaLocation {
            source: source.clone(),
            block: block(key),
            revision: 1,
        };
        let locations = HashMap::from([
            (11, vec![location(11, &same0), location(11, &cross0)]),
            (12, vec![location(12, &same1), location(12, &cross1)]),
        ]);

        let selected =
            select_replica_sources_prefer_row(&[vec![11, 12]], &locations, Some("row04"));
        assert!(
            selected[0]
                .iter()
                .all(|block| block.source.key.instance_id == 9)
        );
    }

    #[test]
    fn selection_fills_blocks_missing_in_same_row_from_cross_row() {
        let mut same0 = source(9, 0, 1);
        let mut same1 = source(9, 1, 1);
        same0.row = Some("row04".to_string());
        same1.row = Some("row04".to_string());
        let mut cross0 = source(7, 0, 2);
        let mut cross1 = source(7, 1, 2);
        cross0.row = Some("row03".to_string());
        cross1.row = Some("row03".to_string());
        let location = |key, source: &ReplicaSource| ReplicaLocation {
            source: source.clone(),
            block: block(key),
            revision: 1,
        };
        let locations = HashMap::from([
            (11, vec![location(11, &cross0)]),
            (12, vec![location(12, &cross1)]),
            (21, vec![location(21, &same0), location(21, &cross0)]),
            (22, vec![location(22, &same1), location(22, &cross1)]),
        ]);

        let selected = select_replica_sources_prefer_row(
            &[vec![11, 12], vec![21, 22]],
            &locations,
            Some("row04"),
        );
        assert_eq!(selected[0][0].source.key.instance_id, 7);
        assert_eq!(selected[1][0].source.key.instance_id, 9);
    }

    #[test]
    #[ignore = "performance benchmark"]
    fn benchmark_replica_index_100k_events() {
        const EVENTS: usize = 100_000;
        for source_count in [1usize, 10, 100] {
            let index = Arc::new(ReplicaIndex::new());
            let started = Instant::now();
            std::thread::scope(|scope| {
                for lane in 0..REPLICA_EVENT_WORKERS {
                    let index = Arc::clone(&index);
                    scope.spawn(move || {
                        for id in 0..EVENTS {
                            let instance = id % source_count;
                            if instance % REPLICA_EVENT_WORKERS != lane {
                                continue;
                            }
                            index.apply(ReplicaEvent {
                                source: source(instance as u64, 0, instance as u128 + 1),
                                revision: (id / source_count + 1) as u64,
                                kind: ReplicaEventKind::Upsert(vec![block(id as u64)]),
                            });
                        }
                    });
                }
            });
            let ingest = started.elapsed();
            assert_eq!(index.entry_count(), EVENTS);
            let sources = (0..source_count)
                .map(|instance| source(instance as u64, 0, instance as u128 + 1).key)
                .collect::<Vec<_>>();
            let lookup_started = Instant::now();
            assert_eq!(
                index
                    .batch_locations(&(0..EVENTS as u64).collect::<Vec<_>>(), None)
                    .len(),
                EVENTS
            );
            let lookup = lookup_started.elapsed();
            let remove_started = Instant::now();
            for source in sources {
                index.remove_source(source);
            }
            let remove = remove_started.elapsed();
            assert_eq!(index.entry_count(), 0);
            println!(
                "sources={source_count} ingest={ingest:?} lookup={lookup:?} remove={remove:?}"
            );
        }
    }
}
