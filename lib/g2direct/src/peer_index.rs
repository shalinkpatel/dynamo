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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
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
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::{EventPublisher, EventSubscriber};

use crate::node_relay::RelaySender;

/// Replica events in the compact positional encoding (see
/// `encode_compact_replica_event`). The suffix names the encoding; deployments
/// never mix builds, so there is no other topic to stay compatible with.
pub const SERVING_G2_REPLICA_TOPIC: &str = "kvbm.serving_g2.replicas.v2";
/// A replica's node-index owner publishes every process's events as one
/// compact-encoded batch per window (see `node_relay`).
pub const SERVING_G2_REPLICA_BATCH_TOPIC: &str = "kvbm.serving_g2.replicas.batch.v2";
pub const SERVING_G2_INDEX_NAMESPACE: &str = "kvbm-serving-g2";
pub const SERVING_G2_MODEL_NAMESPACE_ENV: &str = "DYNAMO_MODEL_NAMESPACE";
pub const SERVING_G2_DEFAULT_MODEL_NAMESPACE: &str = "dynamo";
pub const SERVING_G2_INDEX_COMPONENT: &str = "replica-index";
pub const SERVING_G2_INDEX_QUERY_ENDPOINT: &str = "query";
pub const SERVING_G2_PEER_COMPONENT: &str = "replica-peer";
pub const SERVING_G2_PEER_ENDPOINT: &str = "fetch";
const REPLICA_EVENT_WORKERS: usize = 8;
const REPLICA_BATCH_DEADLINE_MS: u64 = 50;
const REPLICA_BATCH_MAX_BLOCKS: usize = 4_096;
const REPLICA_PUBLISH_QUEUE_CAPACITY: usize = 1_024;
/// Longest a batch may wait on one NATS publish. A stalled publish otherwise
/// backs the queue up until callers' updates are dropped unseen.
const REPLICA_PUBLISH_TIMEOUT: Duration = Duration::from_secs(1);
/// How often the publisher re-checks the connection while idle, to announce a
/// revision gap left by a skipped batch.
const REPLICA_PUBLISHER_TICK: Duration = Duration::from_secs(1);
/// A source is resynced at most this often by one reader, so a source that
/// cannot answer (or keeps losing deltas) costs one snapshot per interval.
const REPLICA_RESYNC_MIN_INTERVAL: Duration = Duration::from_secs(30);
const REPLICA_SUBSCRIBE_RETRY_MAX: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ReplicaPublishConfig {
    batch_deadline: Duration,
    batch_max_blocks: usize,
}

impl Default for ReplicaPublishConfig {
    fn default() -> Self {
        Self {
            batch_deadline: Duration::from_millis(REPLICA_BATCH_DEADLINE_MS),
            batch_max_blocks: REPLICA_BATCH_MAX_BLOCKS,
        }
    }
}

impl ReplicaPublishConfig {
    fn parse(batch_ms: Option<&str>, max_blocks: Option<&str>) -> Result<Self> {
        use dynamo_runtime::config::environment_names::kvbm;
        let parse_number = |name: &str, value: Option<&str>, default: u64| -> Result<u64> {
            match value.map(str::trim).filter(|value| !value.is_empty()) {
                None => Ok(default),
                Some(value) => value
                    .parse()
                    .with_context(|| format!("{name} must be a non-negative integer; got {value}")),
            }
        };
        let batch_ms = parse_number(
            kvbm::DYN_KVBM_G2PB_REPLICA_BATCH_MS,
            batch_ms,
            REPLICA_BATCH_DEADLINE_MS,
        )?;
        let batch_max_blocks = parse_number(
            kvbm::DYN_KVBM_G2PB_REPLICA_BATCH_MAX_BLOCKS,
            max_blocks,
            REPLICA_BATCH_MAX_BLOCKS as u64,
        )?;
        anyhow::ensure!(
            batch_max_blocks > 0,
            "{} must be at least 1",
            kvbm::DYN_KVBM_G2PB_REPLICA_BATCH_MAX_BLOCKS
        );
        Ok(Self {
            batch_deadline: Duration::from_millis(batch_ms),
            batch_max_blocks: usize::try_from(batch_max_blocks)?,
        })
    }

    fn from_env() -> Result<Self> {
        use dynamo_runtime::config::environment_names::kvbm;
        Self::parse(
            optional_utf8_env(kvbm::DYN_KVBM_G2PB_REPLICA_BATCH_MS)?.as_deref(),
            optional_utf8_env(kvbm::DYN_KVBM_G2PB_REPLICA_BATCH_MAX_BLOCKS)?.as_deref(),
        )
    }
}

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

/// Source header of a v2 message, positional so absent options keep their slot:
/// (instance_id, tp_rank, replica_id, tp_world_size, incarnation, block_size, row).
type CompactReplicaSource = (u64, u32, Option<u64>, u32, Uuid, u32, Option<String>);

#[derive(Serialize, Deserialize)]
enum CompactReplicaKind {
    /// Every block has `size_bytes`. `keys` and `checksums` are packed
    /// little-endian u64s; `checksums` is empty when no block has one.
    Upsert {
        size_bytes: u64,
        #[serde(with = "serde_bytes")]
        keys: Vec<u8>,
        #[serde(with = "serde_bytes")]
        checksums: Vec<u8>,
    },
    /// Fallback when sizes or checksum presence differ within one batch.
    UpsertMixed(Vec<(SequenceHash, u64, Option<u64>)>),
    Remove(#[serde(with = "serde_bytes")] Vec<u8>),
    RemoveSource,
}

#[derive(Serialize, Deserialize)]
struct CompactReplicaEvent(CompactReplicaSource, u64, CompactReplicaKind);

fn pack_u64s(values: impl IntoIterator<Item = u64>) -> Vec<u8> {
    values.into_iter().flat_map(u64::to_le_bytes).collect()
}

fn unpack_u64s(packed: &[u8]) -> Result<Vec<u64>> {
    anyhow::ensure!(
        packed.len().is_multiple_of(8),
        "packed u64 array has {} bytes",
        packed.len()
    );
    Ok(packed
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().expect("8-byte chunk")))
        .collect())
}

/// Encode an event for `SERVING_G2_REPLICA_TOPIC`: msgpack arrays instead of
/// named maps, one source header per message, and a uniform upsert batch as one
/// `size_bytes` plus packed keys and checksums (16 bytes per block).
pub fn encode_compact_replica_event(event: &ReplicaEvent) -> Result<Vec<u8>> {
    let source = &event.source;
    let header = (
        source.key.instance_id,
        source.key.tp_rank,
        source.replica_id,
        source.tp_world_size,
        source.incarnation,
        source.block_size,
        source.row.clone(),
    );
    let kind = match &event.kind {
        ReplicaEventKind::Upsert(blocks) => {
            let size_bytes = blocks.first().map_or(0, |block| block.size_bytes);
            let all_checksums = blocks.iter().all(|block| block.checksum.is_some());
            let no_checksums = blocks.iter().all(|block| block.checksum.is_none());
            if blocks.iter().all(|block| block.size_bytes == size_bytes)
                && (all_checksums || no_checksums)
            {
                CompactReplicaKind::Upsert {
                    size_bytes,
                    keys: pack_u64s(blocks.iter().map(|block| block.key)),
                    checksums: pack_u64s(blocks.iter().filter_map(|block| block.checksum)),
                }
            } else {
                CompactReplicaKind::UpsertMixed(
                    blocks
                        .iter()
                        .map(|block| (block.key, block.size_bytes, block.checksum))
                        .collect(),
                )
            }
        }
        ReplicaEventKind::Remove(keys) => {
            CompactReplicaKind::Remove(pack_u64s(keys.iter().copied()))
        }
        ReplicaEventKind::RemoveSource => CompactReplicaKind::RemoveSource,
    };
    Ok(rmp_serde::to_vec(&CompactReplicaEvent(
        header,
        event.revision,
        kind,
    ))?)
}

/// Several sources' events in one message, applied in order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplicaEventBatch {
    pub events: Vec<ReplicaEvent>,
}

/// `SERVING_G2_REPLICA_BATCH_TOPIC`: each event in the compact encoding.
#[derive(Serialize, Deserialize)]
struct CompactReplicaBatch(Vec<serde_bytes::ByteBuf>);

pub fn encode_compact_replica_batch(events: &[ReplicaEvent]) -> Result<Vec<u8>> {
    let events = events
        .iter()
        .map(|event| encode_compact_replica_event(event).map(serde_bytes::ByteBuf::from))
        .collect::<Result<Vec<_>>>()?;
    Ok(rmp_serde::to_vec(&CompactReplicaBatch(events))?)
}

pub fn decode_compact_replica_batch(bytes: &[u8]) -> Result<Vec<ReplicaEvent>> {
    let CompactReplicaBatch(events) = rmp_serde::from_slice(bytes)?;
    events
        .iter()
        .map(|event| decode_compact_replica_event(event))
        .collect()
}

pub fn decode_compact_replica_event(bytes: &[u8]) -> Result<ReplicaEvent> {
    let CompactReplicaEvent(
        (instance_id, tp_rank, replica_id, tp_world_size, incarnation, block_size, row),
        revision,
        kind,
    ) = rmp_serde::from_slice(bytes)?;
    let kind = match kind {
        CompactReplicaKind::Upsert {
            size_bytes,
            keys,
            checksums,
        } => {
            let keys = unpack_u64s(&keys)?;
            let checksums = unpack_u64s(&checksums)?;
            anyhow::ensure!(
                checksums.is_empty() || checksums.len() == keys.len(),
                "compact upsert has {} keys and {} checksums",
                keys.len(),
                checksums.len()
            );
            ReplicaEventKind::Upsert(
                keys.into_iter()
                    .enumerate()
                    .map(|(index, key)| ReplicaBlock {
                        key,
                        size_bytes,
                        checksum: checksums.get(index).copied(),
                    })
                    .collect(),
            )
        }
        CompactReplicaKind::UpsertMixed(blocks) => ReplicaEventKind::Upsert(
            blocks
                .into_iter()
                .map(|(key, size_bytes, checksum)| ReplicaBlock {
                    key,
                    size_bytes,
                    checksum,
                })
                .collect(),
        ),
        CompactReplicaKind::Remove(keys) => ReplicaEventKind::Remove(unpack_u64s(&keys)?),
        CompactReplicaKind::RemoveSource => ReplicaEventKind::RemoveSource,
    };
    Ok(ReplicaEvent {
        source: ReplicaSource {
            key: ReplicaSourceKey {
                instance_id,
                tp_rank,
            },
            replica_id,
            tp_world_size,
            incarnation,
            block_size,
            row,
        },
        revision,
        kind,
    })
}

/// Where a key was last advertised, as a lookup returns it.
///
/// The directory keeps per source, not per key, so `block.size_bytes` is 0,
/// `block.checksum` is `None` and `revision` is the source's latest revision.
/// Verify a read against the checksum the source returns with the lease, never
/// against this.
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

    /// Where each key is advertised. See [`ReplicaLocation`] for which fields
    /// a lookup fills.
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
    /// This source's id in `by_key`, and its record, kept once rather than in
    /// every entry it advertises.
    slot: u32,
    source: Arc<ReplicaSource>,
}

/// The sources advertising one key. Almost every key has exactly one, so it is
/// held inline; a key several replicas hold keeps a small boxed list.
#[derive(Debug, Clone, PartialEq, Eq)]
enum KeyHolders {
    One(u32),
    Many(Box<[u32]>),
}

impl KeyHolders {
    fn slots(&self) -> &[u32] {
        match self {
            Self::One(slot) => std::slice::from_ref(slot),
            Self::Many(slots) => slots,
        }
    }

    fn insert(&mut self, slot: u32) {
        if self.slots().contains(&slot) {
            return;
        }
        let mut slots = self.slots().to_vec();
        slots.push(slot);
        *self = Self::Many(slots.into_boxed_slice());
    }

    /// Drop `slot`; returns whether no source is left.
    fn remove(&mut self, slot: u32) -> bool {
        let slots = self
            .slots()
            .iter()
            .copied()
            .filter(|held| *held != slot)
            .collect::<Vec<_>>();
        match slots.as_slice() {
            [] => true,
            [only] => {
                *self = Self::One(*only);
                false
            }
            _ => {
                *self = Self::Many(slots.into_boxed_slice());
                false
            }
        }
    }
}

/// Sharded exact-replica directory with reverse state per physical producer.
///
/// An entry is just "key -> which sources hold it": each source's record is
/// kept once (`by_slot`), not copied into every page it advertises, which cut
/// the directory from ~550 to ~45 bytes per entry (`tests/directory_memory.rs`).
/// The published size, checksum and per-entry revision are not kept: readers
/// verify against the checksum the source returns with a fetch, so a lookup
/// returns them as zero/none and the source's latest revision.
#[derive(Default)]
pub struct ReplicaIndex {
    by_key: DashMap<SequenceHash, KeyHolders>,
    by_source: DashMap<ReplicaSourceKey, SourceIndexState>,
    by_slot: DashMap<u32, (Arc<ReplicaSource>, u64)>,
    next_slot: std::sync::atomic::AtomicU32,
    /// Instances purged after leaving discovery. Instance ids are discovery
    /// lease ids and are never reused, so a late delta from one is dropped
    /// rather than re-creating entries nothing would purge again.
    departed: DashMap<u64, ()>,
    /// Set while this process cannot receive deltas (NATS down): lookups miss
    /// rather than serve a view that is silently going stale.
    disconnected: AtomicBool,
    /// Sources whose delta stream has a revision gap and need a snapshot.
    resync_pending: DashMap<ReplicaSourceKey, ReplicaSource>,
    /// When each source last entered `resync_pending`, to rate-limit resyncs.
    resync_requested_at: DashMap<ReplicaSourceKey, Instant>,
    resync_wakeup: tokio::sync::Notify,
    pub revision_gaps: AtomicU64,
    pub resyncs: AtomicU64,
    pub resync_failures: AtomicU64,
    /// Times this copy lost its delta stream (NATS disconnected).
    pub disconnects: AtomicU64,
    /// Entries removed because their source refused to lease them.
    pub refused_entries: AtomicU64,
}

impl ReplicaIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Start empty and subscribe in the background, retrying with jittered
    /// backoff, so a process starts (local-only, every lookup a miss) while
    /// NATS is unreachable.
    pub fn subscribe(component: &Component) -> Arc<Self> {
        let index = Arc::new(Self::new());
        let weak = Arc::downgrade(&index);
        spawn_connection_watch(component.clone(), Arc::downgrade(&index));
        let component = component.clone();
        tokio::spawn(async move {
            let mut attempt = 0u32;
            loop {
                // Single events and node-relayed batches. A source publishes on
                // one of them only, so its revisions stay ordered.
                let subscribed = async {
                    let events =
                        EventSubscriber::for_component(&component, SERVING_G2_REPLICA_TOPIC)
                            .await?;
                    let batches =
                        EventSubscriber::for_component(&component, SERVING_G2_REPLICA_BATCH_TOPIC)
                            .await?;
                    anyhow::Ok((events, batches))
                }
                .await;
                match subscribed {
                    Ok((events, batches)) => {
                        spawn_replica_subscriber(events, batches, weak);
                        return;
                    }
                    Err(error) => {
                        attempt += 1;
                        tracing::warn!(%error, attempt, "serving-G2 replica index subscribe failed; retrying");
                        tokio::time::sleep(jittered_backoff(attempt)).await;
                        if weak.strong_count() == 0 {
                            return;
                        }
                    }
                }
            }
        });
        index
    }

    pub fn apply(&self, event: ReplicaEvent) -> bool {
        let source_key = event.source.key;
        if self.departed.contains_key(&source_key.instance_id) {
            return false;
        }
        if self.out_of_slots(source_key) {
            return false;
        }
        let mut source = self
            .by_source
            .entry(source_key)
            .or_insert_with(|| self.new_source_state(&event.source));
        let slot = source.slot;
        if source.incarnation != event.source.incarnation {
            for key in std::mem::take(&mut source.keys) {
                self.remove_replica(key, slot);
            }
            source.incarnation = event.source.incarnation;
            source.revision = 0;
        }
        if source.revision >= event.revision {
            return false;
        }
        if *source.source != event.source {
            source.source = Arc::new(event.source.clone());
        }
        // Revisions are consecutive per incarnation, and a publisher burns one
        // for every batch it could not deliver, so any jump means lost deltas:
        // removals this copy will never see. A first contact past revision 1
        // (this process joined late) is the same case.
        if event.revision > source.revision + 1 {
            self.revision_gaps.fetch_add(1, Ordering::Relaxed);
            self.request_resync(&event.source);
        }
        match event.kind {
            ReplicaEventKind::Upsert(blocks) => {
                for block in blocks {
                    source.keys.insert(block.key);
                    self.upsert_replica(block.key, slot);
                }
            }
            ReplicaEventKind::Remove(keys) => {
                for key in keys {
                    source.keys.remove(&key);
                    self.remove_replica(key, slot);
                }
            }
            ReplicaEventKind::RemoveSource => {
                for key in std::mem::take(&mut source.keys) {
                    self.remove_replica(key, slot);
                }
            }
        }
        source.revision = event.revision;
        self.by_slot
            .insert(slot, (Arc::clone(&source.source), event.revision));
        drop(source);
        // The instance may have been purged while this event was applied.
        if self.departed.contains_key(&source_key.instance_id) {
            self.remove_source(source_key);
            return false;
        }
        true
    }

    /// Slots are never reused (a lookup may still hold a departed source's
    /// slot), so after 2^32 sources new ones are refused rather than given a
    /// live source's slot.
    fn out_of_slots(&self, source_key: ReplicaSourceKey) -> bool {
        let out = !self.by_source.contains_key(&source_key)
            && self.next_slot.load(std::sync::atomic::Ordering::Relaxed) == u32::MAX;
        if out {
            tracing::error!(?source_key, "replica directory is out of source slots");
        }
        out
    }

    fn new_source_state(&self, source: &ReplicaSource) -> SourceIndexState {
        let slot = self
            .next_slot
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let record = Arc::new(source.clone());
        self.by_slot.insert(slot, (Arc::clone(&record), 0));
        SourceIndexState {
            incarnation: source.incarnation,
            revision: 0,
            keys: HashSet::new(),
            slot,
            source: record,
        }
    }

    fn upsert_replica(&self, key: SequenceHash, slot: u32) {
        self.by_key
            .entry(key)
            .and_modify(|holders| holders.insert(slot))
            .or_insert(KeyHolders::One(slot));
    }

    fn remove_replica(&self, key: SequenceHash, slot: u32) {
        // Under the shard lock: drops the key once its last source leaves.
        self.by_key
            .remove_if_mut(&key, |_, holders| holders.remove(slot));
    }

    /// Drop every entry one physical producer advertised. Returns how many.
    fn remove_source(&self, source: ReplicaSourceKey) -> usize {
        let Some((_source, state)) = self.by_source.remove(&source) else {
            return 0;
        };
        let removed = state.keys.len();
        for key in state.keys {
            self.remove_replica(key, state.slot);
        }
        self.by_slot.remove(&state.slot);
        removed
    }

    fn request_resync(&self, source: &ReplicaSource) {
        let now = Instant::now();
        let mut due = false;
        self.resync_requested_at
            .entry(source.key)
            .and_modify(|at| {
                if now.duration_since(*at) >= REPLICA_RESYNC_MIN_INTERVAL {
                    *at = now;
                    due = true;
                }
            })
            .or_insert_with(|| {
                due = true;
                now
            });
        if due {
            self.resync_pending.insert(source.key, source.clone());
            self.resync_wakeup.notify_one();
        }
    }

    /// Wait until a source needs a snapshot and take it.
    pub async fn next_resync(&self) -> ReplicaSource {
        loop {
            let notified = self.resync_wakeup.notified();
            let next = self.resync_pending.iter().next().map(|entry| *entry.key());
            if let Some((_key, source)) = next.and_then(|key| self.resync_pending.remove(&key)) {
                return source;
            }
            notified.await;
        }
    }

    /// Replace everything `source` advertised with a snapshot of what it holds
    /// as of `revision`, and resume its deltas from there. Deltas newer than the
    /// snapshot that already arrived keep the revision where it is; the few
    /// keys they touched may briefly disagree with the source until their next
    /// delta or a refused lease (`forget`) corrects them.
    pub fn install_snapshot(
        &self,
        source: ReplicaSource,
        revision: u64,
        blocks: Vec<ReplicaBlock>,
    ) -> bool {
        let source_key = source.key;
        if self.departed.contains_key(&source_key.instance_id) {
            return false;
        }
        if self.out_of_slots(source_key) {
            return false;
        }
        let mut state = self
            .by_source
            .entry(source_key)
            .or_insert_with(|| self.new_source_state(&source));
        let slot = state.slot;
        if state.incarnation != source.incarnation {
            state.incarnation = source.incarnation;
            state.revision = 0;
        }
        let fresh = blocks.iter().map(|block| block.key).collect::<HashSet<_>>();
        for key in state.keys.iter().filter(|key| !fresh.contains(key)) {
            self.remove_replica(*key, slot);
        }
        for key in &fresh {
            self.upsert_replica(*key, slot);
        }
        state.keys = fresh;
        if *state.source != source {
            state.source = Arc::new(source);
        }
        state.revision = state.revision.max(revision);
        self.by_slot
            .insert(slot, (Arc::clone(&state.source), state.revision));
        drop(state);
        self.resyncs.fetch_add(1, Ordering::Relaxed);
        if self.departed.contains_key(&source_key.instance_id) {
            self.remove_source(source_key);
            return false;
        }
        true
    }

    /// Drop `key` from `source`'s entries: the source refused to lease it, so
    /// the delta that withdrew it was lost. A later upsert re-adds it.
    pub fn forget(&self, key: SequenceHash, source: ReplicaSourceKey) -> bool {
        let slot = self
            .by_source
            .get_mut(&source)
            .and_then(|mut state| state.keys.remove(&key).then_some(state.slot));
        if let Some(slot) = slot {
            self.remove_replica(key, slot);
            self.refused_entries.fetch_add(1, Ordering::Relaxed);
        }
        slot.is_some()
    }

    pub fn counters(&self) -> Vec<(&'static str, u64)> {
        vec![
            (
                "index_revision_gaps",
                self.revision_gaps.load(Ordering::Relaxed),
            ),
            ("index_resyncs", self.resyncs.load(Ordering::Relaxed)),
            (
                "index_resync_failures",
                self.resync_failures.load(Ordering::Relaxed),
            ),
            (
                "index_refused_entries",
                self.refused_entries.load(Ordering::Relaxed),
            ),
            (
                "index_disconnects",
                self.disconnects.load(Ordering::Relaxed),
            ),
        ]
    }

    /// Lookups miss while disconnected. Cumulative transitions into that
    /// state are counted, so the stat exports as a counter.
    pub fn set_disconnected(&self, disconnected: bool) {
        if !self.disconnected.swap(disconnected, Ordering::Relaxed) && disconnected {
            self.disconnects.fetch_add(1, Ordering::Relaxed);
        }
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
        let slots = match self.by_key.get(&key) {
            Some(holders) => holders.slots().to_vec(),
            None => return Vec::new(),
        };
        let mut result = slots
            .into_iter()
            .filter_map(|slot| {
                let entry = self.by_slot.get(&slot)?;
                let (source, revision) = entry.value();
                block_size
                    .is_none_or(|expected| source.block_size == expected)
                    .then(|| ReplicaLocation {
                        source: ReplicaSource::clone(source),
                        block: ReplicaBlock {
                            key,
                            size_bytes: 0,
                            checksum: None,
                        },
                        revision: *revision,
                    })
            })
            .collect::<Vec<_>>();
        result.sort_unstable_by_key(|location| {
            (location.source.key.instance_id, location.source.key.tp_rank)
        });
        result
    }

    /// One sharded map lookup per physical key. Output order matches input.
    /// Every key misses while this copy is disconnected from the delta stream.
    pub fn batch_locations(
        &self,
        keys: &[SequenceHash],
        block_size: Option<u32>,
    ) -> Vec<Vec<ReplicaLocation>> {
        if self.disconnected.load(Ordering::Relaxed) {
            return vec![Vec::new(); keys.len()];
        }
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
    let index = ReplicaIndex::subscribe(component);
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
    shared: Arc<PublisherShared>,
}

#[derive(Default)]
struct PublisherShared {
    /// Last revision the batcher assigned, delivered or not.
    revision: AtomicU64,
    /// An update was dropped: at this queue, or (shared with the node relay's
    /// sender) on its way to the replica's owner. The batcher burns a revision
    /// for it, so readers see the loss as a gap and resync.
    lost: Arc<AtomicBool>,
}

impl ReplicaPublisher {
    /// Never blocks on NATS: while it is down batches are skipped, and every
    /// skipped batch leaves a revision gap that makes readers resync from a
    /// snapshot (see `ReplicaIndex::install_snapshot`).
    pub async fn new(component: &Component, source: ReplicaSource) -> Result<Arc<Self>> {
        let config = ReplicaPublishConfig::from_env()?;
        tracing::info!(?config, "serving-G2 replica publisher");
        let publisher = Arc::new(
            EventPublisher::for_component(component, SERVING_G2_REPLICA_TOPIC)
                .await
                .context("failed to create replica event publisher")?,
        );
        let component = component.clone();
        Ok(Self::spawn(
            source,
            move |event| {
                let publisher = Arc::clone(&publisher);
                async move {
                    publisher
                        .publish_bytes(encode_compact_replica_event(&event)?)
                        .await
                }
            },
            move || nats_connected(&component),
            config,
        ))
    }

    /// Number this source's deltas as usual, but hand each finished event to
    /// the node-index owner, which publishes the replica's events in one batch
    /// per window. The NATS gate still applies: while NATS is down the owner
    /// could not publish either, and the skipped revisions read as a gap.
    pub fn through_relay(
        component: &Component,
        source: ReplicaSource,
        relay: RelaySender,
    ) -> Result<Arc<Self>> {
        let config = ReplicaPublishConfig::from_env()?;
        tracing::info!(
            ?config,
            "serving-G2 replica publisher (through the node relay)"
        );
        let component = component.clone();
        let lost = relay.lost_flag();
        Ok(Self::spawn_with_lost(
            source,
            move |event| {
                std::future::ready(if relay.send(event) {
                    Ok(())
                } else {
                    Err(anyhow::anyhow!("node relay queue rejected an event"))
                })
            },
            move || nats_connected(&component),
            config,
            lost,
        ))
    }

    fn spawn<P, F>(
        source: ReplicaSource,
        publish: P,
        connected: impl Fn() -> bool + Send + Sync + 'static,
        config: ReplicaPublishConfig,
    ) -> Arc<Self>
    where
        P: Fn(ReplicaEvent) -> F + Send + Sync + 'static,
        F: std::future::Future<Output = Result<()>> + Send + 'static,
    {
        Self::spawn_with_lost(source, publish, connected, config, Arc::default())
    }

    /// As `spawn`, with a `lost` flag shared with whatever delivers events.
    fn spawn_with_lost<P, F>(
        source: ReplicaSource,
        publish: P,
        connected: impl Fn() -> bool + Send + Sync + 'static,
        config: ReplicaPublishConfig,
        lost: Arc<AtomicBool>,
    ) -> Arc<Self>
    where
        P: Fn(ReplicaEvent) -> F + Send + Sync + 'static,
        F: std::future::Future<Output = Result<()>> + Send + 'static,
    {
        let (tx, rx) = mpsc::channel(REPLICA_PUBLISH_QUEUE_CAPACITY);
        let shared = Arc::new(PublisherShared {
            revision: AtomicU64::new(0),
            lost,
        });
        tokio::spawn(run_replica_publisher(
            source,
            publish,
            connected,
            config,
            Arc::clone(&shared),
            rx,
        ));
        Arc::new(Self { tx, shared })
    }

    pub fn upsert(&self, blocks: Vec<ReplicaBlock>) -> Result<()> {
        self.publish(ReplicaEventKind::Upsert(blocks))
    }

    pub fn remove(&self, keys: Vec<SequenceHash>) -> Result<()> {
        self.publish(ReplicaEventKind::Remove(keys))
    }

    /// Last revision assigned. Read it before snapshotting the local index:
    /// every change already in that snapshot then arrives at a later revision,
    /// where re-applying it is harmless.
    pub fn revision(&self) -> u64 {
        self.shared.revision.load(Ordering::Acquire)
    }

    fn publish(&self, kind: ReplicaEventKind) -> Result<()> {
        self.tx.try_send(kind).map_err(|error| {
            self.shared.lost.store(true, Ordering::Release);
            anyhow::anyhow!("replica publish queue rejected an event: {error}")
        })
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

/// Assign the next revision and try to deliver it. Returns whether it was
/// delivered. An undelivered revision is never reused, which is what makes the
/// loss visible to readers.
async fn publish_replica_event<F>(
    publish: &impl Fn(ReplicaEvent) -> F,
    connected: &impl Fn() -> bool,
    source: &ReplicaSource,
    shared: &PublisherShared,
    kind: ReplicaEventKind,
) -> bool
where
    F: std::future::Future<Output = Result<()>>,
{
    let revision = shared.revision.fetch_add(1, Ordering::AcqRel) + 1;
    if !connected() {
        return false;
    }
    let event = ReplicaEvent {
        source: source.clone(),
        revision,
        kind,
    };
    match tokio::time::timeout(REPLICA_PUBLISH_TIMEOUT, publish(event)).await {
        Ok(Ok(())) => true,
        Ok(Err(error)) => {
            tracing::warn!(%error, revision, "failed to publish replica event batch");
            false
        }
        Err(_) => {
            tracing::warn!(revision, "replica event publish timed out");
            false
        }
    }
}

async fn run_replica_publisher<P, F>(
    source: ReplicaSource,
    publish: P,
    connected: impl Fn() -> bool,
    config: ReplicaPublishConfig,
    shared: Arc<PublisherShared>,
    mut rx: mpsc::Receiver<ReplicaEventKind>,
) where
    P: Fn(ReplicaEvent) -> F,
    F: std::future::Future<Output = Result<()>>,
{
    // A revision readers have not seen followed by nothing they have: until
    // some later event reaches them they cannot know they missed anything.
    let mut gap_unannounced = false;
    let mut tick = tokio::time::interval(REPLICA_PUBLISHER_TICK);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        let first = tokio::select! {
            next = rx.recv() => match next {
                Some(first) => Some(first),
                None => break,
            },
            _ = tick.tick() => None,
        };
        if shared.lost.swap(false, Ordering::AcqRel) {
            shared.revision.fetch_add(1, Ordering::AcqRel);
            gap_unannounced = true;
        }
        let Some(first) = first else {
            if gap_unannounced && connected() {
                // An empty upsert carries only the revision, exposing the gap.
                gap_unannounced = !publish_replica_event(
                    &publish,
                    &connected,
                    &source,
                    &shared,
                    ReplicaEventKind::Upsert(Vec::new()),
                )
                .await;
            }
            continue;
        };
        let mut pending = PendingReplicaDelta::default();
        pending.absorb(first);

        let deadline = tokio::time::sleep(config.batch_deadline);
        tokio::pin!(deadline);
        while pending.changes.len() < config.batch_max_blocks {
            tokio::select! {
                _ = &mut deadline => break,
                next = rx.recv() => match next {
                    Some(delta) => pending.absorb(delta),
                    None => break,
                }
            }
        }

        for kind in pending.into_kinds(config.batch_max_blocks) {
            gap_unannounced =
                !publish_replica_event(&publish, &connected, &source, &shared, kind).await;
        }
    }
    publish_replica_event(
        &publish,
        &connected,
        &source,
        &shared,
        ReplicaEventKind::RemoveSource,
    )
    .await;
}

/// Equal-jitter exponential backoff, 100 ms doubling up to
/// `REPLICA_SUBSCRIBE_RETRY_MAX`.
fn jittered_backoff(attempt: u32) -> Duration {
    let ceiling = Duration::from_millis(100)
        .saturating_mul(1 << attempt.saturating_sub(1).min(16))
        .min(REPLICA_SUBSCRIBE_RETRY_MAX);
    ceiling / 2 + ceiling.mul_f64(rand::random::<f64>() / 2.0)
}

/// Whether the replica events' NATS transport is connected; true for another
/// event transport, which does not report connectivity here.
fn nats_connected(component: &Component) -> bool {
    let drt = component.drt();
    drt.default_event_transport_kind() != dynamo_runtime::discovery::EventTransportKind::Nats
        || drt.nats_connected().unwrap_or(true)
}

/// Mirror NATS connectivity onto the index: while disconnected every lookup
/// misses. Deltas lost meanwhile show up as revision gaps once they resume.
fn spawn_connection_watch(component: Component, index: std::sync::Weak<ReplicaIndex>) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(REPLICA_PUBLISHER_TICK);
        let mut was_connected = true;
        loop {
            interval.tick().await;
            let Some(index) = index.upgrade() else {
                return;
            };
            let connected = nats_connected(&component);
            if connected != was_connected {
                tracing::warn!(
                    connected,
                    "serving-G2 replica index NATS connectivity changed"
                );
                was_connected = connected;
            }
            index.set_disconnected(!connected);
        }
    });
}

fn spawn_replica_subscriber(
    mut events: EventSubscriber,
    mut batches: EventSubscriber,
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

    let route = move |event: ReplicaEvent| {
        let sticky = event.source.key.instance_id as usize ^ event.source.key.tp_rank as usize;
        workers[sticky % workers.len()].send(event).is_ok()
    };
    tokio::spawn(async move {
        loop {
            // A batch's events go to the same per-source workers as single
            // events, so each source's revisions still apply in order.
            let decoded = tokio::select! {
                Some(envelope) = events.next() => envelope
                    .and_then(|envelope| decode_compact_replica_event(&envelope.payload))
                    .map(|event| vec![event]),
                Some(envelope) = batches.next() => envelope
                    .and_then(|envelope| decode_compact_replica_batch(&envelope.payload)),
                else => break,
            };
            match decoded {
                Ok(events) => {
                    if !events.into_iter().all(&route) {
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
    fn key_holders_track_several_sources_and_shrink_back() {
        let mut holders = KeyHolders::One(3);
        holders.insert(3);
        assert_eq!(holders, KeyHolders::One(3));
        holders.insert(5);
        holders.insert(9);
        assert_eq!(holders.slots(), &[3, 5, 9]);
        assert!(!holders.remove(5));
        assert_eq!(holders.slots(), &[3, 9]);
        assert!(!holders.remove(3));
        assert_eq!(holders, KeyHolders::One(9));
        assert!(!holders.remove(4));
        assert!(holders.remove(9));
    }

    #[test]
    fn a_shared_key_keeps_its_other_sources_and_each_source_its_latest_record() {
        let index = ReplicaIndex::new();
        for instance in [7, 9] {
            assert!(index.apply(ReplicaEvent {
                source: source(instance, 0, 1),
                revision: 1,
                kind: ReplicaEventKind::Upsert(vec![block(11)]),
            }));
        }
        assert_eq!(index.locations(11, None).len(), 2);
        assert!(index.apply(ReplicaEvent {
            source: source(7, 0, 1),
            revision: 2,
            kind: ReplicaEventKind::Remove(vec![11]),
        }));
        let left = index.locations(11, None);
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].source.key.instance_id, 9);
        // A record change (here the row) is seen on lookups without re-adding.
        let mut moved = source(9, 0, 1);
        moved.row = Some("row-b".into());
        assert!(index.apply(ReplicaEvent {
            source: moved,
            revision: 2,
            kind: ReplicaEventKind::Upsert(vec![block(12)]),
        }));
        let at_11 = index.locations(11, None);
        assert_eq!(at_11[0].source.row.as_deref(), Some("row-b"));
        assert_eq!(at_11[0].revision, 2);
        // The last source leaving removes the key.
        assert!(index.apply(ReplicaEvent {
            source: source(9, 0, 1),
            revision: 3,
            kind: ReplicaEventKind::RemoveSource,
        }));
        assert!(index.locations(11, None).is_empty());
        assert!(index.locations(12, None).is_empty());
        assert_eq!(index.entry_count(), 0);
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

    #[tokio::test]
    async fn revision_gap_and_late_join_request_one_rate_limited_resync() {
        let index = ReplicaIndex::new();
        let src = source(7, 0, 1);
        let event = |revision, key| ReplicaEvent {
            source: src.clone(),
            revision,
            kind: ReplicaEventKind::Upsert(vec![block(key)]),
        };
        assert!(index.apply(event(1, 11)));
        assert!(index.apply(event(2, 12)));
        assert_eq!(index.revision_gaps.load(Ordering::Relaxed), 0);
        assert!(index.resync_pending.is_empty());

        // Revision 3 was lost: the delta still applies, and a resync is queued.
        assert!(index.apply(event(4, 14)));
        assert_eq!(index.revision_gaps.load(Ordering::Relaxed), 1);
        assert_eq!(index.locations(14, None).len(), 1);
        assert_eq!(index.next_resync().await, src);

        // Another gap inside the rate-limit window is counted, not re-queued.
        assert!(index.apply(event(6, 16)));
        assert_eq!(index.revision_gaps.load(Ordering::Relaxed), 2);
        assert!(index.resync_pending.is_empty());

        // A source first seen past revision 1 (this copy joined late) is a gap.
        let late = source(9, 0, 2);
        assert!(index.apply(ReplicaEvent {
            source: late.clone(),
            revision: 40,
            kind: ReplicaEventKind::Upsert(vec![block(11)]),
        }));
        assert_eq!(index.next_resync().await, late);
    }

    #[test]
    fn snapshot_replaces_a_sources_entries_and_resumes_its_deltas() {
        let index = ReplicaIndex::new();
        let src = source(7, 0, 1);
        upsert(&index, 7, 0, &[11, 12]);
        upsert(&index, 9, 0, &[12]);

        // 12 was withdrawn in a lost delta; 13 was published in another.
        assert!(index.install_snapshot(src.clone(), 10, vec![block(11), block(13)]));
        assert_eq!(index.locations(11, None).len(), 1);
        assert_eq!(
            index.locations(12, None)[0].source.key.instance_id,
            9,
            "only the other source's entry is left"
        );
        assert_eq!(index.locations(13, None).len(), 1);
        assert_eq!(index.resyncs.load(Ordering::Relaxed), 1);

        // Deltas the snapshot covers are dropped; the next one applies, no gap.
        assert!(!index.apply(ReplicaEvent {
            source: src.clone(),
            revision: 10,
            kind: ReplicaEventKind::Upsert(vec![block(12)]),
        }));
        assert!(index.apply(ReplicaEvent {
            source: src.clone(),
            revision: 11,
            kind: ReplicaEventKind::Remove(vec![11]),
        }));
        assert!(index.locations(11, None).is_empty());
        assert_eq!(index.revision_gaps.load(Ordering::Relaxed), 0);

        // A departed instance's snapshot is ignored.
        index.purge_instance(7);
        assert!(!index.install_snapshot(src, 20, vec![block(11)]));
        assert!(index.locations(11, None).is_empty());
    }

    #[test]
    fn forget_drops_one_sources_entry_and_a_later_upsert_restores_it() {
        let index = ReplicaIndex::new();
        upsert(&index, 7, 0, &[11]);
        upsert(&index, 9, 0, &[11]);
        let refused = source(7, 0, 7).key;
        assert!(index.forget(11, refused));
        assert!(!index.forget(11, refused));
        assert!(!index.forget(99, refused));
        let left = index.locations(11, None);
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].source.key.instance_id, 9);
        assert_eq!(index.refused_entries.load(Ordering::Relaxed), 1);
        // The source's reverse state forgot it too: a source removal is exact.
        index.apply(ReplicaEvent {
            source: source(7, 0, 7),
            revision: 2,
            kind: ReplicaEventKind::Upsert(vec![block(11)]),
        });
        assert_eq!(index.locations(11, None).len(), 2);
    }

    #[test]
    fn lookups_miss_while_disconnected() {
        let index = ReplicaIndex::new();
        upsert(&index, 7, 0, &[11]);
        index.set_disconnected(true);
        index.set_disconnected(true);
        assert_eq!(index.batch_locations(&[11, 12], None), vec![vec![], vec![]]);
        index.set_disconnected(false);
        assert_eq!(index.batch_locations(&[11], None)[0].len(), 1);
        assert_eq!(index.disconnects.load(Ordering::Relaxed), 1);
    }

    /// A publisher whose deliveries land in a vector, behind a connectivity flag.
    fn test_publisher() -> (
        Arc<ReplicaPublisher>,
        Arc<std::sync::Mutex<Vec<ReplicaEvent>>>,
        Arc<AtomicBool>,
    ) {
        test_publisher_with(ReplicaPublishConfig::default())
    }

    /// Records what readers would decode: under v2 every event goes through
    /// the compact encoding and back.
    fn test_publisher_with(
        config: ReplicaPublishConfig,
    ) -> (
        Arc<ReplicaPublisher>,
        Arc<std::sync::Mutex<Vec<ReplicaEvent>>>,
        Arc<AtomicBool>,
    ) {
        let sent = Arc::new(std::sync::Mutex::new(Vec::new()));
        let connected = Arc::new(AtomicBool::new(true));
        let publisher = ReplicaPublisher::spawn(
            source(7, 0, 1),
            {
                let sent = Arc::clone(&sent);
                move |event| {
                    // Through the wire encoding and back, as a reader sees it.
                    let event = encode_compact_replica_event(&event)
                        .and_then(|bytes| decode_compact_replica_event(&bytes));
                    let result = event.map(|event| sent.lock().unwrap().push(event));
                    async { result }
                }
            },
            {
                let connected = Arc::clone(&connected);
                move || connected.load(Ordering::Relaxed)
            },
            config,
        );
        (publisher, sent, connected)
    }

    #[test]
    fn compact_batch_round_trips_every_source_in_order() {
        let events = (0..3u32)
            .flat_map(|rank| {
                (1..=2u64).map(move |revision| ReplicaEvent {
                    source: source(100 + u64::from(rank), rank, 1),
                    revision,
                    kind: ReplicaEventKind::Upsert(vec![ReplicaBlock {
                        key: u64::from(rank) * 10 + revision,
                        size_bytes: 4096,
                        checksum: Some(revision),
                    }]),
                })
            })
            .collect::<Vec<_>>();
        let encoded = encode_compact_replica_batch(&events).unwrap();
        assert_eq!(decode_compact_replica_batch(&encoded).unwrap(), events);
    }

    #[tokio::test(start_paused = true)]
    async fn skipped_batch_reaches_readers_as_a_gap_through_the_wire() {
        let (publisher, sent, connected) = test_publisher_with(ReplicaPublishConfig::default());
        publisher.upsert(vec![block(11)]).unwrap();
        settle().await;
        connected.store(false, Ordering::Relaxed);
        publisher.remove(vec![11]).unwrap();
        settle().await;
        connected.store(true, Ordering::Relaxed);
        settle().await;
        let events = sent.lock().unwrap().clone();
        assert_eq!(
            events
                .iter()
                .map(|event| event.revision)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
        // The gap announcement survives the compact encoding as an empty upsert.
        assert!(matches!(&events[1].kind, ReplicaEventKind::Upsert(blocks) if blocks.is_empty()));
        let index = ReplicaIndex::new();
        for event in events {
            index.apply(event);
        }
        assert_eq!(index.revision_gaps.load(Ordering::Relaxed), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn configured_batch_window_coalesces_updates() {
        let (publisher, sent, _connected) = test_publisher_with(ReplicaPublishConfig {
            batch_deadline: Duration::from_millis(5_000),
            ..ReplicaPublishConfig::default()
        });
        publisher.upsert(vec![block(11)]).unwrap();
        tokio::time::sleep(Duration::from_millis(2_000)).await;
        publisher.upsert(vec![block(12)]).unwrap();
        tokio::time::sleep(Duration::from_millis(1_000)).await;
        assert!(sent.lock().unwrap().is_empty());
        tokio::time::sleep(Duration::from_millis(3_000)).await;
        let events = sent.lock().unwrap().clone();
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0].kind, ReplicaEventKind::Upsert(blocks) if blocks.len() == 2));
    }

    async fn settle() {
        tokio::time::sleep(REPLICA_PUBLISHER_TICK * 2).await;
    }

    #[tokio::test(start_paused = true)]
    async fn batches_skipped_while_disconnected_reach_readers_as_a_gap() {
        let (publisher, sent, connected) = test_publisher();
        publisher.upsert(vec![block(11)]).unwrap();
        settle().await;
        connected.store(false, Ordering::Relaxed);
        // Never blocks, never queues behind NATS: the batch is skipped.
        publisher.remove(vec![11]).unwrap();
        settle().await;
        assert_eq!(sent.lock().unwrap().len(), 1);
        assert_eq!(publisher.revision(), 2);

        // Back up and idle: an empty upsert announces the skipped revision.
        connected.store(true, Ordering::Relaxed);
        settle().await;
        let events = sent.lock().unwrap().clone();
        assert_eq!(events.len(), 2);
        assert_eq!(events[1].revision, 3);
        assert!(matches!(&events[1].kind, ReplicaEventKind::Upsert(blocks) if blocks.is_empty()));

        let index = ReplicaIndex::new();
        for event in events {
            index.apply(event);
        }
        assert_eq!(index.revision_gaps.load(Ordering::Relaxed), 1);
        // Until the resync lands the reader still holds the stale entry; the
        // snapshot (or a refused lease) is what removes it.
        assert_eq!(index.locations(11, None).len(), 1);

        // Nothing more is sent once the gap is announced.
        settle().await;
        assert_eq!(sent.lock().unwrap().len(), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn an_update_dropped_at_the_queue_burns_a_revision() {
        let (publisher, sent, _connected) = test_publisher();
        publisher.upsert(vec![block(11)]).unwrap();
        settle().await;
        publisher.shared.lost.store(true, Ordering::Relaxed);
        settle().await;
        let events = sent.lock().unwrap().clone();
        assert_eq!(
            events
                .iter()
                .map(|event| event.revision)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
    }

    /// The node relay's writer drops an event after the publisher handed it
    /// over (owner unreachable): the shared `lost` flag makes the publisher
    /// burn a revision and announce the gap on its next tick.
    #[tokio::test(start_paused = true)]
    async fn a_relay_drop_after_handoff_burns_a_revision_and_announces_the_gap() {
        let sent = Arc::new(std::sync::Mutex::new(Vec::new()));
        let lost = Arc::new(AtomicBool::new(false));
        let publisher = ReplicaPublisher::spawn_with_lost(
            source(7, 0, 1),
            {
                let sent = Arc::clone(&sent);
                move |event: ReplicaEvent| {
                    sent.lock().unwrap().push(event);
                    async { Ok(()) }
                }
            },
            || true,
            ReplicaPublishConfig::default(),
            Arc::clone(&lost),
        );
        publisher.upsert(vec![block(11)]).unwrap();
        settle().await;
        // The relay writer could not reach the owner with revision 1.
        lost.store(true, Ordering::Release);
        tokio::time::sleep(REPLICA_PUBLISHER_TICK * 2).await;
        let events = sent.lock().unwrap().clone();
        assert_eq!(
            events
                .iter()
                .map(|event| (event.revision, event.kind.clone()))
                .collect::<Vec<_>>(),
            vec![
                (1, ReplicaEventKind::Upsert(vec![block(11)])),
                (3, ReplicaEventKind::Upsert(Vec::new())),
            ]
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_stalled_publish_times_out_instead_of_backing_up_the_queue() {
        let delivered = Arc::new(AtomicU64::new(0));
        let publisher = ReplicaPublisher::spawn(
            source(7, 0, 1),
            {
                let delivered = Arc::clone(&delivered);
                move |event: ReplicaEvent| {
                    let delivered = Arc::clone(&delivered);
                    async move {
                        if event.revision == 1 {
                            std::future::pending::<()>().await;
                        }
                        delivered.fetch_add(1, Ordering::Relaxed);
                        Ok(())
                    }
                }
            },
            || true,
            ReplicaPublishConfig::default(),
        );
        publisher.upsert(vec![block(11)]).unwrap();
        settle().await;
        // Revision 1 timed out; an idle tick announced the gap at revision 2.
        assert_eq!(delivered.load(Ordering::Relaxed), 1);
        assert_eq!(publisher.revision(), 2);
        publisher.upsert(vec![block(12)]).unwrap();
        settle().await;
        assert_eq!(delivered.load(Ordering::Relaxed), 2);
        assert_eq!(publisher.revision(), 3);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn starts_local_only_when_nats_is_unreachable() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let server = format!("nats://127.0.0.1:{}", listener.local_addr().unwrap().port());
        drop(listener);
        let mut config = dynamo_runtime::distributed::DistributedConfig::process_local();
        config.nats_config = Some(
            dynamo_runtime::transports::nats::ClientOptions::builder()
                .server(server)
                .build()
                .unwrap()
                .with_retry_on_initial_connect(),
        );
        config.event_transport_kind = dynamo_runtime::discovery::EventTransportKind::Nats;
        let runtime =
            DistributedRuntime::new(dynamo_runtime::Runtime::from_current().unwrap(), config)
                .await
                .expect("a worker starts while NATS is down");
        assert_eq!(runtime.nats_connected(), Some(false));

        let component = serving_g2_namespace(&runtime, "test")
            .unwrap()
            .component(SERVING_G2_INDEX_COMPONENT)
            .unwrap();
        let publisher = ReplicaPublisher::new(&component, source(7, 0, 1))
            .await
            .unwrap();
        let index = start_replica_index_service(&component).await.unwrap();
        upsert(&index, 9, 0, &[11]);
        tokio::time::sleep(REPLICA_PUBLISHER_TICK * 2).await;
        assert!(index.batch_locations(&[11], None)[0].is_empty());
        assert!(index.counters().contains(&("index_disconnects", 1)));
        // Batches are skipped, never queued behind the dead connection, so the
        // queue keeps draining: several queues' worth of updates are accepted.
        let burst = REPLICA_PUBLISH_QUEUE_CAPACITY as u64 / 2;
        for round in 0..6 {
            for key in round * burst..(round + 1) * burst {
                publisher.upsert(vec![block(key)]).unwrap();
            }
            tokio::time::sleep(Duration::from_millis(REPLICA_BATCH_DEADLINE_MS * 4)).await;
        }
        assert!(publisher.revision() >= 6);
        runtime.shutdown();
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

    fn compact_round_trip(event: ReplicaEvent) -> usize {
        let bytes = encode_compact_replica_event(&event).expect("encode compact");
        assert_eq!(decode_compact_replica_event(&bytes).expect("decode"), event);
        bytes.len()
    }

    #[test]
    fn compact_wire_round_trips_every_event_shape() {
        let mut with_row = source(7, 1, 3);
        with_row.row = Some("row04".to_string());
        let mut legacy = source(9, 0, 4);
        legacy.replica_id = None;
        for source in [with_row, legacy] {
            let mut no_checksum = block(13);
            no_checksum.checksum = None;
            let mut bigger = block(14);
            bigger.size_bytes += 1;
            for kind in [
                ReplicaEventKind::Upsert(vec![block(11), block(12)]),
                ReplicaEventKind::Upsert(vec![no_checksum.clone()]),
                ReplicaEventKind::Upsert(vec![block(11), no_checksum]),
                ReplicaEventKind::Upsert(vec![block(11), bigger]),
                ReplicaEventKind::Upsert(Vec::new()),
                ReplicaEventKind::Remove(vec![11, u64::MAX]),
                ReplicaEventKind::RemoveSource,
            ] {
                compact_round_trip(ReplicaEvent {
                    source: source.clone(),
                    revision: 42,
                    kind,
                });
            }
        }
    }

    #[test]
    fn compact_wire_rejects_truncated_packed_arrays() {
        let bytes = rmp_serde::to_vec(&CompactReplicaEvent(
            (7, 0, None, 1, Uuid::nil(), 64, None),
            1,
            CompactReplicaKind::Remove(vec![0; 7]),
        ))
        .unwrap();
        assert!(decode_compact_replica_event(&bytes).is_err());
        let bytes = rmp_serde::to_vec(&CompactReplicaEvent(
            (7, 0, None, 1, Uuid::nil(), 64, None),
            1,
            CompactReplicaKind::Upsert {
                size_bytes: 1,
                keys: vec![0; 16],
                checksums: vec![0; 8],
            },
        ))
        .unwrap();
        assert!(decode_compact_replica_event(&bytes).is_err());
    }

    /// Payload bytes of the old named encoding (`to_vec_named`, what the event
    /// plane codec emits) against v2, for realistic hash-valued keys. Run with
    /// `--nocapture` for the table quoted in the PR.
    #[test]
    fn compact_wire_is_at_least_2x_smaller_for_upserts() {
        let mut source = source(0x9e37_79b9_7f4a_7c15, 3, 5);
        source.tp_world_size = 8;
        source.replica_id = Some(0xc2b2_ae3d_27d4_eb4f);
        source.row = Some("row04".to_string());
        // Sequence hashes and checksums are uniformly distributed u64s.
        let hash = |seed: u64| compute_hash_v2(&seed.to_le_bytes(), 0);
        // Fixed per-message cost of the event-plane envelope around either payload.
        let envelope = dynamo_runtime::transports::event_plane::EventEnvelope {
            publisher_id: hash(1),
            sequence: 1 << 40,
            published_at: 1_792_686_000_000,
            topic: SERVING_G2_REPLICA_TOPIC.to_string(),
            payload: Default::default(),
        };
        println!(
            "envelope overhead: {} B/message",
            rmp_serde::to_vec_named(&envelope).unwrap().len()
        );
        for count in [1u64, 64, 512, 4096] {
            let upsert = ReplicaEvent {
                source: source.clone(),
                revision: 1_000_000,
                kind: ReplicaEventKind::Upsert(
                    (0..count)
                        .map(|seed| ReplicaBlock {
                            key: hash(seed),
                            size_bytes: 2_359_296,
                            checksum: Some(hash(seed + 1_000_000)),
                        })
                        .collect(),
                ),
            };
            let remove = ReplicaEvent {
                source: source.clone(),
                revision: 1_000_001,
                kind: ReplicaEventKind::Remove((0..count).map(hash).collect()),
            };
            for (name, event, max_ratio) in [("upsert", upsert, 0.5), ("remove", remove, 1.0)] {
                let v1 = rmp_serde::to_vec_named(&event).unwrap().len();
                let v2 = compact_round_trip(event);
                println!(
                    "{name:6} {count:4} blocks: v1 {v1:6} B ({:.1} B/block)  v2 {v2:6} B ({:.1} B/block)  {:.2}x",
                    v1 as f64 / count as f64,
                    v2 as f64 / count as f64,
                    v1 as f64 / v2 as f64
                );
                assert!(
                    (v2 as f64) < v1 as f64 * max_ratio,
                    "{name} {count}: {v2} vs {v1}"
                );
                // v2 per-block cost is the packed payload plus a fixed header.
                let per_block = if name == "upsert" { 16 } else { 8 };
                assert!(
                    v2 <= per_block * count as usize + 96,
                    "{name} {count}: {v2}"
                );
            }
        }
    }

    #[test]
    fn publish_config_defaults_and_env_overrides() {
        let defaults = ReplicaPublishConfig::parse(None, None).unwrap();
        assert_eq!(
            defaults,
            ReplicaPublishConfig {
                batch_deadline: Duration::from_millis(50),
                batch_max_blocks: 4_096,
            }
        );
        assert_eq!(
            ReplicaPublishConfig::parse(Some(""), Some("")).unwrap(),
            defaults
        );
        assert_eq!(
            ReplicaPublishConfig::parse(Some("250"), Some("16384")).unwrap(),
            ReplicaPublishConfig {
                batch_deadline: Duration::from_millis(250),
                batch_max_blocks: 16_384,
            }
        );
        assert!(ReplicaPublishConfig::parse(Some("-1"), None).is_err());
        assert!(ReplicaPublishConfig::parse(None, Some("0")).is_err());
        assert!(ReplicaPublishConfig::parse(None, Some("lots")).is_err());
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
