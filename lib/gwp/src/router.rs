// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The GWP approximate router: a `KvRouter` using etcd discovery and a
//! configurable event plane for cross-replica active-sequence synchronization.
//!
//! This is the critical-path reuse from the design doc (Decision 1), resolved
//! more simply than the `MockDiscovery` sketch: `RuntimeConfigWatch` is just a
//! `watch::Receiver<HashMap<WorkerId, ModelRuntimeConfig>>`, so the topology
//! registry owns one sender and one `KvRouter` per canonical model and feeds
//! each only its eligible workers. The feeds remain local because every
//! replica observes the same topology, while one shared `DistributedRuntime`
//! uses etcd to discover peer model routers and ZMQ or NATS to exchange
//! model-scoped request lifecycle events.
//!
//! Config choices (see `gwp_kv_router_config`):
//! - `use_kv_events: false` — the primary indexer is a local prune-TTL'd radix
//!   tree populated by `record_routing_decision` (the "approximate indexer");
//!   remote endpoints never publish KV events to us.
//! - `skip_initial_worker_wait: false` — REQUIRED for the topology feed: this
//!   flag doubles as "watch worker configs" in `KvScheduler::start`
//!   (`scheduler.rs:94`); setting it to true would freeze the worker set at
//!   construction time. With `DYN_ROUTER_MIN_INITIAL_WORKERS` unset the
//!   constructor does not block on an empty feed.
//! - `router_snapshot_threshold: None` — snapshots go to the NATS object
//!   store, which GWP does not have.
//! - `router_queue_threshold: None` — GWP never parks requests; downstream
//!   endpoints do their own queueing.
//! - `router_replica_sync: true` — replicas exchange add/prefill/free events
//!   over the configured event plane so every selector sees global GWP
//!   in-flight load.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Weak};

use dashmap::DashMap;
use dynamo_kv_router::config::{KvRouterConfig, RouterConfigOverride};
use dynamo_kv_router::indexer::RoutingDecisionHashes;
#[cfg(test)]
use dynamo_kv_router::protocols::BlockHashOptions;
use dynamo_kv_router::protocols::{
    LocalBlockHash, RoutingConstraints, WorkerId, WorkerWithDpRank, compute_block_hash_for_seq,
};
use dynamo_llm::kv_router::{
    ACTIVE_SEQUENCES_SUBJECT, FindBestMatchOutcome, KvRouter,
    metrics::register_global_metrics_with_component,
};
use dynamo_llm::local_model::runtime_config::ModelRuntimeConfig;
use dynamo_runtime::discovery::{
    DiscoveryInstance, DiscoveryQuery, EventChannelQuery, EventTransportKind,
};
use dynamo_runtime::distributed::{DiscoveryBackend, DistributedConfig, RequestPlaneMode};
use dynamo_runtime::metrics::MetricsHierarchy;
use dynamo_runtime::slug::Slug;
use dynamo_runtime::storage::kv;
use dynamo_runtime::transports::event_plane::{
    EventEnvelope, EventPublisher, EventSubscriber, TypedEventSubscriber,
};
use dynamo_runtime::{DistributedRuntime, Runtime};
use serde::{Deserialize, Serialize};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use crate::{
    config::{EndpointId, ModelStagePolicy},
    metrics::GwpMetrics,
    scoring::LocalLoadAnchor,
    worker_selector::{GwpWorkerSelector, SelectionPolicyStore},
};

pub use crate::scoring::{ObservedLoadStore, ObservedWorkerLoad};

/// The sender half of the worker feed. Owned by the topology controller, which
/// publishes a full `HashMap<WorkerId, ModelRuntimeConfig>` generation.
pub type WorkerConfigSender = watch::Sender<HashMap<WorkerId, ModelRuntimeConfig>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouteSelection {
    pub worker: WorkerWithDpRank,
    pub cached_tokens: usize,
}

/// Worker type label used for GWP-tier metrics.
const GWP_WORKER_TYPE: &str = "decode";

fn gwp_kv_router_config(ttl_secs: u64) -> KvRouterConfig {
    KvRouterConfig {
        use_kv_events: false,
        skip_initial_worker_wait: false,
        router_snapshot_threshold: None,
        router_queue_threshold: None,
        router_replica_sync: true,
        router_temperature: 0.0,
        host_cache_hit_weight: 1.0,
        disk_cache_hit_weight: 1.0,
        router_ttl_secs: ttl_secs as f64,
        ..Default::default()
    }
}

/// GWP keeps etcd as its discovery plane while allowing replica events to use
/// either direct ZMQ (the backwards-compatible default) or NATS Core pub-sub.
/// `DYN_EVENT_PLANE=nats` selects NATS and `NATS_SERVER` configures its server.
fn gwp_distributed_config() -> anyhow::Result<DistributedConfig> {
    let event_transport_kind = match std::env::var(
        dynamo_runtime::config::environment_names::event_plane::DYN_EVENT_PLANE,
    ) {
        Ok(value) if value == "nats" => EventTransportKind::Nats,
        Ok(value) if value == "zmq" || value.is_empty() => EventTransportKind::Zmq,
        Err(std::env::VarError::NotPresent) => EventTransportKind::Zmq,
        Ok(value) => anyhow::bail!(
            "invalid DYN_EVENT_PLANE value '{value}'; valid values are 'nats' and 'zmq'"
        ),
        Err(std::env::VarError::NotUnicode(_)) => {
            anyhow::bail!("DYN_EVENT_PLANE must contain valid Unicode")
        }
    };

    Ok(gwp_distributed_config_for(event_transport_kind))
}

fn gwp_distributed_config_for(event_transport_kind: EventTransportKind) -> DistributedConfig {
    DistributedConfig {
        discovery_backend: DiscoveryBackend::KvStore(kv::Selector::Etcd(Box::default())),
        nats_config: (event_transport_kind == EventTransportKind::Nats).then(Default::default),
        request_plane: RequestPlaneMode::Tcp,
        event_transport_kind,
    }
}

/// Canonical model identity used to select one isolated scheduler and event
/// channel. Request aliases resolve to this ID before entering the registry.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(transparent)]
pub struct OracleVersionId(String);

impl OracleVersionId {
    pub fn new(value: impl Into<String>) -> anyhow::Result<Self> {
        let value = value.into();
        anyhow::ensure!(
            !value.trim().is_empty(),
            "oracle version ID must not be empty"
        );
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn transport_key(&self) -> String {
        if Slug::try_from(self.as_str()).is_ok() {
            self.0.clone()
        } else {
            Slug::slugify_unique(self.as_str()).to_string()
        }
    }
}

impl std::fmt::Display for OracleVersionId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Process-wide registry of independent schedulers, keyed by canonical model.
/// Aliases are resolved by the topology before this boundary.
pub struct GwpRouterRegistry {
    routers: DashMap<OracleVersionId, Arc<GwpRouter>>,
    /// Removed routers are quiesced and retained for safe reuse. `KvRouter`
    /// cancellation is tied to the shared runtime, so dropping one model
    /// router would also cancel its siblings until the underlying router
    /// supports scoped shutdown.
    dormant: DashMap<OracleVersionId, Arc<GwpRouter>>,
    create_lock: tokio::sync::Mutex<()>,
    drt: DistributedRuntime,
    metrics: GwpMetrics,
    block_size: u32,
    approx_indexer_ttl_secs: u64,
}

impl GwpRouterRegistry {
    pub async fn new(block_size: u32, approx_indexer_ttl_secs: u64) -> anyhow::Result<Arc<Self>> {
        Self::new_with_distributed_config(
            block_size,
            approx_indexer_ttl_secs,
            gwp_distributed_config()?,
        )
        .await
    }

    #[cfg(test)]
    pub(crate) async fn new_process_local(
        block_size: u32,
        approx_indexer_ttl_secs: u64,
    ) -> anyhow::Result<Arc<Self>> {
        Self::new_with_distributed_config(
            block_size,
            approx_indexer_ttl_secs,
            DistributedConfig::process_local(),
        )
        .await
    }

    async fn new_with_distributed_config(
        block_size: u32,
        approx_indexer_ttl_secs: u64,
        distributed_config: DistributedConfig,
    ) -> anyhow::Result<Arc<Self>> {
        let runtime = Runtime::from_current()?;
        let drt = DistributedRuntime::new(runtime, distributed_config).await?;
        let namespace = drt.namespace("gwp")?;
        let metrics_component = namespace.component("router")?;
        register_global_metrics_with_component(&metrics_component);
        let metrics = GwpMetrics::from_endpoint(&metrics_component.endpoint("egress"))?;
        let registry = Arc::new(Self {
            routers: DashMap::new(),
            dormant: DashMap::new(),
            create_lock: tokio::sync::Mutex::new(()),
            drt,
            metrics,
            block_size,
            approx_indexer_ttl_secs,
        });
        Ok(registry)
    }

    async fn ensure_model(
        &self,
        oracle_version_id: &OracleVersionId,
    ) -> anyhow::Result<Arc<GwpRouter>> {
        if let Some(registered) = self.routers.get(oracle_version_id) {
            return Ok(registered.clone());
        }
        let _guard = self.create_lock.lock().await;
        if let Some(registered) = self.routers.get(oracle_version_id) {
            return Ok(registered.clone());
        }
        if let Some((_, router)) = self.dormant.remove(oracle_version_id) {
            tracing::info!(%oracle_version_id, "reactivated dormant oracle-version router");
            self.routers
                .insert(oracle_version_id.clone(), router.clone());
            return Ok(router);
        }
        let router = GwpRouter::new_for_model(
            self.drt.clone(),
            oracle_version_id.clone(),
            self.block_size,
            self.approx_indexer_ttl_secs,
        )
        .await?;
        tracing::info!(
            %oracle_version_id,
            event_component = %oracle_version_id.transport_key(),
            "created oracle-version router"
        );
        self.routers
            .insert(oracle_version_id.clone(), router.clone());
        Ok(router)
    }

    pub fn model_router(&self, oracle_version_id: &OracleVersionId) -> Option<Arc<GwpRouter>> {
        self.routers
            .get(oracle_version_id)
            .map(|router| router.clone())
    }

    /// Create and feed every scheduler in a complete topology generation.
    /// Removal happens only after the topology snapshot becomes visible so a
    /// request can never observe an old route without its router.
    pub async fn reconcile_topology(
        &self,
        topology: &crate::topology::TopologySnapshot,
        refreshed_workers: &HashSet<WorkerId>,
    ) -> anyhow::Result<()> {
        for model in topology.models.keys() {
            self.ensure_model(&OracleVersionId::new(model.clone())?)
                .await?;
        }

        for model in topology.models.keys() {
            let model = OracleVersionId::new(model.clone())?;
            let router = self
                .routers
                .get(&model)
                .expect("model router created during reconciliation")
                .clone();
            let eligible_endpoints = topology
                .models
                .get(model.as_str())
                .map(|binding| &binding.endpoints)
                .expect("topology model exists");
            let workers: HashMap<_, _> = topology
                .workers
                .iter()
                .filter(|(_, worker)| eligible_endpoints.contains(&worker.endpoint))
                .map(|(worker_id, worker)| (*worker_id, worker.runtime.clone()))
                .collect();
            let loads: HashMap<_, _> = topology
                .workers
                .iter()
                .filter(|(_, worker)| eligible_endpoints.contains(&worker.endpoint))
                .filter_map(|(worker_id, worker)| {
                    worker.observed_load.map(|load| (*worker_id, load))
                })
                .collect();
            let refreshed: HashSet<_> = refreshed_workers
                .iter()
                .copied()
                .filter(|worker_id| workers.contains_key(worker_id))
                .collect();
            router.replace_observed_loads(loads, &refreshed).await?;
            router
                .workers_tx
                .send(workers)
                .map_err(|_| anyhow::anyhow!("worker feed closed for oracle version {model}"))?;
            let mut per_endpoint: HashMap<EndpointId, usize> = eligible_endpoints
                .iter()
                .cloned()
                .map(|endpoint| (endpoint, 0))
                .collect();
            for worker in topology
                .workers
                .values()
                .filter(|worker| eligible_endpoints.contains(&worker.endpoint))
            {
                *per_endpoint.entry(worker.endpoint.clone()).or_default() += 1;
            }
            router.metrics.replace_scheduler_live_workers(per_endpoint);
        }
        Ok(())
    }

    /// Stop admitting removed models after their topology disappears. The
    /// router is quiesced and moved to a dormant cache; in-flight
    /// lifecycle calls retain their `Arc` and can still release local state.
    pub fn retire_absent(&self, topology: &crate::topology::TopologySnapshot) {
        let active: HashSet<_> = topology
            .models
            .keys()
            .filter_map(|model| OracleVersionId::new(model.clone()).ok())
            .collect();
        let removed: Vec<_> = self
            .routers
            .iter()
            .filter(|router| !active.contains(router.key()))
            .map(|router| router.key().clone())
            .collect();
        for oracle_version_id in removed {
            if let Some((_, router)) = self.routers.remove(&oracle_version_id) {
                let _ = router.workers_tx.send(HashMap::new());
                router.observed_loads.replace(HashMap::new());
                router
                    .metrics
                    .replace_scheduler_live_workers(HashMap::new());
                tracing::info!(%oracle_version_id, "moved oracle-version router to dormant cache");
                self.dormant.insert(oracle_version_id, router);
            }
        }
    }

    pub fn metrics(&self) -> &GwpMetrics {
        &self.metrics
    }

    #[cfg(feature = "server")]
    pub(crate) fn register_info_route(&self, topology: crate::topology::TopologyStore) {
        let callback: dynamo_runtime::engine_routes::EngineRouteCallback = Arc::new(move |_| {
            let topology = topology.clone();
            Box::pin(async move { Ok(serde_json::to_value(topology.load().info())?) })
        });
        self.drt.engine_routes().register("info", callback);
    }

    pub fn prometheus_metrics(&self) -> anyhow::Result<String> {
        self.drt.metrics().prometheus_expfmt()
    }

    pub async fn replica_peer_count(&self) -> anyhow::Result<usize> {
        let own_instance_id = self.drt.connection_id();
        let mut peers = HashSet::new();
        let routers: Vec<_> = self.routers.iter().map(|router| router.clone()).collect();
        for router in routers {
            peers.extend(router.replica_peer_ids().await?);
        }
        peers.remove(&own_instance_id);
        Ok(peers.len())
    }

    pub fn shutdown(&self) {
        self.drt.shutdown();
    }

    #[cfg(test)]
    pub(crate) async fn potential_loads(
        &self,
        tokens: &[u32],
    ) -> Vec<dynamo_kv_router::scheduling::PotentialLoad> {
        assert_eq!(self.routers.len(), 1, "test helper requires one model");
        let model = self
            .routers
            .iter()
            .next()
            .expect("model router")
            .key()
            .clone();
        self.potential_loads_for_model(&model, tokens).await
    }

    #[cfg(test)]
    pub(crate) async fn potential_loads_for_model(
        &self,
        oracle_version_id: &OracleVersionId,
        tokens: &[u32],
    ) -> Vec<dynamo_kv_router::scheduling::PotentialLoad> {
        self.model_router(oracle_version_id)
            .expect("model router")
            .potential_loads(tokens)
            .await
    }
}

/// A `KvRouter` plus the pieces GWP needs to keep it alive, exposing only the
/// narrow lifecycle the proxy uses (design doc "Approximate router lifecycle"):
///
/// | proxy event                  | call                        |
/// |------------------------------|-----------------------------|
/// | fall-through decision        | [`GwpRouter::pick`]         |
/// | sticky hit                   | `GwpRouter::add_request`    |
/// | upstream response headers    | [`GwpRouter::mark_prefill_completed`] |
/// | stream end / abort / error   | [`GwpRouter::free`]         |
const ROUTING_DECISIONS_SUBJECT: &str = "gwp_routing_decision_events";

#[derive(Serialize, Deserialize, Debug, Clone)]
struct RoutingDecisionEvent {
    worker: WorkerWithDpRank,
    block_size: u32,
    local_hashes: Vec<LocalBlockHash>,
}

struct DecisionSync {
    publisher: Arc<EventPublisher>,
    ttl: std::time::Duration,
}

/// Assemble the same SchedulingRequest the scheduler scores against, with
/// the two read-only differences: `update_states=false` and no request id.
fn scoring_request(
    isl_tokens: usize,
    cached_by_worker: &HashMap<WorkerWithDpRank, usize>,
    loads: &[dynamo_kv_router::scheduling::PotentialLoad],
    allowed_worker_ids: Option<std::collections::HashSet<WorkerId>>,
    block_size: u32,
) -> dynamo_kv_router::scheduling::SchedulingRequest {
    let effective_overlap_blocks = cached_by_worker
        .iter()
        .map(|(worker, cached)| (*worker, (*cached as f64) / (block_size.max(1) as f64)))
        .collect();
    dynamo_kv_router::scheduling::SchedulingRequest {
        isl_tokens,
        allowed_worker_ids,
        effective_overlap_blocks,
        effective_cached_tokens: cached_by_worker.clone(),
        prefill_tokens: loads
            .iter()
            .map(|load| (worker_of(load), load.potential_prefill_tokens))
            .collect(),
        decode_blocks: loads
            .iter()
            .map(|load| (worker_of(load), load.potential_decode_blocks))
            .collect(),
        active_requests: loads
            .iter()
            .map(|load| (worker_of(load), load.active_requests))
            .collect(),
        update_states: false,
        ..Default::default()
    }
}

fn worker_of(load: &dynamo_kv_router::scheduling::PotentialLoad) -> WorkerWithDpRank {
    WorkerWithDpRank::new(load.worker_id, load.dp_rank)
}

/// Score every candidate and sort by increasing score, then worker id,
/// then dp rank for ties.
fn rank_rows(
    cached_by_worker: HashMap<WorkerWithDpRank, usize>,
    worker_configs: &HashMap<WorkerId, dynamo_llm::local_model::runtime_config::ModelRuntimeConfig>,
    prepared: &crate::scoring::PreparedScoring,
) -> Vec<RankedCandidate> {
    let mut ranked: Vec<_> = cached_by_worker
        .into_iter()
        .map(|(worker, cached_tokens)| {
            let score = prepared.score(worker_configs, worker);
            RankedCandidate {
                worker,
                cached_tokens,
                score: score.logit,
            }
        })
        .collect();
    ranked.sort_by(|a, b| {
        a.score
            .partial_cmp(&b.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            // WorkerWithDpRank orders by (worker_id, dp_rank).
            .then(a.worker.cmp(&b.worker))
    });
    ranked
}

/// One candidate of a read-only bid ranking: the worker, the tokens the
/// indexer believes are already cached there, and the B10 score (a cost;
/// lower is better) comparable only within one ranking.
pub struct RankedCandidate {
    pub worker: WorkerWithDpRank,
    pub cached_tokens: usize,
    pub score: f64,
}

pub struct GwpRouter {
    kv: KvRouter<GwpWorkerSelector>,
    decision_sync: DecisionSync,
    observed_loads: ObservedLoadStore,
    selection_policies: SelectionPolicyStore,
    block_size: u32,
    component_name: String,
    workers_tx: WorkerConfigSender,
    metrics: GwpMetrics,
    /// Keeps the shared-discovery runtime (and its cancellation token, which
    /// the scheduler/indexer background tasks are children of) alive.
    _drt: DistributedRuntime,
}

impl GwpRouter {
    async fn new_for_model(
        drt: DistributedRuntime,
        oracle_version_id: OracleVersionId,
        block_size: u32,
        approx_indexer_ttl_secs: u64,
    ) -> anyhow::Result<Arc<Self>> {
        let namespace = drt.namespace("gwp")?;
        let component_name = oracle_version_id.transport_key();
        let component = namespace.component(&component_name)?;
        let endpoint = component.endpoint("egress");
        let metrics = GwpMetrics::from_endpoint(&endpoint)?;
        let client = endpoint.client().await?;

        let (tx, rx) = watch::channel(HashMap::new());
        let config = gwp_kv_router_config(approx_indexer_ttl_secs);
        let observed_loads = ObservedLoadStore::default();
        let selection_policies = SelectionPolicyStore::default();
        let selector = GwpWorkerSelector::new(observed_loads.clone(), selection_policies.clone());

        let decision_sync = DecisionSync {
            publisher: Arc::new(
                EventPublisher::for_component(&component, ROUTING_DECISIONS_SUBJECT).await?,
            ),
            ttl: std::time::Duration::from_secs(approx_indexer_ttl_secs),
        };
        let subscriber = EventSubscriber::for_component(&component, ROUTING_DECISIONS_SUBJECT)
            .await?
            .typed::<RoutingDecisionEvent>();

        let kv = KvRouter::new(
            endpoint,
            client,
            rx,
            block_size,
            selector,
            Some(config),
            None,
            GWP_WORKER_TYPE,
            None,
            false,
            None,
            None,
        )
        .await?;

        let router = Arc::new(Self {
            kv,
            decision_sync,
            observed_loads,
            selection_policies,
            block_size,
            component_name,
            workers_tx: tx,
            metrics,
            _drt: drt,
        });

        tokio::spawn(Self::consume_peer_decisions(
            Arc::downgrade(&router),
            subscriber,
            router._drt.runtime().child_token(),
        ));

        Ok(router)
    }

    async fn consume_peer_decisions(
        router: Weak<Self>,
        mut subscriber: TypedEventSubscriber<RoutingDecisionEvent>,
        cancel: CancellationToken,
    ) {
        loop {
            let (envelope, event) = tokio::select! {
                _ = cancel.cancelled() => return,
                next = subscriber.next() => match next {
                    None => {
                        if let Some(router) = router.upgrade() {
                            tracing::warn!(component = %router.component_name, "peer routing decision stream ended; sync stopped for this model");
                        }
                        return;
                    }
                    Some(Err(error)) => {
                        tracing::warn!(%error, "failed to receive peer routing decision");
                        continue;
                    }
                    Some(Ok(next)) => next,
                },
            };
            let Some(router) = router.upgrade() else {
                return;
            };
            router.apply_peer_decision(&envelope, event).await;
        }
    }

    /// Drop self-echoes, block-size mismatches, and events older than the TTL;
    /// record the rest like a local confirmation.
    async fn apply_peer_decision(&self, envelope: &EventEnvelope, event: RoutingDecisionEvent) {
        let published =
            std::time::UNIX_EPOCH + std::time::Duration::from_millis(envelope.published_at);
        let stale = std::time::SystemTime::now()
            .duration_since(published)
            .is_ok_and(|age| age > self.decision_sync.ttl);
        if envelope.publisher_id == self._drt.discovery().instance_id()
            || event.block_size != self.block_size
            || stale
        {
            return;
        }
        let hashes = RoutingDecisionHashes::from_local_hashes(event.local_hashes);
        if let Err(error) = self
            .kv
            .record_routing_decision_hashes(hashes, event.worker)
            .await
        {
            tracing::warn!(%error, "failed to apply peer routing decision");
        }
    }

    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    pub fn metrics(&self) -> &GwpMetrics {
        &self.metrics
    }

    async fn replica_peer_ids(&self) -> anyhow::Result<HashSet<u64>> {
        let query = DiscoveryQuery::EventChannels(EventChannelQuery::topic(
            "gwp",
            self.component_name.clone(),
            ACTIVE_SEQUENCES_SUBJECT,
        ));
        let peers = self._drt.discovery().list(query).await?;
        Ok(peers
            .into_iter()
            .filter_map(|instance| match instance {
                DiscoveryInstance::EventChannel { instance_id, .. } => Some(instance_id),
                _ => None,
            })
            .collect())
    }

    /// Replace the provider-observed baseline used by the selector. For workers
    /// refreshed by this update, capture the scheduler's local view at
    /// this exact boundary. Later selections apply the signed local change
    /// after that anchor; publications for other endpoints preserve it.
    pub async fn replace_observed_loads(
        &self,
        loads: HashMap<WorkerId, ObservedWorkerLoad>,
        refreshed_workers: &std::collections::HashSet<WorkerId>,
    ) -> anyhow::Result<()> {
        if refreshed_workers.is_empty() {
            self.observed_loads.replace(loads);
            return Ok(());
        }
        // Discounted, deliberately: these anchors are subtracted from
        // placement-path locals (which apply the discounts) in fuse_load.
        let local = self
            .kv
            .get_potential_loads(&[], None, None, None, true)
            .await?;
        let mut anchors: HashMap<_, _> = refreshed_workers
            .iter()
            .copied()
            .map(|worker_id| (worker_id, LocalLoadAnchor::default()))
            .collect();
        for load in local {
            if let Some(anchor) = anchors.get_mut(&load.worker_id) {
                *anchor = LocalLoadAnchor {
                    prefill_tokens: load.potential_prefill_tokens,
                    decode_blocks: load.potential_decode_blocks,
                    active_requests: load.active_requests,
                };
            }
        }
        self.observed_loads.replace_refreshed(loads, &anchors);
        Ok(())
    }

    /// Read-only candidate ranking for the KVLocator bid: every given
    /// candidate's overlap and load, scored with the same B10 scorer the
    /// scheduler uses, without booking anything or touching scheduler state
    /// (`update_states=false`, no request id, no sequence tracking).
    ///
    /// Reuses the two read-only queries GWP already has —
    /// `kv.get_overlap_scores` for the per-worker cached-token estimate and
    /// `kv.get_potential_loads` for the load view — deliberately costing a
    /// second indexer scan over a fused one, to keep every line of this in
    /// dynamo-gwp. shortcut: if the double scan ever matters, the fix is an
    /// upstream one-scan API proposal, not a fork delta.
    pub async fn rank_candidates(
        &self,
        tokens: &[u32],
        identity: &crate::tokens::RoutingTokenIdentity,
        candidates: &std::collections::HashSet<WorkerWithDpRank>,
        worker_configs: &HashMap<
            WorkerId,
            dynamo_llm::local_model::runtime_config::ModelRuntimeConfig,
        >,
        policy: ModelStagePolicy,
    ) -> anyhow::Result<Vec<RankedCandidate>> {
        let isl_tokens = tokens.len();
        let override_config = RouterConfigOverride {
            overlap_score_credit: (!policy.trie).then_some(0.0),
            ..Default::default()
        };
        let worker_ids: Option<std::collections::HashSet<WorkerId>> =
            Some(candidates.iter().map(|worker| worker.worker_id).collect());

        // The cached-token estimate per candidate, from the same indexer the
        // scheduler scores against. Workers the indexer has never seen rank
        // with zero overlap.
        let mut cached_by_worker: HashMap<WorkerWithDpRank, usize> = HashMap::new();
        if policy.trie && !tokens.is_empty() {
            let overlap = self
                .kv
                .get_overlap_scores(
                    tokens,
                    None,
                    identity.block_mm_infos.as_deref(),
                    identity.cache_salt.as_deref(),
                    false,
                )
                .await?;
            for worker in overlap.workers {
                let cached = ((worker.router_credit_blocks * overlap.block_size as f64).round()
                    as usize)
                    .min(isl_tokens);
                cached_by_worker.insert(
                    WorkerWithDpRank::new(worker.worker_id, worker.dp_rank),
                    cached,
                );
            }
        }
        let candidates: HashMap<WorkerWithDpRank, usize> = candidates
            .iter()
            .map(|worker| (*worker, cached_by_worker.get(worker).copied().unwrap_or(0)))
            .collect();

        let loads = self
            .kv
            .get_potential_loads(
                tokens,
                Some(&override_config),
                identity.block_mm_infos.as_deref(),
                identity.cache_salt.as_deref(),
                true,
            )
            .await?;

        let mut request =
            scoring_request(isl_tokens, &candidates, &loads, worker_ids, self.block_size);
        request.router_config_override = Some(override_config);
        let scorer = crate::scoring::B10Scorer::new(self.observed_loads.clone());
        let prepared = scorer.prepare(&request, policy.load_balancing, self.block_size);
        Ok(rank_rows(candidates, worker_configs, &prepared))
    }

    /// Fall-through decision: score all live workers and provisionally
    /// register the request in the scheduler (`update_states=true` — do NOT
    /// also call `Self::add_request`, that would double-count). The
    /// approximate index is updated only after the endpoint reports the worker
    /// that actually served the request.
    ///
    /// `rid` is the per-request id (NOT the session id — concurrent requests
    /// in one session must not collide in the slot tracker).
    pub async fn pick(
        &self,
        rid: &str,
        tokens: &[u32],
        allowed_worker_ids: Option<std::collections::HashSet<WorkerId>>,
        policy: ModelStagePolicy,
    ) -> anyhow::Result<RouteSelection> {
        self.book(rid, tokens, allowed_worker_ids, policy).await
    }

    // TODO(gwp-provisional-booking): The per-model `trie` stage controls the
    // approximate cache heuristic only. Active-prefix tracking remains part of
    // the generic scheduler. If GWP later needs hashless provisional scheduler
    // slots, add that behind a reviewed Dynamo API: keep scalar request/prefill
    // accounting, install real hashes after HTTP 200, and make the replica
    // transition ordered or atomic. Do not use random hashes; they substantially
    // over-account shared prefixes at high concurrency.

    /// Book a sticky request on its live affine worker while still querying
    /// the indexer for this turn's approximate cached-token count.
    pub async fn book_pinned(
        &self,
        rid: &str,
        tokens: &[u32],
        worker: WorkerWithDpRank,
        policy: ModelStagePolicy,
    ) -> anyhow::Result<RouteSelection> {
        let cached_tokens = if policy.trie {
            (self
                .kv
                .get_overlap_blocks(tokens, None, worker, None)
                .await? as usize)
                .saturating_mul(self.block_size as usize)
                .min(tokens.len())
        } else {
            0
        };
        self.add_request(rid, tokens, cached_tokens, worker).await;
        Ok(RouteSelection {
            worker,
            cached_tokens,
        })
    }

    async fn book(
        &self,
        rid: &str,
        tokens: &[u32],
        allowed_worker_ids: Option<std::collections::HashSet<WorkerId>>,
        policy: ModelStagePolicy,
    ) -> anyhow::Result<RouteSelection> {
        let _policy_guard = self.selection_policies.install(rid, policy.load_balancing);
        let request_override = RouterConfigOverride {
            overlap_score_credit: (!policy.trie).then_some(0.0),
            ..Default::default()
        };
        let result = self
            .kv
            .find_best_match_details(
                Some(rid),
                tokens,
                None,
                Some(&request_override),
                true, // update_states: registers the scheduler slot under `rid`
                false,
                None,
                0.0,
                0,
                false,
                None,
                None,
                None,
                allowed_worker_ids,
                RoutingConstraints::default(),
            )
            .await?;
        match result {
            FindBestMatchOutcome::Routed {
                worker,
                cached_tokens,
                ..
            } => Ok(RouteSelection {
                worker,
                cached_tokens: if policy.trie {
                    cached_tokens.min(tokens.len())
                } else {
                    0
                },
            }),
            FindBestMatchOutcome::Backpressure {
                reason,
                queued_isl_tokens,
                max_queued_isl_tokens,
            } => anyhow::bail!(
                "router backpressure: {reason:?} (queued_isl_tokens={queued_isl_tokens}, max_queued_isl_tokens={max_queued_isl_tokens:?})"
            ),
        }
    }

    async fn add_request(
        &self,
        rid: &str,
        tokens: &[u32],
        cached_tokens: usize,
        worker: WorkerWithDpRank,
    ) {
        self.add_request_with_identity(rid, tokens, &Default::default(), cached_tokens, worker)
            .await;
    }

    async fn add_request_with_identity(
        &self,
        rid: &str,
        tokens: &[u32],
        identity: &crate::tokens::RoutingTokenIdentity,
        cached_tokens: usize,
        worker: WorkerWithDpRank,
    ) {
        self.kv
            .add_request(
                rid.to_string(),
                tokens,
                identity.block_mm_infos.as_deref(),
                cached_tokens.min(tokens.len()),
                None,
                worker,
                identity.cache_salt.clone(),
                None,
            )
            .await;
    }

    /// Replace a provisional booking with the worker reported by the endpoint.
    ///
    /// Replacement uses a distinct request ID. Replica-sync events are
    /// delivered independently, so a delayed `Free` for the provisional ID
    /// can never erase the `AddRequest` for the confirmed worker.
    pub async fn rebook_request(
        &self,
        provisional_rid: &str,
        confirmed_rid: &str,
        tokens: &[u32],
        identity: &crate::tokens::RoutingTokenIdentity,
        cached_tokens: usize,
        worker: WorkerWithDpRank,
    ) {
        self.free(provisional_rid).await;
        self.add_request_with_identity(confirmed_rid, tokens, identity, cached_tokens, worker)
            .await;
    }

    /// Attribute the request prefix to the worker that actually served it.
    pub async fn record_routing_decision(
        &self,
        rid: &str,
        tokens: &[u32],
        worker: WorkerWithDpRank,
    ) {
        self.record_routing_decision_with_identity(rid, tokens, &Default::default(), worker)
            .await;
    }

    pub async fn record_routing_decision_with_identity(
        &self,
        rid: &str,
        tokens: &[u32],
        identity: &crate::tokens::RoutingTokenIdentity,
        worker: WorkerWithDpRank,
    ) {
        let local_hashes =
            compute_block_hash_for_seq(tokens, self.block_size, identity.hash_options());
        if local_hashes.is_empty() {
            return;
        }
        let hashes = RoutingDecisionHashes::from_local_hashes(local_hashes.clone());
        if let Err(error) = self.kv.record_routing_decision_hashes(hashes, worker).await {
            tracing::warn!(rid, %error, "failed to record confirmed routing decision");
            return;
        }
        // Fire-and-forget, same posture as the active-sequence replica events.
        let event = RoutingDecisionEvent {
            worker,
            block_size: self.block_size,
            local_hashes,
        };
        let publisher = Arc::clone(&self.decision_sync.publisher);
        let rid = rid.to_string();
        tokio::spawn(async move {
            if let Err(error) = publisher.publish(&event).await {
                tracing::warn!(rid, %error, "failed to publish routing decision");
            }
        });
    }

    #[cfg(test)]
    pub(crate) async fn cached_blocks(&self, tokens: &[u32], worker: WorkerWithDpRank) -> u32 {
        self.kv
            .get_overlap_blocks(tokens, None, worker, None)
            .await
            .unwrap()
    }

    /// Test view of the scheduler slot count, for pinning that the bid
    /// ranking books nothing.
    #[cfg(all(test, feature = "server"))]
    pub(crate) fn pending_count(&self) -> usize {
        self.kv.pending_count()
    }

    /// Intend + book for the KVLocator lifecycle: provisionally book the
    /// chosen worker under `rid` with the indexer's cached-token estimate —
    /// the sticky booking shape (estimate then add_request), no scoring. The
    /// trie policy matches book_pinned: a model with the trie stage off
    /// books with zero cached tokens, the way the envoy sticky path does.
    pub async fn book_attempt(
        &self,
        rid: &str,
        tokens: &[u32],
        identity: &crate::tokens::RoutingTokenIdentity,
        worker: WorkerWithDpRank,
        trie: bool,
    ) -> usize {
        let cached_tokens = if trie {
            self.cached_tokens(tokens, identity, worker).await
        } else {
            0
        };
        self.add_request_with_identity(rid, tokens, identity, cached_tokens, worker)
            .await;
        cached_tokens
    }

    /// The cached-token estimate for one worker, preserving routing identity.
    pub async fn cached_tokens(
        &self,
        tokens: &[u32],
        identity: &crate::tokens::RoutingTokenIdentity,
        worker: WorkerWithDpRank,
    ) -> usize {
        (self
            .kv
            .get_overlap_blocks(
                tokens,
                identity.block_mm_infos.as_deref(),
                worker,
                identity.cache_salt.as_deref(),
            )
            .await
            .unwrap_or(0) as usize)
            .saturating_mul(self.block_size as usize)
            .min(tokens.len())
    }

    /// Upstream response headers: prefill is done, decode load begins.
    pub async fn mark_prefill_completed(&self, rid: &str) {
        if let Err(error) = self.kv.mark_prefill_completed(rid).await {
            tracing::debug!(rid, %error, "mark_prefill_completed on unknown slot");
        }
    }

    /// Stream end / client abort / upstream error: release the in-flight slot.
    /// Without this the load model monotonically inflates.
    pub async fn free(&self, rid: &str) {
        if let Err(error) = self.kv.free(rid).await {
            tracing::debug!(rid, %error, "free on unknown slot");
        }
    }

    #[cfg(test)]
    pub(crate) async fn potential_loads(
        &self,
        tokens: &[u32],
    ) -> Vec<dynamo_kv_router::scheduling::PotentialLoad> {
        self.kv
            .get_potential_loads(tokens, None, None, None, true)
            .await
            .expect("query potential loads")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelTokenizationConfig;
    use crate::topology::{ModelBinding, ProfileId, ResolvedModelProfile, TopologySnapshot};

    fn topology(models: &[&str]) -> TopologySnapshot {
        let mut snapshot = TopologySnapshot::default();
        for model in models {
            let profile = ProfileId((*model).to_string());
            snapshot.models.insert(
                (*model).to_string(),
                ModelBinding {
                    endpoints: HashSet::new(),
                    profile: profile.clone(),
                },
            );
            snapshot.profiles.insert(
                profile,
                ResolvedModelProfile {
                    stages: ModelStagePolicy::default(),
                    tokenization: ModelTokenizationConfig::Pseudo,
                },
            );
        }
        snapshot
    }

    /// Two replicas on shared file discovery over real ZMQ: B learns A's
    /// confirmation; crafted envelopes hit each drop rule.
    #[tokio::test]
    async fn replica_confirmations_reach_peers_and_stale_events_are_dropped() {
        use std::time::{Duration, SystemTime, UNIX_EPOCH};
        let dir = std::env::temp_dir().join(format!("gwp-sync-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let registry = || async {
            let mut config = DistributedConfig::process_local();
            config.discovery_backend = DiscoveryBackend::KvStore(kv::Selector::File(dir.clone()));
            let registry = GwpRouterRegistry::new_with_distributed_config(4, 30, config)
                .await
                .unwrap();
            let router = registry
                .ensure_model(&OracleVersionId::new("sync-model").unwrap())
                .await
                .unwrap();
            router
                .workers_tx
                .send(HashMap::from([
                    (1, ModelRuntimeConfig::default()),
                    (2, ModelRuntimeConfig::default()),
                ]))
                .unwrap();
            (registry.drt.discovery().instance_id(), router)
        };
        let (ida, ra) = registry().await;
        let (idb, rb) = registry().await;
        let (worker, other) = (
            WorkerWithDpRank::from_worker_id(1),
            WorkerWithDpRank::from_worker_id(2),
        );

        // Republish until receipt: discovery and ZMQ's slow-joiner window are
        // async, and the KVLocator lifecycle tests run their own registries
        // in parallel, so the budget is generous rather than tight.
        let tokens: Vec<u32> = (0..16).collect();
        tokio::time::timeout(Duration::from_secs(30), async {
            while rb.cached_blocks(&tokens, worker).await == 0 {
                ra.record_routing_decision("rid-a", &tokens, worker).await;
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        })
        .await
        .expect("peer learned confirmed prefix");
        assert_eq!(rb.cached_blocks(&tokens, worker).await, 4);
        assert_eq!(rb.cached_blocks(&tokens, other).await, 0);

        // (publisher, age, block_size, expected blocks learned by B for `other`)
        let cases = [
            (ida, Duration::from_secs(31), 4, 0), // stale
            (ida, Duration::ZERO, 8, 0),          // block_size mismatch
            (idb, Duration::ZERO, 4, 0),          // self echo
            (ida, Duration::ZERO, 4, 2),          // fresh
        ];
        for (i, (publisher_id, age, block_size, expected)) in cases.into_iter().enumerate() {
            let tokens: Vec<u32> = (100 * i as u32..100 * i as u32 + 8).collect();
            let envelope = EventEnvelope {
                publisher_id,
                sequence: 0,
                published_at: (SystemTime::now() - age)
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
                topic: ROUTING_DECISIONS_SUBJECT.into(),
                payload: Default::default(),
            };
            let event = RoutingDecisionEvent {
                worker: other,
                block_size,
                local_hashes: compute_block_hash_for_seq(&tokens, 4, BlockHashOptions::default()),
            };
            rb.apply_peer_decision(&envelope, event).await;
            assert_eq!(rb.cached_blocks(&tokens, other).await, expected, "case {i}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    async fn process_local_model_router(
        block_size: u32,
    ) -> (Arc<GwpRouterRegistry>, Arc<GwpRouter>, WorkerConfigSender) {
        let registry = GwpRouterRegistry::new_process_local(block_size, 120)
            .await
            .expect("construct model router registry");
        let model = OracleVersionId::new("model").expect("valid oracle version ID");
        let router = registry.ensure_model(&model).await.expect("model router");
        (registry, router.clone(), router.workers_tx.clone())
    }

    #[test]
    fn production_runtime_uses_etcd_discovery_and_selected_event_plane() {
        let config = gwp_distributed_config_for(EventTransportKind::Zmq);
        match config.discovery_backend {
            DiscoveryBackend::KvStore(kv::Selector::Etcd(options)) => {
                assert!(options.attach_lease);
            }
            other => panic!("expected etcd discovery, got {other:?}"),
        }
        assert_eq!(config.event_transport_kind, EventTransportKind::Zmq);
        assert_eq!(config.request_plane, RequestPlaneMode::Tcp);
        assert!(config.nats_config.is_none());

        let config = gwp_distributed_config_for(EventTransportKind::Nats);
        assert_eq!(config.event_transport_kind, EventTransportKind::Nats);
        assert_eq!(config.request_plane, RequestPlaneMode::Tcp);
        assert!(config.nats_config.is_some());
    }

    #[test]
    fn b10_production_router_enables_replica_sync_and_full_cache_tier_credit() {
        let config = gwp_kv_router_config(120);
        assert!(config.router_replica_sync);
        assert_eq!(config.overlap_score_credit, 1.0);
        assert_eq!(config.host_cache_hit_weight, 1.0);
        assert_eq!(config.disk_cache_hit_weight, 1.0);
    }

    #[test]
    fn model_event_components_are_safe_stable_and_collision_resistant() {
        let transport_key = |value| OracleVersionId::new(value).unwrap().transport_key();
        assert_eq!(transport_key("composer-2-5"), "composer-2-5");
        let slash = transport_key("org/model-v1");
        assert_eq!(slash, transport_key("org/model-v1"));
        assert!(
            slash
                .chars()
                .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_')
        );
        assert_ne!(slash, transport_key("org.model-v1"));
        assert_ne!(transport_key("Model"), transport_key("model"));
    }

    #[tokio::test]
    async fn registry_starts_empty_and_reconciles_active_and_dormant_models() {
        let registry = GwpRouterRegistry::new_process_local(4, 120)
            .await
            .expect("construct empty registry");
        assert!(registry.routers.is_empty());
        assert!(registry.dormant.is_empty());

        let both = topology(&["alpha", "beta"]);
        registry
            .reconcile_topology(&both, &HashSet::new())
            .await
            .expect("create configured routers");
        assert_eq!(registry.routers.len(), 2);

        let beta_id = OracleVersionId::new("beta").unwrap();
        let beta = registry.model_router(&beta_id).expect("beta router");
        let alpha_only = topology(&["alpha"]);
        registry
            .reconcile_topology(&alpha_only, &HashSet::new())
            .await
            .expect("reconcile surviving router");
        registry.retire_absent(&alpha_only);
        assert_eq!(registry.routers.len(), 1);
        assert_eq!(registry.dormant.len(), 1);
        assert!(beta.workers_tx.borrow().is_empty());

        registry
            .reconcile_topology(&both, &HashSet::new())
            .await
            .expect("reactivate removed router");
        assert!(Arc::ptr_eq(
            &beta,
            &registry.model_router(&beta_id).expect("reactivated beta")
        ));
        assert_eq!(registry.routers.len(), 2);
        assert!(registry.dormant.is_empty());
    }

    /// The de-risk test from the design doc: a `KvRouter` constructed with no
    /// etcd and no NATS, fed workers through a bare watch channel, must route,
    /// prefer warm prefixes, and run the full slot lifecycle.
    #[tokio::test]
    async fn kv_router_without_etcd_or_nats() {
        let block_size = 4;
        let (_registry, router, tx) = process_local_model_router(block_size).await;

        // Reflector-style feed: two planner-observed workers appear.
        let mut workers = HashMap::new();
        workers.insert(1u64, ModelRuntimeConfig::default());
        workers.insert(2u64, ModelRuntimeConfig::default());
        tx.send(workers).expect("feed workers");

        // The scheduler ingests the watch asynchronously; give it a beat.
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let tokens: Vec<u32> = (0..32).collect();

        // Fall-through: routes to one of the fed workers.
        let first = router
            .pick("rid-1", &tokens, None, ModelStagePolicy::default())
            .await
            .expect("pick");
        assert!(
            first.worker.worker_id == 1 || first.worker.worker_id == 2,
            "routed to unknown worker {}",
            first.worker.worker_id
        );
        router
            .record_routing_decision("rid-1", &tokens, first.worker)
            .await;
        router.mark_prefill_completed("rid-1").await;
        router.free("rid-1").await;

        // Same prefix, no load anywhere: overlap credit must win — the second
        // request lands on the same worker the indexer recorded.
        let second = router
            .pick("rid-2", &tokens, None, ModelStagePolicy::default())
            .await
            .expect("pick warm");
        assert_eq!(
            first.worker.worker_id, second.worker.worker_id,
            "warm prefix did not stick to the recorded worker"
        );
        router.free("rid-2").await;

        // Sticky-path accounting also estimates this turn's cache hit.
        let sticky = router
            .book_pinned("rid-3", &tokens, first.worker, ModelStagePolicy::default())
            .await
            .expect("book sticky worker");
        assert_eq!(sticky.worker, first.worker);
        assert!(sticky.cached_tokens > 0);
        router.mark_prefill_completed("rid-3").await;
        router.free("rid-3").await;
    }

    #[tokio::test]
    async fn trie_disabled_ignores_warm_overlap_but_keeps_scheduler_load_tracking() {
        let (_registry, router, tx) = process_local_model_router(4).await;
        tx.send(HashMap::from([(1u64, ModelRuntimeConfig::default())]))
            .expect("feed worker");
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let tokens: Vec<u32> = (0..32).collect();
        let worker = WorkerWithDpRank::from_worker_id(1);
        router
            .record_routing_decision("warm", &tokens, worker)
            .await;

        let selection = router
            .pick(
                "trie-disabled",
                &tokens,
                None,
                ModelStagePolicy {
                    affinity: true,
                    trie: false,
                    ..Default::default()
                },
            )
            .await
            .expect("route without trie");
        assert_eq!(selection.cached_tokens, 0);

        let load = router
            .potential_loads(&[])
            .await
            .into_iter()
            .find(|load| load.worker_id == worker.worker_id)
            .expect("worker load");
        assert_eq!(load.active_requests, 1);
        assert!(load.potential_decode_blocks > 0);

        router.free("trie-disabled").await;
    }

    /// Workers fed AFTER construction must become routable (the reflector
    /// starts polling only after the router exists) — guards the
    /// `skip_initial_worker_wait=false` requirement.
    #[tokio::test]
    async fn workers_fed_after_construction_are_routable() {
        let (_registry, router, tx) = process_local_model_router(4).await;

        let mut workers = HashMap::new();
        workers.insert(7u64, ModelRuntimeConfig::default());
        tx.send(workers).expect("feed worker");
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let tokens: Vec<u32> = (100..116).collect();
        let picked = router
            .pick("rid-late", &tokens, None, ModelStagePolicy::default())
            .await
            .expect("pick");
        assert_eq!(picked.worker.worker_id, 7);
        router.free("rid-late").await;
    }

    #[test]
    fn unrelated_observed_publication_preserves_worker_anchor() {
        let store = ObservedLoadStore::default();
        let loads = HashMap::from([
            (1, ObservedWorkerLoad::default()),
            (2, ObservedWorkerLoad::default()),
        ]);
        store.replace_refreshed(
            loads.clone(),
            &HashMap::from([
                (
                    1,
                    LocalLoadAnchor {
                        active_requests: 5,
                        ..Default::default()
                    },
                ),
                (
                    2,
                    LocalLoadAnchor {
                        active_requests: 7,
                        ..Default::default()
                    },
                ),
            ]),
        );

        // Only endpoint/worker 2 received a new planner snapshot. Worker 1's
        // anchor must survive instead of being reset by a global generation.
        store.replace_refreshed(
            loads,
            &HashMap::from([(
                2,
                LocalLoadAnchor {
                    active_requests: 9,
                    ..Default::default()
                },
            )]),
        );
        let snapshot = store.snapshot();
        assert_eq!(snapshot.anchors[&1].active_requests, 5);
        assert_eq!(snapshot.anchors[&2].active_requests, 9);
    }
}
