// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{McmEndpoint, McmRedisConfig};
use crate::config::{ConfigStore, EndpointConfig, EndpointId, GwpConfig, ModelRoute};
use crate::metrics::GwpMetrics;
use crate::topology::{TopologySnapshot, TopologyUpdate, TopologyUpdateSender, TopologyWorker};
use chrono::{DateTime, Utc};
use dynamo_llm::local_model::runtime_config::ModelRuntimeConfig;
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

// Read-only projections of MCM's protobuf JSON, including protobuf's int64
// string representation. The upstream schemas remain owned by MCM.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RouteTable {
    #[serde(alias = "model_version_id")]
    model_version_id: String,
    #[serde(default)]
    namespace: String,
    #[serde(default, alias = "rule_set_id", deserialize_with = "proto_i64")]
    rule_set_id: i64,
    #[serde(default, alias = "weighted_rule_set")]
    weighted_rule_set: Option<WeightedRules>,
    #[serde(default, alias = "fill_first_rule_set")]
    fill_first_rule_set: Option<FillFirstRules>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct WeightedRules {
    #[serde(default, alias = "workload_plane_weights")]
    workload_plane_weights: Vec<WeightedWorkloadPlane>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct WeightedWorkloadPlane {
    #[serde(alias = "workload_plane_name")]
    workload_plane_name: String,
    #[serde(default)]
    weight: i32,
    replicas: Option<i32>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct FillFirstRules {
    #[serde(default, alias = "workload_plane_states")]
    workload_plane_states: Vec<WorkloadPlaneState>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct WorkloadPlaneState {
    #[serde(alias = "workload_plane_name")]
    workload_plane_name: String,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Sample {
    #[serde(alias = "model_deployment_id")]
    model_deployment_id: String,
    #[serde(alias = "workload_plane_name")]
    workload_plane_name: String,
    #[serde(default)]
    namespace: String,
    timestamp: DateTime<Utc>,
    #[serde(alias = "llm_service_metrics_v1")]
    llm_service_metrics_v1: Capacity,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Capacity {
    #[serde(default)]
    capacity: f64,
    #[serde(default)]
    consumption: f64,
    #[serde(default, alias = "worker_ids")]
    worker_ids: Vec<String>,
    #[serde(default, alias = "draining_workers")]
    draining_workers: Vec<DrainingWorker>,
}
#[derive(Clone, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
struct DrainingWorker {
    #[serde(alias = "worker_id")]
    worker_id: String,
    until: DateTime<Utc>,
}

fn worker_id(value: &str) -> anyhow::Result<u64> {
    anyhow::ensure!(
        !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()),
        "worker ID must be a decimal u64"
    );
    Ok(value.parse()?)
}
fn proto_i64<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<i64, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Number {
        String(String),
        Integer(i64),
    }
    match Number::deserialize(deserializer)? {
        Number::String(value) => value.parse().map_err(serde::de::Error::custom),
        Number::Integer(value) => Ok(value),
    }
}

#[derive(Default)]
struct Records {
    routes: Option<String>,
    relayed: HashMap<String, String>,
    observed: HashMap<String, String>,
}
#[derive(Default)]
struct View {
    allowed: HashSet<EndpointId>,
    workers: HashMap<u64, TopologyWorker>,
    revisions: HashMap<String, i64>,
    sample_times: HashMap<EndpointId, DateTime<Utc>>,
    inventory_times: HashMap<EndpointId, DateTime<Utc>>,
    /// Workers built from the last (worker ids, drains) per endpoint. Samples
    /// arrive every second but membership rarely changes.
    inventory: HashMap<EndpointId, Inventory>,
}

struct Inventory {
    worker_ids: Vec<String>,
    draining: Vec<DrainingWorker>,
    workers: HashMap<u64, ModelRuntimeConfig>,
}
impl View {
    fn refresh(
        &mut self,
        config: &McmRedisConfig,
        records: &BTreeMap<String, Records>,
        now: DateTime<Utc>,
    ) {
        self.allowed.clear();
        self.workers.clear();
        let mut ambiguous = HashSet::new();
        for (endpoint_id, endpoint) in &config.endpoints {
            let Some(records) = records.get(&endpoint.model_deployment_id) else {
                continue;
            };
            match self.evaluate(endpoint_id, endpoint, records, config, now) {
                Ok(workers) => {
                    for (id, runtime) in workers {
                        if ambiguous.contains(&id) {
                            continue;
                        }
                        if self.workers.remove(&id).is_some() {
                            ambiguous.insert(id);
                            tracing::error!(
                                worker_id = id,
                                "MCM worker advertised by multiple endpoints; excluding it"
                            );
                            continue;
                        }
                        self.workers.insert(
                            id,
                            TopologyWorker {
                                endpoint: endpoint_id.clone(),
                                runtime,
                                observed_load: None,
                            },
                        );
                    }
                }
                Err(error) => {
                    tracing::warn!(endpoint = %endpoint_id.0, %error, "MCM discovery rejected endpoint observations")
                }
            }
        }
        self.allowed
            .extend(self.workers.values().map(|worker| worker.endpoint.clone()));
        // Forget deployments that are no longer tracked (namespace discovery churns them).
        self.sample_times
            .retain(|id, _| config.endpoints.contains_key(id));
        self.inventory_times
            .retain(|id, _| config.endpoints.contains_key(id));
        self.inventory
            .retain(|id, _| config.endpoints.contains_key(id));
    }
    fn evaluate(
        &mut self,
        id: &EndpointId,
        endpoint: &McmEndpoint,
        records: &Records,
        config: &McmRedisConfig,
        now: DateTime<Utc>,
    ) -> anyhow::Result<HashMap<u64, ModelRuntimeConfig>> {
        let Some(raw) = &records.routes else {
            return Ok(HashMap::new());
        };
        let route: RouteTable = serde_json::from_str(raw)?;
        anyhow::ensure!(
            route.model_version_id == endpoint.model_deployment_id
                && endpoint
                    .namespace
                    .as_ref()
                    .is_none_or(|n| n == &route.namespace),
            "route identity mismatch"
        );
        anyhow::ensure!(
            route.rule_set_id >= 0
                && self
                    .revisions
                    .get(&endpoint.model_deployment_id)
                    .is_none_or(|previous| route.rule_set_id >= *previous),
            "route revision regressed"
        );
        self.revisions
            .insert(endpoint.model_deployment_id.clone(), route.rule_set_id);
        let member = match (&route.weighted_rule_set, &route.fill_first_rule_set) {
            (Some(rules), None) => {
                anyhow::ensure!(rules.workload_plane_weights.iter().all(|w| (0..=100).contains(&w.weight) && w.replicas.is_none_or(|r| r >= 0)), "invalid route weight/replicas");
                rules.workload_plane_weights.iter().any(|w| {
                    w.workload_plane_name == endpoint.workload_plane
                        && w.weight > 0
                        && w.replicas != Some(0)
                })
            }
            // Fill-first replica counts lag a migration (the new plane reads 0, the
            // old one keeps its count); fresh relayed samples decide where workers are.
            (None, Some(rules)) => rules
                .workload_plane_states
                .iter()
                .any(|w| w.workload_plane_name == endpoint.workload_plane),
            _ => anyhow::bail!("route must contain one supported rule set"),
        };
        if !member {
            return Ok(HashMap::new());
        }
        let parse_sample = |raw: &String| match serde_json::from_str::<Sample>(raw) {
            Ok(sample)
                if sample.model_deployment_id == endpoint.model_deployment_id
                    && sample.workload_plane_name == endpoint.workload_plane
                    && endpoint
                        .namespace
                        .as_ref()
                        .is_none_or(|n| n == &sample.namespace)
                    && sample.llm_service_metrics_v1.capacity.is_finite()
                    && sample.llm_service_metrics_v1.consumption.is_finite()
                    && sample.llm_service_metrics_v1.capacity >= 0.0
                    && sample.llm_service_metrics_v1.consumption >= 0.0
                    && sample.timestamp <= now + chrono::Duration::seconds(2) =>
            {
                Some(sample)
            }
            _ => {
                tracing::warn!(endpoint = %id.0, "MCM discovery rejected malformed or mismatched capacity sample");
                None
            }
        };
        let relayed = records
            .relayed
            .get(&endpoint.workload_plane)
            .and_then(parse_sample);
        let observed = records
            .observed
            .get(&endpoint.workload_plane)
            .and_then(parse_sample);
        let sample = [relayed.as_ref(), observed.as_ref()]
            .into_iter()
            .flatten()
            .max_by_key(|s| s.timestamp);
        // Alyx's response-header samples omit workerIds. Only the relayed feed
        // owns inventory, with its own freshness and monotonicity checks: it may
        // advance while still lagging a newer observed capacity/drain sample.
        // Advance both watermarks even when the other feed is unusable.
        let accept_time = |times: &mut HashMap<EndpointId, DateTime<Utc>>, timestamp| {
            if now.signed_duration_since(timestamp).num_milliseconds()
                > i64::try_from(config.max_sample_age_secs.saturating_mul(1000)).unwrap_or(i64::MAX)
                || times.get(id).is_some_and(|previous| timestamp < *previous)
            {
                return false;
            }
            times.insert(id.clone(), timestamp);
            true
        };
        let sample = sample.filter(|s| accept_time(&mut self.sample_times, s.timestamp));
        let inventory = relayed
            .as_ref()
            .filter(|s| accept_time(&mut self.inventory_times, s.timestamp));
        let (Some(sample), Some(inventory)) = (sample, inventory) else {
            return Ok(HashMap::new());
        };
        if sample.llm_service_metrics_v1.capacity == 0.0 {
            return Ok(HashMap::new());
        }
        // Empty/missing IDs cannot prove that a worker is routable. Older LSM
        // samples therefore yield no candidates, without falling back to HTTP.
        let worker_ids = &inventory.llm_service_metrics_v1.worker_ids;
        let draining = &sample.llm_service_metrics_v1.draining_workers;
        if let Some(cached) = self.inventory.get(id)
            && &cached.worker_ids == worker_ids
            && &cached.draining == draining
        {
            return Ok(cached.workers.clone());
        }
        // The default reads the hot-reload config; build it once per rebuild.
        let base = ModelRuntimeConfig::default();
        let mut workers = worker_ids
            .iter()
            .map(|id| Ok((worker_id(id)?, base.clone())))
            .collect::<anyhow::Result<HashMap<_, _>>>()?;
        for draining in draining {
            if let Some(runtime) = workers.get_mut(&worker_id(&draining.worker_id)?) {
                runtime.taints.insert(format!(
                    "{}{}",
                    dynamo_kv_router::scheduling::b10_draining::DRAIN_TAINT_PREFIX,
                    draining.until.to_rfc3339()
                ));
            }
        }
        self.inventory.insert(
            id.clone(),
            Inventory {
                worker_ids: worker_ids.clone(),
                draining: draining.clone(),
                workers: workers.clone(),
            },
        );
        Ok(workers)
    }
    fn topology(&self, config: &GwpConfig) -> anyhow::Result<TopologyUpdate> {
        let mut snapshot = TopologySnapshot::from_config(config, self.workers.clone())?;
        snapshot.endpoints.retain(|id, _| self.allowed.contains(id));
        for binding in snapshot.models.values_mut() {
            binding.endpoints.retain(|id| self.allowed.contains(id));
        }
        Ok(TopologyUpdate {
            snapshot,
            // Inventory updates carry no observed per-worker load baseline.
            refreshed_workers: HashSet::new(),
        })
    }
}

/// The routing config namespace discovery serves: the operator config plus one
/// ingress-less endpoint per discovered (deployment, plane), routed under the
/// deployment ID as its model.
fn with_discovered(config: &GwpConfig, endpoints: &BTreeMap<EndpointId, McmEndpoint>) -> GwpConfig {
    let mut config = config.clone();
    let mut routes: BTreeMap<&str, Vec<EndpointId>> = BTreeMap::new();
    for (id, endpoint) in endpoints {
        config.endpoints.insert(
            id.clone(),
            EndpointConfig {
                ingress_url: None,
                api_key: String::new(),
                planner_url: None,
                planner_api_key: None,
                properties: Default::default(),
            },
        );
        routes
            .entry(&endpoint.model_deployment_id)
            .or_default()
            .push(id.clone());
    }
    config.routes = routes
        .into_iter()
        .map(|(model, endpoints)| ModelRoute {
            models: vec![model.to_string()],
            endpoints,
        })
        .collect();
    let discovered: HashSet<String> = config
        .routes
        .iter()
        .flat_map(|r| r.models.clone())
        .collect();
    config
        .served_alias_model_map
        .retain(|_, canonical| discovered.contains(canonical));
    config
}

fn route_planes(route: &RouteTable) -> Vec<&str> {
    let weighted = route.weighted_rule_set.iter().flat_map(|rules| {
        rules
            .workload_plane_weights
            .iter()
            .map(|w| w.workload_plane_name.as_str())
    });
    let fill_first = route.fill_first_rule_set.iter().flat_map(|rules| {
        rules
            .workload_plane_states
            .iter()
            .map(|w| w.workload_plane_name.as_str())
    });
    weighted.chain(fill_first).collect()
}

/// Endpoints for route tables already matched to the discovery namespace.
fn discovered_endpoints(routes: &[RouteTable]) -> BTreeMap<EndpointId, McmEndpoint> {
    let mut endpoints = BTreeMap::new();
    let mut colliding = HashSet::new();
    for route in routes {
        if crate::router::OracleVersionId::new(route.model_version_id.clone()).is_err() {
            tracing::warn!(model_version_id = %route.model_version_id, "MCM discovery skipped invalid deployment id");
            continue;
        }
        for plane in route_planes(route) {
            let id = EndpointId(format!("{}/{plane}", route.model_version_id));
            if plane.trim().is_empty() {
                continue;
            }
            // Two deployments deriving the same id would swap ownership with SCAN
            // order; skip every claimant.
            if endpoints
                .get(&id)
                .is_some_and(|e: &McmEndpoint| e.model_deployment_id != route.model_version_id)
            {
                colliding.insert(id);
                continue;
            }
            endpoints.insert(
                id,
                // The scan already matched the namespace; LSM samples omit it.
                McmEndpoint {
                    model_deployment_id: route.model_version_id.clone(),
                    workload_plane: plane.to_string(),
                    namespace: None,
                },
            );
        }
    }
    for id in colliding {
        tracing::warn!(endpoint = %id.0, "MCM discovery skipped an endpoint id claimed by several deployments");
        endpoints.remove(&id);
    }
    endpoints
}

const SCAN_TIMEOUT: Duration = Duration::from_secs(10);
/// Bounds the deployments polled each interval, like alyx-lb's dedicated feed.
const MAX_DISCOVERED_ENDPOINTS: usize = 1000;

/// What a published topology was built from. Samples refresh every poll but
/// rarely change membership, and each publish reconciles every model router.
/// The operator config is compared by identity, holding the Arc so a reload can't
/// reuse its address.
struct Published {
    config: Arc<GwpConfig>,
    endpoints: BTreeMap<EndpointId, McmEndpoint>,
    workers: HashMap<u64, TopologyWorker>,
}

pub(super) struct McmDiscovery {
    config: McmRedisConfig,
    last_scan: Option<tokio::time::Instant>,
    /// Route keys of other namespaces; an OV's namespace never changes.
    foreign_routes: HashSet<String>,
    client: redis::Client,
    connection: Option<redis::aio::MultiplexedConnection>,
    metrics: GwpMetrics,
    view: View,
}
impl McmDiscovery {
    pub(super) fn new(config: McmRedisConfig, metrics: GwpMetrics) -> anyhow::Result<Self> {
        let url = std::env::var(&config.url_env).map_err(|_| {
            anyhow::anyhow!(
                "missing Redis discovery URL environment variable {}",
                config.url_env
            )
        })?;
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let client = redis::Client::open(url.as_str())
            .map_err(|_| anyhow::anyhow!("invalid Redis discovery URL in {}", config.url_env))?;
        Ok(Self {
            config,
            client,
            last_scan: None,
            foreign_routes: HashSet::new(),
            connection: None,
            metrics,
            view: View::default(),
        })
    }
    async fn read(&mut self) -> anyhow::Result<BTreeMap<String, Records>> {
        // Redis rejects an empty pipeline; nothing tracked is not a failure.
        if self.config.endpoints.is_empty() {
            return Ok(BTreeMap::new());
        }
        if self.connection.is_none() {
            self.connection = Some(self.client.get_multiplexed_async_connection().await?);
        }
        let versions: std::collections::BTreeSet<_> = self
            .config
            .endpoints
            .values()
            .map(|e| e.model_deployment_id.clone())
            .collect();
        // One bad key (e.g. WRONGTYPE) fails only its own deployment.
        let mut pipeline = redis::pipe();
        pipeline.ignore_errors();
        for version in &versions {
            pipeline.cmd("GET").arg(format!("rt:{version}"));
            pipeline
                .cmd("HGETALL")
                .arg(format!("bis:{version}:v1:metrics"));
            if self.config.observed_metrics {
                pipeline
                    .cmd("HGETALL")
                    .arg(format!("bis:{version}:v1:metrics:observed"));
            }
        }
        let values: Vec<redis::Value> = pipeline
            .query_async(self.connection.as_mut().expect("connected"))
            .await?;
        let mut values = values.into_iter();
        let mut records = BTreeMap::new();
        for version in versions {
            let route = values
                .next()
                .ok_or_else(|| anyhow::anyhow!("missing route pipeline result"))?;
            let relayed = values
                .next()
                .ok_or_else(|| anyhow::anyhow!("missing metrics pipeline result"))?;
            let observed = if self.config.observed_metrics {
                values
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("missing observed metrics pipeline result"))?
            } else {
                redis::Value::Nil
            };
            let parsed = (|| -> redis::RedisResult<Records> {
                Ok(Records {
                    routes: redis::from_redis_value(route)?,
                    relayed: redis::from_redis_value(relayed)?,
                    observed: redis::from_redis_value(observed)?,
                })
            })();
            match parsed {
                Ok(parsed) => {
                    records.insert(version, parsed);
                }
                Err(error) => {
                    tracing::warn!(%version, %error, "MCM discovery skipped unreadable deployment records")
                }
            }
        }
        Ok(records)
    }
    /// Replace `config.endpoints` with every route table in the namespace. A
    /// failed scan keeps the previous set; the poll still withdraws stale routes.
    async fn scan(&mut self) -> anyhow::Result<()> {
        let Some(namespace) = self.config.namespace.clone() else {
            return Ok(());
        };
        if self.connection.is_none() {
            self.connection = Some(self.client.get_multiplexed_async_connection().await?);
        }
        let started = tokio::time::Instant::now();
        let connection = self.connection.as_mut().expect("connected");
        let mut keys = Vec::new();
        let mut cursor = 0u64;
        loop {
            let (next, batch): (u64, Vec<String>) = redis::cmd("SCAN")
                .arg(cursor)
                .arg("MATCH")
                .arg("rt:*")
                .arg("COUNT")
                .arg(1000)
                .query_async(connection)
                .await?;
            keys.extend(batch);
            cursor = next;
            if cursor == 0 {
                break;
            }
        }
        let scanned = keys.len();
        let present: HashSet<&String> = keys.iter().collect();
        self.foreign_routes.retain(|key| present.contains(key));
        drop(present);
        keys.retain(|key| !self.foreign_routes.contains(key));
        let mut routes = Vec::new();
        for chunk in keys.chunks(500) {
            let mut pipeline = redis::pipe();
            pipeline.ignore_errors();
            for key in chunk {
                pipeline.cmd("GET").arg(key);
            }
            let values: Vec<redis::Value> = pipeline.query_async(connection).await?;
            for (key, value) in chunk.iter().zip(values) {
                let Ok(Some(value)) = redis::from_redis_value::<Option<String>>(value) else {
                    continue;
                };
                match serde_json::from_str::<RouteTable>(&value) {
                    Ok(route) if route.namespace != namespace => {
                        self.foreign_routes.insert(key.clone());
                    }
                    Ok(route) => routes.push(route),
                    Err(_) => {}
                }
            }
        }
        // A deployment's route-revision guard lives as long as its route table.
        let present: HashSet<&str> = routes.iter().map(|r| r.model_version_id.as_str()).collect();
        self.view
            .revisions
            .retain(|version, _| present.contains(version.as_str()));
        let mut endpoints = discovered_endpoints(&routes);
        // Only BIS LLM deployments publish LSM samples; others never get workers.
        let versions: Vec<String> = endpoints
            .values()
            .map(|e| e.model_deployment_id.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        let mut pipeline = redis::pipe();
        for version in &versions {
            pipeline
                .cmd("EXISTS")
                .arg(format!("bis:{version}:v1:metrics"));
        }
        let exists: Vec<bool> = if versions.is_empty() {
            Vec::new()
        } else {
            pipeline.query_async(connection).await?
        };
        let live: HashSet<&String> = versions
            .iter()
            .zip(exists)
            .filter_map(|(version, exists)| exists.then_some(version))
            .collect();
        endpoints.retain(|_, endpoint| live.contains(&endpoint.model_deployment_id));
        if endpoints.len() > MAX_DISCOVERED_ENDPOINTS {
            tracing::error!(
                discovered = endpoints.len(),
                "MCM namespace discovery exceeds {MAX_DISCOVERED_ENDPOINTS} endpoints; truncating"
            );
            endpoints = endpoints
                .into_iter()
                .take(MAX_DISCOVERED_ENDPOINTS)
                .collect();
        }
        tracing::info!(
            scanned,
            fetched = keys.len(),
            tracked = live.len(),
            elapsed_ms = started.elapsed().as_millis() as u64,
            "MCM namespace scan"
        );
        self.config.endpoints = endpoints;
        Ok(())
    }
    async fn refresh(&mut self) {
        let due = self
            .last_scan
            .is_none_or(|at| at.elapsed() >= Duration::from_secs(self.config.rescan_interval_secs));
        if self.config.namespace.is_some() && due {
            self.last_scan = Some(tokio::time::Instant::now());
            match tokio::time::timeout(SCAN_TIMEOUT, self.scan()).await {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    self.connection = None;
                    self.last_scan = None;
                    tracing::warn!(%error, "MCM namespace scan failed; keeping previous deployments")
                }
                Err(_) => {
                    self.connection = None;
                    self.last_scan = None;
                    tracing::warn!("MCM namespace scan timed out; keeping previous deployments")
                }
            }
        }
        let result =
            tokio::time::timeout(Duration::from_millis(self.config.timeout_ms), self.read()).await;
        match result {
            Ok(Ok(records)) => {
                self.view.refresh(&self.config, &records, Utc::now());
                self.metrics.record_discovery("ok", self.view.allowed.len());
            }
            result => {
                self.connection = None;
                self.view.allowed.clear();
                self.view.workers.clear();
                self.metrics.record_discovery("error", 0);
                match result {
                    Ok(Err(error)) => {
                        tracing::warn!(%error, "MCM Redis discovery read failed; withdrawing routes")
                    }
                    Err(_) => {
                        tracing::warn!("MCM Redis discovery read timed out; withdrawing routes")
                    }
                    Ok(Ok(_)) => unreachable!(),
                }
            }
        }
    }
    pub(super) fn spawn(
        mut self,
        config: ConfigStore,
        output: TopologyUpdateSender,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut timer =
                tokio::time::interval(Duration::from_millis(self.config.poll_interval_ms));
            timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            // Unchanged state is republished every rescan interval, so an update the
            // controller rejected (e.g. a transient router-creation failure) is retried.
            let republish = Duration::from_secs(self.config.rescan_interval_secs);
            let mut last_published: Option<(Published, tokio::time::Instant)> = None;
            loop {
                tokio::select! {
                    _ = output.closed() => break,
                    _ = timer.tick() => {}
                }
                self.refresh().await;
                let config = config.load();
                if last_published.as_ref().is_some_and(|(last, at)| {
                    at.elapsed() < republish
                        && Arc::ptr_eq(&last.config, &config)
                        && last.endpoints == self.config.endpoints
                        && last.workers == self.view.workers
                }) {
                    continue;
                }
                let published = Published {
                    config: config.clone(),
                    endpoints: self.config.endpoints.clone(),
                    workers: self.view.workers.clone(),
                };
                let update = if self.config.namespace.is_some() {
                    self.view
                        .topology(&with_discovered(&config, &self.config.endpoints))
                } else {
                    self.view.topology(&config)
                };
                let update = match update {
                    Ok(update) => update,
                    Err(error) => {
                        tracing::error!(%error, "invalid MCM topology; retrying next poll");
                        continue;
                    }
                };
                if output.send(update).await.is_err() {
                    break;
                }
                last_published = Some((published, tokio::time::Instant::now()));
            }
        })
    }
}

#[cfg(test)]
mod b10_tests;
