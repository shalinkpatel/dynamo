// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{McmEndpoint, McmRedisConfig};
use crate::config::EndpointId;
use crate::metrics::GwpMetrics;
use crate::topology::{TopologySnapshot, TopologyUpdate, TopologyUpdateSender};
use chrono::{DateTime, Utc};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap, HashSet};
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
    #[serde(default)]
    replicas: i32,
    #[serde(default, alias = "container_concurrency")]
    container_concurrency: i32,
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
struct Capacity {
    #[serde(default)]
    capacity: f64,
    #[serde(default)]
    consumption: f64,
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
    revisions: HashMap<String, i64>,
    sample_times: HashMap<EndpointId, DateTime<Utc>>,
}
impl View {
    fn refresh(
        &mut self,
        config: &McmRedisConfig,
        records: &BTreeMap<String, Records>,
        now: DateTime<Utc>,
    ) {
        self.allowed.clear();
        for (endpoint_id, endpoint) in &config.endpoints {
            let Some(records) = records.get(&endpoint.model_deployment_id) else {
                continue;
            };
            match self.evaluate(endpoint_id, endpoint, records, config, now) {
                Ok(true) => {
                    self.allowed.insert(endpoint_id.clone());
                }
                Ok(false) => {}
                Err(error) => {
                    tracing::warn!(endpoint = %endpoint_id.0, %error, "MCM discovery rejected endpoint observations")
                }
            }
        }
    }
    fn evaluate(
        &mut self,
        id: &EndpointId,
        endpoint: &McmEndpoint,
        records: &Records,
        config: &McmRedisConfig,
        now: DateTime<Utc>,
    ) -> anyhow::Result<bool> {
        let Some(raw) = &records.routes else {
            return Ok(false);
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
            (None, Some(rules)) => rules.workload_plane_states.iter().any(|w| {
                w.workload_plane_name == endpoint.workload_plane
                    && w.replicas > 0
                    && w.container_concurrency > 0
            }),
            _ => anyhow::bail!("route must contain one supported rule set"),
        };
        if !member {
            return Ok(false);
        }
        let sample = [records.relayed.get(&endpoint.workload_plane), records.observed.get(&endpoint.workload_plane)]
            .into_iter().flatten().filter_map(|raw| match serde_json::from_str::<Sample>(raw) {
                Ok(sample) if sample.model_deployment_id == endpoint.model_deployment_id
                    && sample.workload_plane_name == endpoint.workload_plane
                    && endpoint.namespace.as_ref().is_none_or(|n| n == &sample.namespace)
                    && sample.llm_service_metrics_v1.capacity.is_finite()
                    && sample.llm_service_metrics_v1.consumption.is_finite()
                    && sample.llm_service_metrics_v1.capacity >= 0.0
                    && sample.llm_service_metrics_v1.consumption >= 0.0
                    && sample.timestamp <= now + chrono::Duration::seconds(2) => Some(sample),
                _ => { tracing::warn!(endpoint = %id.0, "MCM discovery rejected malformed or mismatched capacity sample"); None }
            }).max_by_key(|s| s.timestamp);
        let Some(sample) = sample else {
            return Ok(false);
        };
        if now
            .signed_duration_since(sample.timestamp)
            .num_milliseconds()
            > i64::try_from(config.max_sample_age_secs.saturating_mul(1000)).unwrap_or(i64::MAX)
        {
            return Ok(false);
        }
        if self
            .sample_times
            .get(id)
            .is_some_and(|previous| sample.timestamp < *previous)
        {
            return Ok(false);
        }
        self.sample_times.insert(id.clone(), sample.timestamp);
        Ok(sample.llm_service_metrics_v1.capacity > 0.0)
    }
    fn filter(&self, snapshot: &TopologySnapshot, refreshed: &HashSet<u64>) -> TopologyUpdate {
        let mut snapshot = snapshot.clone();
        snapshot.endpoints.retain(|id, _| self.allowed.contains(id));
        snapshot
            .workers
            .retain(|_, worker| self.allowed.contains(&worker.endpoint));
        for binding in snapshot.models.values_mut() {
            binding.endpoints.retain(|id| self.allowed.contains(id));
        }
        let refreshed_workers = refreshed
            .iter()
            .filter(|id| snapshot.workers.contains_key(id))
            .copied()
            .collect();
        TopologyUpdate {
            snapshot,
            refreshed_workers,
        }
    }
}

pub(super) struct McmDiscovery {
    config: McmRedisConfig,
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
            connection: None,
            metrics,
            view: View::default(),
        })
    }
    async fn read(&mut self) -> anyhow::Result<BTreeMap<String, Records>> {
        if self.connection.is_none() {
            self.connection = Some(self.client.get_multiplexed_async_connection().await?);
        }
        let versions: std::collections::BTreeSet<_> = self
            .config
            .endpoints
            .values()
            .map(|e| e.model_deployment_id.clone())
            .collect();
        let mut pipeline = redis::pipe();
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
            records.insert(
                version,
                Records {
                    routes: redis::from_redis_value(route)?,
                    relayed: redis::from_redis_value(relayed)?,
                    observed: redis::from_redis_value(observed)?,
                },
            );
        }
        Ok(records)
    }
    async fn refresh(&mut self) {
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
        mut planner: tokio::sync::mpsc::Receiver<TopologyUpdate>,
        output: TopologyUpdateSender,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut latest = None;
            let mut timer =
                tokio::time::interval(Duration::from_millis(self.config.poll_interval_ms));
            timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                let refreshed = tokio::select! {
                    update = planner.recv() => {
                        let Some(update) = update else { break; };
                        latest = Some(update.snapshot);
                        update.refreshed_workers
                    }
                    _ = timer.tick() => { self.refresh().await; HashSet::new() }
                };
                if let Some(snapshot) = &latest
                    && output
                        .send(self.view.filter(snapshot, &refreshed))
                        .await
                        .is_err()
                {
                    break;
                }
            }
        })
    }
}

#[cfg(test)]
mod b10_tests;
