// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Topology acquisition adapters. MCM's existing Kafka consumers materialize
//! worker inventories in Redis. Each backend publishes complete topology snapshots.

use crate::config::{ConfigStore, EndpointId, GwpConfig};
use crate::metrics::GwpMetrics;
use crate::topology::TopologyUpdateSender;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[cfg(feature = "redis")]
mod b10_mcm;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum DiscoveryConfig {
    #[default]
    Planner,
    McmRedis(McmRedisConfig),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct McmRedisConfig {
    /// Name of the environment variable holding the credential-bearing URL.
    pub url_env: String,
    #[serde(default)]
    pub endpoints: BTreeMap<EndpointId, McmEndpoint>,
    #[serde(default)]
    pub namespace: Option<String>,
    /// Redis sorted set of deployment IDs scored by the unix time they were last
    /// listed (alyx-lb's `alyx:dedicated:tracked`): every deployment listed within a
    /// minute is tracked, one endpoint per plane its LSM samples report. No route
    /// tables, no scan.
    #[serde(default)]
    pub tracked_index: Option<String>,
    #[serde(default = "default_rescan_secs")]
    pub rescan_interval_secs: u64,
    #[serde(default = "default_poll_ms")]
    pub poll_interval_ms: u64,
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
    #[serde(default = "default_sample_age_secs")]
    pub max_sample_age_secs: u64,
    /// Include Alyx's data-path samples, choosing the newer timestamp per pair.
    #[serde(default)]
    pub observed_metrics: bool,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct McmEndpoint {
    pub model_deployment_id: String,
    pub workload_plane: String,
    #[serde(default)]
    pub namespace: Option<String>,
}
fn default_rescan_secs() -> u64 {
    30
}
fn default_poll_ms() -> u64 {
    1000
}
fn default_timeout_ms() -> u64 {
    1000
}
fn default_sample_age_secs() -> u64 {
    10
}

impl DiscoveryConfig {
    pub(crate) fn discovers_endpoints(&self) -> bool {
        matches!(self, Self::McmRedis(mcm) if mcm.namespace.is_some() || mcm.tracked_index.is_some())
    }

    pub(crate) fn validate(&self, config: &GwpConfig) -> anyhow::Result<()> {
        let Self::McmRedis(mcm) = self else {
            return Ok(());
        };
        anyhow::ensure!(
            !mcm.url_env.trim().is_empty(),
            "discovery.url_env is required"
        );
        anyhow::ensure!(
            mcm.poll_interval_ms > 0 && mcm.timeout_ms > 0 && mcm.max_sample_age_secs > 0,
            "discovery intervals must be positive"
        );
        if let Some(index) = &mcm.tracked_index {
            anyhow::ensure!(
                !index.is_empty(),
                "discovery.tracked_index must be nonempty"
            );
            anyhow::ensure!(
                mcm.namespace.is_none()
                    && mcm.endpoints.is_empty()
                    && config.endpoints.is_empty()
                    && config.routes.is_empty(),
                "tracked-index discovery derives endpoints and routes; do not configure them or a namespace"
            );
            return Ok(());
        }
        if let Some(namespace) = &mcm.namespace {
            anyhow::ensure!(
                !namespace.is_empty(),
                "discovery.namespace must be nonempty"
            );
            anyhow::ensure!(
                mcm.rescan_interval_secs > 0,
                "discovery intervals must be positive"
            );
            anyhow::ensure!(
                mcm.endpoints.is_empty() && config.endpoints.is_empty() && config.routes.is_empty(),
                "namespace discovery derives endpoints and routes; do not configure them"
            );
            return Ok(());
        }
        anyhow::ensure!(
            mcm.endpoints.keys().eq(config.endpoints.keys()),
            "MCM discovery must map every configured endpoint exactly once"
        );
        let models_by_endpoint = config.models_by_endpoint();
        let mut identities = std::collections::HashSet::new();
        for (id, endpoint) in &mcm.endpoints {
            anyhow::ensure!(
                identities.insert((&endpoint.model_deployment_id, &endpoint.workload_plane)),
                "each MCM deployment/workload-plane pair must have one endpoint ID"
            );
            anyhow::ensure!(
                !endpoint.workload_plane.is_empty() && !endpoint.model_deployment_id.is_empty(),
                "MCM endpoint identities must be nonempty"
            );
            anyhow::ensure!(
                endpoint.namespace.as_ref().is_none_or(|n| !n.is_empty()),
                "MCM namespace must be nonempty when supplied"
            );
            let models = models_by_endpoint
                .get(id)
                .ok_or_else(|| anyhow::anyhow!("MCM endpoint {} has no model route", id.0))?;
            anyhow::ensure!(
                models.len() == 1 && models.contains(&endpoint.model_deployment_id),
                "MCM endpoint {} must serve its canonical model_deployment_id",
                id.0
            );
        }
        Ok(())
    }
}

/// Start exactly one topology producer. MCM discovery has no planner client.
pub async fn spawn(
    config: ConfigStore,
    output: TopologyUpdateSender,
    metrics: GwpMetrics,
) -> anyhow::Result<tokio::task::JoinHandle<()>> {
    let discovery = config.load().discovery.clone();
    match discovery {
        DiscoveryConfig::Planner => Ok(crate::reflector::Reflector::new(config, output)
            .with_metrics(metrics)
            .spawn()),
        DiscoveryConfig::McmRedis(mcm) => {
            #[cfg(feature = "redis")]
            {
                Ok(b10_mcm::McmDiscovery::new(mcm, metrics)?.spawn(config, output))
            }
            #[cfg(not(feature = "redis"))]
            {
                let _ = (mcm, metrics);
                anyhow::bail!("MCM discovery requires the dynamo-gwp redis feature")
            }
        }
    }
}
