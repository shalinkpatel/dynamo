// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Topology acquisition adapters. MCM's existing Kafka consumers materialize
//! routing records in Redis; this layer joins them with planner observations.

use crate::config::{EndpointId, GwpConfig};
use crate::metrics::GwpMetrics;
use crate::topology::{TopologyUpdate, TopologyUpdateSender};
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
    pub endpoints: BTreeMap<EndpointId, McmEndpoint>,
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

/// Connect a topology producer to the controller through the chosen discovery
/// adapter. No request handler performs Redis or Kafka reads.
pub async fn spawn(
    config: DiscoveryConfig,
    mut planner: tokio::sync::mpsc::Receiver<TopologyUpdate>,
    output: TopologyUpdateSender,
    metrics: GwpMetrics,
) -> anyhow::Result<tokio::task::JoinHandle<()>> {
    match config {
        DiscoveryConfig::Planner => Ok(tokio::spawn(async move {
            while let Some(update) = planner.recv().await {
                if output.send(update).await.is_err() {
                    break;
                }
            }
        })),
        DiscoveryConfig::McmRedis(config) => {
            #[cfg(feature = "redis")]
            {
                Ok(b10_mcm::McmDiscovery::new(config, metrics)?.spawn(planner, output))
            }
            #[cfg(not(feature = "redis"))]
            {
                let _ = (config, metrics);
                anyhow::bail!("MCM discovery requires the dynamo-gwp redis feature")
            }
        }
    }
}
