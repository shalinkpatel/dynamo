// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker pod name in the model deployment card.
//!
//! Workers publish their Kubernetes pod name under [`BASETEN_POD_NAME_RUNTIME_KEY`] in the card's
//! `runtime_data`; the B10 coordinator reads it back through the runtime-config watch and the HTTP
//! frontend returns it in `x-baseten-serving-pod`, which beefeater uses for per-replica metrics.

use super::runtime_config::ModelRuntimeConfig;

/// `runtime_data` key holding the worker's Kubernetes pod name.
pub const BASETEN_POD_NAME_RUNTIME_KEY: &str = "baseten_pod_name";

/// Publishes `POD_NAME`, which the `baseten-dynamo-model` chart sets on worker containers from the
/// downward API. There is no hostname fallback: under hostNetwork the hostname is the node's name.
/// A value the caller already set is kept. No-op when `POD_NAME` is unset or blank.
pub fn publish_pod_name_from_env(config: &mut ModelRuntimeConfig) {
    if config
        .runtime_data
        .contains_key(BASETEN_POD_NAME_RUNTIME_KEY)
    {
        return;
    }
    let Some(pod_name) = std::env::var("POD_NAME")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
    else {
        return;
    };
    if let Err(err) = config.set_engine_specific(BASETEN_POD_NAME_RUNTIME_KEY, &pod_name) {
        tracing::warn!(error = %err, "failed to publish the pod name in the model deployment card");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn published(config: &ModelRuntimeConfig) -> Option<String> {
        config
            .get_engine_specific::<String>(BASETEN_POD_NAME_RUNTIME_KEY)
            .unwrap()
    }

    #[test]
    #[serial_test::serial]
    fn b10_publishes_pod_name_env() {
        temp_env::with_vars([("POD_NAME", Some("bt-llm-abc-worker-5d57-tdrf2"))], || {
            let mut config = ModelRuntimeConfig::default();
            publish_pod_name_from_env(&mut config);
            assert_eq!(
                published(&config).as_deref(),
                Some("bt-llm-abc-worker-5d57-tdrf2")
            );
        });
    }

    #[test]
    #[serial_test::serial]
    fn b10_unset_or_blank_pod_name_publishes_nothing() {
        for value in [None, Some("   ")] {
            temp_env::with_vars([("POD_NAME", value)], || {
                let mut config = ModelRuntimeConfig::default();
                publish_pod_name_from_env(&mut config);
                assert_eq!(published(&config), None);
            });
        }
    }

    #[test]
    #[serial_test::serial]
    fn b10_keeps_caller_supplied_pod_name() {
        temp_env::with_vars([("POD_NAME", Some("from-env"))], || {
            let mut config = ModelRuntimeConfig::default();
            config
                .set_engine_specific(BASETEN_POD_NAME_RUNTIME_KEY, "explicit")
                .unwrap();
            publish_pod_name_from_env(&mut config);
            assert_eq!(published(&config).as_deref(), Some("explicit"));
        });
    }
}
