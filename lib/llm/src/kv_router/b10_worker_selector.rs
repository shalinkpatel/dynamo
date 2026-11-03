// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Re-export of `dynamo_kv_router::scheduling::b10_selector`, where `B10WorkerSelector` lives
//! (generic over `WorkerConfigLike`, so offline tools replay it without `dynamo-llm`).

pub use dynamo_kv_router::scheduling::b10_selector::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local_model::runtime_config::ModelRuntimeConfig;
    use dynamo_kv_router::RouterConfigOverride;
    use dynamo_kv_router::protocols::WorkerWithDpRank;
    use dynamo_kv_router::scheduling::SchedulingRequest;
    use dynamo_kv_router::selector::WorkerSelector;
    use std::collections::HashMap;

    fn test_worker_config(dp_start: u32, dp_size: u32) -> ModelRuntimeConfig {
        ModelRuntimeConfig {
            data_parallel_start_rank: dp_start,
            data_parallel_size: dp_size,
            ..Default::default()
        }
    }

    #[test]
    fn b10_virtual_taints_reuse_preferences_overflow_and_hard_constraints() {
        let selector = B10WorkerSelector::new();
        let workers = super::super::b10_virtual_taints::assign(
            (0..64).map(|id| (id, test_worker_config(0, 1))).collect(),
            &std::collections::BTreeMap::from([
                ("b10.virtual/fast".into(), 0.5),
                ("b10.virtual/throughput".into(), 0.5),
            ]),
        );
        let fast = *workers
            .iter()
            .find(|(_, cfg)| cfg.taints.contains("b10.virtual/fast"))
            .unwrap()
            .0;
        let other = *workers
            .iter()
            .find(|(_, cfg)| cfg.taints.contains("b10.virtual/throughput"))
            .unwrap()
            .0;
        let workers: HashMap<_, _> = workers
            .into_iter()
            .filter(|(id, _)| *id == fast || *id == other)
            .collect();
        let mut request = SchedulingRequest {
            maybe_request_id: Some("test".into()),
            isl_tokens: 128,
            track_prefill_tokens: true,
            ..Default::default()
        };
        request.router_config_override = Some(RouterConfigOverride {
            router_temperature: Some(0.0),
            ..Default::default()
        });
        request
            .routing_constraints
            .preferred_taints
            .insert("b10.virtual/fast".into(), 1.0);
        let select = |request: &SchedulingRequest| {
            selector
                .select_worker(&workers, request, request.eligibility(), 64)
                .unwrap()
                .worker
                .worker_id
        };
        assert_eq!(select(&request), fast);
        request
            .prefill_tokens
            .insert(WorkerWithDpRank::new(fast, 0), 1_000_000);
        assert_eq!(select(&request), other);
        request
            .routing_constraints
            .required_taints
            .insert("b10.virtual/fast".into());
        assert_eq!(select(&request), fast);
    }
}
