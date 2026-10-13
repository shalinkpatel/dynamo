// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Baseten: router-side observability for worker draining (`router_honor_draining`).
//!
//! Exported next to the other router queue metrics (frontend `/metrics` and the router
//! component registry) and driven by the scheduler's metrics sync task: a gauge of draining
//! workers, counters for requests steered off draining workers and for fallbacks onto them,
//! and an `info` log when a worker enters or leaves draining.

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use dynamo_kv_router::protocols::WorkerId;
use dynamo_kv_router::scheduling::b10_draining::draining_until;
use dynamo_kv_router::scheduling::queue::b10_draining::B10DrainingCounters;
use dynamo_runtime::component::Component;
use dynamo_runtime::metrics::MetricsHierarchy;
use dynamo_runtime::metrics::prometheus_names::{labels, name_prefix};
use prometheus::{IntCounterVec, IntGaugeVec, Opts};

use crate::kv_router::b10_metrics_helper::bis_const_labels;
use crate::local_model::runtime_config::ModelRuntimeConfig;

pub struct B10DrainingMetrics {
    pub draining_workers: IntGaugeVec,
    pub steered_requests_total: IntCounterVec,
    pub fallback_requests_total: IntCounterVec,
}

pub static B10_DRAINING_METRICS: LazyLock<B10DrainingMetrics> =
    LazyLock::new(B10DrainingMetrics::new);

impl B10DrainingMetrics {
    fn new() -> Self {
        Self {
            draining_workers: IntGaugeVec::new(
                Opts::new(
                    format!("{}_router_draining_workers", name_prefix::FRONTEND),
                    "Workers on this endpoint that published a drain deadline (lame ducks)",
                )
                .const_labels(bis_const_labels()),
                &[labels::WORKER_TYPE],
            )
            .expect("Failed to create router_draining_workers gauge"),
            steered_requests_total: IntCounterVec::new(
                Opts::new(
                    format!(
                        "{}_router_draining_steered_requests_total",
                        name_prefix::FRONTEND
                    ),
                    "Requests selected with draining workers excluded (router_honor_draining)",
                )
                .const_labels(bis_const_labels()),
                &[labels::WORKER_TYPE],
            )
            .expect("Failed to create router_draining_steered_requests_total counter"),
            fallback_requests_total: IntCounterVec::new(
                Opts::new(
                    format!(
                        "{}_router_draining_fallback_requests_total",
                        name_prefix::FRONTEND
                    ),
                    "Requests allowed onto draining workers because no other worker was eligible",
                )
                .const_labels(bis_const_labels()),
                &[labels::WORKER_TYPE],
            )
            .expect("Failed to create router_draining_fallback_requests_total counter"),
        }
    }

    fn register(&self, registry: &prometheus::Registry) -> Result<(), prometheus::Error> {
        registry.register(Box::new(self.draining_workers.clone()))?;
        registry.register(Box::new(self.steered_requests_total.clone()))?;
        registry.register(Box::new(self.fallback_requests_total.clone()))?;
        Ok(())
    }
}

/// Registers the draining metrics with the frontend HTTP `/metrics` registry.
pub fn register_draining_metrics(registry: &prometheus::Registry) -> Result<(), prometheus::Error> {
    B10_DRAINING_METRICS.register(registry)
}

/// Registers the draining metrics with a router component, so they also appear on the
/// router's metrics port.
pub fn register_draining_metrics_with_component(component: &Component) {
    let registry = component.get_metrics_registry();
    let m = &*B10_DRAINING_METRICS;
    registry.add_metric_or_warn(
        Box::new(m.draining_workers.clone()),
        "router_draining_workers",
    );
    registry.add_metric_or_warn(
        Box::new(m.steered_requests_total.clone()),
        "router_draining_steered_requests_total",
    );
    registry.add_metric_or_warn(
        Box::new(m.fallback_requests_total.clone()),
        "router_draining_fallback_requests_total",
    );
}

/// Draining workers and their deadlines (RFC 3339) among `workers`.
fn draining_workers(workers: &HashMap<WorkerId, ModelRuntimeConfig>) -> HashMap<WorkerId, String> {
    workers
        .iter()
        .filter_map(|(&worker_id, config)| {
            draining_until(config).map(|until| (worker_id, until.to_string()))
        })
        .collect()
}

/// Workers that entered draining since `last`, and workers that left it (cleared the state
/// or disappeared from discovery).
fn draining_transitions(
    last: &HashMap<WorkerId, String>,
    current: &HashMap<WorkerId, String>,
) -> (Vec<WorkerId>, Vec<WorkerId>) {
    let mut entered: Vec<WorkerId> = current
        .keys()
        .filter(|worker_id| !last.contains_key(worker_id))
        .copied()
        .collect();
    let mut left: Vec<WorkerId> = last
        .keys()
        .filter(|worker_id| !current.contains_key(worker_id))
        .copied()
        .collect();
    entered.sort_unstable();
    left.sort_unstable();
    (entered, left)
}

/// Per-router state between syncs: the counter values already published and the draining
/// set last seen, so counters grow by deltas and transitions are logged once.
#[derive(Default)]
pub struct B10DrainingSync {
    steered_requests: u64,
    fallback_requests: u64,
    draining: HashMap<WorkerId, String>,
}

impl B10DrainingSync {
    pub fn sync(
        &mut self,
        worker_type: &str,
        workers: &HashMap<WorkerId, ModelRuntimeConfig>,
        counters: &B10DrainingCounters,
    ) {
        let m = &*B10_DRAINING_METRICS;
        let draining = draining_workers(workers);
        let (entered, left) = draining_transitions(&self.draining, &draining);
        for worker_id in entered {
            tracing::info!(
                worker_id,
                worker_type,
                until = %draining[&worker_id],
                "Worker entered draining; new sessions are kept off it"
            );
        }
        for worker_id in left {
            tracing::info!(
                worker_id,
                worker_type,
                removed = !workers.contains_key(&worker_id),
                "Worker left draining"
            );
        }
        m.draining_workers
            .with_label_values(&[worker_type])
            .set(draining.len() as i64);
        self.draining = draining;

        for (counter, last, current) in [
            (
                &m.steered_requests_total,
                &mut self.steered_requests,
                counters.steered_requests(),
            ),
            (
                &m.fallback_requests_total,
                &mut self.fallback_requests,
                counters.fallback_requests(),
            ),
        ] {
            let delta = current.saturating_sub(*last);
            *last = current;
            if delta > 0 {
                counter.with_label_values(&[worker_type]).inc_by(delta);
            }
        }
    }

    /// Workers currently seen as draining.
    pub fn draining_worker_ids(&self) -> HashSet<WorkerId> {
        self.draining.keys().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use dynamo_kv_router::scheduling::b10_draining::DRAIN_TAINT_PREFIX;
    use prometheus::{Encoder, TextEncoder};

    use super::*;

    fn draining_config(until: &str) -> ModelRuntimeConfig {
        ModelRuntimeConfig {
            taints: HashSet::from([
                format!("{DRAIN_TAINT_PREFIX}{until}"),
                "service_tier/priority".to_string(),
            ]),
            ..Default::default()
        }
    }

    #[test]
    fn b10_draining_transitions_report_entries_and_exits_once() {
        let mut workers = HashMap::from([
            (1, draining_config("2026-10-01T12:00:00+00:00")),
            (2, ModelRuntimeConfig::default()),
        ]);
        let first = draining_workers(&workers);
        assert_eq!(
            first,
            HashMap::from([(1, "2026-10-01T12:00:00+00:00".to_string())])
        );
        assert_eq!(
            draining_transitions(&HashMap::new(), &first),
            (vec![1], vec![])
        );
        // Unchanged state is not a transition, nor is a deadline change.
        assert_eq!(draining_transitions(&first, &first), (vec![], vec![]));
        workers.insert(1, draining_config("2026-10-01T13:00:00+00:00"));
        assert_eq!(
            draining_transitions(&first, &draining_workers(&workers)),
            (vec![], vec![])
        );

        // Worker 1 clears, worker 2 starts draining, worker 3 appears already draining.
        workers.insert(1, ModelRuntimeConfig::default());
        workers.insert(2, draining_config("2026-10-01T14:00:00+00:00"));
        workers.insert(3, draining_config("2026-10-01T14:00:00+00:00"));
        let second = draining_workers(&workers);
        assert_eq!(draining_transitions(&first, &second), (vec![2, 3], vec![1]));

        // A draining worker that disappears from discovery has left draining.
        workers.remove(&3);
        assert_eq!(
            draining_transitions(&second, &draining_workers(&workers)),
            (vec![], vec![3])
        );
    }

    #[test]
    fn b10_draining_sync_exports_gauge_and_counter_deltas() {
        let registry = prometheus::Registry::new();
        B10_DRAINING_METRICS.register(&registry).unwrap();
        let counters = B10DrainingCounters::default();
        let mut sync = B10DrainingSync::default();
        let workers = HashMap::from([
            (1, draining_config("2026-10-01T12:00:00+00:00")),
            (2, ModelRuntimeConfig::default()),
        ]);

        counters.steered_requests.store(5, Ordering::Relaxed);
        sync.sync("decode", &workers, &counters);
        assert_eq!(sync.draining_worker_ids(), HashSet::from([1]));

        // Only growth since the last sync is published.
        counters.steered_requests.store(7, Ordering::Relaxed);
        counters.fallback_requests.store(3, Ordering::Relaxed);
        sync.sync("decode", &workers, &counters);
        sync.sync("decode", &HashMap::new(), &counters);
        assert!(sync.draining_worker_ids().is_empty());

        let encoder = TextEncoder::new();
        let mut buffer = Vec::new();
        encoder.encode(&registry.gather(), &mut buffer).unwrap();
        let expected = "\
# HELP dynamo_frontend_router_draining_fallback_requests_total Requests allowed onto draining workers because no other worker was eligible
# TYPE dynamo_frontend_router_draining_fallback_requests_total counter
dynamo_frontend_router_draining_fallback_requests_total{worker_type=\"decode\"} 3
# HELP dynamo_frontend_router_draining_steered_requests_total Requests selected with draining workers excluded (router_honor_draining)
# TYPE dynamo_frontend_router_draining_steered_requests_total counter
dynamo_frontend_router_draining_steered_requests_total{worker_type=\"decode\"} 7
# HELP dynamo_frontend_router_draining_workers Workers on this endpoint that published a drain deadline (lame ducks)
# TYPE dynamo_frontend_router_draining_workers gauge
dynamo_frontend_router_draining_workers{worker_type=\"decode\"} 0
";
        assert_eq!(String::from_utf8(buffer).unwrap(), expected);
    }
}
