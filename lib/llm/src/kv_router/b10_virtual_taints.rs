// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashMap};
use std::time::Duration;

use baseten_configmap::ConfigReader;
use dynamo_kv_router::protocols::WorkerId;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use xxhash_rust::xxh3::Xxh3;

use crate::discovery::RuntimeConfigWatch;
use crate::local_model::runtime_config::ModelRuntimeConfig;

pub(super) fn assign(
    mut workers: HashMap<WorkerId, ModelRuntimeConfig>,
    fractions: &BTreeMap<String, f64>,
) -> HashMap<WorkerId, ModelRuntimeConfig> {
    if fractions.is_empty() || workers.is_empty() {
        return workers;
    }

    // Weighted rendezvous scores accept relative weights: normalizing them all
    // by the same total cannot change the winner, so no unit-sum check is needed.
    // Each worker independently chooses a pool. Membership
    // changes therefore never reassign surviving workers. Fractions are expected
    // shares, not exact quotas; enforcing quotas would require moving survivors.
    for (&id, config) in &mut workers {
        let identity = config
            .stable_routing_id
            .clone()
            .unwrap_or_else(|| format!("worker-{id}"));
        let (taint, _) = fractions
            .iter()
            .map(|(taint, &fraction)| (taint, b10_pool_score(&identity, taint, fraction)))
            .min_by(|(a, a_score), (b, b_score)| a_score.total_cmp(b_score).then(a.cmp(b)))
            .unwrap();
        config.taints.insert(taint.clone());
    }
    workers
}

// Exponential-race weighted rendezvous: minimize -ln(U) / weight. Comparing its
// logarithm avoids overflow for valid, extremely small fractions. Length-frame
// the identity so different (worker, pool) pairs cannot alias by concatenation.
fn b10_pool_score(identity: &str, taint: &str, fraction: f64) -> f64 {
    let mut hash = Xxh3::new();
    hash.update(&(identity.len() as u64).to_le_bytes());
    hash.update(identity.as_bytes());
    hash.update(taint.as_bytes());
    // Midpoints of 2^52 bins give a uniform strictly inside (0, 1), including
    // the smallest and largest hash values, so both logarithms stay finite.
    let uniform = ((hash.digest() >> 12) as f64 + 0.5) / (1u64 << 52) as f64;
    (-uniform.ln()).ln() - fraction.ln()
}

pub(super) fn watch(
    mut base: RuntimeConfigWatch,
    config: ConfigReader,
    cancel: CancellationToken,
) -> RuntimeConfigWatch {
    let mut fractions = config
        .snapshot()
        .routing
        .virtual_taint_worker_fractions
        .clone();
    let (tx, rx) = watch::channel(assign(base.borrow_and_update().clone(), &fractions));
    tokio::spawn(async move {
        // ConfigReader reloads the file itself; only observe its published snapshot.
        let mut interval = tokio::time::interval(Duration::from_secs(1));
        loop {
            tokio::select! {
                _ = cancel.cancelled() => break,
                _ = tx.closed() => break,
                result = base.changed() => {
                    if result.is_err() { break; }
                }
                _ = interval.tick() => {
                    if config.snapshot().routing.virtual_taint_worker_fractions == fractions {
                        continue;
                    }
                }
            }
            fractions = config
                .snapshot()
                .routing
                .virtual_taint_worker_fractions
                .clone();
            // Always derive from discovery, so disabling or renaming a pool leaves no stale taints.
            let effective = assign(base.borrow_and_update().clone(), &fractions);
            if *tx.borrow() != effective {
                tracing::info!(
                    ?fractions,
                    workers = effective.len(),
                    "Updated virtual worker taints"
                );
                if tx.send(effective).is_err() {
                    break;
                }
            }
        }
    });
    rx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_router::b10_worker_selector::B10WorkerSelector;
    use baseten_configmap::UnifiedConfig;
    use dynamo_kv_router::protocols::{
        ActiveLoad, ActiveSequenceEvent, RoutingConstraints, WorkerWithDpRank,
    };
    use dynamo_kv_router::scheduling::{
        LocalScheduler, TierOverlapBlocks, config::RouterQueueDepthTiers, policy::FcfsPolicy,
    };
    use dynamo_kv_router::sequences::{
        ActiveSequencesMultiWorker, SequencePublisher, WorkerLoadObservation,
    };
    use std::collections::HashSet;
    use std::sync::Arc;

    struct B10TestPublisher;
    impl SequencePublisher for B10TestPublisher {
        async fn publish_event(&self, _: &ActiveSequenceEvent) -> anyhow::Result<()> {
            Ok(())
        }
        fn publish_load(&self, _: ActiveLoad) {}
        fn observe_load(&self, _: &WorkerWithDpRank, _: &str, _: WorkerLoadObservation) {}
    }

    fn b10_workers(count: u64) -> HashMap<WorkerId, ModelRuntimeConfig> {
        (0..count)
            .map(|id| {
                (
                    id,
                    ModelRuntimeConfig {
                        stable_routing_id: Some(format!("pod-{id}")),
                        data_parallel_size: 1,
                        taints: HashSet::from(["dynamo.topology/rack=a".into(), "existing".into()]),
                        ..Default::default()
                    },
                )
            })
            .collect()
    }

    fn b10_fractions(fast: f64, throughput: f64) -> BTreeMap<String, f64> {
        BTreeMap::from([
            ("b10.virtual/fast".into(), fast),
            ("b10.virtual/throughput".into(), throughput),
        ])
    }

    fn b10_pool(config: &ModelRuntimeConfig) -> &str {
        config
            .taints
            .iter()
            .find(|taint| taint.starts_with("b10.virtual/"))
            .unwrap()
    }

    #[test]
    fn b10_assignment_preserves_registered_config_and_assigns_one_pool() {
        for count in 0..=10 {
            let base = b10_workers(count);
            assert_eq!(assign(base.clone(), &BTreeMap::new()), base);
            let assigned = assign(base.clone(), &b10_fractions(0.5, 0.5));
            for (id, mut cfg) in assigned {
                assert_eq!(
                    cfg.taints
                        .iter()
                        .filter(|t| t.starts_with("b10.virtual/"))
                        .count(),
                    1
                );
                cfg.taints.remove("b10.virtual/fast");
                cfg.taints.remove("b10.virtual/throughput");
                assert_eq!(cfg, base[&id]);
            }
        }
    }

    #[test]
    fn b10_weighted_rendezvous_approximates_fractions() {
        let fractions = BTreeMap::from([
            ("b10.virtual/a".into(), 0.2),
            ("b10.virtual/b".into(), 0.3),
            ("b10.virtual/c".into(), 0.5),
        ]);
        let assigned = assign(b10_workers(10_000), &fractions);
        for (pool, fraction) in &fractions {
            let count = assigned
                .values()
                .filter(|cfg| b10_pool(cfg) == pool)
                .count();
            assert!((count as f64 / assigned.len() as f64 - fraction).abs() < 0.02);
        }
        for (fast, throughput) in [(0.5, 0.4999995), (0.5, 0.5000005)] {
            assert_eq!(
                assign(b10_workers(10), &b10_fractions(fast, throughput)),
                assign(b10_workers(10), &b10_fractions(0.5, 0.5))
            );
        }
        assert_eq!(
            assign(b10_workers(64), &b10_fractions(2.0, 8.0)),
            assign(b10_workers(64), &b10_fractions(0.2, 0.8))
        );
        assert!(b10_pool_score("worker", "b10.virtual/tiny", f64::from_bits(1)).is_finite());
    }

    #[test]
    fn b10_assignment_survives_discovery_order_and_ephemeral_ids() {
        let original = assign(b10_workers(8), &b10_fractions(0.5, 0.5));
        let re_registered = b10_workers(8)
            .into_iter()
            .map(|(id, cfg)| (100 - id, cfg))
            .collect();
        let replaced = assign(re_registered, &b10_fractions(0.5, 0.5));
        let pools = |workers: HashMap<WorkerId, ModelRuntimeConfig>| {
            workers
                .into_values()
                .map(|cfg| (cfg.stable_routing_id.clone(), b10_pool(&cfg).to_owned()))
                .collect::<BTreeMap<_, _>>()
        };
        assert_eq!(pools(original), pools(replaced));
        let mut no_identity = b10_workers(5);
        for cfg in no_identity.values_mut() {
            cfg.stable_routing_id = None;
        }
        assert_eq!(
            assign(no_identity.clone(), &b10_fractions(0.5, 0.5)),
            assign(no_identity, &b10_fractions(0.5, 0.5))
        );
    }

    #[test]
    fn b10_membership_changes_never_move_surviving_workers() {
        for fractions in [
            b10_fractions(0.5, 0.5),
            BTreeMap::from([
                ("b10.virtual/a".into(), 0.2),
                ("b10.virtual/b".into(), 0.3),
                ("b10.virtual/c".into(), 0.5),
            ]),
        ] {
            let base = b10_workers(64);
            let original = assign(base.clone(), &fractions);
            for removed in 0..64 {
                let mut smaller = base.clone();
                smaller.remove(&removed);
                let assigned = assign(smaller, &fractions);
                assert!(
                    assigned
                        .iter()
                        .all(|(id, cfg)| b10_pool(cfg) == b10_pool(&original[id]))
                );
            }
            let larger = assign(b10_workers(128), &fractions);
            assert!(
                original
                    .iter()
                    .all(|(id, cfg)| b10_pool(cfg) == b10_pool(&larger[id]))
            );
            // Even routers with different membership views agree on shared workers.
            for id in 0..64 {
                let singleton = assign(HashMap::from([(id, base[&id].clone())]), &fractions);
                assert_eq!(b10_pool(&singleton[&id]), b10_pool(&original[&id]));
            }
        }
    }

    #[test]
    fn b10_pool_changes_move_workers_only_to_or_from_the_changed_pool() {
        let base = b10_workers(1_000);
        let original = assign(base.clone(), &b10_fractions(0.25, 0.75));
        let added = assign(
            base.clone(),
            &BTreeMap::from([
                ("b10.virtual/fast".into(), 0.125),
                ("b10.virtual/throughput".into(), 0.375),
                ("b10.virtual/new".into(), 0.5),
            ]),
        );
        let mut moved = 0;
        for (id, cfg) in &added {
            if b10_pool(cfg) != b10_pool(&original[id]) {
                assert_eq!(b10_pool(cfg), "b10.virtual/new");
                moved += 1;
            }
        }
        assert!(moved > 0);
        // Removing that pool restores every original assignment.
        let removed = assign(base.clone(), &b10_fractions(0.25, 0.75));
        assert_eq!(removed, original);
        let increased = assign(base, &b10_fractions(0.5, 0.5));
        let mut moved = 0;
        for (id, cfg) in &increased {
            if b10_pool(cfg) != b10_pool(&original[id]) {
                assert_eq!(b10_pool(cfg), "b10.virtual/fast");
                moved += 1;
            }
        }
        assert!(moved > 0);
    }

    #[tokio::test(start_paused = true)]
    async fn b10_watch_reloads_fractions_membership_and_disable_without_stale_taints() {
        let base = b10_workers(4);
        let (base_tx, base_rx) = watch::channel(base.clone());
        let config = ConfigReader::in_memory(UnifiedConfig::default());
        let cancel = CancellationToken::new();
        let mut effective = watch(base_rx, config.clone(), cancel.clone());
        assert_eq!(*effective.borrow_and_update(), base);

        let mut updated = UnifiedConfig::default();
        updated.routing.virtual_taint_worker_fractions = b10_fractions(0.5, 0.5);
        config.replace(updated.clone());
        tokio::time::timeout(Duration::from_secs(2), effective.changed())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            *effective.borrow_and_update(),
            assign(base.clone(), &b10_fractions(0.5, 0.5))
        );
        assert_eq!(*base_tx.borrow(), base);

        updated.routing.virtual_taint_worker_fractions =
            BTreeMap::from([("b10.virtual/fast".into(), 1.0)]);
        config.replace(updated.clone());
        tokio::time::timeout(Duration::from_secs(2), effective.changed())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            effective
                .borrow_and_update()
                .values()
                .filter(|cfg| b10_pool(cfg) == "b10.virtual/fast")
                .count(),
            4
        );

        let joined = b10_workers(5);
        base_tx.send(joined.clone()).unwrap();
        tokio::time::timeout(Duration::from_secs(2), effective.changed())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            *effective.borrow_and_update(),
            assign(
                joined.clone(),
                &updated.routing.virtual_taint_worker_fractions
            )
        );

        updated.routing.virtual_taint_worker_fractions =
            BTreeMap::from([("b10.virtual/renamed".into(), 1.0)]);
        config.replace(updated);
        tokio::time::timeout(Duration::from_secs(2), effective.changed())
            .await
            .unwrap()
            .unwrap();
        assert!(
            effective
                .borrow_and_update()
                .values()
                .all(|cfg| b10_pool(cfg) == "b10.virtual/renamed")
        );

        config.replace(UnifiedConfig::default());
        tokio::time::timeout(Duration::from_secs(2), effective.changed())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(*effective.borrow_and_update(), joined);
        cancel.cancel();
        assert!(
            tokio::time::timeout(Duration::from_secs(2), effective.changed())
                .await
                .unwrap()
                .is_err()
        );
    }

    #[tokio::test(start_paused = true)]
    async fn b10_virtual_pool_reload_wakes_a_queued_request() {
        let candidates = b10_workers(64);
        let assigned = assign(candidates.clone(), &b10_fractions(0.5, 0.5));
        let mut workers = HashMap::new();
        for pool in ["b10.virtual/fast", "b10.virtual/throughput"] {
            let (&id, _) = assigned
                .iter()
                .find(|(_, cfg)| b10_pool(cfg) == pool)
                .unwrap();
            workers.insert(id, candidates[&id].clone());
        }
        for cfg in workers.values_mut() {
            cfg.max_num_batched_tokens = Some(64);
        }
        let dp_ranges = workers.keys().map(|&id| (id, (0, 1))).collect();
        let (_base_tx, base_rx) = watch::channel(workers);
        let mut settings = UnifiedConfig::default();
        settings.routing.virtual_taint_worker_fractions = b10_fractions(0.5, 0.5);
        let config = ConfigReader::in_memory(settings.clone());
        let cancel = CancellationToken::new();
        let effective = watch(base_rx, config.clone(), cancel.clone());
        let slots = Arc::new(ActiveSequencesMultiWorker::new(
            B10TestPublisher,
            64,
            dp_ranges,
            false,
            0,
            "test",
        ));
        let scheduler = Arc::new(LocalScheduler::new_without_overlap_refresh(
            slots,
            effective,
            Some(0.5),
            RouterQueueDepthTiers::unbounded_cap(),
            64,
            B10WorkerSelector::new(),
            FcfsPolicy,
            None,
            Duration::from_secs(60),
            true,
            cancel.clone(),
            "test",
            true,
        ));
        let schedule = |scheduler: Arc<LocalScheduler<_, _, _, _>>, id: &str| {
            let id = id.to_owned();
            async move {
                scheduler
                    .schedule(
                        Some(id),
                        64,
                        Some(vec![1, 2, 3, 4]),
                        TierOverlapBlocks::default(),
                        HashMap::new(),
                        HashMap::new(),
                        None,
                        true,
                        None,
                        0.0,
                        0,
                        false,
                        None,
                        None,
                        None,
                        None,
                        RoutingConstraints {
                            required_taints: HashSet::from(["b10.virtual/fast".into()]),
                            ..Default::default()
                        },
                        None,
                    )
                    .await
                    .unwrap()
            }
        };
        let first = schedule(scheduler.clone(), "first").await;
        let pending = tokio::spawn(schedule(scheduler.clone(), "pending"));
        tokio::time::timeout(Duration::from_secs(1), async {
            while scheduler.pending_count() != 1 {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .unwrap();

        settings.routing.virtual_taint_worker_fractions =
            BTreeMap::from([("b10.virtual/fast".into(), 1.0)]);
        config.replace(settings);
        let second = tokio::time::timeout(Duration::from_secs(2), pending)
            .await
            .unwrap()
            .unwrap();
        assert_ne!(first.best_worker.worker_id, second.best_worker.worker_id);
        assert_eq!(scheduler.pending_count(), 0);
        cancel.cancel();
    }
}
