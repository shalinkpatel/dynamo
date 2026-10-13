// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Baseten: applies `router_honor_draining` to the scheduler queue. See
//! `scheduling::filter::b10_draining` for the filter itself.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use super::{
    OverlapScoresRefresh, RoutingEligibility, SchedulerQueueActor, SchedulingPolicy,
    SchedulingRequest, SequencePublisher, WorkerConfigLike, WorkerId, WorkerSelector,
};
use crate::scheduling::filter::b10_draining::DrainingOutcome;

/// Monotonic counts of what `router_honor_draining` did to admitted requests; the metrics
/// task turns the growth into Prometheus counters. Probes are not counted.
#[derive(Debug, Default)]
pub struct B10DrainingCounters {
    /// Requests selected with draining workers excluded.
    pub steered_requests: AtomicU64,
    /// Requests that could use a draining worker because nothing else was eligible.
    pub fallback_requests: AtomicU64,
}

impl B10DrainingCounters {
    pub(super) fn record(&self, outcome: DrainingOutcome) {
        match outcome {
            DrainingOutcome::NotApplied => {}
            DrainingOutcome::Excluded => {
                self.steered_requests.fetch_add(1, Ordering::Relaxed);
            }
            DrainingOutcome::FellBack => {
                self.fallback_requests.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    pub fn steered_requests(&self) -> u64 {
        self.steered_requests.load(Ordering::Relaxed)
    }

    pub fn fallback_requests(&self) -> u64 {
        self.fallback_requests.load(Ordering::Relaxed)
    }
}

impl<
    P: SequencePublisher + 'static,
    C: WorkerConfigLike + Send + Sync + 'static,
    S: SchedulingPolicy + 'static,
    Sel: WorkerSelector<C> + Send + 'static,
    RF: OverlapScoresRefresh + Send + Sync + 'static,
> SchedulerQueueActor<P, C, S, Sel, RF>
{
    /// Applies `router_honor_draining`, so the admission busy check and worker selection agree
    /// on which workers count as capacity for new work.
    pub(super) fn honoring_draining<'r>(
        &self,
        eligibility: RoutingEligibility<'r>,
        request: &SchedulingRequest,
        workers: &HashMap<WorkerId, C>,
    ) -> RoutingEligibility<'r> {
        if !self.config.snapshot().routing.router_honor_draining {
            return eligibility;
        }
        eligibility.excluding_draining_workers(workers, request.preferred_worker)
    }

    /// Counts the draining decision behind a selection. Probes (`resp_tx` cleared) are
    /// evaluations, not routed requests, so they are skipped.
    pub(super) fn b10_record_draining_outcome(
        &self,
        request: &SchedulingRequest,
        eligibility: &RoutingEligibility<'_>,
    ) {
        if request.resp_tx.is_some() {
            self.b10_draining_counters
                .record(eligibility.draining_outcome());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;

    use tokio::sync::watch;

    use super::super::tests::{make_queue_with_sender, make_request};
    use super::super::{SchedulerQueue, WorkerId, WorkerWithDpRank};
    use crate::scheduling::filter::b10_draining::b10_drain_taint;
    use crate::test_utils::{NoopSequencePublisher, SimpleWorkerConfig};

    #[tokio::test(flavor = "multi_thread")]
    async fn b10_router_honor_draining_keeps_new_work_off_draining_workers() {
        const PROBES: usize = 64;
        let isl = 512;
        let reader = baseten_configmap::ConfigReader::in_memory(Default::default());
        let (queue, _slots, cfg_tx) = baseten_configmap::with_reader(&reader, || {
            make_queue_with_sender(2, 16, isl, None, None)
        });
        cfg_tx.send_modify(|configs| {
            configs
                .get_mut(&0)
                .unwrap()
                .taints
                .insert(b10_drain_taint());
        });

        let route = |pinned_worker: Option<WorkerWithDpRank>| {
            let queue = Arc::clone(&queue);
            async move {
                let mut workers = HashSet::new();
                for i in 0..PROBES {
                    let (mut request, rx) = make_request(&format!("probe-{i}"), isl);
                    request.update_states = false;
                    request.pinned_worker = pinned_worker;
                    queue.enqueue(request).await;
                    workers.insert(rx.await.unwrap().unwrap().best_worker.worker_id);
                }
                workers
            }
        };

        let counts = || {
            let counters = queue.b10_draining_counters();
            (counters.steered_requests(), counters.fallback_requests())
        };

        // Flag off (default): the draining worker is routable as today.
        assert_eq!(route(None).await, HashSet::from([0, 1]));
        assert_eq!(counts(), (0, 0));

        let mut config = baseten_configmap::UnifiedConfig::default();
        config.routing.router_honor_draining = true;
        reader.replace(config);

        // Flag on: new work avoids the draining worker...
        assert_eq!(route(None).await, HashSet::from([1]));
        assert_eq!(counts(), (PROBES as u64, 0));
        // ...but a pinned or sticky request still reaches it, uncounted.
        assert_eq!(
            route(Some(WorkerWithDpRank::new(0, 0))).await,
            HashSet::from([0])
        );
        assert_eq!(counts(), (PROBES as u64, 0));

        // A probe evaluates placement without being routed, so it is not counted either.
        let (mut probe, _rx) = make_request("probe", isl);
        probe.update_states = false;
        assert_eq!(queue.probe(probe).await.unwrap().worker.worker_id, 1);
        assert_eq!(counts(), (PROBES as u64, 0));

        // With every worker draining, new work still lands somewhere.
        cfg_tx.send_modify(|configs| {
            configs
                .get_mut(&1)
                .unwrap()
                .taints
                .insert(b10_drain_taint());
        });
        assert_eq!(route(None).await, HashSet::from([0, 1]));
        assert_eq!(counts(), (PROBES as u64, PROBES as u64));
    }

    /// Two workers, queueing on any active tokens, worker 0 draining, and
    /// `router_honor_draining` set to `honor_draining`.
    #[allow(clippy::type_complexity)]
    fn b10_make_draining_admission_queue(
        honor_draining: bool,
        isl: usize,
    ) -> (
        Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig>>,
        watch::Sender<HashMap<u64, SimpleWorkerConfig>>,
        baseten_configmap::ConfigReader,
    ) {
        let mut config = baseten_configmap::UnifiedConfig::default();
        config.routing.router_honor_draining = honor_draining;
        let reader = baseten_configmap::ConfigReader::in_memory(config);
        let (queue, _slots, cfg_tx) = baseten_configmap::with_reader(&reader, || {
            make_queue_with_sender(2, 16, isl, Some(0.0), None)
        });
        cfg_tx.send_modify(|configs| {
            configs
                .get_mut(&0)
                .unwrap()
                .taints
                .insert(b10_drain_taint());
        });
        (queue, cfg_tx, reader)
    }

    async fn admit(
        queue: &SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig>,
        request_id: &str,
        isl: usize,
    ) -> WorkerId {
        let (request, rx) = make_request(request_id, isl);
        queue.enqueue(request).await;
        rx.await
            .expect("response sender dropped")
            .expect("request should be admitted")
            .best_worker
            .worker_id
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn b10_router_honor_draining_idle_draining_worker_does_not_admit_past_queue() {
        let isl = 512;
        let (queue, _cfg_tx, reader) = b10_make_draining_admission_queue(true, isl);

        assert_eq!(admit(&queue, "req-1", isl).await, 1);

        // The only non-draining worker is busy, so new work queues even though
        // draining worker 0 is idle.
        let (req2, rx2) = make_request("req-2", isl);
        queue.enqueue(req2).await;
        assert_eq!(queue.pending_count(), 1);

        // The session-affinity worker stays eligible, so it still counts as capacity.
        let (mut sticky, sticky_rx) = make_request("req-sticky", isl);
        sticky.preferred_worker = Some(WorkerWithDpRank::new(0, 0));
        sticky.update_states = false;
        queue.enqueue(sticky).await;
        sticky_rx
            .await
            .expect("response sender dropped")
            .expect("session-affinity request should be admitted");
        assert_eq!(queue.pending_count(), 1);

        // Turning the flag off hot-reloads today's admission: the queued request
        // drains onto the draining worker.
        reader.replace(baseten_configmap::UnifiedConfig::default());
        queue.update().await;
        rx2.await
            .expect("response sender dropped")
            .expect("queued request should be admitted once the flag is off");
        assert_eq!(queue.pending_count(), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn b10_router_honor_draining_off_counts_draining_workers_as_capacity() {
        let isl = 512;
        let (queue, _cfg_tx, _reader) = b10_make_draining_admission_queue(false, isl);

        let first = admit(&queue, "req-1", isl).await;
        let second = admit(&queue, "req-2", isl).await;
        assert_eq!(HashSet::from([first, second]), HashSet::from([0, 1]));
        assert_eq!(queue.pending_count(), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn b10_router_honor_draining_only_draining_workers_admit_as_today() {
        let isl = 512;
        let (queue, cfg_tx, _reader) = b10_make_draining_admission_queue(true, isl);
        cfg_tx.send_modify(|configs| {
            configs
                .get_mut(&1)
                .unwrap()
                .taints
                .insert(b10_drain_taint());
        });

        // Selection falls back to draining workers when nothing else is
        // eligible, so admission counts them: both are used before anything queues.
        let first = admit(&queue, "req-1", isl).await;
        let second = admit(&queue, "req-2", isl).await;
        assert_eq!(HashSet::from([first, second]), HashSet::from([0, 1]));

        let (req3, _rx3) = make_request("req-3", isl);
        queue.enqueue(req3).await;
        assert_eq!(queue.pending_count(), 1);
    }
}
