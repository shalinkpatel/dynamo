// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Coalescing publisher for router [`ActiveLoad`] snapshots.
//!
//! The router publishes a worker's load on every request lifecycle change. Spawning a task per
//! snapshot that awaits the NATS publish piles up one task per change while NATS is unreachable.
//! [`ActiveLoad`] is a whole-worker snapshot, so only the newest per worker rank matters: keep the
//! latest one per `(worker_id, dp_rank)` and let a single task publish them. Memory stays bounded by
//! the number of worker ranks however long NATS is down. Upstream removed per-snapshot tasks in
//! ai-dynamo/dynamo#13861 (router load-state refactor); drop this module with that refactor.

use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;

use dynamo_kv_router::protocols::ActiveLoad;
use dynamo_runtime::transports::event_plane::EventPublisher;
use parking_lot::Mutex;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

type LoadKey = (u64, u32);

pub(crate) trait B10LoadSink: Send + Sync + 'static {
    fn publish_load(&self, load: &ActiveLoad) -> impl Future<Output = anyhow::Result<()>> + Send;
}

impl B10LoadSink for EventPublisher {
    async fn publish_load(&self, load: &ActiveLoad) -> anyhow::Result<()> {
        self.publish(load).await
    }
}

/// Latest unpublished [`ActiveLoad`] per worker rank, drained by one publisher task.
pub(crate) struct B10LoadPublisher {
    pending: Arc<Mutex<HashMap<LoadKey, ActiveLoad>>>,
    ready: Arc<Notify>,
}

impl B10LoadPublisher {
    /// Spawn the publisher task; it stops when `cancellation_token` is cancelled.
    pub(crate) fn spawn<S: B10LoadSink>(sink: S, cancellation_token: CancellationToken) -> Self {
        let publisher = Self {
            pending: Arc::default(),
            ready: Arc::new(Notify::new()),
        };
        tokio::spawn(run_load_publisher(
            sink,
            Arc::clone(&publisher.pending),
            Arc::clone(&publisher.ready),
            cancellation_token,
        ));
        publisher
    }

    /// Record `load` as the newest snapshot for its worker rank, replacing an unpublished one.
    pub(crate) fn publish(&self, load: ActiveLoad) {
        self.pending
            .lock()
            .insert((load.worker_id, load.dp_rank), load);
        self.ready.notify_one();
    }
}

async fn run_load_publisher<S: B10LoadSink>(
    sink: S,
    pending: Arc<Mutex<HashMap<LoadKey, ActiveLoad>>>,
    ready: Arc<Notify>,
    cancellation_token: CancellationToken,
) {
    loop {
        tokio::select! {
            _ = cancellation_token.cancelled() => break,
            _ = ready.notified() => {}
        }
        let loads = std::mem::take(&mut *pending.lock());
        for load in loads.into_values() {
            // Snapshots are advisory, so cancellation drops an in-flight publish rather than
            // delaying shutdown on transport backpressure.
            let result = tokio::select! {
                _ = cancellation_token.cancelled() => return,
                result = sink.publish_load(&load) => result,
            };
            if let Err(e) = result {
                tracing::trace!(
                    "Failed to publish ActiveLoad to NATS for worker (id={}, dp_rank={}): {e:?}",
                    load.worker_id,
                    load.dp_rank
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::sync::mpsc;

    fn load(worker_id: u64, dp_rank: u32, active_decode_blocks: u64) -> ActiveLoad {
        ActiveLoad {
            worker_id,
            dp_rank,
            active_decode_blocks: Some(active_decode_blocks),
            ..Default::default()
        }
    }

    /// Records published loads; blocks each publish until released while `gate` is set.
    struct GatedSink {
        published_tx: mpsc::UnboundedSender<ActiveLoad>,
        release: Arc<Notify>,
        gate: Arc<std::sync::atomic::AtomicBool>,
    }

    impl B10LoadSink for GatedSink {
        async fn publish_load(&self, load: &ActiveLoad) -> anyhow::Result<()> {
            if self.gate.load(std::sync::atomic::Ordering::SeqCst) {
                self.release.notified().await;
            }
            self.published_tx.send(load.clone()).unwrap();
            Ok(())
        }
    }

    fn gated_sink(
        gate: bool,
    ) -> (
        GatedSink,
        mpsc::UnboundedReceiver<ActiveLoad>,
        Arc<Notify>,
        Arc<std::sync::atomic::AtomicBool>,
    ) {
        let (published_tx, published_rx) = mpsc::unbounded_channel();
        let release = Arc::new(Notify::new());
        let gate = Arc::new(std::sync::atomic::AtomicBool::new(gate));
        let sink = GatedSink {
            published_tx,
            release: Arc::clone(&release),
            gate: Arc::clone(&gate),
        };
        (sink, published_rx, release, gate)
    }

    async fn next(rx: &mut mpsc::UnboundedReceiver<ActiveLoad>) -> ActiveLoad {
        tokio::time::timeout(Duration::from_secs(1), rx.recv())
            .await
            .expect("a load should be published")
            .expect("publisher channel open")
    }

    #[tokio::test]
    async fn b10_publishes_latest_snapshot_per_worker_rank_while_sink_is_stalled() {
        let (sink, mut published, release, gate) = gated_sink(true);
        let publisher = B10LoadPublisher::spawn(sink, CancellationToken::new());

        // The first snapshot is taken and its publish stalls, like a publish on a dead connection.
        publisher.publish(load(1, 0, 1));
        tokio::time::sleep(Duration::from_millis(50)).await;
        // While stalled, many snapshots arrive; only the newest per worker rank may be kept.
        for blocks in 2..=1000 {
            publisher.publish(load(1, 0, blocks));
        }
        publisher.publish(load(2, 0, 7));
        assert_eq!(publisher.pending.lock().len(), 2);

        gate.store(false, std::sync::atomic::Ordering::SeqCst);
        release.notify_one();
        assert_eq!(next(&mut published).await, load(1, 0, 1));
        let mut rest = vec![next(&mut published).await, next(&mut published).await];
        rest.sort_by_key(|l| l.worker_id);
        assert_eq!(rest, vec![load(1, 0, 1000), load(2, 0, 7)]);
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(published.try_recv().is_err());
    }

    #[tokio::test]
    async fn b10_keeps_dp_ranks_separate() {
        let (sink, mut published, _release, _gate) = gated_sink(false);
        let publisher = B10LoadPublisher::spawn(sink, CancellationToken::new());
        publisher.publish(load(1, 0, 3));
        publisher.publish(load(1, 1, 4));
        let mut got = vec![next(&mut published).await, next(&mut published).await];
        got.sort_by_key(|l| l.dp_rank);
        assert_eq!(got, vec![load(1, 0, 3), load(1, 1, 4)]);
    }

    #[tokio::test]
    async fn b10_stops_on_cancellation_during_a_stalled_publish() {
        let (sink, _published, _release, _gate) = gated_sink(true);
        let pending = Arc::new(Mutex::new(HashMap::new()));
        let ready = Arc::new(Notify::new());
        let cancellation_token = CancellationToken::new();
        let task = tokio::spawn(run_load_publisher(
            sink,
            Arc::clone(&pending),
            Arc::clone(&ready),
            cancellation_token.clone(),
        ));
        pending.lock().insert((1, 0), load(1, 0, 1));
        ready.notify_one();
        tokio::time::sleep(Duration::from_millis(50)).await;

        cancellation_token.cancel();
        tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .expect("publisher should stop after cancellation")
            .expect("publisher task should not panic");
    }
}
