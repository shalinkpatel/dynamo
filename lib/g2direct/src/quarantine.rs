use std::sync::Mutex;
use std::time::{Duration, Instant};

#[derive(Debug)]
struct G2pbTrackedBlock {
    successful_shards: usize,
    in_flight_writes: usize,
    outstanding_plans: usize,
    quarantine_started: Option<Instant>,
}

#[derive(Debug)]
struct G2pbQuarantineState {
    deadline_elapsed: bool,
    blocks: Vec<G2pbTrackedBlock>,
}

/// What a quarantine tracker reports about the blocks it holds. Implemented by
/// the KVBM Prometheus set and by the HiCache G2 transport's own counters, so
/// the tracker can be shared without exposing one side's metrics to the other.
pub trait QuarantineMetrics {
    fn block_quarantined(&self, bytes: usize);
    fn block_released(&self, bytes: usize, quarantined_for: Duration);
}

/// Tracks physical writes separately from the logical block allocation they share.
///
/// A logical block enters quarantine when the scheduling deadline elapses while
/// any writer (a TP shard here, a NIXL chunk for the HiCache G2 path) can still
/// write to its physical destination. It leaves only after every dispatched
/// write for that block reaches a terminal state.
#[derive(Debug)]
pub struct G2pbQuarantineTracker<M: QuarantineMetrics> {
    required_shards: usize,
    block_bytes: Vec<usize>,
    pub metrics: M,
    state: Mutex<G2pbQuarantineState>,
    retained_transfer_permits: Mutex<Vec<tokio::sync::OwnedSemaphorePermit>>,
    settlement_notify: tokio::sync::Notify,
}

impl<M: QuarantineMetrics> G2pbQuarantineTracker<M> {
    pub fn new(required_shards: usize, block_bytes: Vec<usize>, metrics: M) -> Self {
        let blocks = block_bytes
            .iter()
            .map(|_| G2pbTrackedBlock {
                successful_shards: 0,
                in_flight_writes: 0,
                outstanding_plans: required_shards,
                quarantine_started: None,
            })
            .collect();
        Self {
            required_shards,
            block_bytes,
            metrics,
            state: Mutex::new(G2pbQuarantineState {
                deadline_elapsed: false,
                blocks,
            }),
            retained_transfer_permits: Mutex::new(Vec::new()),
            settlement_notify: tokio::sync::Notify::new(),
        }
    }

    pub fn retain_transfer_permit(&self, permit: tokio::sync::OwnedSemaphorePermit) {
        let mut retained = self
            .retained_transfer_permits
            .lock()
            .expect("G2PB retained transfer permit state poisoned");
        retained.push(permit);
        tracing::error!(
            retained_transfer_operations = retained.len(),
            "retaining G2PB transfer permits with quarantined destinations until process restart"
        );
    }

    pub fn mark_dispatched(&self, block_indices: &[usize]) {
        let now = Instant::now();
        let mut state = self.state.lock().expect("G2PB quarantine state poisoned");
        let deadline_elapsed = state.deadline_elapsed;

        for &block_index in block_indices {
            let block = &mut state.blocks[block_index];
            block.in_flight_writes += 1;
            if deadline_elapsed && block.quarantine_started.is_none() {
                block.quarantine_started = Some(now);
                self.metrics
                    .block_quarantined(self.block_bytes[block_index]);
            }
        }
    }

    pub fn mark_terminal(&self, block_indices: &[usize], success: bool) {
        let now = Instant::now();
        let mut state = self.state.lock().expect("G2PB quarantine state poisoned");

        for &block_index in block_indices {
            let block = &mut state.blocks[block_index];
            assert!(
                block.in_flight_writes > 0,
                "G2PB block {block_index} reached terminal state without a dispatched write"
            );
            block.in_flight_writes -= 1;
            if success {
                block.successful_shards += 1;
            }
            assert!(
                block.outstanding_plans > 0,
                "G2PB block {block_index} reached terminal state without an outstanding plan"
            );
            block.outstanding_plans -= 1;
            if block.in_flight_writes == 0
                && let Some(quarantine_started) = block.quarantine_started.take()
            {
                self.metrics.block_released(
                    self.block_bytes[block_index],
                    now.duration_since(quarantine_started),
                );
            }
        }
        drop(state);
        self.settlement_notify.notify_waiters();
    }

    pub fn mark_not_dispatched(&self, block_indices: &[usize]) {
        let mut state = self.state.lock().expect("G2PB quarantine state poisoned");
        for &block_index in block_indices {
            let block = &mut state.blocks[block_index];
            assert!(
                block.outstanding_plans > 0,
                "G2PB block {block_index} completed without an outstanding plan"
            );
            block.outstanding_plans -= 1;
        }
        drop(state);
        self.settlement_notify.notify_waiters();
    }

    pub fn mark_deadline_elapsed(&self) {
        let now = Instant::now();
        let mut state = self.state.lock().expect("G2PB quarantine state poisoned");
        if state.deadline_elapsed {
            return;
        }
        state.deadline_elapsed = true;

        for (block_index, block) in state.blocks.iter_mut().enumerate() {
            if block.in_flight_writes > 0 && block.quarantine_started.is_none() {
                block.quarantine_started = Some(now);
                self.metrics
                    .block_quarantined(self.block_bytes[block_index]);
            }
        }
    }

    pub fn settlement_notified(&self) -> tokio::sync::futures::Notified<'_> {
        self.settlement_notify.notified()
    }

    pub fn completed_prefix_len(&self) -> usize {
        self.state
            .lock()
            .expect("G2PB quarantine state poisoned")
            .blocks
            .iter()
            .take_while(|block| block.successful_shards == self.required_shards)
            .count()
    }

    pub fn has_in_flight_writes(&self) -> bool {
        self.state
            .lock()
            .expect("G2PB quarantine state poisoned")
            .blocks
            .iter()
            .any(|block| block.in_flight_writes > 0)
    }

    pub fn settled_blocks(&self, first_block_index: usize, block_count: usize) -> Vec<bool> {
        self.state
            .lock()
            .expect("G2PB quarantine state poisoned")
            .blocks[first_block_index..first_block_index + block_count]
            .iter()
            .map(|block| block.outstanding_plans == 0)
            .collect()
    }

    pub fn range_has_outstanding_plans(
        &self,
        first_block_index: usize,
        block_count: usize,
    ) -> bool {
        self.state
            .lock()
            .expect("G2PB quarantine state poisoned")
            .blocks[first_block_index..first_block_index + block_count]
            .iter()
            .any(|block| block.outstanding_plans > 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Default)]
    struct Counters {
        blocks: AtomicUsize,
        bytes: AtomicUsize,
        released: AtomicUsize,
    }

    impl QuarantineMetrics for Arc<Counters> {
        fn block_quarantined(&self, bytes: usize) {
            self.blocks.fetch_add(1, Ordering::Relaxed);
            self.bytes.fetch_add(bytes, Ordering::Relaxed);
        }

        fn block_released(&self, bytes: usize, _quarantined_for: Duration) {
            self.blocks.fetch_sub(1, Ordering::Relaxed);
            self.bytes.fetch_sub(bytes, Ordering::Relaxed);
            self.released.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn quarantine_retains_each_destination_until_all_writers_are_terminal() {
        let metrics = Arc::new(Counters::default());
        let tracker = G2pbQuarantineTracker::new(2, vec![100, 200], metrics.clone());
        tracker.mark_dispatched(&[0, 1]);
        tracker.mark_dispatched(&[0]);
        tracker.mark_deadline_elapsed();
        tracker.mark_deadline_elapsed();
        assert_eq!(metrics.blocks.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.bytes.load(Ordering::Relaxed), 300);

        tracker.mark_terminal(&[0, 1], true);
        assert_eq!(metrics.blocks.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.bytes.load(Ordering::Relaxed), 100);
        assert!(tracker.has_in_flight_writes());
        assert_eq!(tracker.settled_blocks(0, 2), vec![false, false]);

        tracker.mark_terminal(&[0], false);
        tracker.mark_not_dispatched(&[1]);
        assert_eq!(metrics.blocks.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.bytes.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.released.load(Ordering::Relaxed), 2);
        assert!(!tracker.has_in_flight_writes());
        assert!(!tracker.range_has_outstanding_plans(0, 2));
        assert_eq!(tracker.settled_blocks(0, 2), vec![true, true]);
        assert_eq!(tracker.completed_prefix_len(), 0);
    }

    #[test]
    fn dispatch_after_deadline_enters_quarantine_but_skipped_plans_do_not() {
        let metrics = Arc::new(Counters::default());
        let tracker = G2pbQuarantineTracker::new(1, vec![128, 256], metrics.clone());
        tracker.mark_deadline_elapsed();
        tracker.mark_not_dispatched(&[0]);
        assert_eq!(metrics.blocks.load(Ordering::Relaxed), 0);
        tracker.mark_dispatched(&[1]);
        assert_eq!(metrics.bytes.load(Ordering::Relaxed), 256);
        tracker.mark_terminal(&[1], true);
        assert_eq!(metrics.bytes.load(Ordering::Relaxed), 0);
        assert_eq!(tracker.completed_prefix_len(), 0);
    }

    #[test]
    fn completed_prefix_requires_all_shards_and_stops_at_the_first_gap() {
        let tracker = G2pbQuarantineTracker::new(2, vec![100; 3], Arc::new(Counters::default()));
        tracker.mark_dispatched(&[0, 1, 2]);
        tracker.mark_terminal(&[0, 1, 2], true);
        tracker.mark_dispatched(&[0, 2]);
        tracker.mark_terminal(&[0, 2], true);
        tracker.mark_not_dispatched(&[1]);
        assert_eq!(tracker.completed_prefix_len(), 1);
        assert!(!tracker.range_has_outstanding_plans(0, 3));
    }

    #[test]
    fn unknown_transfers_retain_permits_until_the_tracker_is_dropped() {
        let limiter = Arc::new(tokio::sync::Semaphore::new(4));
        let tracker = G2pbQuarantineTracker::new(1, vec![256], Arc::new(Counters::default()));
        tracker.retain_transfer_permit(limiter.clone().try_acquire_many_owned(3).unwrap());
        assert_eq!(limiter.available_permits(), 1);
        drop(tracker);
        assert_eq!(limiter.available_permits(), 4);
    }
}
