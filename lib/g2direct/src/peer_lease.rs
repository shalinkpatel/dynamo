// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend-neutral source-page leases for direct serving-worker reads.

use dynamo_tokens::SequenceHash;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PeerFetchLeaseError {
    #[error("fetch lease was released before it was acquired")]
    Released,
    #[error("one or more requested physical cache keys are not resident")]
    NotFound,
    #[error("active serving-G2 fetch lease limit reached")]
    LimitReached,
    #[error("duplicate physical cache key in fetch lease")]
    DuplicateKey,
    #[error("fetch lease was retried with different physical cache keys or expiry")]
    RetryMismatch,
}

struct LeaseState<I, T> {
    active: HashMap<I, ActiveLease<T>>,
    pinned: HashMap<SequenceHash, usize>,
    released: HashSet<I>,
    release_order: VecDeque<I>,
}

struct ActiveLease<T> {
    keys: Vec<SequenceHash>,
    handles: Vec<Arc<T>>,
    expires_at_unix_ms: u64,
}

/// Retains the handles `T` a fetch leased until the destination acknowledges a
/// terminal NIXL write or the lease expires. What retention protects depends on
/// `T`: KVBM's immutable host blocks cannot be evicted or reused while held,
/// but the HiCache transport's `PublishedPage` is only a snapshot of the page's
/// descriptors and published checksum. It does not pin the engine's slot, which
/// HiCache may reuse at any time; the destination's checksum check catches that.
pub struct PeerFetchLeaseRegistry<I, T> {
    state: Mutex<LeaseState<I, T>>,
    max_active: usize,
    max_released_tombstones: usize,
}

impl<I: Copy + Eq + std::hash::Hash, T> PeerFetchLeaseRegistry<I, T> {
    pub fn new(max_active: usize, max_released_tombstones: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(max_active > 0, "active lease limit must be positive");
        anyhow::ensure!(
            max_released_tombstones > 0,
            "released lease tombstone limit must be positive"
        );
        Ok(Self {
            state: Mutex::new(LeaseState {
                active: HashMap::new(),
                pinned: HashMap::new(),
                released: HashSet::new(),
                release_order: VecDeque::new(),
            }),
            max_active,
            max_released_tombstones,
        })
    }

    pub fn fetch(
        &self,
        lease_id: I,
        keys: &[SequenceHash],
        expires_at_unix_ms: u64,
        mut resolve: impl FnMut(SequenceHash) -> Option<Arc<T>>,
    ) -> Result<Vec<Arc<T>>, PeerFetchLeaseError> {
        if keys.iter().copied().collect::<HashSet<_>>().len() != keys.len() {
            return Err(PeerFetchLeaseError::DuplicateKey);
        }
        let mut state = self.state.lock().expect("peer fetch lease lock poisoned");
        if state.released.contains(&lease_id) {
            return Err(PeerFetchLeaseError::Released);
        }
        if let Some(lease) = state.active.get(&lease_id) {
            if lease.keys != keys || lease.expires_at_unix_ms != expires_at_unix_ms {
                return Err(PeerFetchLeaseError::RetryMismatch);
            }
            return Ok(lease.handles.clone());
        }
        if state.active.len() >= self.max_active {
            return Err(PeerFetchLeaseError::LimitReached);
        }
        let handles = keys
            .iter()
            .map(|key| resolve(*key).ok_or(PeerFetchLeaseError::NotFound))
            .collect::<Result<Vec<_>, _>>()?;
        for key in keys {
            *state.pinned.entry(*key).or_default() += 1;
        }
        state.active.insert(
            lease_id,
            ActiveLease {
                keys: keys.to_vec(),
                handles: handles.clone(),
                expires_at_unix_ms,
            },
        );
        Ok(handles)
    }

    fn release_locked(state: &mut LeaseState<I, T>, lease_id: I, max_released_tombstones: usize) {
        if let Some(lease) = state.active.remove(&lease_id) {
            for key in lease.keys {
                if let Some(count) = state.pinned.get_mut(&key) {
                    *count -= 1;
                    if *count == 0 {
                        state.pinned.remove(&key);
                    }
                }
            }
        }
        if state.released.insert(lease_id) {
            state.release_order.push_back(lease_id);
        }
        while state.release_order.len() > max_released_tombstones {
            if let Some(expired) = state.release_order.pop_front() {
                state.released.remove(&expired);
            }
        }
    }

    pub fn release(&self, lease_id: I) {
        let mut state = self.state.lock().expect("peer fetch lease lock poisoned");
        Self::release_locked(&mut state, lease_id, self.max_released_tombstones);
    }

    pub fn expire(&self, now_unix_ms: u64) -> usize {
        let mut state = self.state.lock().expect("peer fetch lease lock poisoned");
        let expired = state
            .active
            .iter()
            .filter_map(|(lease_id, lease)| {
                (lease.expires_at_unix_ms <= now_unix_ms).then_some(*lease_id)
            })
            .collect::<Vec<_>>();
        for lease_id in &expired {
            Self::release_locked(&mut state, *lease_id, self.max_released_tombstones);
        }
        expired.len()
    }

    #[cfg(test)]
    fn active_count(&self) -> usize {
        self.state
            .lock()
            .expect("peer fetch lease lock poisoned")
            .active
            .len()
    }

    pub fn is_pinned(&self, key: SequenceHash) -> bool {
        self.state
            .lock()
            .expect("peer fetch lease lock poisoned")
            .pinned
            .contains_key(&key)
    }

    #[cfg(test)]
    fn released_tombstone_count(&self) -> usize {
        self.state
            .lock()
            .expect("peer fetch lease lock poisoned")
            .released
            .len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lease::G2pbFetchLeaseId;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn fetch_is_idempotent_and_retains_until_release() {
        let registry = PeerFetchLeaseRegistry::new(2, 2).unwrap();
        let lease = G2pbFetchLeaseId::new();
        let page = Arc::new(7usize);
        let first = registry
            .fetch(lease, &[11], 100, |_| Some(Arc::clone(&page)))
            .unwrap();
        let calls = AtomicUsize::new(0);
        let second = registry
            .fetch(lease, &[11], 100, |_| {
                calls.fetch_add(1, Ordering::Relaxed);
                None
            })
            .unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert!(Arc::ptr_eq(&first[0], &second[0]));
        registry.release(lease);
        assert_eq!(registry.active_count(), 0);
        assert_eq!(
            registry.fetch(lease, &[11], 100, |_| Some(Arc::clone(&page))),
            Err(PeerFetchLeaseError::Released)
        );
    }

    #[test]
    fn release_before_fetch_and_limits_fail_closed() {
        let registry = PeerFetchLeaseRegistry::new(1, 1).unwrap();
        let released = G2pbFetchLeaseId::new();
        registry.release(released);
        assert_eq!(
            registry.fetch(released, &[1], 100, |_| Some(Arc::new(1))),
            Err(PeerFetchLeaseError::Released)
        );
        let active = G2pbFetchLeaseId::new();
        registry
            .fetch(active, &[1], 100, |_| Some(Arc::new(1)))
            .unwrap();
        assert_eq!(
            registry.fetch(G2pbFetchLeaseId::new(), &[2], 100, |_| Some(Arc::new(2))),
            Err(PeerFetchLeaseError::LimitReached)
        );
        registry.release(active);
        registry.release(G2pbFetchLeaseId::new());
        assert_eq!(registry.released_tombstone_count(), 1);
    }

    #[test]
    fn expiry_releases_pins_and_tombstones_the_lease() {
        let registry = PeerFetchLeaseRegistry::new(2, 2).unwrap();
        let lease = G2pbFetchLeaseId::new();
        registry
            .fetch(lease, &[11], 100, |_| Some(Arc::new(7)))
            .unwrap();
        assert!(registry.is_pinned(11));
        assert_eq!(registry.expire(99), 0);
        assert_eq!(registry.expire(100), 1);
        assert!(!registry.is_pinned(11));
        assert_eq!(registry.active_count(), 0);
        assert_eq!(
            registry.fetch(lease, &[11], 100, |_| Some(Arc::new(7))),
            Err(PeerFetchLeaseError::Released)
        );
    }

    #[test]
    fn idempotent_fetch_requires_the_same_keys_and_expiry() {
        let registry = PeerFetchLeaseRegistry::new(2, 2).unwrap();
        let lease = G2pbFetchLeaseId::new();
        registry
            .fetch(lease, &[11], 100, |_| Some(Arc::new(7)))
            .unwrap();
        assert_eq!(
            registry.fetch(lease, &[12], 100, |_| Some(Arc::new(8))),
            Err(PeerFetchLeaseError::RetryMismatch)
        );
        assert_eq!(
            registry.fetch(lease, &[11], 101, |_| Some(Arc::new(7))),
            Err(PeerFetchLeaseError::RetryMismatch)
        );
    }
}
