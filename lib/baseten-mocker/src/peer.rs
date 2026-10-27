// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! G2PB peer directory: workers advertise the pages on their HiCache host tier
//! and look up pages held by other workers.
//!
//! [`PeerIndex`] is the in-process directory (offline replay, `--num-workers`).
//! Live workers get a fleet-wide one over the event plane injected through
//! `MockEngineArgs::peer_directory` (see the `g2pb` module).

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};

type Group = HashMap<u64, HashSet<u64>>;

static GROUPS: LazyLock<Mutex<HashMap<String, Group>>> = LazyLock::new(Default::default);
static NEXT_OWNER: AtomicU64 = AtomicU64::new(0);

/// Where a worker publishes its host pages and finds other workers' pages.
pub trait PeerDirectory: Send + Sync {
    fn publish(&self, hashes: &[u64]);
    fn withdraw(&self, hashes: &[u64]);
    /// Number of leading `hashes` held by some other worker.
    fn contiguous_hits(&self, hashes: &[u64]) -> usize;
}

/// Shareable handle so `MockEngineArgs` stays `Debug + Clone`.
#[derive(Clone)]
pub struct PeerDirectoryHandle(pub std::sync::Arc<dyn PeerDirectory>);

impl std::fmt::Debug for PeerDirectoryHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PeerDirectoryHandle")
    }
}

/// Process-global directory shared by workers with the same group name.
pub struct PeerIndex {
    group: String,
    owner: u64,
}

impl PeerIndex {
    pub fn new(group: &str) -> Self {
        Self {
            group: group.to_string(),
            owner: NEXT_OWNER.fetch_add(1, Ordering::Relaxed),
        }
    }
}

impl PeerDirectory for PeerIndex {
    fn publish(&self, hashes: &[u64]) {
        let mut groups = GROUPS.lock().unwrap();
        let group = groups.entry(self.group.clone()).or_default();
        for &hash in hashes {
            group.entry(hash).or_default().insert(self.owner);
        }
    }

    fn withdraw(&self, hashes: &[u64]) {
        let mut groups = GROUPS.lock().unwrap();
        let Some(group) = groups.get_mut(&self.group) else {
            return;
        };
        for hash in hashes {
            if let Some(owners) = group.get_mut(hash) {
                owners.remove(&self.owner);
                if owners.is_empty() {
                    group.remove(hash);
                }
            }
        }
    }

    fn contiguous_hits(&self, hashes: &[u64]) -> usize {
        let groups = GROUPS.lock().unwrap();
        let Some(group) = groups.get(&self.group) else {
            return 0;
        };
        hashes
            .iter()
            .take_while(|hash| {
                group
                    .get(hash)
                    .is_some_and(|owners| owners.iter().any(|&o| o != self.owner))
            })
            .count()
    }
}

impl Drop for PeerIndex {
    fn drop(&mut self) {
        let mut groups = GROUPS.lock().unwrap();
        if let Some(group) = groups.get_mut(&self.group) {
            group.retain(|_, owners| {
                owners.remove(&self.owner);
                !owners.is_empty()
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hits_only_count_other_owners_and_stop_at_gaps() {
        let a = PeerIndex::new("peer-index-test");
        let b = PeerIndex::new("peer-index-test");
        a.publish(&[1, 2, 4]);
        assert_eq!(b.contiguous_hits(&[1, 2, 3, 4]), 2);
        assert_eq!(a.contiguous_hits(&[1, 2]), 0);
        a.withdraw(&[1]);
        assert_eq!(b.contiguous_hits(&[1, 2]), 0);
        drop(a);
        assert_eq!(b.contiguous_hits(&[2]), 0);
    }
}
