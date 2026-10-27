// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Serving pages from a replica's own L3 (a node-local Mooncake with SSD
//! offload, say) to its peers, through host staging slots.
//!
//! The replica index only describes host-tier pages: once HiCache reuses a
//! page's host slot the page leaves the index, even though the replica's L3
//! still holds it. A replica whose L3 is node-local (not a shared Mooncake
//! cluster) therefore keeps every L3-only page to itself.
//!
//! This module advertises those pages under a second, *staged* key per page
//! ([`hicache_g2_staged_key`]) and serves them in two steps: on a peer's staged
//! fetch the source loads the pages from its L3 into a small set of host
//! staging slots it owns ([`StagingArea`]), then answers like a host-tier fetch
//! with the staging slots' descriptors and the checksums of what landed there.
//! The reader RDMA-reads the slots and verifies them, as for any peer page.
//!
//! Unlike a host-tier page, a staged page *is* pinned by its lease: the staging
//! slots belong to G2Direct, not to HiCache, and stay out of reuse until the
//! reader releases the lease or it expires (plus [`STAGING_LEASE_GRACE`]).
//!
//! What the L3 holds is only partly visible here: this replica learns of a page
//! when it stores or loads it, and of its loss from the L3's KV events when it
//! follows them ([`L3Directory::apply`]). Neither is complete (Mooncake drops
//! some objects without an event, and events can be lost), so a page that keeps
//! failing to load is withdrawn too ([`L3Directory::record_load`]), and the
//! directory is capped.

use super::hicache_g2::HiCachePageSegments;
use super::lease::G2pbFetchLeaseId;
use dynamo_tokens::SequenceHash;

use anyhow::Result;
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Duration;
use xxhash_rust::xxh3::Xxh3;

const HICACHE_G2_STAGED_KEY_DOMAIN: &[u8] = b"b10-hicache-g2-l3-staged-v1";

/// How long past the reader's lease expiry a staging slot stays held when the
/// reader never releases it. The expiry is the reader's wall clock; this covers
/// clock skew between nodes, so a slot is not refilled under a read the reader
/// still counts as in time.
pub const STAGING_LEASE_GRACE: Duration = Duration::from_secs(5);

/// Longest a reader's lease may hold staging slots, whatever expiry it asks
/// for: a reader with a far clock or a long TTL must not pin the pool.
pub const MAX_STAGING_LEASE: Duration = Duration::from_secs(60);

/// How long a slot a failed load wrote into stays out of use. A load that
/// reports failure has not proven every write it started has stopped; a late
/// write landing in the slot's next use would be hashed and served as that
/// next page.
pub const FAILED_SLOT_QUARANTINE: Duration = Duration::from_secs(30);

/// Released lease ids remembered, so a fetch delivered after its own release
/// (a retry, or a reader that timed out and released first) is refused rather
/// than pinning slots no reader will release.
const MAX_RELEASED_TOMBSTONES: usize = 65_536;

/// Failed loads of one page, in a row, after which it is withdrawn although
/// the L3's events still list it: events miss some removals (and can be lost),
/// and an advertised page that never loads costs every reader a lease.
pub const FAILED_LOADS_BEFORE_WITHDRAW: u8 = 3;

/// The directory key a page's L3 copy is advertised under.
///
/// A distinct key, rather than a tier flag on the host entry, so a source can
/// advertise both copies of a page independently, and so readers that predate
/// staging never see these entries: they never query the staged key.
pub fn hicache_g2_staged_key(key: SequenceHash) -> SequenceHash {
    let mut hasher = Xxh3::new();
    hasher.update(HICACHE_G2_STAGED_KEY_DOMAIN);
    hasher.update(&key.to_le_bytes());
    hasher.digest()
}

/// Loads pages from the engine's L3 into staging slots. Implemented by the
/// engine (the Python HiCache backend), which alone knows how its L3 names and
/// lays out a page.
pub trait HiCacheL3Stager: Send + Sync {
    /// Load the page each of `tokens` names, of staging pool `pool`, into the
    /// staging slot at the same position in `slots`. Returns one flag per page.
    fn stage(&self, pool: &str, tokens: &[String], slots: &[usize]) -> Result<Vec<bool>>;
}

/// One page this replica's L3 is believed to hold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagedEntry {
    pub pool: Arc<str>,
    /// What the stager needs to find the page in the L3 (the engine's own key).
    pub token: String,
    pub page_bytes: u64,
    /// The L3 objects the page is stored as ([`hicache_g2_l3_object_id`] of
    /// each name), when the engine names them; see [`L3Directory::apply`].
    objects: Vec<u64>,
    /// Loads that failed in a row; see [`L3Directory::record_load`].
    failed_loads: u8,
    tick: u64,
}

/// Directory id of an L3 object name: a hash, so tracking an object costs a
/// few bytes rather than its name.
pub fn hicache_g2_l3_object_id(object_key: &str) -> u64 {
    let mut hasher = Xxh3::new();
    hasher.update(object_key.as_bytes());
    hasher.digest()
}

/// What happened to one L3 object, from the L3's own event stream.
///
/// Mooncake's master (at our pin) reports an object's removal once, when no
/// copy of it is left in any tier (eviction), or when it is explicitly removed;
/// a copy moving between memory and disk is not reported. So a removal means
/// the object is gone, whatever tier the event names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum L3ObjectEvent {
    Stored {
        object: u64,
    },
    Removed {
        object: u64,
    },
    /// The L3 dropped everything.
    Cleared,
}

/// The pages this replica advertises from its L3, least recently stored or
/// loaded first out once the directory is full.
#[derive(Debug)]
pub struct L3Directory {
    by_key: HashMap<SequenceHash, StagedEntry>,
    by_tick: BTreeMap<u64, SequenceHash>,
    /// Which page names each object (the latest to name it).
    objects: HashMap<u64, SequenceHash>,
    next_tick: u64,
    capacity: usize,
}

impl L3Directory {
    pub fn new(capacity: usize) -> Self {
        Self {
            by_key: HashMap::new(),
            by_tick: BTreeMap::new(),
            objects: HashMap::new(),
            next_tick: 0,
            capacity,
        }
    }

    pub fn len(&self) -> usize {
        self.by_key.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_key.is_empty()
    }

    pub fn contains(&self, key: &SequenceHash) -> bool {
        self.by_key.contains_key(key)
    }

    /// Record (or refresh) `key`, stored as `objects` in the L3 (empty when
    /// the engine does not name them). Returns whether the key is new, which
    /// is when it must be advertised, and the keys pushed out to stay within
    /// capacity, which must leave the replica index. A batch can push out a
    /// key it added itself; the caller nets the two (see `publish_staged`).
    pub fn insert(
        &mut self,
        key: SequenceHash,
        pool: Arc<str>,
        token: String,
        page_bytes: u64,
        objects: Vec<u64>,
    ) -> (bool, Vec<SequenceHash>) {
        if self.capacity == 0 {
            return (false, Vec::new());
        }
        let tick = self.next_tick;
        self.next_tick += 1;
        for object in &objects {
            self.objects.insert(*object, key);
        }
        let added = match self.by_key.get_mut(&key) {
            Some(entry) => {
                self.by_tick.remove(&entry.tick);
                let stale = std::mem::replace(&mut entry.objects, objects);
                entry.tick = tick;
                entry.pool = pool;
                entry.token = token;
                entry.page_bytes = page_bytes;
                entry.failed_loads = 0;
                for object in stale {
                    if !entry.objects.contains(&object) && self.objects.get(&object) == Some(&key) {
                        self.objects.remove(&object);
                    }
                }
                false
            }
            None => {
                self.by_key.insert(
                    key,
                    StagedEntry {
                        pool,
                        token,
                        page_bytes,
                        objects,
                        failed_loads: 0,
                        tick,
                    },
                );
                true
            }
        };
        self.by_tick.insert(tick, key);
        let mut evicted = Vec::new();
        while self.by_key.len() > self.capacity {
            let Some(oldest) = self.by_tick.first_key_value().map(|(_, key)| *key) else {
                break;
            };
            self.remove(&oldest);
            evicted.push(oldest);
        }
        (added, evicted)
    }

    pub fn remove(&mut self, key: &SequenceHash) -> bool {
        match self.by_key.remove(key) {
            Some(entry) => {
                self.by_tick.remove(&entry.tick);
                for object in entry.objects {
                    if self.objects.get(&object) == Some(key) {
                        self.objects.remove(&object);
                    }
                }
                true
            }
            None => false,
        }
    }

    pub fn get(&self, key: &SequenceHash) -> Option<&StagedEntry> {
        self.by_key.get(key)
    }

    /// Every page held, with its bytes: what a resync snapshot advertises.
    pub fn pages(&self) -> impl Iterator<Item = (SequenceHash, u64)> + '_ {
        self.by_key
            .iter()
            .map(|(key, entry)| (*key, entry.page_bytes))
    }

    /// Apply one L3 event. Returns the pages it removed from the directory,
    /// which must leave the replica index: the page naming a removed object,
    /// or every page when the L3 was cleared. Events about objects no page
    /// here names are ignored.
    pub fn apply(&mut self, event: &L3ObjectEvent) -> Vec<SequenceHash> {
        match event {
            L3ObjectEvent::Stored { .. } => Vec::new(),
            L3ObjectEvent::Removed { object } => match self.objects.remove(object) {
                Some(page) if self.remove(&page) => vec![page],
                _ => Vec::new(),
            },
            L3ObjectEvent::Cleared => self.clear(),
        }
    }

    /// Forget every page (the L3 restarted, so its state is unknown). Returns
    /// them, for withdrawal.
    pub fn clear(&mut self) -> Vec<SequenceHash> {
        let pages = self.by_key.keys().copied().collect();
        self.by_key.clear();
        self.by_tick.clear();
        self.objects.clear();
        pages
    }

    /// Record a staging load of `key`. A failed one counts against the page;
    /// returns whether it is now taken as gone (and removed): at once when
    /// the L3's events are not followed or name none of its objects (nothing
    /// else would retire it), else after [`FAILED_LOADS_BEFORE_WITHDRAW`]
    /// failures in a row, since a single failure may be a read the L3 could
    /// not serve rather than a lost page.
    pub fn record_load(&mut self, key: &SequenceHash, loaded: bool, events_followed: bool) -> bool {
        let Some(entry) = self.by_key.get_mut(key) else {
            return false;
        };
        if loaded {
            entry.failed_loads = 0;
            return false;
        }
        entry.failed_loads = entry.failed_loads.saturating_add(1);
        let gone = !events_followed
            || entry.objects.is_empty()
            || entry.failed_loads >= FAILED_LOADS_BEFORE_WITHDRAW;
        if gone {
            self.remove(key);
        }
        gone
    }
}

/// One event as Mooncake's master publishes it (RFC #1527 KV events), keeping
/// only the fields this directory reads.
#[derive(Debug, Deserialize)]
struct WireEvent {
    event_type: String,
    #[serde(default)]
    object_key: Option<String>,
}

/// A published batch: `[timestamp_ms, [event, ...], dp_rank]`.
#[derive(Debug, Deserialize)]
struct WireBatch(
    serde::de::IgnoredAny,
    Vec<WireEvent>,
    #[serde(default)] serde::de::IgnoredAny,
);

/// Decode one Mooncake KV-event payload (the last frame of a message). Events
/// that name no object are skipped: they cannot be applied.
pub fn decode_l3_events(payload: &[u8]) -> Result<Vec<L3ObjectEvent>> {
    let WireBatch(_, events, _) = rmp_serde::from_slice(payload)
        .map_err(|error| anyhow::anyhow!("undecodable L3 event batch: {error}"))?;
    Ok(events
        .into_iter()
        .filter_map(|event| match event.event_type.as_str() {
            "cleared" => Some(L3ObjectEvent::Cleared),
            kind @ ("stored" | "removed") => {
                let object = hicache_g2_l3_object_id(event.object_key.as_deref()?);
                Some(if kind == "stored" {
                    L3ObjectEvent::Stored { object }
                } else {
                    L3ObjectEvent::Removed { object }
                })
            }
            _ => None,
        })
        .collect())
}

/// Staging slots of one engine pool: same geometry as that pool's host pages.
struct StagingPool {
    slots: Vec<HiCachePageSegments>,
    free: Vec<usize>,
    /// Slots a failed load wrote into, and when they may be used again.
    quarantined: Vec<(usize, u64)>,
}

struct StagingLease {
    pool: Arc<str>,
    slots: Vec<usize>,
    /// Held until then even without a release; see [`STAGING_LEASE_GRACE`].
    hold_until_unix_ms: u64,
    /// Set while the stager writes the slots. Never expired then: a second
    /// lease would be handed slots the first stager is still writing.
    staging: bool,
    /// The reader released the lease while it was being staged (it gave up
    /// waiting): the slots stay reserved until the stager finishes.
    release_requested: bool,
}

/// Slots granted to one lease.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagingGrant {
    pub slots: Vec<usize>,
    pub pages: Vec<HiCachePageSegments>,
}

/// Why [`StagingArea::reserve`] granted nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StagingRefusal {
    /// No free slot in the pool (or no such pool).
    Busy,
    /// The lease exists or was released already.
    KnownLease,
}

/// The host staging slots this replica serves L3 pages through, per pool, and
/// the leases holding them.
#[derive(Default)]
pub struct StagingArea {
    pools: HashMap<Arc<str>, StagingPool>,
    leases: HashMap<G2pbFetchLeaseId, StagingLease>,
    released: HashSet<G2pbFetchLeaseId>,
    release_order: VecDeque<G2pbFetchLeaseId>,
}

impl StagingArea {
    /// Add the staging slots of `pool`. A pool registers once.
    pub fn register(&mut self, pool: Arc<str>, slots: Vec<HiCachePageSegments>) -> Result<()> {
        anyhow::ensure!(!slots.is_empty(), "staging pool {pool} has no slots");
        anyhow::ensure!(
            !self.pools.contains_key(&pool),
            "staging pool {pool} is already registered"
        );
        let free = (0..slots.len()).rev().collect();
        self.pools.insert(
            pool,
            StagingPool {
                slots,
                free,
                quarantined: Vec::new(),
            },
        );
        Ok(())
    }

    pub fn has_pool(&self, pool: &str) -> bool {
        self.pools.contains_key(pool)
    }

    pub fn free_slots(&self, pool: &str) -> usize {
        self.pools.get(pool).map_or(0, |pool| pool.free.len())
    }

    /// Return the slots of leases past their hold time, and quarantined slots
    /// past their quarantine, to their pools. Returns how many leases expired.
    pub fn expire(&mut self, now_unix_ms: u64) -> usize {
        for pool in self.pools.values_mut() {
            let (ready, waiting) = std::mem::take(&mut pool.quarantined)
                .into_iter()
                .partition::<Vec<_>, _>(|(_, until)| *until <= now_unix_ms);
            pool.quarantined = waiting;
            pool.free.extend(ready.into_iter().map(|(slot, _)| slot));
        }
        let expired = self
            .leases
            .iter()
            .filter(|(_, lease)| !lease.staging && lease.hold_until_unix_ms <= now_unix_ms)
            .map(|(lease_id, _)| *lease_id)
            .collect::<Vec<_>>();
        for lease_id in &expired {
            self.release(*lease_id);
        }
        expired.len()
    }

    /// Grant up to `pages` free slots of `pool` to `lease_id`, marked as being
    /// staged. The lease is held to `expires_at_unix_ms`, at most
    /// [`MAX_STAGING_LEASE`] from `now_unix_ms`, plus [`STAGING_LEASE_GRACE`].
    pub fn reserve(
        &mut self,
        pool: &Arc<str>,
        pages: usize,
        lease_id: G2pbFetchLeaseId,
        expires_at_unix_ms: u64,
        now_unix_ms: u64,
    ) -> std::result::Result<StagingGrant, StagingRefusal> {
        if self.leases.contains_key(&lease_id) || self.released.contains(&lease_id) {
            return Err(StagingRefusal::KnownLease);
        }
        let staging_pool = self.pools.get_mut(pool).ok_or(StagingRefusal::Busy)?;
        let count = pages.min(staging_pool.free.len());
        if count == 0 {
            return Err(StagingRefusal::Busy);
        }
        let slots = staging_pool.free.split_off(staging_pool.free.len() - count);
        let pages = slots
            .iter()
            .map(|slot| staging_pool.slots[*slot].clone())
            .collect();
        let hold_until = expires_at_unix_ms
            .min(now_unix_ms.saturating_add(MAX_STAGING_LEASE.as_millis() as u64))
            .saturating_add(STAGING_LEASE_GRACE.as_millis() as u64);
        self.leases.insert(
            lease_id,
            StagingLease {
                pool: Arc::clone(pool),
                slots: slots.clone(),
                hold_until_unix_ms: hold_until,
                staging: true,
                release_requested: false,
            },
        );
        Ok(StagingGrant { slots, pages })
    }

    /// The stager is done with `lease_id`: keep its first `keep` slots under
    /// the lease and give back the rest. Slots a failed load may have written
    /// go to quarantine until `quarantine_until_unix_ms` when it is set, else
    /// straight back. Keeping none ends the lease, and so does a release that
    /// arrived while staging: then the kept slots go back too, since their
    /// reader is gone. Returns whether the lease still holds slots.
    pub fn finish_staging(
        &mut self,
        lease_id: G2pbFetchLeaseId,
        keep: usize,
        quarantine_until_unix_ms: Option<u64>,
    ) -> bool {
        let Some(lease) = self.leases.get_mut(&lease_id) else {
            return false;
        };
        lease.staging = false;
        let released = lease.release_requested;
        let unused = lease.slots.split_off(keep.min(lease.slots.len()));
        let pool_name = Arc::clone(&lease.pool);
        let mut kept = Vec::new();
        if keep == 0 || released {
            kept = self
                .leases
                .remove(&lease_id)
                .map_or_else(Vec::new, |lease| lease.slots);
            self.remember_released(lease_id);
        }
        if let Some(pool) = self.pools.get_mut(&pool_name) {
            // Fully written slots nobody will read are free at once.
            pool.free.extend(kept);
            match quarantine_until_unix_ms {
                Some(until) => pool
                    .quarantined
                    .extend(unused.into_iter().map(|slot| (slot, until))),
                None => pool.free.extend(unused),
            }
        }
        keep != 0 && !released
    }

    /// Return a lease's slots, and remember the id so a fetch for it that
    /// arrives later is refused. A lease still being staged keeps its slots
    /// until [`Self::finish_staging`], which then frees or quarantines them:
    /// the stager may still be writing them. Returns whether the lease held
    /// slots.
    pub fn release(&mut self, lease_id: G2pbFetchLeaseId) -> bool {
        self.remember_released(lease_id);
        let Some(lease) = self.leases.get_mut(&lease_id) else {
            return false;
        };
        if lease.staging {
            lease.release_requested = true;
            return true;
        }
        let Some(lease) = self.leases.remove(&lease_id) else {
            return false;
        };
        if let Some(pool) = self.pools.get_mut(&lease.pool) {
            pool.free.extend(lease.slots);
        }
        true
    }

    fn remember_released(&mut self, lease_id: G2pbFetchLeaseId) {
        if self.released.insert(lease_id) {
            self.release_order.push_back(lease_id);
            while self.release_order.len() > MAX_RELEASED_TOMBSTONES {
                if let Some(oldest) = self.release_order.pop_front() {
                    self.released.remove(&oldest);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot(index: u64) -> HiCachePageSegments {
        HiCachePageSegments {
            segments: vec![(0x1000 + index * 0x100, 0x100)],
        }
    }

    fn pool(name: &str) -> Arc<str> {
        Arc::from(name)
    }

    fn id(name: &str) -> u64 {
        hicache_g2_l3_object_id(name)
    }

    fn removed(name: &str) -> L3ObjectEvent {
        L3ObjectEvent::Removed { object: id(name) }
    }

    fn area(slots: u64) -> StagingArea {
        let mut area = StagingArea::default();
        area.register(pool("kv"), (0..slots).map(slot).collect())
            .unwrap();
        area
    }

    const NOW: u64 = 1_000_000;

    #[test]
    fn staged_keys_are_distinct_from_host_keys_and_stable() {
        let key = 0x1234_5678_9abc_def0;
        assert_ne!(hicache_g2_staged_key(key), key);
        assert_eq!(hicache_g2_staged_key(key), hicache_g2_staged_key(key));
        assert_ne!(hicache_g2_staged_key(key), hicache_g2_staged_key(key + 1));
    }

    #[test]
    fn directory_evicts_the_least_recently_recorded_page() {
        let mut directory = L3Directory::new(2);
        assert_eq!(
            directory.insert(1, pool("kv"), "a".into(), 64, vec![]),
            (true, vec![])
        );
        assert_eq!(
            directory.insert(2, pool("kv"), "b".into(), 64, vec![]),
            (true, vec![])
        );
        // Refreshing 1 makes 2 the oldest.
        assert_eq!(
            directory.insert(1, pool("kv"), "a".into(), 64, vec![]),
            (false, vec![])
        );
        assert_eq!(
            directory.insert(3, pool("kv"), "c".into(), 64, vec![]),
            (true, vec![2])
        );
        assert!(directory.contains(&1));
        assert!(!directory.contains(&2));
        assert!(directory.remove(&3));
        assert!(!directory.remove(&3));
        assert_eq!(directory.len(), 1);
    }

    #[test]
    fn a_zero_capacity_directory_records_nothing() {
        let mut directory = L3Directory::new(0);
        assert_eq!(
            directory.insert(1, pool("kv"), "a".into(), 64, vec![]),
            (false, vec![])
        );
        assert!(directory.is_empty());
    }

    #[test]
    fn any_removal_of_one_of_its_objects_withdraws_a_page() {
        // Mooncake reports a removal only once no copy is left (whatever tier
        // it names), and nothing when a copy moves to disk.
        let mut directory = L3Directory::new(8);
        directory.insert(1, pool("kv"), "a".into(), 64, vec![id("a_k"), id("a_v")]);
        assert!(
            directory
                .apply(&L3ObjectEvent::Stored { object: id("a_k") })
                .is_empty()
        );
        assert_eq!(directory.apply(&removed("a_v")), vec![1]);
        assert!(!directory.contains(&1));
        // Its other object is no longer tracked.
        assert!(directory.apply(&removed("a_k")).is_empty());
        assert!(directory.apply(&removed("unknown")).is_empty());
    }

    #[test]
    fn an_object_follows_the_latest_page_to_name_it() {
        let mut directory = L3Directory::new(8);
        directory.insert(1, pool("kv"), "a".into(), 64, vec![id("x")]);
        directory.insert(2, pool("kv"), "b".into(), 64, vec![id("x")]);
        // Dropping the first page does not untrack the second's object.
        assert!(directory.remove(&1));
        assert_eq!(directory.apply(&removed("x")), vec![2]);
    }

    #[test]
    fn clears_and_refreshes_keep_objects_consistent() {
        let mut directory = L3Directory::new(1);
        directory.insert(1, pool("kv"), "a".into(), 64, vec![id("a")]);
        directory.insert(1, pool("kv"), "a".into(), 64, vec![id("a2")]);
        assert!(directory.apply(&removed("a")).is_empty());
        assert_eq!(
            directory.insert(2, pool("kv"), "b".into(), 64, vec![]).1,
            vec![1]
        );
        assert!(directory.apply(&removed("a2")).is_empty());
        assert_eq!(directory.apply(&L3ObjectEvent::Cleared), vec![2]);
        assert!(directory.is_empty());
    }

    #[test]
    fn failed_loads_withdraw_at_once_without_events_and_after_a_streak_with_them() {
        let mut directory = L3Directory::new(8);
        directory.insert(1, pool("kv"), "a".into(), 64, vec![id("a")]);
        assert!(directory.record_load(&1, false, false));
        assert!(!directory.contains(&1));

        directory.insert(2, pool("kv"), "b".into(), 64, vec![id("b")]);
        assert!(!directory.record_load(&2, false, true));
        assert!(!directory.record_load(&2, false, true));
        // A load in between resets the streak.
        assert!(!directory.record_load(&2, true, true));
        assert!(!directory.record_load(&2, false, true));
        assert!(!directory.record_load(&2, false, true));
        assert!(directory.record_load(&2, false, true));
        assert!(!directory.contains(&2));

        // A page whose objects events cannot name goes at once.
        directory.insert(3, pool("kv"), "c".into(), 64, vec![]);
        assert!(directory.record_load(&3, false, true));
        assert!(!directory.record_load(&9, false, true));
    }

    #[test]
    fn decodes_mooncake_event_batches() {
        // As KvEventPublisher::PublishBatch packs it, with the legacy fields.
        let batch = serde_json::json!([
            1_760_000_000_000i64,
            [
                {"event_id": 1, "timestamp": 1, "event_type": "stored", "type": "BlockStored",
                 "model_name": null, "block_size": null, "medium": "cpu", "dp_rank": 0,
                 "group_id": null, "object_key": "k1", "seq_hashes": [], "block_hashes": [],
                 "session_id": null, "base_block_idx": null, "parent_hash": null,
                 "token_ids": null, "parent_block_hash": null},
                {"event_id": 2, "timestamp": 1, "event_type": "removed", "type": "BlockRemoved",
                 "medium": "disk", "object_key": "k2", "seq_hashes": [], "base_block_idx": null},
                {"event_id": 3, "timestamp": 1, "event_type": "removed", "medium": "cpu"},
                {"event_id": 4, "timestamp": 1, "event_type": "cleared", "type": "AllBlocksCleared",
                 "medium": null, "dp_rank": 0}
            ],
            0
        ]);
        let payload = rmp_serde::to_vec(&batch).unwrap();
        assert_eq!(
            decode_l3_events(&payload).unwrap(),
            vec![
                L3ObjectEvent::Stored { object: id("k1") },
                removed("k2"),
                L3ObjectEvent::Cleared
            ]
        );
        assert!(decode_l3_events(b"not msgpack").is_err());
    }

    #[test]
    fn reserve_grants_what_is_free_and_release_returns_it() {
        let mut area = area(3);
        let kv = pool("kv");
        let first = G2pbFetchLeaseId::new();
        let grant = area.reserve(&kv, 2, first, NOW + 1_000, NOW).unwrap();
        assert_eq!((grant.slots.len(), grant.pages.len()), (2, 2));
        // Only one slot is left: a partial grant, not a refusal.
        let second = G2pbFetchLeaseId::new();
        assert_eq!(
            area.reserve(&kv, 4, second, NOW + 1_000, NOW)
                .unwrap()
                .slots
                .len(),
            1
        );
        assert_eq!(
            area.reserve(&kv, 1, G2pbFetchLeaseId::new(), NOW + 1_000, NOW),
            Err(StagingRefusal::Busy)
        );
        assert_eq!(
            area.reserve(&kv, 1, first, NOW + 1_000, NOW),
            Err(StagingRefusal::KnownLease)
        );
        // Served, then released by its reader.
        assert!(area.finish_staging(first, 2, None));
        assert!(area.release(first));
        assert_eq!(area.free_slots("kv"), 2);
        assert!(!area.release(first));
    }

    #[test]
    fn a_fetch_arriving_after_its_release_is_refused() {
        // A reader that timed out releases before the source served it.
        let mut area = area(2);
        let lease = G2pbFetchLeaseId::new();
        assert!(!area.release(lease));
        assert_eq!(
            area.reserve(&pool("kv"), 1, lease, NOW + 1_000, NOW),
            Err(StagingRefusal::KnownLease)
        );
        assert_eq!(area.free_slots("kv"), 2);
    }

    #[test]
    fn finishing_keeps_the_staged_prefix_and_quarantines_failed_slots() {
        let mut area = area(4);
        let kv = pool("kv");
        let lease = G2pbFetchLeaseId::new();
        // Held well past the quarantine, so only quarantine decides the slots.
        area.reserve(&kv, 4, lease, NOW + 50_000, NOW).unwrap();
        let until = NOW + FAILED_SLOT_QUARANTINE.as_millis() as u64;
        area.finish_staging(lease, 1, Some(until));
        assert_eq!(area.free_slots("kv"), 0);
        // Not before the quarantine ends, however often leases expire.
        area.expire(until - 1);
        assert_eq!(area.free_slots("kv"), 0);
        area.expire(until);
        assert_eq!(area.free_slots("kv"), 3);
        area.release(lease);
        assert_eq!(area.free_slots("kv"), 4);
    }

    #[test]
    fn finishing_with_nothing_staged_ends_the_lease() {
        let mut area = area(2);
        let lease = G2pbFetchLeaseId::new();
        area.reserve(&pool("kv"), 2, lease, NOW + 1_000, NOW)
            .unwrap();
        area.finish_staging(lease, 0, None);
        assert_eq!(area.free_slots("kv"), 2);
        assert!(!area.release(lease));
        assert_eq!(
            area.reserve(&pool("kv"), 1, lease, NOW + 1_000, NOW),
            Err(StagingRefusal::KnownLease)
        );
    }

    #[test]
    fn a_release_while_staging_holds_the_slots_until_staging_finishes() {
        let mut area = area(3);
        let kv = pool("kv");
        let lease = G2pbFetchLeaseId::new();
        area.reserve(&kv, 3, lease, NOW + 1_000, NOW).unwrap();
        // The reader gave up while the stager is still writing: nothing is
        // handed out again yet, and the lease id is already refused.
        assert!(area.release(lease));
        assert_eq!(area.free_slots("kv"), 0);
        assert_eq!(
            area.reserve(&kv, 1, G2pbFetchLeaseId::new(), NOW + 1_000, NOW),
            Err(StagingRefusal::Busy)
        );
        assert_eq!(
            area.reserve(&kv, 1, lease, NOW + 1_000, NOW),
            Err(StagingRefusal::KnownLease)
        );
        // Staging ends with one page served and two failed: the lease is gone,
        // the served slot is free (nobody will read it) and the rest wait out
        // the quarantine.
        let until = NOW + FAILED_SLOT_QUARANTINE.as_millis() as u64;
        assert!(!area.finish_staging(lease, 1, Some(until)));
        assert_eq!(area.free_slots("kv"), 1);
        area.expire(until);
        assert_eq!(area.free_slots("kv"), 3);
        assert!(!area.release(lease));
    }

    #[test]
    fn expiry_waits_for_the_grace_and_never_takes_slots_being_staged() {
        let mut area = area(2);
        let kv = pool("kv");
        let grace = STAGING_LEASE_GRACE.as_millis() as u64;
        let staging = G2pbFetchLeaseId::new();
        area.reserve(&kv, 1, staging, NOW + 1_000, NOW).unwrap();
        let served = G2pbFetchLeaseId::new();
        area.reserve(&kv, 1, served, NOW + 1_000, NOW).unwrap();
        area.finish_staging(served, 1, None);
        assert_eq!(area.expire(NOW + 1_000), 0);
        // Past the grace, only the lease whose staging finished is expired.
        assert_eq!(area.expire(NOW + 1_000 + grace), 1);
        assert_eq!(area.free_slots("kv"), 1);
        area.finish_staging(staging, 1, None);
        assert_eq!(area.expire(NOW + 1_000 + grace), 1);
        assert_eq!(area.free_slots("kv"), 2);
    }

    #[test]
    fn a_far_expiry_is_capped() {
        let mut area = area(1);
        let lease = G2pbFetchLeaseId::new();
        area.reserve(&pool("kv"), 1, lease, u64::MAX, NOW).unwrap();
        area.finish_staging(lease, 1, None);
        let cap = (MAX_STAGING_LEASE + STAGING_LEASE_GRACE).as_millis() as u64;
        assert_eq!(area.expire(NOW + cap - 1), 0);
        assert_eq!(area.expire(NOW + cap), 1);
    }

    #[test]
    fn unknown_pools_grant_nothing_and_pools_register_once() {
        let mut area = StagingArea::default();
        assert_eq!(
            area.reserve(&pool("kv"), 1, G2pbFetchLeaseId::new(), NOW + 1, NOW),
            Err(StagingRefusal::Busy)
        );
        area.register(pool("kv"), vec![slot(0)]).unwrap();
        assert!(area.register(pool("kv"), vec![slot(1)]).is_err());
        assert!(area.register(pool("indexer"), vec![]).is_err());
    }
}

#[cfg(test)]
mod subscriber_socket_tests {
    /// `HiCacheG2Direct::subscribe_l3_events` is called from an engine thread
    /// that is not inside the runtime; building the SUB socket there panics
    /// ("there is no reactor running") unless the runtime is entered first.
    #[test]
    fn a_sub_socket_builds_from_a_foreign_thread_once_the_runtime_is_entered() {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        let handle = runtime.handle().clone();
        std::thread::spawn(move || {
            let _runtime = handle.enter();
            tmq::subscribe(&tmq::Context::new())
                .connect("tcp://127.0.0.1:59999")
                .unwrap()
                .subscribe(b"")
                .unwrap();
        })
        .join()
        .unwrap();
    }
}
