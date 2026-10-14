// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Direct G2-to-G2 page sharing for engines that own their own host tier.
//!
//! Direct peer transport uses an advisory NATS
//! replica index, source-page lease, RDMA read, checksum verification — but the
//! host tier belongs to the inference engine (SGLang's HiCache L2), not to KVBM.
//! A page is therefore described by raw byte segments inside a foreign pinned
//! allocation rather than by a KVBM block index, so peers exchange NIXL
//! descriptors directly instead of KVBM block descriptors.
//!
//! Residency is never proven by the index. A page is published with the xxh3 of
//! its bytes and the destination re-hashes what it actually received, so a slot
//! the engine silently recycled fails the fetch instead of serving the wrong
//! tokens. Nothing on the source side can promise more: HiCache never tells this
//! backend when it rewrites a host slot, so no source-side check is sound.

use super::lease::{G2pbFetchLeaseId, release_fetch_lease_bounded, unix_time_ms};
use super::nixl::create_worker_ucx_backend;
use super::nixl::new_hicache_g2_nixl_agent;
use super::peer_index::{
    ReplicaBlock, ReplicaIndex, ReplicaLocation, ReplicaPublisher, ReplicaSource, ReplicaSourceKey,
    SERVING_G2_INDEX_COMPONENT, SERVING_G2_PEER_COMPONENT, baseten_row_from_env,
    g2pb_replica_id_from_env, serving_g2_namespace_from_env, start_replica_index_service,
};
use super::peer_lease::{PeerFetchLeaseError, PeerFetchLeaseRegistry};
use super::quarantine::{G2pbQuarantineTracker, QuarantineMetrics};
use dynamo_tokens::SequenceHash;

use anyhow::{Context, Result};
use dynamo_memory::nixl::NixlDescriptor;
use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::component::Component;
use dynamo_runtime::pipeline::{
    AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream, RouterMode, SingleIn,
    async_trait, network::Ingress, network::egress::push_router::PushRouter,
};
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::stream::{self, StreamExt};
use nixl_sys::{
    Agent as NixlAgent, MemType, RegistrationHandle, XferDescList, XferOp, XferRequest, XferStatus,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock, Weak};
use std::time::{Duration, Instant};
use xxhash_rust::xxh3::Xxh3;

/// Endpoint name on the shared `replica-peer` component. Deliberately distinct
/// from [`super::peer_index::SERVING_G2_PEER_ENDPOINT`] so a KVBM worker and an
/// engine-owned host tier can publish into one index without answering each
/// other's incompatible fetch requests.
pub const HICACHE_G2_PEER_ENDPOINT: &str = "hicache-fetch";
/// Microseconds the NIXL agent's progress thread sleeps between polls. 0 makes
/// UCX poll continuously, which costs a core but removes progress latency from
/// every transfer; the fetch path also busy-polls while a read is outstanding,
/// so the default keeps g2pb's idle-friendly 1 ms.
pub const HICACHE_G2_PROGRESS_DELAY_US_ENV: &str = "DYN_HICACHE_G2_PROGRESS_DELAY_US";
/// Where the payload checksum is re-verified: `destination` (default) or
/// `off`. See [`HiCacheVerify`].
pub const HICACHE_G2_VERIFY_ENV: &str = "DYN_HICACHE_G2_VERIFY";
/// Threads used to hash a batch of pages. Hashing a multi-GiB prefix on one
/// core costs an order of magnitude more than the wire transfer does.
pub const HICACHE_G2_HASH_THREADS_ENV: &str = "DYN_HICACHE_G2_HASH_THREADS";
/// TEST ONLY. The TP rank set here keeps exporting its first pool's NIXL
/// metadata after registering later pools, while still bumping its generation:
/// peers then fail `create_xfer_req` on every page of those pools, which is
/// how an unaddressable peer looks in production. Never set in a deployment.
pub const HICACHE_G2_TEST_FREEZE_METADATA_RANK_ENV: &str =
    "DYN_HICACHE_G2_TEST_FREEZE_METADATA_RANK";
const HICACHE_G2_DEFAULT_HASH_THREADS: usize = 8;
const HICACHE_G2_DEFAULT_PROGRESS_DELAY_US: u64 = 1_000;
const HICACHE_G2_KEY_DOMAIN: &[u8] = b"b10-hicache-g2-v1";
const HICACHE_G2_MAX_ACTIVE_LEASES: usize = 4_096;
const HICACHE_G2_MAX_RELEASED_LEASES: usize = 65_536;
/// Status calls are the progress driver, so the wait loop spins; only check the
/// wall clock every so often.
const HICACHE_G2_DEADLINE_CHECK_POLLS: u32 = 4_096;
/// How often a read that is past its deadline but still in flight is reported.
const HICACHE_G2_QUARANTINE_REPORT_INTERVAL: Duration = Duration::from_secs(5);
/// How often the reaper polls reads that outlived their fetch. Such reads are
/// rare and already late, so this trades latency for an idle core.
const HICACHE_G2_REAPER_POLL_INTERVAL: Duration = Duration::from_millis(1);
/// Longest the idle reaper sleeps before checking that the transport still exists.
const HICACHE_G2_REAPER_IDLE_WAIT: Duration = Duration::from_secs(1);

/// Where a fetched page's payload checksum is re-verified.
///
/// HiCache can recycle a host slot with no notification, so a published page
/// can stop matching its key at any moment and the index is only advisory.
/// Verification is what makes that fail closed instead of serving wrong tokens,
/// and it has to happen on the bytes the destination actually received: a
/// source-side hash can pass and the engine can rewrite the slot before the
/// RDMA read lands, which is why a `Source` mode no longer exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HiCacheVerify {
    /// The destination hashes what it received and compares it with the
    /// checksum the page was published with. Catches a recycled source page
    /// and a corrupted transfer alike.
    Destination,
    /// No verification. Measurement only: a recycled source page silently
    /// yields wrong tokens.
    Off,
}

impl HiCacheVerify {
    pub fn from_env() -> Result<Self> {
        match std::env::var(HICACHE_G2_VERIFY_ENV) {
            Err(_) => Ok(Self::Destination),
            Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
                "destination" | "dest" => Ok(Self::Destination),
                "source" | "src" => anyhow::bail!(
                    "{HICACHE_G2_VERIFY_ENV}=source was removed: a source-side hash cannot \
                     see a host slot the engine rewrites after it, so it verified nothing; \
                     use destination"
                ),
                "off" | "none" => Ok(Self::Off),
                other => anyhow::bail!(
                    "{HICACHE_G2_VERIFY_ENV} must be destination, source or off, got {other:?}"
                ),
            },
        }
    }
}

pub fn hicache_g2_hash_threads() -> Result<usize> {
    let configured = match std::env::var(HICACHE_G2_HASH_THREADS_ENV) {
        Ok(value) => value.trim().parse().with_context(|| {
            format!("{HICACHE_G2_HASH_THREADS_ENV} must be a positive thread count")
        })?,
        Err(_) => HICACHE_G2_DEFAULT_HASH_THREADS,
    };
    anyhow::ensure!(configured > 0, "{HICACHE_G2_HASH_THREADS_ENV} must be > 0");
    let available = std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1);
    Ok(configured.min(available))
}

/// Byte segments of one page inside the publisher's host tier.
///
/// SGLang hands the storage backend one pointer per page for a split k/v pool
/// and two for a fused one, so the count is part of the payload rather than a
/// constant.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HiCachePageSegments {
    pub segments: Vec<(u64, u64)>,
}

impl HiCachePageSegments {
    pub fn total_bytes(&self) -> u64 {
        self.segments.iter().map(|(_, len)| *len).sum()
    }

    /// Split a flat `(ptrs, sizes)` pair, as SGLang's `get_page_buffer_meta`
    /// returns it, into one descriptor per page of `ptrs_per_page` segments.
    /// `publish`, `fetch` and `invalidate_slots` all describe a host slot this
    /// way, which is what lets the published index match a slot across them.
    pub fn split_flat(ptrs: &[u64], sizes: &[u64], ptrs_per_page: usize) -> Result<Vec<Self>> {
        anyhow::ensure!(ptrs_per_page != 0, "segments per page must be positive");
        anyhow::ensure!(
            ptrs.len() == sizes.len(),
            "got {} pointers and {} sizes",
            ptrs.len(),
            sizes.len()
        );
        anyhow::ensure!(
            ptrs.len().is_multiple_of(ptrs_per_page),
            "{} pointers is not a multiple of {ptrs_per_page} segments per page",
            ptrs.len()
        );
        Ok(ptrs
            .chunks(ptrs_per_page)
            .zip(sizes.chunks(ptrs_per_page))
            .map(|(page_ptrs, page_sizes)| Self {
                segments: page_ptrs
                    .iter()
                    .copied()
                    .zip(page_sizes.iter().copied())
                    .collect(),
            })
            .collect())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiCacheFetchRequest {
    pub lease_id: G2pbFetchLeaseId,
    pub keys: Vec<SequenceHash>,
    pub lease_expires_at_unix_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiCacheFetchResponse {
    pub lease_id: G2pbFetchLeaseId,
    pub instance_id: u64,
    pub agent_name: String,
    pub agent_metadata: Vec<u8>,
    /// Generation of the memory map `agent_metadata` describes; bumped by the
    /// source on every `register_host_region`. The reader reloads the peer's
    /// metadata when it advances, see `RemoteMetadataCache`.
    #[serde(default)]
    pub metadata_generation: u64,
    pub pages: Vec<HiCachePageSegments>,
    /// Checksum published for each page, so the destination can verify what it
    /// actually received instead of making the source hash before answering.
    pub checksums: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiCacheReleaseRequest {
    pub lease_id: G2pbFetchLeaseId,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HiCachePeerRequest {
    Fetch(HiCacheFetchRequest),
    Release(HiCacheReleaseRequest),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HiCachePeerResponse {
    Ack,
    Fetch(HiCacheFetchResponse),
}

/// Derive the physical index key for one engine page.
///
/// The caller passes a fully-qualified key string and owns every namespacing
/// decision in it — model, pool, and whether the rank is included. That matters:
/// an MLA KV page is rank-replicated and *must* hash identically on every TP
/// rank (the storage-hit query all-reduces MIN across ranks, so a rank-dependent
/// key would make the agreed prefix zero), while a Mamba/KDA state page is
/// rank-sharded and must not.
pub fn hicache_g2_cache_key(key: &str) -> SequenceHash {
    let mut hasher = Xxh3::new();
    hasher.update(HICACHE_G2_KEY_DOMAIN);
    hasher.update(key.as_bytes());
    hasher.digest()
}

fn hicache_g2_progress_delay_us() -> Result<u64> {
    match std::env::var(HICACHE_G2_PROGRESS_DELAY_US_ENV) {
        Ok(value) => value.trim().parse().with_context(|| {
            format!("{HICACHE_G2_PROGRESS_DELAY_US_ENV} must be a whole number of microseconds")
        }),
        Err(_) => Ok(HICACHE_G2_DEFAULT_PROGRESS_DELAY_US),
    }
}

struct RegisteredRegion {
    addr: u64,
    size: usize,
    /// One handle per registration CHUNK. The region stays one *logical* entry
    /// so `find`/`contains` keep working on the whole pool range and every
    /// reader is unaffected by chunking; only the handles are plural.
    _handles: Vec<RegistrationHandle>,
}

impl RegisteredRegion {
    fn contains(&self, addr: u64, len: u64) -> bool {
        region_contains(self.addr, self.size, addr, len)
    }
}

/// Every host pool the engine asked us to serve, plus the agent metadata that
/// covers all of them. A hybrid model registers several: the KV/MLA pool and
/// one region per Mamba/KDA state buffer.
#[derive(Default)]
struct RegisteredRegions {
    regions: Vec<RegisteredRegion>,
    local_metadata: Vec<u8>,
}

impl RegisteredRegions {
    fn contains(&self, addr: u64, len: u64) -> bool {
        self.regions.iter().any(|region| region.contains(addr, len))
    }

    fn find(&self, addr: u64, size: usize) -> Option<&RegisteredRegion> {
        self.regions
            .iter()
            .find(|region| region.addr == addr && region.size == size)
    }
}

/// This process's NIXL agent name. Readers key every peer map by it, so two
/// processes sharing a name are indistinguishable: a reader holds one of their
/// maps and fails every read of the other's pages, forever.
///
/// It used to be `instance_id ^ tp_rank`. Instance ids are allocated close
/// together (consecutive even numbers in practice), so with TP ranks 1..3 that
/// collided across processes: in prod one agent name carried two processes'
/// pages. The pair below is unique because the instance id is unique per process.
fn hicache_g2_agent_name(instance_id: u64, tp_rank: u32) -> String {
    format!("{instance_id}-{tp_rank}")
}

/// The two NIXL agent calls the metadata cache makes, so the cache's
/// decisions can be tested without an agent.
trait RemoteMetadataImporter {
    /// Import a peer's exported metadata; returns the agent name it describes.
    /// Over a map NIXL already holds for the agent, NIXL 1.4 merges the new
    /// descriptors in, or, if one conflicts with what it holds, rejects the
    /// import and erases that agent's map.
    fn import(&self, metadata: &[u8]) -> Result<String>;
    /// Whether the backend currently holds any metadata for the agent. Cheap:
    /// a lookup in the agent's local remote-section map.
    fn is_loaded(&self, agent_name: &str) -> bool;
}

impl RemoteMetadataImporter for NixlAgent {
    fn import(&self, metadata: &[u8]) -> Result<String> {
        self.load_remote_md(metadata)
            .map_err(|error| anyhow::anyhow!("load_remote_md: {error}"))
    }

    fn is_loaded(&self, agent_name: &str) -> bool {
        self.check_remote_metadata(agent_name, None)
    }
}

#[derive(Debug, PartialEq, Eq)]
enum RemoteMetadataLoad {
    /// NIXL holds this generation (or a newer one) for the peer, and says so.
    AlreadyCurrent,
    Imported {
        previous_generation: Option<u64>,
    },
}

/// Which generation of each peer's memory map NIXL currently holds.
///
/// A peer re-exports its metadata after every `register_host_region` and bumps
/// its generation with it, so a reader that keeps the first snapshot cannot
/// address the pools registered afterwards. The generation is the only reload
/// trigger: `get_local_md()` is not byte-stable, so a digest of the bytes would
/// reload on noise.
///
/// A newer map is imported over the one NIXL holds, never after invalidating
/// it. Invalidation also disconnects the agent's UCX endpoints, which this
/// agent's progress thread and other reads to the same peer may be using; doing
/// it under load corrupted the heap (a double free) in testing. It also failed
/// outright, on every fetch, once nixl-sys's own set of loaded agents had
/// drifted from NIXL's (NIXL invalidates internally on a remote disconnect, and
/// nixl-sys refuses to invalidate an agent it does not list).
#[derive(Default)]
struct RemoteMetadataCache {
    loaded: HashMap<String, u64>,
}

impl RemoteMetadataCache {
    /// Make NIXL hold metadata for `agent_name` covering `generation`.
    ///
    /// INVARIANT: `loaded` names a generation only while NIXL holds it. The
    /// marker is cleared before any import and written only after a successful
    /// one, so a failed refresh leaves the next load to import again instead of
    /// reporting `AlreadyCurrent` over an empty backend. A lower generation than
    /// the one loaded never walks the cache backwards.
    fn load(
        &mut self,
        importer: &impl RemoteMetadataImporter,
        agent_name: &str,
        metadata: &[u8],
        generation: u64,
    ) -> Result<RemoteMetadataLoad> {
        let previous_generation = self.loaded.get(agent_name).copied();
        if previous_generation.is_some_and(|loaded| loaded >= generation) {
            if importer.is_loaded(agent_name) {
                return Ok(RemoteMetadataLoad::AlreadyCurrent);
            }
            // The backend dropped the map behind our back; reconcile by reloading.
            tracing::warn!(
                agent_name,
                loaded_generation = previous_generation,
                "HiCache G2 peer metadata is gone from NIXL; reloading"
            );
        }
        // Nothing below may report the old generation as loaded.
        self.loaded.remove(agent_name);
        // A conflicting import erases the agent's map in NIXL, so the next load
        // imports into an empty slot.
        let loaded_name = importer.import(metadata).with_context(|| {
            format!("peer NIXL metadata for {agent_name} (generation {generation}) was rejected")
        })?;
        anyhow::ensure!(
            loaded_name == agent_name,
            "peer NIXL metadata identifies agent {loaded_name}, expected {agent_name}"
        );
        self.loaded.insert(agent_name.to_string(), generation);
        Ok(RemoteMetadataLoad::Imported {
            previous_generation,
        })
    }

    /// Make the next `load` import whatever the peer sends, whatever its
    /// generation. The map NIXL holds stays in place until then.
    fn forget(&mut self, agent_name: &str) {
        self.loaded.remove(agent_name);
    }
}

/// Destination verification failed on named pages. Typed so the fetch loop can
/// keep the verified prefix and retry only the stale pages elsewhere.
#[derive(Debug)]
struct VerificationFailure {
    source_agent: String,
    source_instance: u64,
    /// Indices into the run, ascending.
    stale_pages: Vec<usize>,
    total_pages: usize,
    /// Checksums of the pages before the first stale one, which verified and
    /// are kept; see [`fetch_runs`].
    verified_checksums: Vec<u64>,
}

impl std::fmt::Display for VerificationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "HiCache G2 read from {} (instance {}) failed verification on {} of {} pages \
             (first stale page {}); see the per-page warnings for hashes and checksums",
            self.source_agent,
            self.source_instance,
            self.stale_pages.len(),
            self.total_pages,
            self.stale_pages.first().copied().unwrap_or(0)
        )
    }
}

impl std::error::Error for VerificationFailure {}

/// The reader could not address the peer's memory: its metadata failed to load,
/// or NIXL refused to build a read against it with every local descriptor
/// inside a registration. Names no page because no page is at fault; the same
/// failure repeats for every page of that pool until the peer's map changes.
#[derive(Debug)]
struct UnaddressablePeer {
    agent_name: String,
    detail: String,
}

impl std::fmt::Display for UnaddressablePeer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "HiCache G2 cannot address peer {}: {}",
            self.agent_name, self.detail
        )
    }
}

impl std::error::Error for UnaddressablePeer {}

/// Consecutive unaddressable failures on one peer pool before the reader drops
/// its copy of the peer's map and imports it again.
const HICACHE_G2_PEER_REIMPORT_AFTER: u32 = 3;
/// Consecutive failures (counting those before the re-import) before the pool
/// is suspended. After a suspension one more failure suspends it again.
const HICACHE_G2_PEER_SUSPEND_AFTER: u32 = 6;
const HICACHE_G2_PEER_SUSPEND_INITIAL: Duration = Duration::from_secs(30);
const HICACHE_G2_PEER_SUSPEND_MAX: Duration = Duration::from_secs(600);

/// One peer process's pages of one size. Health is per pool, not per peer: a
/// peer whose second pool is missing from its exported map still serves its
/// first pool, and a success there must not reset the failing one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PeerPoolKey {
    instance_id: u64,
    page_bytes: u64,
}

#[derive(Debug, PartialEq, Eq)]
enum PeerFailureAction {
    /// Keep trying; the next candidate takes this fetch's pages.
    Retry,
    /// Drop this reader's copy of the peer's map so the next read imports it fresh.
    Reimport,
    /// Skip the pool for this long, and re-import when it comes back.
    Suspend(Duration),
}

struct PeerPoolHealth {
    consecutive_failures: u32,
    suspended_until: Option<Instant>,
    next_suspension: Duration,
}

/// Which peer pools keep failing to be addressable, and which are suspended.
///
/// Without it a peer whose exported map lacks one of its pools costs every
/// reader a lease and a failed `create_xfer_req` on every fetch that touches
/// that pool, indefinitely: the index keeps advertising its pages and nothing
/// else ever drops them.
#[derive(Default)]
struct PeerHealthTracker {
    pools: Mutex<HashMap<PeerPoolKey, PeerPoolHealth>>,
}

impl PeerHealthTracker {
    fn pools(&self) -> std::sync::MutexGuard<'_, HashMap<PeerPoolKey, PeerPoolHealth>> {
        // Every update leaves the map consistent; a panic elsewhere must not
        // turn the health check into a fetch failure.
        self.pools
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn record_failure(&self, key: PeerPoolKey, now: Instant) -> PeerFailureAction {
        let mut pools = self.pools();
        let health = pools.entry(key).or_insert(PeerPoolHealth {
            consecutive_failures: 0,
            suspended_until: None,
            next_suspension: HICACHE_G2_PEER_SUSPEND_INITIAL,
        });
        // A fetch that started before the suspension says nothing new.
        if health.suspended_until.is_some_and(|until| now < until) {
            return PeerFailureAction::Retry;
        }
        health.consecutive_failures += 1;
        if health.consecutive_failures >= HICACHE_G2_PEER_SUSPEND_AFTER {
            let suspension = health.next_suspension;
            health.suspended_until = Some(now + suspension);
            health.next_suspension = (suspension * 2).min(HICACHE_G2_PEER_SUSPEND_MAX);
            // The first read after the suspension is a probe: if it fails too,
            // suspend again at once rather than paying for five more.
            health.consecutive_failures = HICACHE_G2_PEER_SUSPEND_AFTER - 1;
            PeerFailureAction::Suspend(suspension)
        } else if health.consecutive_failures == HICACHE_G2_PEER_REIMPORT_AFTER {
            PeerFailureAction::Reimport
        } else {
            PeerFailureAction::Retry
        }
    }

    /// Forget the pool's failures. Returns how many it had, so a recovery can be logged.
    fn record_success(&self, key: PeerPoolKey) -> u32 {
        let mut pools = self.pools();
        if pools.is_empty() {
            return 0;
        }
        pools
            .remove(&key)
            .map_or(0, |health| health.consecutive_failures)
    }

    /// Whether the peer's pool of `page_bytes` pages, or with `None` any of
    /// its pools, is suspended.
    fn is_suspended(&self, instance_id: u64, page_bytes: Option<u64>, now: Instant) -> bool {
        let pools = self.pools();
        if pools.is_empty() {
            return false;
        }
        let suspended =
            |health: &PeerPoolHealth| health.suspended_until.is_some_and(|until| now < until);
        match page_bytes {
            Some(page_bytes) => pools
                .get(&PeerPoolKey {
                    instance_id,
                    page_bytes,
                })
                .is_some_and(suspended),
            None => pools
                .iter()
                .any(|(key, health)| key.instance_id == instance_id && suspended(health)),
        }
    }

    /// Drop a departed peer's pools; its instance id is never reused.
    fn forget_instance(&self, instance_id: u64) {
        self.pools().retain(|key, _| key.instance_id != instance_id);
    }
}

fn elapsed_ns(since: Instant) -> u64 {
    u64::try_from(since.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn checksum_segments(segments: &[(u64, u64)]) -> u64 {
    let mut hasher = Xxh3::new();
    for (addr, len) in segments {
        // SAFETY: inside a registration the engine keeps alive for the lifetime
        // of its host tier; a concurrent engine write only changes the digest,
        // which is exactly what the verification is there to catch.
        let bytes = unsafe { std::slice::from_raw_parts(*addr as *const u8, *len as usize) };
        hasher.update(bytes);
    }
    hasher.digest()
}

fn region_contains(base: u64, size: usize, addr: u64, len: u64) -> bool {
    addr >= base && addr.saturating_add(len) <= base + size as u64
}

/// What this replica advertised for one key, and the checksum it advertised it with.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PublishedPage {
    segments: HiCachePageSegments,
    checksum: u64,
}

/// This replica's advertised pages, indexed by key and by host slot.
///
/// A slot is identified by its full segment list, exactly as `publish`, `fetch`
/// and `invalidate_slots` describe it (see [`HiCachePageSegments::split_flat`]).
/// Each slot names at most one key, and the entry is kept in step with what the
/// slot holds:
/// - republishing a slot under another key replaces its entry;
/// - refilling it by a peer `fetch` removes its entry before any read is posted
///   into it, since a fetched page is not published;
/// - refilling it with bytes that do not come through this backend (an L3
///   load) removes its entry when the caller first calls `invalidate_slots`;
/// - `unpublish` removes the entry for a key.
///
/// HiCache reuses a host slot without telling this backend, so these are the
/// only signals that a slot's previous key is gone. Same pattern as
/// `PeerServingState::prepare` for the KVBM host tier.
#[derive(Default)]
struct PublishedIndex {
    by_key: HashMap<SequenceHash, PublishedPage>,
    by_slot: HashMap<HiCachePageSegments, SequenceHash>,
}

impl PublishedIndex {
    /// Record `key` at `page`'s slot. Returns the different key that slot held
    /// before, which is no longer served and must leave the replica index.
    fn insert(&mut self, key: SequenceHash, page: PublishedPage) -> Option<SequenceHash> {
        if let Some(previous) = self.by_key.get(&key)
            && previous.segments != page.segments
            && self.by_slot.get(&previous.segments) == Some(&key)
        {
            // The key moved to another slot; its old slot no longer names it.
            self.by_slot.remove(&previous.segments);
        }
        let evicted = self
            .by_slot
            .insert(page.segments.clone(), key)
            .filter(|old_key| *old_key != key);
        if let Some(old_key) = evicted {
            self.by_key.remove(&old_key);
        }
        self.by_key.insert(key, page);
        evicted
    }

    fn remove(&mut self, key: &SequenceHash) -> Option<PublishedPage> {
        let page = self.by_key.remove(key)?;
        if self.by_slot.get(&page.segments) == Some(key) {
            self.by_slot.remove(&page.segments);
        }
        Some(page)
    }

    fn get(&self, key: &SequenceHash) -> Option<&PublishedPage> {
        self.by_key.get(key)
    }

    /// Drop whatever key each of `slots` holds, because the slot is about to be
    /// overwritten. Returns the removed keys, which must leave the replica index.
    fn remove_slots(&mut self, slots: &[HiCachePageSegments]) -> Vec<SequenceHash> {
        let mut removed = Vec::new();
        for slot in slots {
            if let Some(key) = self.by_slot.remove(slot) {
                // `insert` keeps `by_key[key]` pointing at this slot whenever
                // `by_slot[slot] == key`, so the key goes with the slot.
                self.by_key.remove(&key);
                removed.push(key);
            }
        }
        removed
    }
}

/// Remove the keys `published` holds at `slots`, count them on `counter`, and
/// hand them to `withdraw` (the replica-index removal) when there are any.
/// The local removal finishes before `withdraw` runs and before this returns,
/// i.e. before the caller writes into the slots.
fn withdraw_slots(
    published: &Mutex<PublishedIndex>,
    slots: &[HiCachePageSegments],
    counter: &AtomicU64,
    withdraw: impl FnOnce(Vec<SequenceHash>),
) -> Result<usize> {
    let removed = published
        .lock()
        .map_err(|_| anyhow::anyhow!("HiCache G2 published map poisoned"))?
        .remove_slots(slots);
    let count = removed.len();
    if count != 0 {
        counter.fetch_add(count as u64, Ordering::Relaxed);
        withdraw(removed);
    }
    Ok(count)
}

/// Quarantine accounting for reads that outlive their deadline, in NIXL chunks
/// (the unit NIXL reports on, so the unit that can be left in flight).
#[derive(Debug, Default)]
pub struct HiCacheQuarantineCounters {
    /// Chunks currently in quarantine (gauge).
    pub quarantined_chunks: AtomicU64,
    /// Bytes currently in quarantine (gauge).
    pub quarantined_bytes: AtomicU64,
    pub quarantine_released_chunks: AtomicU64,
    pub quarantine_ns: AtomicU64,
}

impl QuarantineMetrics for Arc<HiCacheQuarantineCounters> {
    fn block_quarantined(&self, bytes: usize) {
        self.quarantined_chunks.fetch_add(1, Ordering::Relaxed);
        self.quarantined_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
    }

    fn block_released(&self, bytes: usize, quarantined_for: Duration) {
        self.quarantined_chunks.fetch_sub(1, Ordering::Relaxed);
        self.quarantined_bytes
            .fetch_sub(bytes as u64, Ordering::Relaxed);
        self.quarantine_released_chunks
            .fetch_add(1, Ordering::Relaxed);
        self.quarantine_ns.fetch_add(
            u64::try_from(quarantined_for.as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
    }
}

#[derive(Debug, Default)]
pub struct HiCacheG2Metrics {
    pub quarantine: Arc<HiCacheQuarantineCounters>,
    pub published_pages: AtomicU64,
    pub unpublished_pages: AtomicU64,
    /// Keys dropped because their host slot was republished under another key.
    pub evicted_by_republish: AtomicU64,
    /// Keys dropped because a peer fetch was about to write into their slot.
    pub invalidated_by_fetch: AtomicU64,
    /// Keys dropped by `invalidate_slots`: the engine was about to fill their
    /// slot with bytes from outside this backend (an L3 load).
    pub invalidated_by_caller: AtomicU64,
    pub lookup_hits: AtomicU64,
    pub lookup_misses: AtomicU64,
    pub fetched_pages: AtomicU64,
    pub fetched_bytes: AtomicU64,
    /// Fetched pages this replica then published itself, so peers can read
    /// them from here too. Also counted in `published_pages`.
    pub published_pulled_pages: AtomicU64,
    pub fetch_failures: AtomicU64,
    /// Fetches that returned with reads still in flight (handed to the reaper).
    pub unsettled_fetches: AtomicU64,
    /// Of those, fetches whose reads have since all reached a terminal state.
    pub settled_fetches: AtomicU64,
    pub stale_source_pages: AtomicU64,
    /// Index candidates dropped because their source is no longer a live peer
    /// (it crashed without withdrawing its advertisements).
    pub dead_source_candidates: AtomicU64,
    /// Sources (one per peer TP rank) whose entries the dead-source sweeper
    /// purged from this process's directory, and how many entries that was.
    pub purged_dead_sources: AtomicU64,
    pub purged_dead_entries: AtomicU64,
    /// Index updates this replica failed to hand to the publisher. A dropped
    /// upsert costs peers a hit; a dropped removal costs them a failed lease.
    pub dropped_publications: AtomicU64,
    pub served_pages: AtomicU64,
    /// Reloads of a peer's NIXL metadata after it grew its map. Normal while a
    /// hybrid model's pools register in sequence; steady-state growth means a
    /// peer is still registering.
    pub metadata_reloads: AtomicU64,
    /// Import attempts. In steady state exactly one per peer agent.
    pub metadata_loads: AtomicU64,
    /// Peer reads that failed because the peer's memory could not be addressed
    /// (metadata load or `create_xfer_req` failed). Zero in a healthy fleet.
    pub unaddressable_peer_failures: AtomicU64,
    /// Peer maps dropped and re-imported after repeated unaddressable failures.
    pub peer_metadata_reimports: AtomicU64,
    /// Times a peer pool was suspended, and index candidates skipped for it.
    pub peer_suspensions: AtomicU64,
    pub suspended_peer_candidates: AtomicU64,
    /// Where a fetch's wall clock goes, summed over every fetch.
    pub index_query_ns: AtomicU64,
    pub lease_rpc_ns: AtomicU64,
    pub rdma_ns: AtomicU64,
    pub verify_ns: AtomicU64,
}

impl HiCacheG2Metrics {
    pub fn snapshot(&self) -> Vec<(&'static str, u64)> {
        vec![
            (
                "quarantined_chunks",
                self.quarantine.quarantined_chunks.load(Ordering::Relaxed),
            ),
            (
                "quarantined_bytes",
                self.quarantine.quarantined_bytes.load(Ordering::Relaxed),
            ),
            (
                "quarantine_released_chunks",
                self.quarantine
                    .quarantine_released_chunks
                    .load(Ordering::Relaxed),
            ),
            (
                "quarantine_ms",
                self.quarantine.quarantine_ns.load(Ordering::Relaxed) / 1_000_000,
            ),
            (
                "published_pages",
                self.published_pages.load(Ordering::Relaxed),
            ),
            (
                "unpublished_pages",
                self.unpublished_pages.load(Ordering::Relaxed),
            ),
            (
                "evicted_by_republish",
                self.evicted_by_republish.load(Ordering::Relaxed),
            ),
            (
                "invalidated_by_fetch",
                self.invalidated_by_fetch.load(Ordering::Relaxed),
            ),
            (
                "invalidated_by_caller",
                self.invalidated_by_caller.load(Ordering::Relaxed),
            ),
            (
                "dead_source_candidates",
                self.dead_source_candidates.load(Ordering::Relaxed),
            ),
            (
                "purged_dead_sources",
                self.purged_dead_sources.load(Ordering::Relaxed),
            ),
            (
                "purged_dead_entries",
                self.purged_dead_entries.load(Ordering::Relaxed),
            ),
            (
                "dropped_publications",
                self.dropped_publications.load(Ordering::Relaxed),
            ),
            ("lookup_hits", self.lookup_hits.load(Ordering::Relaxed)),
            ("lookup_misses", self.lookup_misses.load(Ordering::Relaxed)),
            ("fetched_pages", self.fetched_pages.load(Ordering::Relaxed)),
            ("fetched_bytes", self.fetched_bytes.load(Ordering::Relaxed)),
            (
                "published_pulled_pages",
                self.published_pulled_pages.load(Ordering::Relaxed),
            ),
            (
                "fetch_failures",
                self.fetch_failures.load(Ordering::Relaxed),
            ),
            (
                "unsettled_fetches",
                self.unsettled_fetches.load(Ordering::Relaxed),
            ),
            (
                "settled_fetches",
                self.settled_fetches.load(Ordering::Relaxed),
            ),
            (
                "stale_source_pages",
                self.stale_source_pages.load(Ordering::Relaxed),
            ),
            ("served_pages", self.served_pages.load(Ordering::Relaxed)),
            (
                "metadata_reloads",
                self.metadata_reloads.load(Ordering::Relaxed),
            ),
            (
                "metadata_loads",
                self.metadata_loads.load(Ordering::Relaxed),
            ),
            (
                "unaddressable_peer_failures",
                self.unaddressable_peer_failures.load(Ordering::Relaxed),
            ),
            (
                "peer_metadata_reimports",
                self.peer_metadata_reimports.load(Ordering::Relaxed),
            ),
            (
                "peer_suspensions",
                self.peer_suspensions.load(Ordering::Relaxed),
            ),
            (
                "suspended_peer_candidates",
                self.suspended_peer_candidates.load(Ordering::Relaxed),
            ),
            (
                "index_query_ms",
                self.index_query_ns.load(Ordering::Relaxed) / 1_000_000,
            ),
            (
                "lease_rpc_ms",
                self.lease_rpc_ns.load(Ordering::Relaxed) / 1_000_000,
            ),
            ("rdma_ms", self.rdma_ns.load(Ordering::Relaxed) / 1_000_000),
            (
                "verify_ms",
                self.verify_ns.load(Ordering::Relaxed) / 1_000_000,
            ),
        ]
    }
}

#[derive(Debug, Clone)]
pub struct HiCacheG2Config {
    pub tp_rank: u32,
    pub tp_world_size: u32,
    pub page_size: u32,
    pub lease_ttl: Duration,
    /// Cap on the bytes of a single NIXL transfer request, so one prefetch of a
    /// very long prefix is chunked instead of handed to UCX whole.
    pub max_transfer_bytes: u64,
    pub verify: HiCacheVerify,
    pub hash_threads: usize,
}

/// A fetch whose reads were still in flight at its deadline. Their destination
/// pages may still be written by the NIC, so the engine must keep them out of
/// reuse until [`HiCacheG2Direct::is_settled`] reports `ticket` settled.
#[derive(Debug)]
pub struct UnsettledReads {
    pub ticket: u64,
}

impl std::fmt::Display for UnsettledReads {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "HiCache G2 reads were still in flight at the fetch deadline (settlement ticket {})",
            self.ticket
        )
    }
}

impl std::error::Error for UnsettledReads {}

/// What a fetch did: pages completed from the front, and the ticket of reads it
/// left in flight, if any.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HiCacheFetchOutcome {
    pub completed: usize,
    pub unsettled_ticket: Option<u64>,
}

/// A read the reaper keeps alive. `R` is the NIXL request; tests use a stand-in.
struct HeldRead<R = XferRequest> {
    chunk: usize,
    request: R,
}

struct HeldFetch<R> {
    ticket: u64,
    remote_agent: String,
    reads: Vec<HeldRead<R>>,
    tracker: Arc<G2pbQuarantineTracker<Arc<HiCacheQuarantineCounters>>>,
    since: Instant,
    last_report: Instant,
}

struct ReaperState<R> {
    held: Vec<HeldFetch<R>>,
    /// Tickets whose reads all reached a terminal state and that the engine has
    /// not collected yet.
    settled: HashSet<u64>,
}

/// Keeps reads that outlived their fetch alive and polls them to a terminal
/// state, so the prefetch thread returns at its deadline instead of waiting on
/// a stalled peer. The engine holds the destination pages until the ticket
/// settles. A read that never settles holds them until the process exits, as
/// KVBM's G2PB reaper does with its host blocks.
struct UnsettledReadReaper<R = XferRequest> {
    next_ticket: AtomicU64,
    state: Mutex<ReaperState<R>>,
    work: Condvar,
}

impl<R> Default for UnsettledReadReaper<R> {
    fn default() -> Self {
        Self {
            next_ticket: AtomicU64::new(0),
            state: Mutex::new(ReaperState {
                held: Vec::new(),
                settled: HashSet::new(),
            }),
            work: Condvar::new(),
        }
    }
}

impl<R> UnsettledReadReaper<R> {
    fn lock(&self) -> std::sync::MutexGuard<'_, ReaperState<R>> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn hold(
        &self,
        remote_agent: &str,
        reads: Vec<HeldRead<R>>,
        tracker: Arc<G2pbQuarantineTracker<Arc<HiCacheQuarantineCounters>>>,
    ) -> u64 {
        let ticket = self.next_ticket.fetch_add(1, Ordering::Relaxed) + 1;
        let now = Instant::now();
        self.lock().held.push(HeldFetch {
            ticket,
            remote_agent: remote_agent.to_string(),
            reads,
            tracker,
            since: now,
            last_report: now,
        });
        self.work.notify_one();
        ticket
    }

    /// Whether every read of `ticket` is terminal. `true` is reported once: the
    /// engine releases the pages and stops asking.
    fn is_settled(&self, ticket: u64) -> Result<bool> {
        let mut state = self.lock();
        if state.settled.remove(&ticket) {
            return Ok(true);
        }
        anyhow::ensure!(
            state.held.iter().any(|fetch| fetch.ticket == ticket),
            "unknown HiCache G2 settlement ticket {ticket}"
        );
        Ok(false)
    }

    /// Poll every held read once and settle the fetches whose reads are all
    /// terminal. Returns how many fetches are still held.
    fn reap(
        &self,
        metrics: &HiCacheG2Metrics,
        mut poll: impl FnMut(&R) -> Result<XferStatus>,
    ) -> usize {
        let now = Instant::now();
        let mut state = self.lock();
        let ReaperState { held, settled } = &mut *state;
        held.retain_mut(|fetch| {
            fetch.reads.retain(|read| match poll(&read.request) {
                Ok(XferStatus::InProgress) => true,
                Ok(XferStatus::Success) => {
                    fetch.tracker.mark_terminal(&[read.chunk], true);
                    false
                }
                // Terminal failure, as in `drive_to_terminal`.
                Err(_) => {
                    fetch.tracker.mark_terminal(&[read.chunk], false);
                    false
                }
            });
            if fetch.reads.is_empty() {
                settled.insert(fetch.ticket);
                metrics.settled_fetches.fetch_add(1, Ordering::Relaxed);
                tracing::warn!(
                    ticket = fetch.ticket,
                    remote_agent = %fetch.remote_agent,
                    held_ms = now.duration_since(fetch.since).as_millis(),
                    "HiCache G2 reads that outlived their fetch settled; the engine may release \
                     their destination pages"
                );
                return false;
            }
            if now.duration_since(fetch.last_report) >= HICACHE_G2_QUARANTINE_REPORT_INTERVAL {
                fetch.last_report = now;
                tracing::error!(
                    ticket = fetch.ticket,
                    remote_agent = %fetch.remote_agent,
                    reads = fetch.reads.len(),
                    held_ms = now.duration_since(fetch.since).as_millis(),
                    "HiCache G2 reads are still in flight after their fetch returned; their \
                     destination pages stay out of reuse"
                );
            }
            true
        });
        held.len()
    }

    /// Block until there is something to reap or `timeout` passes.
    fn wait_for_work(&self, timeout: Duration) {
        let state = self.lock();
        if state.held.is_empty() {
            let _ = self.work.wait_timeout(state, timeout);
        }
    }
}

/// Reaper thread body. Holds only a weak reference, so it ends with the transport.
fn run_unsettled_read_reaper(state: Weak<HiCacheG2State>) {
    loop {
        let Some(state) = state.upgrade() else {
            return;
        };
        let held = state.reaper.reap(&state.metrics, |request| {
            state
                .agent
                .get_xfer_status(request)
                .map_err(|error| anyhow::anyhow!("NIXL read status unavailable: {error}"))
        });
        if held == 0 {
            state.reaper.wait_for_work(HICACHE_G2_REAPER_IDLE_WAIT);
        } else {
            std::thread::sleep(HICACHE_G2_REAPER_POLL_INTERVAL);
        }
    }
}

/// Shared state both halves of the protocol touch. `publish` runs on the
/// engine's backup thread while `fetch` runs on its prefetch-IO thread and the
/// peer endpoint runs on the Tokio runtime, so every field is behind a lock.
struct HiCacheG2State {
    config: HiCacheG2Config,
    source: ReplicaSource,
    agent: NixlAgent,
    regions: RwLock<RegisteredRegions>,
    published: Mutex<PublishedIndex>,
    leases: PeerFetchLeaseRegistry<G2pbFetchLeaseId, PublishedPage>,
    publisher: Arc<ReplicaPublisher>,
    /// Which generation of each peer's memory map NIXL currently holds. One
    /// lock around check, import and update so two fetches cannot interleave.
    loaded_remotes: Mutex<RemoteMetadataCache>,
    /// This replica's own map generation, bumped on every region registration
    /// and sent with every fetch response.
    map_generation: AtomicU64,
    /// Peer pools this replica keeps failing to address; see [`PeerHealthTracker`].
    peer_health: PeerHealthTracker,
    /// Built once: a page batch is hashed on every publish and every fetch.
    hash_pool: rayon::ThreadPool,
    metrics: HiCacheG2Metrics,
    reaper: UnsettledReadReaper,
}

/// Outcome of building and posting one chunk.
enum PostOutcome {
    /// Nothing reached NIXL; the destination is untouched.
    NotSubmitted(anyhow::Error),
    /// NIXL completed the read synchronously.
    Completed,
    /// The read may be running. `error` is set when `post_xfer_req` failed:
    /// a failed post can still have submitted part of the request, so it is
    /// driven like any other outstanding read rather than dropped.
    Outstanding {
        request: XferRequest,
        error: Option<anyhow::Error>,
    },
}

impl HiCacheG2State {
    fn region_guard(&self) -> Result<std::sync::RwLockReadGuard<'_, RegisteredRegions>> {
        self.regions
            .read()
            .map_err(|_| anyhow::anyhow!("HiCache G2 region lock poisoned"))
    }

    /// xxh3 of every page in place, across `hash_pool`. No copy: this runs on
    /// every published and every fetched page, and a page is multiple MiB. One
    /// core hashes ~8 GiB/s against a ~58 GiB/s wire, so single-threaded hashing
    /// would make a fetch checksum-bound rather than transfer-bound.
    fn checksum_batch(&self, pages: &[HiCachePageSegments]) -> Result<Vec<u64>> {
        // Bounds-check every segment once, up front, so the hashing threads read
        // the ranges without holding the region lock.
        {
            let regions = self.region_guard()?;
            anyhow::ensure!(
                !regions.regions.is_empty(),
                "HiCache G2 has no registered host region yet"
            );
            for page in pages {
                for (addr, len) in &page.segments {
                    anyhow::ensure!(
                        regions.contains(*addr, *len),
                        "page segment [0x{addr:x}, 0x{:x}) is outside every registered host region; registered: [{}]",
                        addr + len,
                        regions
                            .regions
                            .iter()
                            .map(|region| format!("0x{:x}+{}", region.addr, region.size))
                            .collect::<Vec<_>>()
                            .join(",")
                    );
                }
            }
        }
        Ok(self.hash_pool.install(|| {
            pages
                .par_iter()
                .map(|page| checksum_segments(&page.segments))
                .collect()
        }))
    }

    fn publish(&self, keys: &[SequenceHash], pages: Vec<HiCachePageSegments>) -> Result<Vec<bool>> {
        anyhow::ensure!(
            keys.len() == pages.len(),
            "HiCache G2 publish got {} keys for {} pages",
            keys.len(),
            pages.len()
        );
        let checksums = self.checksum_batch(&pages)?;
        self.publish_checksummed(keys, pages, checksums)
    }

    /// Record and advertise pages whose checksums the caller already holds:
    /// hashed just now by `publish`, or verified by the fetch that pulled them.
    fn publish_checksummed(
        &self,
        keys: &[SequenceHash],
        pages: Vec<HiCachePageSegments>,
        checksums: Vec<u64>,
    ) -> Result<Vec<bool>> {
        anyhow::ensure!(
            keys.len() == pages.len() && keys.len() == checksums.len(),
            "HiCache G2 publish got {} keys for {} pages and {} checksums",
            keys.len(),
            pages.len(),
            checksums.len()
        );
        let mut advertised = Vec::with_capacity(keys.len());
        let mut results = Vec::with_capacity(keys.len());
        let mut entries = Vec::with_capacity(keys.len());
        for ((key, segments), checksum) in keys.iter().zip(pages).zip(checksums) {
            let size_bytes = segments.total_bytes();
            entries.push((*key, PublishedPage { segments, checksum }));
            advertised.push(ReplicaBlock {
                key: *key,
                size_bytes,
                checksum: Some(checksum),
            });
            results.push(true);
        }
        let mut evicted = Vec::new();
        {
            let mut published = self
                .published
                .lock()
                .map_err(|_| anyhow::anyhow!("HiCache G2 published map poisoned"))?;
            for (key, page) in entries {
                evicted.extend(published.insert(key, page));
            }
        }
        self.metrics
            .published_pages
            .fetch_add(keys.len() as u64, Ordering::Relaxed);
        if !evicted.is_empty() {
            self.metrics
                .evicted_by_republish
                .fetch_add(evicted.len() as u64, Ordering::Relaxed);
            // Remove before upsert: the publisher coalesces per key with the
            // last write winning, so a key that moved slots in this same batch
            // stays advertised.
            self.advertise("removal", self.publisher.remove(evicted));
        }
        // Advertise immediately, including mid-registration: every fetch
        // response carries the map generation, and a reader reloads a peer's
        // metadata the moment that generation moves.
        self.advertise("upsert", self.publisher.upsert(advertised));
        Ok(results)
    }

    /// The index is advisory: the local published map is the source of truth
    /// for serving, and a reader verifies what it reads. So, as in KVBM, a
    /// publication the publisher rejects is counted and dropped rather than
    /// failing the HiCache backup that triggered it.
    fn advertise(&self, kind: &'static str, result: Result<()>) {
        if let Err(error) = result {
            self.metrics
                .dropped_publications
                .fetch_add(1, Ordering::Relaxed);
            tracing::warn!(%error, kind, "HiCache G2 index publication was dropped");
        }
    }

    fn unpublish(&self, keys: &[SequenceHash]) -> Result<()> {
        let mut removed = Vec::with_capacity(keys.len());
        {
            let mut published = self
                .published
                .lock()
                .map_err(|_| anyhow::anyhow!("HiCache G2 published map poisoned"))?;
            // No exception for keys under an active peer lease: the lease
            // registry handed the reader its own snapshot of the descriptors
            // and answers a retry from that, so the advertisement need not
            // outlive the page. Its read verifies what it received.
            for key in keys {
                if published.remove(key).is_some() {
                    removed.push(*key);
                }
            }
        }
        if removed.is_empty() {
            return Ok(());
        }
        self.metrics
            .unpublished_pages
            .fetch_add(removed.len() as u64, Ordering::Relaxed);
        self.advertise("removal", self.publisher.remove(removed));
        Ok(())
    }

    /// Withdraw every key advertised at one of `slots`, locally and from the
    /// replica index, ahead of the slots being overwritten. Returns how many
    /// keys were withdrawn and adds that to `counter`.
    ///
    /// The local removal is synchronous, so from here on `serve_fetch` no longer
    /// leases these keys. The replica-index removal is queued to the publisher
    /// and reaches peers asynchronously; until it does, a peer can still pick
    /// this replica as a candidate, and its lease fails. A lease granted before
    /// this call keeps its snapshot (see `unpublish`), so its read can race the
    /// overwrite; destination verification rejects what it gets.
    fn invalidate_slots(
        &self,
        slots: &[HiCachePageSegments],
        counter: &AtomicU64,
    ) -> Result<usize> {
        withdraw_slots(&self.published, slots, counter, |keys| {
            self.advertise("removal", self.publisher.remove(keys))
        })
    }

    /// Answer a peer's fetch: lease the pages and hand back this agent's
    /// metadata plus the source descriptors and published checksums.
    fn serve_fetch(&self, request: HiCacheFetchRequest) -> Result<HiCacheFetchResponse> {
        let now_unix_ms = unix_time_ms()?;
        anyhow::ensure!(
            request.lease_expires_at_unix_ms > now_unix_ms,
            "HiCache G2 fetch lease {} expiry is not in the future",
            request.lease_id
        );
        let expired = self.leases.expire(now_unix_ms);
        if expired != 0 {
            tracing::info!(expired, "HiCache G2 source fetch leases expired");
        }

        let leased = {
            let published = self
                .published
                .lock()
                .map_err(|_| anyhow::anyhow!("HiCache G2 published map poisoned"))?;
            self.leases.fetch(
                request.lease_id,
                &request.keys,
                request.lease_expires_at_unix_ms,
                |key| published.get(&key).cloned().map(Arc::new),
            )
        };
        let leased = match leased {
            Ok(leased) => leased,
            Err(PeerFetchLeaseError::NotFound) => {
                anyhow::bail!("HiCache G2 source no longer holds every requested page")
            }
            Err(error) => return Err(error.into()),
        };

        let regions = self.region_guard()?;
        anyhow::ensure!(
            !regions.regions.is_empty(),
            "HiCache G2 has no registered host region yet"
        );
        self.metrics
            .served_pages
            .fetch_add(leased.len() as u64, Ordering::Relaxed);
        Ok(HiCacheFetchResponse {
            lease_id: request.lease_id,
            instance_id: self.source.key.instance_id,
            agent_name: self.agent.name(),
            agent_metadata: regions.local_metadata.clone(),
            // Read under the same guard as the metadata it describes.
            metadata_generation: self.map_generation.load(Ordering::Acquire),
            pages: leased.iter().map(|page| page.segments.clone()).collect(),
            checksums: leased.iter().map(|page| page.checksum).collect(),
        })
    }

    /// Read `source` segments into `destination` segments over NIXL. `deadline`
    /// is the fetch's single absolute deadline; see `fetch_run`.
    fn read_from_peer(
        &self,
        response: &HiCacheFetchResponse,
        keys: &[SequenceHash],
        destination: &[HiCachePageSegments],
        deadline: Instant,
    ) -> Result<u64> {
        anyhow::ensure!(
            response.checksums.len() == destination.len(),
            "HiCache G2 peer returned {} checksums for {} pages",
            response.checksums.len(),
            destination.len()
        );
        anyhow::ensure!(
            response.pages.len() == destination.len(),
            "HiCache G2 peer returned {} pages for {} destinations",
            response.pages.len(),
            destination.len()
        );
        self.load_remote_metadata(
            &response.agent_name,
            &response.agent_metadata,
            response.metadata_generation,
        )
        .map_err(|error| UnaddressablePeer {
            agent_name: response.agent_name.clone(),
            detail: format!("{error:#}"),
        })?;

        let regions = self.region_guard()?;
        anyhow::ensure!(
            !regions.regions.is_empty(),
            "HiCache G2 has no registered host region yet"
        );

        let mut pairs = Vec::new();
        for (source, target) in response.pages.iter().zip(destination) {
            anyhow::ensure!(
                source.segments.len() == target.segments.len(),
                "HiCache G2 page geometry differs: source has {} segments, destination {}",
                source.segments.len(),
                target.segments.len()
            );
            for (&(src_addr, src_len), &(dst_addr, dst_len)) in
                source.segments.iter().zip(target.segments.iter())
            {
                anyhow::ensure!(
                    src_len == dst_len,
                    "HiCache G2 segment length mismatch: source {src_len} B, destination {dst_len} B"
                );
                anyhow::ensure!(
                    regions.contains(dst_addr, dst_len),
                    "HiCache G2 destination [0x{dst_addr:x}, 0x{:x}) is outside every registered \
                     host region",
                    dst_addr + dst_len
                );
                pairs.push((src_addr, dst_addr, src_len));
            }
        }

        let mut chunks: Vec<Vec<(u64, u64, u64)>> = Vec::new();
        let mut chunk_bytes = 0u64;
        for pair in pairs {
            if chunk_bytes + pair.2 > self.config.max_transfer_bytes
                && chunks.last().is_some_and(|last| !last.is_empty())
            {
                chunks.push(Vec::new());
                chunk_bytes = 0;
            }
            if chunks.is_empty() {
                chunks.push(Vec::new());
            }
            chunk_bytes += pair.2;
            chunks
                .last_mut()
                .expect("a chunk was just ensured")
                .push(pair);
        }
        let rdma_started = Instant::now();
        let moved = self.post_all_and_wait(&response.agent_name, &chunks, deadline)?;
        self.metrics
            .rdma_ns
            .fetch_add(elapsed_ns(rdma_started), Ordering::Relaxed);

        if self.config.verify == HiCacheVerify::Destination {
            let verify_started = Instant::now();
            anyhow::ensure!(
                verify_started < deadline,
                "HiCache G2 read from {} landed after the fetch deadline; not verifying or \
                 using the pages",
                response.agent_name
            );
            // Hash what actually landed: the only check that sees a source page
            // the engine recycled between publication and the read, and it also
            // catches a corrupted transfer.
            let digests = self.checksum_batch(destination)?;
            self.metrics
                .verify_ns
                .fetch_add(elapsed_ns(verify_started), Ordering::Relaxed);
            // Name every mismatching page: which pages fail, and whether the
            // same one fails repeatedly, is what separates a source that
            // rewrote a slot from a transport that corrupted one.
            let mut stale_pages = Vec::new();
            for (index, ((got, want), target)) in digests
                .iter()
                .zip(&response.checksums)
                .zip(destination)
                .enumerate()
            {
                if got == want {
                    continue;
                }
                stale_pages.push(index);
                let (dst_addr, dst_len) = target.segments.first().copied().unwrap_or((0, 0));
                tracing::warn!(
                    page_hash = keys.get(index).map(|key| format!("{key:x}")),
                    page_index = index,
                    source_agent = %response.agent_name,
                    source_instance = response.instance_id,
                    advertised_checksum = format!("{want:x}"),
                    computed_checksum = format!("{got:x}"),
                    destination = format!("0x{dst_addr:x}"),
                    bytes = target.total_bytes(),
                    segments = target.segments.len(),
                    first_segment_bytes = dst_len,
                    metadata_generation = response.metadata_generation,
                    "HiCache G2 page failed verification after a peer read"
                );
            }
            if !stale_pages.is_empty() {
                self.metrics
                    .stale_source_pages
                    .fetch_add(stale_pages.len() as u64, Ordering::Relaxed);
                return Err(VerificationFailure {
                    source_agent: response.agent_name.clone(),
                    source_instance: response.instance_id,
                    verified_checksums: response.checksums[..stale_pages[0]].to_vec(),
                    stale_pages,
                    total_pages: destination.len(),
                }
                .into());
            }
        }
        Ok(moved)
    }

    /// Make NIXL hold the peer's current memory map before a transfer is built
    /// against it. See [`RemoteMetadataCache::load`].
    fn load_remote_metadata(
        &self,
        agent_name: &str,
        metadata: &[u8],
        generation: u64,
    ) -> Result<()> {
        let mut loaded = self
            .loaded_remotes
            .lock()
            .map_err(|_| anyhow::anyhow!("HiCache G2 remote metadata map poisoned"))?;
        match loaded.load(&self.agent, agent_name, metadata, generation)? {
            RemoteMetadataLoad::AlreadyCurrent => {}
            RemoteMetadataLoad::Imported {
                previous_generation,
            } => {
                self.metrics.metadata_loads.fetch_add(1, Ordering::Relaxed);
                if let Some(previous_generation) = previous_generation {
                    self.metrics
                        .metadata_reloads
                        .fetch_add(1, Ordering::Relaxed);
                    tracing::info!(
                        agent_name,
                        previous_generation,
                        generation,
                        metadata_bytes = metadata.len(),
                        "HiCache G2 reloaded a peer's NIXL metadata"
                    );
                }
            }
        }
        Ok(())
    }

    /// Act on a peer pool that could not be addressed: count it, and after
    /// repeated failures re-import the peer's map or suspend the pool.
    fn record_unaddressable(&self, key: PeerPoolKey, failure: &UnaddressablePeer) {
        self.metrics
            .unaddressable_peer_failures
            .fetch_add(1, Ordering::Relaxed);
        let action = self.peer_health.record_failure(key, Instant::now());
        let suspend_secs = match action {
            PeerFailureAction::Retry => return,
            PeerFailureAction::Reimport => {
                self.metrics
                    .peer_metadata_reimports
                    .fetch_add(1, Ordering::Relaxed);
                None
            }
            PeerFailureAction::Suspend(suspension) => {
                self.metrics
                    .peer_suspensions
                    .fetch_add(1, Ordering::Relaxed);
                Some(suspension.as_secs())
            }
        };
        // Either way the next read imports whatever the peer sends now, merged
        // over the map NIXL holds, so a map that went stale on this side does
        // not outlive the suspension.
        if let Ok(mut loaded) = self.loaded_remotes.lock() {
            loaded.forget(&failure.agent_name);
        }
        tracing::warn!(
            instance_id = key.instance_id,
            page_bytes = key.page_bytes,
            agent_name = %failure.agent_name,
            suspend_secs,
            error = %failure.detail,
            "HiCache G2 peer pool is repeatedly unaddressable; {}",
            if suspend_secs.is_some() {
                "suspending it; it re-imports its NIXL metadata when it comes back"
            } else {
                "re-importing its NIXL metadata on the next read"
            }
        );
    }

    fn record_peer_success(&self, key: PeerPoolKey) {
        let failures = self.peer_health.record_success(key);
        if failures != 0 {
            tracing::info!(
                instance_id = key.instance_id,
                page_bytes = key.page_bytes,
                failures,
                "HiCache G2 peer pool is addressable again"
            );
        }
    }

    /// Describe a failed `create_xfer_req` in terms of addresses and registration.
    ///
    /// Bounded to a few descriptors: a chunk can carry hundreds, and this runs
    /// on a path that is already failing rather than on the hot path.
    fn log_xfer_req_failure(
        &self,
        remote_agent: &str,
        chunk: &[(u64, u64, u64)],
        error: &impl std::fmt::Display,
    ) -> Option<usize> {
        const MAX_REPORTED: usize = 4;
        let regions = match self.region_guard() {
            Ok(regions) => regions,
            Err(guard_error) => {
                tracing::warn!(%error, %guard_error, remote_agent,
                    "HiCache G2 create_xfer_req failed and the region lock is poisoned");
                return None;
            }
        };
        let uncontained_local = chunk
            .iter()
            .filter(|(_, dst, len)| !regions.contains(*dst, *len))
            .count();
        let sample: Vec<String> = chunk
            .iter()
            .take(MAX_REPORTED)
            .map(|(src, dst, len)| {
                format!(
                    "src=0x{src:x} dst=0x{dst:x} len={len} dst_registered={}",
                    regions.contains(*dst, *len)
                )
            })
            .collect();
        let registered: Vec<String> = regions
            .regions
            .iter()
            .map(|region| format!("0x{:x}+{}", region.addr, region.size))
            .collect();
        tracing::warn!(
            %error,
            remote_agent,
            descriptors = chunk.len(),
            uncontained_local,
            generation = self.map_generation.load(Ordering::Acquire),
            sample = %sample.join(" | "),
            local_regions = %registered.join(","),
            "HiCache G2 create_xfer_req failed: descriptor/registration detail"
        );
        Some(uncontained_local)
    }

    /// Build and post one chunk. `Ok(None)` means NIXL completed it synchronously.
    fn post_chunk(&self, remote_agent: &str, chunk: &[(u64, u64, u64)]) -> PostOutcome {
        let lists = (|| -> Result<(XferDescList, XferDescList)> {
            let mut local = XferDescList::new(MemType::Dram).map_err(|error| {
                anyhow::anyhow!("failed to create local descriptor list: {error}")
            })?;
            let mut remote = XferDescList::new(MemType::Dram).map_err(|error| {
                anyhow::anyhow!("failed to create remote descriptor list: {error}")
            })?;
            for (src_addr, dst_addr, len) in chunk {
                local.add_desc(*dst_addr as usize, *len as usize, 0);
                remote.add_desc(*src_addr as usize, *len as usize, 0);
            }
            Ok((local, remote))
        })();
        let (local, remote) = match lists {
            Ok(lists) => lists,
            Err(error) => return PostOutcome::NotSubmitted(error),
        };
        let request =
            match self
                .agent
                .create_xfer_req(XferOp::Read, &local, &remote, remote_agent, None)
            {
                Ok(request) => request,
                Err(error) => {
                    // A create failure names neither descriptor; say which address
                    // and, for the local side, whether it is inside a registration.
                    let uncontained_local = self.log_xfer_req_failure(remote_agent, chunk, &error);
                    let detail = format!("failed to create NIXL read request: {error}");
                    // With every local descriptor registered, what NIXL could not
                    // resolve is the peer's side: its map, as loaded here, does
                    // not cover the source addresses it just leased to us.
                    return PostOutcome::NotSubmitted(if uncontained_local == Some(0) {
                        UnaddressablePeer {
                            agent_name: remote_agent.to_string(),
                            detail,
                        }
                        .into()
                    } else {
                        anyhow::anyhow!(detail)
                    });
                }
            };
        match self.agent.post_xfer_req(&request, None) {
            Ok(true) => PostOutcome::Outstanding {
                request,
                error: None,
            },
            Ok(false) => PostOutcome::Completed,
            Err(error) => PostOutcome::Outstanding {
                request,
                error: Some(anyhow::anyhow!("failed to post NIXL read request: {error}")),
            },
        }
    }

    fn post_all_and_wait(
        &self,
        remote_agent: &str,
        chunks: &[Vec<(u64, u64, u64)>],
        deadline: Instant,
    ) -> Result<u64> {
        let chunks: Vec<&[(u64, u64, u64)]> = chunks
            .iter()
            .filter(|chunk| !chunk.is_empty())
            .map(Vec::as_slice)
            .collect();
        let chunk_bytes: Vec<usize> = chunks
            .iter()
            .map(|chunk| chunk.iter().map(|(_, _, len)| *len as usize).sum())
            .collect();
        let bytes = chunk_bytes.iter().map(|len| *len as u64).sum();

        // One "block" per chunk with a single writer: a chunk is the unit NIXL
        // reports on, so it is the unit that can be left in flight.
        let tracker = Arc::new(G2pbQuarantineTracker::new(
            1,
            chunk_bytes,
            Arc::clone(&self.metrics.quarantine),
        ));

        let mut requests: Vec<Option<XferRequest>> = Vec::with_capacity(chunks.len());
        let mut in_flight = Vec::with_capacity(chunks.len());
        let mut posting_error = None;
        for (index, chunk) in chunks.iter().enumerate() {
            match self.post_chunk(remote_agent, chunk) {
                PostOutcome::Outstanding { request, error } => {
                    tracker.mark_dispatched(&[index]);
                    requests.push(Some(request));
                    in_flight.push(index);
                    if let Some(error) = error {
                        // Part of the request may have been submitted; stop
                        // posting, but drive this one like the others.
                        posting_error = Some(error);
                        let unposted: Vec<usize> = (index + 1..chunks.len()).collect();
                        tracker.mark_not_dispatched(&unposted);
                        break;
                    }
                }
                PostOutcome::Completed => {
                    tracker.mark_dispatched(&[index]);
                    tracker.mark_terminal(&[index], true);
                    requests.push(None);
                }
                PostOutcome::NotSubmitted(error) => {
                    let unposted: Vec<usize> = (index..chunks.len()).collect();
                    tracker.mark_not_dispatched(&unposted);
                    posting_error = Some(error);
                    break;
                }
            }
        }

        let DrainOutcome { unsettled, result } =
            drive_to_terminal(&tracker, in_flight, deadline, remote_agent, |index| {
                let request = requests[index]
                    .as_ref()
                    .expect("an in-flight index always has a posted request");
                self.agent
                    .get_xfer_status(request)
                    .map_err(|error| anyhow::anyhow!("NIXL read status unavailable: {error}"))
            });
        if !unsettled.is_empty() {
            // Dropping these requests would not stop the NIC writing into the
            // destination; the reaper keeps them and the engine keeps the pages.
            let reads = unsettled
                .into_iter()
                .map(|chunk| HeldRead {
                    chunk,
                    request: requests[chunk]
                        .take()
                        .expect("an unsettled index always has a posted request"),
                })
                .collect();
            let ticket = self.reaper.hold(remote_agent, reads, tracker);
            self.metrics
                .unsettled_fetches
                .fetch_add(1, Ordering::Relaxed);
            tracing::warn!(
                ticket,
                remote_agent,
                read_error = result.err().map(|error| format!("{error:#}")),
                posting_error = posting_error.map(|error| format!("{error:#}")),
                "HiCache G2 fetch returned with reads still in flight"
            );
            return Err(UnsettledReads { ticket }.into());
        }
        drop(requests);
        if let Some(error) = posting_error {
            return Err(error);
        }
        result.map(|()| bytes)
    }
}

/// What `drive_to_terminal` settled and what it left in flight.
struct DrainOutcome {
    /// Chunks still in progress when the deadline passed. The caller hands them
    /// to the reaper; their destination pages must not be reused until they settle.
    unsettled: Vec<usize>,
    /// `Err` if any read failed or the deadline passed.
    result: Result<()>,
}

/// Poll every in-flight read until each is terminal or the fetch deadline passes.
///
/// `Ok(Success)` is a completed read. An `Err` from the status call is a failed
/// one: UCX completes an RMA request with an error only after its endpoint has
/// failed, which is how KVBM's G2PB loads read it too, so the chunk is terminal
/// and its destination can go back to the engine.
///
/// A read still in progress at the deadline is quarantined in `tracker` and
/// returned in `unsettled` rather than waited on: a stalled peer must not stall
/// the engine's prefetch thread and, through the TP prefetch sync, its group.
fn drive_to_terminal(
    tracker: &G2pbQuarantineTracker<Arc<HiCacheQuarantineCounters>>,
    mut in_flight: Vec<usize>,
    deadline: Instant,
    remote_agent: &str,
    mut poll: impl FnMut(usize) -> Result<XferStatus>,
) -> DrainOutcome {
    let mut first_error: Option<anyhow::Error> = None;
    let mut polls: u32 = 0;
    while !in_flight.is_empty() {
        in_flight.retain(|&index| match poll(index) {
            Ok(XferStatus::InProgress) => true,
            Ok(XferStatus::Success) => {
                tracker.mark_terminal(&[index], true);
                false
            }
            Err(error) => {
                tracker.mark_terminal(&[index], false);
                first_error.get_or_insert(error);
                false
            }
        });
        if in_flight.is_empty() {
            break;
        }
        // Instant::now() costs more than a status call; check the clock only
        // occasionally while the read is healthy.
        polls = polls.wrapping_add(1);
        if !polls.is_multiple_of(HICACHE_G2_DEADLINE_CHECK_POLLS) {
            std::hint::spin_loop();
            continue;
        }
        if Instant::now() >= deadline {
            tracker.mark_deadline_elapsed();
            tracing::warn!(
                remote_agent,
                in_flight = in_flight.len(),
                "HiCache G2 read passed its deadline; handing it to the reaper, and its \
                 destination pages stay out of reuse until it settles"
            );
            break;
        }
    }
    let result = if let Some(error) = first_error {
        Err(error.context(format!("NIXL read from {remote_agent} failed")))
    } else if !in_flight.is_empty() {
        Err(anyhow::anyhow!(
            "NIXL read from {remote_agent} did not complete before the fetch deadline"
        ))
    } else {
        Ok(())
    };
    DrainOutcome {
        unsettled: in_flight,
        result,
    }
}

/// Request-plane client for the HiCache peer endpoint.
#[derive(Clone)]
struct HiCachePeerClient {
    router: PushRouter<HiCachePeerRequest, Annotated<HiCachePeerResponse>>,
}

impl HiCachePeerClient {
    async fn new(component: Component) -> Result<Self> {
        let client = component
            .endpoint(HICACHE_G2_PEER_ENDPOINT)
            .client()
            .await?;
        // Peers register asynchronously; a client built before any source has a
        // host tier must not fail, it must simply find no instances yet.
        let router = PushRouter::from_client_no_fault_detection(client, RouterMode::Direct).await?;
        Ok(Self { router })
    }

    fn instance_ids(&self) -> Vec<u64> {
        self.router.client.instance_ids()
    }

    /// Peer instance ids, updated as discovery adds and removes them.
    fn membership(&self) -> tokio::sync::watch::Receiver<Vec<u64>> {
        self.router.client.instance_avail_watcher()
    }

    fn is_live(&self, instance_id: u64) -> bool {
        self.router.client.is_instance_live(instance_id)
    }

    async fn request(
        &self,
        instance_id: u64,
        request: HiCachePeerRequest,
    ) -> Result<HiCachePeerResponse> {
        let mut stream = self
            .router
            .direct(SingleIn::new(request), instance_id)
            .await?;
        let response = stream.next().await.ok_or_else(|| {
            anyhow::anyhow!("HiCache G2 request to instance {instance_id} returned no response")
        })?;
        response.into_result()?.ok_or_else(|| {
            anyhow::anyhow!(
                "HiCache G2 request to instance {instance_id} returned an empty response"
            )
        })
    }

    async fn fetch(
        &self,
        instance_id: u64,
        request: HiCacheFetchRequest,
    ) -> Result<HiCacheFetchResponse> {
        let lease_id = request.lease_id;
        match self
            .request(instance_id, HiCachePeerRequest::Fetch(request))
            .await?
        {
            HiCachePeerResponse::Fetch(response) => {
                anyhow::ensure!(
                    response.lease_id == lease_id,
                    "HiCache G2 fetch response lease {} did not match request lease {lease_id}",
                    response.lease_id
                );
                Ok(response)
            }
            other => anyhow::bail!("unexpected HiCache G2 fetch response: {other:?}"),
        }
    }

    async fn release(&self, instance_id: u64, lease_id: G2pbFetchLeaseId) -> Result<()> {
        match self
            .request(
                instance_id,
                HiCachePeerRequest::Release(HiCacheReleaseRequest { lease_id }),
            )
            .await?
        {
            HiCachePeerResponse::Ack => Ok(()),
            other => anyhow::bail!("unexpected HiCache G2 release response: {other:?}"),
        }
    }
}

/// How long a peer must stay out of discovery before its directory entries are
/// purged. Discovery can briefly drop a live instance (a lease renewal hiccup),
/// and a purged instance is ignored for good, so this errs long.
const DEPARTED_PEER_GRACE: Duration = Duration::from_secs(30);

/// Purge the directory entries of peers that leave discovery. Lookups already
/// skip them (`remote_live_candidates`); this frees them, so a scale-down or
/// crash does not leave its advertisements in every surviving process for
/// good. Event-driven: nothing runs while membership is stable, and each
/// departure gets one delayed re-check. Stops when the transport is dropped.
fn spawn_departed_peer_purger(
    index: Weak<ReplicaIndex>,
    peers: HiCachePeerClient,
    state: Weak<HiCacheG2State>,
) {
    tokio::spawn(async move {
        let mut membership = peers.membership();
        let mut known = membership
            .borrow_and_update()
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        while membership.changed().await.is_ok() {
            let current = membership
                .borrow_and_update()
                .iter()
                .copied()
                .collect::<HashSet<_>>();
            for departed in known.difference(&current).copied() {
                let (index, peers, state) = (index.clone(), peers.clone(), state.clone());
                tokio::spawn(async move {
                    tokio::time::sleep(DEPARTED_PEER_GRACE).await;
                    if peers.is_live(departed) {
                        return;
                    }
                    let (Some(index), Some(state)) = (index.upgrade(), state.upgrade()) else {
                        return;
                    };
                    if departed == state.source.key.instance_id {
                        return;
                    }
                    state.peer_health.forget_instance(departed);
                    // A departed peer can own hundreds of thousands of entries;
                    // keep the purge off the async workers.
                    let Ok((sources, entries)) =
                        tokio::task::spawn_blocking(move || index.purge_instance(departed)).await
                    else {
                        return;
                    };
                    state
                        .metrics
                        .purged_dead_sources
                        .fetch_add(sources as u64, Ordering::Relaxed);
                    state
                        .metrics
                        .purged_dead_entries
                        .fetch_add(entries as u64, Ordering::Relaxed);
                    tracing::info!(
                        instance_id = departed,
                        sources,
                        entries,
                        "purged HiCache G2 directory entries of a peer that left discovery"
                    );
                });
            }
            known = current;
            if index.strong_count() == 0 {
                break;
            }
        }
    });
}

struct HiCachePeerEngine {
    state: Arc<HiCacheG2State>,
}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<HiCachePeerRequest>,
        ManyOut<Annotated<HiCachePeerResponse>>,
        anyhow::Error,
    > for HiCachePeerEngine
{
    async fn generate(
        &self,
        request: SingleIn<HiCachePeerRequest>,
    ) -> Result<ManyOut<Annotated<HiCachePeerResponse>>> {
        let (request, context) = request.into_parts();
        let response = match request {
            HiCachePeerRequest::Fetch(fetch) => match self.state.serve_fetch(fetch) {
                Ok(response) => Annotated::from_data(HiCachePeerResponse::Fetch(response)),
                Err(error) => Annotated::from_error(error.to_string()),
            },
            HiCachePeerRequest::Release(release) => {
                self.state.leases.release(release.lease_id);
                Annotated::from_data(HiCachePeerResponse::Ack)
            }
        };
        Ok(ResponseStream::new(
            Box::pin(stream::iter(vec![response])),
            context.context(),
        ))
    }
}

/// The engine-facing handle: publish my pages, look up a peer's, read them.
pub struct HiCacheG2Direct {
    state: Arc<HiCacheG2State>,
    /// This process's own copy of the directory (fed the same NATS deltas as
    /// every other replica's), so a lookup needs no peer to be reachable.
    index: Arc<ReplicaIndex>,
    peers: HiCachePeerClient,
    runtime: tokio::runtime::Handle,
    _distributed_runtime: Arc<DistributedRuntime>,
}

impl HiCacheG2Direct {
    /// Join the serving-G2 namespace, start the index and peer endpoints, and
    /// build the NIXL agent. The host region is registered separately, once the
    /// engine has allocated its host tier.
    pub async fn new(
        distributed_runtime: Arc<DistributedRuntime>,
        config: HiCacheG2Config,
        ucx_devices: Option<String>,
    ) -> Result<Self> {
        let namespace = serving_g2_namespace_from_env(&distributed_runtime)?;
        let index_component = namespace.component(SERVING_G2_INDEX_COMPONENT)?;
        let peer_component = namespace.component(SERVING_G2_PEER_COMPONENT)?;

        let instance_id = distributed_runtime.connection_id();
        let source = ReplicaSource {
            key: ReplicaSourceKey {
                instance_id,
                tp_rank: config.tp_rank,
            },
            replica_id: Some(g2pb_replica_id_from_env(instance_id, config.tp_world_size)?),
            tp_world_size: config.tp_world_size,
            incarnation: uuid::Uuid::new_v4(),
            block_size: config.page_size,
            row: baseten_row_from_env()?,
        };

        let agent_name = hicache_g2_agent_name(instance_id, config.tp_rank);
        // The progress-thread cadence is a real throughput knob for this path;
        // g2pb's 1 ms default is tuned for a TRT decode worker that must not
        // burn a core while idle. See HICACHE_G2_PROGRESS_DELAY_US_ENV.
        let agent = new_hicache_g2_nixl_agent(&agent_name, hicache_g2_progress_delay_us()?)?;
        // Same UCX profile as a KVBM worker's G2G path.
        create_worker_ucx_backend(&agent, true, ucx_devices.as_deref())?;

        let publisher = ReplicaPublisher::new(&index_component, source.clone()).await?;
        let hash_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.hash_threads)
            .thread_name(|index| format!("hicache-g2-hash-{index}"))
            .build()
            .context("failed to build the HiCache G2 hashing pool")?;
        let state = Arc::new(HiCacheG2State {
            config,
            source,
            agent,
            regions: RwLock::new(RegisteredRegions::default()),
            published: Mutex::new(PublishedIndex::default()),
            leases: PeerFetchLeaseRegistry::new(
                HICACHE_G2_MAX_ACTIVE_LEASES,
                HICACHE_G2_MAX_RELEASED_LEASES,
            )?,
            publisher,
            loaded_remotes: Mutex::new(RemoteMetadataCache::default()),
            map_generation: AtomicU64::new(0),
            peer_health: PeerHealthTracker::default(),
            hash_pool,
            metrics: HiCacheG2Metrics::default(),
            reaper: UnsettledReadReaper::default(),
        });
        let reaper_state = Arc::downgrade(&state);
        std::thread::Builder::new()
            .name("hicache-g2-reaper".to_string())
            .spawn(move || run_unsettled_read_reaper(reaper_state))
            .context("failed to start the HiCache G2 unsettled-read reaper")?;

        // Every replica runs its own copy of the directory: both are fed the
        // same NATS deltas, so no separate indexer process is needed, and
        // lookups read the local copy rather than round-robin over replicas
        // that may be gone.
        let index = start_replica_index_service(&index_component).await?;
        start_peer_service(&peer_component, Arc::clone(&state))?;

        let peers = HiCachePeerClient::new(peer_component).await?;
        // Readers' logs name a peer only by its agent; this line maps it back to
        // a pod, which otherwise takes guesswork once early logs have rotated.
        tracing::info!(
            instance_id,
            agent_name = %agent_name,
            tp_rank = state.config.tp_rank,
            tp_world_size = state.config.tp_world_size,
            replica_id = state.source.replica_id(),
            host = std::env::var("HOSTNAME").ok(),
            "HiCache G2 started"
        );
        spawn_departed_peer_purger(
            Arc::downgrade(&index),
            peers.clone(),
            Arc::downgrade(&state),
        );

        Ok(Self {
            state,
            index,
            peers,
            runtime: tokio::runtime::Handle::current(),
            _distributed_runtime: distributed_runtime,
        })
    }

    pub fn metrics(&self) -> &HiCacheG2Metrics {
        &self.state.metrics
    }

    pub fn instance_id(&self) -> u64 {
        self.state.source.key.instance_id
    }

    /// Register one engine host pool with NIXL, in stride-aligned chunks.
    /// Idempotent for the same window; an overlapping different window is
    /// rejected. A hybrid model registers several pools (MLA KV plus one region
    /// per Mamba/KDA state buffer).
    ///
    /// `page_bytes` must be THIS region's per-page stride, not the pool's summed
    /// page size: a multi-slot pool splits each page across regions, the chunk
    /// grid is built from `page_bytes`, and a page that straddles two memory
    /// registrations fails every `create_xfer_req` that touches it. A region is
    /// always a whole number of pages, so divisibility is an exact check.
    ///
    /// Chunked because one `ibv_reg_mr` over a TB-scale pool holds the process
    /// `mmap_lock` for long enough that kubelet's `/proc/<pid>/smaps` reads
    /// block and the node stops heartbeating; see
    /// `dynamo_memory::REGISTRATION_CHUNK_TARGET_BYTES`.
    pub fn register_host_region(&self, addr: u64, size: usize, page_bytes: usize) -> Result<()> {
        anyhow::ensure!(size > 0, "host region size must be positive");
        anyhow::ensure!(
            page_bytes > 0 && size.is_multiple_of(page_bytes),
            "HiCache G2 host region [0x{addr:x}, +{size}) is not a whole number of \
             {page_bytes}-byte pages (remainder {}); page_bytes must be THIS region's \
             per-page stride, not the pool's summed page size",
            size % page_bytes.max(1)
        );
        let mut regions = self
            .state
            .regions
            .write()
            .map_err(|_| anyhow::anyhow!("HiCache G2 region lock poisoned"))?;
        if regions.find(addr, size).is_some() {
            return Ok(());
        }
        anyhow::ensure!(
            !regions.contains(addr, size as u64),
            "HiCache G2 host region [0x{addr:x}, +{size}) overlaps one already registered"
        );
        let chunk_bytes = dynamo_memory::registration_chunk_bytes(page_bytes);
        let chunks: Vec<(usize, usize)> =
            dynamo_memory::registration_chunks(size, chunk_bytes).collect();
        // Sequential on purpose: the point is to release mmap_lock between
        // chunks, and registering chunks concurrently from the same process
        // would hold it continuously again. UCX_IB_REG_MT_THRESH is set so NIXL
        // does not re-introduce threads underneath this.
        let mut handles = Vec::with_capacity(chunks.len());
        let registration_start = Instant::now();
        for (index, (offset, len)) in chunks.iter().copied().enumerate() {
            let descriptor = NixlDescriptor {
                addr: addr + offset as u64,
                size: len,
                mem_type: MemType::Dram,
                device_id: 0,
            };
            let chunk_start = Instant::now();
            let handle = self.state.agent.register_memory(&descriptor, None).map_err(|error| {
                anyhow::anyhow!(
                    "failed to register host region chunk {index}/{} [0x{:x}, +{len}) with NIXL: {error}",
                    chunks.len(),
                    addr + offset as u64,
                )
            })?;
            handles.push(handle);
            tracing::info!(
                chunk = index + 1,
                chunks = chunks.len(),
                addr = format!("0x{:x}", addr + offset as u64),
                bytes = len,
                took_ms = chunk_start.elapsed().as_millis(),
                "HiCache G2 registered a host pool chunk"
            );
        }
        let registration_ms = registration_start.elapsed().as_millis();
        regions.regions.push(RegisteredRegion {
            addr,
            size,
            _handles: handles,
        });
        // Metadata has to be re-exported after every registration, or peers
        // cannot address the pools registered after the first one. The
        // generation moves with it, under the same write lock, so a peer can
        // never observe a generation that is ahead of the metadata it names.
        let metadata =
            self.state.agent.get_local_md().map_err(|error| {
                anyhow::anyhow!("failed to export local NIXL metadata: {error}")
            })?;
        let freeze = std::env::var(HICACHE_G2_TEST_FREEZE_METADATA_RANK_ENV)
            .ok()
            .and_then(|rank| rank.trim().parse::<u32>().ok())
            == Some(self.state.config.tp_rank);
        if freeze && !regions.local_metadata.is_empty() {
            tracing::warn!(
                tp_rank = self.state.config.tp_rank,
                "{HICACHE_G2_TEST_FREEZE_METADATA_RANK_ENV} is set: keeping the previous NIXL \
                 metadata, so peers cannot address this pool (test only)"
            );
        } else {
            regions.local_metadata = metadata;
        }
        // One bump per pool, after all of its chunks: every peer reloads
        // metadata on a generation increase, so per-chunk bumps would multiply
        // reloads across the fleet.
        let generation = self.state.map_generation.fetch_add(1, Ordering::AcqRel) + 1;
        tracing::info!(
            addr = format!("0x{addr:x}"),
            size,
            page_bytes,
            chunk_bytes,
            chunks = chunks.len(),
            registration_ms,
            regions = regions.regions.len(),
            metadata_bytes = regions.local_metadata.len(),
            generation,
            "HiCache G2 registered an engine host pool"
        );
        Ok(())
    }

    pub fn publish(
        &self,
        keys: &[SequenceHash],
        pages: Vec<HiCachePageSegments>,
    ) -> Result<Vec<bool>> {
        self.state.publish(keys, pages)
    }

    pub fn unpublish(&self, keys: &[SequenceHash]) -> Result<()> {
        self.state.unpublish(keys)
    }

    /// Withdraw whatever this replica advertises at the given host slots, before
    /// the engine overwrites them with bytes that do not come through this
    /// backend (an L3 load). Slots are described as for `publish`/`fetch`: a
    /// flat `(ptrs, sizes)` pair with `ptrs_per_page` segments per page. Returns
    /// the number of index entries removed. `fetch` does this itself for its
    /// destination pages.
    pub fn invalidate_slots(
        &self,
        ptrs: &[u64],
        sizes: &[u64],
        ptrs_per_page: usize,
    ) -> Result<usize> {
        let slots = HiCachePageSegments::split_flat(ptrs, sizes, ptrs_per_page)?;
        self.state
            .invalidate_slots(&slots, &self.state.metrics.invalidated_by_caller)
    }

    /// Which of `keys` some *other* replica advertises, key by key.
    ///
    /// The hybrid pools need this rather than a prefix: a `TRAILING_PAGES` pool
    /// (Mamba/KDA state) only holds a window ending on an offloaded node
    /// boundary, so page 5 can be restorable while 3 and 4 are not.
    pub fn lookup_mask(&self, keys: &[SequenceHash]) -> Result<Vec<bool>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let locations = self.query_locations(keys, None);
        let mask = locations
            .iter()
            .map(|candidates| !candidates.is_empty())
            .collect::<Vec<_>>();
        if mask.iter().any(|hit| *hit) {
            self.state
                .metrics
                .lookup_hits
                .fetch_add(1, Ordering::Relaxed);
        } else {
            self.state
                .metrics
                .lookup_misses
                .fetch_add(1, Ordering::Relaxed);
        }
        Ok(mask)
    }

    /// Longest prefix of `keys` that some *other* replica advertises.
    pub fn lookup_prefix(&self, keys: &[SequenceHash]) -> Result<usize> {
        let locations = self.query_locations(keys, None);
        let mut prefix = 0;
        for candidates in &locations {
            if candidates.is_empty() {
                break;
            }
            prefix += 1;
        }
        if prefix == 0 {
            self.state
                .metrics
                .lookup_misses
                .fetch_add(1, Ordering::Relaxed);
        } else {
            self.state
                .metrics
                .lookup_hits
                .fetch_add(1, Ordering::Relaxed);
        }
        Ok(prefix)
    }

    /// Read the pages for `keys` into `destination`, in order, from the front.
    /// Returns the number of leading pages completed and, when reads outlived
    /// the deadline, the ticket the engine must hold the rest of `destination`
    /// under until [`Self::is_settled`]; see [`fetch_runs`].
    ///
    /// ONE absolute deadline, `lease_ttl` from entry, bounds the whole call: the
    /// index lookup, every peer run including retries on other candidates, each
    /// lease RPC, the RDMA reads and verification. No step starts
    /// once it has passed. Reads still in flight at the deadline are left to the
    /// reaper rather than waited on (see `drive_to_terminal`).
    ///
    /// Errs only on malformed arguments, before any read is posted; a peer or
    /// transport failure shortens `completed` instead. The engine relies on
    /// this to treat an error as "nothing was written".
    ///
    /// The completed pages are published from this replica before it returns;
    /// see [`Self::publish_pulled`].
    pub fn fetch(
        &self,
        keys: &[SequenceHash],
        destination: &[HiCachePageSegments],
    ) -> Result<HiCacheFetchOutcome> {
        anyhow::ensure!(
            keys.len() == destination.len(),
            "HiCache G2 fetch got {} keys for {} destinations",
            keys.len(),
            destination.len()
        );
        if keys.is_empty() {
            return Ok(HiCacheFetchOutcome {
                completed: 0,
                unsettled_ticket: None,
            });
        }
        // The slot's previous key must go before any read is posted, whether or
        // not the fetch then completes: HiCache has handed these slots over for
        // new tokens either way. A pulled page is published only after its bytes
        // land and verify (publish_pulled).
        self.state
            .invalidate_slots(destination, &self.state.metrics.invalidated_by_fetch)?;
        let deadline = Instant::now() + self.state.config.lease_ttl;
        // Every page of one fetch belongs to the same engine pool.
        let page_bytes = destination[0]
            .segments
            .iter()
            .map(|(_, len)| *len)
            .sum::<u64>();
        let mut locations = self.query_locations(keys, Some(page_bytes));
        let FetchedRuns { outcome, checksums } =
            fetch_runs(&mut locations, deadline, |instance_id, range| {
                let pool = PeerPoolKey {
                    instance_id,
                    page_bytes,
                };
                let result = self.fetch_run(
                    instance_id,
                    &keys[range.clone()],
                    &destination[range],
                    deadline,
                );
                match &result {
                    Ok(_) => self.state.record_peer_success(pool),
                    Err(error) => {
                        self.state
                            .metrics
                            .fetch_failures
                            .fetch_add(1, Ordering::Relaxed);
                        if let Some(failure) = error.downcast_ref::<UnaddressablePeer>() {
                            self.state.record_unaddressable(pool, failure);
                        }
                    }
                }
                result
            });
        self.publish_pulled(
            &keys[..outcome.completed],
            &destination[..outcome.completed],
            checksums,
        );
        Ok(outcome)
    }

    /// Advertise the pages a fetch pulled, so peers can read them from this
    /// replica too and not only from the replica that computed them.
    ///
    /// Only the completed prefix is passed in: every page in it landed and, with
    /// destination verification, matched the checksum it was published with,
    /// which is reused here instead of hashing the page again. A page that
    /// failed or failed verification ends the prefix (or is re-read from
    /// another candidate first). Pages under an unsettled ticket are past the
    /// prefix and are never published, not even once the ticket settles: the
    /// fetch reported them as misses, so the engine frees them on settlement,
    /// and the reaper never verified them.
    ///
    /// Publishing before `fetch` returns is what makes this sound. Until the
    /// engine sees the result, nothing can free or rewrite these slots, so the
    /// entry describes bytes that are there. The engine may free a fetched page
    /// right afterwards (SGLang drops a prefetched page that duplicates a tree
    /// node, cut back to the prefix every TP rank fetched, or of an aborted
    /// request, and buffer mode frees its bounce pages after loading them), so
    /// the entry can outlive the page, exactly as for a backed-up page the host
    /// tier evicts. That heals: the slot is next filled either from the GPU,
    /// whose backup republishes the slot and replaces the entry, or by a fetch
    /// or L3 load, which withdraws the slot's entry before writing. A peer that
    /// reads the slot in between receives bytes that fail verification against
    /// the advertised checksum: a miss, never wrong tokens.
    ///
    /// Like `fetch`'s other failures, a failed publication only costs peers a
    /// hit; it is logged and the fetch still reports its pages.
    fn publish_pulled(
        &self,
        keys: &[SequenceHash],
        destination: &[HiCachePageSegments],
        checksums: Vec<u64>,
    ) {
        if keys.is_empty() {
            return;
        }
        match self
            .state
            .publish_checksummed(keys, destination.to_vec(), checksums)
        {
            Ok(_) => {
                self.state
                    .metrics
                    .published_pulled_pages
                    .fetch_add(keys.len() as u64, Ordering::Relaxed);
            }
            Err(error) => {
                tracing::warn!(
                    %error,
                    pages = keys.len(),
                    "HiCache G2 could not publish pages it fetched; peers will not read them \
                     from this replica"
                );
            }
        }
    }

    /// Whether the reads left in flight under `ticket` have all reached a
    /// terminal state, so the engine may reuse their destination pages. Reports
    /// `true` once; an unknown ticket is an error.
    pub fn is_settled(&self, ticket: u64) -> Result<bool> {
        self.state.reaper.is_settled(ticket)
    }

    /// One lease, one set of NIXL reads, one release, against a single peer,
    /// inside the fetch's `deadline`.
    ///
    /// The source's lease expiry is the remaining budget, so a slow lease cannot
    /// buy the transfer time the source has already reclaimed. The release is
    /// spawned on the runtime and bounded by `release_fetch_lease_bounded`;
    /// HiCache's prefetch thread never waits on a stalled source after a
    /// finished transfer.
    ///
    /// Returns the checksum each page was published with, which the read
    /// verified against what landed (unless verification is off).
    fn fetch_run(
        &self,
        instance_id: u64,
        keys: &[SequenceHash],
        destination: &[HiCachePageSegments],
        deadline: Instant,
    ) -> Result<Vec<u64>> {
        let remaining = deadline.saturating_duration_since(Instant::now());
        anyhow::ensure!(
            !remaining.is_zero(),
            "HiCache G2 fetch deadline passed before leasing from instance {instance_id}"
        );
        let expires_at = unix_time_ms()?.saturating_add(u64::try_from(remaining.as_millis())?);
        let lease_id = G2pbFetchLeaseId::new();
        let request = HiCacheFetchRequest {
            lease_id,
            keys: keys.to_vec(),
            lease_expires_at_unix_ms: expires_at,
        };
        let peers = self.peers.clone();
        let lease_started = Instant::now();
        let response = self.runtime.block_on(async move {
            tokio::time::timeout_at(deadline.into(), peers.fetch(instance_id, request))
                .await
                .map_err(|_| {
                    anyhow::anyhow!(
                        "HiCache G2 lease request to instance {instance_id} did not answer \
                         before the fetch deadline ({} ms were left)",
                        remaining.as_millis()
                    )
                })?
        })?;
        self.state
            .metrics
            .lease_rpc_ns
            .fetch_add(elapsed_ns(lease_started), Ordering::Relaxed);

        let result = self
            .state
            .read_from_peer(&response, keys, destination, deadline);

        let owner = self.peers.clone();
        let releaser = self.peers.clone();
        self.runtime.spawn(release_fetch_lease_bounded(
            lease_id,
            instance_id,
            expires_at,
            "HiCache G2 fetch",
            move || owner.instance_ids().contains(&instance_id),
            move || {
                let releaser = releaser.clone();
                async move { releaser.release(instance_id, lease_id).await }
            },
        ));

        let moved = match result {
            Ok(moved) => moved,
            Err(error) => {
                // The pages before the first stale one verified and are in place.
                if let Some(failure) = error.downcast_ref::<VerificationFailure>()
                    && let Some(&first_stale) = failure.stale_pages.first()
                {
                    self.state
                        .metrics
                        .fetched_pages
                        .fetch_add(first_stale as u64, Ordering::Relaxed);
                }
                return Err(error);
            }
        };
        self.state
            .metrics
            .fetched_pages
            .fetch_add(keys.len() as u64, Ordering::Relaxed);
        self.state
            .metrics
            .fetched_bytes
            .fetch_add(moved, Ordering::Relaxed);
        Ok(response.checksums)
    }

    /// Ask the directory where each key lives, dropping this replica's own
    /// entries (a local hit is HiCache's own L2 business, not ours), those of
    /// sources that are no longer live peers, and those of suspended peer pools.
    /// A lookup names no pool (`None`) and skips a peer with any pool suspended:
    /// a page it cannot restore in every pool is not a hit.
    fn query_locations(
        &self,
        keys: &[SequenceHash],
        page_bytes: Option<u64>,
    ) -> Vec<Vec<ReplicaLocation>> {
        let query_started = Instant::now();
        let locations = self
            .index
            .batch_locations(keys, Some(self.state.config.page_size));
        self.state
            .metrics
            .index_query_ns
            .fetch_add(elapsed_ns(query_started), Ordering::Relaxed);
        let live_instances = self.peers.instance_ids().into_iter().collect();
        let (locations, dead) =
            remote_live_candidates(locations, self.state.source.replica_id(), &live_instances);
        if dead != 0 {
            self.state
                .metrics
                .dead_source_candidates
                .fetch_add(dead as u64, Ordering::Relaxed);
        }
        let now = Instant::now();
        let (locations, suspended) = without_suspended_candidates(locations, |instance_id| {
            self.state
                .peer_health
                .is_suspended(instance_id, page_bytes, now)
        });
        if suspended != 0 {
            self.state
                .metrics
                .suspended_peer_candidates
                .fetch_add(suspended as u64, Ordering::Relaxed);
        }
        locations
    }
}

/// Keep the candidates a fetch may lease from: another replica's, and only from
/// a source discovery still lists as a live peer. A crashed source never
/// withdraws its advertisements, so without the liveness filter every reader
/// keeps paying a failed lease on it (KVBM filters the same way). Returns the
/// candidates and how many were dropped as dead.
fn remote_live_candidates(
    locations: Vec<Vec<ReplicaLocation>>,
    own_replica: Option<u64>,
    live_instances: &HashSet<u64>,
) -> (Vec<Vec<ReplicaLocation>>, usize) {
    let mut dead = 0usize;
    let locations = locations
        .into_iter()
        .map(|candidates| {
            candidates
                .into_iter()
                // Filter by *replica*, not by process. At TP8 one replica is
                // eight scheduler processes with eight instance ids, and for MLA
                // their keys are deliberately rank-shared — filtering on
                // instance id would let a rank "fetch" a prefix its own replica
                // already holds.
                .filter(|location| {
                    location.source.replica_id() != own_replica || own_replica.is_none()
                })
                .filter(|location| {
                    let live = live_instances.contains(&location.source.key.instance_id);
                    dead += usize::from(!live);
                    live
                })
                .collect()
        })
        .collect();
    (locations, dead)
}

/// Drop the candidates whose source `suspended` names. Returns the candidates
/// and how many were dropped.
fn without_suspended_candidates(
    locations: Vec<Vec<ReplicaLocation>>,
    suspended: impl Fn(u64) -> bool,
) -> (Vec<Vec<ReplicaLocation>>, usize) {
    let mut dropped = 0usize;
    let locations = locations
        .into_iter()
        .map(|candidates| {
            candidates
                .into_iter()
                .filter(|location| {
                    let skip = suspended(location.source.key.instance_id);
                    dropped += usize::from(skip);
                    !skip
                })
                .collect()
        })
        .collect();
    (locations, dropped)
}

/// Walk `locations` from the front, one peer run at a time, and return how many
/// leading pages were fetched.
///
/// A run is the longest stretch of consecutive pages the chosen peer advertises,
/// so one lease and one set of NIXL requests cover many pages. A failed run
/// never discards a page the peer could still serve:
///
/// - a verification failure names its stale pages, so the verified prefix
///   counts as fetched and only the stale pages lose that candidate;
/// - an unaddressable peer ([`UnaddressablePeer`]) is at fault for every page of
///   the run, not any one of them, so the whole run loses that candidate;
/// - a failure that names no page (lease refused, transfer error, deadline) on a
///   run longer than one page is retried on the first half of the run, and only
///   a single-page failure removes the candidate from that page.
///
/// A run that leaves reads in flight ([`UnsettledReads`]) ends the fetch: its
/// destination may still be written, so no other candidate may read into it.
///
/// Terminates: every iteration advances `completed`, halves the run, or removes
/// a candidate from the first unfinished page. No run starts past `deadline`;
/// the pages fetched so far are still returned.
///
/// `fetch_run` returns the verified checksum of each page of a successful run
/// (a verification failure carries those of its verified prefix), and the
/// result holds one per completed page, for publishing them.
fn fetch_runs(
    locations: &mut [Vec<ReplicaLocation>],
    deadline: Instant,
    mut fetch_run: impl FnMut(u64, Range<usize>) -> Result<Vec<u64>>,
) -> FetchedRuns {
    let mut completed = 0usize;
    let mut checksums = Vec::with_capacity(locations.len());
    let mut run_cap = usize::MAX;
    while completed < locations.len() {
        if Instant::now() >= deadline {
            tracing::warn!(
                completed,
                pages = locations.len(),
                "HiCache G2 fetch deadline passed; not starting another peer run"
            );
            break;
        }
        let Some(instance_id) = locations[completed]
            .first()
            .map(|location| location.source.key.instance_id)
        else {
            break;
        };
        let mut end = completed + 1;
        while end < locations.len()
            && end - completed < run_cap
            && locations[end]
                .iter()
                .any(|location| location.source.key.instance_id == instance_id)
        {
            end += 1;
        }
        let run_len = end - completed;
        match fetch_run(instance_id, completed..end) {
            Ok(run_checksums) => {
                if run_checksums.len() != run_len {
                    // Never publish under a checksum that may belong to another
                    // page; read_from_peer rules this out.
                    tracing::warn!(
                        instance_id,
                        pages = run_len,
                        checksums = run_checksums.len(),
                        "HiCache G2 peer run returned the wrong number of checksums"
                    );
                    break;
                }
                checksums.extend(run_checksums);
                completed = end;
                run_cap = usize::MAX;
            }
            Err(error) => {
                // The run's destination may still be written: stop here rather
                // than reading those pages again from another candidate.
                if let Some(unsettled) = error.downcast_ref::<UnsettledReads>() {
                    return FetchedRuns {
                        outcome: HiCacheFetchOutcome {
                            completed,
                            unsettled_ticket: Some(unsettled.ticket),
                        },
                        checksums,
                    };
                }
                if let Some(failure) = error.downcast_ref::<VerificationFailure>()
                    && !failure.stale_pages.is_empty()
                {
                    let stale_pages = &failure.stale_pages;
                    tracing::warn!(
                        %error,
                        instance_id,
                        pages = run_len,
                        stale = stale_pages.len(),
                        verified_prefix = stale_pages[0],
                        "HiCache G2 peer read failed verification; keeping the verified prefix and \
                         trying the next candidate for the stale pages"
                    );
                    for &page in stale_pages {
                        locations[completed + page]
                            .retain(|location| location.source.key.instance_id != instance_id);
                    }
                    let Some(verified) = failure.verified_checksums.get(..stale_pages[0]) else {
                        // read_from_peer rules this out; see the Ok arm.
                        break;
                    };
                    checksums.extend_from_slice(verified);
                    completed += stale_pages[0];
                    run_cap = usize::MAX;
                } else if error.downcast_ref::<UnaddressablePeer>().is_some() {
                    // Halving would lease from the same unaddressable peer again
                    // for nothing; no page of this run can be read from it.
                    tracing::warn!(
                        %error,
                        instance_id,
                        pages = run_len,
                        "HiCache G2 peer is unaddressable; trying the next candidate for the run"
                    );
                    for candidates in &mut locations[completed..end] {
                        candidates
                            .retain(|location| location.source.key.instance_id != instance_id);
                    }
                    run_cap = usize::MAX;
                } else if run_len > 1 {
                    // The failure does not say which page is at fault; a stale
                    // tail must not cost the healthy head its only copy.
                    run_cap = run_len / 2;
                    tracing::warn!(
                        %error,
                        instance_id,
                        pages = run_len,
                        retry_pages = run_cap,
                        "HiCache G2 peer read failed without naming a page; retrying the first \
                         half of the run"
                    );
                } else {
                    tracing::warn!(
                        %error,
                        instance_id,
                        remaining_candidates = locations[completed].len() - 1,
                        "HiCache G2 peer read failed; trying the next candidate for this page"
                    );
                    locations[completed]
                        .retain(|location| location.source.key.instance_id != instance_id);
                    run_cap = usize::MAX;
                }
            }
        }
    }
    FetchedRuns {
        outcome: HiCacheFetchOutcome {
            completed,
            unsettled_ticket: None,
        },
        checksums,
    }
}

/// What [`fetch_runs`] did, plus the checksum of each completed page.
#[derive(Debug, PartialEq, Eq)]
struct FetchedRuns {
    outcome: HiCacheFetchOutcome,
    /// One per completed page, in order: `checksums.len() == outcome.completed`.
    checksums: Vec<u64>,
}

fn start_peer_service(component: &Component, state: Arc<HiCacheG2State>) -> Result<()> {
    let ingress = Ingress::<SingleIn<HiCachePeerRequest>, ManyOut<Annotated<HiCachePeerResponse>>>::
        for_engine(Arc::new(HiCachePeerEngine { state }))?;
    let endpoint = component.endpoint(HICACHE_G2_PEER_ENDPOINT);
    tokio::spawn(async move {
        if let Err(error) = endpoint
            .endpoint_builder()
            .handler(ingress)
            .graceful_shutdown(true)
            .start()
            .await
        {
            tracing::error!(%error, "HiCache G2 peer endpoint failed");
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn agent_names_are_unique_for_adjacent_instance_ids() {
        // Instance ids come out as consecutive even numbers; every TP4 rank of
        // eight such processes must get its own name.
        let ids: Vec<u64> = (0..8).map(|i| 4_610_454_111_727_008_640 + 2 * i).collect();
        let mut names = HashSet::new();
        let mut old_names = HashSet::new();
        for &id in &ids {
            for rank in 0..4u32 {
                assert!(names.insert(hicache_g2_agent_name(id, rank)));
                old_names.insert(id ^ u64::from(rank));
            }
        }
        // The previous `instance_id ^ tp_rank` scheme collided on exactly these ids.
        assert!(old_names.len() < ids.len() * 4);
    }

    #[test]
    fn cache_keys_are_a_pure_function_of_the_qualified_key() {
        let base = hicache_g2_cache_key("deadbeef_Qwen-Qwen3-8B_0_1");
        assert_eq!(base, hicache_g2_cache_key("deadbeef_Qwen-Qwen3-8B_0_1"));
        // The caller's namespacing is what separates models, ranks and pools.
        assert_ne!(base, hicache_g2_cache_key("deadbeef_Qwen-Qwen3-8B_1_1"));
        assert_ne!(base, hicache_g2_cache_key("deadbeef_Qwen-Qwen3-8B_mla"));
        assert_ne!(
            base,
            hicache_g2_cache_key("deadbeef_Qwen-Qwen3-8B_mla_mamba")
        );
        assert_ne!(base, hicache_g2_cache_key("deadbeee_Qwen-Qwen3-8B_0_1"));
    }

    #[test]
    fn registered_region_bounds_are_half_open() {
        assert!(region_contains(0x1000, 0x100, 0x1000, 0x100));
        assert!(!region_contains(0x1000, 0x100, 0x1000, 0x101));
        assert!(!region_contains(0x1000, 0x100, 0xfff, 1));
        assert!(region_contains(0x1000, 0x100, 0x10ff, 1));
        assert!(!region_contains(0x1000, 0x100, 0x1100, 1));
        // An overflowing length must not wrap into the region.
        assert!(!region_contains(0x1000, 0x100, 0x1000, u64::MAX));
    }

    #[test]
    fn verify_policy_parses_and_defaults_to_the_destination() {
        // SAFETY: single-threaded test, and the variable is read only here.
        unsafe { std::env::remove_var(HICACHE_G2_VERIFY_ENV) };
        assert_eq!(
            HiCacheVerify::from_env().unwrap(),
            HiCacheVerify::Destination
        );
        for (value, expected) in [
            ("destination", HiCacheVerify::Destination),
            ("off", HiCacheVerify::Off),
        ] {
            unsafe { std::env::set_var(HICACHE_G2_VERIFY_ENV, value) };
            assert_eq!(HiCacheVerify::from_env().unwrap(), expected);
        }
        unsafe { std::env::set_var(HICACHE_G2_VERIFY_ENV, "sometimes") };
        assert!(HiCacheVerify::from_env().is_err());
        // Source-side verification was removed, not silently remapped: a
        // deployment still asking for it must fail to start.
        for value in ["source", "SRC"] {
            unsafe { std::env::set_var(HICACHE_G2_VERIFY_ENV, value) };
            let error = HiCacheVerify::from_env().expect_err("source mode is gone");
            assert!(error.to_string().contains("removed"), "{error}");
        }
        unsafe { std::env::remove_var(HICACHE_G2_VERIFY_ENV) };
    }

    #[test]
    fn checksum_segments_covers_every_range_in_order() {
        let first = [1u8, 2, 3, 4];
        let second = [5u8, 6, 7, 8];
        let split = checksum_segments(&[(first.as_ptr() as u64, 4), (second.as_ptr() as u64, 4)]);
        let joined = [1u8, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(split, checksum_segments(&[(joined.as_ptr() as u64, 8)]));
        // Order matters: a page whose k and v were swapped must not verify.
        let swapped = checksum_segments(&[(second.as_ptr() as u64, 4), (first.as_ptr() as u64, 4)]);
        assert_ne!(split, swapped);
    }

    fn future_deadline() -> Instant {
        Instant::now() + Duration::from_secs(60)
    }

    /// Models NIXL 1.4: a re-import over a held map merges into it, unless
    /// `conflict_next`, which rejects it and erases the map; `fail_next` fails
    /// an import into an empty slot.
    #[derive(Default)]
    struct StatefulImporter {
        present: Cell<bool>,
        fail_next: Cell<bool>,
        conflict_next: Cell<bool>,
        imports: Cell<usize>,
    }

    impl RemoteMetadataImporter for StatefulImporter {
        fn import(&self, metadata: &[u8]) -> Result<String> {
            self.imports.set(self.imports.get() + 1);
            if self.present.get() && self.conflict_next.replace(false) {
                self.present.set(false);
                anyhow::bail!("conflicting metadata; map erased");
            }
            anyhow::ensure!(!self.fail_next.replace(false), "import failed");
            self.present.set(true);
            Ok(std::str::from_utf8(metadata)?.to_owned())
        }

        fn is_loaded(&self, _agent_name: &str) -> bool {
            self.present.get()
        }
    }

    #[test]
    fn metadata_is_reused_until_its_generation_advances() {
        let importer = StatefulImporter::default();
        let mut cache = RemoteMetadataCache::default();
        cache.load(&importer, "peer", b"peer", 1).unwrap();
        for generation in [1, 0] {
            assert_eq!(
                cache.load(&importer, "peer", b"peer", generation).unwrap(),
                RemoteMetadataLoad::AlreadyCurrent
            );
        }
        assert_eq!(importer.imports.get(), 1);
        // The newer map is merged over the held one; NIXL never goes without one.
        assert_eq!(
            cache.load(&importer, "peer", b"peer", 2).unwrap(),
            RemoteMetadataLoad::Imported {
                previous_generation: Some(1)
            }
        );
        assert!(importer.present.get());
        assert_eq!(importer.imports.get(), 2);
        assert_eq!(cache.loaded["peer"], 2);
    }

    #[test]
    fn failed_refresh_clears_loaded_state_and_allows_retry() {
        for retry_generation in [1, 2] {
            let importer = StatefulImporter::default();
            let mut cache = RemoteMetadataCache::default();
            cache.load(&importer, "peer", b"peer", 1).unwrap();
            importer.conflict_next.set(true);
            assert!(cache.load(&importer, "peer", b"peer", 2).is_err());
            assert!(!importer.present.get());
            assert!(cache.loaded.is_empty());
            assert_eq!(
                cache
                    .load(&importer, "peer", b"peer", retry_generation)
                    .unwrap(),
                RemoteMetadataLoad::Imported {
                    previous_generation: None
                }
            );
            assert!(importer.present.get());
            assert_eq!(cache.loaded["peer"], retry_generation);
            assert_eq!(importer.imports.get(), 3);
        }
    }

    #[test]
    fn forgetting_a_peer_reimports_its_current_generation() {
        let importer = StatefulImporter::default();
        let mut cache = RemoteMetadataCache::default();
        cache.load(&importer, "peer", b"peer", 2).unwrap();
        cache.forget("peer");
        // The held map stays until the re-import merges over it.
        assert!(importer.present.get());
        assert!(cache.loaded.is_empty());
        assert_eq!(
            cache.load(&importer, "peer", b"peer", 2).unwrap(),
            RemoteMetadataLoad::Imported {
                previous_generation: None
            }
        );
        assert_eq!(importer.imports.get(), 2);
        assert_eq!(cache.loaded["peer"], 2);
    }

    #[test]
    fn backend_invalidation_allows_same_generation_recovery() {
        let importer = StatefulImporter::default();
        let mut cache = RemoteMetadataCache::default();
        cache.load(&importer, "peer", b"peer", 1).unwrap();
        importer.present.set(false);
        cache.load(&importer, "peer", b"peer", 1).unwrap();
        assert!(importer.present.get());
        assert_eq!(importer.imports.get(), 2);
    }

    #[test]
    fn a_failed_first_import_is_retried() {
        let importer = StatefulImporter::default();
        let mut cache = RemoteMetadataCache::default();
        importer.fail_next.set(true);
        assert!(cache.load(&importer, "peer", b"peer", 1).is_err());
        assert!(cache.loaded.is_empty());
        cache.load(&importer, "peer", b"peer", 1).unwrap();
        assert_eq!(cache.loaded["peer"], 1);
    }

    #[test]
    fn metadata_for_another_agent_is_rejected() {
        let importer = StatefulImporter::default();
        let mut cache = RemoteMetadataCache::default();
        assert!(cache.load(&importer, "peer", b"other", 1).is_err());
        assert!(cache.loaded.is_empty());
    }

    fn tracker(chunks: usize) -> G2pbQuarantineTracker<Arc<HiCacheQuarantineCounters>> {
        let tracker = G2pbQuarantineTracker::new(
            1,
            vec![4096; chunks],
            Arc::new(HiCacheQuarantineCounters::default()),
        );
        tracker.mark_dispatched(&(0..chunks).collect::<Vec<_>>());
        tracker
    }

    #[test]
    fn a_status_error_is_a_terminal_failure_and_siblings_still_drain() {
        let tracker = tracker(2);
        let mut polls = [0; 2];
        let outcome = drive_to_terminal(&tracker, vec![0, 1], future_deadline(), "peer", |index| {
            polls[index] += 1;
            anyhow::ensure!(index != 0, "status unavailable");
            Ok(if polls[index] < 4 {
                XferStatus::InProgress
            } else {
                XferStatus::Success
            })
        });
        assert!(outcome.result.is_err());
        assert!(outcome.unsettled.is_empty());
        // The failed read is terminal after one poll; its sibling ran to success.
        assert_eq!(polls, [1, 4]);
        assert!(!tracker.has_in_flight_writes());
        assert_eq!(tracker.completed_prefix_len(), 0);
    }

    #[test]
    fn reads_in_flight_at_the_deadline_are_left_unsettled_not_waited_on() {
        let tracker = tracker(2);
        let mut polls = 0u32;
        let outcome = drive_to_terminal(
            &tracker,
            vec![0, 1],
            Instant::now() - Duration::from_secs(1),
            "peer",
            |index| {
                polls += 1;
                Ok(if index == 0 {
                    XferStatus::Success
                } else {
                    XferStatus::InProgress
                })
            },
        );
        assert!(outcome.result.is_err());
        assert_eq!(outcome.unsettled, vec![1]);
        // Returned at the first clock check instead of polling to completion.
        assert!(polls <= HICACHE_G2_DEADLINE_CHECK_POLLS + 1);
        assert!(tracker.has_in_flight_writes());
        let counters = &tracker.metrics;
        assert_eq!(counters.quarantined_chunks.load(Ordering::Relaxed), 1);
        assert_eq!(counters.quarantined_bytes.load(Ordering::Relaxed), 4096);
    }

    #[test]
    fn healthy_reads_never_enter_quarantine() {
        let tracker = tracker(1);
        let mut polls = 0;
        let outcome = drive_to_terminal(&tracker, vec![0], future_deadline(), "peer", |_| {
            polls += 1;
            Ok(if polls < 3 {
                XferStatus::InProgress
            } else {
                XferStatus::Success
            })
        });
        assert!(outcome.result.is_ok());
        assert!(outcome.unsettled.is_empty());
        assert_eq!(
            tracker.metrics.quarantined_chunks.load(Ordering::Relaxed),
            0
        );
        assert_eq!(tracker.completed_prefix_len(), 1);
    }

    /// A fetch of `chunks` NIXL chunks whose reads all outlived the deadline,
    /// held under fake request ids equal to the chunk index.
    fn held_fetch(
        reaper: &UnsettledReadReaper<usize>,
        chunks: usize,
    ) -> (u64, Arc<HiCacheQuarantineCounters>) {
        let quarantine = Arc::new(HiCacheQuarantineCounters::default());
        let tracker = Arc::new(G2pbQuarantineTracker::new(
            1,
            vec![4096; chunks],
            quarantine.clone(),
        ));
        let all: Vec<usize> = (0..chunks).collect();
        tracker.mark_dispatched(&all);
        tracker.mark_deadline_elapsed();
        let reads = all
            .into_iter()
            .map(|chunk| HeldRead {
                chunk,
                request: chunk,
            })
            .collect();
        (reaper.hold("peer", reads, tracker), quarantine)
    }

    #[test]
    fn a_ticket_settles_when_every_held_read_is_terminal_and_reports_once() {
        let reaper = UnsettledReadReaper::<usize>::default();
        let metrics = HiCacheG2Metrics::default();
        let (ticket, quarantine) = held_fetch(&reaper, 2);
        assert_eq!(quarantine.quarantined_chunks.load(Ordering::Relaxed), 2);

        // Chunk 1 is still being written: the ticket, and both pages, stay held.
        let done = [true, false];
        assert_eq!(
            reaper.reap(&metrics, |&chunk| Ok(if done[chunk] {
                XferStatus::Success
            } else {
                XferStatus::InProgress
            })),
            1
        );
        assert!(!reaper.is_settled(ticket).unwrap());
        assert_eq!(quarantine.quarantined_chunks.load(Ordering::Relaxed), 1);

        // A status error is terminal too.
        assert_eq!(
            reaper.reap(&metrics, |_| anyhow::bail!("status unavailable")),
            0
        );
        assert!(reaper.is_settled(ticket).unwrap());
        // Reported once: the engine released the pages and never asks again.
        assert!(reaper.is_settled(ticket).is_err());
        assert_eq!(quarantine.quarantined_chunks.load(Ordering::Relaxed), 0);
        assert_eq!(
            quarantine
                .quarantine_released_chunks
                .load(Ordering::Relaxed),
            2
        );
        assert_eq!(metrics.settled_fetches.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn a_read_that_never_settles_holds_its_ticket_and_others_still_settle() {
        let reaper = UnsettledReadReaper::<usize>::default();
        let metrics = HiCacheG2Metrics::default();
        let (stuck, _) = held_fetch(&reaper, 1);
        let (healthy, _) = held_fetch(&reaper, 1);
        assert_ne!(stuck, healthy);
        assert!(reaper.is_settled(stuck + healthy).is_err());
        // Both fetches use request id 0; tell them apart by settling in turn.
        let mut polled = 0;
        let held = reaper.reap(&metrics, |_| {
            polled += 1;
            Ok(if polled == 1 {
                XferStatus::InProgress
            } else {
                XferStatus::Success
            })
        });
        assert_eq!(held, 1);
        assert!(!reaper.is_settled(stuck).unwrap());
        assert!(reaper.is_settled(healthy).unwrap());
        for _ in 0..3 {
            assert_eq!(reaper.reap(&metrics, |_| Ok(XferStatus::InProgress)), 1);
            assert!(!reaper.is_settled(stuck).unwrap());
        }
    }

    fn page(addr: u64) -> PublishedPage {
        PublishedPage {
            segments: HiCachePageSegments {
                segments: vec![(addr, 64)],
            },
            checksum: 1,
        }
    }

    #[test]
    fn published_index_tracks_slot_replacement_moves_and_removal() {
        let mut index = PublishedIndex::default();
        assert_eq!(index.insert(1, page(0x1000)), None);
        assert_eq!(index.insert(2, page(0x1000)), Some(1));
        assert!(index.get(&1).is_none());
        assert_eq!(index.insert(2, page(0x1000)), None);
        assert_eq!(index.by_key.len(), 1);
        assert_eq!(index.by_slot.len(), 1);
        assert_eq!(index.insert(2, page(0x2000)), None);
        assert_eq!(index.insert(3, page(0x1000)), None);
        assert_eq!(index.get(&2), Some(&page(0x2000)));
        assert!(index.remove(&2).is_some());
        assert!(index.remove(&2).is_none());
        assert_eq!(index.by_slot.len(), 1);
        assert!(index.remove(&3).is_some());
        assert!(index.by_key.is_empty());
        assert!(index.by_slot.is_empty());
    }

    fn slot(segments: &[(u64, u64)]) -> HiCachePageSegments {
        HiCachePageSegments {
            segments: segments.to_vec(),
        }
    }

    fn published_at(segments: &[(u64, u64)]) -> PublishedPage {
        PublishedPage {
            segments: slot(segments),
            checksum: 1,
        }
    }

    #[test]
    fn remove_slots_drops_only_the_keys_at_destination_slots() {
        let mut index = PublishedIndex::default();
        index.insert(1, page(0x1000));
        index.insert(2, page(0x2000));
        index.insert(3, page(0x3000));
        let mut removed = index.remove_slots(&[page(0x1000).segments, page(0x3000).segments]);
        removed.sort_unstable();
        assert_eq!(removed, vec![1, 3]);
        assert!(index.get(&1).is_none());
        assert!(index.get(&3).is_none());
        // The unrelated slot is untouched.
        assert_eq!(index.get(&2), Some(&page(0x2000)));
        assert_eq!(index.by_slot.len(), 1);
        assert_eq!(index.by_key.len(), 1);
        // Slots that hold nothing, and slots already invalidated, are no-ops.
        assert!(
            index
                .remove_slots(&[page(0x1000).segments, page(0x4000).segments])
                .is_empty()
        );
        // A slot refilled by a fetch and later republished is advertised again.
        assert_eq!(index.insert(4, page(0x1000)), None);
        assert_eq!(index.get(&4), Some(&page(0x1000)));
    }

    #[test]
    fn remove_slots_matches_multi_segment_pages_by_whole_slot() {
        let mut index = PublishedIndex::default();
        // Split k/v pool: one k and one v segment per page.
        let kv_a = [(0x1_0000, 64), (0x9_0000, 64)];
        let kv_b = [(0x1_0040, 64), (0x9_0040, 64)];
        index.insert(1, published_at(&kv_a));
        index.insert(2, published_at(&kv_b));
        // The same slot description as `publish` used is matched.
        assert_eq!(index.remove_slots(&[slot(&kv_a)]), vec![1]);
        assert!(index.get(&1).is_none());
        assert_eq!(index.get(&2), Some(&published_at(&kv_b)));
        // A key that moved slots is only removed by its current slot.
        let kv_c = [(0x1_0080, 64), (0x9_0080, 64)];
        index.insert(2, published_at(&kv_c));
        assert!(index.remove_slots(&[slot(&kv_b)]).is_empty());
        assert_eq!(index.remove_slots(&[slot(&kv_c)]), vec![2]);
        assert!(index.by_key.is_empty());
        assert!(index.by_slot.is_empty());
    }

    #[test]
    fn withdraw_slots_counts_and_withdraws_from_the_replica_index() {
        let published = Mutex::new(PublishedIndex::default());
        {
            let mut index = published.lock().unwrap();
            index.insert(1, page(0x1000));
            index.insert(2, page(0x2000));
        }
        let metrics = HiCacheG2Metrics::default();
        let mut withdrawn = Vec::new();
        let count = withdraw_slots(
            &published,
            &[page(0x1000).segments, page(0x5000).segments],
            &metrics.invalidated_by_fetch,
            |keys| withdrawn.extend(keys),
        )
        .unwrap();
        assert_eq!(count, 1);
        assert_eq!(withdrawn, vec![1]);
        assert!(published.lock().unwrap().get(&1).is_none());
        assert!(published.lock().unwrap().get(&2).is_some());

        // Nothing removed: nothing counted and nothing sent to the index.
        let count = withdraw_slots(
            &published,
            &[page(0x1000).segments],
            &metrics.invalidated_by_caller,
            |_| panic!("no key to withdraw"),
        )
        .unwrap();
        assert_eq!(count, 0);

        let count = withdraw_slots(
            &published,
            &[page(0x2000).segments],
            &metrics.invalidated_by_caller,
            |keys| withdrawn.extend(keys),
        )
        .unwrap();
        assert_eq!(count, 1);
        assert_eq!(withdrawn, vec![1, 2]);

        let snapshot = metrics.snapshot().into_iter().collect::<HashMap<_, _>>();
        assert_eq!(snapshot["invalidated_by_fetch"], 1);
        assert_eq!(snapshot["invalidated_by_caller"], 1);
    }

    #[test]
    fn split_flat_groups_pointers_into_pages() {
        let pages = HiCachePageSegments::split_flat(&[1, 2, 3, 4], &[10, 20, 30, 40], 2).unwrap();
        assert_eq!(
            pages,
            vec![slot(&[(1, 10), (2, 20)]), slot(&[(3, 30), (4, 40)])]
        );
        assert!(
            HiCachePageSegments::split_flat(&[], &[], 1)
                .unwrap()
                .is_empty()
        );
        assert!(HiCachePageSegments::split_flat(&[1], &[10], 0).is_err());
        assert!(HiCachePageSegments::split_flat(&[1, 2], &[10], 1).is_err());
        assert!(HiCachePageSegments::split_flat(&[1, 2, 3], &[1, 2, 3], 2).is_err());
    }

    #[test]
    fn a_page_pulled_into_a_slot_replaces_what_the_slot_advertised() {
        let mut index = PublishedIndex::default();
        assert_eq!(index.insert(1, page(0x1000)), None);
        // A fetch read key 2 into the same slot and publishes it with the
        // checksum it verified.
        let pulled = PublishedPage {
            checksum: 0xfeed,
            ..page(0x1000)
        };
        assert_eq!(index.insert(2, pulled.clone()), Some(1));
        assert!(index.get(&1).is_none());
        assert_eq!(index.get(&2), Some(&pulled));
    }

    #[test]
    fn published_pulled_pages_is_exported() {
        let metrics = HiCacheG2Metrics::default();
        metrics
            .published_pulled_pages
            .fetch_add(3, Ordering::Relaxed);
        assert!(metrics.snapshot().contains(&("published_pulled_pages", 3)));
    }

    /// Stand-in verified checksums: `instance_id * 100 + page`.
    fn sums(instance_id: u64, range: Range<usize>) -> Vec<u64> {
        range.map(|page| instance_id * 100 + page as u64).collect()
    }

    #[test]
    fn every_completed_page_carries_the_checksum_its_run_verified() {
        let mut locations = vec![
            vec![location(7, 0)],
            vec![location(7, 1)],
            vec![location(9, 2)],
        ];
        let fetched = fetch_runs(&mut locations, future_deadline(), |instance_id, range| {
            Ok(sums(instance_id, range))
        });
        assert_eq!(fetched.outcome.completed, 3);
        assert_eq!(fetched.checksums, vec![700, 701, 902]);
    }

    #[test]
    fn pages_that_failed_or_have_no_candidate_are_not_in_the_published_prefix() {
        let mut locations = vec![
            vec![location(7, 0)],
            vec![location(9, 1)],
            vec![location(7, 2)],
        ];
        let fetched = fetch_runs(&mut locations, future_deadline(), |instance_id, range| {
            anyhow::ensure!(instance_id != 9, "transfer failed");
            Ok(sums(instance_id, range))
        });
        assert_eq!(fetched.outcome.completed, 1);
        assert_eq!(fetched.checksums, vec![700]);
    }

    #[test]
    fn only_the_prefix_before_an_unsettled_run_is_published() {
        let mut locations = vec![
            vec![location(7, 0)],
            vec![location(7, 1)],
            vec![location(9, 2)],
            vec![location(9, 3)],
        ];
        let fetched = fetch_runs(&mut locations, future_deadline(), |instance_id, range| {
            if instance_id == 9 {
                return Err(UnsettledReads { ticket: 4 }.into());
            }
            Ok(sums(instance_id, range))
        });
        // Pages 2 and 3 may still be written and the engine counts them as
        // misses: no checksum, so they are never published.
        assert_eq!(
            fetched,
            FetchedRuns {
                outcome: HiCacheFetchOutcome {
                    completed: 2,
                    unsettled_ticket: Some(4),
                },
                checksums: vec![700, 701],
            }
        );
    }

    #[test]
    fn a_run_with_the_wrong_number_of_checksums_ends_the_prefix() {
        let mut locations = vec![vec![location(7, 0)], vec![location(9, 1)]];
        let fetched = fetch_runs(&mut locations, future_deadline(), |instance_id, range| {
            Ok(if instance_id == 9 {
                Vec::new()
            } else {
                sums(instance_id, range)
            })
        });
        assert_eq!(fetched.outcome.completed, 1);
        assert_eq!(fetched.checksums, vec![700]);
    }

    fn location(instance_id: u64, key: SequenceHash) -> ReplicaLocation {
        ReplicaLocation {
            source: ReplicaSource {
                key: ReplicaSourceKey {
                    instance_id,
                    tp_rank: 0,
                },
                replica_id: Some(instance_id),
                tp_world_size: 1,
                incarnation: uuid::Uuid::from_u128(instance_id as u128),
                block_size: 64,
                row: None,
            },
            block: ReplicaBlock {
                key,
                size_bytes: 64,
                checksum: None,
            },
            revision: 0,
        }
    }

    #[test]
    fn candidates_from_dead_sources_and_our_own_replica_are_dropped() {
        let locations = vec![
            vec![location(7, 0), location(8, 0), location(9, 0)],
            vec![location(8, 1)],
        ];
        let live = HashSet::from([7, 9]);
        let (kept, dead) = remote_live_candidates(locations, Some(9), &live);
        let instances = |candidates: &Vec<ReplicaLocation>| {
            candidates
                .iter()
                .map(|location| location.source.key.instance_id)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            kept.iter().map(instances).collect::<Vec<_>>(),
            vec![vec![7], vec![]]
        );
        assert_eq!(dead, 2);
    }

    #[test]
    fn a_page_advertised_only_by_a_dead_source_ends_the_prefix() {
        let mut locations = remote_live_candidates(
            vec![vec![location(7, 0)], vec![location(8, 1)]],
            None,
            &HashSet::from([7]),
        )
        .0;
        let mut calls = Vec::new();
        let completed = fetch_runs(&mut locations, future_deadline(), |instance_id, range| {
            calls.push((instance_id, range.clone()));
            Ok(sums(instance_id, range))
        })
        .outcome
        .completed;
        assert_eq!(completed, 1);
        assert_eq!(calls, vec![(7, 0..1)]);
    }

    #[test]
    fn a_stale_tail_preserves_the_only_healthy_copy_of_the_first_page() {
        let mut locations = vec![vec![location(7, 0)], vec![location(7, 1), location(9, 1)]];
        let mut calls = Vec::new();
        let completed = fetch_runs(&mut locations, future_deadline(), |instance_id, range| {
            calls.push((instance_id, range.clone()));
            anyhow::ensure!(instance_id != 7 || !range.contains(&1), "stale tail");
            Ok(sums(instance_id, range))
        })
        .outcome
        .completed;
        assert_eq!(completed, 2);
        assert_eq!(calls, vec![(7, 0..2), (7, 0..1), (7, 1..2), (9, 1..2)]);
        assert_eq!(locations[0].len(), 1);
    }

    #[test]
    fn verification_failure_retries_only_from_the_first_stale_page() {
        let mut locations: Vec<_> = (0..4)
            .map(|key| vec![location(7, key), location(9, key)])
            .collect();
        let mut calls = Vec::new();
        let fetched = fetch_runs(&mut locations, future_deadline(), |instance_id, range| {
            calls.push((instance_id, range.clone()));
            if instance_id == 7 {
                return Err(VerificationFailure {
                    source_agent: "peer".into(),
                    source_instance: 7,
                    stale_pages: vec![1],
                    total_pages: 4,
                    verified_checksums: sums(7, 0..1),
                }
                .into());
            }
            Ok(sums(instance_id, range))
        });
        assert_eq!(fetched.outcome.completed, 4);
        // The verified page keeps the checksum it was read with; the stale ones
        // take those of the candidate that served them.
        assert_eq!(fetched.checksums, vec![700, 901, 902, 903]);
        assert_eq!(calls, vec![(7, 0..4), (9, 1..4)]);
        assert_eq!(
            locations.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![2, 1, 2, 2]
        );
    }

    #[test]
    fn the_prefix_stops_where_no_candidate_is_left() {
        let mut locations = vec![
            vec![location(7, 0), location(9, 0)],
            vec![location(7, 1)],
            vec![location(9, 2)],
        ];
        let completed = fetch_runs(&mut locations, future_deadline(), |instance_id, range| {
            assert!(range.start < 2);
            anyhow::ensure!(instance_id != 7, "peer unavailable");
            Ok(sums(instance_id, range))
        })
        .outcome
        .completed;
        assert_eq!(completed, 1);
        assert!(locations[1].is_empty());
    }

    #[test]
    fn no_run_starts_once_the_deadline_has_passed() {
        let mut locations = vec![vec![location(7, 0)]];
        assert_eq!(
            fetch_runs(&mut locations, Instant::now(), |_, _| panic!(
                "deadline passed"
            ))
            .outcome,
            HiCacheFetchOutcome {
                completed: 0,
                unsettled_ticket: None,
            }
        );
        assert_eq!(locations[0].len(), 1);
    }

    #[test]
    fn an_unaddressable_peer_loses_the_whole_run_without_halving() {
        let mut locations: Vec<_> = (0..4)
            .map(|key| vec![location(7, key), location(9, key)])
            .collect();
        let mut calls = Vec::new();
        let completed = fetch_runs(&mut locations, future_deadline(), |instance_id, range| {
            calls.push((instance_id, range.clone()));
            if instance_id == 7 {
                return Err(UnaddressablePeer {
                    agent_name: "peer".into(),
                    detail: "failed to create NIXL read request".into(),
                }
                .into());
            }
            Ok(sums(instance_id, range))
        })
        .outcome
        .completed;
        assert_eq!(completed, 4);
        assert_eq!(calls, vec![(7, 0..4), (9, 0..4)]);
        assert!(locations.iter().all(|candidates| candidates.len() == 1));
    }

    #[test]
    fn an_unaddressable_peer_is_recognised_through_added_context() {
        let error = anyhow::Error::from(UnaddressablePeer {
            agent_name: "peer".into(),
            detail: "rejected".into(),
        })
        .context("reading from instance 7");
        assert!(error.downcast_ref::<UnaddressablePeer>().is_some());
    }

    fn pool(instance_id: u64) -> PeerPoolKey {
        PeerPoolKey {
            instance_id,
            page_bytes: 667_392,
        }
    }

    #[test]
    fn repeated_unaddressable_failures_reimport_then_suspend_with_backoff() {
        let tracker = PeerHealthTracker::default();
        let start = Instant::now();
        let actions: Vec<_> = (0..HICACHE_G2_PEER_SUSPEND_AFTER)
            .map(|_| tracker.record_failure(pool(7), start))
            .collect();
        assert_eq!(
            actions,
            vec![
                PeerFailureAction::Retry,
                PeerFailureAction::Retry,
                PeerFailureAction::Reimport,
                PeerFailureAction::Retry,
                PeerFailureAction::Retry,
                PeerFailureAction::Suspend(HICACHE_G2_PEER_SUSPEND_INITIAL),
            ]
        );
        assert!(tracker.is_suspended(7, Some(pool(7).page_bytes), start));
        // Reads that were already running when it was suspended change nothing.
        assert_eq!(
            tracker.record_failure(pool(7), start + Duration::from_secs(1)),
            PeerFailureAction::Retry
        );
        let expired = start + HICACHE_G2_PEER_SUSPEND_INITIAL;
        assert!(!tracker.is_suspended(7, Some(pool(7).page_bytes), expired));
        // A failed probe suspends again at once, for twice as long.
        assert_eq!(
            tracker.record_failure(pool(7), expired),
            PeerFailureAction::Suspend(HICACHE_G2_PEER_SUSPEND_INITIAL * 2)
        );
        let mut now = expired;
        let mut last = Duration::ZERO;
        for _ in 0..10 {
            now += HICACHE_G2_PEER_SUSPEND_MAX;
            if let PeerFailureAction::Suspend(suspension) = tracker.record_failure(pool(7), now) {
                last = suspension;
            }
        }
        assert_eq!(last, HICACHE_G2_PEER_SUSPEND_MAX);
    }

    #[test]
    fn peer_pools_recover_independently() {
        let tracker = PeerHealthTracker::default();
        let now = Instant::now();
        let other_pool = PeerPoolKey {
            instance_id: 7,
            page_bytes: 12_307,
        };
        for _ in 0..HICACHE_G2_PEER_SUSPEND_AFTER {
            tracker.record_failure(pool(7), now);
        }
        tracker.record_failure(pool(9), now);
        // The same peer's other pool succeeding does not vouch for this one.
        assert_eq!(tracker.record_success(other_pool), 0);
        assert!(tracker.is_suspended(7, Some(pool(7).page_bytes), now));
        assert!(!tracker.is_suspended(7, Some(other_pool.page_bytes), now));
        // A lookup, which names no pool, skips a peer with any pool suspended.
        assert!(tracker.is_suspended(7, None, now));
        assert!(!tracker.is_suspended(9, Some(pool(9).page_bytes), now));
        assert_eq!(tracker.record_success(pool(9)), 1);
        assert_eq!(
            tracker.record_failure(pool(9), now),
            PeerFailureAction::Retry
        );
        tracker.forget_instance(7);
        assert!(!tracker.is_suspended(7, Some(pool(7).page_bytes), now));
        assert_eq!(
            tracker.record_failure(pool(7), now),
            PeerFailureAction::Retry
        );
    }

    #[test]
    fn suspended_candidates_are_skipped() {
        let (kept, dropped) = without_suspended_candidates(
            vec![vec![location(7, 0), location(9, 0)], vec![location(7, 1)]],
            |instance_id| instance_id == 7,
        );
        assert_eq!(dropped, 2);
        assert_eq!(kept[0].len(), 1);
        assert_eq!(kept[0][0].source.key.instance_id, 9);
        assert!(kept[1].is_empty());
    }

    #[test]
    fn an_unsettled_run_ends_the_fetch_without_rereading_its_pages() {
        let mut locations: Vec<_> = (0..3)
            .map(|key| vec![location(7, key), location(9, key)])
            .collect();
        let mut calls = Vec::new();
        let mut first = true;
        let outcome = fetch_runs(&mut locations, future_deadline(), |instance_id, range| {
            calls.push((instance_id, range.clone()));
            if std::mem::take(&mut first) {
                return Err(UnsettledReads { ticket: 5 }.into());
            }
            Ok(sums(instance_id, range))
        });
        assert_eq!(
            outcome,
            FetchedRuns {
                outcome: HiCacheFetchOutcome {
                    completed: 0,
                    unsettled_ticket: Some(5),
                },
                checksums: vec![],
            }
        );
        // No second read into the pages the first one may still be writing.
        assert_eq!(calls, vec![(7, 0..3)]);
    }
}
