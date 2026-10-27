// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang HiCache host tier (L2) and G2PB peer reads, hashes only.
//!
//! Follows SGLang's HiRadixCache: `write_through` copies every page stored on
//! device to host, `write_back` copies pages as they leave the device. Host
//! pages still on device are never evicted from host; the rest are evicted
//! leaf-first in LRU order. A request's pages after its device prefix are
//! served from host while contiguous, then from peers if they hold at least
//! SGLang's prefetch threshold. Host-tier stores and removals are emitted as
//! KV events for the router (`CPU_PINNED`) and mirrored to the peer directory.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::time::Duration;

use dynamo_kv_router::protocols::{
    BlockHashOptions, ExternalSequenceBlockHash, KvCacheEventData, KvCacheRemoveData,
    KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash, compute_block_hash_for_seq,
    compute_seq_hash_for_block,
};
use serde::{Deserialize, Serialize};

use crate::peer::{PeerDirectory, PeerDirectoryHandle, PeerIndex};

/// SGLang's default `--hicache-storage-prefetch-threshold` (tokens).
const PEER_PREFETCH_THRESHOLD: usize = 256;
const PREFETCH_POLL_MS: f64 = 5.0;

/// HiCache write policy (sglang `--hicache-write-policy`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WritePolicy {
    #[default]
    WriteThrough,
    WriteBack,
}

/// Mocker `sglang.hicache` config. Sizes and bandwidths are per rank (GPU);
/// see the crate README for how to calibrate them.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HiCacheArgs {
    /// Host pool size in GB per rank (sglang `--hicache-size`). Requires
    /// `kv_bytes_per_token`.
    pub size: f64,
    pub write_policy: WritePolicy,
    /// Per-rank host-to-device load-back bandwidth in GB/s. Unset: instant.
    pub bandwidth_gb_per_sec: Option<f64>,
    /// Enables G2PB. Offline, workers in one process with the same group
    /// share pages; live mockers use the fleet-wide G2Direct index.
    pub peer_group: Option<String>,
    /// Per-rank peer read bandwidth in GB/s, as achieved at real fetch sizes
    /// (not link rate). Unset: instant. Concurrent reads do not share it.
    pub peer_bandwidth_gb_per_sec: Option<f64>,
    /// Attention TP size. MHA/GQA KV heads are split across TP ranks; with
    /// fewer KV heads than ranks, set the KV head count.
    pub tp_size: usize,
    /// Decode context parallel size (MLA only): each DCP rank holds an
    /// interleaved slice of every page.
    pub dcp_size: usize,
    /// MLA: the latent KV is replicated on every TP rank.
    pub mla: bool,
}

impl Default for HiCacheArgs {
    fn default() -> Self {
        Self {
            size: 0.0,
            write_policy: WritePolicy::default(),
            bandwidth_gb_per_sec: None,
            peer_group: None,
            peer_bandwidth_gb_per_sec: None,
            tp_size: 1,
            dcp_size: 1,
            mla: false,
        }
    }
}

impl HiCacheArgs {
    pub fn validate(&self, kv_bytes_per_token: Option<usize>) -> Result<(), String> {
        let positive = |v: f64| v.is_finite() && v > 0.0;
        if !positive(self.size)
            || !self.bandwidth_gb_per_sec.is_none_or(positive)
            || !self.peer_bandwidth_gb_per_sec.is_none_or(positive)
        {
            return Err("sglang.hicache size and bandwidths must be finite and > 0".into());
        }
        if kv_bytes_per_token.is_none_or(|b| b == 0) {
            return Err("sglang.hicache requires kv_bytes_per_token > 0".into());
        }
        if self.tp_size == 0 || self.dcp_size == 0 || (self.dcp_size > 1 && !self.mla) {
            return Err(
                "sglang.hicache tp_size and dcp_size must be >= 1; dcp_size > 1 needs mla".into(),
            );
        }
        Ok(())
    }

    /// KV bytes one rank holds per token, from the model's `kv_bytes_per_token`
    /// (all layers and heads).
    pub fn rank_kv_bytes(&self, kv_bytes_per_token: usize) -> usize {
        let ranks = if self.mla {
            self.dcp_size
        } else {
            self.tp_size
        };
        kv_bytes_per_token.div_ceil(ranks)
    }

    /// Tags of the ranks holding distinct parts of a page, like G2Direct's key
    /// suffixes: one per MHA/GQA TP rank, one per MLA DCP rank (MLA is the
    /// same on every TP rank). Different geometries never share keys.
    fn rank_tags(&self) -> Vec<u64> {
        let tags: Vec<String> = match (self.mla, self.dcp_size) {
            (false, _) => (0..self.tp_size)
                .map(|r| format!("{r}_{}", self.tp_size))
                .collect(),
            (true, 1) => vec!["mla".into()],
            (true, dcp) => (0..dcp).map(|r| format!("mla_dcp{dcp}_{r}")).collect(),
        };
        let hash = |tag: String| dynamo_tokens::compute_hash_v2(tag.as_bytes(), 0);
        tags.into_iter().map(hash).collect()
    }
}

/// Prefix tokens a request reuses beyond its device prefix.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrefixHits {
    /// Served from this worker's host tier.
    pub host_tokens: usize,
    /// Read from a peer's host tier into this one.
    pub peer_tokens: usize,
}

/// A host-tier KV event and its per-page token ids.
pub type HostEvent = (KvCacheEventData, Option<Vec<Vec<u32>>>);

#[derive(Clone, Debug)]
struct Page {
    local: u64,
    parent: Option<u64>,
    tokens: Vec<u32>,
}

struct HostPage {
    page: Page,
    last_used: u64,
}

pub struct HiCache {
    args: HiCacheArgs,
    page_size: usize,
    /// KV bytes per token on one rank.
    rank_kv_bytes: usize,
    capacity_pages: usize,
    peer: Option<Arc<dyn PeerDirectory>>,
    /// See [`HiCacheArgs::rank_tags`].
    rank_tags: Vec<u64>,
    /// Pages on device, by sequence hash.
    device: HashMap<u64, Page>,
    host: HashMap<u64, HostPage>,
    /// Host pages per parent sequence hash.
    host_children: HashMap<u64, usize>,
    /// Host-evictable pages (off device, no host children) by last use.
    evictable: BTreeSet<(u64, u64)>,
    clock: u64,
    stored: Vec<(u64, Page)>,
    removed: Vec<u64>,
    events: Vec<HostEvent>,
    /// Peer updates held until [`Self::publish_peer_updates`]: (publish?, hashes).
    peer_pending: Option<Vec<(bool, Vec<u64>)>>,
    /// Simulated time, from [`Self::set_now_ms`].
    now_ms: f64,
    /// Per request id: when its peer read finishes and how many pages it read.
    prefetches: HashMap<u128, (f64, usize)>,
    /// Peer pages the request being admitted read; `None` (no prefetch
    /// modeled, e.g. direct callers): whatever peers hold.
    admit_peer_pages: Option<usize>,
}

impl HiCache {
    /// `peer` overrides the in-process directory for `args.peer_group`.
    /// `kv_bytes_per_token` is the model's, over all layers and heads.
    pub fn new(
        args: &HiCacheArgs,
        page_size: usize,
        kv_bytes_per_token: usize,
        peer: Option<&PeerDirectoryHandle>,
    ) -> Self {
        let peer = match (peer, &args.peer_group) {
            (Some(handle), _) => Some(handle.0.clone()),
            (None, Some(group)) => Some(Arc::new(PeerIndex::new(group)) as Arc<dyn PeerDirectory>),
            (None, None) => None,
        };
        // shortcut: assumes a 64-token engine page; G2Direct keys a DCP page
        // widened by dcp_size.
        if args.dcp_size > 1 && page_size < 64 * args.dcp_size {
            tracing::warn!(
                page_size,
                dcp_size = args.dcp_size,
                "under DCP, G2Direct keys widened pages: set block_size to page_size x dcp_size (512 for DCP8)"
            );
        }
        let rank_kv_bytes = args.rank_kv_bytes(kv_bytes_per_token);
        let host_tokens = (args.size * 1e9) as usize / rank_kv_bytes;
        tracing::info!(
            rank_kv_bytes,
            host_tokens = host_tokens / page_size * page_size,
            "HiCache host tier"
        );
        Self {
            args: args.clone(),
            page_size,
            rank_kv_bytes,
            capacity_pages: host_tokens / page_size,
            peer,
            rank_tags: args.rank_tags(),
            device: HashMap::new(),
            host: HashMap::new(),
            host_children: HashMap::new(),
            evictable: BTreeSet::new(),
            clock: 0,
            stored: Vec::new(),
            removed: Vec::new(),
            events: Vec::new(),
            peer_pending: None,
            now_ms: 0.0,
            prefetches: HashMap::new(),
            admit_peer_pages: None,
        }
    }

    pub fn set_now_ms(&mut self, now_ms: f64) {
        self.now_ms = now_ms;
        self.admit_peer_pages = None;
    }

    /// Like SGLang's `wait_complete` storage prefetch, a request reads the
    /// pages peers hold after its device and host prefix while it waits in
    /// the queue, so the read delays only that request. The first call starts
    /// the read of what fits in the host tier; true until it finishes. Once it
    /// has, admitting the request takes at most the pages that were read.
    pub fn prefetch_pending(
        &mut self,
        request: u128,
        token_ids: &[u64],
        device_len: usize,
    ) -> bool {
        if !self.prefetches.contains_key(&request) {
            let pages = self.pages(token_ids, device_len);
            let host = self.host_hit_pages(&pages);
            // Room on host: all but pages still on device (host eviction
            // frees off-device chains leaf by leaf) and the host hits this
            // request pins.
            let on_device = self.host.keys().filter(|s| self.device.contains_key(s));
            let pinned = on_device.count() + host;
            let room = self.capacity_pages.saturating_sub(pinned);
            let peer = self.peer_hit_pages(&pages[host..], room);
            let read = self.transfer(peer * self.page_size, self.args.peer_bandwidth_gb_per_sec);
            let ready_ms = self.now_ms + read.as_secs_f64() * 1000.0;
            self.prefetches.insert(request, (ready_ms, peer));
        }
        let (ready_ms, pages) = self.prefetches[&request];
        let pending = ready_ms > self.now_ms;
        self.admit_peer_pages = (!pending).then_some(pages);
        pending
    }

    /// Forget a request's peer read once it is admitted.
    /// shortcut: requests dropped while waiting keep their entry.
    pub fn end_prefetch(&mut self, request: u128) {
        self.prefetches.remove(&request);
        self.admit_peer_pages = None;
    }

    /// When an idle scheduler should look again: when the next peer read
    /// ends, or sooner so a new arrival is not stuck behind it.
    /// shortcut: polls every `PREFETCH_POLL_MS`, so an arrival at an idle
    /// worker waits up to that long.
    pub fn next_prefetch_wakeup_ms(&self) -> Option<f64> {
        let pending = self.prefetches.values().map(|&(t, _)| t);
        let next = pending.filter(|&t| t > self.now_ms).reduce(f64::min)?;
        Some(next.min(self.now_ms + PREFETCH_POLL_MS))
    }

    /// Hold peer updates until [`Self::publish_peer_updates`], so a live
    /// mocker advertises pages when its simulated pass ends, like its router
    /// events.
    pub fn defer_peer_updates(&mut self) {
        self.peer_pending.get_or_insert_default();
    }

    pub fn publish_peer_updates(&mut self) {
        let (Some(peer), Some(pending)) = (&self.peer, self.peer_pending.as_mut()) else {
            return;
        };
        for (publish, keys) in pending.drain(..) {
            if publish {
                peer.publish(&keys);
            } else {
                peer.withdraw(&keys);
            }
        }
    }

    /// Every rank's peer key for each of `hashes`.
    fn rank_keys(&self, hashes: &[u64]) -> Vec<u64> {
        let tags = &self.rank_tags;
        tags.iter()
            .flat_map(|&t| hashes.iter().map(move |&h| peer_key(h, t)))
            .collect()
    }

    fn peer_update(&mut self, publish: bool, hashes: Vec<u64>) {
        let hashes = self.rank_keys(&hashes);
        match (&self.peer, self.peer_pending.as_mut()) {
            (None, _) => {}
            (Some(_), Some(pending)) => pending.push((publish, hashes)),
            (Some(peer), None) if publish => peer.publish(&hashes),
            (Some(peer), None) => peer.withdraw(&hashes),
        }
    }

    /// Follow a device-tier KV event published by the mocker.
    pub fn on_device_event(&mut self, data: &KvCacheEventData, token_ids: Option<&[Vec<u32>]>) {
        match data {
            KvCacheEventData::Stored(store) => {
                let mut parent = store.parent_hash.map(|h| h.0);
                for (i, block) in store.blocks.iter().enumerate() {
                    let seq = block.block_hash.0;
                    let page = Page {
                        local: block.tokens_hash.0,
                        parent,
                        tokens: token_ids
                            .and_then(|t| t.get(i).cloned())
                            .unwrap_or_default(),
                    };
                    parent = Some(seq);
                    self.device.insert(seq, page.clone());
                    if self.host.contains_key(&seq) {
                        self.refresh(seq);
                    } else if self.args.write_policy == WritePolicy::WriteThrough {
                        self.backup(seq, page);
                    }
                }
            }
            KvCacheEventData::Removed(remove) => {
                for hash in &remove.block_hashes {
                    let seq = hash.0;
                    let Some(page) = self.device.remove(&seq) else {
                        continue;
                    };
                    if self.host.contains_key(&seq) {
                        self.touch(seq);
                    } else if self.args.write_policy == WritePolicy::WriteBack {
                        self.backup(seq, page);
                    }
                }
            }
            KvCacheEventData::Cleared => {
                self.device.clear();
                let off_device: Vec<u64> = self.host.keys().copied().collect();
                for seq in off_device {
                    self.refresh(seq);
                }
            }
        }
        self.flush();
    }

    /// Serve the pages of `token_ids` after `device_len` from host while
    /// contiguous, then read peer pages into host. The caller loads the
    /// returned tokens to device.
    pub fn take_prefix(&mut self, token_ids: &[u64], device_len: usize) -> PrefixHits {
        let pages = self.pages(token_ids, device_len);
        let host = self.host_hit_pages(&pages);
        let peer_pages = self.peer_hit_pages(&pages[host..], self.admit_cap());
        // Taken pages go to device next; mark them so now, or backing up later
        // pages of this take could evict them from host.
        for (seq, page) in &pages[..host] {
            self.device.insert(*seq, page.clone());
            self.touch(*seq);
        }
        let mut peer = 0;
        for (seq, page) in pages[host..].iter().take(peer_pages) {
            if !self.backup(*seq, page.clone()) {
                break;
            }
            self.device.insert(*seq, page.clone());
            self.refresh(*seq);
            peer += 1;
        }
        self.flush();
        PrefixHits {
            host_tokens: host * self.page_size,
            peer_tokens: peer * self.page_size,
        }
    }

    /// Tokens [`Self::take_prefix`] would serve. Read-only.
    pub fn peek_prefix(&self, token_ids: &[u64], device_len: usize) -> usize {
        let pages = self.pages(token_ids, device_len);
        let host = self.host_hit_pages(&pages);
        (host + self.peer_hit_pages(&pages[host..], self.admit_cap())) * self.page_size
    }

    /// Host-tier KV events since the last call.
    pub fn drain_events(&mut self) -> Vec<HostEvent> {
        std::mem::take(&mut self.events)
    }

    /// Prefill time with load-back of host and peer tokens (peer pages are on
    /// host by then) overlapping compute layer by layer.
    /// `cached_by_tier`: reused prefix tokens from device, host and peers.
    /// shortcut: load-back is batch-level and concurrent transfers do not
    /// share bandwidth.
    pub fn prefill_time(&self, compute: Duration, cached_by_tier: [usize; 3]) -> Duration {
        let [_, host_tokens, peer_tokens] = cached_by_tier;
        compute.max(self.transfer(host_tokens + peer_tokens, self.args.bandwidth_gb_per_sec))
    }

    /// Wire time for `tokens` at `gb_per_sec`; independent of the mocker's
    /// compute `speedup_ratio`.
    fn transfer(&self, tokens: usize, gb_per_sec: Option<f64>) -> Duration {
        gb_per_sec.map_or(Duration::ZERO, |bw| {
            Duration::from_secs_f64((tokens * self.rank_kv_bytes) as f64 / (bw * 1e9))
        })
    }

    /// Router hashes of the full pages of `token_ids` after `skip_tokens`;
    /// the same recurrence as the router and the SGLang mocker's events.
    fn pages(&self, token_ids: &[u64], skip_tokens: usize) -> Vec<(u64, Page)> {
        let tokens: Vec<u32> = token_ids.iter().map(|&t| t as u32).collect();
        let locals =
            compute_block_hash_for_seq(&tokens, self.page_size as u32, BlockHashOptions::default());
        let seqs = compute_seq_hash_for_block(&locals);
        let skip = (skip_tokens / self.page_size).min(seqs.len());
        (skip..seqs.len())
            .map(|i| {
                let page = Page {
                    local: locals[i].0,
                    parent: i.checked_sub(1).map(|p| seqs[p]),
                    tokens: tokens[i * self.page_size..(i + 1) * self.page_size].to_vec(),
                };
                (seqs[i], page)
            })
            .collect()
    }

    /// Leading `pages` on this host tier.
    fn host_hit_pages(&self, pages: &[(u64, Page)]) -> usize {
        let on_host = |(seq, _): &&(u64, Page)| self.host.contains_key(seq);
        pages.iter().take_while(on_host).count()
    }

    /// Leading `pages` peers hold on every rank, at most `cap`; 0 below the
    /// prefetch threshold. Like SGLang's G2Direct lookup, each rank looks up
    /// its own keys and the hit is the minimum over ranks.
    fn peer_hit_pages(&self, pages: &[(u64, Page)], cap: usize) -> usize {
        let Some(peer) = &self.peer else {
            return 0;
        };
        let rank_hits = self.rank_tags.iter().map(|&tag| {
            let keys: Vec<u64> = pages.iter().map(|(seq, _)| peer_key(*seq, tag)).collect();
            peer.contiguous_hits(&keys)
        });
        let n = rank_hits.min().unwrap_or(0).min(cap);
        if n * self.page_size >= PEER_PREFETCH_THRESHOLD {
            n
        } else {
            0
        }
    }

    /// Peer pages the request being admitted may take.
    fn admit_cap(&self) -> usize {
        self.admit_peer_pages.unwrap_or(usize::MAX)
    }

    /// Copy a page to host, evicting to make room. False when it cannot fit.
    fn backup(&mut self, seq: u64, page: Page) -> bool {
        if self.host.contains_key(&seq) {
            self.touch(seq);
            return true;
        }
        while self.host.len() >= self.capacity_pages && self.evict_one() {}
        if self.host.len() >= self.capacity_pages {
            return false;
        }
        if let Some(parent) = page.parent {
            *self.host_children.entry(parent).or_default() += 1;
            self.refresh(parent);
        }
        self.clock += 1;
        self.host.insert(
            seq,
            HostPage {
                page: page.clone(),
                last_used: self.clock,
            },
        );
        self.refresh(seq);
        self.stored.push((seq, page));
        true
    }

    fn evict_one(&mut self) -> bool {
        let Some((_, seq)) = self.evictable.pop_first() else {
            return false;
        };
        let entry = self.host.remove(&seq).expect("evictable pages are on host");
        if let Some(parent) = entry.page.parent {
            if let Some(count) = self.host_children.get_mut(&parent) {
                *count -= 1;
                if *count == 0 {
                    self.host_children.remove(&parent);
                }
            }
            self.refresh(parent);
        }
        // A page stored since the last flush was never advertised: just drop it.
        match self.stored.iter().position(|(s, _)| *s == seq) {
            Some(i) => {
                self.stored.remove(i);
            }
            None => self.removed.push(seq),
        }
        true
    }

    fn touch(&mut self, seq: u64) {
        let Some(entry) = self.host.get_mut(&seq) else {
            return;
        };
        self.evictable.remove(&(entry.last_used, seq));
        self.clock += 1;
        entry.last_used = self.clock;
        self.refresh(seq);
    }

    /// Recompute whether `seq` is host-evictable.
    fn refresh(&mut self, seq: u64) {
        let Some(entry) = self.host.get(&seq) else {
            return;
        };
        let key = (entry.last_used, seq);
        if self.device.contains_key(&seq) || self.host_children.contains_key(&seq) {
            self.evictable.remove(&key);
        } else {
            self.evictable.insert(key);
        }
    }

    /// Turn pending host stores and removals into KV events and peer updates.
    fn flush(&mut self) {
        if !self.removed.is_empty() {
            let removed = std::mem::take(&mut self.removed);
            self.peer_update(false, removed.clone());
            let block_hashes = removed.into_iter().map(ExternalSequenceBlockHash).collect();
            self.events.push((
                KvCacheEventData::Removed(KvCacheRemoveData { block_hashes }),
                None,
            ));
        }
        let stored = std::mem::take(&mut self.stored);
        if !stored.is_empty() {
            self.peer_update(true, stored.iter().map(|(seq, _)| *seq).collect());
        }
        // One event per run of pages chained parent to child.
        let mut runs: Vec<Vec<(u64, Page)>> = Vec::new();
        for (seq, page) in stored {
            match runs.last_mut() {
                Some(run) if page.parent == run.last().map(|(s, _)| *s) => run.push((seq, page)),
                _ => runs.push(vec![(seq, page)]),
            }
        }
        for run in runs {
            let parent_hash = run[0].1.parent.map(ExternalSequenceBlockHash);
            let tokens = run.iter().map(|(_, p)| p.tokens.clone()).collect();
            let blocks = run
                .iter()
                .map(|(seq, p)| KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(*seq),
                    tokens_hash: LocalBlockHash(p.local),
                    mm_extra_info: None,
                })
                .collect();
            self.events.push((
                KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash,
                    start_position: None,
                    blocks,
                }),
                Some(tokens),
            ));
        }
    }
}

/// A rank's peer key for a page.
fn peer_key(seq_hash: u64, rank_tag: u64) -> u64 {
    dynamo_tokens::compute_next_sequence_hash(seq_hash, rank_tag)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: usize = 4;

    fn args(pages: usize, write_policy: WritePolicy) -> HiCacheArgs {
        HiCacheArgs {
            size: (pages * PAGE * 1000) as f64 / 1e9,
            write_policy,
            ..Default::default()
        }
    }

    /// The device events the SGLang mocker publishes for `tokens`.
    fn device_stored(cache: &HiCache, tokens: &[u64]) -> (KvCacheEventData, Vec<Vec<u32>>) {
        let pages = cache.pages(tokens, 0);
        let data = KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash: None,
            start_position: None,
            blocks: pages
                .iter()
                .map(|(seq, p)| KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(*seq),
                    tokens_hash: LocalBlockHash(p.local),
                    mm_extra_info: None,
                })
                .collect(),
        });
        (data, pages.into_iter().map(|(_, p)| p.tokens).collect())
    }

    fn device_removed(cache: &HiCache, tokens: &[u64]) -> KvCacheEventData {
        let block_hashes = cache
            .pages(tokens, 0)
            .into_iter()
            .map(|(seq, _)| ExternalSequenceBlockHash(seq))
            .collect();
        KvCacheEventData::Removed(KvCacheRemoveData { block_hashes })
    }

    fn store(cache: &mut HiCache, tokens: &[u64]) {
        let (data, token_ids) = device_stored(cache, tokens);
        cache.on_device_event(&data, Some(&token_ids));
    }

    fn remove(cache: &mut HiCache, tokens: &[u64]) {
        let data = device_removed(cache, tokens);
        cache.on_device_event(&data, None);
    }

    /// `(stored?, blocks)` per host event.
    fn events(cache: &mut HiCache) -> Vec<(bool, usize)> {
        cache
            .drain_events()
            .into_iter()
            .map(|(data, _)| match data {
                KvCacheEventData::Stored(s) => (true, s.blocks.len()),
                KvCacheEventData::Removed(r) => (false, r.block_hashes.len()),
                KvCacheEventData::Cleared => unreachable!(),
            })
            .collect()
    }

    #[test]
    fn page_hashes_match_the_router() {
        let cache = HiCache::new(&args(8, WritePolicy::WriteThrough), PAGE, 1000, None);
        let tokens: Vec<u64> = (100..108).collect();
        let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
        let locals = compute_block_hash_for_seq(&tokens_u32, PAGE as u32, Default::default());
        let seqs = dynamo_kv_router::protocols::compute_seq_hash_for_block(&locals);
        let pages = cache.pages(&tokens, 0);
        assert_eq!(pages.iter().map(|(s, _)| *s).collect::<Vec<_>>(), seqs);
    }

    #[test]
    fn write_through_serves_pages_evicted_from_device() {
        let mut cache = HiCache::new(&args(8, WritePolicy::WriteThrough), PAGE, 1000, None);
        let tokens: Vec<u64> = (0..8).collect();
        store(&mut cache, &tokens);
        assert_eq!(events(&mut cache), vec![(true, 2)]);
        assert_eq!(cache.peek_prefix(&tokens, 0), 8);

        remove(&mut cache, &tokens);
        assert_eq!(events(&mut cache), vec![], "host copy stays");
        let hits = cache.take_prefix(&tokens, 0);
        assert_eq!(
            hits,
            PrefixHits {
                host_tokens: 8,
                peer_tokens: 0
            }
        );
        assert_eq!(cache.take_prefix(&tokens, 4).host_tokens, 4);
    }

    #[test]
    fn write_back_copies_on_device_removal_only() {
        let mut cache = HiCache::new(&args(8, WritePolicy::WriteBack), PAGE, 1000, None);
        let tokens: Vec<u64> = (0..8).collect();
        store(&mut cache, &tokens);
        assert_eq!(events(&mut cache), vec![]);
        remove(&mut cache, &tokens);
        assert_eq!(events(&mut cache), vec![(true, 2)]);
        assert_eq!(cache.take_prefix(&tokens, 0).host_tokens, 8);
    }

    #[test]
    fn full_host_evicts_lru_off_device_leaf_first() {
        let mut cache = HiCache::new(&args(2, WritePolicy::WriteThrough), PAGE, 1000, None);
        let a: Vec<u64> = (0..8).collect();
        let b: Vec<u64> = (100..108).collect();
        store(&mut cache, &a);
        // a is on device, so b's copy does not fit.
        store(&mut cache, &b);
        assert_eq!(events(&mut cache), vec![(true, 2)]);
        assert_eq!(cache.peek_prefix(&b, 0), 0);

        remove(&mut cache, &a);
        remove(&mut cache, &b);
        store(&mut cache, &b);
        // a's leaf page goes first, then its parent.
        assert_eq!(events(&mut cache), vec![(false, 2), (true, 2)]);
        assert_eq!(cache.peek_prefix(&a, 0), 0);
        assert_eq!(cache.peek_prefix(&b, 0), 8);
    }

    #[test]
    fn peers_serve_pages_above_the_prefetch_threshold() {
        let group = "baseten-mocker-peer-test";
        let peer_args = HiCacheArgs {
            peer_group: Some(group.into()),
            ..args(1024, WritePolicy::WriteThrough)
        };
        let mut a = HiCache::new(&peer_args, 64, 1000, None);
        let mut b = HiCache::new(&peer_args, 64, 1000, None);
        let tokens: Vec<u64> = (0..576).collect();
        store(&mut a, &tokens[..512]);

        assert_eq!(b.peek_prefix(&tokens, 0), 512);
        let hits = b.take_prefix(&tokens, 0);
        assert_eq!(
            hits,
            PrefixHits {
                host_tokens: 0,
                peer_tokens: 512
            }
        );
        assert_eq!(
            b.take_prefix(&tokens, 0).host_tokens,
            512,
            "now on b's host"
        );

        let short: Vec<u64> = (5000..5192).collect();
        store(&mut a, &short);
        let c = HiCache::new(&peer_args, 64, 1000, None);
        assert_eq!(
            c.peek_prefix(&short, 0),
            0,
            "192 tokens is below the threshold"
        );
    }

    #[test]
    fn validate_rejects_bad_sizes_bandwidths_and_bytes() {
        let ok = args(1, WritePolicy::WriteThrough);
        assert!(ok.validate(Some(1)).is_ok());
        assert!(ok.validate(None).is_err());
        assert!(ok.validate(Some(0)).is_err());
        let zero_bandwidth = HiCacheArgs {
            bandwidth_gb_per_sec: Some(0.0),
            ..ok.clone()
        };
        assert!(zero_bandwidth.validate(Some(1)).is_err());
        for (tp_size, dcp_size, mla) in [(0, 1, false), (1, 0, true), (8, 8, false)] {
            let bad = HiCacheArgs {
                tp_size,
                dcp_size,
                mla,
                ..ok.clone()
            };
            assert!(
                bad.validate(Some(1)).is_err(),
                "tp {tp_size} dcp {dcp_size} mla {mla}"
            );
        }
        let parsed: HiCacheArgs =
            serde_json::from_str(r#"{"size": 6, "write_policy": "write_back"}"#).unwrap();
        assert_eq!(parsed.write_policy, WritePolicy::WriteBack);
        assert!(serde_json::from_str::<HiCacheArgs>(r#"{"size": 6, "x": 1}"#).is_err());
    }

    #[test]
    fn page_evicted_before_flush_is_never_advertised() {
        // One host page; the two-page node's removal backs up a, then evicts
        // it for b, all before one flush.
        let group = "baseten-mocker-ghost-test";
        let peer_args = HiCacheArgs {
            peer_group: Some(group.into()),
            ..args(1, WritePolicy::WriteBack)
        };
        let mut owner = HiCache::new(&peer_args, PAGE, 1000, None);
        let reader = HiCache::new(&peer_args, PAGE, 1000, None);
        let tokens: Vec<u64> = (0..8).collect();
        store(&mut owner, &tokens);
        remove(&mut owner, &tokens);
        assert_eq!(events(&mut owner), vec![(true, 1)]);
        let pages = owner.pages(&tokens, 0);
        let hashes = owner.rank_keys(&pages.iter().map(|(s, _)| *s).collect::<Vec<_>>());
        let peer = reader.peer.as_ref().unwrap();
        assert_eq!(peer.contiguous_hits(&hashes[..1]), 0, "a is gone");
        assert_eq!(peer.contiguous_hits(&hashes[1..]), 1, "b is on host");
    }

    #[test]
    fn backing_up_a_resident_page_is_a_no_op() {
        let mut cache = HiCache::new(&args(8, WritePolicy::WriteThrough), PAGE, 1000, None);
        let tokens: Vec<u64> = (0..8).collect();
        let pages = cache.pages(&tokens, 0);
        for (seq, page) in &pages {
            assert!(cache.backup(*seq, page.clone()));
        }
        assert!(cache.backup(pages[1].0, pages[1].1.clone()));
        assert_eq!(cache.host.len(), 2);
        assert_eq!(cache.host_children.get(&pages[0].0), Some(&1));
        assert_eq!(cache.evictable.len(), 1, "one entry for the leaf");
    }

    #[test]
    fn deferred_peer_updates_wait_for_publish() {
        let group = "baseten-mocker-defer-test";
        let peer_args = HiCacheArgs {
            peer_group: Some(group.into()),
            ..args(1024, WritePolicy::WriteThrough)
        };
        let mut owner = HiCache::new(&peer_args, 64, 1000, None);
        owner.defer_peer_updates();
        let reader = HiCache::new(&peer_args, 64, 1000, None);
        let tokens: Vec<u64> = (0..576).collect();
        store(&mut owner, &tokens[..512]);
        assert_eq!(reader.peek_prefix(&tokens, 0), 0, "pass not over yet");
        owner.publish_peer_updates();
        assert_eq!(reader.peek_prefix(&tokens, 0), 512);
    }

    fn read_test_cache(group: &str, host_tokens: usize) -> HiCache {
        let peer_args = HiCacheArgs {
            size: (host_tokens * 1000) as f64 / 1e9,
            peer_group: Some(group.into()),
            peer_bandwidth_gb_per_sec: Some(1e-3), // 1 MB/s: 1 ms per token
            ..Default::default()
        };
        HiCache::new(&peer_args, 64, 1000, None)
    }

    #[test]
    fn admission_takes_only_the_pages_the_prefetch_read() {
        let mut owner = read_test_cache("prefetch-span-test", 1024);
        let mut reader = read_test_cache("prefetch-span-test", 1024);
        let tokens: Vec<u64> = (0..576).collect();
        store(&mut owner, &tokens[..256]);
        assert!(reader.prefetch_pending(1, &tokens, 0), "256 ms read");
        assert_eq!(reader.next_prefetch_wakeup_ms(), Some(PREFETCH_POLL_MS));
        store(&mut owner, &tokens[..512]);

        reader.set_now_ms(300.0);
        assert!(!reader.prefetch_pending(1, &tokens, 0));
        assert_eq!(reader.peek_prefix(&tokens, 0), 256);
        assert_eq!(reader.take_prefix(&tokens, 0).peer_tokens, 256);
    }

    #[test]
    fn prefetch_skips_peer_pages_the_host_tier_cannot_take() {
        let mut owner = read_test_cache("prefetch-room-test", 1024);
        let mut reader = read_test_cache("prefetch-room-test", 128);
        let tokens: Vec<u64> = (0..576).collect();
        store(&mut owner, &tokens[..512]);
        assert!(!reader.prefetch_pending(1, &tokens, 0), "nothing fits");
        assert_eq!(reader.take_prefix(&tokens, 0).peer_tokens, 0);
    }

    #[test]
    fn host_capacity_and_transfers_use_per_rank_bytes() {
        let model = 8000; // KV bytes per token over all layers and heads
        let geometry = |tp_size, dcp_size, mla| HiCacheArgs {
            size: 64.0 * 8000.0 / 1e9, // 64 tokens of the whole model
            tp_size,
            dcp_size,
            mla,
            ..Default::default()
        };
        let capacity = |a: &HiCacheArgs| HiCache::new(a, 4, model, None).capacity_pages * 4;
        assert_eq!(capacity(&geometry(1, 1, false)), 64);
        assert_eq!(
            capacity(&geometry(8, 1, false)),
            512,
            "GQA heads split over TP"
        );
        assert_eq!(
            capacity(&geometry(8, 1, true)),
            64,
            "MLA replicated over TP"
        );
        assert_eq!(capacity(&geometry(8, 8, true)), 512, "MLA split over DCP");
        let mla_tp8 = HiCache::new(&geometry(8, 1, true), 4, model, None);
        let gqa_tp8 = HiCache::new(&geometry(8, 1, false), 4, model, None);
        let gbps = Some(1e-6); // 1 KB/s
        assert_eq!(mla_tp8.transfer(1, gbps), Duration::from_secs(8));
        assert_eq!(gqa_tp8.transfer(1, gbps), Duration::from_secs(1));
    }

    #[test]
    fn peers_share_pages_only_with_the_same_kv_layout() {
        let cache = |group: &str, tp_size, dcp_size, mla| {
            let a = HiCacheArgs {
                peer_group: Some(group.into()),
                tp_size,
                dcp_size,
                mla,
                ..args(1024, WritePolicy::WriteThrough)
            };
            HiCache::new(&a, 64, 1000, None)
        };
        let tokens: Vec<u64> = (0..576).collect();
        let shares = |group: &str, owner: (usize, usize, bool), reader: (usize, usize, bool)| {
            let mut o = cache(group, owner.0, owner.1, owner.2);
            store(&mut o, &tokens[..512]);
            cache(group, reader.0, reader.1, reader.2).peek_prefix(&tokens, 0) == 512
        };
        assert!(shares("layout-gqa-same", (4, 1, false), (4, 1, false)));
        assert!(!shares("layout-gqa-tp", (4, 1, false), (2, 1, false)));
        assert!(
            shares("layout-mla-tp", (8, 1, true), (4, 1, true)),
            "MLA is TP-replicated"
        );
        assert!(!shares("layout-mla-dcp", (8, 8, true), (8, 4, true)));
        assert!(!shares("layout-mla-gqa", (1, 1, true), (1, 1, false)));
    }

    #[test]
    fn a_page_hits_only_when_every_rank_holds_it() {
        let group = "layout-tp-complete";
        let tp2 = HiCacheArgs {
            peer_group: Some(group.into()),
            tp_size: 2,
            ..args(1024, WritePolicy::WriteThrough)
        };
        let reader = HiCache::new(&tp2, 64, 1000, None);
        let tokens: Vec<u64> = (0..576).collect();
        let seqs: Vec<u64> = reader.pages(&tokens, 0).iter().map(|(s, _)| *s).collect();
        // Another replica whose rank 1 never published.
        let rank0: Vec<u64> = seqs
            .iter()
            .map(|&s| peer_key(s, reader.rank_tags[0]))
            .collect();
        let other = PeerIndex::new(group);
        other.publish(&rank0);
        assert_eq!(reader.peek_prefix(&tokens, 0), 0);
        let rank1: Vec<u64> = seqs
            .iter()
            .map(|&s| peer_key(s, reader.rank_tags[1]))
            .collect();
        other.publish(&rank1[..4]);
        assert_eq!(reader.peek_prefix(&tokens, 0), 256, "min over ranks");
    }

    #[test]
    fn prefetch_counts_off_device_host_chains_as_room() {
        let mut owner = read_test_cache("prefetch-chain-test", 1024);
        let mut reader = read_test_cache("prefetch-chain-test", 512);
        // The reader's host is full of one off-device chain: only its leaf is
        // evictable right now, but eviction frees the whole chain.
        let old: Vec<u64> = (10_000..10_512).collect();
        store(&mut reader, &old);
        remove(&mut reader, &old);
        let tokens: Vec<u64> = (0..576).collect();
        store(&mut owner, &tokens[..512]);
        assert!(reader.prefetch_pending(1, &tokens, 0), "512 tokens to read");
        reader.set_now_ms(1000.0);
        assert!(!reader.prefetch_pending(1, &tokens, 0));
        assert_eq!(reader.take_prefix(&tokens, 0).peer_tokens, 512);
    }
}
