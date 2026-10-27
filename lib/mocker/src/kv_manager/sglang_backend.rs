// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang KV manager — wraps [`RadixCache`] with request-level lifecycle
//! operations and KV event publishing.
//!
//! Like SGLang, KV events describe radix-tree contents: pages are stored when
//! inserted into the tree and removed when evicted.

use crate::cache::radix_cache::{NodeId, RadixCache};
use crate::common::kv_cache_trace;
use crate::common::protocols::{KvEventPublishers, MockEngineArgs};
use baseten_mocker::{HiCache, PrefixHits};
use dynamo_kv_router::protocols::{KvCacheEvent, KvCacheEventData, StorageTier};

/// Result of `allocate_for_request`.
pub struct AllocResult {
    /// Number of tokens matched from the prefix cache.
    pub prefix_len: usize,
    /// Pool token indices for the allocated input (1 per token).
    pub kv_indices: Vec<usize>,
    /// The deepest matched node in the radix tree (used for lock/unlock).
    /// This is the prefix match point, not the new tokens — new tokens are
    /// only in kv_indices and get inserted into the tree on completion.
    pub last_node: NodeId,
    /// Baseten HiCache: part of `prefix_len` served from host and peers.
    pub hicache_hits: PrefixHits,
}

pub struct SglangKvManager {
    cache: RadixCache,
    kv_event_publishers: KvEventPublishers,
    dp_rank: u32,
    next_event_id: u64,
    /// Baseten: SGLang HiCache host tier and G2PB peers.
    hicache: Option<HiCache>,
}

impl SglangKvManager {
    pub fn new(
        total_tokens: usize,
        page_size: usize,
        kv_event_publishers: KvEventPublishers,
        dp_rank: u32,
    ) -> Self {
        Self {
            cache: RadixCache::new(total_tokens, page_size),
            kv_event_publishers,
            dp_rank,
            next_event_id: 0,
            hicache: None,
        }
    }

    /// Baseten: enable the HiCache host tier configured by `sglang.hicache`.
    pub fn with_hicache(mut self, args: &MockEngineArgs) -> Self {
        let hicache = args.sglang.as_ref().and_then(|s| s.hicache.as_ref());
        if let (Some(hicache), Some(bytes)) = (hicache, args.kv_bytes_per_token) {
            let page_size = self.cache.page_size();
            let peer = args.peer_directory.as_ref();
            self.hicache = Some(HiCache::new(hicache, page_size, bytes, peer));
        }
        self
    }

    pub fn cache(&self) -> &RadixCache {
        &self.cache
    }

    pub fn cache_mut(&mut self) -> &mut RadixCache {
        &mut self.cache
    }

    /// Try to allocate KV cache for a new request.
    /// Returns `None` if the pool doesn't have enough token slots (OOM).
    pub fn allocate_for_request(&mut self, token_ids: &[u64]) -> Option<AllocResult> {
        self.allocate_for_request_capped(token_ids, token_ids.len())
    }

    /// [`Self::allocate_for_request`], reusing at most `max_prefix` cached
    /// tokens.
    pub fn allocate_for_request_capped(
        &mut self,
        token_ids: &[u64],
        max_prefix: usize,
    ) -> Option<AllocResult> {
        let (mut prefix_len, mut last_node) = self.cache.match_prefix(&token_ids[..max_prefix]);
        // Host loads and new tokens all fit in this.
        let needed = token_ids.len() - prefix_len;
        let available = self.cache.available_tokens();
        if available < needed {
            self.cache.inc_lock_ref(last_node);
            self.cache.evict(needed - available);
            self.cache.dec_lock_ref(last_node);
            if self.cache.available_tokens() < needed {
                self.flush_events();
                return None;
            }
        }
        let hicache_hits =
            self.load_hicache_hits(&token_ids[..max_prefix], &mut prefix_len, &mut last_node);

        let mut kv_indices = self.collect_path_indices(last_node);
        let new_tokens = token_ids.len() - prefix_len;
        let new_indices = self
            .cache
            .token_pool
            .allocate(new_tokens)
            .expect("slots checked");
        kv_indices.extend_from_slice(&new_indices);

        self.cache.inc_lock_ref(last_node);
        self.flush_events();
        self.log_trace("allocation", new_tokens);

        Some(AllocResult {
            prefix_len,
            kv_indices,
            last_node,
            hicache_hits,
        })
    }

    /// Baseten: load the pages after the device prefix that HiCache serves
    /// from host or peers into the tree, extending the prefix.
    fn load_hicache_hits(
        &mut self,
        token_ids: &[u64],
        prefix_len: &mut usize,
        last_node: &mut NodeId,
    ) -> PrefixHits {
        self.flush_events();
        let Some(hicache) = self.hicache.as_mut() else {
            return PrefixHits::default();
        };
        let hits = hicache.take_prefix(token_ids, *prefix_len);
        let loaded = hits.host_tokens + hits.peer_tokens;
        if loaded > 0 {
            let end = *prefix_len + loaded;
            let mut indices = self.collect_path_indices(*last_node);
            indices.extend(
                self.cache
                    .token_pool
                    .allocate(loaded)
                    .expect("slots checked"),
            );
            self.cache.insert(&token_ids[..end], &indices);
            (*prefix_len, *last_node) = self.cache.match_prefix(&token_ids[..end]);
        }
        hits
    }

    /// Baseten HiCache, if enabled.
    pub fn hicache_mut(&mut self) -> Option<&mut HiCache> {
        self.hicache.as_mut()
    }

    /// Tokens of `token_ids` a fresh request would reuse, including Baseten
    /// HiCache host and peer pages, or `None` while its peer read is pending.
    pub fn peek_prefix_len(&mut self, request: u128, token_ids: &[u64]) -> Option<usize> {
        let device = self.cache.prefix_match_len(token_ids);
        let Some(h) = self.hicache.as_mut() else {
            return Some(device);
        };
        let pending = h.prefetch_pending(request, token_ids, device);
        (!pending).then(|| device + h.peek_prefix(token_ids, device))
    }

    /// Continue an in-flight request from an already materialized prefix.
    ///
    /// This is used by chunked-prefill continuation where the request still
    /// owns token slots for a prefix that may extend past the radix-tree's
    /// page-aligned cached prefix.
    pub fn allocate_after_prefix(
        &mut self,
        token_ids: &[u64],
        prefix_len: usize,
        prefix_indices: &[usize],
        last_node: NodeId,
    ) -> Option<AllocResult> {
        let new_tokens = token_ids.len().saturating_sub(prefix_len);
        let new_indices = self.cache.token_pool.allocate(new_tokens)?;

        let mut kv_indices = prefix_indices[..prefix_len].to_vec();
        kv_indices.extend_from_slice(&new_indices);

        self.cache.inc_lock_ref(last_node);
        self.log_trace("allocation", new_tokens);

        Some(AllocResult {
            prefix_len,
            kv_indices,
            last_node,
            hicache_hits: PrefixHits::default(),
        })
    }

    /// Cache a completed request's full sequence into the radix tree.
    ///
    /// Inserts the full token sequence so future requests can reuse it,
    /// then unlocks the path.
    pub fn cache_finished_req(
        &mut self,
        token_ids: &[u64],
        kv_indices: &[usize],
        last_node: NodeId,
    ) {
        // Like SGLang, free slots computed for tokens the tree already holds
        // (a fully cached prompt recomputes its last page).
        let (_, cached_node) = self.cache.match_prefix(token_ids);
        let cached = self.collect_path_indices(cached_node);
        let duplicates: Vec<usize> = (cached.iter().zip(kv_indices))
            .filter(|(tree, own)| tree != own)
            .map(|(_, &own)| own)
            .collect();
        self.free_indices(&duplicates);
        self.cache.insert(token_ids, kv_indices);
        self.cache.dec_lock_ref(last_node);
        self.flush_events();
    }

    /// Cache a partial sequence after a chunked prefill step.
    ///
    /// Inserts the partial sequence, then transfers the lock from the old
    /// path to the new (extended) path. The request is still active, so the
    /// new deepest node stays locked.
    ///
    /// Returns the new `last_node` that the caller should use for
    /// subsequent calls.
    pub fn cache_unfinished_req(
        &mut self,
        token_ids: &[u64],
        kv_indices: &[usize],
        last_node: NodeId,
    ) -> NodeId {
        self.cache.insert(token_ids, kv_indices);

        // Find the new deepest node after insert
        let (_, new_last_node) = self.cache.match_prefix(token_ids);

        // Transfer lock: release old path, protect new path
        self.cache.dec_lock_ref(last_node);
        self.cache.inc_lock_ref(new_last_node);
        self.flush_events();

        new_last_node
    }

    /// Allocate a single token slot for decode output.
    pub fn allocate_decode_token(&mut self) -> Option<usize> {
        let indices = self.cache.token_pool.allocate(1)?;
        self.log_trace("allocation", 1);
        Some(indices[0])
    }

    /// Free a request without caching (e.g., aborted request).
    ///
    /// Unlocks the path without inserting into the tree.
    pub fn free_request(&mut self, last_node: NodeId) {
        self.cache.dec_lock_ref(last_node);
    }

    /// Return request-owned token slots (never in the tree) to the free pool.
    pub fn free_indices(&mut self, indices: &[usize]) {
        if indices.is_empty() {
            return;
        }
        self.cache.token_pool.free(indices);
        self.log_trace("free", indices.len());
    }

    /// Collect token indices from the matched prefix path by walking root→last_node.
    fn collect_path_indices(&self, last_node: NodeId) -> Vec<usize> {
        let mut path = Vec::new();
        let mut current = last_node;
        while let Some(parent) = self.cache.node(current).parent {
            path.push(current);
            current = parent;
        }
        path.iter()
            .rev()
            .flat_map(|&id| self.cache.node(id).value.iter().copied())
            .collect()
    }

    /// Evict tokens from the cache, publish BlockRemoved events, and log a trace.
    pub fn evict(&mut self, num_tokens: usize) {
        let (evicted, _) = self.cache.evict(num_tokens);
        self.flush_events();
        self.log_trace("eviction", evicted);
    }

    fn log_trace(&self, event: &str, num_tokens: usize) {
        kv_cache_trace::log_sglang_trace(&kv_cache_trace::SglangCacheState {
            event,
            dp_rank: self.dp_rank,
            num_tokens,
            page_size: self.cache.page_size(),
            available_tokens: self.cache.available_tokens(),
            evictable_tokens: self.cache.evictable_size,
            protected_tokens: self.cache.protected_size,
            total_tokens: self.cache.total_tokens(),
        });
    }

    /// Publish queued tree changes, and Baseten HiCache's host-tier changes,
    /// to the router.
    fn flush_events(&mut self) {
        for (data, token_ids) in std::mem::take(&mut self.cache.events) {
            if let Some(hicache) = self.hicache.as_mut() {
                hicache.on_device_event(&data, token_ids.as_deref());
            }
            self.publish(data, token_ids, StorageTier::Device);
        }
        let host_events = self.hicache.as_mut().map(HiCache::drain_events);
        for (data, token_ids) in host_events.into_iter().flatten() {
            self.publish(data, token_ids, StorageTier::HostPinned);
        }
    }

    fn publish(
        &mut self,
        data: KvCacheEventData,
        token_ids: Option<Vec<Vec<u32>>>,
        tier: StorageTier,
    ) {
        if self.kv_event_publishers.is_empty() {
            return;
        }
        let event = KvCacheEvent {
            event_id: self.next_event_id,
            data,
            dp_rank: self.dp_rank,
        };
        self.next_event_id += 1;
        if let Err(e) =
            self.kv_event_publishers
                .publish_with_storage_tier(event, token_ids.as_deref(), tier)
        {
            tracing::warn!("Failed to publish SGLang KV event: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::Mutex;

    use crate::common::protocols::KvCacheEventSink;

    struct MockSink {
        events: Mutex<Vec<KvCacheEvent>>,
    }

    impl MockSink {
        fn new() -> Self {
            Self {
                events: Mutex::new(Vec::new()),
            }
        }

        fn event_count(&self) -> usize {
            self.events.lock().unwrap().len()
        }

        fn clone_events(&self) -> Vec<KvCacheEvent> {
            self.events.lock().unwrap().clone()
        }
    }

    impl KvCacheEventSink for MockSink {
        fn publish(&self, event: KvCacheEvent) -> anyhow::Result<()> {
            self.events.lock().unwrap().push(event);
            Ok(())
        }
    }

    fn manager_with_sink(total: usize, page: usize) -> (SglangKvManager, Arc<MockSink>) {
        let sink = Arc::new(MockSink::new());
        let mgr = SglangKvManager::new(
            total,
            page,
            KvEventPublishers::new(Some(sink.clone()), None),
            0,
        );
        (mgr, sink)
    }

    #[test]
    fn test_allocate_cache_miss() {
        let mut mgr = SglangKvManager::new(100, 1, KvEventPublishers::default(), 0);

        let result = mgr.allocate_for_request(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(result.prefix_len, 0);
        assert_eq!(result.kv_indices.len(), 5);
        assert_eq!(mgr.cache().token_pool.available(), 95);
    }

    #[test]
    fn test_allocate_cache_hit() {
        let mut mgr = SglangKvManager::new(100, 1, KvEventPublishers::default(), 0);

        // First request: allocate and cache
        let r1 = mgr.allocate_for_request(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(r1.kv_indices.len(), 5); // 5 pages (page_size=1)
        mgr.cache_finished_req(&[1, 2, 3, 4, 5], &r1.kv_indices, r1.last_node);

        // Second request with shared prefix
        let r2 = mgr.allocate_for_request(&[1, 2, 3, 4, 5, 6, 7]).unwrap();
        assert_eq!(r2.prefix_len, 5);
        assert_eq!(r2.kv_indices.len(), 7); // 5 reused + 2 new pages
        assert_eq!(mgr.cache().token_pool.available(), 93); // 100 - 5 - 2
    }

    #[test]
    fn test_free_request_without_caching() {
        let mut mgr = SglangKvManager::new(100, 1, KvEventPublishers::default(), 0);

        let result = mgr.allocate_for_request(&[1, 2, 3]).unwrap();
        mgr.free_request(result.last_node);

        // Path is unlocked, tokens still allocated in pool
        assert_eq!(mgr.cache().protected_size, 0);
    }

    #[test]
    fn test_event_publishing() {
        let (mut mgr, sink) = manager_with_sink(100, 1);

        // Like SGLang, pages are published when inserted into the tree.
        let r = mgr.allocate_for_request(&[1, 2, 3]).unwrap();
        assert_eq!(sink.event_count(), 0);
        mgr.cache_finished_req(&[1, 2, 3], &r.kv_indices, r.last_node);
        let events = sink.clone_events();
        let [
            KvCacheEvent {
                data: KvCacheEventData::Stored(store),
                ..
            },
        ] = events.as_slice()
        else {
            panic!("expected one stored event, got {events:?}");
        };
        assert_eq!(store.blocks.len(), 3);

        // Second request with full cache hit → no new events
        let r2 = mgr.allocate_for_request(&[1, 2, 3]).unwrap();
        assert_eq!(r2.prefix_len, 3);
        assert_eq!(sink.event_count(), 1);
    }

    #[test]
    fn test_page_hashes_match_router_and_removals_reuse_them() {
        let (mut mgr, sink) = manager_with_sink(100, 4);
        let tokens: Vec<u64> = (100..108).collect();
        let r = mgr.allocate_for_request(&tokens).unwrap();
        mgr.cache_finished_req(&tokens, &r.kv_indices, r.last_node);
        mgr.evict(8);

        let events = sink.clone_events();
        let KvCacheEventData::Stored(store) = &events[0].data else {
            panic!("expected stored event");
        };
        let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
        let router = dynamo_kv_router::protocols::compute_block_hash_for_seq(
            &tokens_u32,
            4,
            Default::default(),
        );
        let locals: Vec<_> = store.blocks.iter().map(|b| b.tokens_hash).collect();
        assert_eq!(locals, router);
        let seqs: Vec<_> = store.blocks.iter().map(|b| b.block_hash.0).collect();
        assert_eq!(
            seqs,
            dynamo_kv_router::protocols::compute_seq_hash_for_block(&router)
        );

        let KvCacheEventData::Removed(remove) = &events[1].data else {
            panic!("expected removed event");
        };
        let stored: Vec<_> = store.blocks.iter().map(|b| b.block_hash).collect();
        assert_eq!(remove.block_hashes, stored);
    }

    #[test]
    fn test_allocate_oom() {
        let mut mgr = SglangKvManager::new(3, 1, KvEventPublishers::default(), 0);

        let _r = mgr.allocate_for_request(&[1, 2, 3]).unwrap();
        // Pool is full
        let result = mgr.allocate_for_request(&[4, 5, 6]);
        assert!(result.is_none());
    }

    #[test]
    fn test_chunked_prefill_parent_hash() {
        let (mut mgr, sink) = manager_with_sink(32, 1);
        let tokens = [11, 22, 33, 44, 55, 66];

        let alloc1 = mgr.allocate_for_request(&tokens[..3]).unwrap();
        let last = mgr.cache_unfinished_req(&tokens[..3], &alloc1.kv_indices, alloc1.last_node);
        let alloc2 = mgr
            .allocate_after_prefix(&tokens, 3, &alloc1.kv_indices, last)
            .unwrap();
        mgr.free_request(last);
        mgr.cache_unfinished_req(&tokens, &alloc2.kv_indices, alloc2.last_node);

        let events = sink.clone_events();
        assert_eq!(events.len(), 2, "expected two stored events");
        let KvCacheEventData::Stored(store1) = &events[0].data else {
            panic!("expected first event to be Stored");
        };
        let KvCacheEventData::Stored(store2) = &events[1].data else {
            panic!("expected second event to be Stored");
        };
        assert!(store1.parent_hash.is_none());
        assert_eq!(store2.parent_hash, Some(store1.blocks[2].block_hash));
        assert_eq!(
            store2.blocks.len(),
            3,
            "second chunk should only emit new blocks"
        );
    }
}
