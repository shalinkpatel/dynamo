// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

impl ConcurrentRadixTreeCompressed {
    #[cfg(test)]
    pub(crate) fn run_cleanup_for_test(&self) {
        self.sweep_stale_children();
    }

    pub(super) fn sweep_stale_children(&self) {
        let mut queue = VecDeque::from([self.root.clone()]);
        queue.extend(self.anchor_nodes.iter().map(|entry| entry.value().clone()));
        let mut edges = Vec::new();

        while let Some(parent) = queue.pop_front() {
            let children = parent.child_edges_snapshot();
            for (key, child) in children {
                queue.push_back(child.clone());
                edges.push(CleanupEdge {
                    parent: Arc::downgrade(&parent),
                    key,
                    child: Arc::downgrade(&child),
                });
            }
        }

        for edge in edges.into_iter().rev() {
            let Some(parent) = edge.parent.upgrade() else {
                continue;
            };
            let Some(child) = edge.child.upgrade() else {
                continue;
            };
            parent.remove_child_if_stale_leaf(edge.key, &child);
        }
    }

    fn dead_leaf_edge(
        &self,
        worker_lookup: &WorkerLookup,
        node: &SharedNode,
    ) -> Option<CleanupEdge> {
        let (key, parent_tail) = node.dead_leaf_key_and_tail()?;
        let parent = if self.root.has_child(key, node) {
            self.root.clone()
        } else {
            let tail = parent_tail?;
            let parent = self
                .anchor_nodes
                .get(&tail)
                .map(|entry| entry.value().clone())
                .or_else(|| worker_lookup.get(&tail).cloned())?;
            if !parent.contains_edge_hash(tail) || !parent.has_child(key, node) {
                return None;
            }
            parent
        };
        Some(CleanupEdge {
            parent: Arc::downgrade(&parent),
            key,
            child: Arc::downgrade(node),
        })
    }

    fn push_pending_unlink(&self, lookup: &mut LaneLookup, edge: CleanupEdge) {
        const PENDING_LIMIT: usize = 4096;
        lookup.pending_unlinks.push_back(edge);
        if lookup.pending_unlinks.len() >= PENDING_LIMIT {
            self.drain_pending_unlinks(lookup, PENDING_LIMIT / 4);
        }
    }

    pub(super) fn queue_split_dead_leaf(
        &self,
        lookup: &mut LaneLookup,
        prefix: &SharedNode,
        suffix: &SharedNode,
    ) {
        if let Some((key, _)) = suffix.dead_leaf_key_and_tail() {
            self.push_pending_unlink(
                lookup,
                CleanupEdge {
                    parent: Arc::downgrade(prefix),
                    key,
                    child: Arc::downgrade(suffix),
                },
            );
        }
    }

    pub(super) fn drain_pending_unlinks(&self, lookup: &mut LaneLookup, limit: usize) {
        for _ in 0..limit.min(lookup.pending_unlinks.len()) {
            let edge = lookup.pending_unlinks.pop_front().unwrap();
            let (Some(parent), Some(child)) = (edge.parent.upgrade(), edge.child.upgrade()) else {
                continue;
            };
            parent.remove_child_if_stale_leaf(edge.key, &child);
        }
    }

    /// Apply a remove operation (eviction).
    ///
    /// For each evicted block hash, finds its position in the node via `edge_index` (O(1)).
    /// Updates the worker's match index without splitting the tree:
    /// - `pos >= current_cutoff`: no-op (already beyond coverage)
    /// - `pos < current_cutoff`: `new_cutoff = pos`; moves worker to `worker_cutoffs`
    ///   or removes entirely if `new_cutoff == 0`.
    ///
    /// Lookup entries for the newly uncovered suffix are removed eagerly so
    /// later duplicate remove events fast-path through the missing-hash case.
    pub(super) fn apply_removed(
        &self,
        lookup: &mut LaneLookup,
        worker: WorkerWithDpRank,
        op: KvCacheRemoveData,
        id: u64,
    ) -> Result<(), KvCacheEventError> {
        if !lookup.contains_key(&worker) {
            return Err(KvCacheEventError::BlockNotFound);
        }

        'outer: for block_hash in op.block_hashes {
            let mut cur_node = match self.resolve_lookup(
                lookup,
                worker,
                block_hash,
                LookupRepairDirection::TowardHead,
            ) {
                Some(n) => n,
                None => {
                    if let Some(worker_lookup) = lookup.get_mut(&worker) {
                        worker_lookup.remove(&block_hash);
                    }
                    tracing::debug!(
                        worker_id = worker.worker_id.to_string(),
                        dp_rank = worker.dp_rank,
                        id,
                        block_hash = ?block_hash,
                        "Block not found during remove; skipping"
                    );
                    continue;
                }
            };

            loop {
                match cur_node.remove_worker_for_hash(worker, block_hash) {
                    Some(outcome) => {
                        if let Some(wl) = lookup.get_mut(&worker) {
                            for hash in outcome.stale_hashes {
                                wl.remove(&hash);
                            }
                        }
                        if let Some(edge) = lookup
                            .get(&worker)
                            .and_then(|wl| self.dead_leaf_edge(wl, &cur_node))
                        {
                            self.push_pending_unlink(lookup, edge);
                        }
                        continue 'outer;
                    }
                    None => {
                        // Hash was moved to a descendant by a concurrent split.
                        match Self::find_in_subtree(&cur_node, block_hash) {
                            Some(resolved) => {
                                self.repair_lookup_for_resolved_node(
                                    lookup,
                                    block_hash,
                                    &resolved,
                                    LookupRepairDirection::TowardHead,
                                );
                                #[cfg(feature = "bench")]
                                self.bench_metrics
                                    .lookup_repair_scans
                                    .fetch_add(1, Ordering::Relaxed);
                                cur_node = resolved;
                                // Retry the inner loop with the resolved node.
                            }
                            None => {
                                // Hash not found anywhere -- evicted by a concurrent clear.
                                tracing::debug!(
                                    worker_id = worker.worker_id.to_string(),
                                    dp_rank = worker.dp_rank,
                                    id,
                                    block_hash = ?block_hash,
                                    "Block not found in subtree during remove; skipping"
                                );
                                if let Some(wl) = lookup.get_mut(&worker) {
                                    wl.remove(&block_hash);
                                }
                                continue 'outer;
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    pub(super) fn remove_or_clear_worker_blocks(
        &self,
        lookup: &mut LaneLookup,
        worker_id: WorkerId,
        keep_worker: bool,
    ) {
        let workers: Vec<WorkerWithDpRank> = lookup
            .keys()
            .filter(|w| w.worker_id == worker_id)
            .copied()
            .collect();

        for worker in workers {
            if let Some(worker_lookup) = lookup.remove(&worker) {
                let mut seen = FxHashSet::<usize>::default();
                for node in worker_lookup.values() {
                    let ptr = Arc::as_ptr(node) as usize;
                    if !seen.insert(ptr) {
                        continue;
                    }
                    node.drop_worker(worker);
                    if let Some(edge) = self.dead_leaf_edge(&worker_lookup, node) {
                        self.push_pending_unlink(lookup, edge);
                    }
                }

                if keep_worker {
                    lookup.insert(worker, FxHashMap::default());
                }
            }
        }
    }

    pub(super) fn remove_worker_dp_rank(
        &self,
        lookup: &mut LaneLookup,
        worker_id: WorkerId,
        dp_rank: DpRank,
    ) {
        let key = WorkerWithDpRank { worker_id, dp_rank };
        if let Some(worker_lookup) = lookup.remove(&key) {
            let mut seen = FxHashSet::<usize>::default();
            for node in worker_lookup.values() {
                let ptr = Arc::as_ptr(node) as usize;
                if !seen.insert(ptr) {
                    continue;
                }
                node.drop_worker(key);
                if let Some(edge) = self.dead_leaf_edge(&worker_lookup, node) {
                    self.push_pending_unlink(lookup, edge);
                }
            }
        }
    }

    pub(super) fn clear_all_blocks(&self, lookup: &mut LaneLookup, worker_id: WorkerId) {
        self.remove_or_clear_worker_blocks(lookup, worker_id, true);
    }
}
