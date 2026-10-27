// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! G2PB peer directory for live SGLang mockers: the real G2Direct replica
//! index and publisher over the event plane, without NIXL. Mocker replicas
//! sharing `DYNAMO_MODEL_NAMESPACE` advertise host pages to and find pages on
//! each other exactly as G2Direct replicas do; the page read is simulated.
//!
//! shortcut: no discovery-driven purge, so pages of a replica that dies without
//! shutting down stay visible until it restarts.

use std::sync::Arc;

use anyhow::Result;
use dynamo_g2direct::peer_index::{
    ReplicaBlock, ReplicaIndex, ReplicaPublisher, ReplicaSource, ReplicaSourceKey,
    SERVING_G2_INDEX_COMPONENT, baseten_row_from_env, serving_g2_namespace_from_env,
};
use dynamo_runtime::component::Component;
use dynamo_runtime::traits::DistributedRuntimeProvider;

use crate::hicache::HiCacheArgs;
use crate::peer::{PeerDirectory, PeerDirectoryHandle};

/// A fleet-wide peer directory for one mocker DP rank, when `hicache` asks for
/// G2PB and the mocker runs under a distributed runtime.
pub async fn peer_directory(
    hicache: Option<&HiCacheArgs>,
    component: Option<&Component>,
    dp_rank: u32,
    page_size: usize,
    kv_bytes_per_token: Option<usize>,
) -> Option<PeerDirectoryHandle> {
    let (Some(component), Some(bytes_per_token)) = (component, kv_bytes_per_token) else {
        return None;
    };
    let hicache = hicache?;
    hicache.peer_group.as_ref()?;
    let page_bytes = (hicache.rank_kv_bytes(bytes_per_token) * page_size) as u64;
    match G2pbPeerDirectory::new(component, dp_rank, page_size as u32, page_bytes).await {
        Ok(directory) => Some(PeerDirectoryHandle(Arc::new(directory))),
        Err(e) => {
            tracing::error!("Failed to start the G2PB peer directory for dp_rank {dp_rank}: {e}");
            None
        }
    }
}

struct G2pbPeerDirectory {
    publisher: Arc<ReplicaPublisher>,
    index: Arc<ReplicaIndex>,
    replica_id: u64,
    page_size: u32,
    page_bytes: u64,
}

impl G2pbPeerDirectory {
    /// Each DP rank is its own replica so ranks of one worker read each other.
    async fn new(
        component: &Component,
        dp_rank: u32,
        page_size: u32,
        page_bytes: u64,
    ) -> Result<Self> {
        let drt = component.drt();
        let index_component =
            serving_g2_namespace_from_env(drt)?.component(SERVING_G2_INDEX_COMPONENT)?;
        let instance_id = drt.connection_id();
        let replica_id = dynamo_tokens::compute_hash_v2(
            &[instance_id.to_le_bytes(), u64::from(dp_rank).to_le_bytes()].concat(),
            0,
        );
        let source = ReplicaSource {
            key: ReplicaSourceKey {
                instance_id,
                tp_rank: dp_rank,
            },
            replica_id: Some(replica_id),
            tp_world_size: 1,
            incarnation: uuid::Uuid::new_v4(),
            block_size: page_size,
            row: baseten_row_from_env()?,
        };
        Ok(Self {
            publisher: ReplicaPublisher::new(&index_component, source).await?,
            index: ReplicaIndex::subscribe(&index_component),
            replica_id,
            page_size,
            page_bytes,
        })
    }
}

impl PeerDirectory for G2pbPeerDirectory {
    fn publish(&self, hashes: &[u64]) {
        let blocks = hashes
            .iter()
            .map(|&key| ReplicaBlock {
                key,
                size_bytes: self.page_bytes,
                checksum: None,
            })
            .collect();
        if let Err(error) = self.publisher.upsert(blocks) {
            tracing::warn!(%error, "G2PB mocker publish dropped");
        }
    }

    fn withdraw(&self, hashes: &[u64]) {
        if let Err(error) = self.publisher.remove(hashes.to_vec()) {
            tracing::warn!(%error, "G2PB mocker withdraw dropped");
        }
    }

    fn contiguous_hits(&self, hashes: &[u64]) -> usize {
        remote_prefix(&self.index, hashes, self.page_size, self.replica_id)
    }
}

/// Leading `hashes` held by a replica other than `mine`, as G2Direct's lookup
/// counts them.
fn remote_prefix(index: &ReplicaIndex, hashes: &[u64], page_size: u32, mine: u64) -> usize {
    index
        .batch_locations(hashes, Some(page_size))
        .iter()
        .take_while(|locations| {
            locations
                .iter()
                .any(|l| l.source.replica_id() != Some(mine))
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_g2direct::peer_index::{ReplicaEvent, ReplicaEventKind};

    fn upsert(replica_id: u64, page_size: u32, keys: &[u64]) -> ReplicaEvent {
        ReplicaEvent {
            source: ReplicaSource {
                key: ReplicaSourceKey {
                    instance_id: replica_id,
                    tp_rank: 0,
                },
                replica_id: Some(replica_id),
                tp_world_size: 1,
                incarnation: uuid::Uuid::from_u128(u128::from(replica_id)),
                block_size: page_size,
                row: None,
            },
            revision: 1,
            kind: ReplicaEventKind::Upsert(
                keys.iter()
                    .map(|&key| ReplicaBlock {
                        key,
                        size_bytes: 1,
                        checksum: None,
                    })
                    .collect(),
            ),
        }
    }

    #[test]
    fn remote_prefix_skips_own_pages_and_other_page_sizes() {
        let index = ReplicaIndex::new();
        index.apply(upsert(1, 64, &[10, 11]));
        index.apply(upsert(2, 64, &[10, 11, 12]));
        index.apply(upsert(3, 32, &[13]));

        assert_eq!(remote_prefix(&index, &[10, 11, 12, 13], 64, 1), 3);
        assert_eq!(remote_prefix(&index, &[10, 11, 12], 64, 2), 2);
        assert_eq!(remote_prefix(&index, &[13], 64, 1), 0);
    }
}
