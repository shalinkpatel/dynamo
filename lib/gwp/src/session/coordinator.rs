// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use dynamo_llm::{
    protocols::common::extensions::SessionAffinityId,
    session_affinity::{AffinityCoordinator, AffinityTarget, MAX_SESSION_AFFINITY_ID_BYTES},
};

fn coordinator_session_id(session_id: &str) -> SessionAffinityId {
    if session_id.len() > MAX_SESSION_AFFINITY_ID_BYTES {
        SessionAffinityId::new(blake3::hash(session_id.as_bytes()).to_hex().to_string())
    } else {
        SessionAffinityId::new(session_id)
    }
}

// TTL is fixed when the coordinator is constructed; config reload preserves it.
#[async_trait]
impl AffinityStore for AffinityCoordinator {
    fn backend_name(&self) -> &'static str {
        "pubsub"
    }

    async fn peek(&self, session_id: &str) -> anyhow::Result<Option<WorkerWithDpRank>> {
        Ok(self
            .query_target(&coordinator_session_id(session_id), None)?
            .map(|target| WorkerWithDpRank {
                worker_id: target.worker_id,
                dp_rank: target.dp_rank.unwrap_or_default(),
            }))
    }

    async fn get(&self, session_id: &str) -> anyhow::Result<Option<WorkerWithDpRank>> {
        let session_id = coordinator_session_id(session_id);
        let target = self.query_target(&session_id, None)?;
        if let Some(target) = target {
            self.record_confirmed_target(&session_id, target)?;
        }
        Ok(target.map(|target| WorkerWithDpRank {
            worker_id: target.worker_id,
            dp_rank: target.dp_rank.unwrap_or_default(),
        }))
    }

    async fn put(
        &self,
        session_id: &str,
        worker: WorkerWithDpRank,
        _ttl: Duration,
    ) -> anyhow::Result<()> {
        self.record_confirmed_target(
            &coordinator_session_id(session_id),
            AffinityTarget {
                worker_id: worker.worker_id,
                dp_rank: Some(worker.dp_rank),
            },
        )
    }

    async fn remove(&self, _session_id: &str) -> anyhow::Result<bool> {
        anyhow::bail!("pubsub session affinity has no distributed delete; bindings expire by TTL")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn external_session_ids_preserve_short_ids_and_hash_oversized_ids() {
        let ttl = Duration::from_secs(10);
        let store = AffinityCoordinator::new(ttl).unwrap();
        let ids = ["x".repeat(256), "x".repeat(257), "é".repeat(129)];
        assert_eq!(coordinator_session_id(&ids[0]).as_str(), ids[0]);
        for id in &ids[1..] {
            assert_eq!(coordinator_session_id(id).as_str().len(), 64);
        }
        for (index, id) in ids.iter().enumerate() {
            store
                .put(
                    id,
                    WorkerWithDpRank {
                        worker_id: index as u64,
                        dp_rank: 0,
                    },
                    ttl,
                )
                .await
                .unwrap();
        }
        for (index, id) in ids.iter().enumerate() {
            let expected = Some(WorkerWithDpRank {
                worker_id: index as u64,
                dp_rank: 0,
            });
            assert_eq!(store.peek(id).await.unwrap(), expected);
            assert_eq!(store.get(id).await.unwrap(), expected);
        }
    }

    #[tokio::test(start_paused = true)]
    async fn confirmed_binding_preserves_worker_rank_and_expires() {
        let ttl = Duration::from_secs(10);
        let store = AffinityCoordinator::new(ttl).unwrap();
        let worker = WorkerWithDpRank {
            worker_id: 42,
            dp_rank: 3,
        };
        store
            .put_binding(
                "session",
                AffinityBinding::with_endpoint(worker, EndpointId("endpoint".into())),
                ttl,
            )
            .await
            .unwrap();
        assert_eq!(
            store.peek_binding("session").await.unwrap(),
            Some(AffinityBinding::worker_only(worker))
        );
        tokio::time::advance(Duration::from_secs(9)).await;
        assert_eq!(store.peek("session").await.unwrap(), Some(worker));
        tokio::time::advance(Duration::from_secs(2)).await;
        assert_eq!(store.peek("session").await.unwrap(), None);
    }
}
