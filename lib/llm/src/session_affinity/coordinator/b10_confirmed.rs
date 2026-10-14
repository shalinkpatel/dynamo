// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Confirmed placements for callers that do not acquire generation leases.
use super::*;

impl AffinityCoordinator {
    /// Record the worker that actually served a request, refreshing its idle TTL
    /// and publishing through the coordinator's existing best-effort transport.
    pub fn record_confirmed_target(
        &self,
        session_id: &SessionAffinityId,
        target: AffinityTarget,
    ) -> Result<(), Error> {
        let version = self.inner.next_affinity_version();
        let outcome = self.inner.apply_replica_update(SessionAffinityUpdate {
            session_id: session_id.as_str().to_owned(),
            worker_id: target.worker_id,
            dp_rank: target.dp_rank,
            router_id: version.router_id,
            version_epoch: version.epoch,
            version_counter: version.counter,
            version_router_id: version.router_id,
        });
        match outcome {
            ReplicaApplyOutcome::RejectedCapacity => {
                return Err(resource_exhausted("session affinity entry limit reached"));
            }
            ReplicaApplyOutcome::IgnoredInitializing => {
                return Err(invalid_argument(
                    "session affinity selection is in progress",
                ));
            }
            ReplicaApplyOutcome::RejectedSessionId => {
                return Err(invalid_argument(
                    "session affinity ID exceeds the byte limit",
                ));
            }
            ReplicaApplyOutcome::IgnoredStale => return Ok(()),
            ReplicaApplyOutcome::Inserted
            | ReplicaApplyOutcome::Refreshed
            | ReplicaApplyOutcome::ReplacedExpired
            | ReplicaApplyOutcome::ReplacedNewer => {}
        }
        self.inner
            .publish_replica_update(session_id.as_str(), target, version);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn b10_confirmed_placement_overwrites_and_publishes() {
        let coordinator = AffinityCoordinator::new(Duration::from_secs(10)).unwrap();
        let peer = AffinityCoordinator::new(Duration::from_secs(10)).unwrap();
        let mut updates = coordinator.enable_test_replica(7, 16);
        let session = SessionAffinityId::new("session".to_owned());
        let first = AffinityTarget {
            worker_id: 1,
            dp_rank: Some(0),
        };
        let second = AffinityTarget {
            worker_id: 2,
            dp_rank: Some(1),
        };
        coordinator
            .record_confirmed_target(&session, first)
            .unwrap();
        let old_update = updates.recv().await.unwrap();
        coordinator
            .record_confirmed_target(&session, second)
            .unwrap();
        peer.apply_replica_message_for_test(updates.recv().await.unwrap());
        peer.apply_replica_message_for_test(old_update);
        assert_eq!(peer.query_target(&session, None).unwrap(), Some(second));
    }
}
