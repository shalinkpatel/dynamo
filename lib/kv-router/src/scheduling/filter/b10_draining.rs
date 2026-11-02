// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Baseten: keeps new work off draining (lame-duck) workers.
//!
//! A worker drains itself through `POST /engine/drain` (see `local_model::b10_drain` in
//! `dynamo-llm`), which publishes a `b10.drain/until=<RFC 3339>` taint on its base model card;
//! this module reads it back from the worker config and, unless `router_honor_draining` is
//! set to `false`, excludes the worker from selection while any other worker is eligible.

use std::collections::HashMap;

use super::{RoutingEligibility, worker_config_for_rank};
use crate::protocols::{WorkerConfigLike, WorkerId, WorkerWithDpRank};

/// Prefix of the taint a draining worker publishes; the rest is the RFC 3339 deadline.
/// Requests never name it in `required_taints`/`preferred_taints`, so it changes nothing for
/// routers with `router_honor_draining: false`.
pub const DRAIN_TAINT_PREFIX: &str = "b10.drain/until=";

/// The drain deadline a worker published (RFC 3339, as written), if it is draining. Every
/// reader of drain state (router filter, router metrics, planner) reads this taint.
///
/// A worker is draining exactly while it publishes the taint; the deadline itself is
/// information for the planner, not a condition the router judges against its own clock.
/// llm-operator owns both the cancel and the pod removal, so a passed deadline still keeps
/// new sessions off the worker until the operator clears it or removes the pod.
pub fn draining_until<C: WorkerConfigLike>(config: &C) -> Option<&str> {
    config
        .taints()
        .iter()
        .find_map(|taint| taint.strip_prefix(DRAIN_TAINT_PREFIX))
}

/// Whether the worker currently asks routers to keep new work off it.
pub fn is_draining<C: WorkerConfigLike>(config: &C) -> bool {
    draining_until(config).is_some()
}

/// Whether new work may land on draining workers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DrainingFilter {
    Allow,
    /// Skip draining workers, except the request's session-affinity rank: the session's
    /// cache lives on that exact rank.
    Exclude {
        session_worker: Option<WorkerWithDpRank>,
    },
    /// Draining workers exist but excluding them would leave nothing eligible, so they
    /// are allowed. Behaves like `Allow`; kept apart so the fallback can be counted.
    FellBack,
}

/// What `excluding_draining_workers` decided for a request, for metrics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrainingOutcome {
    /// No draining worker was in play (none draining, or the request was pinned).
    NotApplied,
    /// Draining workers were kept out of selection.
    Excluded,
    /// Draining workers were the only eligible ones, so selection may use them.
    FellBack,
}

impl<'a> RoutingEligibility<'a> {
    /// Keeps new work off draining workers while any other worker is eligible.
    ///
    /// Pinned requests (explicit worker hints and sticky sessions) are unaffected, because
    /// `validate_worker_rank` never looks at draining state. `session_worker`, the request's
    /// soft session-affinity rank, stays eligible so its session keeps its cache; the other
    /// ranks of that worker are still skipped. When excluding draining workers would leave
    /// nothing eligible, the result is unchanged.
    pub fn excluding_draining_workers<C: WorkerConfigLike>(
        self,
        workers: &HashMap<WorkerId, C>,
        session_worker: Option<WorkerWithDpRank>,
    ) -> Self {
        if self.pinned_worker.is_some() {
            return self;
        }
        if !workers.values().any(is_draining) {
            return self;
        }
        // A stale rank (worker gone or resized) exempts nothing rather than a whole worker.
        let session_worker =
            session_worker.filter(|&worker| worker_config_for_rank(workers, worker).is_ok());
        let excluding = Self {
            draining: DrainingFilter::Exclude { session_worker },
            ..self
        };
        if excluding.has_eligible_worker(
            workers
                .iter()
                .map(|(&worker_id, config)| (worker_id, config)),
        ) {
            excluding
        } else {
            Self {
                draining: DrainingFilter::FellBack,
                ..self
            }
        }
    }

    pub fn draining_outcome(&self) -> DrainingOutcome {
        match self.draining {
            DrainingFilter::Allow => DrainingOutcome::NotApplied,
            DrainingFilter::Exclude { .. } => DrainingOutcome::Excluded,
            DrainingFilter::FellBack => DrainingOutcome::FellBack,
        }
    }

    /// Draining check shared by the worker-level and rank-level eligibility paths: a
    /// draining worker passes only where `is_session` matches the session-affinity rank. At
    /// worker level that is "hosts the rank" (`|s| s.worker_id == worker_id`); at rank level
    /// it is the exact rank (`|s| s == worker`).
    #[inline]
    pub(super) fn admits_draining<C: WorkerConfigLike>(
        &self,
        config: &C,
        is_session: impl FnOnce(WorkerWithDpRank) -> bool,
    ) -> bool {
        match self.draining {
            DrainingFilter::Allow | DrainingFilter::FellBack => true,
            DrainingFilter::Exclude { session_worker } => {
                !is_draining(config) || session_worker.is_some_and(is_session)
            }
        }
    }
}

/// A drain taint whose deadline stays in the future for the whole test.
#[cfg(test)]
pub(crate) fn b10_drain_taint() -> String {
    format!("{DRAIN_TAINT_PREFIX}2099-01-01T00:00:00+00:00")
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::protocols::RoutingConstraints;
    use crate::test_utils::SimpleWorkerConfig;

    fn draining() -> SimpleWorkerConfig {
        SimpleWorkerConfig {
            taints: HashSet::from([b10_drain_taint()]),
            ..Default::default()
        }
    }

    /// Worker 1 is draining, worker 2 is not.
    fn one_draining_worker() -> HashMap<WorkerId, SimpleWorkerConfig> {
        HashMap::from([(1, draining()), (2, SimpleWorkerConfig::default())])
    }

    fn eligible_ranks(
        eligibility: RoutingEligibility<'_>,
        workers: &HashMap<WorkerId, SimpleWorkerConfig>,
    ) -> Vec<WorkerWithDpRank> {
        let mut ranks = Vec::new();
        eligibility.for_each_eligible_worker_rank(workers, |worker, _| ranks.push(worker));
        ranks.sort_by_key(|worker| (worker.worker_id, worker.dp_rank));
        ranks
    }

    #[test]
    fn b10_excluding_draining_workers_skips_them_for_new_work() {
        let workers = one_draining_worker();
        let constraints = RoutingConstraints::default();
        let eligibility = RoutingEligibility::new(None, None, None, &constraints);
        let both = vec![WorkerWithDpRank::new(1, 0), WorkerWithDpRank::new(2, 0)];
        assert_eq!(eligible_ranks(eligibility, &workers), both);
        assert_eq!(eligibility.draining_outcome(), DrainingOutcome::NotApplied);

        let excluding = eligibility.excluding_draining_workers(&workers, None);
        assert_eq!(
            eligible_ranks(excluding, &workers),
            vec![WorkerWithDpRank::new(2, 0)]
        );
        assert!(!excluding.allows_worker(1, &workers[&1]));
        assert_eq!(excluding.draining_outcome(), DrainingOutcome::Excluded);

        // The deadline is information, not a condition: a passed deadline still drains until
        // the worker clears it, so the router never depends on its clock.
        let mut workers = workers;
        workers.get_mut(&1).unwrap().taints =
            HashSet::from([format!("{DRAIN_TAINT_PREFIX}2000-01-01T00:00:00+00:00")]);
        let excluding = eligibility.excluding_draining_workers(&workers, None);
        assert_eq!(
            eligible_ranks(excluding, &workers),
            vec![WorkerWithDpRank::new(2, 0)]
        );

        // Clearing the taint is what makes it routable again; other taints don't drain.
        workers.get_mut(&1).unwrap().taints = HashSet::from(["service_tier/priority".to_string()]);
        let excluding = eligibility.excluding_draining_workers(&workers, None);
        assert_eq!(eligible_ranks(excluding, &workers), both);
        assert_eq!(excluding.draining_outcome(), DrainingOutcome::NotApplied);
    }

    #[test]
    fn b10_excluding_draining_workers_keeps_only_the_session_affinity_rank() {
        // Draining worker 1 has two data-parallel ranks; the session lives on rank 1.
        let mut workers = one_draining_worker();
        workers.get_mut(&1).unwrap().data_parallel_size = 2;
        let constraints = RoutingConstraints::default();
        let eligibility = RoutingEligibility::new(None, None, None, &constraints)
            .excluding_draining_workers(&workers, Some(WorkerWithDpRank::new(1, 1)));

        assert_eq!(
            eligible_ranks(eligibility, &workers),
            vec![WorkerWithDpRank::new(1, 1), WorkerWithDpRank::new(2, 0)]
        );
        assert!(eligibility.allows_worker(1, &workers[&1]));

        // A rank the worker no longer has exempts nothing.
        let eligibility = RoutingEligibility::new(None, None, None, &constraints)
            .excluding_draining_workers(&workers, Some(WorkerWithDpRank::new(1, 7)));
        assert_eq!(
            eligible_ranks(eligibility, &workers),
            vec![WorkerWithDpRank::new(2, 0)]
        );
    }

    #[test]
    fn b10_excluding_draining_workers_keeps_pinned_draining_worker() {
        let workers = one_draining_worker();
        let constraints = RoutingConstraints::default();
        let pinned = WorkerWithDpRank::new(1, 0);
        let eligibility = RoutingEligibility::new(None, None, Some(pinned), &constraints)
            .excluding_draining_workers(&workers, None);

        assert!(eligibility.validate_worker_rank(&workers, pinned).is_ok());
        assert_eq!(eligible_ranks(eligibility, &workers), vec![pinned]);
        assert_eq!(eligibility.draining_outcome(), DrainingOutcome::NotApplied);
    }

    #[test]
    fn b10_excluding_draining_workers_falls_back_when_only_draining_workers_are_eligible() {
        let workers = HashMap::from([(1, draining()), (2, draining())]);
        let constraints = RoutingConstraints::default();
        let eligibility = RoutingEligibility::new(None, None, None, &constraints)
            .excluding_draining_workers(&workers, None);

        assert_eq!(
            eligible_ranks(eligibility, &workers),
            vec![WorkerWithDpRank::new(1, 0), WorkerWithDpRank::new(2, 0)]
        );
        assert_eq!(eligibility.draining_outcome(), DrainingOutcome::FellBack);
    }

    #[test]
    fn b10_excluding_draining_workers_falls_back_when_other_workers_are_ineligible() {
        let workers = one_draining_worker();
        let constraints = RoutingConstraints::default();
        let overloaded = HashSet::from([2]);
        let eligibility = RoutingEligibility::new(None, Some(&overloaded), None, &constraints)
            .excluding_draining_workers(&workers, None);
        assert_eq!(
            eligible_ranks(eligibility, &workers),
            vec![WorkerWithDpRank::new(1, 0)]
        );
        assert_eq!(eligibility.draining_outcome(), DrainingOutcome::FellBack);

        let allowed = HashSet::from([1]);
        let eligibility = RoutingEligibility::new(Some(&allowed), None, None, &constraints)
            .excluding_draining_workers(&workers, None);
        assert_eq!(
            eligible_ranks(eligibility, &workers),
            vec![WorkerWithDpRank::new(1, 0)]
        );
    }
}
