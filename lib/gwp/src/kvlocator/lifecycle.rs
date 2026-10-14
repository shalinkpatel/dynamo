// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Attempt-scoped serialization over GWP's booking and response lifecycle.

use std::sync::Arc;
use std::time::Duration;

use dynamo_kvlocator::lifecycle::{AlreadyCommitted, AlreadyRecorded, InvalidAttempt, Lifecycle};
use dynamo_kvlocator::pb;
use tokio::sync::Mutex;
use tokio::time::Instant;

use crate::config::EndpointId;
use crate::core::{GwpCore, INFLIGHT_MAX_AGE, ScheduleError};
use dynamo_kv_router::protocols::WorkerWithDpRank;

/// Bounds notification replay after termination, including booking expiry.
const TOMBSTONE_TTL: Duration = Duration::from_secs(600);
const ATTEMPT_SWEEP_INTERVAL: Duration = Duration::from_secs(30);
type AttemptMap = dashmap::DashMap<String, Arc<Mutex<Attempt>>>;

#[derive(Clone)]
pub struct GwpLifecycle {
    core: GwpCore,
    attempts: Arc<AttemptMap>,
}

struct PreparedAdmission {
    native: crate::core::NativeAdmission,
    topology: Arc<crate::topology::TopologySnapshot>,
    context_topology: Option<Arc<crate::topology::TopologySnapshot>>,
    router: Option<Arc<crate::router::GwpRouter>>,
}

struct CommitContext {
    model_id: String,
    endpoint: EndpointId,
    tokens: Vec<u32>,
    identity: crate::tokens::RoutingTokenIdentity,
    session_id: Option<String>,
}

struct Attempt {
    commit: Option<CommitContext>,
    outcome: Option<pb::response_open_outcome_request::Outcome>,
    prepared_admission: Option<PreparedAdmission>,
    booked: bool,
    admission_recorded: bool,
    created_at: Instant,
    terminal_at: Option<Instant>,
}

impl Default for Attempt {
    fn default() -> Self {
        Self {
            commit: None,
            outcome: None,
            prepared_admission: None,
            booked: false,
            admission_recorded: false,
            created_at: Instant::now(),
            terminal_at: None,
        }
    }
}

impl GwpLifecycle {
    pub fn new(core: GwpCore) -> Self {
        let attempts = Arc::new(AttemptMap::new());
        let weak_attempts = Arc::downgrade(&attempts);
        let janitor_core = core.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(ATTEMPT_SWEEP_INTERVAL);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                interval.tick().await;
                let Some(attempts) = weak_attempts.upgrade() else {
                    return;
                };
                Self::sweep(&janitor_core, &attempts).await;
            }
        });
        Self { core, attempts }
    }

    fn attempt(&self, attempt_id: &str) -> Arc<Mutex<Attempt>> {
        self.attempts.entry(attempt_id.into()).or_default().clone()
    }

    fn rid(attempt_id: &str) -> String {
        format!("kvl:{attempt_id}")
    }

    async fn expire(core: &GwpCore, attempt_id: &str, attempt: &mut Attempt) {
        if attempt.terminal_at.is_none() && attempt.created_at.elapsed() >= INFLIGHT_MAX_AGE {
            attempt.terminal_at = Some(Instant::now());
            core.finish_kvlocator_attempt(&Self::rid(attempt_id), "expired")
                .await;
        }
    }

    async fn sweep(core: &GwpCore, attempts: &AttemptMap) {
        // Do not hold a DashMap guard while waiting on an attempt's mutex.
        let entries: Vec<_> = attempts
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();
        for (id, record) in entries {
            let mut attempt = record.lock().await;
            Self::expire(core, &id, &mut attempt).await;
            if attempt
                .terminal_at
                .is_some_and(|time| time.elapsed() >= TOMBSTONE_TTL)
            {
                attempts.remove(&id);
            }
        }
    }

    fn validate_commit<'a>(
        &self,
        request: &'a pb::CommitIntendRequest,
    ) -> anyhow::Result<(&'a pb::Placement, crate::tokens::RoutingTokenIdentity)> {
        let input = request
            .input
            .as_ref()
            .ok_or_else(|| InvalidAttempt("input is required".into()))?;
        let Some(pb::commit_intend_request::ChosenPlacement::Chosen(chosen)) =
            &request.chosen_placement
        else {
            return Err(InvalidAttempt("exactly one chosen placement is required".into()).into());
        };
        let (_, identity) = super::scoring::validate_input(input)
            .map_err(|error| InvalidAttempt(error.to_string()))?;
        let topology = self.core.topology.load();
        let (_, binding) = topology
            .model_binding(&input.model_id)
            .ok_or_else(|| InvalidAttempt("model is not known to GWP routing".into()))?;
        let worker = WorkerWithDpRank::new(chosen.worker_id, chosen.dp_rank);
        let eligible = super::scoring::eligible_workers(&topology, binding, input)
            .is_some_and(|(workers, _)| workers.contains(&worker));
        if !eligible
            || topology.worker_endpoint(worker.worker_id)
                != Some(&EndpointId(chosen.endpoint_id.clone()))
        {
            return Err(InvalidAttempt(
                "chosen placement does not satisfy input eligibility and constraints".into(),
            )
            .into());
        }
        Ok((chosen, identity))
    }

    async fn intend_inner(&self, request: &pb::CommitIntendRequest) -> anyhow::Result<()> {
        validate_attempt_id(&request.attempt_id)?;
        let record = self.attempt(&request.attempt_id);
        let mut attempt = record.lock().await;
        Self::expire(&self.core, &request.attempt_id, &mut attempt).await;
        if attempt.booked {
            return Err(AlreadyCommitted.into());
        }
        if attempt.terminal_at.is_some() && attempt.commit.is_some() {
            return Err(InvalidAttempt(
                "attempt is terminal; a new dispatch requires a new attempt_id".into(),
            )
            .into());
        }
        if request.request_id.is_empty() {
            return Err(InvalidAttempt("request_id must not be empty".into()).into());
        }
        let (chosen, identity) = self.validate_commit(request)?;
        let input = request.input.as_ref().expect("validated input");
        let tokens = &input.bid.as_ref().expect("validated bid").tokens;
        let worker = WorkerWithDpRank::new(chosen.worker_id, chosen.dp_rank);
        let session_id = input
            .bid
            .as_ref()
            .and_then(|bid| bid.session_id.clone())
            .filter(|sid| !sid.is_empty());
        let context = CommitContext {
            model_id: input.model_id.clone(),
            endpoint: EndpointId(chosen.endpoint_id.clone()),
            tokens: tokens.to_vec(),
            identity: identity.clone(),
            session_id: session_id.clone(),
        };
        if attempt.terminal_at.is_some() {
            attempt.commit = Some(context);
            self.apply_admission(&request.attempt_id, &mut attempt, None)
                .await?;
            return Err(InvalidAttempt(
                "attempt is terminal; a new dispatch requires a new attempt_id".into(),
            )
            .into());
        }
        if let Some(prepared) = attempt.prepared_admission.as_mut()
            && let Err(error) =
                self.bind_admission_context(prepared, &context, self.core.topology.load())
        {
            attempt.prepared_admission = None;
            attempt.outcome = None;
            return Err(error);
        }
        if matches!(
            attempt.outcome,
            Some(pb::response_open_outcome_request::Outcome::Denied(_))
        ) {
            attempt.terminal_at.get_or_insert_with(Instant::now);
            attempt.commit = Some(context);
            self.apply_admission(&request.attempt_id, &mut attempt, None)
                .await?;
            return Err(InvalidAttempt(
                "attempt is terminal; a new dispatch requires a new attempt_id".into(),
            )
            .into());
        }
        let result = self
            .core
            .book_kvlocator_attempt(
                &Self::rid(&request.attempt_id),
                &input.model_id,
                session_id,
                tokens.to_vec(),
                identity,
                &EndpointId(chosen.endpoint_id.clone()),
                worker,
            )
            .await;
        if let Err(error) = result {
            attempt.terminal_at = Some(Instant::now());
            return Err(match error {
                ScheduleError::InvalidModel(message) => InvalidAttempt(message).into(),
                other => other.into(),
            });
        }
        attempt.commit = Some(context);
        attempt.booked = true;
        self.apply_admission(&request.attempt_id, &mut attempt, None)
            .await?;
        Ok(())
    }

    fn prepare_admission(
        &self,
        admission: &pb::Admission,
        commit: Option<&CommitContext>,
    ) -> anyhow::Result<PreparedAdmission> {
        let decode = match (admission.decode_worker_id, admission.decode_dp_rank) {
            (Some(id), Some(rank)) => Some(WorkerWithDpRank::new(id, rank)),
            (None, None) => None,
            _ => {
                return Err(InvalidAttempt(
                    "decode worker ID and DP rank must be set together".into(),
                )
                .into());
            }
        };
        let native = crate::core::NativeAdmission {
            prefill: WorkerWithDpRank::new(admission.prefill_worker_id, admission.prefill_dp_rank),
            decode,
            cached_tokens: usize::try_from(admission.estimated_overlap_tokens)
                .unwrap_or(usize::MAX),
        };
        let topology = self.core.topology.load();
        let mut prepared = PreparedAdmission {
            native,
            topology: topology.clone(),
            context_topology: None,
            router: None,
        };
        if let Some(commit) = commit {
            self.bind_admission_context(&mut prepared, commit, topology)?;
        } else {
            for worker in [Some(prepared.native.prefill), prepared.native.decode]
                .into_iter()
                .flatten()
            {
                crate::topology::validate_admission_worker(&prepared.topology, None, worker)
                    .map_err(|error| InvalidAttempt(error.to_string()))?;
            }
        }
        Ok(prepared)
    }

    fn bind_admission_context(
        &self,
        prepared: &mut PreparedAdmission,
        commit: &CommitContext,
        topology: Arc<crate::topology::TopologySnapshot>,
    ) -> anyhow::Result<()> {
        if prepared.context_topology.is_some() {
            return Ok(());
        }
        let endpoint = &commit.endpoint;
        if topology.endpoint(endpoint).is_some() {
            for worker in [Some(prepared.native.prefill), prepared.native.decode]
                .into_iter()
                .flatten()
            {
                crate::topology::validate_admission_worker(
                    &prepared.topology,
                    Some(endpoint),
                    worker,
                )
                .map_err(|error| InvalidAttempt(error.to_string()))?;
            }
            let unknown_workers: Vec<_> = [Some(prepared.native.prefill), prepared.native.decode]
                .into_iter()
                .flatten()
                .filter(|worker| {
                    prepared
                        .topology
                        .worker_endpoint(worker.worker_id)
                        .is_none()
                })
                .collect();
            self.core
                .topology
                .confirm_admission_workers(endpoint, &unknown_workers)
                .map_err(|error| InvalidAttempt(error.to_string()))?;
            prepared.router = topology
                .model_binding(&commit.model_id)
                .map(|(model, _)| crate::router::OracleVersionId::new(model.clone()))
                .transpose()?
                .and_then(|model| self.core.router.model_router(&model));
        }
        prepared.context_topology = Some(topology);
        Ok(())
    }

    async fn apply_admission(
        &self,
        attempt_id: &str,
        attempt: &mut Attempt,
        prepared: Option<PreparedAdmission>,
    ) -> anyhow::Result<()> {
        if attempt.admission_recorded || (!attempt.booked && attempt.terminal_at.is_none()) {
            return Ok(());
        }
        let admission = match attempt.outcome.as_ref() {
            Some(pb::response_open_outcome_request::Outcome::Admitted(admission)) => {
                Some(*admission)
            }
            Some(pb::response_open_outcome_request::Outcome::Denied(denied)) => denied.admission,
            Some(pb::response_open_outcome_request::Outcome::Failed(_)) | None => None,
        };
        let Some(admission) = admission else {
            return Ok(());
        };
        let Some(commit) = &attempt.commit else {
            return Ok(());
        };
        let prepared = match prepared.or_else(|| attempt.prepared_admission.take()) {
            Some(prepared) => Ok(prepared),
            None => self.prepare_admission(&admission, Some(commit)),
        };
        let mut prepared = match prepared {
            Ok(prepared) => prepared,
            Err(error) => {
                attempt.outcome = None;
                return Err(error);
            }
        };
        if let Err(error) =
            self.bind_admission_context(&mut prepared, commit, self.core.topology.load())
        {
            attempt.outcome = None;
            return Err(error);
        }
        let result = if attempt.terminal_at.is_some() {
            let Some(commit) = &attempt.commit else {
                return Ok(());
            };
            self.record_terminal_admission(
                attempt_id,
                commit,
                &prepared.native,
                prepared
                    .context_topology
                    .as_deref()
                    .expect("prepared commit context"),
                prepared.router.as_deref(),
            )
            .await
        } else if attempt.booked {
            self.core
                .response_started_for_attempt(&Self::rid(attempt_id), prepared.native)
                .await
                .map_err(|error| InvalidAttempt(error.to_string()).into())
        } else {
            return Ok(());
        };
        if result.is_err() {
            // A rejected notification must not prevent a corrected opening outcome.
            attempt.outcome = None;
        } else {
            attempt.admission_recorded = true;
        }
        result
    }

    async fn record_terminal_admission(
        &self,
        attempt_id: &str,
        commit: &CommitContext,
        admission: &crate::core::NativeAdmission,
        topology: &crate::topology::TopologySnapshot,
        router: Option<&crate::router::GwpRouter>,
    ) -> anyhow::Result<()> {
        let Some(router) = router else {
            return Ok(());
        };
        let endpoint = &commit.endpoint;
        let Some((_, binding)) = topology.model_binding(&commit.model_id) else {
            return Ok(());
        };
        // Retired routes must not receive cache or affinity updates.
        if !binding.endpoints.contains(endpoint) || topology.endpoint(endpoint).is_none() {
            return Ok(());
        }
        let policy = topology
            .profile(binding)
            .ok_or_else(|| anyhow::anyhow!("model has no routing profile"))?
            .stages;
        let worker = admission.prefill;
        if policy.trie {
            router
                .record_routing_decision_with_identity(
                    &Self::rid(attempt_id),
                    &commit.tokens,
                    &commit.identity,
                    worker,
                )
                .await;
        }
        if policy.affinity
            && let Some(session) = commit.session_id.clone()
        {
            let affinity = self.core.affinity.clone();
            let ttl = Duration::from_secs(self.core.config.load().session.ttl_secs);
            let binding = crate::session::AffinityBinding::with_endpoint(worker, endpoint.clone());
            tokio::spawn(async move {
                if let Err(error) = affinity.put_binding(&session, binding, ttl).await {
                    tracing::warn!(%error, "affinity put failed for terminal admission");
                }
            });
        }
        Ok(())
    }
    async fn open_inner(&self, request: &pb::ResponseOpenOutcomeRequest) -> anyhow::Result<()> {
        validate_attempt_id(&request.attempt_id)?;
        let record = self.attempt(&request.attempt_id);
        let mut attempt = record.lock().await;
        Self::expire(&self.core, &request.attempt_id, &mut attempt).await;
        if attempt.outcome.is_some() {
            return Err(AlreadyRecorded.into());
        }
        let outcome = request
            .outcome
            .as_ref()
            .ok_or_else(|| InvalidAttempt("exactly one outcome is required".into()))?;
        let terminal_status = match outcome {
            pb::response_open_outcome_request::Outcome::Failed(failed) => {
                Some(open_failure_status(failed)?)
            }
            pb::response_open_outcome_request::Outcome::Denied(denied) => {
                Some(denial_status(denied)?)
            }
            pb::response_open_outcome_request::Outcome::Admitted(_) => None,
        };
        let prepared = if !attempt.admission_recorded && attempt.prepared_admission.is_none() {
            let admission = match outcome {
                pb::response_open_outcome_request::Outcome::Admitted(admission) => Some(admission),
                pb::response_open_outcome_request::Outcome::Denied(denied) => {
                    denied.admission.as_ref()
                }
                pb::response_open_outcome_request::Outcome::Failed(_) => None,
            };
            admission
                .map(|admission| self.prepare_admission(admission, attempt.commit.as_ref()))
                .transpose()?
        } else {
            None
        };
        if let Some(prepared) = prepared {
            attempt.prepared_admission = Some(prepared);
        }
        attempt.outcome = Some(outcome.clone());
        match outcome {
            pb::response_open_outcome_request::Outcome::Admitted(_) => {
                self.apply_admission(&request.attempt_id, &mut attempt, None)
                    .await?;
            }
            pb::response_open_outcome_request::Outcome::Denied(denied) => {
                if attempt.commit.is_some() || denied.admission.is_none() {
                    attempt.terminal_at.get_or_insert_with(Instant::now);
                    self.core
                        .response_started(
                            &Self::rid(&request.attempt_id),
                            None,
                            terminal_status.expect("denial validated"),
                        )
                        .await;
                }
                self.apply_admission(&request.attempt_id, &mut attempt, None)
                    .await?;
            }
            pb::response_open_outcome_request::Outcome::Failed(_) => {
                attempt.terminal_at.get_or_insert_with(Instant::now);
                self.core
                    .response_started(
                        &Self::rid(&request.attempt_id),
                        None,
                        terminal_status.expect("failure kind validated"),
                    )
                    .await;
            }
        }
        Ok(())
    }

    async fn finish_inner(&self, request: &pb::ResponseFinishRequest) -> anyhow::Result<()> {
        validate_attempt_id(&request.attempt_id)?;
        let record = self.attempt(&request.attempt_id);
        let mut attempt = record.lock().await;
        if attempt.terminal_at.is_some() {
            return Ok(());
        }
        let reason = finish_reason(request)?;
        attempt.terminal_at = Some(Instant::now());
        self.core
            .finish_kvlocator_attempt(&Self::rid(&request.attempt_id), reason)
            .await;
        Ok(())
    }
}

#[async_trait::async_trait]
impl Lifecycle for GwpLifecycle {
    async fn intend(&self, request: &pb::CommitIntendRequest) -> anyhow::Result<()> {
        let backend = self.clone();
        let request = request.clone();
        // Cancellation of an RPC must not interrupt an accepted booking transaction.
        tokio::spawn(async move { backend.intend_inner(&request).await }).await?
    }

    async fn open(&self, request: &pb::ResponseOpenOutcomeRequest) -> anyhow::Result<()> {
        let backend = self.clone();
        let request = request.clone();
        tokio::spawn(async move { backend.open_inner(&request).await }).await?
    }

    async fn finish(&self, request: &pb::ResponseFinishRequest) -> anyhow::Result<()> {
        let backend = self.clone();
        let request = request.clone();
        tokio::spawn(async move { backend.finish_inner(&request).await }).await?
    }
}

/// The finish reason as the core's request_finished spells it.
fn finish_reason(request: &pb::ResponseFinishRequest) -> anyhow::Result<&'static str> {
    let reason = pb::response_finish_request::Reason::try_from(request.reason)
        .map_err(|_| InvalidAttempt("unknown finish reason".into()))?;
    Ok(match reason {
        pb::response_finish_request::Reason::Completed => "completed",
        pb::response_finish_request::Reason::Cancelled => "cancelled",
        pb::response_finish_request::Reason::Failed => "failed",
        pb::response_finish_request::Reason::Unspecified => {
            return Err(InvalidAttempt("reason must not be UNSPECIFIED".into()).into());
        }
    })
}

/// DeniedGeneration carries no HTTP status; map the denial kind to the
/// closest upstream status the core's response path already understands
/// (non-200 releases the booking).
fn denial_status(denied: &pb::DeniedGeneration) -> anyhow::Result<u16> {
    use pb::denied_request::Kind;
    let request = denied
        .denied
        .as_ref()
        .ok_or_else(|| InvalidAttempt("denied request is required".into()))?;
    let kind =
        Kind::try_from(request.kind).map_err(|_| InvalidAttempt("unknown denial kind".into()))?;
    let (status, required) = match kind {
        Kind::RouterBackpressure => (429, Some(("reason", &request.reason))),
        Kind::RequiredComponentsDown => (503, Some(("name", &request.name))),
        Kind::Cancelled => (499, None),
        Kind::ProtocolError => (400, Some(("received", &request.received))),
        Kind::NextRouterUnreachable | Kind::FirstWorkerEventFailed | Kind::WorkerErrorResponse => {
            (500, Some(("error", &request.error)))
        }
        Kind::Unspecified => {
            return Err(InvalidAttempt("denial kind must not be UNSPECIFIED".into()).into());
        }
    };
    if let Some((field, value)) = required
        && value.is_none()
    {
        return Err(InvalidAttempt(format!("denial is missing {field}")).into());
    }
    Ok(status)
}

fn validate_attempt_id(id: &str) -> anyhow::Result<()> {
    uuid::Uuid::parse_str(id).map_err(|_| InvalidAttempt("attempt_id must be a UUID".into()))?;
    Ok(())
}

fn open_failure_status(failed: &pb::OpenFailure) -> anyhow::Result<u16> {
    use pb::open_failure::Kind;
    Ok(
        match Kind::try_from(failed.kind)
            .map_err(|_| InvalidAttempt("unknown open-failure kind".into()))?
        {
            Kind::Transport => 502,
            Kind::Protocol => 500,
            Kind::Cancelled => 499,
            Kind::DeadlineExceeded => 504,
            Kind::Unspecified => {
                return Err(
                    InvalidAttempt("open-failure kind must not be UNSPECIFIED".into()).into(),
                );
            }
        },
    )
}

#[cfg(test)]
mod lifecycle_tests {
    use super::*;
    use crate::config::{EndpointConfig, GwpConfig, ModelRoute, RoutingConfig};
    use crate::router::{GwpRouterRegistry, OracleVersionId};
    use crate::session::InMemoryAffinityStore;
    use crate::topology::{TopologySnapshot, TopologyStore, TopologyWorker};
    use GwpCore;
    use std::collections::{BTreeMap, HashMap};
    use std::sync::Arc;

    const ENDPOINT: &str = "kvl-endpoint";
    const WORKER: u64 = 10;
    const TOKENS: &[u32] = &[1, 2, 3, 4, 5, 6, 7, 8];

    async fn make_lifecycle() -> GwpLifecycle {
        make_lifecycle_with_store(Arc::new(InMemoryAffinityStore::new())).await
    }

    async fn make_lifecycle_with_store(
        store: std::sync::Arc<dyn crate::session::AffinityStore>,
    ) -> GwpLifecycle {
        let config = GwpConfig {
            endpoints: BTreeMap::from([(
                EndpointId(ENDPOINT.into()),
                EndpointConfig {
                    ingress_url: url::Url::parse("https://endpoint.example.com:8443/v1").unwrap(),
                    api_key: "ck".into(),
                    planner_url: url::Url::parse("http://localhost:0/deep/health").unwrap(),
                    planner_api_key: None,
                    properties: Default::default(),
                },
            )]),
            routes: vec![ModelRoute {
                models: vec!["m".into()],
                endpoints: vec![EndpointId(ENDPOINT.into())],
            }],
            routing: RoutingConfig {
                block_size: 4,
                ..Default::default()
            },
            ..Default::default()
        };
        let snapshot = TopologySnapshot::from_config(
            &config,
            HashMap::from([(
                WORKER,
                TopologyWorker {
                    endpoint: EndpointId(ENDPOINT.into()),
                    runtime: dynamo_llm::local_model::runtime_config::ModelRuntimeConfig::default(),
                    observed_load: None,
                },
            )]),
        )
        .unwrap();
        let router = GwpRouterRegistry::new_process_local(4, 120).await.unwrap();
        router
            .reconcile_topology(&snapshot, &snapshot.workers.keys().copied().collect())
            .await
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        GwpLifecycle::new(GwpCore::new(
            router,
            store,
            TopologyStore::new(snapshot),
            Arc::new(config),
        ))
    }

    fn attempt_id(label: &str) -> String {
        let hash = dynamo_kv_router::protocols::compute_block_hash(label.as_bytes()).0;
        uuid::Uuid::from_u128(hash as u128).to_string()
    }

    fn commit_with_session(
        attempt: &str,
        request: &str,
        endpoint: &str,
        sid: &str,
    ) -> pb::CommitIntendRequest {
        let mut commit = commit(attempt, request, endpoint);
        commit
            .input
            .as_mut()
            .unwrap()
            .bid
            .as_mut()
            .unwrap()
            .session_id = Some(sid.into());
        commit
    }

    #[tokio::test]
    async fn session_carrying_attempt_binds_affinity() {
        let store = Arc::new(InMemoryAffinityStore::new());
        let lifecycle = make_lifecycle_with_store(store.clone()).await;

        // The skip-empty-sids fix must not break real bindings: a commit
        // carrying a session still binds the admitted worker under it. The
        // write is spawned, so poll.
        lifecycle
            .intend(&commit_with_session("sid-1", "r1", ENDPOINT, "sess-a"))
            .await
            .expect("commit");
        lifecycle.open(&admitted("sid-1")).await.expect("admission");
        let mut bound = None;
        for _ in 0..100 {
            bound = crate::session::AffinityStore::peek(store.as_ref(), "sess-a")
                .await
                .unwrap();
            if bound.is_some() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert_eq!(
            bound,
            Some(WorkerWithDpRank::new(WORKER, 0)),
            "the session still binds to the admitted worker"
        );
    }

    fn commit(attempt: &str, request: &str, endpoint: &str) -> pb::CommitIntendRequest {
        pb::CommitIntendRequest {
            attempt_id: attempt_id(attempt),
            request_id: request.into(),
            input: Some(pb::LocateRequest {
                model_id: "m".into(),
                bid: Some(pb::BidRequest {
                    tokens: TOKENS.to_vec(),
                    ..Default::default()
                }),
                eligible_endpoint_ids: vec![endpoint.into()],
                ..Default::default()
            }),
            chosen_placement: Some(pb::commit_intend_request::ChosenPlacement::Chosen(
                pb::Placement {
                    endpoint_id: endpoint.into(),
                    worker_id: WORKER,
                    dp_rank: 0,
                },
            )),
        }
    }

    fn admitted(attempt: &str) -> pb::ResponseOpenOutcomeRequest {
        pb::ResponseOpenOutcomeRequest {
            attempt_id: attempt_id(attempt),
            outcome: Some(pb::response_open_outcome_request::Outcome::Admitted(
                pb::Admission {
                    prefill_worker_id: WORKER,
                    prefill_dp_rank: 0,
                    estimated_overlap_tokens: 8,
                    ..Default::default()
                },
            )),
        }
    }

    fn finish(
        attempt: &str,
        reason: pb::response_finish_request::Reason,
    ) -> pb::ResponseFinishRequest {
        pb::ResponseFinishRequest {
            attempt_id: attempt_id(attempt),
            reason: reason as i32,
        }
    }

    async fn active_requests(lifecycle: &GwpLifecycle) -> usize {
        lifecycle
            .core
            .router
            .model_router(&OracleVersionId::new("m").unwrap())
            .unwrap()
            .potential_loads(TOKENS)
            .await
            .into_iter()
            .map(|load| load.active_requests)
            .sum()
    }

    #[tokio::test]
    async fn b10_commit_is_single_use_regardless_of_payload() {
        let lifecycle = make_lifecycle().await;
        let request = commit("a1", "r1", ENDPOINT);
        lifecycle.intend(&request).await.unwrap();
        let created_at = lifecycle
            .attempt(&request.attempt_id)
            .lock()
            .await
            .created_at;
        for duplicate in [
            request.clone(),
            commit("a1", "r2", ENDPOINT),
            pb::CommitIntendRequest {
                attempt_id: request.attempt_id.clone(),
                ..Default::default()
            },
        ] {
            let error = lifecycle.intend(&duplicate).await.unwrap_err();
            assert!(error.is::<AlreadyCommitted>());
            assert_eq!(
                error.to_string(),
                "attempt_id already committed; did you retry a previously delivered request?"
            );
            assert_eq!(active_requests(&lifecycle).await, 1);
        }
        assert_eq!(
            lifecycle
                .attempt(&request.attempt_id)
                .lock()
                .await
                .created_at,
            created_at
        );
        lifecycle
            .finish(&finish(
                "a1",
                pb::response_finish_request::Reason::Completed,
            ))
            .await
            .unwrap();
        assert!(
            lifecycle
                .intend(&request)
                .await
                .unwrap_err()
                .is::<AlreadyCommitted>()
        );
        assert_eq!(active_requests(&lifecycle).await, 0);
    }

    #[tokio::test]
    async fn admission_rebooks_and_finish_frees_exactly_once() {
        let lifecycle = make_lifecycle().await;
        lifecycle
            .intend(&commit("a1", "r1", ENDPOINT))
            .await
            .expect("commit");
        lifecycle
            .open(&admitted("a1"))
            .await
            .expect("admission applies");
        assert_eq!(active_requests(&lifecycle).await, 1);

        // Replayed admission acks without double-booking.
        assert!(
            lifecycle
                .open(&admitted("a1"))
                .await
                .unwrap_err()
                .is::<AlreadyRecorded>()
        );
        assert_eq!(active_requests(&lifecycle).await, 1);

        lifecycle
            .finish(&finish(
                "a1",
                pb::response_finish_request::Reason::Completed,
            ))
            .await
            .expect("finish frees");
        assert_eq!(active_requests(&lifecycle).await, 0);

        // Replayed finish acks; the tombstone rejects a re-commit.
        lifecycle
            .finish(&finish(
                "a1",
                pb::response_finish_request::Reason::Completed,
            ))
            .await
            .expect("replayed finish acks");
        let error = lifecycle
            .intend(&commit("a1", "r1", ENDPOINT))
            .await
            .expect_err("terminal attempt must not be booked again");
        assert!(error.is::<AlreadyCommitted>());
    }

    #[tokio::test]
    async fn denial_before_commit_tombstones_and_rejects_late_commit() {
        let lifecycle = make_lifecycle().await;
        lifecycle
            .open(&pb::ResponseOpenOutcomeRequest {
                attempt_id: attempt_id("a1"),
                outcome: Some(pb::response_open_outcome_request::Outcome::Denied(
                    pb::DeniedGeneration {
                        denied: Some(pb::DeniedRequest {
                            kind: pb::denied_request::Kind::Cancelled as i32,
                            ..Default::default()
                        }),
                        admission: None,
                    },
                )),
            })
            .await
            .expect("early denial retained");
        assert_eq!(active_requests(&lifecycle).await, 0);

        let error = lifecycle
            .intend(&commit("a1", "r1", ENDPOINT))
            .await
            .expect_err("commit after terminal denial rejected");
        assert!(error.downcast_ref::<InvalidAttempt>().is_some());
    }

    #[tokio::test]
    async fn admission_before_commit_is_retained_and_applied() {
        let lifecycle = make_lifecycle().await;
        lifecycle.open(&admitted("a1")).await.expect("retained");
        lifecycle
            .intend(&commit("a1", "r1", ENDPOINT))
            .await
            .expect("commit adopts");
        assert_eq!(active_requests(&lifecycle).await, 1, "actual worker booked");

        lifecycle
            .finish(&finish("a1", pb::response_finish_request::Reason::Failed))
            .await
            .expect("finish frees");
        assert_eq!(active_requests(&lifecycle).await, 0);
    }

    #[tokio::test]
    async fn finish_before_commit_tombstones() {
        let lifecycle = make_lifecycle().await;
        lifecycle
            .finish(&finish(
                "a1",
                pb::response_finish_request::Reason::Cancelled,
            ))
            .await
            .expect("early finish retained");
        let error = lifecycle
            .intend(&commit("a1", "r1", ENDPOINT))
            .await
            .expect_err("commit after finish rejected");
        assert!(error.downcast_ref::<InvalidAttempt>().is_some());
    }
    #[tokio::test]
    async fn b10_commit_enforces_all_eligibility_filters() {
        let lifecycle = make_lifecycle().await;
        for field in 0..4 {
            let mut request = commit(&format!("a{field}"), "r1", ENDPOINT);
            let input = request.input.as_mut().unwrap();
            match field {
                0 => input.eligible_endpoint_ids.clear(),
                1 => input.allowed_worker_ids = vec![WORKER + 1],
                2 => {
                    input.constraints = Some(pb::RoutingConstraints {
                        required_taints: vec!["missing".into()],
                        ..Default::default()
                    })
                }
                3 => {
                    let Some(pb::commit_intend_request::ChosenPlacement::Chosen(chosen)) =
                        request.chosen_placement.as_mut()
                    else {
                        unreachable!()
                    };
                    chosen.dp_rank = 1;
                }
                _ => unreachable!(),
            }
            assert!(lifecycle.intend(&request).await.is_err());
        }
        assert_eq!(active_requests(&lifecycle).await, 0);
    }

    #[tokio::test]
    async fn b10_concurrent_commits_book_once_regardless_of_payload() {
        for changed in [false, true] {
            let lifecycle = make_lifecycle().await;
            let first = commit("a1", "r1", ENDPOINT);
            let mut second = first.clone();
            if changed {
                second.input.as_mut().unwrap().bid.as_mut().unwrap().tokens[0] += 1;
            }
            let (a, b) = tokio::join!(lifecycle.intend(&first), lifecycle.intend(&second));
            assert_ne!(a.is_ok(), b.is_ok());
            assert!(
                a.err()
                    .or_else(|| b.err())
                    .unwrap()
                    .is::<AlreadyCommitted>()
            );
            assert_eq!(active_requests(&lifecycle).await, 1);
        }
    }

    #[tokio::test]
    async fn b10_finish_and_commit_are_serialized_in_both_orders() {
        for finish_first in [true, false] {
            let lifecycle = make_lifecycle().await;
            let request = commit("a1", "r1", ENDPOINT);
            let ending = finish("a1", pb::response_finish_request::Reason::Completed);
            let record = lifecycle.attempt(&attempt_id("a1"));
            let guard = record.lock().await;
            let mut commit_call = Box::pin(lifecycle.intend(&request));
            let mut finish_call = Box::pin(lifecycle.finish(&ending));
            if finish_first {
                assert!(futures::poll!(&mut finish_call).is_pending());
                tokio::task::yield_now().await;
                assert!(futures::poll!(&mut commit_call).is_pending());
            } else {
                assert!(futures::poll!(&mut commit_call).is_pending());
                tokio::task::yield_now().await;
                assert!(futures::poll!(&mut finish_call).is_pending());
            }
            drop(guard);
            let (booking, finished) = tokio::join!(commit_call, finish_call);
            finished.unwrap();
            assert_eq!(booking.is_err(), finish_first);
            assert_eq!(active_requests(&lifecycle).await, 0);
            assert!(lifecycle.intend(&request).await.is_err());
        }
    }

    #[tokio::test]
    async fn b10_expiry_releases_load_and_keeps_a_tombstone() {
        let lifecycle = make_lifecycle().await;
        let request = commit("a1", "r1", ENDPOINT);
        lifecycle.intend(&request).await.unwrap();
        lifecycle.attempt(&attempt_id("a1")).lock().await.created_at =
            Instant::now() - INFLIGHT_MAX_AGE;
        GwpLifecycle::sweep(&lifecycle.core, &lifecycle.attempts).await;
        assert_eq!(active_requests(&lifecycle).await, 0);
        assert!(lifecycle.intend(&request).await.is_err());
        GwpLifecycle::sweep(&lifecycle.core, &lifecycle.attempts).await;
        lifecycle.open(&admitted("a1")).await.unwrap();
        assert_eq!(active_requests(&lifecycle).await, 0);
        assert!(lifecycle.intend(&request).await.is_err());
        assert!(
            lifecycle
                .attempt(&attempt_id("a1"))
                .lock()
                .await
                .terminal_at
                .is_some()
        );
    }

    #[tokio::test]
    async fn b10_opening_is_single_use_and_finish_is_first_wins() {
        let lifecycle = make_lifecycle().await;
        lifecycle
            .intend(&commit("a1", "r1", ENDPOINT))
            .await
            .unwrap();
        lifecycle.open(&admitted("a1")).await.unwrap();
        let mut changed = admitted("a1");
        let Some(pb::response_open_outcome_request::Outcome::Admitted(admission)) =
            changed.outcome.as_mut()
        else {
            unreachable!()
        };
        admission.prefill_worker_id += 1;
        for duplicate in [
            changed,
            admitted("a1"),
            pb::ResponseOpenOutcomeRequest {
                attempt_id: attempt_id("a1"),
                outcome: None,
            },
        ] {
            assert!(
                lifecycle
                    .open(&duplicate)
                    .await
                    .unwrap_err()
                    .is::<AlreadyRecorded>()
            );
        }
        assert_eq!(active_requests(&lifecycle).await, 1);
        lifecycle
            .finish(&finish(
                "a1",
                pb::response_finish_request::Reason::Completed,
            ))
            .await
            .unwrap();
        let terminal_at = lifecycle
            .attempt(&attempt_id("a1"))
            .lock()
            .await
            .terminal_at;
        for reason in [
            pb::response_finish_request::Reason::Failed,
            pb::response_finish_request::Reason::Unspecified,
        ] {
            lifecycle.finish(&finish("a1", reason)).await.unwrap();
        }
        assert_eq!(
            lifecycle
                .attempt(&attempt_id("a1"))
                .lock()
                .await
                .terminal_at,
            terminal_at
        );
        assert_eq!(active_requests(&lifecycle).await, 0);
    }

    #[tokio::test]
    async fn b10_admission_preserves_multimodal_identity_in_confirmed_cache() {
        let lifecycle = make_lifecycle().await;
        let mut request = commit("a1", "r1", ENDPOINT);
        request
            .input
            .as_mut()
            .unwrap()
            .bid
            .as_mut()
            .unwrap()
            .mm_routing_args = Some(pb::MmRoutingArgs {
            blocks: vec![pb::MmRoutingBlock {
                present: true,
                objects: vec![pb::MmRoutingObject {
                    mm_hash: 42,
                    offsets: vec![pb::TokenRange { start: 0, end: 4 }],
                }],
            }],
        });
        request
            .input
            .as_mut()
            .unwrap()
            .bid
            .as_mut()
            .unwrap()
            .cache_salt = Some("tenant-a".into());
        lifecycle.intend(&request).await.unwrap();
        lifecycle.open(&admitted("a1")).await.unwrap();
        let scorer = super::super::scoring::GwpScorer::new(
            lifecycle.core.router.clone(),
            lifecycle.core.topology.clone(),
            lifecycle.core.affinity.clone(),
        );
        use dynamo_kvlocator::scorer::Scorer;
        let input = request.input.as_ref().unwrap();
        let response = scorer.rank(input).await.unwrap();
        assert_eq!(
            response.placements_v1.unwrap().candidates[0].estimated_overlap_tokens,
            8
        );
        let mut changed = input.clone();
        changed
            .bid
            .as_mut()
            .unwrap()
            .mm_routing_args
            .as_mut()
            .unwrap()
            .blocks[0]
            .objects[0]
            .mm_hash += 1;
        let response = scorer.rank(&changed).await.unwrap();
        assert_eq!(
            response.placements_v1.unwrap().candidates[0].estimated_overlap_tokens,
            0
        );
        lifecycle
            .finish(&finish(
                "a1",
                pb::response_finish_request::Reason::Completed,
            ))
            .await
            .unwrap();
        assert_eq!(active_requests(&lifecycle).await, 0);
    }
    async fn cancel_queued_rpc(
        future: impl std::future::Future<Output = anyhow::Result<()>>,
        record: Arc<Mutex<Attempt>>,
    ) {
        let guard = record.lock().await;
        let mut rpc = Box::pin(future);
        assert!(futures::poll!(&mut rpc).is_pending());
        tokio::task::yield_now().await;
        drop(rpc);
        drop(guard);
    }

    #[tokio::test]
    async fn b10_cancelled_rpc_does_not_interrupt_accounting_transactions() {
        let lifecycle = make_lifecycle().await;
        let request = commit("cancelled", "r1", ENDPOINT);
        let record = lifecycle.attempt(&attempt_id("cancelled"));
        let created_at = record.lock().await.created_at;
        cancel_queued_rpc(lifecycle.intend(&request), record.clone()).await;
        assert!(
            lifecycle
                .intend(&request)
                .await
                .unwrap_err()
                .is::<AlreadyCommitted>()
        );
        assert_eq!(active_requests(&lifecycle).await, 1);
        assert_eq!(record.lock().await.created_at, created_at);
        let opening = admitted("cancelled");
        cancel_queued_rpc(lifecycle.open(&opening), record.clone()).await;
        assert!(
            lifecycle
                .open(&opening)
                .await
                .unwrap_err()
                .is::<AlreadyRecorded>()
        );
        assert_eq!(active_requests(&lifecycle).await, 1);
        let ending = finish("cancelled", pb::response_finish_request::Reason::Completed);
        cancel_queued_rpc(lifecycle.finish(&ending), record).await;
        lifecycle.finish(&ending).await.unwrap();
        assert_eq!(active_requests(&lifecycle).await, 0);
        assert!(lifecycle.intend(&request).await.is_err());
    }

    #[tokio::test]
    async fn b10_terminal_admissions_record_prefix_without_recreating_load() {
        for event_order in 0..3 {
            let lifecycle = make_lifecycle().await;
            let mut request = commit("a1", "r1", ENDPOINT);
            request
                .input
                .as_mut()
                .unwrap()
                .bid
                .as_mut()
                .unwrap()
                .session_id = Some("late-session".into());
            let opening = admitted("a1");
            if event_order < 2 {
                lifecycle.intend(&request).await.unwrap();
            }
            if event_order == 1 {
                let Some(pb::response_open_outcome_request::Outcome::Admitted(admission)) =
                    opening.outcome.clone()
                else {
                    unreachable!()
                };
                lifecycle
                    .open(&pb::ResponseOpenOutcomeRequest {
                        attempt_id: attempt_id("a1"),
                        outcome: Some(pb::response_open_outcome_request::Outcome::Denied(
                            pb::DeniedGeneration {
                                denied: Some(pb::DeniedRequest {
                                    kind: pb::denied_request::Kind::RequiredComponentsDown as i32,
                                    name: Some("decode".into()),
                                    ..Default::default()
                                }),
                                admission: Some(admission),
                            },
                        )),
                    })
                    .await
                    .unwrap();
            } else {
                lifecycle
                    .finish(&finish(
                        "a1",
                        pb::response_finish_request::Reason::Completed,
                    ))
                    .await
                    .unwrap();
                lifecycle.open(&opening).await.unwrap();
            }
            assert_eq!(active_requests(&lifecycle).await, 0);
            assert!(lifecycle.intend(&request).await.is_err());
            let router = lifecycle
                .core
                .router
                .model_router(&OracleVersionId::new("m").unwrap())
                .unwrap();
            assert_eq!(
                router
                    .cached_tokens(
                        TOKENS,
                        &Default::default(),
                        WorkerWithDpRank::new(WORKER, 0)
                    )
                    .await,
                TOKENS.len()
            );
            tokio::task::yield_now().await;
            assert_eq!(
                lifecycle.core.affinity.peek("late-session").await.unwrap(),
                Some(WorkerWithDpRank::new(WORKER, 0))
            );
            if event_order != 1 {
                assert!(
                    lifecycle
                        .open(&opening)
                        .await
                        .unwrap_err()
                        .is::<AlreadyRecorded>()
                );
            }
            assert_eq!(active_requests(&lifecycle).await, 0);
        }
    }
    #[tokio::test]
    async fn b10_non_uuid_attempts_cannot_alias_confirmed_scheduler_ids() {
        let lifecycle = make_lifecycle().await;
        for bad_id in ["a", "a:confirmed:2", ""] {
            let mut request = commit("a1", "r1", ENDPOINT);
            request.attempt_id = bad_id.into();
            assert!(lifecycle.intend(&request).await.is_err());
            let mut opening = admitted("a1");
            opening.attempt_id = bad_id.into();
            assert!(lifecycle.open(&opening).await.is_err());
            let mut ending = finish("a1", pb::response_finish_request::Reason::Completed);
            ending.attempt_id = bad_id.into();
            assert!(lifecycle.finish(&ending).await.is_err());
        }
        assert!(lifecycle.attempts.is_empty());
        assert_eq!(active_requests(&lifecycle).await, 0);
    }

    #[tokio::test]
    async fn b10_invalid_failure_kind_does_not_poison_a_corrected_notification() {
        let lifecycle = make_lifecycle().await;
        lifecycle
            .intend(&commit("a1", "r1", ENDPOINT))
            .await
            .unwrap();
        for kind in [0, 999] {
            let error = lifecycle
                .open(&pb::ResponseOpenOutcomeRequest {
                    attempt_id: attempt_id("a1"),
                    outcome: Some(pb::response_open_outcome_request::Outcome::Failed(
                        pb::OpenFailure {
                            kind,
                            ..Default::default()
                        },
                    )),
                })
                .await
                .expect_err("invalid kind rejected");
            assert!(error.downcast_ref::<InvalidAttempt>().is_some());
        }
        assert_eq!(active_requests(&lifecycle).await, 1);
        lifecycle
            .open(&pb::ResponseOpenOutcomeRequest {
                attempt_id: attempt_id("a1"),
                outcome: Some(pb::response_open_outcome_request::Outcome::Failed(
                    pb::OpenFailure {
                        kind: pb::open_failure::Kind::Transport as i32,
                        ..Default::default()
                    },
                )),
            })
            .await
            .unwrap();
        assert_eq!(active_requests(&lifecycle).await, 0);
    }

    #[tokio::test]
    async fn b10_split_admission_tracks_decoder_but_indexes_prefill_worker() {
        let lifecycle = make_lifecycle().await;
        let mut snapshot = (*lifecycle.core.topology.load()).clone();
        snapshot.workers.insert(
            WORKER + 1,
            TopologyWorker {
                endpoint: EndpointId(ENDPOINT.into()),
                runtime: Default::default(),
                observed_load: None,
            },
        );
        lifecycle
            .core
            .router
            .reconcile_topology(&snapshot, &snapshot.workers.keys().copied().collect())
            .await
            .unwrap();
        lifecycle.core.topology.replace(snapshot);
        let mut request = commit("split", "r1", ENDPOINT);
        request
            .input
            .as_mut()
            .unwrap()
            .bid
            .as_mut()
            .unwrap()
            .session_id = Some("split-session".into());
        lifecycle.intend(&request).await.unwrap();
        let mut opening = admitted("split");
        let Some(pb::response_open_outcome_request::Outcome::Admitted(admission)) =
            opening.outcome.as_mut()
        else {
            unreachable!()
        };
        admission.decode_worker_id = Some(WORKER + 1);
        admission.decode_dp_rank = Some(0);
        lifecycle.open(&opening).await.unwrap();
        assert!(
            lifecycle
                .open(&opening)
                .await
                .unwrap_err()
                .is::<AlreadyRecorded>()
        );
        let router = lifecycle
            .core
            .router
            .model_router(&OracleVersionId::new("m").unwrap())
            .unwrap();
        let loads = router.potential_loads(TOKENS).await;
        assert_eq!(
            loads
                .iter()
                .find(|load| load.worker_id == WORKER)
                .unwrap()
                .active_requests,
            0
        );
        assert_eq!(
            loads
                .iter()
                .find(|load| load.worker_id == WORKER + 1)
                .unwrap()
                .active_requests,
            1
        );
        assert_eq!(
            router
                .cached_tokens(
                    TOKENS,
                    &Default::default(),
                    WorkerWithDpRank::new(WORKER, 0)
                )
                .await,
            TOKENS.len()
        );
        assert_eq!(
            router
                .cached_tokens(
                    TOKENS,
                    &Default::default(),
                    WorkerWithDpRank::new(WORKER + 1, 0)
                )
                .await,
            0
        );
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if let Some(worker) = lifecycle.core.affinity.get("split-session").await.unwrap() {
                    assert_eq!(worker.worker_id, WORKER);
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        lifecycle
            .finish(&finish(
                "split",
                pb::response_finish_request::Reason::Completed,
            ))
            .await
            .unwrap();
        assert_eq!(active_requests(&lifecycle).await, 0);
    }

    #[tokio::test]
    async fn b10_rejected_early_or_decode_admission_accepts_a_corrected_outcome() {
        for early in [false, true] {
            for decoder in [false, true] {
                let lifecycle = make_lifecycle().await;
                let mut snapshot = (*lifecycle.core.topology.load()).clone();
                snapshot.workers.insert(
                    WORKER + 1,
                    TopologyWorker {
                        endpoint: EndpointId("other-endpoint".into()),
                        runtime: Default::default(),
                        observed_load: None,
                    },
                );
                lifecycle.core.topology.replace(snapshot);
                let mut opening = admitted("recover");
                let Some(pb::response_open_outcome_request::Outcome::Admitted(admission)) =
                    opening.outcome.as_mut()
                else {
                    unreachable!()
                };
                if decoder {
                    admission.decode_worker_id = Some(WORKER + 1);
                    admission.decode_dp_rank = Some(0);
                } else {
                    admission.prefill_worker_id = WORKER + 1;
                }
                let request = commit("recover", "r1", ENDPOINT);
                if early {
                    lifecycle.open(&opening).await.unwrap();
                    assert!(lifecycle.intend(&request).await.is_err());
                } else {
                    lifecycle.intend(&request).await.unwrap();
                    assert!(lifecycle.open(&opening).await.is_err());
                }
                assert!(
                    lifecycle
                        .attempt(&attempt_id("recover"))
                        .lock()
                        .await
                        .outcome
                        .is_none()
                );
                assert_eq!(active_requests(&lifecycle).await, usize::from(!early));
                if early {
                    assert!(!lifecycle.attempt(&request.attempt_id).lock().await.booked);
                    assert!(
                        lifecycle
                            .attempt(&request.attempt_id)
                            .lock()
                            .await
                            .commit
                            .is_none()
                    );
                    assert_eq!(active_requests(&lifecycle).await, 0);
                }
                lifecycle.open(&admitted("recover")).await.unwrap();
                if early {
                    lifecycle.intend(&request).await.unwrap();
                } else {
                    assert!(
                        lifecycle
                            .intend(&request)
                            .await
                            .unwrap_err()
                            .is::<AlreadyCommitted>()
                    );
                }
                assert_eq!(active_requests(&lifecycle).await, 1);
                lifecycle
                    .finish(&finish(
                        "recover",
                        pb::response_finish_request::Reason::Completed,
                    ))
                    .await
                    .unwrap();
                assert_eq!(active_requests(&lifecycle).await, 0);
            }
        }
    }

    #[tokio::test]
    async fn b10_deferred_worker_confirmation_is_atomic() {
        let lifecycle = make_lifecycle().await;
        let mut opening = admitted("confirm-pair");
        let Some(pb::response_open_outcome_request::Outcome::Admitted(admission)) =
            opening.outcome.as_mut()
        else {
            unreachable!()
        };
        admission.prefill_worker_id = WORKER + 1;
        admission.decode_worker_id = Some(WORKER + 2);
        admission.decode_dp_rank = Some(0);
        lifecycle.open(&opening).await.unwrap();

        let mut snapshot = (*lifecycle.core.topology.load()).clone();
        snapshot.workers.insert(
            WORKER + 2,
            TopologyWorker {
                endpoint: EndpointId("other-endpoint".into()),
                runtime: Default::default(),
                observed_load: None,
            },
        );
        lifecycle.core.topology.replace(snapshot);
        let request = commit("confirm-pair", "r1", ENDPOINT);
        assert!(lifecycle.intend(&request).await.is_err());
        assert!(
            lifecycle
                .core
                .topology
                .load()
                .worker_endpoint(WORKER + 1)
                .is_none()
        );
        assert_eq!(active_requests(&lifecycle).await, 0);
        assert!(
            lifecycle
                .attempt(&request.attempt_id)
                .lock()
                .await
                .outcome
                .is_none()
        );
        lifecycle.open(&admitted("confirm-pair")).await.unwrap();
        lifecycle.intend(&request).await.unwrap();
        assert_eq!(active_requests(&lifecycle).await, 1);
        lifecycle
            .finish(&finish(
                "confirm-pair",
                pb::response_finish_request::Reason::Completed,
            ))
            .await
            .unwrap();
        assert_eq!(active_requests(&lifecycle).await, 0);
    }

    #[tokio::test]
    async fn b10_deferred_confirmation_revalidates_live_dp_ranges() {
        let lifecycle = make_lifecycle().await;
        let mut opening = admitted("confirm-rank");
        let Some(pb::response_open_outcome_request::Outcome::Admitted(admission)) =
            opening.outcome.as_mut()
        else {
            unreachable!()
        };
        admission.prefill_worker_id = WORKER + 1;
        admission.decode_worker_id = Some(WORKER + 2);
        admission.decode_dp_rank = Some(0);
        lifecycle.open(&opening).await.unwrap();

        let mut snapshot = (*lifecycle.core.topology.load()).clone();
        let mut runtime = snapshot.workers[&WORKER].runtime.clone();
        runtime.data_parallel_start_rank = 1;
        runtime.data_parallel_size = 1;
        snapshot.workers.insert(
            WORKER + 2,
            TopologyWorker {
                endpoint: EndpointId(ENDPOINT.into()),
                runtime,
                observed_load: None,
            },
        );
        lifecycle.core.topology.replace(snapshot);
        let request = commit("confirm-rank", "r1", ENDPOINT);
        assert!(lifecycle.intend(&request).await.is_err());
        assert!(
            lifecycle
                .core
                .topology
                .load()
                .worker_endpoint(WORKER + 1)
                .is_none()
        );
        assert_eq!(active_requests(&lifecycle).await, 0);
        assert!(
            lifecycle
                .attempt(&request.attempt_id)
                .lock()
                .await
                .outcome
                .is_none()
        );
        lifecycle.open(&admitted("confirm-rank")).await.unwrap();
        lifecycle.intend(&request).await.unwrap();
        assert_eq!(active_requests(&lifecycle).await, 1);
        lifecycle
            .finish(&finish(
                "confirm-rank",
                pb::response_finish_request::Reason::Completed,
            ))
            .await
            .unwrap();
        assert_eq!(active_requests(&lifecycle).await, 0);
    }

    #[tokio::test]
    async fn b10_invalid_admitted_dp_ranks_do_not_hide_load_or_poison_outcomes() {
        for terminal in [false, true] {
            for decoder in [false, true] {
                for rank in [1, 4] {
                    let lifecycle = make_lifecycle().await;
                    let mut snapshot = (*lifecycle.core.topology.load()).clone();
                    let runtime = &mut snapshot.workers.get_mut(&WORKER).unwrap().runtime;
                    runtime.data_parallel_start_rank = 2;
                    runtime.data_parallel_size = 2;
                    lifecycle
                        .core
                        .router
                        .reconcile_topology(&snapshot, &snapshot.workers.keys().copied().collect())
                        .await
                        .unwrap();
                    lifecycle.core.topology.replace(snapshot);
                    let mut request = commit("dp-recover", "r1", ENDPOINT);
                    let Some(pb::commit_intend_request::ChosenPlacement::Chosen(chosen)) =
                        request.chosen_placement.as_mut()
                    else {
                        unreachable!()
                    };
                    chosen.dp_rank = 2;
                    lifecycle.intend(&request).await.unwrap();
                    if terminal {
                        lifecycle
                            .finish(&finish(
                                "dp-recover",
                                pb::response_finish_request::Reason::Completed,
                            ))
                            .await
                            .unwrap();
                    }
                    let mut opening = admitted("dp-recover");
                    let Some(pb::response_open_outcome_request::Outcome::Admitted(admission)) =
                        opening.outcome.as_mut()
                    else {
                        unreachable!()
                    };
                    admission.prefill_dp_rank = if decoder { 2 } else { rank };
                    admission.decode_worker_id = Some(WORKER);
                    admission.decode_dp_rank = Some(if decoder { rank } else { 2 });
                    let error = lifecycle
                        .open(&opening)
                        .await
                        .expect_err("out-of-range rank rejected");
                    assert!(error.downcast_ref::<InvalidAttempt>().is_some());
                    assert!(
                        lifecycle
                            .attempt(&attempt_id("dp-recover"))
                            .lock()
                            .await
                            .outcome
                            .is_none()
                    );
                    assert_eq!(active_requests(&lifecycle).await, usize::from(!terminal));
                    let Some(pb::response_open_outcome_request::Outcome::Admitted(admission)) =
                        opening.outcome.as_mut()
                    else {
                        unreachable!()
                    };
                    admission.prefill_dp_rank = 2;
                    admission.decode_dp_rank = Some(3);
                    lifecycle.open(&opening).await.unwrap();
                    let router = lifecycle
                        .core
                        .router
                        .model_router(&OracleVersionId::new("m").unwrap())
                        .unwrap();
                    let loads = router.potential_loads(TOKENS).await;
                    assert_eq!(
                        loads
                            .iter()
                            .find(|load| load.dp_rank == 2)
                            .unwrap()
                            .active_requests,
                        0
                    );
                    assert_eq!(
                        loads
                            .iter()
                            .find(|load| load.dp_rank == 3)
                            .unwrap()
                            .active_requests,
                        usize::from(!terminal)
                    );
                    assert_eq!(
                        router
                            .cached_tokens(
                                TOKENS,
                                &Default::default(),
                                WorkerWithDpRank::new(WORKER, 2)
                            )
                            .await,
                        TOKENS.len()
                    );
                    lifecycle
                        .finish(&finish(
                            "dp-recover",
                            pb::response_finish_request::Reason::Completed,
                        ))
                        .await
                        .unwrap();
                    assert_eq!(active_requests(&lifecycle).await, 0);
                }
            }
        }
    }

    #[tokio::test]
    async fn b10_malformed_denials_leave_the_attempt_correctable() {
        use pb::denied_request::Kind;
        let malformed = [
            None,
            Some(Kind::Unspecified as i32),
            Some(999),
            Some(Kind::RouterBackpressure as i32),
            Some(Kind::RequiredComponentsDown as i32),
            Some(Kind::NextRouterUnreachable as i32),
            Some(Kind::ProtocolError as i32),
            Some(Kind::FirstWorkerEventFailed as i32),
            Some(Kind::WorkerErrorResponse as i32),
        ];
        for early in [false, true] {
            for kind in malformed {
                let lifecycle = make_lifecycle().await;
                let request = commit("denial-recover", "r1", ENDPOINT);
                if !early {
                    lifecycle.intend(&request).await.unwrap();
                }
                let mut opening = pb::ResponseOpenOutcomeRequest {
                    attempt_id: request.attempt_id.clone(),
                    outcome: Some(pb::response_open_outcome_request::Outcome::Denied(
                        pb::DeniedGeneration {
                            denied: kind.map(|kind| pb::DeniedRequest {
                                kind,
                                ..Default::default()
                            }),
                            admission: None,
                        },
                    )),
                };
                assert!(
                    lifecycle
                        .open(&opening)
                        .await
                        .unwrap_err()
                        .downcast_ref::<InvalidAttempt>()
                        .is_some()
                );
                {
                    let attempt = lifecycle.attempt(&request.attempt_id);
                    let attempt = attempt.lock().await;
                    assert!(attempt.outcome.is_none());
                    assert!(attempt.terminal_at.is_none());
                }
                assert_eq!(active_requests(&lifecycle).await, usize::from(!early));
                let Some(pb::response_open_outcome_request::Outcome::Denied(denied)) =
                    opening.outcome.as_mut()
                else {
                    unreachable!()
                };
                denied.denied = Some(pb::DeniedRequest {
                    kind: Kind::Cancelled as i32,
                    ..Default::default()
                });
                lifecycle.open(&opening).await.unwrap();
                assert_eq!(active_requests(&lifecycle).await, 0);
                assert!(lifecycle.intend(&request).await.is_err());
            }
        }
    }

    #[tokio::test]
    async fn b10_invalid_late_commits_do_not_retain_context_or_confirm_cache() {
        for denial in [false, true] {
            for malformed in 0..6 {
                let lifecycle = make_lifecycle().await;
                let request = commit("terminal-context", "r1", ENDPOINT);
                let mut opening = admitted("terminal-context");
                if denial {
                    let Some(pb::response_open_outcome_request::Outcome::Admitted(admission)) =
                        opening.outcome
                    else {
                        unreachable!()
                    };
                    opening.outcome = Some(pb::response_open_outcome_request::Outcome::Denied(
                        pb::DeniedGeneration {
                            denied: Some(pb::DeniedRequest {
                                kind: pb::denied_request::Kind::Cancelled as i32,
                                ..Default::default()
                            }),
                            admission: Some(admission),
                        },
                    ));
                } else {
                    lifecycle
                        .finish(&finish(
                            "terminal-context",
                            pb::response_finish_request::Reason::Completed,
                        ))
                        .await
                        .unwrap();
                }
                lifecycle.open(&opening).await.unwrap();
                let mut invalid = request.clone();
                match malformed {
                    0 => invalid
                        .input
                        .as_mut()
                        .unwrap()
                        .bid
                        .as_mut()
                        .unwrap()
                        .tokens
                        .clear(),
                    1 => invalid.input.as_mut().unwrap().allowed_worker_ids = vec![WORKER + 1],
                    2 => invalid.input.as_mut().unwrap().model_id = "unknown".into(),
                    3 => invalid.chosen_placement = None,
                    4 | 5 => {
                        let Some(pb::commit_intend_request::ChosenPlacement::Chosen(chosen)) =
                            invalid.chosen_placement.as_mut()
                        else {
                            unreachable!()
                        };
                        if malformed == 4 {
                            chosen.endpoint_id = "other-endpoint".into();
                        } else {
                            chosen.dp_rank = 1;
                        }
                    }
                    _ => unreachable!(),
                }
                assert!(
                    lifecycle
                        .intend(&invalid)
                        .await
                        .unwrap_err()
                        .downcast_ref::<InvalidAttempt>()
                        .is_some()
                );
                assert!(
                    lifecycle
                        .attempt(&request.attempt_id)
                        .lock()
                        .await
                        .commit
                        .is_none()
                );
                let router = lifecycle
                    .core
                    .router
                    .model_router(&OracleVersionId::new("m").unwrap())
                    .unwrap();
                assert_eq!(
                    router
                        .cached_tokens(
                            TOKENS,
                            &Default::default(),
                            WorkerWithDpRank::new(WORKER, 0)
                        )
                        .await,
                    0
                );
                assert_eq!(active_requests(&lifecycle).await, 0);
                assert!(lifecycle.intend(&request).await.is_err());
                assert!(
                    lifecycle
                        .attempt(&request.attempt_id)
                        .lock()
                        .await
                        .commit
                        .is_some()
                );
                assert_eq!(
                    router
                        .cached_tokens(
                            TOKENS,
                            &Default::default(),
                            WorkerWithDpRank::new(WORKER, 0)
                        )
                        .await,
                    TOKENS.len()
                );
                assert_eq!(active_requests(&lifecycle).await, 0);
            }
        }
    }

    #[tokio::test]
    async fn b10_invalid_embedded_admission_does_not_terminate_a_rejected_denial() {
        for early in [false, true] {
            for malformed in 0..6 {
                if early && malformed >= 4 {
                    continue;
                }
                let lifecycle = make_lifecycle().await;
                let mut snapshot = (*lifecycle.core.topology.load()).clone();
                snapshot.workers.insert(
                    WORKER + 1,
                    TopologyWorker {
                        endpoint: EndpointId("other-endpoint".into()),
                        runtime: Default::default(),
                        observed_load: None,
                    },
                );
                lifecycle.core.topology.replace(snapshot);
                let request = commit("denial-admission", "r1", ENDPOINT);
                if !early {
                    lifecycle.intend(&request).await.unwrap();
                }
                let Some(pb::response_open_outcome_request::Outcome::Admitted(mut admission)) =
                    admitted("denial-admission").outcome
                else {
                    unreachable!()
                };
                match malformed {
                    0 => admission.decode_worker_id = Some(WORKER),
                    1 => admission.decode_dp_rank = Some(0),
                    2 => admission.prefill_dp_rank = 1,
                    3 => {
                        admission.decode_worker_id = Some(WORKER);
                        admission.decode_dp_rank = Some(1);
                    }
                    4 => admission.prefill_worker_id = WORKER + 1,
                    5 => {
                        admission.decode_worker_id = Some(WORKER + 1);
                        admission.decode_dp_rank = Some(0);
                    }
                    _ => unreachable!(),
                }
                let opening = pb::ResponseOpenOutcomeRequest {
                    attempt_id: request.attempt_id.clone(),
                    outcome: Some(pb::response_open_outcome_request::Outcome::Denied(
                        pb::DeniedGeneration {
                            denied: Some(pb::DeniedRequest {
                                kind: pb::denied_request::Kind::Cancelled as i32,
                                ..Default::default()
                            }),
                            admission: Some(admission),
                        },
                    )),
                };
                assert!(
                    lifecycle
                        .open(&opening)
                        .await
                        .unwrap_err()
                        .downcast_ref::<InvalidAttempt>()
                        .is_some()
                );
                let attempt = lifecycle.attempt(&request.attempt_id);
                let attempt = attempt.lock().await;
                assert!(attempt.terminal_at.is_none());
                assert!(attempt.outcome.is_none());
                drop(attempt);
                assert_eq!(active_requests(&lifecycle).await, usize::from(!early));
                lifecycle.open(&admitted("denial-admission")).await.unwrap();
                if early {
                    lifecycle.intend(&request).await.unwrap();
                }
                assert_eq!(active_requests(&lifecycle).await, 1);
                lifecycle
                    .finish(&finish(
                        "denial-admission",
                        pb::response_finish_request::Reason::Completed,
                    ))
                    .await
                    .unwrap();
                assert_eq!(active_requests(&lifecycle).await, 0);
            }
        }
    }

    #[tokio::test]
    async fn b10_denial_reconciliation_keeps_its_validated_topology_snapshot() {
        for change in 0..3 {
            let lifecycle = make_lifecycle().await;
            let request = commit("snapshot-denial", "r1", ENDPOINT);
            lifecycle.intend(&request).await.unwrap();
            let Some(pb::response_open_outcome_request::Outcome::Admitted(admission)) =
                admitted("snapshot-denial").outcome
            else {
                unreachable!()
            };
            let record = lifecycle.attempt(&request.attempt_id);
            let mut attempt = record.lock().await;
            let prepared = lifecycle
                .prepare_admission(&admission, attempt.commit.as_ref())
                .unwrap();
            let router = prepared.router.as_ref().unwrap().clone();
            let mut snapshot = (*lifecycle.core.topology.load()).clone();
            let other_endpoint = snapshot.endpoints[&EndpointId(ENDPOINT.into())].clone();
            snapshot
                .endpoints
                .insert(EndpointId("other-endpoint".into()), other_endpoint);
            let worker = snapshot.workers.get_mut(&WORKER).unwrap();
            match change {
                0 => worker.runtime.data_parallel_start_rank = 2,
                1 => worker.endpoint = EndpointId("other-endpoint".into()),
                2 => {
                    snapshot.models.remove("m");
                    lifecycle.core.router.retire_absent(&snapshot);
                    assert!(
                        lifecycle
                            .core
                            .router
                            .model_router(&OracleVersionId::new("m").unwrap())
                            .is_none()
                    );
                }
                _ => unreachable!(),
            }
            lifecycle.core.topology.replace(snapshot);
            attempt.outcome = Some(pb::response_open_outcome_request::Outcome::Denied(
                pb::DeniedGeneration {
                    denied: Some(pb::DeniedRequest {
                        kind: pb::denied_request::Kind::Cancelled as i32,
                        ..Default::default()
                    }),
                    admission: Some(admission),
                },
            ));
            attempt.terminal_at = Some(Instant::now());
            lifecycle
                .core
                .response_started(&GwpLifecycle::rid(&request.attempt_id), None, 499)
                .await;
            lifecycle
                .apply_admission(&request.attempt_id, &mut attempt, Some(prepared))
                .await
                .unwrap();
            assert!(attempt.admission_recorded);
            let opening = pb::ResponseOpenOutcomeRequest {
                attempt_id: request.attempt_id.clone(),
                outcome: attempt.outcome.clone(),
            };
            assert!(attempt.outcome.is_some());
            drop(attempt);
            assert_eq!(router.pending_count(), 0);
            assert!(
                router
                    .potential_loads(TOKENS)
                    .await
                    .iter()
                    .all(|load| load.active_requests == 0)
            );
            assert!(
                lifecycle
                    .open(&opening)
                    .await
                    .unwrap_err()
                    .is::<AlreadyRecorded>()
            );
            assert_eq!(
                router
                    .cached_tokens(
                        TOKENS,
                        &Default::default(),
                        WorkerWithDpRank::new(WORKER, 0)
                    )
                    .await,
                TOKENS.len()
            );
        }
    }

    #[tokio::test]
    async fn b10_repeated_precommit_openings_are_rejected_without_revalidation() {
        for denial in [false, true] {
            for removed in [false, true] {
                let lifecycle = make_lifecycle().await;
                let mut original = (*lifecycle.core.topology.load()).clone();
                let runtime = &mut original.workers.get_mut(&WORKER).unwrap().runtime;
                runtime.data_parallel_start_rank = 2;
                runtime.data_parallel_size = 2;
                lifecycle
                    .core
                    .router
                    .reconcile_topology(&original, &original.workers.keys().copied().collect())
                    .await
                    .unwrap();
                lifecycle.core.topology.replace(original.clone());
                let mut request = commit("pending-replay", "r1", ENDPOINT);
                let Some(pb::commit_intend_request::ChosenPlacement::Chosen(chosen)) =
                    request.chosen_placement.as_mut()
                else {
                    unreachable!()
                };
                chosen.dp_rank = 2;
                let Some(pb::response_open_outcome_request::Outcome::Admitted(mut admission)) =
                    admitted("pending-replay").outcome
                else {
                    unreachable!()
                };
                admission.prefill_dp_rank = 2;
                let outcome = if denial {
                    pb::response_open_outcome_request::Outcome::Denied(pb::DeniedGeneration {
                        denied: Some(pb::DeniedRequest {
                            kind: pb::denied_request::Kind::Cancelled as i32,
                            ..Default::default()
                        }),
                        admission: Some(admission),
                    })
                } else {
                    pb::response_open_outcome_request::Outcome::Admitted(admission)
                };
                let opening = pb::ResponseOpenOutcomeRequest {
                    attempt_id: request.attempt_id.clone(),
                    outcome: Some(outcome),
                };
                lifecycle.open(&opening).await.unwrap();
                let mut changed = original.clone();
                if removed {
                    changed.workers.remove(&WORKER);
                } else {
                    changed
                        .workers
                        .get_mut(&WORKER)
                        .unwrap()
                        .runtime
                        .data_parallel_start_rank = 4;
                }
                lifecycle.core.topology.replace(changed);
                assert!(lifecycle.prepare_admission(&admission, None).is_err());
                assert!(
                    lifecycle
                        .open(&opening)
                        .await
                        .unwrap_err()
                        .is::<AlreadyRecorded>()
                );
                assert_eq!(
                    lifecycle.attempt(&request.attempt_id).lock().await.outcome,
                    opening.outcome
                );
                assert_eq!(active_requests(&lifecycle).await, 0);
                lifecycle.core.topology.replace(original);
                let result = lifecycle.intend(&request).await;
                assert_eq!(result.is_err(), denial);
                assert!(
                    lifecycle
                        .attempt(&request.attempt_id)
                        .lock()
                        .await
                        .admission_recorded
                );
                assert_eq!(active_requests(&lifecycle).await, usize::from(!denial));
                lifecycle
                    .finish(&finish(
                        "pending-replay",
                        pb::response_finish_request::Reason::Completed,
                    ))
                    .await
                    .unwrap();
                assert_eq!(active_requests(&lifecycle).await, 0);
            }
        }
    }

    #[tokio::test]
    async fn b10_late_commit_uses_accepted_prefill_snapshot_with_a_stable_choice() {
        for denial in [false, true] {
            let lifecycle = make_lifecycle().await;
            let mut snapshot = (*lifecycle.core.topology.load()).clone();
            let runtime = dynamo_llm::local_model::runtime_config::ModelRuntimeConfig {
                data_parallel_start_rank: 2,
                data_parallel_size: 2,
                ..Default::default()
            };
            snapshot.workers.insert(
                WORKER + 1,
                TopologyWorker {
                    endpoint: EndpointId(ENDPOINT.into()),
                    runtime,
                    observed_load: None,
                },
            );
            lifecycle
                .core
                .router
                .reconcile_topology(&snapshot, &snapshot.workers.keys().copied().collect())
                .await
                .unwrap();
            lifecycle.core.topology.replace(snapshot.clone());
            let request = commit("late-snapshot", "r1", ENDPOINT);
            let Some(pb::response_open_outcome_request::Outcome::Admitted(mut admission)) =
                admitted("late-snapshot").outcome
            else {
                unreachable!()
            };
            admission.prefill_worker_id = WORKER + 1;
            admission.prefill_dp_rank = 2;
            admission.decode_worker_id = Some(WORKER);
            admission.decode_dp_rank = Some(0);
            let outcome = if denial {
                pb::response_open_outcome_request::Outcome::Denied(pb::DeniedGeneration {
                    denied: Some(pb::DeniedRequest {
                        kind: pb::denied_request::Kind::Cancelled as i32,
                        ..Default::default()
                    }),
                    admission: Some(admission),
                })
            } else {
                pb::response_open_outcome_request::Outcome::Admitted(admission)
            };
            let opening = pb::ResponseOpenOutcomeRequest {
                attempt_id: request.attempt_id.clone(),
                outcome: Some(outcome),
            };
            lifecycle.open(&opening).await.unwrap();
            snapshot
                .workers
                .get_mut(&(WORKER + 1))
                .unwrap()
                .runtime
                .data_parallel_start_rank = 4;
            lifecycle
                .core
                .router
                .reconcile_topology(&snapshot, &snapshot.workers.keys().copied().collect())
                .await
                .unwrap();
            lifecycle.core.topology.replace(snapshot);
            assert!(
                lifecycle
                    .open(&opening)
                    .await
                    .unwrap_err()
                    .is::<AlreadyRecorded>()
            );
            assert_eq!(lifecycle.intend(&request).await.is_err(), denial);
            let record = lifecycle.attempt(&request.attempt_id);
            let attempt = record.lock().await;
            assert!(attempt.admission_recorded);
            assert!(attempt.prepared_admission.is_none());
            assert_eq!(attempt.outcome, opening.outcome);
            drop(attempt);
            let router = lifecycle
                .core
                .router
                .model_router(&OracleVersionId::new("m").unwrap())
                .unwrap();
            assert_eq!(
                router
                    .cached_tokens(
                        TOKENS,
                        &Default::default(),
                        WorkerWithDpRank::new(WORKER + 1, 2)
                    )
                    .await,
                TOKENS.len()
            );
            assert_eq!(active_requests(&lifecycle).await, usize::from(!denial));
            assert!(
                router
                    .potential_loads(TOKENS)
                    .await
                    .iter()
                    .filter(|load| load.worker_id == WORKER + 1)
                    .all(|load| load.active_requests == 0)
            );
            lifecycle
                .finish(&finish(
                    "late-snapshot",
                    pb::response_finish_request::Reason::Completed,
                ))
                .await
                .unwrap();
            assert_eq!(router.pending_count(), 0);
            assert_eq!(active_requests(&lifecycle).await, 0);
        }
    }

    #[tokio::test]
    async fn b10_deferred_denial_ownership_rejection_preserves_independent_termination() {
        for termination in 0..3 {
            let lifecycle = make_lifecycle().await;
            let mut snapshot = (*lifecycle.core.topology.load()).clone();
            let other = snapshot.endpoints[&EndpointId(ENDPOINT.into())].clone();
            snapshot
                .endpoints
                .insert(EndpointId("other-endpoint".into()), other);
            snapshot.workers.insert(
                WORKER + 1,
                TopologyWorker {
                    endpoint: EndpointId("other-endpoint".into()),
                    runtime: Default::default(),
                    observed_load: None,
                },
            );
            lifecycle.core.topology.replace(snapshot);
            let request = commit("pending-denial", "r1", ENDPOINT);
            let Some(pb::response_open_outcome_request::Outcome::Admitted(mut admission)) =
                admitted("pending-denial").outcome
            else {
                unreachable!()
            };
            admission.prefill_worker_id = WORKER + 1;
            lifecycle
                .open(&pb::ResponseOpenOutcomeRequest {
                    attempt_id: request.attempt_id.clone(),
                    outcome: Some(pb::response_open_outcome_request::Outcome::Denied(
                        pb::DeniedGeneration {
                            denied: Some(pb::DeniedRequest {
                                kind: pb::denied_request::Kind::Cancelled as i32,
                                ..Default::default()
                            }),
                            admission: Some(admission),
                        },
                    )),
                })
                .await
                .unwrap();
            assert!(
                lifecycle
                    .attempt(&request.attempt_id)
                    .lock()
                    .await
                    .terminal_at
                    .is_none()
            );
            match termination {
                0 => {}
                1 => lifecycle
                    .finish(&finish(
                        "pending-denial",
                        pb::response_finish_request::Reason::Cancelled,
                    ))
                    .await
                    .unwrap(),
                2 => {
                    lifecycle
                        .attempt(&request.attempt_id)
                        .lock()
                        .await
                        .created_at = Instant::now() - INFLIGHT_MAX_AGE;
                    GwpLifecycle::sweep(&lifecycle.core, &lifecycle.attempts).await;
                }
                _ => unreachable!(),
            }
            assert!(
                lifecycle
                    .intend(&request)
                    .await
                    .unwrap_err()
                    .downcast_ref::<InvalidAttempt>()
                    .is_some()
            );
            assert_eq!(active_requests(&lifecycle).await, 0);
            let record = lifecycle.attempt(&request.attempt_id);
            let attempt = record.lock().await;
            assert!(attempt.outcome.is_none());
            assert_eq!(attempt.terminal_at.is_some(), termination != 0);
            drop(attempt);
            lifecycle.open(&admitted("pending-denial")).await.unwrap();
            assert_eq!(lifecycle.intend(&request).await.is_err(), termination != 0);
            assert_eq!(
                active_requests(&lifecycle).await,
                usize::from(termination == 0)
            );
            if termination == 0 {
                lifecycle
                    .finish(&finish(
                        "pending-denial",
                        pb::response_finish_request::Reason::Completed,
                    ))
                    .await
                    .unwrap();
            }
            assert_eq!(active_requests(&lifecycle).await, 0);
        }
    }

    #[tokio::test]
    async fn b10_valid_deferred_denial_terminates_without_ever_booking() {
        let lifecycle = make_lifecycle().await;
        let request = commit("valid-deferred-denial", "r1", ENDPOINT);
        let Some(pb::response_open_outcome_request::Outcome::Admitted(admission)) =
            admitted("valid-deferred-denial").outcome
        else {
            unreachable!()
        };
        lifecycle
            .open(&pb::ResponseOpenOutcomeRequest {
                attempt_id: request.attempt_id.clone(),
                outcome: Some(pb::response_open_outcome_request::Outcome::Denied(
                    pb::DeniedGeneration {
                        denied: Some(pb::DeniedRequest {
                            kind: pb::denied_request::Kind::Cancelled as i32,
                            ..Default::default()
                        }),
                        admission: Some(admission),
                    },
                )),
            })
            .await
            .unwrap();
        assert_eq!(active_requests(&lifecycle).await, 0);
        assert!(lifecycle.intend(&request).await.is_err());
        let record = lifecycle.attempt(&request.attempt_id);
        let attempt = record.lock().await;
        assert!(attempt.terminal_at.is_some());
        assert!(attempt.admission_recorded);
        assert!(!attempt.booked);
        drop(attempt);
        assert_eq!(active_requests(&lifecycle).await, 0);
    }

    #[tokio::test]
    async fn b10_cross_endpoint_admission_is_rejected_without_false_confirmation() {
        let lifecycle = make_lifecycle().await;
        let request = commit("a1", "r1", ENDPOINT);
        lifecycle.intend(&request).await.unwrap();
        let mut snapshot = (*lifecycle.core.topology.load()).clone();
        let other_endpoint = snapshot.endpoints[&EndpointId(ENDPOINT.into())].clone();
        snapshot
            .endpoints
            .insert(EndpointId("other-endpoint".into()), other_endpoint);
        snapshot.workers.insert(
            WORKER + 1,
            TopologyWorker {
                endpoint: EndpointId("other-endpoint".into()),
                runtime: Default::default(),
                observed_load: None,
            },
        );
        lifecycle.core.topology.replace(snapshot);
        let mut opening = admitted("a1");
        let Some(pb::response_open_outcome_request::Outcome::Admitted(admission)) =
            opening.outcome.as_mut()
        else {
            unreachable!()
        };
        admission.prefill_worker_id = WORKER + 1;
        let error = lifecycle
            .open(&opening)
            .await
            .expect_err("wrong owner rejected");
        assert!(error.downcast_ref::<InvalidAttempt>().is_some());
        assert!(
            !lifecycle
                .attempt(&attempt_id("a1"))
                .lock()
                .await
                .admission_recorded
        );
        let router = lifecycle
            .core
            .router
            .model_router(&OracleVersionId::new("m").unwrap())
            .unwrap();
        assert_eq!(
            router
                .cached_tokens(
                    TOKENS,
                    &Default::default(),
                    WorkerWithDpRank::new(WORKER, 0)
                )
                .await,
            0
        );
        assert_eq!(active_requests(&lifecycle).await, 1);
        assert!(
            lifecycle
                .attempt(&attempt_id("a1"))
                .lock()
                .await
                .outcome
                .is_none()
        );
        lifecycle.open(&admitted("a1")).await.unwrap();
        assert_eq!(active_requests(&lifecycle).await, 1);
        assert!(
            lifecycle
                .attempt(&attempt_id("a1"))
                .lock()
                .await
                .admission_recorded
        );
        assert_eq!(
            router
                .cached_tokens(
                    TOKENS,
                    &Default::default(),
                    WorkerWithDpRank::new(WORKER, 0)
                )
                .await,
            TOKENS.len()
        );
        lifecycle
            .finish(&finish(
                "a1",
                pb::response_finish_request::Reason::Completed,
            ))
            .await
            .unwrap();
        assert_eq!(active_requests(&lifecycle).await, 0);
    }
}
