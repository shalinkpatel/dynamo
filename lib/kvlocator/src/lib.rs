// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The KVLocator wire contract, backend seams, and gRPC handlers.
//!
//! GWP implements the scorer and attempt lifecycle and serves these handlers
//! on its existing gRPC listener, using its shared readiness and accounting.

mod b10_token_buffer;

pub const FILE_DESCRIPTOR_SET: &[u8] = tonic::include_file_descriptor_set!("kvlocator_descriptor");

pub mod pb {
    tonic::include_proto!("dynamo.b10.generation.v1");
}

/// The lifecycle seam: booking and the attempt lifecycle backend. The
/// handlers drive this trait (CommitIntend, ResponseOpenOutcome,
/// ResponseFinish). The backend owns attempt identity, serialization, and
/// reconciliation; handlers map backend results to gRPC statuses.
pub mod lifecycle {
    use async_trait::async_trait;

    use crate::pb;

    /// A caller error on the attempt path (committing an already-finished
    /// attempt or choosing an ineligible placement):
    /// the service maps it to INVALID_ARGUMENT. AlreadyCommitted and AlreadyRecorded
    /// map to ALREADY_EXISTS; other backend failures map to INTERNAL.
    #[derive(Debug)]
    pub struct InvalidAttempt(pub String);

    impl std::fmt::Display for InvalidAttempt {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    impl std::error::Error for InvalidAttempt {}

    #[derive(Debug)]
    pub struct AlreadyCommitted;

    impl std::fmt::Display for AlreadyCommitted {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(
                "attempt_id already committed; did you retry a previously delivered request?",
            )
        }
    }

    impl std::error::Error for AlreadyCommitted {}

    #[derive(Debug)]
    pub struct AlreadyRecorded;

    impl std::fmt::Display for AlreadyRecorded {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("opening outcome already recorded for attempt_id")
        }
    }

    impl std::error::Error for AlreadyRecorded {}

    /// Booking and attempt lifecycle backend. See the module docs.
    #[async_trait]
    pub trait Lifecycle: Send + Sync {
        /// Intend + book, one step: provisionally book the chosen placement.
        /// A previously committed attempt_id returns AlreadyCommitted, regardless of payload.
        async fn intend(&self, request: &pb::CommitIntendRequest) -> anyhow::Result<()>;

        /// Apply the open outcome. Admitted keeps the attempt active until
        /// finish; denied and failed are terminal and release once.
        /// Subsequent openings return AlreadyRecorded regardless of payload.
        async fn open(&self, request: &pb::ResponseOpenOutcomeRequest) -> anyhow::Result<()>;

        /// End an admitted attempt. Replaying finish for an already-ended
        /// attempt is a no-op regardless of reason.
        async fn finish(&self, request: &pb::ResponseFinishRequest) -> anyhow::Result<()>;
    }
}

/// The scorer seam: read-only placement ranking for the bid phase. Scorers
/// compose (weighted, best-of), and the handlers see only this trait, never
/// a concrete scoring implementation.
pub mod scorer {
    use async_trait::async_trait;

    use crate::pb;

    /// A caller error (unknown model, bad placement versions): the service
    /// maps it to INVALID_ARGUMENT. Everything else in a scorer's anyhow
    /// error is INTERNAL. Scorers raise this instead of plain bail! for
    /// anything the caller could fix by changing the request.
    #[derive(Debug)]
    pub struct InvalidRequest(pub String);

    impl std::fmt::Display for InvalidRequest {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    impl std::error::Error for InvalidRequest {}

    /// Read-only placement ranking. See the module docs.
    #[async_trait]
    pub trait Scorer: Send + Sync {
        /// Rank every eligible placement, ordered by increasing score (a
        /// cost; comparable within one response). Book nothing, change no
        /// state.
        async fn rank(&self, request: &pb::LocateRequest) -> anyhow::Result<pb::LocateResponse>;
    }
}

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use tonic::{Request, Response, Status};
use tracing::Instrument;

use lifecycle::Lifecycle;
use pb::kv_locator_service_server::KvLocatorService as KVLocatorService;
use scorer::Scorer;

/// A Locate carries the full token sequence of a request; keep the same
/// decoding budget the GWP authorization path uses.
pub const GRPC_MAX_MESSAGE_BYTES: usize = 64 * 1024 * 1024;

/// When a call's headers arrived. The handler starts once the body is received
/// and decoded, so its start minus this is what the token payload costs.
#[derive(Clone, Copy)]
pub struct ReceivedAt(pub Instant);

/// Interceptor stamping [`ReceivedAt`]; tonic runs it on the headers, before the body.
pub fn mark_received(mut request: Request<()>) -> Result<Request<()>, Status> {
    request.extensions_mut().insert(ReceivedAt(Instant::now()));
    Ok(request)
}

/// Observes each token-carrying call: method, token count, and time from its
/// headers to the handler's end.
pub type CallObserver = Arc<dyn Fn(&'static str, usize, Duration) + Send + Sync>;

/// Native placement RPCs require readiness; completion RPCs remain available
/// during drain so Alyx can release admitted attempts.
#[derive(Clone)]
pub struct GrpcKVLocator {
    ready: Arc<dyn Fn() -> bool + Send + Sync>,
    scorer: Option<Arc<dyn Scorer>>,
    lifecycle: Option<Arc<dyn Lifecycle>>,
    observer: Option<CallObserver>,
    // Locate and CommitIntend calls in progress on this replica.
    inflight: Arc<AtomicUsize>,
}

impl GrpcKVLocator {
    /// Read readiness directly from the serving process's shared lifecycle.
    pub fn new(ready: impl Fn() -> bool + Send + Sync + 'static) -> Self {
        Self {
            ready: Arc::new(ready),
            scorer: None,
            lifecycle: None,
            observer: None,
            inflight: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Wire the per-call observer (the serving process's metrics).
    pub fn with_observer(mut self, observer: CallObserver) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Run a token-carrying call inside a span with its token count, the time
    /// its body took to arrive and decode, and the calls in flight beside it.
    async fn traced<T>(
        &self,
        method: &'static str,
        tokens: usize,
        received: Option<ReceivedAt>,
        call: impl std::future::Future<Output = T>,
    ) -> T {
        let start = received.map_or_else(Instant::now, |at| at.0);
        // Decrements on drop, so a cancelled call still leaves the count.
        struct Inflight(Arc<AtomicUsize>);
        impl Drop for Inflight {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::Relaxed);
            }
        }
        let inflight = self.inflight.fetch_add(1, Ordering::Relaxed);
        let _guard = Inflight(self.inflight.clone());
        let span = tracing::info_span!(
            "kvlocator.call",
            kvlocator.method = method,
            // i64: tracing-opentelemetry exports only i64 (not usize or u64) as a number.
            kvlocator.tokens = tokens as i64,
            kvlocator.received_ms = start.elapsed().as_secs_f64() * 1e3,
            kvlocator.inflight = inflight as i64,
        );
        let out = call.instrument(span).await;
        if let Some(observer) = &self.observer {
            observer(method, tokens, start.elapsed());
        }
        out
    }

    /// Wire the scorer seam (the bid phase).
    pub fn with_scorer(mut self, scorer: Arc<dyn Scorer>) -> Self {
        self.scorer = Some(scorer);
        self
    }

    /// Wire the lifecycle seam (intend+book, open, finish).
    pub fn with_lifecycle(mut self, lifecycle: Arc<dyn Lifecycle>) -> Self {
        self.lifecycle = Some(lifecycle);
        self
    }

    fn require_ready(&self) -> Result<(), Status> {
        if !(self.ready)() {
            return Err(Status::unavailable(
                "kv locator is not ready (lifecycle still converging)",
            ));
        }
        Ok(())
    }

    fn require_scorer(&self) -> Result<Arc<dyn Scorer>, Status> {
        self.scorer
            .clone()
            .ok_or_else(|| Status::unimplemented("kv locator: no scorer is wired"))
    }

    fn require_lifecycle(&self) -> Result<Arc<dyn Lifecycle>, Status> {
        self.lifecycle
            .clone()
            .ok_or_else(|| Status::unimplemented("kv locator: no lifecycle is wired"))
    }
}

// The span records only the gRPC code; log the reason inside it so traces name the check.
fn map_lifecycle_error(error: anyhow::Error) -> Status {
    if error.is::<lifecycle::AlreadyCommitted>() || error.is::<lifecycle::AlreadyRecorded>() {
        tracing::debug!(%error, "KVLocator call rejected");
        return Status::already_exists(error.to_string());
    }
    tracing::warn!(%error, "KVLocator call rejected");
    match error.downcast_ref::<lifecycle::InvalidAttempt>() {
        Some(invalid) => Status::invalid_argument(&invalid.0),
        None => Status::internal(error.to_string()),
    }
}

#[tonic::async_trait]
impl KVLocatorService for GrpcKVLocator {
    async fn locate(
        &self,
        request: Request<pb::LocateRequest>,
    ) -> Result<Response<pb::LocateResponse>, Status> {
        self.require_ready()?;
        let scorer = self.require_scorer()?;
        let received = request.extensions().get::<ReceivedAt>().copied();
        let request = request.into_inner();
        let tokens = request.bid.as_ref().map_or(0, |bid| bid.token_count());
        match self
            .traced("Locate", tokens, received, scorer.rank(&request))
            .await
        {
            Ok(response) => Ok(Response::new(response)),
            // A caller error the request could have avoided; everything
            // else is ours.
            Err(error) => {
                tracing::warn!(%error, "KVLocator Locate rejected");
                match error.downcast_ref::<scorer::InvalidRequest>() {
                    Some(invalid) => Err(Status::invalid_argument(&invalid.0)),
                    None => Err(Status::internal(error.to_string())),
                }
            }
        }
    }

    async fn commit_intend(
        &self,
        request: Request<pb::CommitIntendRequest>,
    ) -> Result<Response<()>, Status> {
        self.require_ready()?;
        let lifecycle = self.require_lifecycle()?;
        let received = request.extensions().get::<ReceivedAt>().copied();
        let request = request.into_inner();
        if request.attempt_id.is_empty() {
            return Err(Status::invalid_argument("attempt_id must not be empty"));
        }
        let tokens = request
            .input
            .as_ref()
            .and_then(|input| input.bid.as_ref())
            .map_or(0, |bid| bid.token_count());
        self.traced("CommitIntend", tokens, received, lifecycle.intend(&request))
            .await
            .map(|_| Response::new(()))
            .map_err(map_lifecycle_error)
    }

    async fn response_open_outcome(
        &self,
        request: Request<pb::ResponseOpenOutcomeRequest>,
    ) -> Result<Response<()>, Status> {
        let lifecycle = self.require_lifecycle()?;
        let request = request.into_inner();
        if request.attempt_id.is_empty() {
            return Err(Status::invalid_argument("attempt_id must not be empty"));
        }
        lifecycle
            .open(&request)
            .await
            .map(|_| Response::new(()))
            .map_err(map_lifecycle_error)
    }

    async fn response_finish(
        &self,
        request: Request<pb::ResponseFinishRequest>,
    ) -> Result<Response<()>, Status> {
        let lifecycle = self.require_lifecycle()?;
        let request = request.into_inner();
        if request.attempt_id.is_empty() {
            return Err(Status::invalid_argument("attempt_id must not be empty"));
        }
        lifecycle
            .finish(&request)
            .await
            .map(|_| Response::new(()))
            .map_err(map_lifecycle_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A canned scorer for proving the service's error mapping; the real one
    /// lives in dynamo-gwp.
    enum RankOutcome {
        Ok(pb::LocateResponse),
        Invalid(String),
        Internal(String),
    }

    struct FakeScorer {
        outcome: RankOutcome,
    }

    #[async_trait::async_trait]
    impl Scorer for FakeScorer {
        async fn rank(&self, _request: &pb::LocateRequest) -> anyhow::Result<pb::LocateResponse> {
            match &self.outcome {
                RankOutcome::Ok(response) => Ok(response.clone()),
                // Raised natively so the handler's downcast sees it.
                RankOutcome::Invalid(message) => {
                    Err(scorer::InvalidRequest(message.clone()).into())
                }
                RankOutcome::Internal(message) => Err(anyhow::anyhow!(message.clone())),
            }
        }
    }

    #[tokio::test]
    async fn token_calls_report_size_and_time_since_headers() {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let record = seen.clone();
        let locator = GrpcKVLocator::new(|| true)
            .with_scorer(Arc::new(FakeScorer {
                outcome: RankOutcome::Ok(pb::LocateResponse::default()),
            }))
            .with_observer(Arc::new(move |method, tokens, elapsed| {
                record.lock().unwrap().push((method, tokens, elapsed));
            }));
        let mut request = Request::new(pb::LocateRequest {
            bid: Some(pb::BidRequest {
                token_buffer: Some(pb::Tokens {
                    payload: dynamo_tokens::b10_token_buffer::encode(&[1; 300]).into(),
                }),
                ..Default::default()
            }),
            ..Default::default()
        });
        let headers = Instant::now() - Duration::from_millis(50);
        request.extensions_mut().insert(ReceivedAt(headers));
        locator.locate(request).await.unwrap();

        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!((seen[0].0, seen[0].1), ("Locate", 300));
        assert!(
            seen[0].2 >= Duration::from_millis(50),
            "timed from the headers"
        );
        assert_eq!(locator.inflight.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn a_dropped_call_leaves_the_inflight_count() {
        let locator = GrpcKVLocator::new(|| true);
        let call = locator.traced("Locate", 0, None, std::future::pending::<()>());
        let polled = tokio::time::timeout(Duration::from_millis(10), call).await;
        assert!(polled.is_err(), "the call never finishes and is dropped");
        assert_eq!(locator.inflight.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn locate_maps_scorer_results_and_errors_to_statuses() {
        let ok = pb::LocateResponse::default();
        let locator = GrpcKVLocator::new(|| true).with_scorer(Arc::new(FakeScorer {
            outcome: RankOutcome::Ok(ok.clone()),
        }));
        let response = locator
            .locate(Request::new(pb::LocateRequest::default()))
            .await
            .expect("ranked response passes through");
        assert_eq!(response.into_inner(), ok);

        // A caller error maps to INVALID_ARGUMENT.
        let locator = GrpcKVLocator::new(|| true).with_scorer(Arc::new(FakeScorer {
            outcome: RankOutcome::Invalid("bad request".into()),
        }));
        let error = locator
            .locate(Request::new(pb::LocateRequest::default()))
            .await
            .unwrap_err();
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
        assert_eq!(error.message(), "bad request");

        // Anything else is INTERNAL.
        let locator = GrpcKVLocator::new(|| true).with_scorer(Arc::new(FakeScorer {
            outcome: RankOutcome::Internal("scorer exploded".into()),
        }));
        let error = locator
            .locate(Request::new(pb::LocateRequest::default()))
            .await
            .unwrap_err();
        assert_eq!(error.code(), tonic::Code::Internal);
    }

    /// A canned lifecycle for proving the handler mappings; the real one
    /// lives in dynamo-gwp.
    struct FakeLifecycle {
        invalid: Option<String>,
        internal: Option<String>,
    }

    #[async_trait::async_trait]
    impl lifecycle::Lifecycle for FakeLifecycle {
        async fn intend(&self, _request: &pb::CommitIntendRequest) -> anyhow::Result<()> {
            match (&self.invalid, &self.internal) {
                (Some(message), _) => Err(lifecycle::InvalidAttempt(message.clone()).into()),
                (_, Some(message)) => Err(anyhow::anyhow!(message.clone())),
                _ => Ok(()),
            }
        }
        async fn open(&self, _request: &pb::ResponseOpenOutcomeRequest) -> anyhow::Result<()> {
            self.intend(&pb::CommitIntendRequest::default()).await
        }
        async fn finish(&self, _request: &pb::ResponseFinishRequest) -> anyhow::Result<()> {
            self.intend(&pb::CommitIntendRequest::default()).await
        }
    }

    #[tokio::test]
    async fn lifecycle_handlers_validate_and_map_errors() {
        // Field validation is the handler's, before the backend is touched.
        let locator = GrpcKVLocator::new(|| true).with_lifecycle(Arc::new(FakeLifecycle {
            invalid: None,
            internal: None,
        }));
        let error = locator
            .commit_intend(Request::new(pb::CommitIntendRequest::default()))
            .await
            .unwrap_err();
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
        assert!(error.message().contains("attempt_id"));

        // A backend caller error maps to INVALID_ARGUMENT.
        let locator = GrpcKVLocator::new(|| true).with_lifecycle(Arc::new(FakeLifecycle {
            invalid: Some("already finished".into()),
            internal: None,
        }));
        let error = locator
            .response_finish(Request::new(pb::ResponseFinishRequest {
                attempt_id: "a1".into(),
                reason: 1,
            }))
            .await
            .unwrap_err();
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
        assert_eq!(error.message(), "already finished");

        let error = map_lifecycle_error(lifecycle::AlreadyCommitted.into());
        assert_eq!(error.code(), tonic::Code::AlreadyExists);
        assert!(error.message().contains("previously delivered request"));
        let error = map_lifecycle_error(lifecycle::AlreadyRecorded.into());
        assert_eq!(error.code(), tonic::Code::AlreadyExists);
        assert!(error.message().contains("already recorded"));

        // Anything else is INTERNAL.
        let locator = GrpcKVLocator::new(|| true).with_lifecycle(Arc::new(FakeLifecycle {
            invalid: None,
            internal: Some("core exploded".into()),
        }));
        let error = locator
            .response_open_outcome(Request::new(pb::ResponseOpenOutcomeRequest {
                attempt_id: "a1".into(),
                outcome: Some(pb::response_open_outcome_request::Outcome::Denied(
                    pb::DeniedGeneration::default(),
                )),
            }))
            .await
            .unwrap_err();
        assert_eq!(error.code(), tonic::Code::Internal);
    }

    #[tokio::test]
    async fn fails_closed_until_ready_then_reports_missing_seams() {
        let ready = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let shared = ready.clone();
        let locator = GrpcKVLocator::new(move || shared.load(std::sync::atomic::Ordering::Acquire));
        assert_eq!(
            locator
                .locate(Request::new(pb::LocateRequest::default()))
                .await
                .unwrap_err()
                .code(),
            tonic::Code::Unavailable
        );

        // Ready but nothing wired: the seam, not the transport, is missing.
        ready.store(true, std::sync::atomic::Ordering::Release);
        assert_eq!(
            locator
                .locate(Request::new(pb::LocateRequest::default()))
                .await
                .unwrap_err()
                .code(),
            tonic::Code::Unimplemented
        );
        assert_eq!(
            locator
                .commit_intend(Request::new(pb::CommitIntendRequest::default()))
                .await
                .unwrap_err()
                .code(),
            tonic::Code::Unimplemented
        );
    }
}
