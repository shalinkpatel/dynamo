// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The KVLocator wire contract, backend seams, and gRPC handlers.
//!
//! GWP implements the scorer and attempt lifecycle and serves these handlers
//! on its existing gRPC listener, using its shared readiness and accounting.

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

use tonic::{Request, Response, Status};

use lifecycle::Lifecycle;
use pb::kv_locator_service_server::KvLocatorService as KVLocatorService;
use scorer::Scorer;

/// A Locate carries the full token sequence of a request; keep the same
/// decoding budget the GWP authorization path uses.
pub const GRPC_MAX_MESSAGE_BYTES: usize = 64 * 1024 * 1024;

/// Native placement RPCs require readiness; completion RPCs remain available
/// during drain so Alyx can release admitted attempts.
#[derive(Clone)]
pub struct GrpcKVLocator {
    ready: Arc<dyn Fn() -> bool + Send + Sync>,
    scorer: Option<Arc<dyn Scorer>>,
    lifecycle: Option<Arc<dyn Lifecycle>>,
}

impl GrpcKVLocator {
    /// Read readiness directly from the serving process's shared lifecycle.
    pub fn new(ready: impl Fn() -> bool + Send + Sync + 'static) -> Self {
        Self {
            ready: Arc::new(ready),
            scorer: None,
            lifecycle: None,
        }
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

fn map_lifecycle_error(error: anyhow::Error) -> Status {
    if error.is::<lifecycle::AlreadyCommitted>() || error.is::<lifecycle::AlreadyRecorded>() {
        return Status::already_exists(error.to_string());
    }
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
        let request = request.into_inner();
        match scorer.rank(&request).await {
            Ok(response) => Ok(Response::new(response)),
            // A caller error the request could have avoided; everything
            // else is ours.
            Err(error) => match error.downcast_ref::<scorer::InvalidRequest>() {
                Some(invalid) => Err(Status::invalid_argument(&invalid.0)),
                None => Err(Status::internal(error.to_string())),
            },
        }
    }

    async fn commit_intend(
        &self,
        request: Request<pb::CommitIntendRequest>,
    ) -> Result<Response<()>, Status> {
        self.require_ready()?;
        let lifecycle = self.require_lifecycle()?;
        let request = request.into_inner();
        if request.attempt_id.is_empty() {
            return Err(Status::invalid_argument("attempt_id must not be empty"));
        }
        lifecycle
            .intend(&request)
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
