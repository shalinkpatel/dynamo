// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native, synchronous observation of either coordinator boundary.

use crate::{
    DeniedGenerationRequest, GenerationAdmission, GenerationOutcome, GenerationRequest,
    RequestContext,
};
use anyhow::Result;
use dynamo_runtime::pipeline::{EngineStream, ResponseStream};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{StreamExt, future::BoxFuture};
use rmpv::Value;

/// Why observation ended. Stream closure does not imply generation success:
/// observers interpret terminal worker messages themselves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationObservationEnd {
    StreamClosed,
    Denied,
    Error,
    /// The setup future or response stream was dropped, including cancellation.
    Dropped,
}

/// One observer per generation attempt. Callbacks run inline on the task or
/// thread calling, polling, or dropping generation;
/// implementations must not block, enter Python, or retain borrowed payloads.
/// `on_finish` runs exactly once on closure or drop, including dropped unpolled
/// setup futures and streams. Use it for terminal accounting; a consumer-defined
/// `Drop` implementation must not repeat that accounting.
/// A stream error item is delivered to `on_chunk`; setup errors use `on_error`.
/// Panics are not isolated and must not be used for observer error reporting.
pub trait GenerationObserver: Send {
    fn on_admission(&mut self, _admission: &GenerationAdmission) {}
    fn on_chunk(&mut self, _chunk: &Annotated<Value>) {}
    fn on_denied(&mut self, _denied: &DeniedGenerationRequest) {}
    fn on_error(&mut self, _error: &anyhow::Error) {}
    fn on_finish(&mut self, end: GenerationObservationEnd);
}

/// Creates request-local state before setup is polled. Resolve labels and shared
/// handles here without blocking; clean up any partial state before returning an
/// error. An error disables observation for this attempt; generation
/// still dispatches and preserves its original outcome.
pub trait GenerationObserverFactory: Send + Sync {
    fn begin(
        &self,
        context: &RequestContext,
        request: &GenerationRequest,
    ) -> Result<Box<dyn GenerationObserver>>;
}

pub(crate) struct ObserverGuard {
    observer: Box<dyn GenerationObserver>,
    finished: bool,
}

impl ObserverGuard {
    pub(crate) fn new(observer: Box<dyn GenerationObserver>) -> Self {
        Self {
            observer,
            finished: false,
        }
    }
    fn finish(&mut self, end: GenerationObservationEnd) {
        if !self.finished {
            self.finished = true;
            self.observer.on_finish(end);
        }
    }
    fn wrap(
        mut self,
        mut stream: EngineStream<Annotated<Value>>,
    ) -> EngineStream<Annotated<Value>> {
        let context = stream.context();
        let output = async_stream::stream! {
            while let Some(item) = stream.next().await {
                self.observer.on_chunk(&item);
                yield item;
            }
            self.finish(GenerationObservationEnd::StreamClosed);
        };
        ResponseStream::new(Box::pin(output), context)
    }
}

impl Drop for ObserverGuard {
    fn drop(&mut self) {
        self.finish(GenerationObservationEnd::Dropped);
    }
}

pub(crate) fn observe_generation<'a>(
    mut observer: ObserverGuard,
    future: BoxFuture<'a, Result<GenerationOutcome>>,
) -> BoxFuture<'a, Result<GenerationOutcome>> {
    Box::pin(async move {
        match future.await {
            Ok(GenerationOutcome::Connected(mut generated)) => {
                observer.observer.on_admission(&generated.admission);
                generated.stream = observer.wrap(generated.stream);
                Ok(GenerationOutcome::Connected(generated))
            }
            Ok(GenerationOutcome::Denied(denied)) => {
                observer.observer.on_denied(&denied);
                observer.finish(GenerationObservationEnd::Denied);
                Ok(GenerationOutcome::Denied(denied))
            }
            Err(error) => {
                observer.observer.on_error(&error);
                observer.finish(GenerationObservationEnd::Error);
                Err(error)
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DeniedRequest, GeneratedRequest, GenerationCoordinatorClient, GenerationOptions};
    use dynamo_runtime::pipeline::{AsyncEngineContext, context::Controller};
    use std::sync::{Arc, Mutex};

    #[derive(Debug, PartialEq)]
    enum Event {
        Begin,
        Dispatch,
        Admission,
        Chunk,
        Denied,
        Error,
        Finish(GenerationObservationEnd),
    }
    type Events = Arc<Mutex<Vec<Event>>>;
    struct Probe(Events);
    impl GenerationObserverFactory for Probe {
        fn begin(
            &self,
            _: &RequestContext,
            _: &GenerationRequest,
        ) -> Result<Box<dyn GenerationObserver>> {
            self.0.lock().unwrap().push(Event::Begin);
            Ok(Box::new(Probe(self.0.clone())))
        }
    }
    impl GenerationObserver for Probe {
        fn on_admission(&mut self, _: &GenerationAdmission) {
            self.0.lock().unwrap().push(Event::Admission);
        }
        fn on_chunk(&mut self, _: &Annotated<Value>) {
            self.0.lock().unwrap().push(Event::Chunk);
        }
        fn on_denied(&mut self, _: &DeniedGenerationRequest) {
            self.0.lock().unwrap().push(Event::Denied);
        }
        fn on_error(&mut self, _: &anyhow::Error) {
            self.0.lock().unwrap().push(Event::Error);
        }
        fn on_finish(&mut self, end: GenerationObservationEnd) {
            self.0.lock().unwrap().push(Event::Finish(end));
        }
    }
    enum Mode {
        Stream,
        Denied,
        Error,
    }
    struct Client {
        probe: Probe,
        mode: Mode,
    }
    impl GenerationCoordinatorClient for Client {
        fn observer(&self) -> Option<&dyn GenerationObserverFactory> {
            Some(&self.probe)
        }
        fn generate_inner(
            &self,
            context: RequestContext,
            _: GenerationRequest,
            _: GenerationOptions,
        ) -> BoxFuture<'_, Result<GenerationOutcome>> {
            Box::pin(async move {
                self.probe.0.lock().unwrap().push(Event::Dispatch);
                match self.mode {
                    Mode::Error => anyhow::bail!("transport failed"),
                    Mode::Denied => Ok(GenerationOutcome::Denied(DeniedGenerationRequest {
                        denied: DeniedRequest::Cancelled(),
                        admission: None,
                        usage: None,
                    })),
                    Mode::Stream => Ok(GenerationOutcome::Connected(GeneratedRequest {
                        admission: GenerationAdmission::default(),
                        stream: ResponseStream::new(
                            Box::pin(futures::stream::iter([
                                Annotated::from_data(Value::from(1)),
                                Annotated::from_data(Value::from(2)),
                            ])),
                            context.inner(),
                        ),
                    })),
                }
            })
        }
    }
    fn context() -> RequestContext {
        RequestContext::new(
            Arc::new(Controller::new("observer-test".into())),
            None,
            Default::default(),
        )
    }
    fn request() -> GenerationRequest {
        GenerationRequest {
            routing_request: Default::default(),
            primary_worker_request: Value::Nil,
            decode_worker_request: None,
        }
    }
    fn client(mode: Mode) -> Client {
        Client {
            probe: Probe(Default::default()),
            mode,
        }
    }

    #[test]
    fn b10_observer_finalizes_unpolled_setup() {
        let client = client(Mode::Stream);
        let future = client.generate(context(), request(), Default::default());
        assert_eq!(*client.probe.0.lock().unwrap(), vec![Event::Begin]);
        drop(future);
        assert_eq!(
            *client.probe.0.lock().unwrap(),
            vec![
                Event::Begin,
                Event::Finish(GenerationObservationEnd::Dropped)
            ]
        );
    }

    #[tokio::test]
    async fn b10_observer_reports_denial_and_transport_error_once() {
        for (mode, event, end) in [
            (
                Mode::Denied,
                Event::Denied,
                GenerationObservationEnd::Denied,
            ),
            (Mode::Error, Event::Error, GenerationObservationEnd::Error),
        ] {
            let client = client(mode);
            let _ = client
                .generate(context(), request(), Default::default())
                .await;
            assert_eq!(
                *client.probe.0.lock().unwrap(),
                vec![Event::Begin, Event::Dispatch, event, Event::Finish(end)]
            );
        }
    }

    #[tokio::test]
    async fn b10_observer_stream_closure_and_drops() {
        // Covers unpolled streams, cancellation after output, and full closure.
        for consumed in 0..=2 {
            let client = client(Mode::Stream);
            let GenerationOutcome::Connected(mut generated) = client
                .generate(context(), request(), Default::default())
                .await
                .unwrap()
            else {
                panic!("expected stream")
            };
            for expected in 1..=consumed {
                let item = generated.stream.next().await.unwrap();
                assert_eq!(item.data, Some(Value::from(expected)));
            }
            let end = if consumed == 2 {
                assert!(generated.stream.next().await.is_none());
                GenerationObservationEnd::StreamClosed
            } else {
                GenerationObservationEnd::Dropped
            };
            drop(generated);
            let mut expected = vec![Event::Begin, Event::Dispatch, Event::Admission];
            expected.extend((0..consumed).map(|_| Event::Chunk));
            expected.push(Event::Finish(end));
            assert_eq!(*client.probe.0.lock().unwrap(), expected);
        }
    }
    struct RejectFactory;
    impl GenerationObserverFactory for RejectFactory {
        fn begin(
            &self,
            _: &RequestContext,
            _: &GenerationRequest,
        ) -> Result<Box<dyn GenerationObserver>> {
            anyhow::bail!("invalid metrics configuration")
        }
    }
    struct RejectClient(Client);
    impl GenerationCoordinatorClient for RejectClient {
        fn observer(&self) -> Option<&dyn GenerationObserverFactory> {
            Some(&RejectFactory)
        }
        fn generate_inner(
            &self,
            context: RequestContext,
            request: GenerationRequest,
            options: GenerationOptions,
        ) -> BoxFuture<'_, Result<GenerationOutcome>> {
            self.0.generate_inner(context, request, options)
        }
    }
    #[tokio::test]
    async fn b10_observer_factory_failure_preserves_generation_outcome() {
        for mode in [Mode::Stream, Mode::Denied, Mode::Error] {
            let client = RejectClient(client(mode));
            let outcome = client
                .generate(context(), request(), Default::default())
                .await;
            match client.0.mode {
                Mode::Stream => {
                    let GenerationOutcome::Connected(mut generated) = outcome.unwrap() else {
                        panic!("expected stream")
                    };
                    for expected in [1, 2] {
                        assert_eq!(
                            generated.stream.next().await.unwrap().data,
                            Some(Value::from(expected))
                        );
                    }
                    assert!(generated.stream.next().await.is_none());
                }
                Mode::Denied => assert!(matches!(
                    outcome.unwrap(),
                    GenerationOutcome::Denied(DeniedGenerationRequest {
                        denied: DeniedRequest::Cancelled(),
                        ..
                    })
                )),
                Mode::Error => assert_eq!(outcome.unwrap_err().to_string(), "transport failed"),
            }
            // No callbacks were installed; dispatch and its outcome are unchanged.
            assert_eq!(*client.0.probe.0.lock().unwrap(), vec![Event::Dispatch]);
        }
    }
    #[tokio::test]
    async fn b10_observer_runs_independently_on_both_http_boundaries() {
        let serving = Arc::new(client(Mode::Stream));
        let service = Arc::new(crate::GenerationCoordinatorService::new(
            serving.clone(),
            crate::DisaggregationStrategy::Aggregated,
        ));
        let running = service.start("127.0.0.1:0".parse().unwrap()).await.unwrap();
        let sending: Events = Default::default();
        let remote = crate::RemoteGenerationCoordinator::new(running.endpoint_url())
            .unwrap()
            .with_observer(Arc::new(Probe(sending.clone())));
        let mut request = request();
        request.primary_worker_request = Value::Map(vec![]);
        request.routing_request.tokens = vec![1, 2];
        let GenerationOutcome::Connected(mut generated) = remote
            .generate(context(), request, Default::default())
            .await
            .unwrap()
        else {
            panic!("expected stream")
        };
        let mut chunks = 0;
        while let Some(item) = generated.stream.next().await {
            assert!(item.error.is_none(), "{item:?}");
            chunks += 1;
        }
        assert_eq!(chunks, 2);
        drop(generated);
        assert_eq!(
            *sending.lock().unwrap(),
            vec![
                Event::Begin,
                Event::Admission,
                Event::Chunk,
                Event::Chunk,
                Event::Finish(GenerationObservationEnd::StreamClosed)
            ]
        );
        assert_eq!(
            *serving.probe.0.lock().unwrap(),
            vec![
                Event::Begin,
                Event::Dispatch,
                Event::Admission,
                Event::Chunk,
                Event::Chunk,
                Event::Finish(GenerationObservationEnd::StreamClosed)
            ]
        );
        running.shutdown().await.unwrap();
    }
    struct ImmediateClient {
        probe: Probe,
        enabled: bool,
    }
    impl GenerationCoordinatorClient for ImmediateClient {
        fn observer(&self) -> Option<&dyn GenerationObserverFactory> {
            self.enabled.then_some(&self.probe)
        }
        fn generate_inner(
            &self,
            _: RequestContext,
            _: GenerationRequest,
            _: GenerationOptions,
        ) -> BoxFuture<'_, Result<GenerationOutcome>> {
            // Synchronous dispatch would already be too early if begin happened
            // inside an async body instead of at the coordinator entry point.
            self.probe.0.lock().unwrap().push(Event::Dispatch);
            Box::pin(futures::future::pending())
        }
    }

    #[test]
    fn b10_observer_begins_before_synchronous_dispatch() {
        let client = ImmediateClient {
            probe: Probe(Default::default()),
            enabled: true,
        };
        let future = client.generate(context(), request(), Default::default());
        assert_eq!(
            *client.probe.0.lock().unwrap(),
            vec![Event::Begin, Event::Dispatch]
        );
        drop(future);
        assert_eq!(
            *client.probe.0.lock().unwrap(),
            vec![
                Event::Begin,
                Event::Dispatch,
                Event::Finish(GenerationObservationEnd::Dropped)
            ]
        );
    }

    #[test]
    fn b10_observer_disabled_dispatches_without_callbacks() {
        let client = ImmediateClient {
            probe: Probe(Default::default()),
            enabled: false,
        };
        let future = client.generate(context(), request(), Default::default());
        drop(future);
        assert_eq!(*client.probe.0.lock().unwrap(), vec![Event::Dispatch]);
    }

    struct PendingClient(Probe);
    impl GenerationCoordinatorClient for PendingClient {
        fn observer(&self) -> Option<&dyn GenerationObserverFactory> {
            Some(&self.0)
        }
        fn generate_inner(
            &self,
            _: RequestContext,
            _: GenerationRequest,
            _: GenerationOptions,
        ) -> BoxFuture<'_, Result<GenerationOutcome>> {
            Box::pin(async {
                self.0.0.lock().unwrap().push(Event::Dispatch);
                futures::future::pending().await
            })
        }
    }

    #[tokio::test]
    async fn b10_observer_finalizes_pending_setup_once() {
        let client = PendingClient(Probe(Default::default()));
        let mut future = client.generate(context(), request(), Default::default());
        assert!(futures::poll!(future.as_mut()).is_pending());
        drop(future);
        assert_eq!(
            *client.0.0.lock().unwrap(),
            vec![
                Event::Begin,
                Event::Dispatch,
                Event::Finish(GenerationObservationEnd::Dropped)
            ]
        );
    }

    #[tokio::test]
    async fn b10_observer_finalizes_while_chunk_poll_is_pending() {
        let events: Events = Default::default();
        let guard = ObserverGuard::new(Box::new(Probe(events.clone())));
        let source = futures::stream::iter([Annotated::from_data(Value::from(1))])
            .chain(futures::stream::pending());
        let mut stream = guard.wrap(ResponseStream::new(Box::pin(source), context().inner()));
        assert_eq!(stream.next().await.unwrap().data, Some(Value::from(1)));
        let mut next = stream.next();
        assert!(futures::poll!(&mut next).is_pending());
        drop(next);
        drop(stream);
        assert_eq!(
            *events.lock().unwrap(),
            vec![
                Event::Chunk,
                Event::Finish(GenerationObservationEnd::Dropped)
            ]
        );
    }
    struct DifferentContextClient {
        probe: Probe,
        returned_context: Arc<dyn AsyncEngineContext>,
    }
    impl GenerationCoordinatorClient for DifferentContextClient {
        fn observer(&self) -> Option<&dyn GenerationObserverFactory> {
            Some(&self.probe)
        }
        fn generate_inner(
            &self,
            _: RequestContext,
            _: GenerationRequest,
            _: GenerationOptions,
        ) -> BoxFuture<'_, Result<GenerationOutcome>> {
            let returned_context = self.returned_context.clone();
            Box::pin(async move {
                Ok(GenerationOutcome::Connected(GeneratedRequest {
                    stream: ResponseStream::new(
                        Box::pin(futures::stream::empty()),
                        returned_context,
                    ),
                    admission: Default::default(),
                }))
            })
        }
    }
    #[tokio::test]
    async fn b10_observer_preserves_returned_stream_context() {
        let returned_context: Arc<dyn AsyncEngineContext> =
            Arc::new(Controller::new("returned-context".into()));
        let client = DifferentContextClient {
            probe: Probe(Default::default()),
            returned_context: returned_context.clone(),
        };
        let GenerationOutcome::Connected(generated) = client
            .generate(context(), request(), Default::default())
            .await
            .unwrap()
        else {
            panic!("expected stream")
        };
        let observed_context = generated.stream.context();
        assert!(
            Arc::ptr_eq(&observed_context, &returned_context),
            "observer replaced the stream context supplied by generate_inner"
        );
    }
}
