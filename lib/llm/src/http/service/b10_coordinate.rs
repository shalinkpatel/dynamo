// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `POST /v1/coordinate` on the frontend's HTTP server.
//!
//! Beefeater delivers a global frontend's coordinate requests to the OV's frontend
//! Service, so the frontend serves the generation coordinator's own route. The
//! coordinator lives in `dynamo-b10-client`, which depends on this crate: it registers
//! its router here when it starts, and the route looks the router up per request, so
//! it does not matter whether the coordinator or the HTTP service starts first. Until
//! a coordinator registers, the route answers 404. Shutdown keeps the registration
//! alive through the frontend's propagation grace period and request drain.

use std::sync::{Arc, LazyLock};

use axum::{
    Router,
    body::Body,
    http::{Request, StatusCode},
    response::{IntoResponse, Response},
    routing::post,
};
use tokio::sync::RwLock;
use tokio_util::{sync::CancellationToken, task::TaskTracker};
use tower::ServiceExt;

pub const COORDINATE_PATH: &str = "/v1/coordinate";

static COORDINATOR: RwLock<Option<Arc<Registration>>> = RwLock::const_new(None);

// Tokens cover each HTTP server's entire serving future, including graceful drain.
pub(crate) static FRONTENDS: LazyLock<TaskTracker> = LazyLock::new(TaskTracker::new);

/// A runtime-owned registration; unregister it as part of graceful shutdown.
pub struct Registration {
    router: Router,
}

impl Registration {
    pub async fn unregister(self: Arc<Self>) {
        // Closing allows wait() to finish when empty; later frontends can still add tokens.
        FRONTENDS.close();
        FRONTENDS.wait().await;
        let mut slot = COORDINATOR.write().await;
        let previous = if slot
            .as_ref()
            .is_some_and(|current| Arc::ptr_eq(current, &self))
        {
            slot.take()
        } else {
            None
        };
        drop(slot);
        drop(previous);
    }
}

/// Publish the latest coordinator. Already-cancelled registrations have no effect.
pub async fn register(router: Router, shutdown: CancellationToken) -> Arc<Registration> {
    let registration = Arc::new(Registration { router });
    let mut slot = COORDINATOR.write().await;
    let previous = if shutdown.is_cancelled() {
        None
    } else {
        slot.replace(registration.clone())
    };
    drop(slot);
    drop(previous);
    registration
}

/// Internal: left out of the OpenAPI route docs.
pub(crate) fn coordinate_router() -> Router {
    Router::new().route(COORDINATE_PATH, post(coordinate))
}

async fn coordinate(request: Request<Body>) -> Response {
    let coordinator = {
        let slot = COORDINATOR.read().await;
        slot.as_ref().map(|entry| entry.router.clone())
    };
    match coordinator {
        Some(router) => router
            .oneshot(request)
            .await
            .unwrap_or_else(|never| match never {}),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::service::service_v2::HttpService;
    use axum::body::Bytes;

    #[tokio::test]
    async fn frontend_serves_the_registered_coordinator() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}{COORDINATE_PATH}", listener.local_addr().unwrap());
        let service = HttpService::builder().build().unwrap();
        let cancel = CancellationToken::new();
        let handle = service.spawn_with_listener(cancel.clone(), listener).await;
        let http = reqwest::Client::new();
        let send = |body: &'static [u8]| {
            http.post(&url)
                .header("content-type", "application/json")
                .body(body)
                .send()
        };

        // The only test in this crate that registers, so nothing is registered yet.
        assert_eq!(
            send(b"unused").await.unwrap().status(),
            reqwest::StatusCode::NOT_FOUND
        );

        // The registered router receives the request as sent, path and body included.
        let registration = register(
            Router::new().route(
                COORDINATE_PATH,
                post(|body: Bytes| async move { [b"echo:".as_slice(), &body].concat() }),
            ),
            CancellationToken::new(),
        )
        .await;
        let response = send(b"frames").await.unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        assert_eq!(response.bytes().await.unwrap().as_ref(), b"echo:frames");
        // An older runtime's cleanup must leave the replacement active.
        let shutdown = CancellationToken::new();
        let finish = CancellationToken::new();
        let latest = register(
            Router::new().route(
                COORDINATE_PATH,
                post({
                    let finish = finish.clone();
                    move || {
                        let finish = finish.clone();
                        async move {
                            Body::from_stream(async_stream::stream! {
                                yield Ok::<_, std::convert::Infallible>(Bytes::from_static(b"new"));
                                finish.cancelled().await;
                                yield Ok(Bytes::from_static(b"last"));
                            })
                        }
                    }
                }),
            ),
            shutdown.clone(),
        )
        .await;
        let old_cleanup = tokio::spawn(registration.unregister());
        let mut streaming = send(b"new").await.unwrap();
        assert_eq!(streaming.chunk().await.unwrap().unwrap(), "new");
        shutdown.cancel();
        let cleanup = tokio::spawn(latest.unregister());
        cancel.cancel();
        // Late-routed requests still succeed during Axum's propagation grace period.
        assert_eq!(
            send(b"draining")
                .await
                .unwrap()
                .chunk()
                .await
                .unwrap()
                .unwrap(),
            "new"
        );
        // Stopping admission must not end an active token stream, even 30 seconds later.
        service.state().cancel_token().cancelled().await;
        tokio::time::pause();
        tokio::time::advance(std::time::Duration::from_secs(30)).await;
        tokio::task::yield_now().await;
        assert!(!handle.is_finished());
        assert!(!cleanup.is_finished());
        tokio::time::resume();
        finish.cancel();
        assert_eq!(streaming.chunk().await.unwrap().unwrap(), "last");
        assert!(streaming.chunk().await.unwrap().is_none());
        handle.await.unwrap().unwrap();
        old_cleanup.await.unwrap();
        cleanup.await.unwrap();
        assert!(COORDINATOR.read().await.is_none());
    }
}
