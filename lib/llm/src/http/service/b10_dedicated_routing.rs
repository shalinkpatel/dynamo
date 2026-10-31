// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Preserve dedicated routing while presenting API paths to the frontend router.

use axum::{
    Router,
    extract::Request,
    http::{StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};

pub const PREFIX_HEADER: &str = "x-baseten-internal-routing-prefix";

/// Split the same environment/deployment sync prefix that Alyx uses for routing.
fn split_prefix(path: &str) -> Option<(&str, &str)> {
    let rest = path
        .strip_prefix("/environments/")
        .or_else(|| path.strip_prefix("/deployment/"))?;
    let (name, rest) = rest.split_once('/')?;
    if name.is_empty() {
        return None;
    }
    let suffix = rest.strip_prefix("sync")?;
    if !suffix.is_empty() && !suffix.starts_with('/') {
        return None;
    }
    Some(path.split_at(path.len() - suffix.len()))
}

pub(super) fn wrap(router: Router, remote: bool) -> Router {
    if !remote {
        return router;
    }
    // Router::layer runs after route lookup; wrap the router itself to rewrite first.
    Router::new()
        .fallback_service(router)
        .layer(axum::middleware::from_fn(normalize))
}

async fn normalize(mut request: Request, next: Next) -> Response {
    // This is an internal carrier, never a routing instruction from the caller.
    request.headers_mut().remove(PREFIX_HEADER);
    if let Some((prefix, suffix)) = split_prefix(request.uri().path()) {
        // Forwarded-header cleanup must not discard the routing carrier as hop-by-hop.
        if request
            .headers()
            .get_all(header::CONNECTION)
            .iter()
            .any(|value| {
                value
                    .to_str()
                    .unwrap_or("")
                    .split(',')
                    .any(|name| name.trim().eq_ignore_ascii_case(PREFIX_HEADER))
            })
        {
            return StatusCode::BAD_REQUEST.into_response();
        }
        let prefix = prefix.parse().expect("URI path is a valid header value");
        let suffix = if suffix.is_empty() { "/" } else { suffix };
        let path_and_query = match request.uri().query() {
            Some(query) => format!("{suffix}?{query}"),
            None => suffix.to_owned(),
        };
        let mut uri = request.uri().clone().into_parts();
        uri.path_and_query = Some(path_and_query.parse().expect("suffix of a valid URI"));
        *request.uri_mut() = axum::http::Uri::from_parts(uri).expect("valid URI parts");
        request.headers_mut().insert(PREFIX_HEADER, prefix);
    }
    next.run(request).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{Body, to_bytes},
        http::StatusCode,
        routing::post,
    };
    use tower::ServiceExt;

    #[tokio::test]
    async fn b10_prefix_routing_is_remote_only() {
        for remote in [false, true] {
            for path in [
                "/v1/chat/completions",
                "/v1/completions",
                "/v1/responses",
                "/v1/messages",
            ] {
                let router = wrap(
                    Router::new().route(
                        path,
                        post(|request: Request| async move {
                            format!(
                                "{}|{}",
                                request
                                    .headers()
                                    .get(PREFIX_HEADER)
                                    .map_or("", |v| v.to_str().unwrap()),
                                request.uri()
                            )
                        }),
                    ),
                    remote,
                );
                if remote {
                    let request = Request::builder()
                        .method("POST")
                        .uri(format!("/deployment/abc/sync{path}"))
                        .header(header::CONNECTION, PREFIX_HEADER)
                        .body(Body::empty())
                        .unwrap();
                    assert_eq!(
                        router.clone().oneshot(request).await.unwrap().status(),
                        StatusCode::BAD_REQUEST
                    );
                }
                for prefix in [
                    "",
                    "/environments/staging/sync",
                    "/deployment/abc/sync",
                    "/environments//sync",
                    "/deployment/abc/sync-extra",
                ] {
                    let response = router
                        .clone()
                        .oneshot(
                            Request::builder()
                                .method("POST")
                                .uri(format!("{prefix}{path}?a=b"))
                                .header(PREFIX_HEADER, "spoof")
                                .body(Body::empty())
                                .unwrap(),
                        )
                        .await
                        .unwrap();
                    let routes = prefix.is_empty() || (remote && split_prefix(prefix).is_some());
                    assert_eq!(
                        response.status(),
                        if routes {
                            StatusCode::OK
                        } else {
                            StatusCode::NOT_FOUND
                        }
                    );
                    if routes {
                        let carrier = if remote { prefix } else { "spoof" };
                        assert_eq!(
                            to_bytes(response.into_body(), 1024).await.unwrap(),
                            format!("{carrier}|{path}?a=b")
                        );
                    }
                }
            }
        }
    }
}
