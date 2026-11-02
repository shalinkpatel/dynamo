// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-request accounting for the inference routes, ahead of protocol translation.
//!
//! The Python processor writes one `dynamo` record per request it receives
//! (`common/request_logger.py` in the monorepo), and the unified-logs collector lifts
//! that object into `LogAttributes['dynamo']` of ClickHouse `model_logs`. Two things
//! happen before the processor and are invisible there:
//!
//! - **Which API the client called.** `/v1/messages` and `/v1/responses` become chat
//!   requests in their handlers, and tool-bank turns every server-tools request into
//!   `/v1/chat/completions` calls before Dynamo sees it. [`layer`] stamps the route's
//!   surface into an internal header that [`insert_api_surface_metadata`] copies into
//!   context metadata; tool-bank's `x-baseten-client-api` header overrides it.
//! - **Requests rejected before the engine** (rate limit, not ready, unknown model, body
//!   and field validation, `generate` failures before the processor). [`layer`] writes
//!   the `dynamo` record for any error response whose request never reached the engine.
//!   [`mark_if_engine_saw`] marks the rest: a successful `generate`, or an `Err` carrying
//!   the [`HttpError`] the Python processor raises after logging its own record
//!   (`HttpAsyncEngine` blocks on the first stream item, so processor errors surface
//!   from `generate` itself).

use std::collections::{BTreeMap, HashSet};
use std::sync::{LazyLock, Mutex};
use std::time::SystemTime;

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderValue};
use axum::middleware::Next;
use axum::response::Response;

use super::baseten::get_or_create_context_id;
use super::error::HttpError;

/// Resolved API surface: what the client called.
pub const API_SURFACE_METADATA_KEY: &str = "baseten.api_surface";
/// `"true"` when tool-bank sent the request on the client's behalf (server-side tools).
pub const SERVER_TOOLS_METADATA_KEY: &str = "baseten.server_tools";
/// Set by tool-bank on its model calls to the protocol its client used.
pub const CLIENT_API_HEADER: &str = "x-baseten-client-api";
/// Internal: the route's surface, always overwritten by [`layer`] so clients cannot set it.
const ROUTE_API_SURFACE_HEADER: &str = "x-baseten-internal-api-surface";

pub const CHAT_COMPLETIONS: &str = "chat_completions";
pub const COMPLETIONS: &str = "completions";
pub const RESPONSES: &str = "responses";
pub const MESSAGES: &str = "messages";
const SURFACES: [&str; 4] = [CHAT_COMPLETIONS, COMPLETIONS, RESPONSES, MESSAGES];

const MAX_ERROR_BODY_BYTES: usize = 64 * 1024;

/// Image identity, as the processor's record carries it (`GIT_SHA`, `DYNAMO_BACKEND`
/// are baked into the image; the frontend runs in the same process).
static IMAGE: LazyLock<(String, String)> = LazyLock::new(|| {
    let var = |k| std::env::var(k).unwrap_or_default();
    (var("GIT_SHA"), var("DYNAMO_BACKEND"))
});
const MAX_ERROR_CHARS: usize = 1000;

// ponytail: one set keyed by context id. Two concurrent requests sharing an id (the client
// reusing X-Baseten-Request-Id) can hide one rejection record; key per request if that matters.
static DISPATCHED: LazyLock<Mutex<HashSet<String>>> = LazyLock::new(Default::default);

/// Mark `context_id` dispatched when `result` shows the engine saw the request: `Ok`, or
/// an error carrying the processor's [`HttpError`] (it logged before raising).
pub(crate) fn mark_if_engine_saw<T>(context_id: &str, result: &anyhow::Result<T>) {
    if engine_saw(result) {
        mark_dispatched(context_id);
    }
}

fn engine_saw<T>(result: &anyhow::Result<T>) -> bool {
    match result {
        Ok(_) => true,
        Err(e) => e.chain().any(|cause| cause.is::<HttpError>()),
    }
}

/// Note that the engine accepted this request, so the processor owns its record.
fn mark_dispatched(context_id: &str) {
    DISPATCHED
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(context_id.to_owned());
}

fn take_dispatched(context_id: &str) -> bool {
    DISPATCHED
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .remove(context_id)
}

/// Copy the resolved surface into context metadata, replacing client-supplied
/// `x-dynamo-meta-baseten.*` entries.
pub(crate) fn insert_api_surface_metadata(
    metadata: &mut BTreeMap<String, String>,
    headers: &HeaderMap,
) {
    metadata.remove(API_SURFACE_METADATA_KEY);
    metadata.remove(SERVER_TOOLS_METADATA_KEY);
    let client_api = client_api(headers);
    if client_api.is_some() {
        metadata.insert(SERVER_TOOLS_METADATA_KEY.into(), "true".into());
    }
    if let Some(surface) = client_api.or_else(|| surface_header(headers, ROUTE_API_SURFACE_HEADER))
    {
        metadata.insert(API_SURFACE_METADATA_KEY.into(), surface.into());
    }
}

fn surface_header(headers: &HeaderMap, name: &str) -> Option<&'static str> {
    let value = headers.get(name)?.to_str().ok()?;
    SURFACES.into_iter().find(|s| *s == value)
}

fn client_api(headers: &HeaderMap) -> Option<&'static str> {
    surface_header(headers, CLIENT_API_HEADER)
}

/// Wrap one inference route in [`layer`] for `surface`.
pub(crate) fn route<S: Clone + Send + Sync + 'static>(
    method_router: axum::routing::MethodRouter<S>,
    surface: &'static str,
) -> axum::routing::MethodRouter<S> {
    // route_layer: a 405 for another method is not a request.
    method_router.route_layer(axum::middleware::from_fn_with_state(surface, layer))
}

/// Route layer for one inference surface: stamp the surface, and write the `dynamo`
/// record for error responses the engine never saw.
pub async fn layer(
    State(surface): State<&'static str>,
    mut request: Request,
    next: Next,
) -> Response {
    let headers = request.headers_mut();
    headers.insert(ROUTE_API_SURFACE_HEADER, HeaderValue::from_static(surface));
    // Fix the request id now so the handler derives the same context id as this record.
    if headers
        .get("x-baseten-request-id")
        .is_none_or(|v| v.is_empty())
        && let Ok(id) = HeaderValue::from_str(&uuid::Uuid::new_v4().to_string())
    {
        headers.insert("x-baseten-request-id", id);
    }
    let context_id = get_or_create_context_id(request.headers());
    // Waypoints inspections are internal replays; they are not traffic.
    if request
        .extensions()
        .get::<super::waypoints::Inspection>()
        .is_some()
    {
        let response = next.run(request).await;
        take_dispatched(&context_id);
        return response;
    }
    let api_surface = client_api(request.headers()).unwrap_or(surface);
    let server_tools = client_api(request.headers()).is_some();
    let model_name = header_string(request.headers(), "x-baseten-model-slug");
    let start = SystemTime::now();

    let response = next.run(request).await;

    let dispatched = take_dispatched(&context_id);
    let status = response.status();
    // The router's JSON error middleware (outside this route layer) turns a body
    // deserialization 422 into the 400 the client sees.
    let http_status = match status.as_u16() {
        422 => 400,
        code => code,
    };
    if dispatched || !(status.is_client_error() || status.is_server_error()) {
        return response;
    }
    let (parts, body) = response.into_parts();
    // Read the error text only from a body of known small size (handler errors are
    // buffered JSON); anything else passes through untouched, since a failed bounded
    // read would drop the client's body. Content-Length is not set yet at this layer.
    let small = axum::body::HttpBody::size_hint(&body)
        .upper()
        .is_some_and(|len| len <= MAX_ERROR_BODY_BYTES as u64);
    let (body, error, error_type) = if small {
        let bytes = axum::body::to_bytes(body, MAX_ERROR_BODY_BYTES)
            .await
            .unwrap_or_default();
        let (error, error_type) = error_from_body(&bytes);
        (Body::from(bytes), error, error_type)
    } else {
        (body, String::new(), String::new())
    };
    let record = Rejection {
        context_id: &context_id,
        route_surface: surface,
        api_surface,
        server_tools,
        model_name: &model_name,
        http_status,
        error: &error,
        error_type: &error_type,
        start,
        end: SystemTime::now(),
    }
    .to_json();
    #[cfg(test)]
    tests::EMITTED.lock().unwrap().push(record.clone());
    tracing::info!(
        unified_model_logs = true,
        dynamo = record.to_string().as_str(),
        b10_request_id = record["b10_request_id"].as_str().unwrap_or_default(),
        context_id = %context_id,
        "Inference request log line (frontend rejection): {api_surface} {http_status}"
    );
    Response::from_parts(parts, body)
}

fn header_string(headers: &HeaderMap, name: &str) -> String {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_owned()
}

/// OpenAI (`{"message", "type"}` or `{"error": {...}}`) and Anthropic
/// (`{"error": {"type", "message"}}`) error bodies; anything else is kept as text.
fn error_from_body(bytes: &[u8]) -> (String, String) {
    let json: serde_json::Value = serde_json::from_slice(bytes).unwrap_or_default();
    let obj = json.get("error").filter(|e| e.is_object()).unwrap_or(&json);
    let field = |k: &str| obj.get(k).and_then(|v| v.as_str()).map(str::to_owned);
    let error = field("message").unwrap_or_else(|| String::from_utf8_lossy(bytes).into_owned());
    let error_type = field("type").unwrap_or_default();
    (error.chars().take(MAX_ERROR_CHARS).collect(), error_type)
}

struct Rejection<'a> {
    context_id: &'a str,
    route_surface: &'static str,
    api_surface: &'static str,
    server_tools: bool,
    model_name: &'a str,
    http_status: u16,
    error: &'a str,
    error_type: &'a str,
    start: SystemTime,
    end: SystemTime,
}

impl Rejection<'_> {
    /// Same field names as the processor's record; empty values are omitted like there.
    fn to_json(&self) -> serde_json::Value {
        // context id: {org_namespace}--{b10_request_id}--{model_version_id}[--{extras}]
        let mut parts = self.context_id.split("--");
        let mut part = || parts.next().filter(|p| *p != "none").unwrap_or_default();
        let (org_namespace, b10_request_id, model_version_id) = (part(), part(), part());
        let iso = |t: SystemTime| {
            chrono::DateTime::<chrono::Utc>::from(t)
                .format("%Y-%m-%dT%H:%M:%S%.3fZ")
                .to_string()
        };
        let duration_ms = self
            .end
            .duration_since(self.start)
            .unwrap_or_default()
            .as_secs_f64()
            * 1000.0;
        let request_type = if self.route_surface == COMPLETIONS {
            "completion"
        } else {
            "chat"
        };
        let mut record = serde_json::json!({
            "stage": "frontend",
            "status": "error",
            "http_status": self.http_status,
            "request_type": request_type,
            "api_surface": self.api_surface,
            "start_time": iso(self.start),
            "end_time": iso(self.end),
            "duration_ms": (duration_ms * 1000.0).round() / 1000.0,
            "context_id": self.context_id,
            "b10_request_id": b10_request_id,
            "org_namespace": org_namespace,
            "model_version_id": model_version_id,
            "model_name": self.model_name,
            "image_git_sha": IMAGE.0.as_str(),
            "backend": IMAGE.1.as_str(),
            "error": self.error,
            "error_type": self.error_type,
        });
        if self.server_tools {
            record["request_features"] = serde_json::json!(["server_tools"]);
        }
        let obj = record.as_object_mut().expect("record is an object");
        obj.retain(|_, v| v.as_str() != Some(""));
        record
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;
    use axum::routing::post;
    use axum::{Router, middleware};
    use tower::ServiceExt;

    /// Records [`layer`] wrote, for assertions (tests run in parallel: filter by request id).
    pub(super) static EMITTED: Mutex<Vec<serde_json::Value>> = Mutex::new(Vec::new());

    fn b10_emitted(b10_request_id: &str) -> Vec<serde_json::Value> {
        EMITTED
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r["b10_request_id"] == b10_request_id)
            .cloned()
            .collect()
    }

    /// One route returning `status`, marking the request dispatched first when asked.
    fn b10_router(status: StatusCode, dispatch: bool) -> Router {
        Router::new()
            .route(
                "/v1/messages",
                post(move |headers: HeaderMap| async move {
                    if dispatch {
                        // What the generate sites see when the processor raised after logging.
                        let result: anyhow::Result<()> = Err(anyhow::Error::new(HttpError {
                            code: status.as_u16(),
                            message: "from the processor".into(),
                        })
                        .context("Failed to generate completions"));
                        mark_if_engine_saw(&get_or_create_context_id(&headers), &result);
                    }
                    (status, r#"{"message":"nope","type":"Overload"}"#)
                }),
            )
            .layer(middleware::from_fn_with_state(MESSAGES, layer))
    }

    #[tokio::test]
    async fn b10_layer_records_rejections_the_engine_never_saw() {
        b10_call(
            b10_router(StatusCode::TOO_MANY_REQUESTS, false),
            &[
                ("x-baseten-request-id", "rej-429"),
                ("x-baseten-org-namespace", "org-a"),
                (CLIENT_API_HEADER, RESPONSES),
            ],
        )
        .await;
        let records = b10_emitted("rej-429");
        assert_eq!(records.len(), 1, "{records:?}");
        let r = &records[0];
        assert_eq!(r["stage"], "frontend");
        assert_eq!(r["http_status"], 429);
        assert_eq!(r["api_surface"], RESPONSES);
        assert_eq!(r["request_features"], serde_json::json!(["server_tools"]));
        assert_eq!(r["org_namespace"], "org-a");
        assert_eq!(r["error"], "nope");
        assert_eq!(r["error_type"], "Overload");

        b10_call(
            b10_router(StatusCode::UNPROCESSABLE_ENTITY, false),
            &[("x-baseten-request-id", "rej-422")],
        )
        .await;
        assert_eq!(
            b10_emitted("rej-422")[0]["http_status"],
            400,
            "client sees 400"
        );
    }

    #[tokio::test]
    async fn b10_layer_skips_dispatched_and_successful_requests() {
        for (id, status, dispatch) in [
            (
                "skip-dispatched-500",
                StatusCode::INTERNAL_SERVER_ERROR,
                true,
            ),
            // KV-backpressure 429 the processor already logged: no second record.
            ("skip-processor-429", StatusCode::TOO_MANY_REQUESTS, true),
            ("skip-200", StatusCode::OK, false),
        ] {
            b10_call(
                b10_router(status, dispatch),
                &[("x-baseten-request-id", id)],
            )
            .await;
            assert!(b10_emitted(id).is_empty(), "{id}");
        }
    }

    fn headers(pairs: &[(&'static str, &'static str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(*k, HeaderValue::from_static(v));
        }
        h
    }

    #[test]
    fn b10_metadata_uses_route_surface_and_drops_client_spoofing() {
        let mut metadata = BTreeMap::from([
            (API_SURFACE_METADATA_KEY.to_string(), "spoofed".to_string()),
            (SERVER_TOOLS_METADATA_KEY.to_string(), "true".to_string()),
        ]);
        insert_api_surface_metadata(
            &mut metadata,
            &headers(&[(ROUTE_API_SURFACE_HEADER, MESSAGES)]),
        );
        assert_eq!(metadata.get(API_SURFACE_METADATA_KEY).unwrap(), MESSAGES);
        assert!(!metadata.contains_key(SERVER_TOOLS_METADATA_KEY));
    }

    #[test]
    fn b10_metadata_prefers_tool_bank_client_api() {
        let mut metadata = BTreeMap::new();
        insert_api_surface_metadata(
            &mut metadata,
            &headers(&[
                (ROUTE_API_SURFACE_HEADER, CHAT_COMPLETIONS),
                (CLIENT_API_HEADER, RESPONSES),
            ]),
        );
        assert_eq!(metadata.get(API_SURFACE_METADATA_KEY).unwrap(), RESPONSES);
        assert_eq!(metadata.get(SERVER_TOOLS_METADATA_KEY).unwrap(), "true");
    }

    #[test]
    fn b10_metadata_ignores_unknown_client_api() {
        let mut metadata = BTreeMap::new();
        insert_api_surface_metadata(
            &mut metadata,
            &headers(&[
                (ROUTE_API_SURFACE_HEADER, CHAT_COMPLETIONS),
                (CLIENT_API_HEADER, "bogus"),
            ]),
        );
        assert_eq!(
            metadata.get(API_SURFACE_METADATA_KEY).unwrap(),
            CHAT_COMPLETIONS
        );
        assert!(!metadata.contains_key(SERVER_TOOLS_METADATA_KEY));
    }

    #[test]
    fn b10_error_body_shapes() {
        let openai = br#"{"message":"Too many requests","type":"Overload","code":429}"#;
        assert_eq!(
            error_from_body(openai),
            ("Too many requests".into(), "Overload".into())
        );
        let anthropic =
            br#"{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}"#;
        assert_eq!(
            error_from_body(anthropic),
            ("slow down".into(), "rate_limit_error".into())
        );
        assert_eq!(
            error_from_body(b"plain text"),
            ("plain text".into(), String::new())
        );
    }

    #[test]
    fn b10_rejection_record_matches_processor_fields() {
        let t = SystemTime::UNIX_EPOCH + std::time::Duration::from_millis(1_700_000_000_123);
        let record = Rejection {
            context_id: "org-abc--deadbeef--none",
            route_surface: COMPLETIONS,
            api_surface: COMPLETIONS,
            server_tools: false,
            model_name: "",
            http_status: 429,
            error: "Too many requests",
            error_type: "Overload",
            start: t,
            end: t + std::time::Duration::from_micros(2500),
        }
        .to_json();
        assert_eq!(
            record,
            serde_json::json!({
                "stage": "frontend",
                "status": "error",
                "http_status": 429,
                "request_type": "completion",
                "api_surface": "completions",
                "start_time": "2023-11-14T22:13:20.123Z",
                "end_time": "2023-11-14T22:13:20.125Z",
                "duration_ms": 2.5,
                "context_id": "org-abc--deadbeef--none",
                "b10_request_id": "deadbeef",
                "org_namespace": "org-abc",
                "error": "Too many requests",
                "error_type": "Overload",
            })
        );
    }

    async fn b10_call(
        router: Router,
        pairs: &[(&'static str, &'static str)],
    ) -> (StatusCode, String) {
        let mut request = Request::post("/v1/messages").body(Body::empty()).unwrap();
        *request.headers_mut() = headers(pairs);
        let response = router.oneshot(request).await.unwrap();
        let status = response.status();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, String::from_utf8(body.to_vec()).unwrap())
    }

    /// The handler sees the stamped surface and a fixed request id, and an error
    /// body passes through unchanged whether or not the layer logged it.
    #[tokio::test]
    async fn b10_layer_stamps_surface_and_preserves_error_body() {
        let router = Router::new()
            .route(
                "/v1/messages",
                post(|headers: HeaderMap| async move {
                    let mut metadata = BTreeMap::new();
                    insert_api_surface_metadata(&mut metadata, &headers);
                    assert_eq!(metadata.get(API_SURFACE_METADATA_KEY).unwrap(), MESSAGES);
                    let context_id = get_or_create_context_id(&headers);
                    assert!(context_id.contains("--req-1--"), "{context_id}");
                    (StatusCode::TOO_MANY_REQUESTS, r#"{"message":"slow"}"#)
                }),
            )
            .layer(middleware::from_fn_with_state(MESSAGES, layer));
        let (status, body) = b10_call(
            router,
            &[
                (ROUTE_API_SURFACE_HEADER, "spoofed"),
                ("x-baseten-request-id", "req-1"),
            ],
        )
        .await;
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(body, r#"{"message":"slow"}"#);
    }

    #[tokio::test]
    async fn b10_layer_clears_dispatch_mark() {
        let router = Router::new()
            .route(
                "/v1/messages",
                post(|headers: HeaderMap| async move {
                    mark_dispatched(&get_or_create_context_id(&headers));
                    StatusCode::INTERNAL_SERVER_ERROR
                }),
            )
            .layer(middleware::from_fn_with_state(MESSAGES, layer));
        let pairs = [("x-baseten-request-id", "req-dispatched")];
        let context_id = get_or_create_context_id(&headers(&pairs));
        let (status, _) = b10_call(router, &pairs).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert!(!take_dispatched(&context_id), "layer must consume the mark");
    }

    #[test]
    fn b10_engine_saw_ok_and_processor_errors_only() {
        let processor = || {
            anyhow::Error::new(HttpError {
                code: 400,
                message: "bad".into(),
            })
        };
        assert!(engine_saw(&Ok(())));
        assert!(engine_saw::<()>(&Err(processor())));
        assert!(engine_saw::<()>(&Err(processor().context("wrapped"))));
        assert!(!engine_saw::<()>(&Err(anyhow::anyhow!("no instances"))));
    }

    #[tokio::test]
    async fn b10_layer_passes_large_or_unsized_error_bodies_through() {
        let big = "x".repeat(MAX_ERROR_BODY_BYTES + 1);
        let expected = big.clone();
        let router = Router::new()
            .route(
                "/v1/messages",
                post(move || {
                    let big = big.clone();
                    async move { (StatusCode::BAD_REQUEST, big) }
                }),
            )
            .layer(middleware::from_fn_with_state(MESSAGES, layer));
        let (status, body) = b10_call(router, &[("x-baseten-request-id", "big-body")]).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body, expected, "client keeps the full body");
        let records = b10_emitted("big-body");
        assert_eq!(records.len(), 1);
        assert!(records[0].get("error").is_none(), "{:?}", records[0]);
    }
}
