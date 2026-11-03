// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::protocol::codec::encode_request;
use crate::protocol::{self, GenerationResponseFrameV1, generation_response_frame_v1};
use crate::{
    CancellationPolicy, DeniedGenerationRequest, DeniedRequest, GeneratedRequest,
    GenerationAdmission, GenerationCoordinatorClient, GenerationOptions, GenerationOutcome,
    GenerationRequest, RequestContext, RouteOptions,
};
use anyhow::{Context, Result, bail};
use baseten_configmap::ConfigReader;
use dynamo_llm::http::service::forwarded_headers;
use dynamo_llm::session_affinity::{AffinityLease, AffinityTarget};
use dynamo_runtime::pipeline::{EngineStream, ResponseStream};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::future::BoxFuture;
use futures::{Stream, StreamExt};
use prost::Message;
use reqwest::{Client, Url};
use rmpv::Value;
use std::pin::Pin;
use std::sync::Arc;
use tokio_util::codec::{FramedRead, LengthDelimitedCodec};
use tokio_util::io::StreamReader;

mod b10_served_model_name;
mod pool;
mod scoring;
use pool::RemotePool;

enum RemoteEndpoints {
    Fixed(Url),
    Pool(Box<RemotePool>),
}

type FramedResponse = Pin<Box<dyn Stream<Item = Result<GenerationResponseFrameV1>> + Send>>;

/// Non-success HTTP status from the remote generation service, kept typed so
/// the frontend can return the upstream status (e.g. 529) instead of a 500.
#[derive(Debug)]
pub struct RemoteStatusError {
    pub status: reqwest::StatusCode,
    pub detail: String,
}

impl std::fmt::Display for RemoteStatusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "generation service returned HTTP {}: {}",
            self.status, self.detail
        )
    }
}

impl std::error::Error for RemoteStatusError {}

/// Native protobuf HTTP client with optional load- and session-aware remote selection.
pub struct RemoteGenerationCoordinator {
    endpoints: RemoteEndpoints,
    client: Client,
}

impl RemoteGenerationCoordinator {
    pub fn new(endpoint: impl AsRef<str>) -> Result<Self> {
        let endpoint = Url::parse(endpoint.as_ref()).context("invalid generation service URL")?;
        if !matches!(endpoint.scheme(), "http" | "https") {
            bail!("generation service URL must use http or https");
        }
        Ok(Self {
            endpoints: RemoteEndpoints::Fixed(endpoint),
            client: Client::new(),
        })
    }

    /// Reloadable remotes sharing one HTTP connection pool; distributed affinity requires a runtime.
    pub fn from_config(config: ConfigReader) -> Result<Self> {
        Self::from_runtime_config(config, None, None)
    }

    pub(crate) fn from_runtime_config(
        config: ConfigReader,
        namespace: Option<&dynamo_runtime::component::Namespace>,
        default_remote: Option<String>,
    ) -> Result<Self> {
        let client = Client::new();
        let endpoints = RemoteEndpoints::Pool(Box::new(RemotePool::new(
            config,
            client.clone(),
            namespace,
            default_remote,
        )?));
        Ok(Self { endpoints, client })
    }

    async fn generate_remote(
        &self,
        context: RequestContext,
        request: GenerationRequest,
        options: GenerationOptions,
    ) -> Result<GenerationOutcome> {
        validate_remote_options(&options)?;
        let served_model_name =
            b10_served_model_name::from_worker_request(&request.primary_worker_request);
        let mut wire_request = encode_request(&context, request)?;
        let mut headers = forwarded_headers::take(&mut wire_request.metadata)?;
        headers.remove(b10_served_model_name::HEADER);
        if let Some(name) = served_model_name {
            headers.insert(b10_served_model_name::HEADER, name);
        }
        // Nest the remote's spans under this frontend's, not beside it under the caller's.
        if let Some(trace) = context.trace_context() {
            headers.insert("traceparent", trace.create_traceparent().parse()?);
        }
        let request_context = context.inner();
        let session = wire_request
            .metadata
            .get(dynamo_llm::protocols::common::extensions::SESSION_AFFINITY_CONTEXT_KEY)
            .map(String::as_str)
            .filter(|id| !id.is_empty())
            .or_else(|| {
                wire_request
                    .routing
                    .as_ref()
                    .and_then(|routing| routing.session_id.as_deref())
            });
        if let Some(session_id) = session {
            headers.insert("x-session-affinity", session_id.parse()?);
        }
        let (mut endpoint, affinity) = match &self.endpoints {
            RemoteEndpoints::Pool(pool) => {
                let bid = protocol::BidRequestV1 {
                    token_buffer: wire_request.token_buffer.clone(),
                    tokens: wire_request.tokens.clone(),
                    mm_routing_args: wire_request.mm_routing_args.clone(),
                    cache_salt: wire_request
                        .routing
                        .as_ref()
                        .and_then(|routing| routing.cache_salt.clone()),
                    session_id: session.map(str::to_owned),
                };
                let allowed = wire_request
                    .routing
                    .as_ref()
                    .map_or(&[][..], |routing| routing.allowed_worker_ids.as_slice());
                tokio::select! {
                    result = pool.select(session, bid, allowed, &context) => result?,
                    _ = request_context.stopped() => return Ok(cancelled_outcome()),
                    _ = request_context.killed() => return Ok(cancelled_outcome()),
                }
            }
            RemoteEndpoints::Fixed(endpoint) => (endpoint.clone(), None),
        };
        b10_coordinate_path(&mut endpoint, &mut headers)?;
        let send = self
            .client
            .post(endpoint)
            .headers(headers)
            .header(
                reqwest::header::CONTENT_TYPE,
                protocol::REQUEST_CONTENT_TYPE,
            )
            .header(reqwest::header::ACCEPT, protocol::RESPONSE_CONTENT_TYPE)
            .body(wire_request.encode_to_vec())
            .send();
        let response = tokio::select! {
            response = send => response.context("generation service request failed")?,
            _ = request_context.stopped() => return Ok(cancelled_outcome()),
            _ = request_context.killed() => return Ok(cancelled_outcome()),
        };

        if !response.status().is_success() {
            let status = response.status();
            let detail = response
                .text()
                .await
                .unwrap_or_else(|_| "response body unavailable".to_string());
            return Err(RemoteStatusError { status, detail }.into());
        }

        let mut reader = response_frames(response);
        let first = tokio::select! {
            frame = reader.next() => frame.transpose()?,
            _ = request_context.stopped() => return Ok(cancelled_outcome()),
            _ = request_context.killed() => return Ok(cancelled_outcome()),
        }
        .context("generation service ended before an admission or denial")?;
        match first.frame {
            Some(generation_response_frame_v1::Frame::Admission(admission)) => {
                let admission: GenerationAdmission = admission.into();
                let lease = affinity
                    .map(|selection| {
                        selection.complete_selection(AffinityTarget {
                            worker_id: admission.prefill_worker_id,
                            dp_rank: None,
                        })
                    })
                    .transpose()?
                    .flatten();
                connected_outcome(context, admission, reader, lease)
            }
            Some(generation_response_frame_v1::Frame::Denied(denied)) => {
                let mut denied: DeniedGenerationRequest = denied.try_into()?;
                denied.usage = first.usage;
                if let (Some(selection), Some(admission)) = (affinity, denied.admission) {
                    selection.complete_selection(AffinityTarget {
                        worker_id: admission.prefill_worker_id,
                        dp_rank: None,
                    })?;
                }
                Ok(GenerationOutcome::Denied(denied))
            }
            Some(generation_response_frame_v1::Frame::Error(error)) => {
                bail!("generation service protocol error: {}", error.message)
            }
            Some(generation_response_frame_v1::Frame::ChunkMsgpack(_)) => {
                bail!("generation service sent a chunk before admission")
            }
            None => bail!("generation service sent an empty response frame"),
        }
    }
}

fn b10_coordinate_path(endpoint: &mut Url, headers: &mut reqwest::header::HeaderMap) -> Result<()> {
    if let Some(prefix) =
        headers.remove(dynamo_llm::http::service::b10_dedicated_routing::PREFIX_HEADER)
    {
        endpoint.set_path(&format!("{}/v1/coordinate", prefix.to_str()?));
    }
    Ok(())
}

async fn fetch_worker_loads(client: &Client, endpoint: &Url) -> Result<Vec<crate::WorkerLoad>> {
    let response = client
        .get(endpoint.join("worker_loads")?)
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await?
        .error_for_status()?;
    Ok(serde_json::from_slice(&response.bytes().await?)?)
}

async fn fetch_bid(
    client: &Client,
    endpoint: &Url,
    request: protocol::BidRequestV1,
) -> Result<protocol::BidResponseV1> {
    request.validate()?;
    let response = client
        .post(endpoint.join("bid")?)
        .header(
            reqwest::header::CONTENT_TYPE,
            protocol::REQUEST_CONTENT_TYPE,
        )
        .header(reqwest::header::ACCEPT, protocol::REQUEST_CONTENT_TYPE)
        .timeout(std::time::Duration::from_secs(5))
        .body(request.encode_to_vec())
        .send()
        .await?
        .error_for_status()?;
    Ok(protocol::BidResponseV1::decode(response.bytes().await?)?)
}

#[cfg(test)]
mod reload_tests;

#[cfg(test)]
mod affinity_tests;

fn cancelled_outcome() -> GenerationOutcome {
    GenerationOutcome::Denied(DeniedGenerationRequest {
        denied: DeniedRequest::Cancelled(),
        admission: None,
        usage: None,
    })
}

impl GenerationCoordinatorClient for RemoteGenerationCoordinator {
    fn bid(
        &self,
        request: protocol::BidRequestV1,
    ) -> BoxFuture<'_, Result<protocol::BidResponseV1>> {
        Box::pin(async {
            let request = request.into_token_buffer()?;
            match &self.endpoints {
                RemoteEndpoints::Fixed(endpoint) => {
                    fetch_bid(&self.client, endpoint, request).await
                }
                RemoteEndpoints::Pool(pool) => pool.bid(request).await,
            }
        })
    }
    fn start(&self) -> BoxFuture<'_, Result<()>> {
        Box::pin(async {
            if let RemoteEndpoints::Pool(pool) = &self.endpoints {
                pool.start().await?;
            }
            Ok(())
        })
    }
    fn worker_loads(&self) -> BoxFuture<'_, Result<Vec<crate::WorkerLoad>>> {
        Box::pin(async {
            match &self.endpoints {
                RemoteEndpoints::Pool(pool) => pool.worker_loads().await,
                RemoteEndpoints::Fixed(endpoint) => {
                    fetch_worker_loads(&self.client, endpoint).await
                }
            }
        })
    }
    fn generate(
        &self,
        context: RequestContext,
        request: GenerationRequest,
        options: GenerationOptions,
    ) -> BoxFuture<'_, Result<GenerationOutcome>> {
        Box::pin(self.generate_remote(context, request, options))
    }
}

fn validate_remote_options(options: &GenerationOptions) -> Result<()> {
    for (name, route) in [("primary", &options.primary), ("decode", &options.decode)] {
        if !route.require_available.is_empty()
            || route.cancellation != CancellationPolicy::Cancellable
            || route.max_reroutes != RouteOptions::default().max_reroutes
            || route.tracing_enabled
            || route.wait_for_first_response
            || route.phase.is_some()
        {
            bail!(
                "remote generation does not accept service-local {name} RouteOptions; configure them on the generation service"
            );
        }
    }
    Ok(())
}

fn connected_outcome(
    context: RequestContext,
    admission: GenerationAdmission,
    mut reader: FramedResponse,
    affinity_lease: Option<AffinityLease>,
) -> Result<GenerationOutcome> {
    let engine_context = context.inner();
    let cancellation = Arc::clone(&engine_context);
    let output = async_stream::stream! {
        let _affinity_lease = affinity_lease;
        loop {
            let frame = tokio::select! {
                frame = reader.next() => frame,
                _ = cancellation.stopped() => break,
                _ = cancellation.killed() => break,
            };
            match frame {
                Some(Ok(frame)) => match frame.frame {
                    Some(generation_response_frame_v1::Frame::ChunkMsgpack(chunk)) => {
                        match rmp_serde::from_slice::<Annotated<Value>>(&chunk) {
                            Ok(chunk) => yield chunk,
                            Err(error) => {
                                yield Annotated::from_error(format!(
                                    "invalid generation service chunk: {error}"
                                ));
                                break;
                            }
                        }
                    }
                    Some(generation_response_frame_v1::Frame::Error(error)) => {
                        yield Annotated::from_error(error.message);
                        break;
                    }
                    Some(generation_response_frame_v1::Frame::Admission(_)) => {
                        yield Annotated::from_error(
                            "generation service sent more than one admission",
                        );
                        break;
                    }
                    Some(generation_response_frame_v1::Frame::Denied(_)) => {
                        yield Annotated::from_error(
                            "generation service denied an already-admitted stream",
                        );
                        break;
                    }
                    None => {
                        yield Annotated::from_error(
                            "generation service sent an empty response frame",
                        );
                        break;
                    }
                },
                None => break,
                Some(Err(error)) => {
                    yield Annotated::from_error(format!(
                        "generation service response failed: {error}"
                    ));
                    break;
                }
            }
        }
    };
    let stream: EngineStream<Annotated<Value>> =
        ResponseStream::new(Box::pin(output), engine_context);
    Ok(GenerationOutcome::Connected(GeneratedRequest {
        stream,
        admission,
    }))
}

fn response_frames(response: reqwest::Response) -> FramedResponse {
    let body = response
        .bytes_stream()
        .map(|chunk| chunk.map_err(std::io::Error::other));
    let codec = LengthDelimitedCodec::builder()
        .max_frame_length(protocol::MAX_FRAME_BYTES)
        .new_codec();
    FramedRead::new(StreamReader::new(body), codec)
        .map(|frame| {
            GenerationResponseFrameV1::decode(frame?)
                .context("invalid generation service protobuf frame")
        })
        .boxed()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::codec::map_value;
    use crate::protocol::{AdmissionV1, NewRequestV1};
    use axum::Router;
    use axum::body::{Body, Bytes};
    use axum::extract::State;
    use axum::http::{HeaderMap, Response, StatusCode, header};
    use axum::routing::post;
    use dynamo_kv_router::protocols::{BlockExtraInfo, BlockMmObjectInfo, RoutingConstraints};
    use dynamo_runtime::logging::DistributedTraceContext;
    use dynamo_runtime::pipeline::AsyncEngineContext;
    use dynamo_runtime::pipeline::context::Controller;
    use futures::StreamExt;
    use std::collections::{BTreeMap, HashSet};
    use std::convert::Infallible;
    use std::sync::{Arc, Mutex};

    #[test]
    fn b10_coordinate_url_uses_only_the_arrived_prefix() {
        use dynamo_llm::http::service::b10_dedicated_routing::PREFIX_HEADER;
        for prefix in [
            None,
            Some("/environments/staging/sync"),
            Some("/deployment/abc/sync"),
        ] {
            let configured = if prefix.is_none() {
                "http://alyx/v1/coordinate?a=b"
            } else {
                "http://alyx/environments/legacy/sync/v1/coordinate?a=b"
            };
            let mut endpoint = Url::parse(configured).unwrap();
            let mut headers = HeaderMap::new();
            headers.insert(header::HOST, "model-abc.api.baseten.co".parse().unwrap());
            if let Some(prefix) = prefix {
                headers.insert(PREFIX_HEADER, prefix.parse().unwrap());
            }
            b10_coordinate_path(&mut endpoint, &mut headers).unwrap();
            assert_eq!(
                endpoint.as_str(),
                format!("http://alyx{}/v1/coordinate?a=b", prefix.unwrap_or(""))
            );
            assert!(!headers.contains_key(PREFIX_HEADER));
            assert_eq!(headers[header::HOST], "model-abc.api.baseten.co");
        }
    }

    #[derive(Clone)]
    struct TestServerState {
        captured: Arc<Mutex<Option<(NewRequestV1, HeaderMap)>>>,
        response: Vec<u8>,
    }

    async fn coordinate(
        State(state): State<TestServerState>,
        headers: HeaderMap,
        body: Bytes,
    ) -> Response<Body> {
        let request = NewRequestV1::decode(body).expect("valid request protobuf");
        protocol::validate_request(&request).expect("semantically valid request");
        *state.captured.lock().unwrap() = Some((request, headers));

        // Deliberately split both the frame header and payload. This exercises
        // the real streaming decoder rather than relying on HTTP chunk boundaries.
        let response = state.response;
        let split_at = [2, 11, response.len()];
        let mut start = 0;
        let chunks = split_at.into_iter().map(move |end| {
            let chunk = Bytes::copy_from_slice(&response[start..end]);
            start = end;
            Ok::<_, Infallible>(chunk)
        });
        Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, protocol::RESPONSE_CONTENT_TYPE)
            .body(Body::from_stream(futures::stream::iter(chunks)))
            .unwrap()
    }

    fn value(json: serde_json::Value) -> Value {
        serde_json::from_value(json).unwrap()
    }

    #[tokio::test]
    async fn remote_client_sends_typed_request_and_streams_framed_response() {
        for session_id in [Some("session-a"), None] {
            for model in [Some("test-model"), None] {
                b10_remote_client_roundtrip(session_id, model).await;
            }
        }
    }

    async fn b10_remote_client_roundtrip(session_id: Option<&str>, model: Option<&str>) {
        let first_chunk = Annotated::from_data(value(serde_json::json!({"token": 17})));
        let second_chunk = Annotated::from_data(value(serde_json::json!({"finish": true})));
        let frames = [
            GenerationResponseFrameV1 {
                usage: None,
                frame: Some(generation_response_frame_v1::Frame::Admission(
                    AdmissionV1 {
                        estimated_overlap_tokens: 64,
                        best_overlap_blocks: 3,
                        prefill_worker_id: 10,
                        prefill_dp_rank: 1,
                        decode_worker_id: Some(20),
                        decode_dp_rank: Some(2),
                        // Servers that predate the field omit it.
                        prefill_duration_ms: None,
                    },
                )),
            },
            GenerationResponseFrameV1 {
                usage: None,
                frame: Some(generation_response_frame_v1::Frame::ChunkMsgpack(
                    rmp_serde::to_vec_named(&first_chunk).unwrap(),
                )),
            },
            GenerationResponseFrameV1 {
                usage: None,
                frame: Some(generation_response_frame_v1::Frame::ChunkMsgpack(
                    rmp_serde::to_vec_named(&second_chunk).unwrap(),
                )),
            },
        ];
        let response = frames
            .iter()
            .flat_map(protocol::encode_response_frame)
            .collect::<Vec<_>>();
        let captured = Arc::new(Mutex::new(None));
        let app = Router::new()
            .route("/v1/coordinate", post(coordinate))
            .with_state(TestServerState {
                captured: Arc::clone(&captured),
                response,
            });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let coordinator =
            RemoteGenerationCoordinator::new(format!("http://{address}/v1/coordinate")).unwrap();
        let inner: Arc<dyn AsyncEngineContext> =
            Arc::new(Controller::new("request-123".to_string()));
        let trace_context: DistributedTraceContext = serde_json::from_value(serde_json::json!({
            "trace_id": "0123456789abcdef0123456789abcdef",
            "span_id": "0123456789abcdef",
            "parent_id": "fedcba9876543210",
            "tracestate": "vendor=value",
            "x_request_id": "external-123",
            "request_id": "trace-request-123"
        }))
        .unwrap();
        let context = RequestContext::new(
            inner,
            Some(trace_context),
            BTreeMap::from([
                ("tenant".to_string(), "acme".to_string()),
                (
                    forwarded_headers::FORWARDED_HEADERS_METADATA_KEY.to_string(),
                    serde_json::json!([
                        [
                            "traceparent",
                            b"00-0123456789abcdef0123456789abcdef-1111111111111111-01".to_vec()
                        ],
                        [
                            b10_served_model_name::HEADER,
                            b"client-supplied-model".to_vec()
                        ]
                    ])
                    .to_string(),
                ),
            ]),
        );
        let mut worker_request = value(serde_json::json!({
            "model": model,
            "streaming": true,
            "sampling_params": {"temperature": 0.25, "max_tokens": 8},
            "dynamic_temperature_rules": [{"after": 4, "temperature": 0.1}],
            "lora": "adapter-a",
            "cache_salt": "salt-a",
            "priority": -4,
            "service_tier": "priority",
            "user": session_id,
            "mm_args": {
                "mm_kwargs": ["image-kwargs"],
                "media_token_id": 32000,
                "mm_hashes": ["image-hash"],
                "mm_positions": [2],
                "mm_lengths": [4]
            }
        }));
        // Resolved media crosses as MessagePack binary, which a JSON fixture
        // cannot express; attach it to the parsed map directly.
        let Value::Map(entries) = &mut worker_request else {
            panic!("worker request must be a map")
        };
        let (_, Value::Map(mm_args)) = entries
            .iter_mut()
            .find(|(key, _)| key.as_str() == Some("mm_args"))
            .expect("mm_args present")
        else {
            panic!("mm_args must be a map")
        };
        mm_args.push((
            Value::from("mm_native_inputs"),
            Value::Array(vec![Value::Map(vec![
                (Value::from("type"), Value::from("image_url")),
                (Value::from("data"), Value::Binary(b"image-bytes".to_vec())),
            ])]),
        ));
        let routing_request = crate::RouterRequestNew {
            tokens: vec![1, 2, 3, 4],
            block_mm_infos: Some(vec![
                Some(BlockExtraInfo {
                    mm_objects: vec![BlockMmObjectInfo {
                        mm_hash: 99,
                        offsets: vec![(2, 4)],
                    }],
                }),
                None,
            ]),
            routing_constraints: RoutingConstraints {
                required_taints: HashSet::from(["gpu".to_string()]),
                preferred_taints: [("zone-a".to_string(), 0.75)].into_iter().collect(),
            },
            allowed_worker_ids: Some(HashSet::from([7, 5])),
            priority_jump: 1.5,
            priority_load_shed_percent: 25,
            do_not_queue: true,
        };
        let outcome = coordinator
            .generate(
                context,
                GenerationRequest {
                    routing_request,
                    primary_worker_request: worker_request,
                    decode_worker_request: None,
                },
                GenerationOptions::default(),
            )
            .await
            .unwrap();

        let GenerationOutcome::Connected(mut generated) = outcome else {
            panic!("expected an admitted request")
        };
        assert_eq!(generated.admission.prefill_worker_id, 10);
        assert_eq!(generated.admission.decode_worker_id, Some(20));
        assert_eq!(generated.admission.prefill_duration, None);
        assert_eq!(
            generated.stream.next().await.unwrap().data,
            first_chunk.data
        );
        assert_eq!(
            generated.stream.next().await.unwrap().data,
            second_chunk.data
        );
        assert!(generated.stream.next().await.is_none());

        let (captured, headers) = captured.lock().unwrap().take().unwrap();
        assert_eq!(
            headers
                .get(b10_served_model_name::HEADER)
                .map(|value| value.to_str().unwrap()),
            model
        );
        assert_eq!(
            headers
                .get("traceparent")
                .map(|value| value.to_str().unwrap()),
            Some("00-0123456789abcdef0123456789abcdef-0123456789abcdef-01")
        );
        assert_eq!(
            headers
                .get("x-session-affinity")
                .map(|value| value.to_str().unwrap()),
            session_id
        );
        assert_eq!(captured.request_id, "request-123");
        assert_eq!(
            captured
                .trace_context
                .as_ref()
                .unwrap()
                .x_request_id
                .as_deref(),
            Some("external-123")
        );
        assert_eq!(
            captured.metadata.get("tenant").map(String::as_str),
            Some("acme")
        );
        let request = captured;
        let worker: Value = rmp_serde::from_slice(&request.worker_msgpack).unwrap();
        assert_eq!(worker["model"].as_str(), model);
        assert!(request.tokens.is_empty());
        assert_eq!(
            request.token_buffer.as_ref().unwrap().payload.as_ref(),
            &[1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0],
        );
        assert_eq!(worker["lora"].as_str(), Some("adapter-a"));
        let routing = request.routing.unwrap();
        assert_eq!(routing.session_id.as_deref(), session_id);
        assert!(routing.do_not_queue);
        assert_eq!(routing.allowed_worker_ids, vec![5, 7]);
        assert_eq!(request.mm_routing_args.unwrap().blocks.len(), 2);
        let native = &request.mm_payloads.as_ref().unwrap().native_inputs[0];
        assert_eq!(native.kind, "image_url");
        assert_eq!(native.data.as_slice(), b"image-bytes".as_slice());
        assert!(
            native.url.is_empty(),
            "new writers must not emit a base64 URL"
        );
        assert_eq!(request.mm_payloads.unwrap().hashes, vec!["image-hash"]);
        let sampling: Value = rmp_serde::from_slice(&request.worker_msgpack).unwrap();
        assert_eq!(
            map_value(&sampling, "sampling_params")
                .and_then(|params| map_value(params, "max_tokens"))
                .and_then(Value::as_u64),
            Some(8)
        );

        server.abort();
    }
}
