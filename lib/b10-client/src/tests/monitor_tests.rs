// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Monitor gate end to end through the real `GenerationCoordinator`: scripted
//! worker stream, channel-fed score rows, assertions on the client stream.

use std::sync::Arc;

use anyhow::{Result, anyhow};
use dynamo_runtime::pipeline::EngineStream;
use dynamo_runtime::protocols::annotated::Annotated;
use futures::StreamExt;
use futures::channel::mpsc;
use rmpv::Value;
use tokio::time::Instant;

use super::{
    RouterGuardClientForTesting, build_test_context, generation_coordinator, jv,
    make_worker_request, route_response_new,
};
use crate::monitor::MonitorMessage;
use crate::monitor::tests::{
    DEADLINE, HOLD, REQUEST_ID, assert_ended, assert_held, assert_terminal, check, chunk, chunk_of,
    event, events, finish, msg, next, output,
};
use crate::{
    DisaggregationStrategy, GenerationCoordinator, GenerationOptions, GenerationOutcome,
    GenerationRequest, RouterRequestNew,
};

type Worker = Arc<RouterGuardClientForTesting>;
type Stream = EngineStream<Annotated<Value>>;

// e2e: the whole path through GenerationCoordinator
// ---------------------------------------------------------------------------

/// `make_worker_request` plus the opt-in list the binding would send.
fn worker_request(requested: Option<&[&str]>) -> Value {
    let mut request = serde_json::to_value(make_worker_request()).unwrap();
    request["requested_monitors"] = serde_json::json!(requested);
    serde_json::from_value(request).unwrap()
}

/// One monitored generation: the real coordinator, the coordinator-side gate, the
/// scripted worker stream, and the score channel that feeds the gate.
struct Fixture {
    coordinator: GenerationCoordinator,
    router: Worker,
    worker: Worker,
    scores: mpsc::UnboundedSender<MonitorMessage>,
    /// Exactly the items the worker will produce, in order.
    produced: Vec<Annotated<Value>>,
}

impl Fixture {
    /// Aggregated deployment whose gate deploys the fixture `on`-scoped, with the
    /// always-on monitor when `always_on`.
    async fn new(on: &str, always_on: bool) -> Self {
        let router =
            RouterGuardClientForTesting::new(vec![1], vec![1], vec![route_response_new(1)]);
        let worker = RouterGuardClientForTesting::new(vec![1], vec![1], vec![]);
        let (gate, scores) = crate::monitor::tests::gate(on, always_on).await;
        let coordinator = generation_coordinator(
            router.clone(),
            worker.clone(),
            None,
            None,
            DisaggregationStrategy::Aggregated,
            Some(gate),
        );
        Self {
            coordinator,
            router,
            worker,
            scores,
            produced: Vec::new(),
        }
    }

    /// Script the worker's chunks, yielding between them the way a decoding engine
    /// does (the shared reader and the per-request task need a turn first).
    fn script(&mut self, chunks: &[Annotated<Value>]) {
        self.produced = chunks.to_vec();
        let chunks = self.produced.clone();
        *self.worker.stream_override.lock().unwrap() = Some(Box::pin(async_stream::stream! {
            for item in chunks {
                tokio::task::yield_now().await;
                tokio::task::yield_now().await;
                tokio::task::yield_now().await;
                yield item;
            }
        }));
    }

    /// Send one score row per position from `start`, then let the gate file them.
    async fn rows(&self, start: u64, values: &[f32]) {
        self.scores
            .unbounded_send(msg(REQUEST_ID, start, values))
            .expect("scores open");
        for _ in 0..8 {
            tokio::task::yield_now().await;
        }
    }

    /// Whether the worker's route context was stopped (`ctx.stop_generating`).
    fn worker_stopped(&self) -> bool {
        let contexts = self.worker.route_contexts();
        contexts.last().is_some_and(|ctx| ctx.is_stopped())
    }

    /// Run one generation through the coordinator's front door.
    async fn generate(&self, requested: Option<&[&str]>, prompt: Vec<u32>) -> Result<Stream> {
        let routing_request = RouterRequestNew {
            tokens: prompt,
            ..Default::default()
        };
        let request = GenerationRequest {
            routing_request,
            primary_worker_request: worker_request(requested),
            decode_worker_request: None,
        };
        let context = build_test_context(REQUEST_ID);
        let options = GenerationOptions::default();
        match self.coordinator.generate(context, request, options).await? {
            GenerationOutcome::Connected(generated) => Ok(generated.stream),
            GenerationOutcome::Denied(denied) => Err(anyhow!("denied: {denied:?}")),
        }
    }

    /// Open a generation with the `harm` opt-in and `prompt` routing tokens.
    async fn monitored(&self, prompt: Vec<u32>) -> Stream {
        self.generate(Some(&["harm"]), prompt)
            .await
            .expect("monitored")
    }
}

/// Unmonitored (no opt-in, no always-on monitor): the client stream is the worker's
/// own items, carries no monitor field, and never waits for a row.
#[tokio::test(start_paused = true)]
async fn unmonitored_stream_is_byte_identical_and_never_waits() {
    let mut fixture = Fixture::new("output", false).await;
    fixture.script(&[chunk(&[0]), chunk(&[1]), chunk(&[2]), finish(&[3])]);
    let stream = fixture.generate(None, vec![]).await.expect("unmonitored");

    let items: Vec<Annotated<Value>> = tokio::time::timeout(DEADLINE, stream.collect())
        .await
        .expect("stream ends");
    assert_eq!(items.len(), fixture.produced.len());
    for (item, produced) in items.iter().zip(&fixture.produced) {
        assert_eq!(format!("{item:#?}"), format!("{produced:#?}"), "untouched");
        assert!(events(item, 0).is_nil(), "no monitor field unmonitored");
    }
    assert_eq!(fixture.worker.calls().len(), 1, "one worker open, no rows");
}

/// Monitored pass-through: every chunk carries `monitor_events`; order and content untouched.
#[tokio::test(start_paused = true)]
async fn monitored_events_ride_their_chunk_and_content_is_untouched() {
    let mut fixture = Fixture::new("output", false).await;
    fixture.script(&[
        chunk(&[0, 1]),
        chunk(&[2, 3]),
        chunk(&[4, 5]),
        chunk(&[6, 7]),
        finish(&[8]),
    ]);
    let mut out = fixture.monitored(vec![]).await;

    // A token's row rides the chunk carrying the NEXT token, so each chunk scores the
    // tokens before its newest one: its rows must be sent before it is pulled.
    let scripted: [(u64, Vec<f32>); 5] = [
        (0, vec![0.25, 0.25]),
        (1, vec![0.25, 0.25]),
        (3, vec![0.0, 0.25]),
        (5, vec![0.25, 0.75]),
        (7, vec![0.25]),
    ];
    let expected = [jv!({}), jv!({}), jv!({}), event(0.75), jv!({})];
    for (i, (start, values)) in scripted.iter().enumerate() {
        fixture.rows(*start, values).await;
        check(
            &next(&mut out).await,
            &fixture.produced[i],
            &[expected[i].clone()],
        );
    }
    assert_ended(&mut out).await;
}

/// The gate holds a chunk until the rows for its scored tokens exist.
#[tokio::test(start_paused = true)]
async fn chunk_waits_for_its_rows_then_flows() {
    let mut fixture = Fixture::new("output", false).await;
    fixture.script(&[chunk(&[0, 1]), chunk(&[2, 3]), finish(&[4])]);
    let mut out = fixture.monitored(vec![]).await;

    fixture.rows(0, &[0.25, 0.25]).await;
    check(&next(&mut out).await, &fixture.produced[0], &[jv!({})]);

    // Row 2 is withheld: chunk 1 needs it and is held, and the hold has not expired
    // either -- which the quiet release below proves.
    let started = Instant::now();
    assert_held(&mut out).await;
    assert!(Instant::now() - started < HOLD, "hold has not expired");
    fixture.rows(2, &[0.25, 0.25]).await;
    check(&next(&mut out).await, &fixture.produced[1], &[jv!({})]);
    check(&next(&mut out).await, &fixture.produced[2], &[jv!({})]);
    assert_ended(&mut out).await;
}

/// The response's final token has no forward pass, so it never blocks release; the
/// finish chunk is quiet, and an empty finish chunk needs no rows at all.
#[tokio::test(start_paused = true)]
async fn finish_chunks_release_without_the_final_token_row() {
    let mut fixture = Fixture::new("output", false).await;
    fixture.script(&[chunk(&[0, 1]), finish(&[2, 3])]);
    let mut out = fixture.monitored(vec![]).await;

    fixture.rows(0, &[0.25, 0.25]).await;
    check(&next(&mut out).await, &fixture.produced[0], &[jv!({})]);
    // Row 2 scores the finish chunk's first token; row 3 (the last) never arrives.
    fixture.rows(2, &[0.25]).await;
    check(&next(&mut out).await, &fixture.produced[1], &[jv!({})]);
    assert_ended(&mut out).await;

    // Separate empty finish chunk, no rows sent for it: released, quiet.
    let mut fixture = Fixture::new("output", false).await;
    fixture.script(&[chunk(&[0, 1]), chunk_of(&[(0, &[])], true)]);
    let mut out = fixture.monitored(vec![]).await;
    fixture.rows(0, &[0.6, 0.25]).await;
    check(&next(&mut out).await, &fixture.produced[0], &[event(0.6)]);
    check(&next(&mut out).await, &fixture.produced[1], &[jv!({})]);
    assert_ended(&mut out).await;
}

/// A high value is reported on the evaluating chunk and never acted on: the
/// coordinator carries events, the frontend processor decides to stop.
#[tokio::test(start_paused = true)]
async fn high_value_is_reported_not_enforced() {
    let mut fixture = Fixture::new("output", false).await;
    fixture.script(&[chunk(&[0, 1]), finish(&[2, 3])]);
    let mut out = fixture.monitored(vec![]).await;

    // Rows lead the chunks: row 0 releases chunk 0 quietly, rows 1 and 2 release
    // chunk 1, which scores the 0.95 row of token 1.
    fixture.rows(0, &[0.25, 0.95, 0.0]).await;
    check(&next(&mut out).await, &fixture.produced[0], &[jv!({})]);
    check(&next(&mut out).await, &fixture.produced[1], &[event(0.95)]);
    assert_ended(&mut out).await;
    assert!(
        !fixture.worker_stopped(),
        "coordinator never stops on a value"
    );
}

/// Nothing arrives: after the hold the first withheld chunk becomes a `content_filter`
/// terminal chunk with `monitor_error`, and the stream stops.
#[tokio::test(start_paused = true)]
async fn hold_timeout_stops_with_monitor_error() {
    let mut fixture = Fixture::new("output", false).await;
    fixture.script(&[chunk(&[0, 1]), chunk(&[2, 3])]);
    let mut out = fixture.monitored(vec![]).await;

    let started = Instant::now();
    let item = next(&mut out).await;
    assert_terminal(&item, 0);
    assert_eq!(events(&item, 0), jv!({}));
    let reason = output(&item, 0)["monitor_error"].as_str().unwrap();
    assert!(reason.contains("did not arrive"), "{reason}");
    assert!(Instant::now() - started >= HOLD, "waited for the hold");
    assert_ended(&mut out).await;
    assert!(fixture.worker_stopped());
}

/// Prompt events ride the first chunk; a prompt-scoped monitor never fires on output.
#[tokio::test(start_paused = true)]
async fn prompt_events_ride_the_first_chunk() {
    let mut fixture = Fixture::new("prompt", false).await;
    fixture.script(&[chunk(&[0, 1]), chunk(&[2, 3]), finish(&[4])]);
    let mut out = fixture.monitored(vec![1, 2, 3]).await;

    // Three routing tokens => prompt positions 0..3, plus the first chunk's own first
    // output token (position 3): rows 0..4 release it, and its prompt rows fire.
    fixture.rows(0, &[0.75, 0.25, 0.25, 0.0]).await;
    check(&next(&mut out).await, &fixture.produced[0], &[event(0.75)]);
    // A loud output row raises nothing for a prompt-scoped monitor.
    fixture.rows(4, &[0.75, 0.75]).await;
    check(&next(&mut out).await, &fixture.produced[1], &[jv!({})]);
    fixture.rows(6, &[0.75]).await;
    check(&next(&mut out).await, &fixture.produced[2], &[jv!({})]);
}

/// An unknown monitor name fails the request before routing starts.
#[tokio::test(start_paused = true)]
async fn unknown_requested_monitor_fails_before_routing() {
    let fixture = Fixture::new("output", false).await;
    let Err(error) = fixture.generate(Some(&["nope"]), vec![]).await else {
        panic!("an unknown monitor must fail the request")
    };
    let error = format!("{error:#}");
    assert!(error.contains("unknown monitors"), "{error}");
    let routed = fixture.router.calls();
    assert!(
        routed
            .iter()
            .all(|(_, data)| data["method"].as_str() != Some("new")),
        "routing never started"
    );
}

/// An always-on monitor monitors the request with no opt-in at all.
#[tokio::test(start_paused = true)]
async fn always_on_monitor_monitors_without_an_opt_in() {
    let mut fixture = Fixture::new("output", true).await;
    fixture.script(&[chunk(&[0, 1]), finish(&[2])]);
    let mut out = fixture.generate(None, vec![]).await.expect("always on");

    fixture.rows(0, &[0.75, 0.25]).await;
    check(
        &next(&mut out).await,
        &fixture.produced[0],
        &[jv!({"sustained": 0.75})],
    );
    check(&next(&mut out).await, &fixture.produced[1], &[jv!({})]);
}

/// Monitored generation requires a single choice: a MonitorMessage carries no
/// choice index, so n > 1 is rejected before routing.
#[tokio::test(start_paused = true)]
async fn monitored_generation_rejects_multiple_choices() {
    let fixture = Fixture::new("output", false).await;
    let mut request = serde_json::to_value(make_worker_request()).unwrap();
    request["requested_monitors"] = serde_json::json!(["harm"]);
    request["sampling_options"] = serde_json::json!({ "n": 2 });
    let primary_worker_request: Value = serde_json::from_value(request).unwrap();
    let request = GenerationRequest {
        routing_request: RouterRequestNew {
            tokens: vec![],
            ..Default::default()
        },
        primary_worker_request,
        decode_worker_request: None,
    };
    let context = build_test_context(REQUEST_ID);
    let Err(error) = fixture
        .coordinator
        .generate(context, request, GenerationOptions::default())
        .await
    else {
        panic!("n > 1 must be rejected");
    };
    assert!(format!("{error:#}").contains("n = 1"), "{error}");
    assert!(
        fixture
            .router
            .calls()
            .iter()
            .all(|(_, data)| data["method"].as_str() != Some("new")),
        "routing never started"
    );
}

// ---------------------------------------------------------------------------
