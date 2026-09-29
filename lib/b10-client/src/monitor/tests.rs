// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Monitor gate unit tests (config, evaluate, gate, event plane) and the test
//! helpers shared with the e2e tests in `crate::tests::monitor_tests`.

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use dynamo_runtime::component::Namespace;
use dynamo_runtime::distributed::DistributedConfig;
use dynamo_runtime::pipeline::context::Controller;
use dynamo_runtime::pipeline::{AsyncEngineContext, EngineStream, ResponseStream};
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::transports::event_plane::EventPublisher;
use dynamo_runtime::{DistributedRuntime, Runtime};
use futures::FutureExt;
use futures::StreamExt;
use futures::channel::mpsc;
use rmpv::Value;
use tokio::time::Instant;

use super::*;
use crate::tests::jv;

type Stream = EngineStream<Annotated<Value>>;

// `pub(crate)` items are shared with the e2e tests in `crate::tests::monitor_tests`.
/// Request id the coordinator registers; the sidecar echoes it on every row.
pub(crate) const REQUEST_ID: &str = "mon-req";
/// Gate hold: long enough that a legitimate row release is never the timeout.
pub(crate) const HOLD: Duration = Duration::from_millis(200);
/// A stalled gate is a bug, not something to hang on (paused clock: instant).
pub(crate) const DEADLINE: Duration = Duration::from_secs(5);
/// Prompt positions in the gate tests.
const PROMPT_LEN: u64 = 2;

/// The realistic deployment fixture: `harm` on request and the always-on
/// `sustained`, both linear over the single `harm` stream.
const FIXTURE: &str = include_str!("../../../llm/tests/fixtures/monitoring.toml");

/// The fixture deployment, `on`-scoped, with `sustained` dropped when it is not wanted.
fn deployment(on: &str, always_on: bool) -> String {
    let mut doc: toml::Value = FIXTURE.parse().expect("fixture is valid TOML");
    let monitors = doc["monitoring"]["monitors"]
        .as_table_mut()
        .expect("fixture declares monitors");
    monitors["harm"]["config"]["on"] = on.into();
    if always_on {
        monitors["sustained"]["config"]["on"] = on.into();
    } else {
        monitors.remove("sustained");
    }
    toml::to_string(&doc).expect("deployment serializes")
}

/// One monitor `m` over streams `a`, `b` with extra `[monitoring]` keys,
/// monitor keys, and `config` body.
fn one(section: &str, monitor: &str, config: &str) -> String {
    format!(
        "[monitoring]\nversion = 1\nstreams = [\"a\", \"b\"]\n{section}\n\
         [monitoring.monitors.m]\nreturn_when = \"always\"\nevent_threshold = 0.5\n{monitor}\n\
         [monitoring.monitors.m.config]\n{config}\n"
    )
}

/// One worker chunk: `(index, sampled tokens)` per output, `finish_reason` "stop" when finished.
pub(crate) fn chunk_of(outs: &[(u64, &[u64])], finished: bool) -> Annotated<Value> {
    let outputs: Vec<Value> = outs
        .iter()
        .map(|(index, tokens)| {
            let mut output = jv!({"index": index, "token_ids_diff": tokens});
            if let (true, Value::Map(map)) = (finished, &mut output) {
                map.push((Value::from("finish_reason"), Value::from("stop")));
            }
            output
        })
        .collect();
    Annotated::from_data(jv!({"finished": finished, "outputs": outputs}))
}

/// One output, `tokens` sampled, not finished.
pub(crate) fn chunk(tokens: &[u64]) -> Annotated<Value> {
    chunk_of(&[(0, tokens)], false)
}

/// Final chunk for output 0: finished, `finish_reason` "stop", `tokens` sampled.
pub(crate) fn finish(tokens: &[u64]) -> Annotated<Value> {
    chunk_of(&[(0, tokens)], true)
}

/// `outputs[i]` of a chunk.
pub(crate) fn output(item: &Annotated<Value>, i: usize) -> &Value {
    &item.data.as_ref().expect("chunk carries data")["outputs"][i]
}

/// `outputs[i].monitor_events`; `Nil` when the field is absent.
pub(crate) fn events(item: &Annotated<Value>, i: usize) -> Value {
    output(item, i)["monitor_events"].clone()
}

/// The event a row of `v` raises under linear calibration (f32 row, widened f64).
pub(crate) fn event(v: f32) -> Value {
    jv!({"harm": f64::from(v)})
}

/// The client got exactly the produced item, plus `monitor_events` on every output.
pub(crate) fn check(item: &Annotated<Value>, produced: &Annotated<Value>, fired: &[Value]) {
    let mut expected = serde_json::to_value(produced.data.as_ref().unwrap()).unwrap();
    let outputs = expected["outputs"].as_array_mut().unwrap();
    for (output, fired) in outputs.iter_mut().zip(fired) {
        output["monitor_events"] = serde_json::to_value(fired).unwrap();
    }
    let received = serde_json::to_value(item.data.as_ref().unwrap()).unwrap();
    assert_eq!(received, expected);
}

/// The substituted terminal chunk: finished, `content_filter`, no tokens.
pub(crate) fn assert_terminal(item: &Annotated<Value>, i: usize) {
    let output = output(item, i);
    let finished = item.data.as_ref().unwrap()["finished"].as_bool();
    assert_eq!(finished, Some(true));
    assert_eq!(output["finish_reason"].as_str(), Some("content_filter"));
    assert!(output["token_ids_diff"].as_array().unwrap().is_empty());
}

/// Nothing may be released while a chunk's rows are still missing.
pub(crate) async fn assert_held(stream: &mut Stream) {
    let window = tokio::time::timeout(Duration::from_millis(50), stream.next()).await;
    assert!(window.is_err(), "chunk released before its rows");
}

/// One score row per position from `start` for a single-stream sidecar.
pub(crate) fn msg(request_id: &str, start: u64, rows: &[f32]) -> MonitorMessage {
    MonitorMessage {
        request_id: request_id.to_string(),
        start,
        rows: rows.iter().map(|&v| vec![v]).collect(),
    }
}

/// The next chunk, failing rather than hanging on a stalled gate.
pub(crate) async fn next(stream: &mut Stream) -> Annotated<Value> {
    let chunk = tokio::time::timeout(DEADLINE, stream.next()).await;
    chunk.expect("chunk arrives").expect("chunk")
}

/// The stream ends here: no further chunk, and no hang.
pub(crate) async fn assert_ended(stream: &mut Stream) {
    let item = tokio::time::timeout(DEADLINE, stream.next())
        .await
        .expect("stream ends");
    assert!(item.is_none(), "extra chunk: {item:?}");
}

// ---------------------------------------------------------------------------
// config: parsing, validation, monitor selection
// ---------------------------------------------------------------------------

#[test]
fn parses_example_config() {
    let c = MonitoringConfig::parse(FIXTURE).unwrap();
    assert_eq!(c.names(), ["harm", "sustained"]);
    assert_eq!(c.width, 1);
    let (harm, sustained) = (&c.monitors[0], &c.monitors[1]);
    assert_eq!(harm.return_when, ReturnWhen::OnRequest);
    assert_eq!(harm.columns, [(0, 1.0)]);
    assert_eq!((harm.on, harm.repeat), (Scope::Output, Repeat::Always));
    // Explicit config keys parse.
    assert_eq!((harm.temperature, harm.bias), (1.0, 0.0));
    assert_eq!(harm.calibration, Calibration::Linear);
    assert_eq!(sustained.return_when, ReturnWhen::Always);
    assert_eq!(sustained.columns, [(0, 1.0)]);
    assert_eq!(sustained.on, Scope::Output);
    assert_eq!(sustained.calibration, Calibration::Linear);
    // Defaults for omitted config keys.
    assert_eq!((sustained.temperature, sustained.bias), (1.0, 0.0));
    assert_eq!(sustained.repeat, Repeat::Always);
    let minimal = |config: &str| MonitoringConfig::parse(&one("", "", config)).unwrap();
    assert_eq!(
        minimal("probe = \"a\"").monitors[0].calibration,
        Calibration::Sigmoid
    );
    assert_eq!(
        minimal("probe = \"a\"\nrepeat = \"once\"")
            .monitors
            .remove(0)
            .repeat,
        Repeat::Once
    );
    // No [monitoring.monitors] deploys nothing.
    let empty = MonitoringConfig::parse("[monitoring]\nversion = 1").unwrap();
    assert!(empty.monitors.is_empty());
}

#[test]
fn rejects_invalid_configs() {
    let probe = "probe = \"a\"";
    let cases = [
        (format!("[other]\n{}", one("", "", probe)), "unknown field"),
        (one("foo = 1", "", probe), "unknown field"),
        (
            one("[monitoring.capture]\nenabled = true", "", probe),
            "capture",
        ),
        (
            "[monitoring]\nversion = 2".into(),
            "unsupported monitoring version",
        ),
        (
            "[monitoring]\nstreams = []".into(),
            "missing field `version`",
        ),
        (
            "[monitoring]\nversion = 1\n[monitoring.monitors]".into(),
            "at least one monitor",
        ),
        (
            one("", "", probe).replace("streams = [\"a\", \"b\"]", ""),
            "streams",
        ),
        (one("", "", probe).replace("\"b\"", "\"a\""), "repeat"),
        (one("", "window_size = 3", probe), "unknown field"),
        (
            one("", "", probe).replace("event_threshold = 0.5", ""),
            "missing field `event_threshold`",
        ),
        (
            one("", "", probe).replace("\"always\"", "\"never\""),
            "unknown variant",
        ),
        (one("", "", probe).replace("0.5", "true"), "invalid type"),
        (
            one("", "", probe).replace("0.5", "nan"),
            "event_threshold must be a finite",
        ),
        (
            one("", "", probe)
                .replace(".m]", ".\" \"]")
                .replace(".m.", ".\" \"."),
            "non-empty",
        ),
        (
            one("", "", "probe = \"a\"\naggregation = \"ema\""),
            "unknown field",
        ),
        (
            one("", "", "probe = \"a\"\nprobes = {b = 1.0}"),
            "exactly one of probe, probes",
        ),
        (
            one("", "", "on = \"output\""),
            "exactly one of probe, probes",
        ),
        (one("", "", "probes = {}"), "non-empty"),
        (
            one("", "", "probes = {a = inf}"),
            "probes.a must be a finite",
        ),
        (one("", "", "probe = \"c\""), "unknown probe \"c\""),
        (
            one("", "", "probe = \"a\"\ntemperature = 0"),
            "temperature must be positive",
        ),
        (
            one("", "", "probe = \"a\"\nbias = nan"),
            "bias must be a finite",
        ),
        (
            one("", "", "probe = \"a\"\non = \"input\""),
            "unknown variant",
        ),
        (
            one("", "", "probe = \"a\"\ncalibration = \"tanh\""),
            "unknown variant",
        ),
        (
            one("", "", "probe = \"a\"\nrepeat = \"twice\""),
            "unknown variant",
        ),
        (
            one("", "stop_threshold = 0.4", probe),
            "must be >= event_threshold",
        ),
        (
            one(
                "",
                "stop_threshold = 0.9",
                "probe = \"a\"\nrepeat = \"once\"",
            ),
            "must equal it",
        ),
    ];
    for (text, why) in cases {
        let err = format!("{:#}", MonitoringConfig::parse(&text).unwrap_err());
        assert!(err.contains(why), "want {why:?} for\n{text}\ngot: {err}");
    }
}

#[test]
fn selects_requested_monitors() {
    let c = MonitoringConfig::parse(FIXTURE).unwrap();
    let names = |requested: Option<&[String]>| {
        c.select(requested)
            .map(|s| s.map(|s| s.into_iter().map(|m| m.name).collect::<Vec<_>>()))
    };
    assert_eq!(names(None).unwrap(), Some(vec!["sustained".to_string()]));
    assert_eq!(
        names(Some(&[])).unwrap(),
        Some(vec!["sustained".to_string()])
    );
    let harm = ["harm".to_string()];
    assert_eq!(
        names(Some(&harm)).unwrap(),
        Some(vec!["harm".into(), "sustained".into()])
    );
    let err = names(Some(&["nope".to_string()])).unwrap_err().to_string();
    assert!(err.contains("unknown monitors [\"nope\"]"), "{err}");
    // Only on-request monitors: no opt-in is unmonitored, an empty list is monitored.
    let c = MonitoringConfig::parse(&deployment("output", false)).unwrap();
    assert!(c.select(None).unwrap().is_none());
    assert!(c.select(Some(&[])).unwrap().is_some_and(|s| s.is_empty()));
}

// ---------------------------------------------------------------------------
// evaluate: per-row calibration and event semantics
// ---------------------------------------------------------------------------

/// `m`'s value over `rows` in `scope`, or `None` when it did not fire.
fn fire(config: &str, fired: &mut HashSet<String>, rows: &[[f32; 2]], scope: Scope) -> Option<f64> {
    let c = MonitoringConfig::parse(&one("", "", config)).unwrap();
    let rows: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
    let mut events = BTreeMap::new();
    evaluate(&c.monitors, fired, &rows, scope, &mut events);
    events.get("m").copied()
}

#[test]
fn evaluate_semantics() {
    let fresh = || HashSet::new();
    let out = Scope::Output;
    let a = "probe = \"a\"";
    // Sigmoid(0) = 0.5 meets event_threshold 0.5: equality qualifies.
    assert_eq!(fire(a, &mut fresh(), &[[0.0, 9.0]], out), Some(0.5));
    assert_eq!(fire(a, &mut fresh(), &[[-0.1, 9.0]], out), None);
    // logit / temperature + bias = 2 / 2 - 1 = 0.
    let tb = "probe = \"a\"\ntemperature = 2.0\nbias = -1.0";
    assert_eq!(fire(tb, &mut fresh(), &[[2.0, 0.0]], out), Some(0.5));
    // Linear emits the calibrated logit; the largest qualifying value per segment.
    let lin = "probe = \"a\"\ncalibration = \"linear\"";
    let rows = [[0.75, 0.0], [3.0, 0.0], [0.25, 0.0]];
    assert_eq!(fire(lin, &mut fresh(), &rows, out), Some(3.0));
    // Weighted sum of probes: 0.5 * 2 + 2 * 0.25.
    let sum = "probes = {a = 0.5, b = 2.0}\ncalibration = \"linear\"";
    assert_eq!(fire(sum, &mut fresh(), &[[2.0, 0.25]], out), Some(1.5));
    // Scope: output by default; prompt and both.
    assert_eq!(fire(lin, &mut fresh(), &[[1.0, 0.0]], Scope::Prompt), None);
    let prompt = "probe = \"a\"\ncalibration = \"linear\"\non = \"prompt\"";
    assert_eq!(
        fire(prompt, &mut fresh(), &[[1.0, 0.0]], Scope::Prompt),
        Some(1.0)
    );
    assert_eq!(fire(prompt, &mut fresh(), &[[1.0, 0.0]], out), None);
    let both = "probe = \"a\"\ncalibration = \"linear\"\non = \"both\"";
    assert_eq!(
        fire(both, &mut fresh(), &[[1.0, 0.0]], Scope::Prompt),
        Some(1.0)
    );
    assert_eq!(fire(both, &mut fresh(), &[[1.0, 0.0]], out), Some(1.0));
    // repeat = once: the first qualifying value, then silent across segments.
    let once = "probe = \"a\"\ncalibration = \"linear\"\nrepeat = \"once\"";
    let mut fired = fresh();
    assert_eq!(
        fire(once, &mut fired, &[[1.0, 0.0], [5.0, 0.0]], out),
        Some(1.0)
    );
    assert_eq!(fire(once, &mut fired, &[[7.0, 0.0]], out), None);
    let mut fired = fresh();
    assert_eq!(fire(lin, &mut fired, &[[1.0, 0.0]], out), Some(1.0));
    assert_eq!(fire(lin, &mut fired, &[[7.0, 0.0]], out), Some(7.0));
}

// ---------------------------------------------------------------------------
// gate: holding, hold failures, registration
// ---------------------------------------------------------------------------

/// A gate over the fixture deployment (channel feed), plus its score sender.
pub(crate) async fn gate(
    on: &str,
    always_on: bool,
) -> (Arc<MonitorGate>, mpsc::UnboundedSender<MonitorMessage>) {
    let (tx, rx) = mpsc::unbounded();
    let feed: MonitorFeed = Box::pin(rx);
    let config = MonitoringConfig::parse(&deployment(on, always_on)).expect("config parses");
    let handle = tokio::runtime::Handle::current();
    let gate = MonitorGate::new(config, HOLD, feed, &handle);
    (gate, tx)
}

/// Register [`REQUEST_ID`] (the always-on monitors only) and wrap `chunks`.
fn run(g: &MonitorGate, chunks: Vec<Annotated<Value>>) -> (Stream, Arc<dyn AsyncEngineContext>) {
    let ctx: Arc<dyn AsyncEngineContext> = Arc::new(Controller::new(REQUEST_ID.to_string()));
    let inner = ResponseStream::new(Box::pin(futures::stream::iter(chunks)), ctx.clone());
    let mailbox = g
        .subscribe(REQUEST_ID, None, PROMPT_LEN)
        .unwrap()
        .expect("monitored");
    (MonitorGate::wrap(inner, mailbox, ctx.clone()), ctx)
}

/// A chunk is held until its rows exist, however they arrive; prompt events
/// ride the first chunk; the final token may stay unscored.
#[tokio::test(start_paused = true)]
async fn holds_until_rows_cover_the_chunk() {
    let (g, tx) = gate("both", true).await;
    let (mut out, ctx) = run(&g, vec![chunk(&[10, 11]), finish(&[12])]);
    tx.unbounded_send(msg(REQUEST_ID, 0, &[0.0, 0.75])).unwrap(); // prompt
    let wait = Duration::from_millis(100);
    assert!(
        tokio::time::timeout(wait, out.next()).await.is_err(),
        "held: output rows missing"
    );
    // Out of order, with a radix replay of the prompt: first copy wins, so no stop.
    tx.unbounded_send(msg(REQUEST_ID, 3, &[0.25])).unwrap();
    tx.unbounded_send(msg(REQUEST_ID, 0, &[9.0, 9.0])).unwrap();
    tx.unbounded_send(msg(REQUEST_ID, 2, &[0.25])).unwrap();
    let started = Instant::now();
    let first = out.next().await.unwrap();
    assert_eq!(output(&first, 0)["token_ids_diff"], jv!([10, 11]));
    assert_eq!(
        events(&first, 0),
        jv!({"sustained": 0.75}),
        "prompt event on the first chunk"
    );
    // Position 4 is the final token: released unscored, quiet.
    let last = out.next().await.unwrap();
    assert_eq!(events(&last, 0), jv!({}));
    assert_eq!(output(&last, 0)["finish_reason"].as_str(), Some("stop"));
    assert!(out.next().await.is_none());
    assert_eq!(Instant::now(), started, "released as soon as covered");
    assert!(!ctx.is_stopped());
}

/// A hold timeout or an impossible row never releases held content: the chunk
/// becomes content_filter with `monitor_error` and empty `monitor_events`.
#[tokio::test(start_paused = true)]
async fn hold_failure_stops_with_monitor_error() {
    // A 2-wide row for a 1-stream deployment can never cover position 0.
    let bad = MonitorMessage {
        request_id: REQUEST_ID.into(),
        start: 0,
        rows: vec![vec![0.0, 0.0]],
    };
    // Prompt scope makes the first chunk wait for rows 0..PROMPT_LEN.
    for (rows, reason) in [(None, "did not arrive"), (Some(bad), "bad score row")] {
        let (g, tx) = gate("both", true).await;
        let (mut out, ctx) = run(&g, vec![chunk(&[10]), chunk(&[11])]);
        if let Some(bad) = rows {
            tx.unbounded_send(bad).unwrap();
        }
        let started = Instant::now();
        let item = out.next().await.unwrap();
        assert_eq!(
            output(&item, 0)["finish_reason"].as_str(),
            Some("content_filter")
        );
        assert_eq!(
            output(&item, 0)["token_ids_diff"],
            jv!([]),
            "held tokens withheld"
        );
        assert_eq!(events(&item, 0), jv!({}));
        let error = output(&item, 0)["monitor_error"].as_str().unwrap();
        assert!(error.contains(reason), "{error}");
        assert_eq!(Instant::now() - started >= HOLD, reason == "did not arrive");
        assert!(ctx.is_stopped());
        assert!(out.next().await.is_none());
    }
}

/// No always-on monitor and no opt-in: not registered, so never wrapped.
#[tokio::test]
async fn unmonitored_request_is_not_registered() {
    let (g, _tx) = gate("output", false).await;
    assert!(g.subscribe(REQUEST_ID, None, PROMPT_LEN).unwrap().is_none());
    assert!(g.slots.lock().unwrap().is_empty());
    assert!(
        g.subscribe(REQUEST_ID, Some(&["nope".to_string()]), PROMPT_LEN)
            .is_err()
    );
    assert_eq!(g.config.names(), ["harm"]);
}

/// A Migration retry re-registers the request id with a new sender before the
/// stale mailbox drops; that drop must not unregister the live sender.
#[tokio::test(start_paused = true)]
async fn retry_reregistration_keeps_the_new_sender() {
    let (g, tx) = gate("output", true).await;
    let first = g
        .subscribe(REQUEST_ID, None, PROMPT_LEN)
        .unwrap()
        .expect("monitored");
    let ctx: Arc<dyn AsyncEngineContext> = Arc::new(Controller::new(REQUEST_ID.to_string()));
    let inner = ResponseStream::new(
        Box::pin(futures::stream::iter(vec![finish(&[10, 11])])),
        ctx.clone(),
    );
    // The retry replaces the slot, then the stale first mailbox drops.
    let second = g
        .subscribe(REQUEST_ID, None, PROMPT_LEN)
        .unwrap()
        .expect("monitored");
    let mut out = MonitorGate::wrap(inner, second, ctx.clone());
    drop(first);
    // Position 2 scores the chunk; it reaches the gate via the retry's sender.
    tx.unbounded_send(msg(REQUEST_ID, 2, &[0.75])).unwrap();
    let item = next(&mut out).await;
    assert_eq!(
        output(&item, 0)["finish_reason"].as_str(),
        Some("stop"),
        "released, not filtered"
    );
    assert_eq!(events(&item, 0), jv!({"sustained": 0.75}));
    assert_ended(&mut out).await;
}

// ---------------------------------------------------------------------------

// event plane: manual round trip over the real transport
// ---------------------------------------------------------------------------

/// Local runtime with the ZMQ event plane (no NATS needed). `DYN_EVENT_PLANE`
/// is honoured by the runtime; leave it unset for the default.
async fn namespace() -> Namespace {
    let rt = DistributedRuntime::new(
        Runtime::from_current().unwrap(),
        DistributedConfig::process_local(),
    )
    .await
    .unwrap();
    rt.namespace("monitor-test").unwrap()
}

/// Round trip over the real event plane; a non-MonitorMessage payload is dropped.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a discoverable event plane (etcd/NATS or a ZMQ broker); run manually"]
async fn event_plane_round_trip() {
    let ns = namespace().await;
    let topic = "monitors".to_string();
    let mut feed = MonitorGate::subscribe_namespace(&ns, &topic).await.unwrap();
    let publisher = EventPublisher::for_namespace(&ns, topic.clone())
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await; // let the subscription settle
    publisher
        .publish(&msg(REQUEST_ID, 0, &[0.25]))
        .await
        .unwrap();
    publisher.publish(&"not a monitor message").await.unwrap();
    publisher
        .publish(&msg(REQUEST_ID, 1, &[0.75, 0.5]))
        .await
        .unwrap();
    let got = tokio::time::timeout(Duration::from_secs(2), async {
        vec![feed.next().await.unwrap(), feed.next().await.unwrap()]
    })
    .await
    .expect("two decoded messages");
    assert_eq!(
        got,
        vec![
            msg(REQUEST_ID, 0, &[0.25]),
            msg(REQUEST_ID, 1, &[0.75, 0.5])
        ]
    );
    assert!(feed.next().now_or_never().is_none(), "garbage was dropped");
}
