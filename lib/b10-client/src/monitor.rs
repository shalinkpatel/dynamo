// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Coordinator-side monitor gate. A probe sidecar publishes per-position score
//! rows ([`MonitorMessage`]) keyed by request id; the gate holds each chunk until
//! its rows arrive, evaluates the configured monitors, and sets `monitor_events`
//! on every output. Stop policy lives in the frontend.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::generation::{map_get_mut, set_map_entry};
use anyhow::{Context as _, Result, anyhow, bail, ensure};
use dynamo_runtime::component::Namespace;
use dynamo_runtime::config::environment_names::llm::monitor as env;
use dynamo_runtime::pipeline::{AsyncEngineContext, EngineStream, ResponseStream};
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::EventSubscriber;
use futures::{Stream, StreamExt};
use rmpv::Value;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, watch};

pub const MONITORING_VERSION: i64 = 1;

/// Score rows for one request: `rows[k]` scores absolute position `start + k`,
/// one f32 logit per stream in `streams` order. A replayed position keeps its first copy.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MonitorMessage {
    pub request_id: String,
    pub start: u64,
    pub rows: Vec<Vec<f32>>,
}

/// Inbound rows for every request on this gate; never leaves the coordinator.
pub type MonitorFeed = Pin<Box<dyn Stream<Item = MonitorMessage> + Send>>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Scope {
    Prompt,
    #[default]
    Output,
    Both,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ReturnWhen {
    Always,
    OnRequest,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Calibration {
    #[default]
    Sigmoid,
    Linear,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Repeat {
    #[default]
    Always,
    Once,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigFile {
    monitoring: Section,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Section {
    version: i64,
    /// The sidecar's stream (column) order; required with monitors.
    #[serde(default)]
    streams: Vec<String>,
    monitors: Option<BTreeMap<String, Entry>>,
    capture: Option<toml::Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    return_when: ReturnWhen,
    event_threshold: f64,
    /// Frontend stop policy; the coordinator only validates it.
    stop_threshold: Option<f64>,
    #[serde(default)]
    config: Engine,
}

/// The monitor's opaque `config` table.
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Engine {
    probe: Option<String>,
    probes: Option<BTreeMap<String, f64>>,
    #[serde(default)]
    on: Scope,
    temperature: Option<f64>,
    #[serde(default)]
    bias: f64,
    #[serde(default)]
    calibration: Calibration,
    #[serde(default)]
    repeat: Repeat,
}

/// A deployed monitor bound to score-row columns.
#[derive(Clone, Debug)]
struct Monitor {
    name: String,
    return_when: ReturnWhen,
    event_threshold: f64,
    /// (column, weight); the monitor's logit is their weighted sum.
    columns: Vec<(usize, f64)>,
    on: Scope,
    temperature: f64,
    bias: f64,
    calibration: Calibration,
    repeat: Repeat,
}

impl Monitor {
    /// `logit / temperature + bias`, squashed by sigmoid or emitted as is.
    fn value(&self, row: &[f32]) -> f64 {
        let logit: f64 = self
            .columns
            .iter()
            .map(|&(c, w)| w * f64::from(row[c]))
            .sum();
        let z = logit / self.temperature + self.bias;
        match self.calibration {
            Calibration::Linear => z,
            // Two branches avoid overflow for extreme logits.
            Calibration::Sigmoid if z >= 0.0 => 1.0 / (1.0 + (-z).exp()),
            Calibration::Sigmoid => z.exp() / (1.0 + z.exp()),
        }
    }
}

/// The parsed `DYN_MONITOR_CONFIG` file: what this coordinator deploys.
#[derive(Clone, Debug, Default)]
pub struct MonitoringConfig {
    monitors: Vec<Monitor>,
    /// Floats per score row: one per sidecar stream.
    width: usize,
}

impl MonitoringConfig {
    /// Port of `load_monitoring_config` + `bind_monitors`. Unknown keys, missing
    /// fields and probes the sidecar does not serve fail here.
    pub fn parse(text: &str) -> Result<Self> {
        let Section {
            version,
            streams,
            monitors,
            capture,
        } = toml::from_str::<ConfigFile>(text)?.monitoring;
        ensure!(
            capture.is_none(),
            "[monitoring.capture] is not supported; this server deploys monitors only"
        );
        ensure!(
            version == MONITORING_VERSION,
            "unsupported monitoring version {version} (this server implements {MONITORING_VERSION})"
        );
        let Some(monitors) = monitors else {
            return Ok(Self::default());
        };
        ensure!(
            !monitors.is_empty(),
            "[monitoring.monitors] requires at least one monitor"
        );
        ensure!(
            !streams.is_empty(),
            "[monitoring] streams (the sidecar's column order) is required with monitors"
        );
        ensure!(
            streams.iter().collect::<HashSet<_>>().len() == streams.len(),
            "probe sidecar stream names repeat: {streams:?}"
        );
        let monitors = monitors
            .into_iter()
            .map(|(name, entry)| bind(name, entry, &streams))
            .collect::<Result<_>>()?;
        Ok(Self {
            monitors,
            width: streams.len(),
        })
    }

    /// Every deployed monitor, always-on and on-request.
    pub fn names(&self) -> Vec<String> {
        self.monitors.iter().map(|m| m.name.clone()).collect()
    }

    /// Port of `select_monitors`: `None` runs the always-on monitors (unmonitored
    /// when there are none); a list, even empty, also runs the listed ones.
    fn select(&self, requested: Option<&[String]>) -> Result<Option<Vec<Monitor>>> {
        let unknown: Vec<&String> = requested
            .unwrap_or_default()
            .iter()
            .filter(|name| !self.monitors.iter().any(|m| &m.name == *name))
            .collect();
        ensure!(
            unknown.is_empty(),
            "unknown monitors {unknown:?}; deployed monitors are {:?}",
            self.names()
        );
        let selected: Vec<Monitor> = self
            .monitors
            .iter()
            .filter(|m| {
                m.return_when == ReturnWhen::Always
                    || requested.is_some_and(|names| names.contains(&m.name))
            })
            .cloned()
            .collect();
        Ok((requested.is_some() || !selected.is_empty()).then_some(selected))
    }
}

fn bind(name: String, entry: Entry, streams: &[String]) -> Result<Monitor> {
    let Entry {
        return_when,
        event_threshold,
        stop_threshold,
        config,
    } = entry;
    ensure!(
        !name.trim().is_empty(),
        "monitor name must be a non-empty string"
    );
    let temperature = config.temperature.unwrap_or(1.0);
    for (field, value) in [
        ("event_threshold", Some(event_threshold)),
        ("stop_threshold", stop_threshold),
        ("temperature", Some(temperature)),
        ("bias", Some(config.bias)),
    ] {
        ensure!(
            value.is_none_or(f64::is_finite),
            "monitor {name:?}: {field} must be a finite number"
        );
    }
    ensure!(
        temperature > 0.0,
        "monitor {name:?}: temperature must be positive"
    );
    if let Some(stop) = stop_threshold {
        ensure!(
            stop >= event_threshold,
            "monitor {name:?}: stop_threshold ({stop}) must be >= event_threshold ({event_threshold})"
        );
        // repeat = once fires once at event_threshold, so a higher stop is unreachable.
        ensure!(
            config.repeat != Repeat::Once || stop == event_threshold,
            "monitor {name:?}: repeat = once fires at event_threshold, so stop_threshold must equal it"
        );
    }
    let probes: Vec<(String, f64)> = match (config.probe, config.probes) {
        (Some(probe), None) => vec![(probe, 1.0)],
        (None, Some(probes)) => probes.into_iter().collect(),
        _ => bail!("monitor {name:?}: set exactly one of probe, probes"),
    };
    ensure!(
        !probes.is_empty(),
        "monitor {name:?}: probes must be non-empty (name = weight) pairs"
    );
    let columns = probes
        .iter()
        .map(|(probe, weight)| {
            ensure!(
                weight.is_finite(),
                "monitor {name:?}: probes.{probe} must be a finite number"
            );
            let column = streams.iter().position(|s| s == probe).ok_or_else(|| {
                anyhow!("monitor {name:?}: unknown probe {probe:?}; the sidecar serves {streams:?}")
            })?;
            Ok((column, *weight))
        })
        .collect::<Result<_>>()?;
    Ok(Monitor {
        name,
        return_when,
        event_threshold,
        columns,
        on: config.on,
        temperature,
        bias: config.bias,
        calibration: config.calibration,
        repeat: config.repeat,
    })
}

/// Calibrate `rows` and merge the largest qualifying value per monitor into
/// `events`; `repeat = once` fires once per choice.
fn evaluate(
    monitors: &[Monitor],
    fired: &mut HashSet<String>,
    rows: &[&[f32]],
    scope: Scope,
    events: &mut BTreeMap<String, f64>,
) {
    for m in monitors {
        if (m.on != scope && m.on != Scope::Both)
            || (m.repeat == Repeat::Once && fired.contains(&m.name))
        {
            continue;
        }
        for row in rows {
            let value = m.value(row);
            if value >= m.event_threshold {
                // Values may be negative under linear calibration.
                let held = events.entry(m.name.clone()).or_insert(value);
                *held = held.max(value);
                // fired only gates repeat = once; skip the insert for always.
                if m.repeat == Repeat::Once {
                    fired.insert(m.name.clone());
                    break;
                }
            }
        }
    }
}

/// One request's rows by absolute position.
// ponytail: kept for the request's lifetime; prune evaluated positions if memory matters.
#[derive(Default)]
struct Rows {
    rows: BTreeMap<u64, Vec<f32>>,
    /// A malformed row: coverage is impossible.
    error: Option<String>,
}

impl Rows {
    fn file(&mut self, m: MonitorMessage, width: usize) {
        for (i, row) in m.rows.into_iter().enumerate() {
            let position = m.start.saturating_add(i as u64);
            if row.len() != width || !row.iter().all(|v| v.is_finite()) {
                self.error.get_or_insert_with(|| {
                    format!("bad score row at position {position}: expected {width} finite scores")
                });
                return;
            }
            self.rows.entry(position).or_insert(row);
        }
    }

    fn covers(&self, start: u64, end: u64) -> bool {
        self.rows.range(start..end).count() as u64 >= end - start
    }

    fn get(&self, start: u64, end: u64) -> Vec<&[f32]> {
        self.rows
            .range(start..end)
            .map(|(_, r)| r.as_slice())
            .collect()
    }
}

/// Messages buffered per request between the shared reader and its task.
const REQUEST_QUEUE: usize = 256;

type Slots = Arc<Mutex<HashMap<String, mpsc::Sender<MonitorMessage>>>>;

pub struct MonitorGate {
    config: MonitoringConfig,
    hold: Duration,
    slots: Slots,
    handle: tokio::runtime::Handle,
    /// Shared transport reader: decode + demux only. Aborted on drop.
    reader: tokio::task::JoinHandle<()>,
}

impl MonitorGate {
    /// Subscribe once to `topic` and start the shared reader on `handle`. `hold`
    /// bounds how long a chunk waits for its rows.
    pub fn new(
        config: MonitoringConfig,
        hold: Duration,
        mut feed: MonitorFeed,
        handle: &tokio::runtime::Handle,
    ) -> Arc<Self> {
        let slots: Slots = Arc::default();
        let reader = handle.spawn({
            let slots = Arc::clone(&slots);
            async move {
                let mut dropped = 0u64;
                while let Some(m) = feed.next().await {
                    // Cooperative yield so a feed that is never Pending cannot pin a worker.
                    tokio::task::consume_budget().await;
                    // Unknown ids: finished, or another coordinator's request.
                    let Some(tx) = slots.lock().unwrap().get(&m.request_id).cloned() else {
                        continue;
                    };
                    if tx.try_send(m).is_err() {
                        dropped += 1;
                        if dropped.is_power_of_two() {
                            tracing::warn!(dropped, "monitor request queue full");
                        }
                    }
                }
            }
        });
        Arc::new(Self {
            config,
            hold,
            slots,
            handle: handle.clone(),
            reader,
        })
    }

    /// Subscribe to the namespace event plane: decode and drop undecodable events.
    async fn subscribe_namespace(ns: &Namespace, topic: &str) -> Result<MonitorFeed> {
        let sub = EventSubscriber::for_namespace(ns, topic.to_string())
            .await?
            .typed::<MonitorMessage>();
        let feed = futures::stream::unfold(sub, |mut sub| async move {
            loop {
                match sub.next().await? {
                    Ok((_, m)) => return Some((m, sub)),
                    Err(e) => tracing::warn!(%e, "dropping undecodable monitor event"),
                }
            }
        })
        .fuse();
        Ok(Box::pin(feed) as MonitorFeed)
    }

    /// `Ok(None)` when `DYN_MONITOR_CONFIG` is unset or deploys no monitors;
    /// an invalid config or failed subscription is an error (fail closed).
    pub async fn from_env(ns: Namespace) -> Result<Option<Arc<Self>>> {
        let Some(path) = std::env::var(env::DYN_MONITOR_CONFIG)
            .ok()
            .filter(|p| !p.trim().is_empty())
        else {
            return Ok(None);
        };
        let config = std::fs::read_to_string(&path)
            .map_err(anyhow::Error::from)
            .and_then(|text| MonitoringConfig::parse(&text))
            .with_context(|| format!("invalid monitoring config {path}"))?;
        if config.monitors.is_empty() {
            tracing::info!(%path, "monitoring config deploys no monitors; monitor gate off");
            return Ok(None);
        }
        let hold = Duration::from_millis(
            std::env::var(env::DYN_MONITOR_HOLD_TIMEOUT_MS)
                .ok()
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(2000),
        );
        let topic = std::env::var(env::DYN_MONITOR_TOPIC)
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "monitors".to_string());
        let handle = ns.drt().runtime().primary();
        let namespace = ns.name();
        let monitors = config.names();
        let feed = Self::subscribe_namespace(&ns, &topic)
            .await
            .with_context(|| format!("monitor gate: event-plane subscribe failed on {topic}"))?;
        let gate = Self::new(config, hold, feed, &handle);
        tracing::info!(?monitors, ?hold, %namespace, topic, "monitor gate enabled");
        Ok(Some(gate))
    }

    /// Select the request's monitors and register it before routing.
    /// `Ok(None)`: unmonitored, the stream is not wrapped and never waits.
    pub fn subscribe(
        &self,
        request_id: &str,
        requested: Option<&[String]>,
        prompt_len: u64,
    ) -> Result<Option<Mailbox>> {
        let Some(monitors) = self.config.select(requested)? else {
            return Ok(None);
        };
        let (tx, mut rx) = mpsc::channel::<MonitorMessage>(REQUEST_QUEUE);
        let (filed, rows) = watch::channel(Rows::default());
        self.slots
            .lock()
            .unwrap()
            .insert(request_id.to_string(), tx.clone());
        // Per-request task: files this request's rows; ends when Mailbox::drop
        // unregisters the sender.
        let width = self.config.width;
        self.handle.spawn(async move {
            while let Some(m) = rx.recv().await {
                filed.send_modify(|rows| rows.file(m, width));
            }
        });
        Ok(Some(Mailbox {
            rows,
            monitors,
            prompt_len,
            hold: self.hold,
            key: request_id.to_string(),
            slots: Arc::clone(&self.slots),
            tx,
        }))
    }

    /// Hold each chunk until its rows arrive, then set `monitor_events` on every
    /// output. Hold timeout or bad row: `content_filter` + `monitor_error`.
    pub fn wrap(
        inner: EngineStream<Annotated<Value>>,
        mailbox: Mailbox,
        ctx: Arc<dyn AsyncEngineContext>,
    ) -> EngineStream<Annotated<Value>> {
        let stream_ctx = Arc::clone(&ctx);
        let output = async_stream::stream! {
            let mut inner = inner;
            let mut rows = mailbox.rows.clone();
            let (monitors, prompt_len, hold) = (&mailbox.monitors, mailbox.prompt_len, mailbox.hold);
            let prompt_rows = monitors.iter().any(|m| m.on != Scope::Output);
            let mut choices: HashMap<u64, Choice> = HashMap::new();
            while let Some(mut item) = inner.next().await {
                let Some(outputs) = outputs_mut(&mut item) else {
                    yield item; // worker errors and output-less chunks pass untouched
                    continue;
                };
                // Token k's row is produced by the forward that samples k+1, so a chunk
                // waits for rows covering every output token before its newest one and
                // evaluates the newly covered positions.
                let mut segments = Vec::with_capacity(outputs.len());
                let mut need = Vec::new();
                for (i, out) in outputs.iter().enumerate() {
                    let index = out["index"].as_u64().unwrap_or(i as u64);
                    let tokens = out["token_ids_diff"].as_array().map_or(0, Vec::len) as u64;
                    let choice = choices.entry(index).or_insert_with(|| Choice::new(prompt_len));
                    let first = !std::mem::replace(&mut choice.started, true);
                    choice.tokens += tokens;
                    let end = (prompt_len + choice.tokens).saturating_sub(1).max(choice.scored);
                    if first && prompt_rows {
                        need.push((0, prompt_len));
                    }
                    need.push((choice.scored, end));
                    segments.push((index, choice.scored, end, first));
                    choice.scored = end;
                }
                let ready = |r: &Rows| r.error.is_some() || need.iter().all(|&(a, b)| r.covers(a, b));
                let error = match tokio::time::timeout(hold, rows.wait_for(ready)).await {
                    Ok(Ok(r)) => r.error.clone(),
                    Ok(Err(_)) => Some("monitor score feed closed".to_string()),
                    Err(_) => Some(format!("monitor scores did not arrive within {hold:?}")),
                };
                if let Some(reason) = error {
                    tracing::warn!(request_id = %ctx.id(), %reason, "monitor hold failed; stopping");
                    for out in outputs.iter_mut() {
                        if let Value::Map(out) = out {
                            set_map_entry(out, "monitor_events", Value::Map(vec![]));
                            set_map_entry(out, "monitor_error", Value::from(reason.as_str()));
                        }
                    }
                    ctx.stop_generating();
                    yield Annotated::from_data(content_filter_chunk(item));
                    return;
                }
                {
                    let r = rows.borrow();
                    for (out, &(index, start, end, first)) in outputs.iter_mut().zip(&segments) {
                        let fired = &mut choices.get_mut(&index).expect("filed above").fired;
                        let mut events = BTreeMap::new();
                        if first && prompt_rows {
                            evaluate(monitors, fired, &r.get(0, prompt_len), Scope::Prompt, &mut events);
                        }
                        evaluate(monitors, fired, &r.get(start, end), Scope::Output, &mut events);
                        if let Value::Map(out) = out {
                            let map = events
                                .iter()
                                .map(|(k, v)| (Value::from(k.as_str()), Value::from(*v)));
                            set_map_entry(out, "monitor_events", Value::Map(map.collect()));
                        }
                    }
                }
                yield item;
            }
        };
        ResponseStream::new(Box::pin(output), stream_ctx)
    }
}

impl Drop for MonitorGate {
    fn drop(&mut self) {
        self.reader.abort();
    }
}

/// Per-choice monitor state, keyed by output index.
struct Choice {
    /// Output tokens seen so far.
    tokens: u64,
    /// Absolute position evaluated through; starts at `prompt_len`.
    scored: u64,
    started: bool,
    /// `repeat = once` monitors that already fired.
    fired: HashSet<String>,
}

impl Choice {
    fn new(prompt_len: u64) -> Self {
        Self {
            tokens: 0,
            scored: prompt_len,
            started: false,
            fired: HashSet::new(),
        }
    }
}

/// One request's registration; dropping it unregisters the request id unless a
/// retry already replaced the slot with a new sender.
pub struct Mailbox {
    rows: watch::Receiver<Rows>,
    monitors: Vec<Monitor>,
    prompt_len: u64,
    hold: Duration,
    key: String,
    slots: Slots,
    /// This registration's sender; drop removes the slot only if it is still ours.
    tx: mpsc::Sender<MonitorMessage>,
}

impl Drop for Mailbox {
    fn drop(&mut self) {
        let mut slots = self.slots.lock().unwrap();
        // A Migration retry may re-register this id with a new sender before the
        // stale mailbox drops; only remove the slot when it is still ours.
        if slots
            .get(&self.key)
            .is_some_and(|tx| tx.same_channel(&self.tx))
        {
            slots.remove(&self.key);
        }
    }
}

fn outputs_mut(item: &mut Annotated<Value>) -> Option<&mut Vec<Value>> {
    let Some(Value::Map(map)) = item.data.as_mut() else {
        return None;
    };
    match map_get_mut(map, "outputs") {
        Some(Value::Array(outputs)) => Some(outputs),
        _ => None,
    }
}

/// Tripping chunk with `finished` set and every output rewritten to no tokens +
/// `content_filter`; siblings (`usage`, `request_id`, `monitor_events`) preserved.
fn content_filter_chunk(mut from: Annotated<Value>) -> Value {
    let mut base = match from.data.take() {
        Some(Value::Map(m)) => m,
        _ => Vec::new(),
    };
    let outputs: Vec<Value> = match base.iter_mut().find(|(k, _)| k.as_str() == Some("outputs")) {
        Some((_, Value::Array(outs))) if !outs.is_empty() => {
            for o in outs.iter_mut() {
                let mut m = match std::mem::replace(o, Value::Nil) {
                    Value::Map(m) => m,
                    _ => Vec::new(),
                };
                set_map_entry(&mut m, "token_ids_diff", Value::Array(vec![]));
                set_map_entry(&mut m, "finish_reason", Value::from("content_filter"));
                m.retain(|(k, _)| k.as_str() != Some("text") && k.as_str() != Some("token_ids"));
                *o = Value::Map(m);
            }
            std::mem::take(outs)
        }
        _ => vec![Value::Map(vec![
            (Value::from("index"), Value::from(0u64)),
            (Value::from("token_ids_diff"), Value::Array(vec![])),
            (Value::from("finish_reason"), Value::from("content_filter")),
        ])],
    };
    set_map_entry(&mut base, "outputs", Value::Array(outputs));
    set_map_entry(&mut base, "finished", Value::from(true));
    Value::Map(base)
}

#[cfg(test)]
pub(crate) mod tests;
