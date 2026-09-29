// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend side of the monitoring TOML (`DYN_MONITOR_CONFIG`, shared with the
//! coordinator): `GET /monitors` and the per-monitor `stop_threshold` policy.

use std::collections::{BTreeMap, HashMap};
use std::sync::LazyLock;

use crate::protocols::openai::chat_completions::{
    NvCreateChatCompletionResponse, NvCreateChatCompletionStreamResponse,
};
use anyhow::{Context as _, Result, ensure};
use axum::{Json, Router, http::Method, routing::get};
use dynamo_protocols::types::{ChatCompletionStreamResponseDelta, FinishReason, MonitorMeta};
use dynamo_runtime::config::environment_names::llm::monitor::DYN_MONITOR_CONFIG;
use serde::Deserialize;
use serde_json::{Value, json};

use super::RouteDoc;

/// Deployed monitor names and the values at which a stream is stopped.
#[derive(Debug, Default, PartialEq)]
pub struct MonitorPolicy {
    names: Vec<String>,
    stop: HashMap<String, f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    monitoring: Monitoring,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Monitoring {
    version: i64,
    #[serde(default)]
    #[allow(dead_code)]
    streams: Vec<String>,
    #[serde(default)]
    monitors: BTreeMap<String, Entry>,
    capture: Option<toml::Value>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    #[allow(dead_code)]
    return_when: String,
    event_threshold: f64,
    stop_threshold: Option<f64>,
    #[allow(dead_code)]
    config: Option<toml::Value>,
}

static POLICY: LazyLock<Result<Option<MonitorPolicy>>> = LazyLock::new(MonitorPolicy::from_env);

impl MonitorPolicy {
    /// Force-load the policy, surfacing a malformed file as an error so
    /// [`monitors_router`] can fail the frontend at startup.
    pub fn init() -> Result<Option<&'static Self>> {
        match &*POLICY {
            Ok(policy) => Ok(policy.as_ref()),
            Err(e) => Err(anyhow::anyhow!("{e:#}")).context("invalid DYN_MONITOR_CONFIG"),
        }
    }

    /// The process-wide policy; `None` when `DYN_MONITOR_CONFIG` is unset. Startup
    /// already failed on a malformed file, so a load error reads as absent here.
    pub fn global() -> Option<&'static Self> {
        POLICY.as_ref().ok().and_then(|p| p.as_ref())
    }

    fn from_env() -> Result<Option<Self>> {
        let Some(path) = std::env::var(DYN_MONITOR_CONFIG)
            .ok()
            .filter(|p| !p.trim().is_empty())
        else {
            return Ok(None);
        };
        let text = std::fs::read_to_string(&path).with_context(|| format!("reading {path}"))?;
        Self::parse(&text)
            .with_context(|| format!("invalid monitoring config {path}"))
            .map(Some)
    }

    pub fn parse(text: &str) -> Result<Self> {
        let monitoring = toml::from_str::<File>(text)?.monitoring;
        ensure!(
            monitoring.version == 1,
            "unsupported monitoring version {} (this server implements 1)",
            monitoring.version
        );
        ensure!(
            monitoring.capture.is_none(),
            "[monitoring.capture] is not supported"
        );
        // BTreeMap keys are already sorted.
        let names: Vec<String> = monitoring.monitors.keys().cloned().collect();
        let mut stop = HashMap::new();
        for (name, entry) in &monitoring.monitors {
            if let Some(threshold) = entry.stop_threshold {
                ensure!(
                    threshold.is_finite(),
                    "monitor {name:?}: stop_threshold must be a finite number"
                );
                ensure!(
                    threshold >= entry.event_threshold,
                    "monitor {name:?}: stop_threshold must be >= event_threshold"
                );
                stop.insert(name.clone(), threshold);
            }
        }
        Ok(Self { names, stop })
    }

    /// The `GET /monitors` document.
    pub fn document(&self) -> Value {
        json!({"version": 1, "monitors": self.names, "capture": {"enabled": false}})
    }

    /// First (monitor, value) at or above its `stop_threshold`.
    pub fn tripped(&self, meta: &MonitorMeta) -> Option<(&str, f64)> {
        let events = meta.monitor_events.as_ref()?;
        self.stop.iter().find_map(|(name, threshold)| {
            let value = *events.get(name)?;
            (value >= *threshold).then_some((name.as_str(), value))
        })
    }

    /// If any choice trips, rewrite the chunk into the terminal `content_filter`
    /// chunk (no content, events kept) and return what tripped.
    pub fn stop_chunk(
        &self,
        chunk: &mut NvCreateChatCompletionStreamResponse,
    ) -> Option<(String, f64)> {
        let trip = chunk
            .inner
            .choices
            .iter()
            .find_map(|c| c.delta.monitor.as_ref().and_then(|m| self.tripped(m)))
            .map(|(name, value)| (name.to_string(), value))?;
        for choice in &mut chunk.inner.choices {
            choice.delta = ChatCompletionStreamResponseDelta {
                role: choice.delta.role,
                monitor: choice.delta.monitor.take(),
                content: None,
                function_call: None,
                tool_calls: None,
                refusal: None,
                reasoning_content: None,
            };
            choice.finish_reason = Some(FinishReason::ContentFilter);
            choice.logprobs = None;
        }
        Some(trip)
    }

    /// Non-streaming: if the folded max trips, blank the message and finish with
    /// `content_filter`.
    pub fn stop_message(
        &self,
        response: &mut NvCreateChatCompletionResponse,
    ) -> Option<(String, f64)> {
        let trip = response
            .inner
            .choices
            .iter()
            .find_map(|c| c.message.monitor.as_ref().and_then(|m| self.tripped(m)))
            .map(|(name, value)| (name.to_string(), value))?;
        for choice in &mut response.inner.choices {
            choice.message.content = None;
            choice.message.tool_calls = None;
            choice.message.reasoning_content = None;
            choice.finish_reason = Some(FinishReason::ContentFilter);
            choice.logprobs = None;
        }
        Some(trip)
    }
}

/// `GET /monitors`, registered only when `DYN_MONITOR_CONFIG` is set. Forces the
/// policy to load, so a malformed file fails the frontend at startup.
pub fn monitors_router() -> Result<(Vec<RouteDoc>, Router)> {
    let Some(policy) = MonitorPolicy::init()? else {
        return Ok((vec![], Router::new()));
    };
    let doc = policy.document();
    Ok((
        vec![RouteDoc::new(Method::GET, "/monitors")],
        Router::new().route("/monitors", get(move || async move { Json(doc) })),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The deployment the coordinator tests use too: `harm` (stop at 0.9) and the
    /// always-on `sustained` (no stop threshold).
    const CONFIG: &str = include_str!("../../../tests/fixtures/monitoring.toml");

    fn meta(name: &str, value: f64) -> MonitorMeta {
        MonitorMeta {
            monitor_events: Some(HashMap::from([(name.to_string(), value)])),
            monitor_error: None,
        }
    }

    #[test]
    fn parses_names_and_stop_thresholds() {
        let policy = MonitorPolicy::parse(CONFIG).unwrap();
        assert_eq!(
            policy.document(),
            json!({"version": 1, "monitors": ["harm", "sustained"], "capture": {"enabled": false}})
        );
        assert_eq!(policy.tripped(&meta("harm", 0.9)), Some(("harm", 0.9)));
        assert_eq!(policy.tripped(&meta("harm", 0.89)), None);
        assert_eq!(
            policy.tripped(&meta("sustained", 1.0)),
            None,
            "no stop_threshold"
        );
    }

    /// Each case is the fixture with one edit.
    #[test]
    fn rejects_invalid_frontend_configs() {
        let err = |t: String| format!("{:#}", MonitorPolicy::parse(&t).unwrap_err());
        let edit = |from: &str, to: &str| {
            assert!(CONFIG.contains(from), "fixture lacks {from:?}");
            CONFIG.replace(from, to)
        };
        assert!(err(edit("version = 1", "version = 2")).contains("version"));
        assert!(
            err(format!("{CONFIG}\n[monitoring.capture]\nenabled = true\n")).contains("capture")
        );
        assert!(
            err(edit(
                "streams = [\"harm\"]",
                "streams = [\"harm\"]\nbogus = 1"
            ))
            .contains("unknown field")
        );
        assert!(
            err(edit(
                "return_when = \"always\"",
                "return_when = \"always\"\nbogus = 1"
            ))
            .contains("unknown field")
        );
        assert!(
            err(edit("stop_threshold = 0.9", "stop_threshold = 0.4"))
                .contains("stop_threshold must be >=")
        );
        assert!(err(edit("stop_threshold = 0.9", "stop_threshold = nan")).contains("finite"));
    }

    #[test]
    fn stop_chunk_blanks_content_and_keeps_events() {
        let policy = MonitorPolicy::parse(CONFIG).unwrap();
        let mut chunk: NvCreateChatCompletionStreamResponse = serde_json::from_value(json!({
            "id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{"index": 0, "delta": {"content": "bad", "monitor_events": {"harm": 0.95}}}]
        }))
        .unwrap();
        assert_eq!(
            policy.stop_chunk(&mut chunk),
            Some(("harm".to_string(), 0.95))
        );
        let wire = serde_json::to_value(&chunk).unwrap();
        let choice = &wire["choices"][0];
        assert_eq!(choice["finish_reason"], "content_filter");
        assert!(choice["delta"].get("content").is_none());
        assert_eq!(choice["delta"]["monitor_events"]["harm"], 0.95);
        assert_eq!(
            policy.stop_chunk(&mut chunk.clone()),
            Some(("harm".to_string(), 0.95))
        );
    }
}
