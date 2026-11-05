// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{
    DeniedGenerationRequest, GenerationAdmission, GenerationObservationEnd, GenerationObserver,
    GenerationObserverFactory, GenerationRequest, RequestContext,
};
use anyhow::{Result, ensure};
use dynamo_runtime::protocols::annotated::Annotated;
use llm_runtime_metrics::{
    __private::{RequestMetricsOptions, RequestTraceOptions},
    AtomicCounter, AtomicHistogram, PrometheusSample, RequestFeatures, RequestMetricsFactory,
    RequestMetricsRequest, RoutedWorkers, ServiceTier, StreamChunkMetrics, flatten_snapshot,
};
use rmpv::Value;
use serde::Deserialize;
use std::{
    collections::{BTreeMap, HashMap},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};

/// Bucket boundaries and formatted VM labels come from the existing Python exporter.
#[derive(Deserialize)]
pub struct HistogramConfig {
    pub bounds: Vec<f64>,
    pub ranges: Vec<String>,
    pub labels: Vec<String>,
}

#[derive(Deserialize)]
pub struct MetricsConfig {
    pub labels: HashMap<String, String>,
    #[serde(default)]
    pub factory_labels: Option<HashMap<String, String>>,
    pub histograms: BTreeMap<String, HistogramConfig>,
    pub factory: RequestMetricsOptions,
    #[serde(default)]
    pub session_metrics_enabled: bool,
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct SeriesKey {
    billing_identity: Option<(String, String)>,
    billing_tier: String,
    effective_tier: String,
    has_session: Option<bool>,
}

pub struct GenerationMetrics {
    config: MetricsConfig,
    factory: RequestMetricsFactory,
    tiers: Mutex<BTreeMap<SeriesKey, Arc<Series>>>,
    session_metrics_enabled: AtomicBool,
}

struct MetricsObserverFactory {
    metrics: Arc<GenerationMetrics>,
    block_size: u32,
    request_identity: bool,
}

impl GenerationObserverFactory for MetricsObserverFactory {
    fn begin(
        &self,
        context: &RequestContext,
        request: &GenerationRequest,
    ) -> Result<Box<dyn GenerationObserver>> {
        let recorder = self.metrics.begin_with_identity(
            context,
            request,
            self.block_size,
            self.request_identity,
        )?;
        Ok(Box::new(recorder))
    }
}

struct Series {
    created_at: f64,
    e2e_tier: String,
    labels: HashMap<String, String>,
    histograms: BTreeMap<String, AtomicHistogram>,
    input: AtomicCounter,
    cached: AtomicCounter,
    uncached: AtomicCounter,
    output: AtomicCounter,
    missed: BTreeMap<&'static str, AtomicCounter>,
    unpredicted: AtomicCounter,
    requests: AtomicCounter,
}

const REASONS: [&str; 6] = [
    "unrouted",
    "no_engine_reading",
    "held_not_reused",
    "stale_index",
    "routed_away",
    "cold",
];

impl GenerationMetrics {
    pub fn new(config: MetricsConfig) -> Result<Self> {
        for name in [
            "llm_time_to_first_token_seconds",
            "llm_time_per_output_token_seconds",
            "llm_inter_token_latency_seconds",
            "llm_request_time_per_output_token_seconds",
            "llm_processor_input_tokens_histogram",
            "llm_processor_output_tokens_histogram",
            "llm_request_tps",
            "llm_e2e_latency",
        ] {
            ensure!(
                config.histograms.contains_key(name),
                "missing histogram {name}"
            );
        }
        for histogram in config.histograms.values() {
            for label in &histogram.labels {
                ensure!(
                    config.labels.contains_key(label)
                        || matches!(label.as_str(), "service_tier" | "has_session"),
                    "unknown histogram label {label}"
                );
            }
            ensure!(
                histogram.ranges.len() == histogram.bounds.len() + 1,
                "histogram ranges must include overflow"
            );
            let mut bounds = histogram.bounds.clone();
            bounds.push(f64::INFINITY);
            AtomicHistogram::new(bounds)?;
        }
        let factory = RequestMetricsFactory::new(config.factory.clone())?;
        Ok(Self {
            session_metrics_enabled: AtomicBool::new(config.session_metrics_enabled),
            config,
            factory,
            tiers: Mutex::new(BTreeMap::new()),
        })
    }

    pub fn observer_factory(
        self: &Arc<Self>,
        block_size: u32,
    ) -> Arc<dyn GenerationObserverFactory> {
        Arc::new(MetricsObserverFactory {
            metrics: self.clone(),
            block_size,
            request_identity: false,
        })
    }

    /// Sending-side legacy series use the request's billing identity.
    pub fn sending_observer_factory(
        self: &Arc<Self>,
        block_size: u32,
    ) -> Arc<dyn GenerationObserverFactory> {
        Arc::new(MetricsObserverFactory {
            metrics: self.clone(),
            block_size,
            request_identity: true,
        })
    }

    pub fn begin(
        &self,
        context: &RequestContext,
        request: &GenerationRequest,
        block_size: u32,
    ) -> Result<GenerationRecorder> {
        self.begin_with_identity(context, request, block_size, false)
    }

    fn begin_with_identity(
        &self,
        context: &RequestContext,
        request: &GenerationRequest,
        block_size: u32,
        request_identity: bool,
    ) -> Result<GenerationRecorder> {
        let started = Instant::now();
        let payload = &request.primary_worker_request;
        let tier = context
            .metadata()
            .get("x-baseten-service-tier")
            .map(String::as_str)
            .unwrap_or("");
        let session_id = context
            .metadata()
            .get("dynamo.llm.session_affinity")
            .map(String::as_str)
            .or_else(|| payload["user"].as_str())
            .filter(|s| !s.is_empty());
        let has_session = self
            .session_metrics_enabled
            .load(Ordering::Relaxed)
            .then_some(session_id.is_some());
        let e2e_tier = payload["service_tier"]
            .as_str()
            .filter(|s| !s.is_empty())
            .unwrap_or(tier);
        let billing_identity = request_identity
            .then(|| {
                let mut parts = context.id().split("--");
                let namespace = parts.next()?;
                parts.next()?;
                let version = parts.next()?;
                let normalize = |value: &str| {
                    if value.eq_ignore_ascii_case("none") {
                        String::new()
                    } else {
                        value.to_owned()
                    }
                };
                Some((normalize(namespace), normalize(version)))
            })
            .flatten();
        let key = SeriesKey {
            billing_identity,
            billing_tier: tier.to_owned(),
            effective_tier: e2e_tier.to_owned(),
            has_session,
        };
        let series = {
            let mut tiers = self.tiers.lock().unwrap();
            if let Some(series) = tiers.get(&key) {
                series.clone()
            } else {
                let mut labels = self.config.labels.clone();
                if let Some((namespace, version)) = &key.billing_identity {
                    labels.insert("exported_namespace".into(), namespace.clone());
                    labels.insert("model_version_id".into(), version.clone());
                }
                labels.insert("service_tier".into(), tier.into());
                if let Some(has_session) = has_session {
                    labels.insert("has_session".into(), has_session.to_string());
                }
                let histograms = self
                    .config
                    .histograms
                    .iter()
                    .map(|(name, config)| {
                        let mut bounds = config.bounds.clone();
                        bounds.push(f64::INFINITY);
                        Ok((name.clone(), AtomicHistogram::new(bounds)?))
                    })
                    .collect::<Result<_>>()?;
                let series = Arc::new(Series {
                    created_at: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs_f64(),
                    e2e_tier: e2e_tier.to_owned(),
                    labels,
                    histograms,
                    input: AtomicCounter::default(),
                    cached: AtomicCounter::default(),
                    uncached: AtomicCounter::default(),
                    output: AtomicCounter::default(),
                    missed: REASONS
                        .into_iter()
                        .map(|name| (name, AtomicCounter::default()))
                        .collect(),
                    unpredicted: AtomicCounter::default(),
                    requests: AtomicCounter::default(),
                });
                tiers.insert(key, series.clone());
                series
            }
        };
        series.requests.inc(1);
        let tracker = self.factory.new_request_labeled(
            self.factory
                .request_log_enabled()
                .then_some(request.routing_request.tokens.as_slice()),
            request.routing_request.tokens.len(),
            RequestFeatures::from(payload["b10_metrics_features"].as_u64().unwrap_or(0) as u8),
            session_id,
            context.id().split("--").nth(1),
            payload["service_tier"]
                .as_str()
                .and_then(|s| s.parse::<ServiceTier>().ok()),
            None,
        )?;
        let histogram = |name: &str| {
            series
                .histograms
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("missing histogram {name}"))
        };
        let chunks = StreamChunkMetrics::new(
            histogram("llm_time_to_first_token_seconds")?,
            histogram("llm_time_per_output_token_seconds")?,
            histogram("llm_inter_token_latency_seconds")?,
            &series.output,
        );
        Ok(GenerationRecorder {
            series,
            tracker,
            chunks,
            started,
            first: None,
            last: None,
            first_tokens: 0,
            output: 0,
            input: request.routing_request.tokens.len() as u64 + 1,
            cached: None,
            full_hit: None,
            input_recorded: false,
            cached_recorded: false,
            kv_recorded: false,
            admission: None,
            block_size: u64::from(block_size),
            finalized: false,
            failed: false,
            successful_terminal: false,
            tokens: Vec::new(),
        })
    }

    pub fn configure(
        &self,
        trace: RequestTraceOptions,
        session_metrics_enabled: bool,
    ) -> Result<()> {
        self.session_metrics_enabled
            .store(session_metrics_enabled, Ordering::Relaxed);
        self.factory.configure_request_log(&trace)
    }

    pub fn samples(&self) -> Vec<PrometheusSample> {
        let factory_labels = self
            .config
            .factory_labels
            .as_ref()
            .unwrap_or(&self.config.labels);
        let mut out = flatten_snapshot(self.factory.snapshot(), Some(factory_labels));
        let tiers: Vec<_> = self.tiers.lock().unwrap().values().cloned().collect();
        for series in tiers {
            let mut sample =
                |name: String, value: f64, labels: HashMap<String, String>, created: bool| {
                    if created {
                        out.push(PrometheusSample {
                            name: format!("{}_created", name.trim_end_matches("_total")),
                            value: series.created_at,
                            labels: labels.clone(),
                            metric_type: "gauge".into(),
                        });
                    }
                    out.push(PrometheusSample {
                        name,
                        value,
                        labels,
                        metric_type: "counter".into(),
                    });
                };
            for (name, counter) in [
                ("llm_input_tokens_total", &series.input),
                ("llm_cached_input_tokens_total", &series.cached),
                ("llm_uncached_input_tokens_total", &series.uncached),
                ("llm_output_tokens_total", &series.output),
            ] {
                let mut labels = series.labels.clone();
                labels.remove("has_session");
                sample(name.into(), counter.get() as f64, labels, true);
            }
            if series.labels.contains_key("has_session") {
                sample(
                    "llm_session_requests_total".into(),
                    series.requests.get() as f64,
                    series.labels.clone(),
                    true,
                );
            }
            let reduced: HashMap<_, _> = series
                .labels
                .iter()
                .filter(|(key, _)| matches!(key.as_str(), "model_name" | "service_tier"))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            sample(
                "llm_kv_unpredicted_cached_input_tokens_total".into(),
                series.unpredicted.get() as f64,
                reduced.clone(),
                true,
            );
            for (reason, count) in &series.missed {
                let mut labels = reduced.clone();
                labels.insert("reason".into(), (*reason).into());
                sample(
                    "llm_kv_missed_input_tokens_total".into(),
                    count.get() as f64,
                    labels,
                    true,
                );
            }
            for (name, histogram) in &series.histograms {
                let (buckets, count, sum) = histogram.snapshot();
                if count == 0 {
                    continue;
                }
                let config = &self.config.histograms[name];
                let mut labels: HashMap<_, _> = config
                    .labels
                    .iter()
                    .map(|key| (key.clone(), series.labels[key].clone()))
                    .collect();
                if name == "llm_e2e_latency" {
                    labels.insert("service_tier".into(), series.e2e_tier.clone());
                }
                sample(
                    format!("{name}_count_total"),
                    count as f64,
                    labels.clone(),
                    false,
                );
                sample(format!("{name}_sum_total"), sum, labels.clone(), false);
                for (range, count) in config.ranges.iter().zip(buckets) {
                    if count > 0 {
                        let mut labels = labels.clone();
                        labels.insert("vmrange".into(), range.clone());
                        sample(format!("{name}_bucket_total"), count as f64, labels, false);
                    }
                }
            }
        }
        // Model-only histograms share labels across service tiers.
        let mut merged: BTreeMap<(String, BTreeMap<String, String>), PrometheusSample> =
            BTreeMap::new();
        for sample in out {
            let key = (
                sample.name.clone(),
                sample.labels.clone().into_iter().collect(),
            );
            merged
                .entry(key)
                .and_modify(|old| {
                    if sample.name.ends_with("_created") {
                        old.value = old.value.min(sample.value);
                    } else {
                        old.value += sample.value;
                    }
                })
                .or_insert(sample);
        }
        merged.into_values().collect()
    }
}

pub struct GenerationRecorder {
    series: Arc<Series>,
    tracker: RequestMetricsRequest,
    chunks: StreamChunkMetrics,
    started: Instant,
    first: Option<Instant>,
    last: Option<Instant>,
    first_tokens: u64,
    output: u64,
    input: u64,
    cached: Option<u64>,
    full_hit: Option<u64>,
    input_recorded: bool,
    cached_recorded: bool,
    kv_recorded: bool,
    admission: Option<GenerationAdmission>,
    block_size: u64,
    finalized: bool,
    failed: bool,
    successful_terminal: bool,
    tokens: Vec<u32>,
}

impl GenerationRecorder {
    pub fn admission(&mut self, admission: GenerationAdmission) {
        self.tracker.set_routed_workers(RoutedWorkers {
            prefill_worker_id: Some(admission.prefill_worker_id),
            prefill_dp_rank: Some(admission.prefill_dp_rank),
            decode_worker_id: admission.decode_worker_id,
            decode_dp_rank: admission.decode_dp_rank,
        });
        self.admission = Some(admission);
    }

    fn observe(&self, name: &str, value: f64) {
        if let Some(histogram) = self.series.histograms.get(name) {
            histogram.observe(value);
        }
    }

    fn record_input(&mut self) {
        if !self.input_recorded {
            self.series.input.inc(self.input);
            self.observe("llm_processor_input_tokens_histogram", self.input as f64);
            self.input_recorded = true;
        }
    }

    pub fn usage(&mut self, usage: crate::protocol::TokenUsageV1) {
        self.cache(usage.cached_tokens);
        if let Some(total) = usage.output_tokens {
            let delta = total.saturating_sub(self.output);
            self.output += delta;
            self.series.output.inc(delta);
            self.tracker
                .reconcile_output_usage(self.output as usize, self.cached);
        }
    }

    fn cache(&mut self, cached: Option<u64>) {
        if self.cached.unwrap_or(0) == 0 && cached.is_some() {
            self.cached = cached;
            self.tracker.record_tokens(&[], self.cached, true, None);
        }
        if !self.cached_recorded
            && let Some(cached) = cached.filter(|v| *v > 0)
        {
            self.series.cached.inc(cached);
            self.cached_recorded = true;
        }
    }

    pub fn chunk(&mut self, data: &Value) {
        if self.finalized {
            return;
        }
        self.record_input();
        self.cache(data["cached_tokens"].as_u64().or_else(|| {
            data["completion_usage"]["prompt_tokens_details"]["cached_tokens"].as_u64()
        }));
        let kv = &data["kv_cache_metrics"];
        if kv.is_map() && !self.kv_recorded {
            self.kv_recorded = true;
            self.full_hit = kv["num_full_kv_hit_tokens"].as_u64();
            let reused = kv["num_reused_blocks"].as_u64().unwrap_or(0);
            self.observe(
                "llm_kv_cache_hit_rate",
                kv["kv_cache_hit_rate"].as_f64().unwrap_or(0.0),
            );
            if self.series.labels.contains_key("has_session") {
                self.observe(
                    "llm_session_kv_cache_hit_rate",
                    kv["kv_cache_hit_rate"].as_f64().unwrap_or(0.0),
                );
            }
            self.observe(
                "llm_kv_cache_missed_blocks",
                kv["num_missed_blocks"].as_u64().unwrap_or(0) as f64,
            );
            self.observe("llm_kv_cache_reused_blocks", reused as f64);
            if let Some(admission) = self.admission {
                let input = self.input.max(1);
                let blocks = input.div_ceil(self.block_size.max(1));
                let chosen = admission.estimated_overlap_tokens / self.block_size.max(1);
                self.observe(
                    "llm_kv_cache_estimated_prefix_hit_rate",
                    admission.estimated_overlap_tokens as f64 / input as f64,
                );
                self.observe(
                    "llm_kv_cache_best_prefix_hit_rate",
                    (admission.best_overlap_blocks * self.block_size) as f64 / input as f64,
                );
                self.observe(
                    "llm_kv_cache_hit_rate_skew",
                    (chosen.min(blocks) as f64 - reused.saturating_sub(1).min(blocks) as f64)
                        / blocks as f64
                        + 1.0,
                );
                if admission.best_overlap_blocks > 0 {
                    self.observe(
                        "llm_kv_cache_hit_rate_efficiency",
                        chosen.min(blocks) as f64 / admission.best_overlap_blocks as f64,
                    );
                }
            }
        }
        self.tokens.clear();
        let mut success = false;
        let mut failed = false;
        if let Some(outputs) = data["outputs"].as_array() {
            for output in outputs {
                if let Some(tokens) = output["token_ids_diff"].as_array() {
                    self.tokens.extend(
                        tokens
                            .iter()
                            .filter_map(|v| v.as_u64().and_then(|n| u32::try_from(n).ok())),
                    );
                }
                if let Some(reason) = output["finish_reason"].as_str() {
                    failed |= matches!(reason, "cancelled" | "timeout" | "error");
                    success |= matches!(reason, "stop" | "length");
                }
            }
        } else if let Some(tokens) = data["token_ids"].as_array() {
            self.tokens.extend(
                tokens
                    .iter()
                    .filter_map(|v| v.as_u64().and_then(|n| u32::try_from(n).ok())),
            );
            success = data["finish_reason"]
                .as_str()
                .is_some_and(|s| matches!(s, "stop" | "length" | "eos"));
        }
        let count = self.tokens.len() as u64;
        if count > 0 {
            let now = Instant::now();
            self.tracker
                .record_tokens(&self.tokens, self.cached, true, None);
            self.chunks
                .record(
                    count,
                    self.first
                        .is_none()
                        .then(|| now.duration_since(self.started).as_secs_f64()),
                    self.last.map(|last| now.duration_since(last).as_secs_f64()),
                )
                .expect("live recorder");
            if self.first.is_none() {
                self.first = Some(now);
                self.first_tokens = count;
            }
            self.last = Some(now);
            self.output += count;
        }
        self.failed |= failed;
        // Prefill handoffs retain `finished: true` but lose their finish reason.
        // Keep accounting open for final cache/usage metadata after token completion.
        self.successful_terminal |= success && data["finished"].as_bool() != Some(false);
    }

    pub fn finish(&mut self, success: bool) {
        if self.finalized {
            return;
        }
        self.finalized = true;
        self.record_input();
        let elapsed = self.started.elapsed().as_secs_f64();
        self.observe("llm_e2e_latency", elapsed);
        self.observe("llm_processor_output_tokens_histogram", self.output as f64);
        if self.output > 0 && elapsed > 0.0 {
            self.observe("llm_request_tps", self.output as f64 / elapsed);
        }
        if let (Some(first), Some(last)) = (self.first, self.last) {
            let count = self.output - self.first_tokens;
            if count > 0 && last > first {
                self.observe(
                    "llm_request_time_per_output_token_seconds",
                    last.duration_since(first).as_secs_f64() / count as f64,
                );
            }
        }
        let cached = self.cached.unwrap_or(0).min(self.input);
        self.series.uncached.inc(self.input - cached);
        let miss = |reason: &str, count| self.series.missed[reason].inc(count);
        if let Some(admission) = self.admission {
            if self.cached.is_some() {
                let held = self.full_hit.unwrap_or(0).min(self.input).max(cached);
                let chosen = admission.estimated_overlap_tokens.min(self.input);
                let best = (admission.best_overlap_blocks * self.block_size)
                    .min(self.input)
                    .max(chosen);
                miss("held_not_reused", held - cached);
                miss("stale_index", chosen.saturating_sub(held));
                miss("routed_away", best.saturating_sub(chosen.max(held)));
                miss("cold", self.input - best.max(held));
                self.series.unpredicted.inc(cached.saturating_sub(chosen));
            } else {
                miss("no_engine_reading", self.input);
            }
        } else {
            miss("unrouted", self.input - cached);
        }
        if success {
            self.tracker.success();
        } else {
            self.tracker.cancel();
        }
        self.chunks.close();
    }
}

impl GenerationObserver for GenerationRecorder {
    fn on_admission(&mut self, admission: &GenerationAdmission) {
        self.admission(*admission);
    }

    fn on_chunk(&mut self, chunk: &Annotated<Value>) {
        if chunk.error.is_some() || chunk.event.as_deref() == Some("error") {
            self.failed = true;
        } else if let Some(data) = &chunk.data {
            self.chunk(data);
        }
    }

    fn on_denied(&mut self, denied: &DeniedGenerationRequest) {
        if let Some(admission) = denied.admission {
            self.admission(admission);
        }
        if let Some(usage) = denied.usage {
            self.usage(usage);
        }
    }

    fn on_finish(&mut self, _end: GenerationObservationEnd) {
        self.finish(self.successful_terminal && !self.failed);
    }
}

impl Drop for GenerationRecorder {
    fn drop(&mut self) {
        self.finish(self.successful_terminal && !self.failed);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::RouterRequestNew;
    use dynamo_runtime::pipeline::context::Controller;

    pub(crate) fn metrics() -> GenerationMetrics {
        let names = [
            "llm_time_to_first_token_seconds",
            "llm_time_per_output_token_seconds",
            "llm_inter_token_latency_seconds",
            "llm_request_time_per_output_token_seconds",
            "llm_processor_input_tokens_histogram",
            "llm_processor_output_tokens_histogram",
            "llm_request_tps",
            "llm_e2e_latency",
        ];
        GenerationMetrics::new(MetricsConfig {
            labels: HashMap::from([
                ("model_name".into(), "model".into()),
                ("exported_namespace".into(), "deployment-org".into()),
                ("model_version_id".into(), "deployment".into()),
            ]),
            factory_labels: None,
            histograms: names
                .into_iter()
                .map(|name| {
                    (
                        name.into(),
                        HistogramConfig {
                            bounds: vec![1.0],
                            ranges: vec!["0...1".into(), "1...+Inf".into()],
                            labels: vec!["model_name".into(), "service_tier".into()],
                        },
                    )
                })
                .collect(),
            factory: RequestMetricsOptions::default(),
            session_metrics_enabled: false,
        })
        .unwrap()
    }

    pub(crate) fn begin(metrics: &GenerationMetrics) -> GenerationRecorder {
        let context = RequestContext::new(
            Arc::new(Controller::new(
                "billing-org--request--billing-version".into(),
            )),
            None,
            BTreeMap::new(),
        );
        metrics
            .begin(
                &context,
                &GenerationRequest {
                    routing_request: RouterRequestNew {
                        tokens: vec![1, 2, 3],
                        ..Default::default()
                    },
                    primary_worker_request: Value::Nil,
                    decode_worker_request: None,
                },
                2,
            )
            .unwrap()
    }

    pub(crate) fn value(metrics: &GenerationMetrics, name: &str) -> f64 {
        metrics
            .samples()
            .iter()
            .filter(|sample| sample.name == name)
            .map(|sample| sample.value)
            .sum()
    }

    fn chunk(value: serde_json::Value) -> Value {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn legacy_counter_creation_timestamps_are_gauges() {
        let metrics = metrics();
        drop(begin(&metrics));
        let samples = metrics.samples();
        for name in [
            "llm_input_tokens_created",
            "llm_output_tokens_created",
            "llm_cached_input_tokens_created",
            "llm_uncached_input_tokens_created",
        ] {
            let sample = samples.iter().find(|s| s.name == name).unwrap();
            assert_eq!(sample.metric_type, "gauge");
            assert!(sample.value > 1_000_000_000.0);
        }
        assert!(
            !samples
                .iter()
                .any(|s| s.name == "llm_e2e_latency_count_created")
        );
    }

    #[test]
    fn invalid_log_reload_preserves_existing_configuration() {
        let metrics = metrics();
        let trace = RequestTraceOptions {
            request_log_enabled: true,
            s3_config: Some(llm_runtime_metrics::__private::S3SinkConfig {
                bucket: "test".into(),
                region: "us-east-1".into(),
                endpoint: None,
                proxy_url: Some("ftp://localhost".into()),
                access_key_id: None,
                secret_access_key: None,
                prefix: "test".into(),
            }),
            ..Default::default()
        };
        assert!(metrics.configure(trace, true).is_err());
        assert!(!metrics.factory.request_log_enabled());
        assert!(metrics.session_metrics_enabled.load(Ordering::Relaxed));
    }

    fn observer_request() -> GenerationRequest {
        GenerationRequest {
            routing_request: RouterRequestNew {
                tokens: vec![1, 2, 3],
                ..Default::default()
            },
            primary_worker_request: Value::Nil,
            decode_worker_request: None,
        }
    }

    #[test]
    fn b10_metrics_observer_drop_releases_inflight_once() {
        let metrics = Arc::new(metrics());
        let context = RequestContext::new(
            Arc::new(Controller::new("observer-drop".into())),
            None,
            BTreeMap::new(),
        );
        let observer = metrics
            .observer_factory(2)
            .begin(&context, &observer_request())
            .unwrap();
        assert_eq!(value(&metrics, "current_inflight_requests"), 1.0);
        drop(observer);
        assert_eq!(value(&metrics, "current_inflight_requests"), 0.0);
        assert_eq!(value(&metrics, "total_pre_first_token_cancellations"), 1.0);
        assert_eq!(value(&metrics, "llm_input_tokens_total"), 4.0);
    }

    #[test]
    fn b10_metrics_observer_success_records_terminal_accounting() {
        let metrics = Arc::new(metrics());
        let context = RequestContext::new(
            Arc::new(Controller::new("observer-success".into())),
            None,
            BTreeMap::new(),
        );
        let mut observer = metrics
            .observer_factory(2)
            .begin(&context, &observer_request())
            .unwrap();
        observer.on_admission(&GenerationAdmission {
            prefill_worker_id: 7,
            ..Default::default()
        });
        observer.on_chunk(&Annotated::from_data(chunk(serde_json::json!({
            "cached_tokens": 2,
            "finished": true,
            "outputs": [{"token_ids_diff": [4, 5], "finish_reason": "stop"}]
        }))));
        observer.on_finish(GenerationObservationEnd::StreamClosed);
        drop(observer);
        assert_eq!(value(&metrics, "total_successful_requests"), 1.0);
        assert_eq!(value(&metrics, "total_pre_20_token_cancellations"), 0.0);
        assert_eq!(value(&metrics, "llm_output_tokens_total"), 2.0);
        assert_eq!(value(&metrics, "llm_cached_input_tokens_total"), 2.0);
        assert_eq!(value(&metrics, "llm_uncached_input_tokens_total"), 2.0);
        assert_eq!(
            value(&metrics, "llm_time_to_first_token_seconds_count_total"),
            1.0
        );
        assert_eq!(value(&metrics, "llm_e2e_latency_count_total"), 1.0);
    }

    #[test]
    fn b10_metrics_observer_denied_records_usage_without_success() {
        use crate::DeniedRequest;

        let metrics = Arc::new(metrics());
        let context = RequestContext::new(
            Arc::new(Controller::new("observer-denied".into())),
            None,
            BTreeMap::new(),
        );
        let mut observer = metrics
            .observer_factory(2)
            .begin(&context, &observer_request())
            .unwrap();
        observer.on_denied(&DeniedGenerationRequest {
            usage: Some(crate::protocol::TokenUsageV1 {
                input_tokens: Some(3),
                cached_tokens: Some(2),
                output_tokens: Some(1),
            }),
            denied: DeniedRequest::Cancelled(),
            admission: Some(GenerationAdmission {
                estimated_overlap_tokens: 4,
                best_overlap_blocks: 2,
                prefill_worker_id: 1,
                prefill_dp_rank: 3,
                decode_worker_id: None,
                decode_dp_rank: None,
                prefill_duration: None,
            }),
        });
        observer.on_finish(GenerationObservationEnd::Denied);
        drop(observer);
        assert_eq!(value(&metrics, "total_successful_requests"), 0.0);
        assert_eq!(value(&metrics, "llm_output_tokens_total"), 1.0);
        assert_eq!(value(&metrics, "total_output_tokens"), 1.0);
        assert_eq!(
            value(&metrics, "llm_processor_output_tokens_histogram_sum_total"),
            1.0
        );
        assert_eq!(value(&metrics, "total_pre_20_token_cancellations"), 1.0);
        assert_eq!(
            value(&metrics, "llm_time_to_first_token_seconds_count_total"),
            0.0
        );
    }

    #[test]
    fn e2e_keeps_effective_tier_while_token_counters_use_billing_tier() {
        let metrics = metrics();
        let context = RequestContext::new(
            Arc::new(Controller::new("org--request--version".into())),
            None,
            BTreeMap::new(),
        );
        let request = GenerationRequest {
            routing_request: RouterRequestNew {
                tokens: vec![1],
                ..Default::default()
            },
            primary_worker_request: chunk(serde_json::json!({"service_tier":"fast"})),
            decode_worker_request: None,
        };
        drop(metrics.begin(&context, &request, 32).unwrap());
        let samples = metrics.samples();
        assert_eq!(
            samples
                .iter()
                .find(|s| s.name == "llm_e2e_latency_count_total")
                .unwrap()
                .labels["service_tier"],
            "fast"
        );
        assert_eq!(
            samples
                .iter()
                .find(|s| s.name == "llm_input_tokens_total")
                .unwrap()
                .labels["service_tier"],
            ""
        );
    }

    #[test]
    fn drop_without_poll_records_one_cancel_and_no_latency_sample() {
        let metrics = metrics();
        let recorder = begin(&metrics);
        assert_eq!(value(&metrics, "current_inflight_requests"), 1.0);
        drop(recorder);
        assert_eq!(value(&metrics, "current_inflight_requests"), 0.0);
        assert_eq!(value(&metrics, "total_pre_first_token_cancellations"), 1.0);
        assert_eq!(value(&metrics, "llm_input_tokens_total"), 4.0);
        assert_eq!(
            value(&metrics, "llm_time_to_first_token_seconds_count_total"),
            0.0
        );
        assert_eq!(value(&metrics, "llm_e2e_latency_count_total"), 1.0);
    }

    #[test]
    fn live_partial_tokens_survive_cancel_with_deployment_labels() {
        let metrics = metrics();
        let mut recorder = begin(&metrics);
        recorder.chunk(&chunk(
            serde_json::json!({"outputs":[{"token_ids_diff":[4,5]}]}),
        ));
        assert_eq!(value(&metrics, "llm_output_tokens_total"), 2.0);
        assert_eq!(value(&metrics, "current_inflight_output_tokens"), 2.0);
        assert!(
            metrics
                .samples()
                .iter()
                .filter(|s| s.name == "llm_output_tokens_total")
                .all(|s| s.labels["model_version_id"] == "deployment")
        );
        drop(recorder);
        assert_eq!(value(&metrics, "llm_output_tokens_total"), 2.0);
        assert_eq!(value(&metrics, "current_inflight_requests"), 0.0);
        assert_eq!(value(&metrics, "total_pre_20_token_cancellations"), 1.0);
    }

    #[test]
    fn terminal_token_keeps_late_cache_metadata_until_accounting_closes() {
        let metrics = metrics();
        let mut recorder = begin(&metrics);
        recorder.chunk(&chunk(serde_json::json!({"finished":true,"outputs":[{"token_ids_diff":[4],"finish_reason":"stop"}]})));
        recorder.chunk(&chunk(serde_json::json!({"cached_tokens":2,"outputs":[]})));
        drop(recorder);
        assert_eq!(value(&metrics, "total_successful_requests"), 1.0);
        assert_eq!(value(&metrics, "llm_cached_input_tokens_total"), 2.0);
        assert_eq!(value(&metrics, "llm_uncached_input_tokens_total"), 2.0);
        assert_eq!(value(&metrics, "llm_output_tokens_total"), 1.0);
    }

    #[test]
    fn success_then_drop_is_finalized_once_with_late_cache_reading() {
        let metrics = metrics();
        let mut recorder = begin(&metrics);
        recorder.chunk(&chunk(
            serde_json::json!({"outputs":[{"token_ids_diff":[4,5,6]}]}),
        ));
        recorder.chunk(&chunk(serde_json::json!({"cached_tokens":2,"outputs":[{"token_ids_diff":[7,8],"finish_reason":"stop"}]})));
        drop(recorder);
        assert_eq!(value(&metrics, "total_successful_requests"), 1.0);
        assert_eq!(value(&metrics, "total_pre_20_token_cancellations"), 0.0);
        assert_eq!(value(&metrics, "llm_cached_input_tokens_total"), 2.0);
        assert_eq!(value(&metrics, "llm_uncached_input_tokens_total"), 2.0);
        assert_eq!(value(&metrics, "llm_e2e_latency_count_total"), 1.0);
        assert_eq!(
            value(&metrics, "llm_inter_token_latency_seconds_count_total"),
            2.0
        );
        assert_eq!(
            value(
                &metrics,
                "llm_request_time_per_output_token_seconds_count_total"
            ),
            1.0
        );
    }
    struct NativeMetricsClient {
        observer: Arc<dyn GenerationObserverFactory>,
    }
    impl crate::GenerationCoordinatorClient for NativeMetricsClient {
        fn observer(&self) -> Option<&dyn GenerationObserverFactory> {
            Some(self.observer.as_ref())
        }
        fn generate_inner(
            &self,
            context: RequestContext,
            _: GenerationRequest,
            _: crate::GenerationOptions,
        ) -> futures::future::BoxFuture<'_, Result<crate::GenerationOutcome>> {
            Box::pin(async move {
                Ok(crate::GenerationOutcome::Connected(
                    crate::GeneratedRequest {
                        admission: GenerationAdmission {
                            prefill_worker_id: 1,
                            ..Default::default()
                        },
                        stream: dynamo_runtime::pipeline::ResponseStream::new(
                            Box::pin(futures::stream::iter([Annotated::from_data(chunk(
                                serde_json::json!({
                                    "cached_tokens": 2, "finished": true,
                                    "outputs": [{"token_ids_diff": [4, 5], "finish_reason": "stop"}]
                                }),
                            ))])),
                            context.inner(),
                        ),
                    },
                ))
            })
        }
    }

    #[tokio::test]
    async fn b10_metrics_record_once_on_both_http_boundaries_with_existing_identities() {
        use crate::GenerationCoordinatorClient;
        use futures::StreamExt;
        let receiving = Arc::new(metrics());
        let service = Arc::new(crate::GenerationCoordinatorService::new(
            Arc::new(NativeMetricsClient {
                observer: receiving.observer_factory(2),
            }),
            crate::DisaggregationStrategy::Aggregated,
        ));
        let server = service.start("127.0.0.1:0".parse().unwrap()).await.unwrap();
        let mut sending = metrics();
        sending.config.factory_labels =
            Some(HashMap::from([("model_name".into(), "model".into())]));
        let sending = Arc::new(sending);
        let remote = crate::RemoteGenerationCoordinator::new(server.endpoint_url())
            .unwrap()
            .with_observer(sending.sending_observer_factory(2));
        let identities = [
            ("billing-a", "version-a"),
            ("billing-b", "version-b"),
            ("NoNe", "NONE"),
        ];
        for (namespace, version) in identities {
            let context = RequestContext::new(
                Arc::new(Controller::new(format!(
                    "{namespace}--request--{version}--opaque"
                ))),
                None,
                BTreeMap::new(),
            );
            let mut request = observer_request();
            request.primary_worker_request = Value::Map(vec![]);
            let crate::GenerationOutcome::Connected(mut generated) = remote
                .generate(context, request, Default::default())
                .await
                .unwrap()
            else {
                panic!("expected stream")
            };
            while let Some(item) = generated.stream.next().await {
                assert!(item.error.is_none());
            }
        }
        for metrics in [&sending, &receiving] {
            assert_eq!(value(metrics, "llm_input_tokens_total"), 12.0);
            assert_eq!(value(metrics, "llm_output_tokens_total"), 6.0);
            assert_eq!(value(metrics, "total_successful_requests"), 3.0);
            assert_eq!(value(metrics, "current_inflight_requests"), 0.0);
            let mut keys = std::collections::BTreeSet::new();
            for sample in metrics.samples() {
                assert!(
                    keys.insert((
                        sample.name,
                        sample.labels.into_iter().collect::<BTreeMap<_, _>>()
                    )),
                    "duplicate full labelset"
                );
            }
        }
        let sender_samples = sending.samples();
        for (namespace, version) in [
            ("billing-a", "version-a"),
            ("billing-b", "version-b"),
            ("", ""),
        ] {
            let sample = sender_samples
                .iter()
                .find(|sample| {
                    sample.name == "llm_input_tokens_total"
                        && sample.labels["exported_namespace"] == namespace
                        && sample.labels["model_version_id"] == version
                })
                .unwrap();
            assert_eq!(sample.value, 4.0);
        }
        let core = sender_samples
            .iter()
            .find(|sample| sample.name == "total_successful_requests")
            .unwrap();
        assert_eq!(
            core.labels.get("model_name").map(String::as_str),
            Some("model")
        );
        assert!(!core.labels.contains_key("exported_namespace"));
        assert!(!core.labels.contains_key("model_version_id"));
        let receiving_samples = receiving.samples();
        let counter = receiving_samples
            .iter()
            .find(|sample| sample.name == "llm_input_tokens_total")
            .unwrap();
        assert_eq!(counter.labels["exported_namespace"], "deployment-org");
        assert_eq!(counter.labels["model_version_id"], "deployment");
        assert_eq!(counter.value, 12.0);
        server.shutdown().await.unwrap();
    }
}
