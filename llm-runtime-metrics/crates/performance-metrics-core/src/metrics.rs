use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const DEFAULT_WINDOW_SECONDS: f64 = 60.0;
const PRUNE_EVERY_WRITES: u32 = 64;
const MAX_SERIES: usize = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricKind {
    Rate,
    Distribution,
    Ratio,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSnapshot {
    pub kind: MetricKind,
    pub rate_per_second: Option<f64>,
    pub average: Option<f64>,
    pub quantiles: HashMap<String, f64>,
    pub numerator_sum: Option<f64>,
    pub denominator_sum: Option<f64>,
    pub ratio: Option<f64>,
}

pub type MetricLabels = BTreeMap<String, String>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSeriesSnapshot {
    pub name: String,
    pub labels: MetricLabels,
    pub snapshot: MetricSnapshot,
}

#[derive(Debug, Clone)]
struct PerformanceMetricSpec {
    kind: MetricKind,
    quantiles: Vec<f64>,
    sample_period_seconds: Option<f64>,
    window_seconds: f64,
}

#[derive(Debug, Clone)]
enum MetricState {
    Rate {
        counts: VecDeque<(Instant, u64)>,
    },
    Distribution {
        values: VecDeque<(Instant, f64)>,
    },
    Ratio {
        pairs: VecDeque<(Instant, f64, f64)>,
    },
}

#[derive(Debug, Clone)]
struct MetricEntry {
    name: String,
    labels: MetricLabels,
    spec: PerformanceMetricSpec,
    state: MetricState,
    writes_since_prune: u32,
}

#[derive(Clone)]
pub struct PerformanceMetricsRegistry {
    inner: Arc<Mutex<HashMap<String, MetricEntry>>>,
}

#[derive(Clone)]
pub struct RateMetricHandle {
    registry: PerformanceMetricsRegistry,
    series_id: String,
}

#[derive(Clone)]
pub struct DistributionMetricHandle {
    registry: PerformanceMetricsRegistry,
    series_id: String,
}

#[derive(Clone)]
pub struct RatioMetricHandle {
    registry: PerformanceMetricsRegistry,
    series_id: String,
}

impl Default for PerformanceMetricsRegistry {
    fn default() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl PerformanceMetricsRegistry {
    pub fn new_rate_metric(
        &self,
        name: impl Into<String>,
        quantiles: Vec<f64>,
        sample_period_seconds: Option<f64>,
        window_seconds: Option<f64>,
    ) -> anyhow::Result<RateMetricHandle> {
        self.new_rate_metric_with_labels(
            name,
            MetricLabels::new(),
            quantiles,
            sample_period_seconds,
            window_seconds,
        )
    }

    pub fn new_rate_metric_with_labels(
        &self,
        name: impl Into<String>,
        labels: MetricLabels,
        quantiles: Vec<f64>,
        sample_period_seconds: Option<f64>,
        window_seconds: Option<f64>,
    ) -> anyhow::Result<RateMetricHandle> {
        let name = name.into();
        let series_id = metric_series_id(name.as_str(), &labels);
        self.register(
            series_id.clone(),
            name,
            labels,
            PerformanceMetricSpec {
                kind: MetricKind::Rate,
                quantiles: normalize_quantiles(quantiles),
                sample_period_seconds: sample_period_seconds
                    .filter(|v| v.is_finite() && *v > 0.0)
                    .map(|v| v.max(0.1)),
                window_seconds: window_seconds
                    .filter(|v| v.is_finite() && *v > 0.0)
                    .map(|v| v.max(0.1))
                    .unwrap_or(DEFAULT_WINDOW_SECONDS),
            },
        )?;
        Ok(RateMetricHandle {
            registry: self.clone(),
            series_id,
        })
    }

    pub fn get_or_create_rate_metric_with_labels(
        &self,
        name: impl Into<String>,
        labels: MetricLabels,
        quantiles: Vec<f64>,
        sample_period_seconds: Option<f64>,
        window_seconds: Option<f64>,
    ) -> anyhow::Result<RateMetricHandle> {
        let name = name.into();
        let series_id = metric_series_id(name.as_str(), &labels);
        self.get_or_create(
            series_id.clone(),
            name,
            labels,
            PerformanceMetricSpec {
                kind: MetricKind::Rate,
                quantiles: normalize_quantiles(quantiles),
                sample_period_seconds: sample_period_seconds
                    .filter(|v| v.is_finite() && *v > 0.0)
                    .map(|v| v.max(0.1)),
                window_seconds: window_seconds
                    .filter(|v| v.is_finite() && *v > 0.0)
                    .map(|v| v.max(0.1))
                    .unwrap_or(DEFAULT_WINDOW_SECONDS),
            },
        )?;
        Ok(RateMetricHandle {
            registry: self.clone(),
            series_id,
        })
    }

    pub fn new_distribution_metric(
        &self,
        name: impl Into<String>,
        quantiles: Vec<f64>,
        window_seconds: Option<f64>,
    ) -> anyhow::Result<DistributionMetricHandle> {
        self.new_distribution_metric_with_labels(
            name,
            MetricLabels::new(),
            quantiles,
            window_seconds,
        )
    }

    pub fn new_distribution_metric_with_labels(
        &self,
        name: impl Into<String>,
        labels: MetricLabels,
        quantiles: Vec<f64>,
        window_seconds: Option<f64>,
    ) -> anyhow::Result<DistributionMetricHandle> {
        let name = name.into();
        let series_id = metric_series_id(name.as_str(), &labels);
        self.register(
            series_id.clone(),
            name,
            labels,
            PerformanceMetricSpec {
                kind: MetricKind::Distribution,
                quantiles: normalize_quantiles(quantiles),
                sample_period_seconds: None,
                window_seconds: window_seconds
                    .filter(|v| v.is_finite() && *v > 0.0)
                    .map(|v| v.max(0.1))
                    .unwrap_or(DEFAULT_WINDOW_SECONDS),
            },
        )?;
        Ok(DistributionMetricHandle {
            registry: self.clone(),
            series_id,
        })
    }

    pub fn get_or_create_distribution_metric_with_labels(
        &self,
        name: impl Into<String>,
        labels: MetricLabels,
        quantiles: Vec<f64>,
        window_seconds: Option<f64>,
    ) -> anyhow::Result<DistributionMetricHandle> {
        let name = name.into();
        let series_id = metric_series_id(name.as_str(), &labels);
        self.get_or_create(
            series_id.clone(),
            name,
            labels,
            PerformanceMetricSpec {
                kind: MetricKind::Distribution,
                quantiles: normalize_quantiles(quantiles),
                sample_period_seconds: None,
                window_seconds: window_seconds
                    .filter(|v| v.is_finite() && *v > 0.0)
                    .map(|v| v.max(0.1))
                    .unwrap_or(DEFAULT_WINDOW_SECONDS),
            },
        )?;
        Ok(DistributionMetricHandle {
            registry: self.clone(),
            series_id,
        })
    }

    pub fn new_ratio_metric(
        &self,
        name: impl Into<String>,
        quantiles: Vec<f64>,
        window_seconds: Option<f64>,
    ) -> anyhow::Result<RatioMetricHandle> {
        self.new_ratio_metric_with_labels(name, MetricLabels::new(), quantiles, window_seconds)
    }

    pub fn new_ratio_metric_with_labels(
        &self,
        name: impl Into<String>,
        labels: MetricLabels,
        quantiles: Vec<f64>,
        window_seconds: Option<f64>,
    ) -> anyhow::Result<RatioMetricHandle> {
        let name = name.into();
        let series_id = metric_series_id(name.as_str(), &labels);
        self.register(
            series_id.clone(),
            name,
            labels,
            PerformanceMetricSpec {
                kind: MetricKind::Ratio,
                quantiles: normalize_quantiles(quantiles),
                sample_period_seconds: None,
                window_seconds: window_seconds
                    .filter(|v| v.is_finite() && *v > 0.0)
                    .map(|v| v.max(0.1))
                    .unwrap_or(DEFAULT_WINDOW_SECONDS),
            },
        )?;
        Ok(RatioMetricHandle {
            registry: self.clone(),
            series_id,
        })
    }

    pub fn get_or_create_ratio_metric_with_labels(
        &self,
        name: impl Into<String>,
        labels: MetricLabels,
        quantiles: Vec<f64>,
        window_seconds: Option<f64>,
    ) -> anyhow::Result<RatioMetricHandle> {
        let name = name.into();
        let series_id = metric_series_id(name.as_str(), &labels);
        self.get_or_create(
            series_id.clone(),
            name,
            labels,
            PerformanceMetricSpec {
                kind: MetricKind::Ratio,
                quantiles: normalize_quantiles(quantiles),
                sample_period_seconds: None,
                window_seconds: window_seconds
                    .filter(|v| v.is_finite() && *v > 0.0)
                    .map(|v| v.max(0.1))
                    .unwrap_or(DEFAULT_WINDOW_SECONDS),
            },
        )?;
        Ok(RatioMetricHandle {
            registry: self.clone(),
            series_id,
        })
    }

    pub fn snapshot_all(&self) -> HashMap<String, MetricSnapshot> {
        let now = Instant::now();
        let mut guard = self.inner.lock().expect("metrics mutex poisoned");
        guard
            .iter_mut()
            .map(|(_, entry)| {
                prune_entry(now, entry);
                entry.writes_since_prune = 0;
                (
                    format_metric_series_key(entry.name.as_str(), &entry.labels),
                    snapshot_entry(now, entry),
                )
            })
            .collect()
    }

    pub fn snapshot_all_series(&self) -> Vec<MetricSeriesSnapshot> {
        let now = Instant::now();
        let mut guard = self.inner.lock().expect("metrics mutex poisoned");
        let mut out = guard
            .iter_mut()
            .map(|(_, entry)| {
                prune_entry(now, entry);
                entry.writes_since_prune = 0;
                MetricSeriesSnapshot {
                    name: entry.name.clone(),
                    labels: entry.labels.clone(),
                    snapshot: snapshot_entry(now, entry),
                }
            })
            .collect::<Vec<_>>();
        out.sort_by(|a, b| {
            a.name
                .cmp(&b.name)
                .then_with(|| a.labels.iter().cmp(b.labels.iter()))
        });
        out
    }

    pub fn unregister_metric(&self, name: &str) {
        self.unregister_metric_with_labels(name, &MetricLabels::new());
    }

    pub fn unregister_metric_with_labels(&self, name: &str, labels: &MetricLabels) {
        let mut guard = self.inner.lock().expect("metrics mutex poisoned");
        guard.remove(metric_series_id(name, labels).as_str());
    }

    fn register(
        &self,
        series_id: String,
        name: String,
        labels: MetricLabels,
        spec: PerformanceMetricSpec,
    ) -> anyhow::Result<()> {
        let mut guard = self.inner.lock().expect("metrics mutex poisoned");
        insert_entry(&mut guard, series_id, name, labels, spec, false)?;
        Ok(())
    }

    fn get_or_create(
        &self,
        series_id: String,
        name: String,
        labels: MetricLabels,
        spec: PerformanceMetricSpec,
    ) -> anyhow::Result<()> {
        let mut guard = self.inner.lock().expect("metrics mutex poisoned");
        insert_entry(&mut guard, series_id, name, labels, spec, true)?;
        Ok(())
    }

    fn record_count(&self, name: &str, count: u64) -> anyhow::Result<()> {
        if count == 0 {
            return Ok(());
        }
        let now = Instant::now();
        let mut guard = self.inner.lock().expect("metrics mutex poisoned");
        let entry = guard
            .get_mut(name)
            .with_context(|| format!("unknown metric '{}'", name))?;
        if entry.spec.kind != MetricKind::Rate {
            anyhow::bail!("metric '{}' is not a rate metric", name);
        }
        maybe_prune_entry(now, entry);
        if let MetricState::Rate { counts } = &mut entry.state {
            counts.push_back((now, count));
        }
        Ok(())
    }

    fn record_value(&self, name: &str, value: f64) -> anyhow::Result<()> {
        if !value.is_finite() {
            anyhow::bail!("value must be finite");
        }
        let now = Instant::now();
        let mut guard = self.inner.lock().expect("metrics mutex poisoned");
        let entry = guard
            .get_mut(name)
            .with_context(|| format!("unknown metric '{}'", name))?;
        if entry.spec.kind != MetricKind::Distribution {
            anyhow::bail!("metric '{}' is not a distribution metric", name);
        }
        maybe_prune_entry(now, entry);
        if let MetricState::Distribution { values } = &mut entry.state {
            values.push_back((now, value));
        }
        Ok(())
    }

    fn record_ratio(&self, name: &str, numerator: f64, denominator: f64) -> anyhow::Result<()> {
        if !numerator.is_finite()
            || !denominator.is_finite()
            || numerator < 0.0
            || denominator < 0.0
        {
            anyhow::bail!("invalid ratio inputs");
        }
        let now = Instant::now();
        let mut guard = self.inner.lock().expect("metrics mutex poisoned");
        let entry = guard
            .get_mut(name)
            .with_context(|| format!("unknown metric '{}'", name))?;
        if entry.spec.kind != MetricKind::Ratio {
            anyhow::bail!("metric '{}' is not a ratio metric", name);
        }
        maybe_prune_entry(now, entry);
        if let MetricState::Ratio { pairs } = &mut entry.state {
            pairs.push_back((now, numerator, denominator));
        }
        Ok(())
    }
}

impl RateMetricHandle {
    pub fn record_count(&self, count: u64) -> anyhow::Result<()> {
        self.registry.record_count(self.series_id.as_str(), count)
    }
}

impl DistributionMetricHandle {
    pub fn record_value(&self, value: f64) -> anyhow::Result<()> {
        self.registry.record_value(self.series_id.as_str(), value)
    }
}

impl RatioMetricHandle {
    pub fn record_ratio(&self, numerator: f64, denominator: f64) -> anyhow::Result<()> {
        self.registry
            .record_ratio(self.series_id.as_str(), numerator, denominator)
    }
}

fn metric_series_id(name: &str, labels: &MetricLabels) -> String {
    format_metric_series_key(name, labels)
}

fn format_metric_series_key(name: &str, labels: &MetricLabels) -> String {
    if labels.is_empty() {
        return name.to_string();
    }

    let rendered = labels
        .iter()
        .map(|(key, value)| format!(r#"{key}="{}""#, escape_label_value(value.as_str())))
        .collect::<Vec<_>>()
        .join(",");
    format!("{name}{{{rendered}}}")
}

fn escape_label_value(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\n', "\\n")
        .replace('"', "\\\"")
}

fn insert_entry(
    guard: &mut HashMap<String, MetricEntry>,
    series_id: String,
    name: String,
    labels: MetricLabels,
    spec: PerformanceMetricSpec,
    idempotent: bool,
) -> anyhow::Result<bool> {
    if let Some(existing) = guard.get(&series_id) {
        if idempotent {
            if existing.spec.kind != spec.kind {
                anyhow::bail!(
                    "metric series '{}' already exists as {:?}, requested {:?}",
                    format_metric_series_key(name.as_str(), &labels),
                    existing.spec.kind,
                    spec.kind
                );
            }
            return Ok(false);
        }
        anyhow::bail!(
            "metric series '{}' already exists",
            format_metric_series_key(name.as_str(), &labels)
        );
    }
    if guard.len() >= MAX_SERIES {
        anyhow::bail!(
            "metric cardinality limit reached ({MAX_SERIES} series); refusing to register '{}'",
            format_metric_series_key(name.as_str(), &labels)
        );
    }
    let state = match spec.kind {
        MetricKind::Rate => MetricState::Rate {
            counts: VecDeque::new(),
        },
        MetricKind::Distribution => MetricState::Distribution {
            values: VecDeque::new(),
        },
        MetricKind::Ratio => MetricState::Ratio {
            pairs: VecDeque::new(),
        },
    };
    guard.insert(
        series_id,
        MetricEntry {
            name,
            labels,
            spec,
            state,
            writes_since_prune: 0,
        },
    );
    Ok(true)
}

fn normalize_quantiles(mut quantiles: Vec<f64>) -> Vec<f64> {
    quantiles.retain(|q| q.is_finite());
    quantiles.iter_mut().for_each(|q| *q = q.clamp(0.0, 1.0));
    quantiles.sort_unstable_by(f64::total_cmp);
    quantiles.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    quantiles
}

fn prune_entry(now: Instant, entry: &mut MetricEntry) {
    let window = Duration::from_secs_f64(entry.spec.window_seconds.max(0.1));
    let lower = now - window;
    match &mut entry.state {
        MetricState::Rate { counts } => {
            while let Some((ts, _)) = counts.front() {
                if *ts < lower {
                    counts.pop_front();
                } else {
                    break;
                }
            }
        }
        MetricState::Distribution { values } => {
            while let Some((ts, _)) = values.front() {
                if *ts < lower {
                    values.pop_front();
                } else {
                    break;
                }
            }
        }
        MetricState::Ratio { pairs } => {
            while let Some((ts, _, _)) = pairs.front() {
                if *ts < lower {
                    pairs.pop_front();
                } else {
                    break;
                }
            }
        }
    }
}

fn maybe_prune_entry(now: Instant, entry: &mut MetricEntry) {
    entry.writes_since_prune = entry.writes_since_prune.saturating_add(1);
    if entry.writes_since_prune < PRUNE_EVERY_WRITES {
        return;
    }
    prune_entry(now, entry);
    entry.writes_since_prune = 0;
}

fn snapshot_entry(now: Instant, entry: &MetricEntry) -> MetricSnapshot {
    match &entry.state {
        MetricState::Rate { counts } => {
            let window = entry.spec.window_seconds.max(0.1);
            let total: u64 = counts.iter().map(|(_, c)| *c).sum();
            let sample_period = entry.spec.sample_period_seconds.unwrap_or(1.0).max(0.1);
            let quantiles = if counts.is_empty() {
                HashMap::new()
            } else {
                let samples = build_rate_samples(now, counts, window, sample_period);
                quantiles_map(samples.as_slice(), entry.spec.quantiles.as_slice())
            };
            MetricSnapshot {
                kind: MetricKind::Rate,
                rate_per_second: Some(total as f64 / window),
                average: None,
                quantiles,
                numerator_sum: None,
                denominator_sum: None,
                ratio: None,
            }
        }
        MetricState::Distribution { values } => {
            let series = values.iter().map(|(_, v)| *v).collect::<Vec<_>>();
            let average = if series.is_empty() {
                None
            } else {
                Some(avg(series.as_slice()))
            };
            MetricSnapshot {
                kind: MetricKind::Distribution,
                rate_per_second: None,
                average,
                quantiles: quantiles_map(series.as_slice(), entry.spec.quantiles.as_slice()),
                numerator_sum: None,
                denominator_sum: None,
                ratio: None,
            }
        }
        MetricState::Ratio { pairs } => {
            let numerator_sum: f64 = pairs.iter().map(|(_, n, _)| *n).sum();
            let denominator_sum: f64 = pairs.iter().map(|(_, _, d)| *d).sum();
            let ratio = if denominator_sum > 0.0 {
                Some(numerator_sum / denominator_sum)
            } else {
                None
            };
            let sample_ratios = pairs
                .iter()
                .filter_map(|(_, n, d)| if *d > 0.0 { Some(*n / *d) } else { None })
                .collect::<Vec<_>>();
            MetricSnapshot {
                kind: MetricKind::Ratio,
                rate_per_second: None,
                average: None,
                quantiles: quantiles_map(sample_ratios.as_slice(), entry.spec.quantiles.as_slice()),
                numerator_sum: (!pairs.is_empty()).then_some(numerator_sum),
                denominator_sum: (!pairs.is_empty()).then_some(denominator_sum),
                ratio,
            }
        }
    }
}

fn build_rate_samples(
    now: Instant,
    counts: &VecDeque<(Instant, u64)>,
    window_seconds: f64,
    sample_period_seconds: f64,
) -> Vec<f64> {
    let bins = (window_seconds / sample_period_seconds).ceil().max(1.0) as usize;
    let start = now - Duration::from_secs_f64(window_seconds);
    let mut sums = vec![0u64; bins];
    for (ts, count) in counts {
        if *ts < start || *ts > now {
            continue;
        }
        let offset = ts.duration_since(start).as_secs_f64();
        let idx = (offset / sample_period_seconds).floor() as usize;
        sums[idx.min(bins - 1)] = sums[idx.min(bins - 1)].saturating_add(*count);
    }

    sums.into_iter()
        .enumerate()
        .map(|(idx, c)| {
            let bucket_start = idx as f64 * sample_period_seconds;
            let bucket_end = ((idx + 1) as f64 * sample_period_seconds).min(window_seconds);
            let bucket_width = (bucket_end - bucket_start).max(1e-9);
            c as f64 / bucket_width
        })
        .collect()
}

fn quantiles_map(values: &[f64], quantiles: &[f64]) -> HashMap<String, f64> {
    if quantiles.is_empty() {
        return HashMap::new();
    }
    if values.is_empty() {
        return HashMap::new();
    }

    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    quantiles
        .iter()
        .map(|q| {
            let qq = q.clamp(0.0, 1.0);
            let idx = ((sorted.len() - 1) as f64 * qq).round() as usize;
            (quantile_label(qq), sorted[idx.min(sorted.len() - 1)])
        })
        .collect::<HashMap<_, _>>()
}

fn quantile_label(q: f64) -> String {
    let mut s = format!("{:.6}", q);
    while s.ends_with('0') {
        s.pop();
    }
    if s.ends_with('.') {
        s.push('0');
    }
    s
}

fn avg(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::time::Duration;

    #[test]
    fn snapshot_map_returns_registered_metrics() {
        let registry = PerformanceMetricsRegistry::default();
        let rate = registry
            .new_rate_metric("requests", vec![0.5, 0.9], Some(1.0), Some(30.0))
            .unwrap();
        let dist = registry
            .new_distribution_metric("ttft_ms", vec![0.5, 0.9], Some(30.0))
            .unwrap();
        let ratio = registry
            .new_ratio_metric("kv_hit_rate", vec![0.5, 0.9], Some(30.0))
            .unwrap();

        rate.record_count(5).unwrap();
        dist.record_value(40.0).unwrap();
        dist.record_value(60.0).unwrap();
        ratio.record_ratio(8.0, 10.0).unwrap();

        let snapshot = registry.snapshot_all();
        assert_eq!(snapshot["requests"].kind, MetricKind::Rate);
        assert_eq!(snapshot["ttft_ms"].kind, MetricKind::Distribution);
        assert_eq!(snapshot["kv_hit_rate"].kind, MetricKind::Ratio);
    }

    #[test]
    fn snapshot_series_preserves_labels() {
        let registry = PerformanceMetricsRegistry::default();
        let mut labels = MetricLabels::new();
        labels.insert("feature".to_string(), "tools".to_string());
        let dist = registry
            .new_distribution_metric_with_labels("ttft_ms", labels, vec![0.5], Some(30.0))
            .unwrap();

        dist.record_value(42.0).unwrap();

        let series = registry.snapshot_all_series();
        assert_eq!(series.len(), 1);
        assert_eq!(series[0].name, "ttft_ms");
        assert_eq!(series[0].labels["feature"], "tools");
        assert_eq!(series[0].snapshot.average, Some(42.0));
    }

    #[test]
    fn get_or_create_is_idempotent_across_kinds() {
        let registry = PerformanceMetricsRegistry::default();
        let mut labels = MetricLabels::new();
        labels.insert("service_tier".to_string(), "fast".to_string());

        let dist_a = registry
            .get_or_create_distribution_metric_with_labels(
                "ttft_ms",
                labels.clone(),
                vec![0.5, 0.9],
                Some(30.0),
            )
            .unwrap();
        let dist_b = registry
            .get_or_create_distribution_metric_with_labels(
                "ttft_ms",
                labels.clone(),
                vec![0.5, 0.9],
                Some(30.0),
            )
            .unwrap();
        assert_eq!(dist_a.series_id, dist_b.series_id);

        let rate_a = registry
            .get_or_create_rate_metric_with_labels(
                "requests",
                labels.clone(),
                vec![0.5],
                Some(1.0),
                Some(30.0),
            )
            .unwrap();
        let rate_b = registry
            .get_or_create_rate_metric_with_labels(
                "requests",
                labels.clone(),
                vec![0.5],
                Some(1.0),
                Some(30.0),
            )
            .unwrap();
        assert_eq!(rate_a.series_id, rate_b.series_id);

        let ratio_a = registry
            .get_or_create_ratio_metric_with_labels(
                "kv_hit_rate",
                labels.clone(),
                vec![0.5],
                Some(30.0),
            )
            .unwrap();
        let ratio_b = registry
            .get_or_create_ratio_metric_with_labels("kv_hit_rate", labels, vec![0.5], Some(30.0))
            .unwrap();
        assert_eq!(ratio_a.series_id, ratio_b.series_id);

        let series = registry.snapshot_all_series();
        assert_eq!(
            series.len(),
            3,
            "idempotent calls must not duplicate series"
        );
    }

    #[test]
    fn get_or_create_records_into_existing_series() {
        let registry = PerformanceMetricsRegistry::default();
        let mut labels = MetricLabels::new();
        labels.insert("service_tier".to_string(), "flex".to_string());

        let dist = registry
            .get_or_create_distribution_metric_with_labels(
                "ttft_ms",
                labels.clone(),
                vec![0.5],
                Some(30.0),
            )
            .unwrap();
        dist.record_value(100.0).unwrap();

        let dist_again = registry
            .get_or_create_distribution_metric_with_labels("ttft_ms", labels, vec![0.5], Some(30.0))
            .unwrap();
        dist_again.record_value(200.0).unwrap();

        let series = registry.snapshot_all_series();
        assert_eq!(series.len(), 1);
        assert_eq!(series[0].snapshot.average, Some(150.0));
    }

    #[test]
    fn get_or_create_rejects_kind_mismatch() {
        let registry = PerformanceMetricsRegistry::default();
        let mut labels = MetricLabels::new();
        labels.insert("service_tier".to_string(), "fast".to_string());

        registry
            .get_or_create_distribution_metric_with_labels(
                "ttft_ms",
                labels.clone(),
                vec![0.5],
                Some(30.0),
            )
            .unwrap();

        let result = registry.get_or_create_rate_metric_with_labels(
            "ttft_ms",
            labels,
            vec![0.5],
            Some(1.0),
            Some(30.0),
        );
        let err = match result {
            Err(err) => err.to_string(),
            Ok(_) => panic!("expected kind-mismatch error"),
        };
        assert!(
            err.contains("already exists as") && err.contains("Rate"),
            "expected kind-mismatch error, got: {err}"
        );
    }

    #[test]
    fn register_still_bails_on_duplicate_series() {
        let registry = PerformanceMetricsRegistry::default();
        let mut labels = MetricLabels::new();
        labels.insert("service_tier".to_string(), "default".to_string());

        registry
            .new_distribution_metric_with_labels("ttft_ms", labels.clone(), vec![0.5], Some(30.0))
            .unwrap();
        let result =
            registry.new_distribution_metric_with_labels("ttft_ms", labels, vec![0.5], Some(30.0));
        let err = match result {
            Err(err) => err.to_string(),
            Ok(_) => panic!("expected duplicate-registration error"),
        };
        assert!(
            err.contains("already exists"),
            "expected duplicate-registration error, got: {err}"
        );
    }

    #[test]
    fn get_or_create_enforces_cardinality_cap() {
        let registry = PerformanceMetricsRegistry::default();
        for i in 0..MAX_SERIES {
            let mut labels = MetricLabels::new();
            labels.insert("i".to_string(), i.to_string());
            registry
                .get_or_create_distribution_metric_with_labels(
                    "ttft_ms",
                    labels,
                    vec![0.5],
                    Some(30.0),
                )
                .unwrap();
        }
        let mut labels = MetricLabels::new();
        labels.insert("i".to_string(), MAX_SERIES.to_string());
        let result = registry.get_or_create_distribution_metric_with_labels(
            "ttft_ms",
            labels,
            vec![0.5],
            Some(30.0),
        );
        let err = match result {
            Err(err) => err.to_string(),
            Ok(_) => panic!("expected cardinality-cap error"),
        };
        assert!(
            err.contains("cardinality limit"),
            "expected cardinality-cap error, got: {err}"
        );
    }

    #[test]
    fn normalize_quantiles_clamps_sorts_and_dedups() {
        let qs = normalize_quantiles(vec![0.9, f64::NAN, -1.0, 0.5, 1.5, 0.5, 0.0]);
        assert_eq!(qs, vec![0.0, 0.5, 0.9, 1.0]);
    }

    #[test]
    fn build_rate_samples_includes_boundaries() {
        let now = Instant::now();
        let window_seconds = 2.0;
        let sample_period = 1.0;
        let start = now - Duration::from_secs_f64(window_seconds);

        let mut counts = VecDeque::new();
        counts.push_back((start, 1));
        counts.push_back((now, 1));

        let samples = build_rate_samples(now, &counts, window_seconds, sample_period);
        assert_eq!(samples.len(), 2);
        assert!((samples[0] - 1.0).abs() < 1e-9);
        assert!((samples[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn snapshot_ratio_handles_zero_denominator() {
        let registry = PerformanceMetricsRegistry::default();
        let ratio = registry
            .new_ratio_metric("hit_rate", vec![0.5], Some(10.0))
            .unwrap();
        ratio.record_ratio(10.0, 0.0).unwrap();
        ratio.record_ratio(5.0, 5.0).unwrap();

        let snapshot = registry.snapshot_all();
        let ratio_snapshot = &snapshot["hit_rate"];
        assert_eq!(ratio_snapshot.kind, MetricKind::Ratio);
        assert_eq!(ratio_snapshot.numerator_sum, Some(15.0));
        assert_eq!(ratio_snapshot.denominator_sum, Some(5.0));
        assert_eq!(ratio_snapshot.ratio, Some(3.0));
        assert!(ratio_snapshot.quantiles.contains_key("0.5"));
    }

    #[test]
    fn empty_distribution_omits_quantiles() {
        let registry = PerformanceMetricsRegistry::default();
        registry
            .new_distribution_metric("ttft_ms", vec![0.5, 0.9], Some(10.0))
            .unwrap();

        let snapshot = registry.snapshot_all();
        let dist_snapshot = &snapshot["ttft_ms"];
        assert_eq!(dist_snapshot.kind, MetricKind::Distribution);
        assert_eq!(dist_snapshot.average, None);
        assert!(dist_snapshot.quantiles.is_empty());
    }

    #[test]
    fn empty_rate_keeps_zero_rate_and_omits_quantiles() {
        let registry = PerformanceMetricsRegistry::default();
        registry
            .new_rate_metric("requests", vec![0.5, 0.9], Some(1.0), Some(10.0))
            .unwrap();

        let snapshot = registry.snapshot_all();
        let rate_snapshot = &snapshot["requests"];
        assert_eq!(rate_snapshot.kind, MetricKind::Rate);
        assert_eq!(rate_snapshot.rate_per_second, Some(0.0));
        assert!(rate_snapshot.quantiles.is_empty());
    }

    #[test]
    fn empty_ratio_omits_sums_ratio_and_quantiles() {
        let registry = PerformanceMetricsRegistry::default();
        registry
            .new_ratio_metric("hit_rate", vec![0.5, 0.9], Some(10.0))
            .unwrap();

        let snapshot = registry.snapshot_all();
        let ratio_snapshot = &snapshot["hit_rate"];
        assert_eq!(ratio_snapshot.kind, MetricKind::Ratio);
        assert_eq!(ratio_snapshot.numerator_sum, None);
        assert_eq!(ratio_snapshot.denominator_sum, None);
        assert_eq!(ratio_snapshot.ratio, None);
        assert!(ratio_snapshot.quantiles.is_empty());
    }

    #[test]
    fn tiny_window_prunes_old_samples() {
        let registry = PerformanceMetricsRegistry::default();
        let dist = registry
            .new_distribution_metric("ttft_ms", vec![0.5], Some(0.1))
            .unwrap();
        dist.record_value(100.0).unwrap();
        std::thread::sleep(Duration::from_millis(130));
        dist.record_value(10.0).unwrap();

        let snapshot = registry.snapshot_all();
        let avg = snapshot["ttft_ms"].average.unwrap_or_default();
        assert!(avg < 50.0, "expected old sample to be pruned, avg={avg}");
    }

    #[test]
    fn write_path_prunes_old_samples_without_snapshot() {
        let registry = PerformanceMetricsRegistry::default();
        let dist = registry
            .new_distribution_metric("ttft_ms", vec![0.5], Some(0.1))
            .unwrap();
        dist.record_value(100.0).unwrap();
        std::thread::sleep(Duration::from_millis(130));
        for _ in 0..(PRUNE_EVERY_WRITES - 1) {
            dist.record_value(10.0).unwrap();
        }

        let guard = registry.inner.lock().expect("metrics mutex poisoned");
        let entry = guard.get("ttft_ms").unwrap();
        match &entry.state {
            MetricState::Distribution { values } => {
                assert_eq!(values.len(), (PRUNE_EVERY_WRITES - 1) as usize);
                assert_eq!(values.front().map(|(_, value)| *value), Some(10.0));
            }
            _ => panic!("expected distribution metric state"),
        }
    }
}
