use prometheus::proto::{Counter, Gauge, LabelPair, Metric, MetricFamily, MetricType};
use prometheus::{Encoder, TextEncoder};
use std::collections::{BTreeMap, HashMap};

#[derive(Clone, Debug, serde::Serialize)]
pub struct PrometheusSample {
    pub name: String,
    pub value: f64,
    pub labels: HashMap<String, String>,
    pub metric_type: String,
}
pub fn flatten_snapshot(
    snapshot: crate::RequestMetricsSnapshot,
    base_labels: Option<&HashMap<String, String>>,
) -> Vec<PrometheusSample> {
    let mut out = flatten_series(snapshot.series, base_labels);
    out.extend(flatten_totals(snapshot.totals, base_labels));
    out
}

fn flatten_series(
    rolling: Vec<crate::MetricSeriesSnapshot>,
    base_labels: Option<&HashMap<String, String>>,
) -> Vec<PrometheusSample> {
    let mut out = Vec::new();
    for series in rolling {
        let metric_name = series.name;
        let extra_labels = Some(
            series
                .labels
                .into_iter()
                .collect::<HashMap<String, String>>(),
        );
        let snapshot = series.snapshot;
        if let Some(v) = snapshot.rate_per_second {
            out.push(make_sample(
                format!(
                    "{}_rate_per_second",
                    sanitize_metric_name(metric_name.as_str())
                ),
                v,
                base_labels,
                extra_labels.clone(),
                "gauge",
            ));
        }
        if let Some(v) = snapshot.average {
            out.push(make_sample(
                format!("{}_average", sanitize_metric_name(metric_name.as_str())),
                v,
                base_labels,
                extra_labels.clone(),
                "gauge",
            ));
        }
        if let Some(v) = snapshot.numerator_sum {
            out.push(make_sample(
                format!(
                    "{}_numerator_sum",
                    sanitize_metric_name(metric_name.as_str())
                ),
                v,
                base_labels,
                extra_labels.clone(),
                "gauge",
            ));
        }
        if let Some(v) = snapshot.denominator_sum {
            out.push(make_sample(
                format!(
                    "{}_denominator_sum",
                    sanitize_metric_name(metric_name.as_str())
                ),
                v,
                base_labels,
                extra_labels.clone(),
                "gauge",
            ));
        }
        if let Some(v) = snapshot.ratio {
            out.push(make_sample(
                format!("{}_ratio", sanitize_metric_name(metric_name.as_str())),
                v,
                base_labels,
                extra_labels.clone(),
                "gauge",
            ));
        }
        for (q, v) in snapshot.quantiles {
            let mut extra = extra_labels.clone().unwrap_or_default();
            extra.insert("quantile".to_string(), q);
            out.push(make_sample(
                format!("{}_quantile", sanitize_metric_name(metric_name.as_str())),
                v,
                base_labels,
                Some(extra),
                "gauge",
            ));
        }
    }
    out
}

fn flatten_totals(
    totals: crate::RequestMetricsTotalsSnapshot,
    base_labels: Option<&HashMap<String, String>>,
) -> Vec<PrometheusSample> {
    let mut out = Vec::new();
    let counters = [
        ("total_requests", totals.total_requests as f64),
        (
            "total_successful_requests",
            totals.total_successful_requests as f64,
        ),
        (
            "total_pre_first_token_cancellations",
            totals.total_pre_first_token_cancellations as f64,
        ),
        (
            "total_pre_20_token_cancellations",
            totals.total_pre_20_token_cancellations as f64,
        ),
        (
            "total_mid_stream_cancellations",
            totals.total_mid_stream_cancellations as f64,
        ),
        ("total_input_tokens", totals.total_input_tokens as f64),
        ("total_output_tokens", totals.total_output_tokens as f64),
        (
            "total_net_new_input_tokens",
            totals.total_net_new_input_tokens as f64,
        ),
        (
            "total_kv_cache_hits_tokens",
            totals.total_kv_cache_hits_tokens as f64,
        ),
        (
            "total_kv_cache_lookup_tokens",
            totals.total_kv_cache_lookup_tokens as f64,
        ),
        (
            "total_itl_samples_recorded",
            totals.total_itl_samples_recorded as f64,
        ),
        (
            "total_request_log_rows_persisted",
            totals.total_request_log_rows_persisted as f64,
        ),
        (
            "total_request_log_rows_dropped_queue_full",
            totals.total_request_log_rows_dropped_queue_full as f64,
        ),
        (
            "total_request_log_rows_dropped_flush_error",
            totals.total_request_log_rows_dropped_flush_error as f64,
        ),
    ];
    for (name, value) in counters {
        out.push(make_sample(
            sanitize_metric_name(name),
            value,
            base_labels,
            None,
            "counter",
        ));
    }
    if let Some(v) = totals.kv_cache_hit_rate_lifetime {
        out.push(make_sample(
            "kv_cache_hit_rate_lifetime".to_string(),
            v,
            base_labels,
            None,
            "gauge",
        ));
    }
    let gauges = [
        (
            "current_inflight_requests",
            totals.current_inflight_requests as f64,
        ),
        ("mean_inflight_requests", totals.mean_inflight_requests),
        (
            "current_inflight_input_tokens",
            totals.current_inflight_input_tokens as f64,
        ),
        (
            "mean_inflight_input_tokens",
            totals.mean_inflight_input_tokens,
        ),
        (
            "current_inflight_output_tokens",
            totals.current_inflight_output_tokens as f64,
        ),
        (
            "mean_inflight_output_tokens",
            totals.mean_inflight_output_tokens,
        ),
        (
            "current_inflight_first_to_fifth_token_requests",
            totals.current_inflight_first_to_fifth_token_requests as f64,
        ),
        (
            "mean_inflight_first_to_fifth_token_requests",
            totals.mean_inflight_first_to_fifth_token_requests,
        ),
        (
            "current_inflight_first_to_fifth_token_input_tokens",
            totals.current_inflight_first_to_fifth_token_input_tokens as f64,
        ),
        (
            "mean_inflight_first_to_fifth_token_input_tokens",
            totals.mean_inflight_first_to_fifth_token_input_tokens,
        ),
    ];
    for (name, value) in gauges {
        out.push(make_sample(
            sanitize_metric_name(name),
            value,
            base_labels,
            None,
            "gauge",
        ));
    }
    out
}

fn make_sample(
    name: String,
    value: f64,
    base_labels: Option<&HashMap<String, String>>,
    extra_labels: Option<HashMap<String, String>>,
    metric_type: &str,
) -> PrometheusSample {
    let mut labels = base_labels.cloned().unwrap_or_default();
    if let Some(extra) = extra_labels {
        labels.extend(extra);
    }
    let labels = labels
        .into_iter()
        .map(|(k, v)| (sanitize_label_name(k.as_str()), v))
        .collect();
    PrometheusSample {
        name,
        value,
        labels,
        metric_type: metric_type.to_string(),
    }
}

fn sanitize_metric_name(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        return "metric".to_string();
    }
    if out.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        out.insert(0, '_');
    }
    out
}

fn sanitize_label_name(value: &str) -> String {
    sanitize_metric_name(value)
}

fn metric_type_from_str(metric_type: &str) -> anyhow::Result<MetricType> {
    match metric_type {
        "counter" => Ok(MetricType::COUNTER),
        "gauge" => Ok(MetricType::GAUGE),
        _ => Err(anyhow::Error::msg(format!(
            "unsupported prometheus metric type: {metric_type}"
        ))),
    }
}

pub fn encode_samples_as_prometheus_text(samples: Vec<PrometheusSample>) -> anyhow::Result<String> {
    let mut grouped: BTreeMap<(String, String), Vec<PrometheusSample>> = BTreeMap::new();
    for sample in samples {
        grouped
            .entry((sample.name.clone(), sample.metric_type.clone()))
            .or_default()
            .push(sample);
    }

    let mut families = Vec::with_capacity(grouped.len());
    for ((name, metric_type), mut samples) in grouped {
        let parsed_type = metric_type_from_str(metric_type.as_str())?;
        samples.sort_by(|a, b| {
            let mut a_labels = a.labels.iter().collect::<Vec<_>>();
            let mut b_labels = b.labels.iter().collect::<Vec<_>>();
            a_labels.sort_by(|x, y| x.0.cmp(y.0).then_with(|| x.1.cmp(y.1)));
            b_labels.sort_by(|x, y| x.0.cmp(y.0).then_with(|| x.1.cmp(y.1)));
            a_labels.cmp(&b_labels)
        });

        let mut family = MetricFamily::new();
        family.set_name(name.clone());
        family.set_help(name);
        family.set_field_type(parsed_type);

        for sample in samples {
            let mut metric = Metric::new();
            let mut labels = sample.labels.into_iter().collect::<Vec<_>>();
            labels.sort_by(|a, b| a.0.cmp(&b.0));
            for (k, v) in labels {
                let mut pair = LabelPair::new();
                pair.set_name(sanitize_label_name(k.as_str()));
                pair.set_value(v);
                metric.mut_label().push(pair);
            }

            match parsed_type {
                MetricType::COUNTER => {
                    let mut counter = Counter::new();
                    counter.set_value(sample.value);
                    metric.set_counter(counter);
                }
                MetricType::GAUGE => {
                    let mut gauge = Gauge::new();
                    gauge.set_value(sample.value);
                    metric.set_gauge(gauge);
                }
                _ => {
                    return Err(anyhow::Error::msg(format!(
                        "unsupported prometheus type in encoder: {metric_type}"
                    )));
                }
            }

            family.mut_metric().push(metric);
        }
        families.push(family);
    }

    let mut out = Vec::new();
    TextEncoder::new()
        .encode(&families, &mut out)
        .map_err(|e| anyhow::Error::msg(e.to_string()))?;
    String::from_utf8(out).map_err(|e| anyhow::Error::msg(e.to_string()))
}
