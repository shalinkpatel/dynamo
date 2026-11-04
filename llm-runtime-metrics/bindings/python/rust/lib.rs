use numpy::PyReadonlyArray1;
use performance_metrics_core::{
    __private::{
        DEFAULT_BLOCK_SIZE as RS_DEFAULT_BLOCK_SIZE,
        DEFAULT_LOCAL_OUTPUT_DIR as RS_DEFAULT_LOCAL_OUTPUT_DIR,
        DEFAULT_REQUEST_LOG_BATCH_SIZE as RS_DEFAULT_REQUEST_LOG_BATCH_SIZE,
        DEFAULT_REQUEST_LOG_QUEUE_CAPACITY as RS_DEFAULT_REQUEST_LOG_QUEUE_CAPACITY,
        DistributionMetricHandle as RsDistributionMetricHandle, FileFormat as RsFileFormat,
        PerformanceMetricsRegistry as RsPerformanceMetricsRegistry,
        RateMetricHandle as RsRateMetricHandle, RatioMetricHandle as RsRatioMetricHandle,
        RequestMetricsOptions as RsRequestMetricsOptions,
        RequestTraceOptions as RsRequestTraceOptions, S3SinkConfig as RsS3SinkConfig,
    },
    AnonymizeConfig as RsAnonymizeConfig, MetricSeriesSnapshot as RsMetricSeriesSnapshot,
    ParquetCompactResult as RsParquetCompactResult, RequestFeatures as RsRequestFeatures,
    RequestMetricsFactory as RsRequestMetricsFactory,
    RequestMetricsRequest as RsRequestMetricsRequest, RoutedWorkers as RsRoutedWorkers,
    S3ParquetCompactConfig as RsS3ParquetCompactConfig, ServiceTier as RsServiceTier,
    compact_s3_parquet_to_local as RsCompactS3ParquetToLocal,
    hash_b10_request_id as rs_hash_b10_request_id,
};
use performance_metrics_core::{encode_samples_as_prometheus_text, flatten_snapshot};
use pyo3::types::{PyBytes, PyDict};
use pyo3::{
    exceptions::{PyKeyboardInterrupt, PyRuntimeError, PyValueError},
    prelude::*,
};
use std::collections::{BTreeMap, HashMap};
use std::str::FromStr;

fn default_anonymize_pseudonymize_columns() -> Vec<String> {
    RsAnonymizeConfig::published().columns_to_pseudonymize
}

fn default_anonymize_drop_columns() -> Vec<String> {
    RsAnonymizeConfig::published().columns_to_drop
}

#[pyclass(name = "RateMetric")]
pub struct PyRateMetric {
    inner: RsRateMetricHandle,
}

#[pymethods]
impl PyRateMetric {
    #[pyo3(signature = (count = 1))]
    fn record_count(&self, count: u64) -> PyResult<()> {
        self.inner
            .record_count(count)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pyclass(name = "DistributionMetric")]
pub struct PyDistributionMetric {
    inner: RsDistributionMetricHandle,
}

#[pymethods]
impl PyDistributionMetric {
    fn record_value(&self, value: f64) -> PyResult<()> {
        self.inner
            .record_value(value)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pyclass(name = "RatioMetric")]
pub struct PyRatioMetric {
    inner: RsRatioMetricHandle,
}

#[pymethods]
impl PyRatioMetric {
    fn record_ratio(&self, numerator: f64, denominator: f64) -> PyResult<()> {
        self.inner
            .record_ratio(numerator, denominator)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pyclass(name = "PerformanceMetricsRegistry")]
pub struct PyPerformanceMetricsRegistry {
    inner: RsPerformanceMetricsRegistry,
}

#[pymethods]
impl PyPerformanceMetricsRegistry {
    #[new]
    fn new() -> Self {
        Self {
            inner: RsPerformanceMetricsRegistry::default(),
        }
    }

    #[pyo3(signature = (name, quantiles = None, sample_period_seconds = 1.0, window_seconds = None))]
    fn new_rate_metric(
        &self,
        name: String,
        quantiles: Option<Vec<f64>>,
        sample_period_seconds: f64,
        window_seconds: Option<f64>,
    ) -> PyResult<PyRateMetric> {
        self.inner
            .new_rate_metric(
                name,
                quantiles.unwrap_or_default(),
                Some(sample_period_seconds),
                window_seconds,
            )
            .map(|inner| PyRateMetric { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (name, quantiles = None, window_seconds = None))]
    fn new_distribution_metric(
        &self,
        name: String,
        quantiles: Option<Vec<f64>>,
        window_seconds: Option<f64>,
    ) -> PyResult<PyDistributionMetric> {
        self.inner
            .new_distribution_metric(name, quantiles.unwrap_or_default(), window_seconds)
            .map(|inner| PyDistributionMetric { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (name, quantiles = None, window_seconds = None))]
    fn new_ratio_metric(
        &self,
        name: String,
        quantiles: Option<Vec<f64>>,
        window_seconds: Option<f64>,
    ) -> PyResult<PyRatioMetric> {
        self.inner
            .new_ratio_metric(name, quantiles.unwrap_or_default(), window_seconds)
            .map(|inner| PyRatioMetric { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn snapshot_all(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyDict::new(py);
        for (name, metric) in self.inner.snapshot_all() {
            out.set_item(name, Py::new(py, metric_snapshot_to_typed(metric))?)?;
        }
        Ok(out.into_any().unbind())
    }

    fn snapshot_all_series(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = self
            .inner
            .snapshot_all_series()
            .into_iter()
            .map(metric_series_snapshot_to_typed)
            .map(|series| Py::new(py, series))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(out.into_pyobject(py)?.unbind())
    }

    fn unregister_metric(&self, name: String) {
        self.inner.unregister_metric(name.as_str());
    }
}

#[pyclass(name = "RequestMetricsFactory")]
pub struct PyRequestMetricsFactory {
    inner: RsRequestMetricsFactory,
}

#[pyclass(name = "RequestMetrics")]
pub struct PyRequestMetricsRequest {
    inner: RsRequestMetricsRequest,
}

#[pyclass(name = "MetricSnapshot")]
#[derive(Clone)]
pub struct PyMetricSnapshot {
    #[pyo3(get)]
    kind: String,
    #[pyo3(get)]
    rate_per_second: Option<f64>,
    #[pyo3(get)]
    average: Option<f64>,
    #[pyo3(get)]
    quantiles: HashMap<String, f64>,
    #[pyo3(get)]
    numerator_sum: Option<f64>,
    #[pyo3(get)]
    denominator_sum: Option<f64>,
    #[pyo3(get)]
    ratio: Option<f64>,
}

#[pyclass(name = "RequestMetricsTotalsSnapshot")]
#[derive(Clone)]
pub struct PyRequestMetricsTotalsSnapshot {
    #[pyo3(get)]
    total_requests: u64,
    #[pyo3(get)]
    total_successful_requests: u64,
    #[pyo3(get)]
    total_pre_first_token_cancellations: u64,
    #[pyo3(get)]
    total_pre_20_token_cancellations: u64,
    #[pyo3(get)]
    total_mid_stream_cancellations: u64,
    #[pyo3(get)]
    total_input_tokens: u64,
    #[pyo3(get)]
    total_output_tokens: u64,
    #[pyo3(get)]
    total_net_new_input_tokens: u64,
    #[pyo3(get)]
    total_kv_cache_hits_tokens: u64,
    #[pyo3(get)]
    total_kv_cache_lookup_tokens: u64,
    #[pyo3(get)]
    total_itl_samples_recorded: u64,
    #[pyo3(get)]
    total_request_log_rows_persisted: u64,
    #[pyo3(get)]
    total_request_log_rows_dropped_queue_full: u64,
    #[pyo3(get)]
    total_request_log_rows_dropped_flush_error: u64,
    #[pyo3(get)]
    kv_cache_hit_rate_lifetime: Option<f64>,
    #[pyo3(get)]
    current_inflight_requests: u64,
    #[pyo3(get)]
    mean_inflight_requests: f64,
    #[pyo3(get)]
    current_inflight_input_tokens: u64,
    #[pyo3(get)]
    mean_inflight_input_tokens: f64,
    #[pyo3(get)]
    current_inflight_output_tokens: u64,
    #[pyo3(get)]
    mean_inflight_output_tokens: f64,
    #[pyo3(get)]
    current_inflight_first_to_fifth_token_requests: u64,
    #[pyo3(get)]
    mean_inflight_first_to_fifth_token_requests: f64,
    #[pyo3(get)]
    current_inflight_first_to_fifth_token_input_tokens: u64,
    #[pyo3(get)]
    mean_inflight_first_to_fifth_token_input_tokens: f64,
}

#[pyclass(name = "RequestMetricsSnapshot")]
pub struct PyRequestMetricsSnapshot {
    series: Vec<PyMetricSeriesSnapshot>,
    totals: PyRequestMetricsTotalsSnapshot,
}

#[pyclass(name = "CompactResult")]
#[derive(Clone)]
pub struct PyCompactResult {
    #[pyo3(get)]
    source_objects: usize,
    #[pyo3(get)]
    rows_written: usize,
    #[pyo3(get)]
    output_path: String,
    #[pyo3(get)]
    min_timestamp: u64,
}

#[pyclass(name = "MetricSeriesSnapshot")]
#[derive(Clone)]
pub struct PyMetricSeriesSnapshot {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    labels: BTreeMap<String, String>,
    snapshot: PyMetricSnapshot,
}

#[pyclass(name = "PrometheusSample")]
#[derive(Clone)]
pub struct PyPrometheusSample {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    value: f64,
    #[pyo3(get)]
    labels: HashMap<String, String>,
    #[pyo3(get)]
    metric_type: String,
}

#[derive(Clone)]
#[pyclass(frozen, name = "AtomicHistogram")]
pub struct PyAtomicHistogram {
    inner: performance_metrics_core::AtomicHistogram,
}

#[pymethods]
impl PyAtomicHistogram {
    #[new]
    fn new(upper_bounds: Vec<f64>) -> PyResult<Self> {
        performance_metrics_core::AtomicHistogram::new(upper_bounds)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    #[staticmethod]
    fn from_vm(min_exponent: i32, max_exponent: i32, buckets_per_decimal: u32) -> PyResult<Self> {
        performance_metrics_core::AtomicHistogram::from_vm(
            min_exponent,
            max_exponent,
            buckets_per_decimal,
        )
        .map(|inner| Self { inner })
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    #[staticmethod]
    fn from_ranges(bucket_ranges: Vec<(f64, f64)>) -> PyResult<Self> {
        performance_metrics_core::AtomicHistogram::from_ranges(bucket_ranges)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    fn observe(&self, amount: f64) {
        self.inner.observe(amount);
    }
    fn observe_many(&self, amount: f64, count: i64) {
        self.inner.observe_many(amount, count);
    }
    fn snapshot(&self) -> (Vec<u64>, u64, f64) {
        self.inner.snapshot()
    }
    fn is_shared(&self) -> bool {
        self.inner.is_shared()
    }
    fn bucket_len(&self) -> usize {
        self.inner.bucket_len()
    }
}

#[derive(Clone, Default)]
#[pyclass(frozen, name = "AtomicCounter")]
pub struct PyAtomicCounter {
    inner: performance_metrics_core::AtomicCounter,
}
#[pymethods]
impl PyAtomicCounter {
    #[new]
    fn new() -> Self {
        Self::default()
    }
    #[pyo3(signature = (amount = 1))]
    fn inc(&self, amount: u64) {
        self.inner.inc(amount);
    }
    fn get(&self) -> u64 {
        self.inner.get()
    }
}

#[pyclass(name = "StreamChunkMetrics")]
pub struct PyStreamChunkMetrics {
    inner: performance_metrics_core::StreamChunkMetrics,
}
#[pymethods]
impl PyStreamChunkMetrics {
    #[new]
    fn new(
        ttft: &PyAtomicHistogram,
        tpot: &PyAtomicHistogram,
        itl: &PyAtomicHistogram,
        output_tokens: &PyAtomicCounter,
    ) -> Self {
        Self {
            inner: performance_metrics_core::StreamChunkMetrics::new(
                &ttft.inner,
                &tpot.inner,
                &itl.inner,
                &output_tokens.inner,
            ),
        }
    }
    #[pyo3(signature = (token_count, ttft_seconds = None, chunk_seconds = None))]
    fn record(
        &self,
        token_count: u64,
        ttft_seconds: Option<f64>,
        chunk_seconds: Option<f64>,
    ) -> PyResult<()> {
        self.inner
            .record(token_count, ttft_seconds, chunk_seconds)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
    fn close(&mut self) {
        self.inner.close();
    }
}

#[pymethods]
impl PyRequestMetricsSnapshot {
    fn series(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = self
            .series
            .iter()
            .cloned()
            .map(|series| Py::new(py, series))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(out.into_pyobject(py)?.unbind())
    }

    fn totals(&self, py: Python<'_>) -> PyResult<Py<PyRequestMetricsTotalsSnapshot>> {
        Py::new(py, self.totals.clone())
    }
}

#[pymethods]
impl PyMetricSeriesSnapshot {
    fn snapshot(&self, py: Python<'_>) -> PyResult<Py<PyMetricSnapshot>> {
        Py::new(py, self.snapshot.clone())
    }
}

#[pymethods]
impl PyRequestMetricsFactory {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        block_size = RS_DEFAULT_BLOCK_SIZE,
        request_log_enabled = false,
        request_log_hash_salt = None,
        request_log_batch_size = RS_DEFAULT_REQUEST_LOG_BATCH_SIZE,
        request_log_queue_capacity = RS_DEFAULT_REQUEST_LOG_QUEUE_CAPACITY,
        local_file = true,
        request_log_output_dir = RS_DEFAULT_LOCAL_OUTPUT_DIR.to_string(),
        s3 = false,
        s3_bucket = None,
        s3_region = "us-east-1".to_string(),
        s3_endpoint = None,
        s3_access_key_id = None,
        s3_secret_access_key = None,
        s3_prefix = "llm-request-metrics".to_string(),
        file_format = "json".to_string(),
        model_name = Some("unset_model_name".to_string()),
        org_id = String::new(),
        metric_prefix = "request_metrics".to_string(),
        metrics_window_seconds = 60.0,
        metrics_quantiles = vec![0.5, 0.9, 0.99],
        itl_sample_rate = 0.05,
        s3_proxy_url = None,
        component_name = None
    ))]
    fn new(
        py: Python<'_>,
        block_size: usize,
        request_log_enabled: bool,
        request_log_hash_salt: Option<String>,
        request_log_batch_size: usize,
        request_log_queue_capacity: usize,
        local_file: bool,
        request_log_output_dir: String,
        s3: bool,
        s3_bucket: Option<String>,
        s3_region: String,
        s3_endpoint: Option<String>,
        s3_access_key_id: Option<String>,
        s3_secret_access_key: Option<String>,
        s3_prefix: String,
        file_format: String,
        model_name: Option<String>,
        org_id: String,
        metric_prefix: String,
        metrics_window_seconds: f64,
        metrics_quantiles: Vec<f64>,
        itl_sample_rate: f64,
        s3_proxy_url: Option<String>,
        component_name: Option<String>,
    ) -> PyResult<Self> {
        let kv_cache_hit_rate_quantiles = kv_cache_quantiles(metrics_quantiles.clone());

        let inner = RsRequestMetricsFactory::new(RsRequestMetricsOptions {
            trace: RsRequestTraceOptions {
                block_size,
                request_log_hash_salt,
                model_name: model_name.unwrap_or_else(|| "unset_model_name".to_string()),
                org_id,
                component_name,
                ..Default::default()
            },
            metric_prefix,
            request_quantiles: metrics_quantiles.clone(),
            request_sample_period_seconds: Some(1.0),
            request_window_seconds: Some(metrics_window_seconds),
            input_tokens_quantiles: metrics_quantiles.clone(),
            input_tokens_sample_period_seconds: Some(1.0),
            input_tokens_window_seconds: Some(metrics_window_seconds),
            ttft_quantiles: metrics_quantiles.clone(),
            ttft_window_seconds: Some(metrics_window_seconds),
            time_to_fifth_token_quantiles: metrics_quantiles.clone(),
            time_to_fifth_token_window_seconds: Some(metrics_window_seconds),
            first_to_fifth_token_itl_quantiles: metrics_quantiles.clone(),
            first_to_fifth_token_itl_window_seconds: Some(metrics_window_seconds),
            ttft_per_input_token_quantiles: metrics_quantiles.clone(),
            ttft_per_input_token_window_seconds: Some(metrics_window_seconds),
            itl_quantiles: vec![0.1, 0.5, 0.9, 0.99, 0.999],
            itl_window_seconds: Some(metrics_window_seconds),
            pre_first_token_cancellation_quantiles: Vec::new(),
            pre_first_token_cancellation_sample_period_seconds: Some(1.0),
            pre_first_token_cancellation_window_seconds: Some(metrics_window_seconds),
            pre_20_token_cancellation_quantiles: Vec::new(),
            pre_20_token_cancellation_sample_period_seconds: Some(1.0),
            pre_20_token_cancellation_window_seconds: Some(metrics_window_seconds),
            mid_stream_cancellation_quantiles: Vec::new(),
            mid_stream_cancellation_sample_period_seconds: Some(1.0),
            mid_stream_cancellation_window_seconds: Some(metrics_window_seconds),
            successful_request_quantiles: Vec::new(),
            successful_request_sample_period_seconds: Some(1.0),
            successful_request_window_seconds: Some(metrics_window_seconds),
            kv_cache_hit_rate_quantiles,
            kv_cache_hit_rate_window_seconds: Some(metrics_window_seconds),
            itl_sample_rate,
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let factory = Self { inner };
        factory.configure_request_log(
            py,
            request_log_enabled,
            request_log_batch_size,
            request_log_queue_capacity,
            local_file,
            request_log_output_dir,
            s3,
            s3_bucket,
            s3_region,
            s3_endpoint,
            s3_access_key_id,
            s3_secret_access_key,
            s3_prefix,
            file_format,
            s3_proxy_url,
        )?;
        Ok(factory)
    }

    /// Turn request logging on/off or re-point it without resetting metrics.
    /// See the README; raises ValueError and keeps the current setting if invalid.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        request_log_enabled,
        request_log_batch_size = RS_DEFAULT_REQUEST_LOG_BATCH_SIZE,
        request_log_queue_capacity = RS_DEFAULT_REQUEST_LOG_QUEUE_CAPACITY,
        local_file = true,
        request_log_output_dir = RS_DEFAULT_LOCAL_OUTPUT_DIR.to_string(),
        s3 = false,
        s3_bucket = None,
        s3_region = "us-east-1".to_string(),
        s3_endpoint = None,
        s3_access_key_id = None,
        s3_secret_access_key = None,
        s3_prefix = "llm-request-metrics".to_string(),
        file_format = "json".to_string(),
        s3_proxy_url = None
    ))]
    fn configure_request_log(
        &self,
        py: Python<'_>,
        request_log_enabled: bool,
        request_log_batch_size: usize,
        request_log_queue_capacity: usize,
        local_file: bool,
        request_log_output_dir: String,
        s3: bool,
        s3_bucket: Option<String>,
        s3_region: String,
        s3_endpoint: Option<String>,
        s3_access_key_id: Option<String>,
        s3_secret_access_key: Option<String>,
        s3_prefix: String,
        file_format: String,
        s3_proxy_url: Option<String>,
    ) -> PyResult<()> {
        let local_output_dir = local_file.then(|| std::path::PathBuf::from(request_log_output_dir));

        let file_format = match file_format.as_str() {
            "json" => RsFileFormat::Json,
            "parquet" => RsFileFormat::Parquet,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Invalid file_format: {}. Must be 'json' or 'parquet'",
                    file_format
                )));
            }
        };

        let s3_config = if s3 {
            let bucket = s3_bucket
                .ok_or_else(|| PyValueError::new_err("s3=True requires s3_bucket to be set"))?;
            Some(RsS3SinkConfig {
                bucket,
                region: s3_region,
                endpoint: s3_endpoint,
                proxy_url: s3_proxy_url,
                access_key_id: s3_access_key_id,
                secret_access_key: s3_secret_access_key,
                prefix: s3_prefix,
            })
        } else {
            None
        };

        let options = RsRequestTraceOptions {
            request_log_enabled,
            request_log_batch_size,
            request_log_queue_capacity,
            local_output_dir,
            s3_config,
            file_format,
            ..Default::default()
        };
        let inner = self.inner.clone();
        py.allow_threads(move || inner.configure_request_log(&options))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter]
    fn request_log_enabled(&self) -> bool {
        self.inner.request_log_enabled()
    }

    #[pyo3(signature = (input_token_ids, features = 0, session_id = None, service_tier = None, dp_rank = None, b10_request_id = None))]
    fn new_request(
        &self,
        input_token_ids: &Bound<'_, PyAny>,
        features: u8,
        session_id: Option<String>,
        service_tier: Option<String>,
        dp_rank: Option<i64>,
        b10_request_id: Option<String>,
    ) -> PyResult<PyRequestMetricsRequest> {
        let tier = match service_tier.as_deref() {
            Some(value) => Some(
                RsServiceTier::from_str(value).map_err(|e| PyValueError::new_err(e.to_string()))?,
            ),
            None => None,
        };
        let dp_rank = match dp_rank {
            Some(value) if (0..=31).contains(&value) => Some(value as u8),
            // Values outside 0..=31 (negative, >= 32) normalize to None.
            _ => None,
        };
        // Fast path: when logging is disabled, just get the length without copying.
        // shortcut: a request that lands in the instant logging turns on can take
        // this path and be logged without input hashes; fine for an operator toggle.
        if !self.inner.request_log_enabled() {
            // this branch is around 100-200x faster for 50k tokens
            // since we don't need to create a rust vec to get len() if log is disabled
            let count = extract_length(input_token_ids)?;
            return Ok(PyRequestMetricsRequest {
                inner: self
                    .inner
                    .new_request_labeled(
                        None,
                        count,
                        RsRequestFeatures::from(features),
                        session_id.as_deref(),
                        b10_request_id.as_deref(),
                        tier,
                        dp_rank,
                    )
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            });
        }

        // Full path: extract and copy tokens for hashing
        let input_token_ids = extract_list_or_numpy_u32(input_token_ids)?;
        Ok(PyRequestMetricsRequest {
            inner: self
                .inner
                .new_request_labeled(
                    Some(input_token_ids.as_slice()),
                    input_token_ids.len(),
                    RsRequestFeatures::from(features),
                    session_id.as_deref(),
                    b10_request_id.as_deref(),
                    tier,
                    dp_rank,
                )
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        })
    }

    fn snapshot(&self) -> PyRequestMetricsSnapshot {
        request_metrics_snapshot_to_typed(self.inner.snapshot())
    }

    #[pyo3(signature = (base_labels = None))]
    fn prometheus_samples(
        &self,
        base_labels: Option<HashMap<String, String>>,
    ) -> Vec<PyPrometheusSample> {
        flatten_snapshot(self.inner.snapshot(), base_labels.as_ref())
            .into_iter()
            .map(|s| PyPrometheusSample {
                name: s.name,
                value: s.value,
                labels: s.labels,
                metric_type: s.metric_type,
            })
            .collect()
    }

    #[pyo3(signature = (base_labels = None))]
    fn prometheus_strfmt(&self, base_labels: Option<HashMap<String, String>>) -> PyResult<String> {
        let samples = flatten_snapshot(self.inner.snapshot(), base_labels.as_ref());
        encode_samples_as_prometheus_text(samples).map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pymethods]
impl PyRequestMetricsRequest {
    #[pyo3(signature = (token_ids, cached_tokens = None, is_diff = true, speculation_ratio = None))]
    fn record_tokens(
        &mut self,
        token_ids: &Bound<'_, PyAny>,
        cached_tokens: Option<u64>,
        is_diff: bool,
        speculation_ratio: Option<f64>,
    ) -> PyResult<()> {
        let token_ids = extract_list_or_numpy_u32(token_ids)?;
        self.inner.record_tokens(
            token_ids.as_slice(),
            cached_tokens,
            is_diff,
            speculation_ratio,
        );
        Ok(())
    }

    fn success(&mut self) {
        self.inner.success();
    }

    fn cancel(&mut self) {
        self.inner.cancel();
    }

    /// Record the router's chosen workers on the log row; ignored once the row is written.
    /// Each call replaces all four values, so omitted arguments are stored as null.
    #[pyo3(signature = (prefill_worker_id = None, prefill_dp_rank = None, decode_worker_id = None, decode_dp_rank = None))]
    fn set_routed_workers(
        &mut self,
        prefill_worker_id: Option<u64>,
        prefill_dp_rank: Option<u32>,
        decode_worker_id: Option<u64>,
        decode_dp_rank: Option<u32>,
    ) {
        self.inner.set_routed_workers(RsRoutedWorkers {
            prefill_worker_id,
            prefill_dp_rank,
            decode_worker_id,
            decode_dp_rank,
        });
    }

    fn total_hashes<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyBytes>> {
        self.inner
            .total_hashes()
            .iter()
            .map(|hash| PyBytes::new(py, hash))
            .collect()
    }

    fn features(&self) -> u8 {
        self.inner.features().bits()
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    s3_bucket,
    output_path,
    s3_region = "us-east-1".to_string(),
    s3_prefix = String::new(),
    s3_endpoint = None,
    s3_access_key_id = None,
    s3_secret_access_key = None,
    max_concurrent_downloads = 128,
    anonymize = false,
    s3_proxy_url = None
))]
fn compact_s3_parquet_to_local(
    py: Python<'_>,
    s3_bucket: String,
    output_path: String,
    s3_region: String,
    s3_prefix: String,
    s3_endpoint: Option<String>,
    s3_access_key_id: Option<String>,
    s3_secret_access_key: Option<String>,
    max_concurrent_downloads: usize,
    anonymize: bool,
    s3_proxy_url: Option<String>,
) -> PyResult<PyCompactResult> {
    let anonymize_config = if anonymize {
        Some(RsAnonymizeConfig::published())
    } else {
        None
    };

    let result = py
        .allow_threads(move || {
            RsCompactS3ParquetToLocal(RsS3ParquetCompactConfig {
                s3: RsS3SinkConfig {
                    bucket: s3_bucket,
                    region: s3_region,
                    endpoint: s3_endpoint,
                    proxy_url: s3_proxy_url,
                    access_key_id: s3_access_key_id,
                    secret_access_key: s3_secret_access_key,
                    prefix: s3_prefix,
                },
                output_path: std::path::PathBuf::from(output_path),
                max_concurrent_downloads,
                anonymize: anonymize_config,
            })
        })
        .map_err(|e| {
            let msg = e.to_string();
            if msg.contains("interrupted by Ctrl+C") {
                PyKeyboardInterrupt::new_err(msg)
            } else {
                PyValueError::new_err(msg)
            }
        })?;

    Ok(compact_result_to_typed(result))
}

#[pyfunction]
#[pyo3(signature = (
    input_path,
    output_path,
    columns_to_pseudonymize = default_anonymize_pseudonymize_columns(),
    columns_to_drop = default_anonymize_drop_columns(),
    min_timestamp = 0
))]
fn anonymize_parquet(
    input_path: String,
    output_path: String,
    columns_to_pseudonymize: Vec<String>,
    columns_to_drop: Vec<String>,
    min_timestamp: u64,
) -> PyResult<u64> {
    let columns_to_pseudonymize_refs: Vec<&str> =
        columns_to_pseudonymize.iter().map(|s| s.as_str()).collect();
    let columns_to_drop_refs: Vec<&str> = columns_to_drop.iter().map(|s| s.as_str()).collect();

    performance_metrics_core::anonymize_parquet_file(
        std::path::Path::new(&input_path),
        std::path::Path::new(&output_path),
        &columns_to_pseudonymize_refs,
        &columns_to_drop_refs,
        min_timestamp,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Hash a Baseten request id exactly as request-log rows store it.
#[pyfunction]
fn hash_b10_request_id(b10_request_id: &str) -> String {
    rs_hash_b10_request_id(b10_request_id)
}

#[pymodule]
fn _core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPerformanceMetricsRegistry>()?;
    m.add_class::<PyRateMetric>()?;
    m.add_class::<PyDistributionMetric>()?;
    m.add_class::<PyRatioMetric>()?;
    m.add_class::<PyRequestMetricsFactory>()?;
    m.add_class::<PyRequestMetricsRequest>()?;
    m.add_class::<PyMetricSnapshot>()?;
    m.add_class::<PyMetricSeriesSnapshot>()?;
    m.add_class::<PyRequestMetricsTotalsSnapshot>()?;
    m.add_class::<PyRequestMetricsSnapshot>()?;
    m.add_class::<PyPrometheusSample>()?;
    m.add_class::<PyAtomicHistogram>()?;
    m.add_class::<PyAtomicCounter>()?;
    m.add_class::<PyStreamChunkMetrics>()?;
    m.add_class::<PyCompactResult>()?;
    m.add_function(wrap_pyfunction!(compact_s3_parquet_to_local, m)?)?;
    m.add_function(wrap_pyfunction!(anonymize_parquet, m)?)?;
    m.add_function(wrap_pyfunction!(hash_b10_request_id, m)?)?;
    m.add("REQUEST_FEATURE_NONE", RsRequestFeatures::NONE.bits())?;
    m.add(
        "REQUEST_FEATURE_XGRAMMAR",
        RsRequestFeatures::XGRAMMAR.bits(),
    )?;
    m.add("REQUEST_FEATURE_TOOLS", RsRequestFeatures::TOOLS.bits())?;
    m.add("REQUEST_FEATURE_IMAGE", RsRequestFeatures::IMAGE.bits())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

/// Extract just the length from a Python list or NumPy array without copying data.
fn extract_length(token_ids: &Bound<'_, PyAny>) -> PyResult<usize> {
    // Try NumPy array first (fastest path)
    if let Ok(arr) = token_ids.extract::<PyReadonlyArray1<'_, u32>>() {
        return arr.len().map_err(|e| PyValueError::new_err(e.to_string()));
    }
    if let Ok(arr) = token_ids.extract::<PyReadonlyArray1<'_, i64>>() {
        return arr.len().map_err(|e| PyValueError::new_err(e.to_string()));
    }
    // Fall back to Python list
    if let Ok(list) = token_ids.downcast::<pyo3::types::PyList>() {
        return Ok(list.len());
    }
    Err(PyValueError::new_err(
        "token_ids must be list[int] or numpy.ndarray (uint32/int64)",
    ))
}

fn extract_list_or_numpy_u32(token_ids: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
    if let Ok(arr) = token_ids.extract::<PyReadonlyArray1<'_, u32>>() {
        return arr
            .as_slice()
            .map(|s| s.to_vec())
            .map_err(|e| PyValueError::new_err(e.to_string()));
    }

    if let Ok(arr) = token_ids.extract::<PyReadonlyArray1<'_, i64>>() {
        let values = arr
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut out = Vec::with_capacity(values.len());
        for &v in values {
            let u = u32::try_from(v)
                .map_err(|_| PyValueError::new_err("numpy int64 values must fit in u32"))?;
            out.push(u);
        }
        return Ok(out);
    }

    if let Ok(ids) = token_ids.extract::<Vec<u32>>() {
        return Ok(ids);
    }
    Err(PyValueError::new_err(
        "token_ids must be list[int] or numpy.ndarray (uint32/int64)",
    ))
}

fn metric_snapshot_to_typed(m: performance_metrics_core::MetricSnapshot) -> PyMetricSnapshot {
    PyMetricSnapshot {
        kind: match m.kind {
            performance_metrics_core::MetricKind::Rate => "rate".to_string(),
            performance_metrics_core::MetricKind::Distribution => "distribution".to_string(),
            performance_metrics_core::MetricKind::Ratio => "ratio".to_string(),
        },
        rate_per_second: m.rate_per_second,
        average: m.average,
        quantiles: m.quantiles,
        numerator_sum: m.numerator_sum,
        denominator_sum: m.denominator_sum,
        ratio: m.ratio,
    }
}

fn compact_result_to_typed(result: RsParquetCompactResult) -> PyCompactResult {
    PyCompactResult {
        source_objects: result.source_objects,
        rows_written: result.rows_written,
        output_path: result.output_path.display().to_string(),
        min_timestamp: result.min_timestamp,
    }
}

fn metric_series_snapshot_to_typed(series: RsMetricSeriesSnapshot) -> PyMetricSeriesSnapshot {
    PyMetricSeriesSnapshot {
        name: series.name,
        labels: series.labels,
        snapshot: metric_snapshot_to_typed(series.snapshot),
    }
}

fn request_metrics_snapshot_to_typed(
    snapshot: performance_metrics_core::RequestMetricsSnapshot,
) -> PyRequestMetricsSnapshot {
    PyRequestMetricsSnapshot {
        series: snapshot
            .series
            .into_iter()
            .map(metric_series_snapshot_to_typed)
            .collect(),
        totals: totals_to_typed(snapshot.totals),
    }
}

fn totals_to_typed(
    totals: performance_metrics_core::RequestMetricsTotalsSnapshot,
) -> PyRequestMetricsTotalsSnapshot {
    PyRequestMetricsTotalsSnapshot {
        total_requests: totals.total_requests,
        total_successful_requests: totals.total_successful_requests,
        total_pre_first_token_cancellations: totals.total_pre_first_token_cancellations,
        total_pre_20_token_cancellations: totals.total_pre_20_token_cancellations,
        total_mid_stream_cancellations: totals.total_mid_stream_cancellations,
        total_input_tokens: totals.total_input_tokens,
        total_output_tokens: totals.total_output_tokens,
        total_net_new_input_tokens: totals.total_net_new_input_tokens,
        total_kv_cache_hits_tokens: totals.total_kv_cache_hits_tokens,
        total_kv_cache_lookup_tokens: totals.total_kv_cache_lookup_tokens,
        total_itl_samples_recorded: totals.total_itl_samples_recorded,
        total_request_log_rows_persisted: totals.total_request_log_rows_persisted,
        total_request_log_rows_dropped_queue_full: totals.total_request_log_rows_dropped_queue_full,
        total_request_log_rows_dropped_flush_error: totals
            .total_request_log_rows_dropped_flush_error,
        kv_cache_hit_rate_lifetime: totals.kv_cache_hit_rate_lifetime,
        current_inflight_requests: totals.current_inflight_requests,
        mean_inflight_requests: totals.mean_inflight_requests,
        current_inflight_input_tokens: totals.current_inflight_input_tokens,
        mean_inflight_input_tokens: totals.mean_inflight_input_tokens,
        current_inflight_output_tokens: totals.current_inflight_output_tokens,
        mean_inflight_output_tokens: totals.mean_inflight_output_tokens,
        current_inflight_first_to_fifth_token_requests: totals
            .current_inflight_first_to_fifth_token_requests,
        mean_inflight_first_to_fifth_token_requests: totals
            .mean_inflight_first_to_fifth_token_requests,
        current_inflight_first_to_fifth_token_input_tokens: totals
            .current_inflight_first_to_fifth_token_input_tokens,
        mean_inflight_first_to_fifth_token_input_tokens: totals
            .mean_inflight_first_to_fifth_token_input_tokens,
    }
}

fn kv_cache_quantiles(mut base: Vec<f64>) -> Vec<f64> {
    base.extend([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]);
    base.retain(|q| q.is_finite());
    for q in &mut base {
        *q = q.clamp(0.0, 1.0);
    }
    base.sort_unstable_by(f64::total_cmp);
    base.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    base
}

#[cfg(test)]
mod tests {
    use super::*;
    use performance_metrics_core::{
        __private::RequestMetricsOptions as RsRequestMetricsOptions,
        RequestMetricsFactory as RsRequestMetricsFactory,
    };
    use std::collections::{HashMap, HashSet};

    #[test]
    fn request_metrics_factory_exports_under_10k_series() {
        let factory = RsRequestMetricsFactory::new(RsRequestMetricsOptions::default()).unwrap();
        let mut req = factory
            .new_request_labeled(
                Some(&[1, 2, 3, 4]),
                4,
                RsRequestFeatures::NONE,
                None,
                None,
                None,
                None,
            )
            .unwrap();
        req.record_tokens(&[10, 11, 12], Some(2), true, None);
        req.success();

        let mut base_labels = HashMap::new();
        base_labels.insert("engine".to_string(), "vllm".to_string());
        base_labels.insert("model".to_string(), "demo".to_string());

        let samples = flatten_snapshot(factory.snapshot(), Some(&base_labels));

        let mut series = HashSet::new();
        for sample in samples {
            let mut labels = sample.labels.into_iter().collect::<Vec<_>>();
            labels.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            series.insert((sample.name, labels));
        }

        assert!(series.len() < 10_000, "series count was {}", series.len());
    }

    #[test]
    fn request_metrics_feature_exports_use_feature_label() {
        let factory = RsRequestMetricsFactory::new(RsRequestMetricsOptions::default()).unwrap();
        let mut req = factory
            .new_request_labeled(
                Some(&[1, 2, 3, 4]),
                4,
                RsRequestFeatures::TOOLS | RsRequestFeatures::IMAGE,
                None,
                None,
                None,
                None,
            )
            .unwrap();
        req.record_tokens(&[10, 11, 12], Some(2), true, None);
        req.record_tokens(&[13], None, true, None);
        req.success();

        let mut base_labels = HashMap::new();
        base_labels.insert("engine".to_string(), "vllm".to_string());

        let samples = flatten_snapshot(factory.snapshot(), Some(&base_labels));

        let mut series = HashSet::new();
        for sample in samples {
            let mut labels = sample.labels.into_iter().collect::<Vec<_>>();
            labels.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            series.insert((sample.name, labels));
        }

        assert!(series.contains(&(
            "request_metrics_ttft_ms_average".to_string(),
            vec![
                ("engine".to_string(), "vllm".to_string()),
                ("feature".to_string(), "all".to_string()),
            ],
        )));
        assert!(series.contains(&(
            "request_metrics_ttft_ms_average".to_string(),
            vec![
                ("engine".to_string(), "vllm".to_string()),
                ("feature".to_string(), "tools".to_string()),
            ],
        )));
        assert!(series.contains(&(
            "request_metrics_ttft_ms_average".to_string(),
            vec![
                ("engine".to_string(), "vllm".to_string()),
                ("feature".to_string(), "image".to_string()),
            ],
        )));
        assert!(series.contains(&(
            "request_metrics_ttft_ms_average".to_string(),
            vec![
                ("engine".to_string(), "vllm".to_string()),
                ("feature".to_string(), "none".to_string()),
            ],
        )));
        assert!(series.contains(&(
            "request_metrics_request_rate_per_second".to_string(),
            vec![
                ("engine".to_string(), "vllm".to_string()),
                ("feature".to_string(), "all".to_string()),
            ],
        )));
        assert!(series.contains(&(
            "request_metrics_request_rate_per_second".to_string(),
            vec![
                ("engine".to_string(), "vllm".to_string()),
                ("feature".to_string(), "tools".to_string()),
            ],
        )));
        assert!(series.contains(&(
            "request_metrics_request_rate_per_second".to_string(),
            vec![
                ("engine".to_string(), "vllm".to_string()),
                ("feature".to_string(), "image".to_string()),
            ],
        )));
        assert!(series.contains(&(
            "request_metrics_request_rate_per_second".to_string(),
            vec![
                ("engine".to_string(), "vllm".to_string()),
                ("feature".to_string(), "none".to_string()),
            ],
        )));
        assert!(!series.iter().any(|(name, _)| name.contains("_with_tools")));
        assert!(!series.iter().any(|(name, _)| name.contains("_with_image")));
    }
}
