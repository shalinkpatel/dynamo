mod compaction;
mod metrics;
mod request_trace;

pub use compaction::{
    AnonymizeConfig, ParquetCompactResult, S3ParquetCompactConfig, anonymize_parquet_file,
    compact_s3_parquet_to_local, compact_s3_parquet_to_local_async,
};
pub use metrics::{MetricKind, MetricSeriesSnapshot, MetricSnapshot};
pub use request_trace::{
    FileFormat, HashBytes, RequestFeatures, RequestMetricsFactory, RequestMetricsRequest,
    RequestMetricsSnapshot, RequestMetricsTotalsSnapshot, RoutedWorkers, ServiceTier,
    hash_b10_request_id,
};

#[doc(hidden)]
pub mod __private {
    pub use crate::metrics::{
        DistributionMetricHandle, MetricLabels, PerformanceMetricsRegistry, RateMetricHandle,
        RatioMetricHandle,
    };
    pub use crate::request_trace::{
        DEFAULT_BLOCK_SIZE, DEFAULT_LOCAL_OUTPUT_DIR, DEFAULT_REQUEST_LOG_BATCH_SIZE,
        DEFAULT_REQUEST_LOG_QUEUE_CAPACITY, FileFormat, RequestLogEntry, RequestMetricsOptions,
        RequestTrace, RequestTraceFactory, RequestTraceOptions, S3SinkConfig,
    };
}

mod stream;
pub use stream::{AtomicCounter, AtomicHistogram, StreamChunkMetrics};

mod export;
pub use export::{PrometheusSample, encode_samples_as_prometheus_text, flatten_snapshot};

#[cfg(test)]
mod config_tests {
    #[test]
    fn native_config_preserves_defaults_and_trace_options() {
        let options: super::__private::RequestMetricsOptions = serde_json::from_str(r#"{"metric_prefix":"processor","trace":{"model_name":"model","local_output_dir":null,"file_format":"parquet"},"ttft_window_seconds":60}"#).unwrap();
        assert_eq!(options.metric_prefix, "processor");
        assert_eq!(options.trace.model_name, "model");
        assert_eq!(options.trace.local_output_dir, None);
        assert_eq!(options.trace.file_format, super::FileFormat::Parquet);
        assert_eq!(options.ttft_window_seconds, Some(60.0));
        assert_eq!(
            options.itl_sample_rate,
            super::__private::RequestMetricsOptions::default().itl_sample_rate
        );
    }
}
