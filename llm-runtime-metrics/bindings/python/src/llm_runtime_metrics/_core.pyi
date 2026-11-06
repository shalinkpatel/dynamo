from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

# Token ids accepted by the bindings. The Rust side takes `PyAny` and tries, in
# order, a zero-copy `uint32` numpy array, an `int64` numpy array, then a
# Python list of ints. numpy is optional at runtime, hence the guarded import.
_TokenIds = Sequence[int] | npt.NDArray[np.uint32] | npt.NDArray[np.int64]

REQUEST_FEATURE_NONE: Final[int]
REQUEST_FEATURE_XGRAMMAR: Final[int]
REQUEST_FEATURE_TOOLS: Final[int]
REQUEST_FEATURE_IMAGE: Final[int]
__version__: Final[str]

class RateMetric:
    def record_count(self, count: int = 1) -> None: ...

class DistributionMetric:
    def record_value(self, value: float) -> None: ...

class RatioMetric:
    def record_ratio(self, numerator: float, denominator: float) -> None: ...

class PerformanceMetricsRegistry:
    def __init__(self) -> None: ...
    def new_rate_metric(
        self,
        name: str,
        quantiles: Sequence[float] | None = None,
        sample_period_seconds: float = 1.0,
        window_seconds: float | None = None,
    ) -> RateMetric: ...
    def new_distribution_metric(
        self,
        name: str,
        quantiles: Sequence[float] | None = None,
        window_seconds: float | None = None,
    ) -> DistributionMetric: ...
    def new_ratio_metric(
        self,
        name: str,
        quantiles: Sequence[float] | None = None,
        window_seconds: float | None = None,
    ) -> RatioMetric: ...
    def snapshot_all(self) -> dict[str, MetricSnapshot]: ...
    def snapshot_all_series(self) -> list[MetricSeriesSnapshot]: ...
    def unregister_metric(self, name: str) -> None: ...

class MetricSnapshot:
    kind: str
    rate_per_second: float | None
    average: float | None
    quantiles: dict[str, float]
    numerator_sum: float | None
    denominator_sum: float | None
    ratio: float | None

class RequestMetricsTotalsSnapshot:
    total_requests: int
    total_successful_requests: int
    total_pre_first_token_cancellations: int
    total_pre_20_token_cancellations: int
    total_mid_stream_cancellations: int
    total_input_tokens: int
    total_output_tokens: int
    total_net_new_input_tokens: int
    total_kv_cache_hits_tokens: int
    total_kv_cache_lookup_tokens: int
    total_itl_samples_recorded: int
    total_request_log_rows_persisted: int
    total_request_log_rows_dropped_queue_full: int
    total_request_log_rows_dropped_flush_error: int
    kv_cache_hit_rate_lifetime: float | None
    current_inflight_requests: int
    mean_inflight_requests: float
    current_inflight_input_tokens: int
    mean_inflight_input_tokens: float
    current_inflight_output_tokens: int
    mean_inflight_output_tokens: float
    current_inflight_first_to_fifth_token_requests: int
    mean_inflight_first_to_fifth_token_requests: float
    current_inflight_first_to_fifth_token_input_tokens: int
    mean_inflight_first_to_fifth_token_input_tokens: float

class MetricSeriesSnapshot:
    name: str
    labels: dict[str, str]
    def snapshot(self) -> MetricSnapshot: ...

class RequestMetricsSnapshot:
    def series(self) -> list[MetricSeriesSnapshot]: ...
    def totals(self) -> RequestMetricsTotalsSnapshot: ...

class PrometheusSample:
    name: str
    value: float
    labels: dict[str, str]
    metric_type: str

class CompactResult:
    source_objects: int
    rows_written: int
    output_path: str
    min_timestamp: int

class AtomicHistogram:
    def __init__(self, upper_bounds: Sequence[float]) -> None: ...
    @staticmethod
    def from_ranges(
        bucket_ranges: Sequence[tuple[float, float]],
    ) -> AtomicHistogram: ...
    @staticmethod
    def from_vm(
        min_exponent: int,
        max_exponent: int,
        buckets_per_decimal: int,
    ) -> AtomicHistogram: ...
    def observe(self, amount: float) -> None: ...
    def observe_many(self, amount: float, count: int) -> None: ...
    def snapshot(self) -> tuple[list[int], int, float]: ...
    def bucket_len(self) -> int: ...
    def is_shared(self) -> bool: ...

class AtomicCounter:
    def __init__(self) -> None: ...
    def inc(self, amount: int = 1) -> None: ...
    def get(self) -> int: ...

class StreamChunkMetrics:
    def __init__(
        self,
        ttft: AtomicHistogram,
        tpot: AtomicHistogram,
        itl: AtomicHistogram,
        output_tokens: AtomicCounter,
    ) -> None: ...
    def record(
        self,
        token_count: int,
        ttft_seconds: float | None = None,
        chunk_seconds: float | None = None,
    ) -> None: ...
    def close(self) -> None: ...

class RequestMetricsFactory:
    def __init__(
        self,
        block_size: int = 64,
        request_log_enabled: bool = False,
        request_log_hash_salt: str | None = None,
        request_log_batch_size: int = 2048,
        request_log_queue_capacity: int = 4096,
        local_file: bool = True,
        request_log_output_dir: str = "/tmp/records",
        s3: bool = False,
        s3_bucket: str | None = None,
        s3_region: str = "us-east-1",
        s3_endpoint: str | None = None,
        s3_access_key_id: str | None = None,
        s3_secret_access_key: str | None = None,
        s3_prefix: str = "records",
        file_format: str = "json",
        model_name: str | None = "unset_model_name",
        org_id: str = "",
        metric_prefix: str = "request_metrics",
        metrics_window_seconds: float = 60.0,
        metrics_quantiles: Sequence[float] = (0.5, 0.9, 0.99),
        itl_sample_rate: float = 0.05,
        s3_proxy_url: str | None = None,
        component_name: str | None = None,
    ) -> None: ...
    def configure_request_log(
        self,
        request_log_enabled: bool,
        request_log_batch_size: int = 2048,
        request_log_queue_capacity: int = 4096,
        local_file: bool = True,
        request_log_output_dir: str = "/tmp/records",
        s3: bool = False,
        s3_bucket: str | None = None,
        s3_region: str = "us-east-1",
        s3_endpoint: str | None = None,
        s3_access_key_id: str | None = None,
        s3_secret_access_key: str | None = None,
        s3_prefix: str = "llm-request-metrics",
        file_format: str = "json",
        s3_proxy_url: str | None = None,
    ) -> None:
        """Turn request logging on/off or re-point it; see the README."""
    @property
    def request_log_enabled(self) -> bool: ...
    def new_request(
        self,
        input_token_ids: _TokenIds,
        features: int = 0,
        session_id: str | None = None,
        service_tier: str | None = None,
        dp_rank: int | None = None,
        b10_request_id: str | None = None,
        org_id: str | None = None,
    ) -> RequestMetrics: ...
    def snapshot(self) -> RequestMetricsSnapshot: ...
    def prometheus_samples(
        self,
        base_labels: Mapping[str, str] | None = None,
    ) -> list[PrometheusSample]: ...
    def prometheus_strfmt(
        self, base_labels: Mapping[str, str] | None = None
    ) -> str: ...

class RequestMetrics:
    def record_tokens(
        self,
        token_ids: _TokenIds,
        cached_tokens: int | None = None,
        is_diff: bool = True,
        speculation_ratio: float | None = None,
    ) -> None: ...
    def success(self) -> None: ...
    def cancel(self) -> None: ...
    def set_routed_workers(
        self,
        prefill_worker_id: int | None = None,
        prefill_dp_rank: int | None = None,
        decode_worker_id: int | None = None,
        decode_dp_rank: int | None = None,
    ) -> None: ...
    def total_hashes(self) -> list[bytes]: ...
    def features(self) -> int: ...

def compact_s3_parquet_to_local(
    s3_bucket: str,
    output_path: str,
    s3_region: str = "us-east-1",
    s3_prefix: str = "",
    s3_endpoint: str | None = None,
    s3_access_key_id: str | None = None,
    s3_secret_access_key: str | None = None,
    max_concurrent_downloads: int = 128,
    anonymize: bool = False,
    s3_proxy_url: str | None = None,
) -> CompactResult: ...
def anonymize_parquet(
    input_path: str,
    output_path: str,
    columns_to_pseudonymize: Sequence[str] = [
        "provided_session_id",
        "request_org_id",
        "org_id",
        "total_hashes",
        "prefill_worker_id",
        "decode_worker_id",
    ],
    columns_to_drop: Sequence[str] = ["request_org_id", "org_id", "b10_request_id"],
    min_timestamp: int = 0,
) -> int: ...
def hash_b10_request_id(b10_request_id: str) -> str: ...
