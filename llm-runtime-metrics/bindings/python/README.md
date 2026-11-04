# llm-runtime-metrics (Python)

Python bindings for request metrics and Prometheus export.

Install from PyPI as `llm-runtime-metrics`.

Import in Python as:

```python
import llm_runtime_metrics
```

The supported package root focuses on the request-metrics workflow.

## S3 request-log proxy

Set `s3_proxy_url` to route S3 requests through an HTTP forward proxy, such as
a sidecar listening on `http://127.0.0.1:3128`:

```python
factory = RequestMetricsFactory(
    request_log_enabled=True,
    s3=True,
    s3_bucket="request-logs",
    s3_region="us-east-1",
    s3_proxy_url="http://127.0.0.1:3128",
)
```

The S3 bucket, region, signing, and endpoint remain unchanged. For HTTPS S3
endpoints, the proxy must support HTTP `CONNECT`. Omit `s3_proxy_url` for a
direct connection. `compact_s3_parquet_to_local` accepts the same option for
downloads. Use `s3_endpoint` only for an S3-compatible endpoint override.

In request-log records, `model_name` is stored as supplied. `org_id` is hashed
without a salt. `provided_session_id` is hashed without a salt so the same ID
has the same digest across deployments. `request_log_hash_salt` applies only to
the token-block hashes in `total_hashes`; without it, those hashes are unkeyed.
Timing, token counts, flags, and other metadata are stored as plain values.

Pass `b10_request_id` to `new_request` to store its unkeyed, domain-separated
hash (an empty string is stored as null); `hash_b10_request_id` computes the same digest for joins. The digest is
the lowercase hex of `blake3(b"pm-b10-request-id-v1" + raw_id.encode())`, with no
key, and is stable across versions. Call
`set_routed_workers` on the request before it finishes to record the prefill
and decode worker ids and DP ranks; each call replaces all four values, so
omitted arguments are stored as null. `anonymize_parquet` drops `b10_request_id`
and replaces worker ids with dense per-file ids by default. Each column is
numbered on its own, so after anonymizing, prefill worker 1 and decode worker 1
are not the same machine, and ids do not match across files. The worker id and
DP rank columns are meaningful only on rows with `__version__` 0.0.20 or later;
on older rows, null means "not recorded", not "not routed".

## Change request logging at runtime

`configure_request_log(request_log_enabled, ...)` takes the constructor's
request-log arguments and turns logging on or off, or moves it to a new sink,
without resetting metrics:

```python
factory.configure_request_log(True, s3=True, s3_bucket="request-logs")
factory.configure_request_log(False)  # factory.request_log_enabled is now False
```

A request keeps the sink that was active when it started: turning logging off
does not drop requests already in flight, and turning it on does not log
requests that started while it was off. When a replaced worker's last reference
goes away it closes its queue and flushes what is left, waiting up to 2s.
Repeating the same settings is a no-op. An invalid setting raises `ValueError`
and keeps the current one. While logging is off there is no recorder thread and
no token hashing, so `total_hashes()` is empty for requests that start then.

Set `component_name` on `RequestMetricsFactory` to identify the type of
component that wrote each request log, for example `"frontend"` or `"worker"`.
The library generates `component_id` once per process and shares it among all
factories in that process. Both fields are stored as plain text in JSON and
Parquet records; `component_name` is optional.

## Add LLM Metrics To An Existing Prometheus Server

```python
from prometheus_client import CollectorRegistry, start_http_server
from llm_runtime_metrics import (
    REQUEST_FEATURE_IMAGE,
    REQUEST_FEATURE_TOOLS,
    RequestMetricsCollector,
    RequestMetricsFactory,
)

# Reuse your existing registry if you already have one.
registry = CollectorRegistry()

factory = RequestMetricsFactory(
    request_log_enabled=False,
    metric_prefix="llm_runtime",
    metrics_window_seconds=60.0,
    metrics_quantiles=[0.5, 0.9, 0.99],
)

# Registers a custom collector that pulls fresh samples from `factory` at scrape time.
RequestMetricsCollector(
    factory,
    base_labels={"service": "text-generation", "engine": "vllm"},
    registry=registry,
)

# If your app already exposes /metrics, wire this into that server instead.
start_http_server(8000, registry=registry)


# Example lifecycle hooks in your inference code:
def on_request_start(prompt_token_ids: list[int]):
    features = REQUEST_FEATURE_TOOLS | REQUEST_FEATURE_IMAGE
    return factory.new_request(prompt_token_ids, features=features)


def on_stream_step(req_metrics, full_output_token_ids: list[int], cached_tokens: int | None):
    # Use `is_diff=False` when passing cumulative token ids.
    req_metrics.record_tokens(full_output_token_ids, cached_tokens=cached_tokens, is_diff=False)


def on_request_success(req_metrics):
    req_metrics.success()


def on_request_cancel(req_metrics):
    req_metrics.cancel()
```

The exported request latency distributions include `request_metrics_ttft_ms`,
`request_metrics_time_to_fifth_token_ms`,
`request_metrics_first_to_fifth_token_itl_ms`, and `request_metrics_itl_ms`.
`request_metrics_time_to_fifth_token_ms` is the elapsed time from request start
until the fifth output token is observed by `record_tokens`.

`request_metrics_first_to_fifth_token_itl_ms` is the elapsed time between
observing the first and fifth output tokens.

Request logs include `mean_output_tokens_per_step`: output tokens divided by the
number of nonempty `record_tokens` updates. It is null when no output tokens were
recorded. One update is the observed step; callers that batch multiple engine
iterations into one update get tokens per update, not tokens per engine iteration.

The inflight gauges also include early-decode pressure between the first and
fifth output token:
`current_inflight_first_to_fifth_token_requests`,
`mean_inflight_first_to_fifth_token_requests`,
`current_inflight_first_to_fifth_token_input_tokens`, and
`mean_inflight_first_to_fifth_token_input_tokens`. The input-token gauges count
prompt tokens attached to requests that have emitted at least one output token
but have not emitted their fifth token yet.

Available request feature bits:

- `REQUEST_FEATURE_NONE`
- `REQUEST_FEATURE_XGRAMMAR`
- `REQUEST_FEATURE_TOOLS`
- `REQUEST_FEATURE_IMAGE`

If you need plain text output instead of a collector, call:

```python
text = factory.prometheus_strfmt({"service": "text-generation", "engine": "vllm"})
```

## Synchronous stream chunk metrics

`AtomicHistogram` and `AtomicCounter` can share their native storage with a
`StreamChunkMetrics` handle. Resolve metric labels once per request; each chunk
then needs one Python-to-Rust call, with no per-chunk label lookup, Python counter
lock, task scheduling, or GIL release. This API is independent of the request
tracing workflow and does not change its metrics or timing semantics.

```python
from llm_runtime_metrics import AtomicCounter, AtomicHistogram, StreamChunkMetrics

# Reuse the same series across requests with the same labels.
ttft = AtomicHistogram.from_vm(-4, 10, 12)
tpot = AtomicHistogram.from_vm(-4, 10, 12)
itl = AtomicHistogram.from_vm(-4, 10, 12)
output_tokens = AtomicCounter()
stream = StreamChunkMetrics(ttft, tpot, itl, output_tokens)
try:
    stream.record(4, ttft_seconds=0.5)  # first chunk: prefill, not decode latency
    stream.record(4, chunk_seconds=0.04)  # one TPOT sample, four ITL samples
finally:
    stream.close()
```

The caller supplies seconds measured at its chosen chunk-delivery boundary.
Provide `ttft_seconds` only for the first chunk and omit `chunk_seconds` when no
previous chunk exists. Nonpositive/nonfinite durations are ignored; output tokens
are still counted. TPOT and ITL both observe seconds divided by token count; TPOT
has one observation per chunk and ITL one per token. Zero-token chunks do not add
latency observations. Negative token counts are rejected before any update.

Histogram snapshots remain `(buckets, count, sum)`; counter snapshots use `get()`.
A live stream owns shared histogram storage: `histogram.is_shared()` stays true
until all stream handles close or are dropped. Exporters that evict idle series
must retain shared histograms, including between TTFT and the next observation.
`close()` is idempotent; recording after close raises `RuntimeError`.

`AtomicCounterCollector(name, documentation, labelnames, registry=...)` adapts
native integer counters to `prometheus_client`. Its `labels(**labels)` resolves a
stable `AtomicCounter` handle; `collect()` reads it at scrape time. Counter names,
labels, zero-valued children, and optional `_created` samples follow Prometheus
counter conventions. Like `prometheus_client.Counter`, its series are retained.
The caller continues to own histogram names, bucket ranges, and their exporter.

Run `python examples/stream_chunk_bench.py` from the repository root to compare
cached Python observations with the fused native call. This is a microbenchmark,
not a measurement of end-to-end serving capacity.
