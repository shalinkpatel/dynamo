from __future__ import annotations

import json
import time
from pathlib import Path

import llm_runtime_metrics
import numpy as np
import pytest
from llm_runtime_metrics import (
    REQUEST_FEATURE_IMAGE,
    REQUEST_FEATURE_TOOLS,
    RequestMetricsCollector,
    RequestMetricsFactory,
    prometheus_strfmt,
)
from prometheus_client import CollectorRegistry, generate_latest


def _strip_volatile_inflight_mean_samples(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.startswith("mean_inflight_requests{"):
            continue
        if line.startswith("mean_inflight_input_tokens{"):
            continue
        if line.startswith("mean_inflight_output_tokens{"):
            continue
        if line.startswith("mean_inflight_first_to_fifth_token_requests{"):
            continue
        if line.startswith("mean_inflight_first_to_fifth_token_input_tokens{"):
            continue
        lines.append(line)
    return "\n".join(lines) + "\n"


def test_root_api_hides_low_level_registry() -> None:
    assert not hasattr(llm_runtime_metrics, "PerformanceMetricsRegistry")


def test_root_api_exports_compaction_helpers() -> None:
    assert hasattr(llm_runtime_metrics, "compact_s3_parquet_to_local")
    assert callable(llm_runtime_metrics.compact_s3_parquet_to_local)
    assert hasattr(llm_runtime_metrics, "CompactResult")


def test_request_metrics_factory_unified_e2e(tmp_path: Path) -> None:
    factory = RequestMetricsFactory(
        request_log_enabled=True,
        request_log_batch_size=1,
        local_file=True,
        request_log_output_dir=str(tmp_path),
        s3=False,
        request_log_hash_salt="fixed-salt",
    )

    req = factory.new_request(list(range(128)))
    req.record_tokens([128, 129, 130, 131, 132], cached_tokens=64, is_diff=True)
    req.success()

    hashes = req.total_hashes()
    assert isinstance(hashes, list)
    assert len(hashes) > 0

    snap = factory.snapshot()
    series = snap.series()
    totals = snap.totals()
    ttft = find_metric_snapshot(series, "request_metrics_ttft_ms", feature="all")
    fifth_token = find_metric_snapshot(
        series, "request_metrics_time_to_fifth_token_ms", feature="all"
    )
    first_to_fifth_itl = find_metric_snapshot(
        series, "request_metrics_first_to_fifth_token_itl_ms", feature="all"
    )
    kv = find_metric_snapshot(series, "request_metrics_kv_cache_hit_rate")
    itl = find_metric_snapshot(series, "request_metrics_itl_ms", feature="all")
    successful = find_metric_snapshot(series, "request_metrics_successful_request")
    assert ttft.average is not None
    assert fifth_token.average is not None
    assert first_to_fifth_itl.average is not None
    assert kv.ratio == pytest.approx(0.5, rel=1e-9)
    assert itl.average is None
    assert successful.rate_per_second is not None
    assert any(
        item.name == "request_metrics_ttft_ms" and item.labels["feature"] == "all"
        for item in series
    )

    samples = factory.prometheus_samples({"model": "demo"})
    names = {sample.name for sample in samples}
    assert "request_metrics_ttft_ms_average" in names
    assert "request_metrics_time_to_fifth_token_ms_average" in names
    assert "request_metrics_first_to_fifth_token_itl_ms_average" in names
    assert "request_metrics_kv_cache_hit_rate_ratio" in names
    assert "current_inflight_first_to_fifth_token_requests" in names
    assert "mean_inflight_first_to_fifth_token_requests" in names
    assert "current_inflight_first_to_fifth_token_input_tokens" in names
    assert "mean_inflight_first_to_fifth_token_input_tokens" in names
    assert "total_requests" in names
    assert all(sample.labels.get("model") == "demo" for sample in samples)

    assert totals.total_requests == 1
    assert totals.total_successful_requests == 1
    assert totals.total_input_tokens == 128
    assert totals.total_output_tokens == 5
    assert totals.total_kv_cache_hits_tokens == 64
    assert totals.total_kv_cache_lookup_tokens == 128
    assert totals.kv_cache_hit_rate_lifetime == pytest.approx(0.5, rel=1e-9)
    assert totals.current_inflight_first_to_fifth_token_requests == 0
    assert totals.current_inflight_first_to_fifth_token_input_tokens == 0

    files = []
    for _ in range(50):
        files = list(tmp_path.glob("*.json"))
        if files:
            break
        time.sleep(0.01)
    assert files, "expected at least one request log JSON file"


def test_request_metrics_factory_does_not_log_by_default(tmp_path: Path) -> None:
    factory = RequestMetricsFactory(request_log_output_dir=str(tmp_path))
    req = factory.new_request(list(range(16)))
    req.record_tokens([16, 17], cached_tokens=8, is_diff=True)
    req.success()

    time.sleep(0.05)
    assert not list(tmp_path.glob("*.json"))


def test_request_metrics_request_log_session_id_is_optional(tmp_path: Path) -> None:
    factory = RequestMetricsFactory(
        request_log_enabled=True,
        request_log_batch_size=1,
        local_file=True,
        request_log_output_dir=str(tmp_path),
        s3=False,
        request_log_hash_salt="fixed-salt",
        model_name="test-model",
        component_name="frontend",
    )

    with_session = factory.new_request(list(range(16)), session_id="session-123")
    with_session.record_tokens([16, 17], cached_tokens=8, is_diff=True)
    with_session.success()

    without_session = factory.new_request(list(range(16)))
    without_session.record_tokens([16, 17], cached_tokens=8, is_diff=True)
    without_session.success()

    files = []
    for _ in range(50):
        files = list(tmp_path.glob("*.json"))
        if len(files) >= 2:
            break
        time.sleep(0.01)
    assert len(files) >= 2, "expected request log JSON files for both requests"

    entries = []
    for path in files:
        entries.extend(json.loads(path.read_text()))

    with_session_entry = next(
        entry
        for entry in entries
        if entry["input_tokens"] == 16 and entry.get("provided_session_id")
    )
    without_session_entry = next(
        entry
        for entry in entries
        if entry["input_tokens"] == 16 and not entry.get("provided_session_id")
    )

    # Session ID should be hashed (64 hex chars), not the original
    assert with_session_entry["provided_session_id"] != "session-123"
    assert len(with_session_entry["provided_session_id"]) == 64
    assert with_session_entry["model_name"] == "test-model"
    assert with_session_entry["component_name"] == "frontend"
    assert len(with_session_entry["component_id"]) == 32
    assert without_session_entry["component_id"] == with_session_entry["component_id"]
    assert not without_session_entry.get("provided_session_id")


RAW_B10_REQUEST_ID = "wp5ab0192f1e8a27d4c5b8e3f2a1d0c9b8e7f6"
WORKER_COLUMNS = (
    "prefill_worker_id",
    "prefill_dp_rank",
    "decode_worker_id",
    "decode_dp_rank",
)


def _read_json_entries(tmp_path: Path, expected: int) -> list[dict]:
    entries: list[dict] = []
    for _ in range(100):
        entries = [
            entry
            for path in tmp_path.glob("*.json")
            for entry in json.loads(path.read_text())
        ]
        if len(entries) >= expected:
            break
        time.sleep(0.01)
    assert len(entries) == expected
    return entries


def test_hash_b10_request_id_is_a_hex_digest() -> None:
    digest = llm_runtime_metrics.hash_b10_request_id(RAW_B10_REQUEST_ID)
    assert len(digest) == 64
    assert int(digest, 16) >= 0
    assert digest != RAW_B10_REQUEST_ID
    assert digest == llm_runtime_metrics.hash_b10_request_id(RAW_B10_REQUEST_ID)


def test_hash_b10_request_id_matches_golden_digest() -> None:
    # Permanent join key: blake3(b"pm-b10-request-id-v1" + raw id). Must match Rust.
    assert (
        llm_runtime_metrics.hash_b10_request_id(RAW_B10_REQUEST_ID)
        == "a619fd8ae35a3cd121bfcbd90349aac1588a26034f868965d546e8650f472594"
    )


def test_request_log_records_hashed_b10_request_id_and_routed_workers(
    tmp_path: Path,
) -> None:
    factory = RequestMetricsFactory(
        request_log_enabled=True,
        request_log_batch_size=1,
        local_file=True,
        request_log_output_dir=str(tmp_path),
        s3=False,
    )

    succeeded = factory.new_request(list(range(1)), b10_request_id=RAW_B10_REQUEST_ID)
    succeeded.set_routed_workers(
        prefill_worker_id=7900896516392480724,
        prefill_dp_rank=0,
        decode_worker_id=3121732886823530259,
        decode_dp_rank=1,
    )
    succeeded.success()
    # The row is already written; a late call is ignored without raising.
    succeeded.set_routed_workers(prefill_worker_id=1)

    canceled = factory.new_request(list(range(2)), b10_request_id=RAW_B10_REQUEST_ID)
    canceled.set_routed_workers(prefill_worker_id=11, prefill_dp_rank=0)
    canceled.cancel()

    dropped = factory.new_request(list(range(3)), b10_request_id=RAW_B10_REQUEST_ID)
    dropped.set_routed_workers(prefill_worker_id=12, decode_worker_id=22)
    del dropped

    factory.new_request(list(range(4))).success()
    factory.new_request(list(range(5)), b10_request_id="").success()

    entries = _read_json_entries(tmp_path, 5)
    assert all(
        RAW_B10_REQUEST_ID not in path.read_text() for path in tmp_path.glob("*.json")
    )
    by_input = {entry["input_tokens"]: entry for entry in entries}
    digest = llm_runtime_metrics.hash_b10_request_id(RAW_B10_REQUEST_ID)

    def workers(entry: dict) -> tuple:
        return tuple(entry[column] for column in WORKER_COLUMNS)

    assert by_input[1]["b10_request_id"] == digest
    assert workers(by_input[1]) == (7900896516392480724, 0, 3121732886823530259, 1)
    assert by_input[2]["b10_request_id"] == digest
    assert by_input[2]["request_canceled"] is True
    assert workers(by_input[2]) == (11, 0, None, None)
    assert by_input[3]["request_canceled"] is True
    assert workers(by_input[3]) == (12, None, 22, None)
    assert by_input[4]["b10_request_id"] is None
    assert workers(by_input[4]) == (None, None, None, None)
    assert by_input[5]["b10_request_id"] is None


def test_anonymize_parquet_defaults_drop_b10_request_id_and_densify_workers(
    tmp_path: Path,
) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    records = tmp_path / "records"
    factory = RequestMetricsFactory(
        request_log_enabled=True,
        request_log_batch_size=3,
        local_file=True,
        request_log_output_dir=str(records),
        s3=False,
        file_format="parquet",
    )
    for prefill, decode in [(901, 902), (905, None), (901, 902)]:
        req = factory.new_request(list(range(64)), b10_request_id=RAW_B10_REQUEST_ID)
        req.set_routed_workers(
            prefill_worker_id=prefill, prefill_dp_rank=0, decode_worker_id=decode
        )
        req.success()
    del factory, req
    for _ in range(100):
        if list(records.glob("*.parquet")):
            break
        time.sleep(0.01)
    (source,) = records.glob("*.parquet")

    output = tmp_path / "anonymized.parquet"
    llm_runtime_metrics.anonymize_parquet(str(source), str(output))

    table = pq.read_table(output)
    assert "b10_request_id" not in table.column_names
    assert table.column("prefill_worker_id").to_pylist() == [1, 2, 1]
    assert table.column("decode_worker_id").to_pylist() == [1, None, 1]
    assert table.column("prefill_dp_rank").to_pylist() == [0, 0, 0]
    assert table.column("decode_dp_rank").to_pylist() == [None, None, None]


def test_prometheus_collector_and_strfmt() -> None:
    factory = RequestMetricsFactory(request_log_enabled=False)
    req = factory.new_request(list(range(128)))
    req.record_tokens([128, 129, 130], cached_tokens=64, is_diff=True)
    req.success()

    registry = CollectorRegistry()
    RequestMetricsCollector(factory, base_labels={"engine": "vllm"}, registry=registry)
    text = generate_latest(registry).decode("utf-8")
    assert "request_metrics_ttft_ms_average" in text
    assert 'engine="vllm"' in text
    assert "total_requests_total" in text

    strfmt = factory.prometheus_strfmt({"engine": "vllm"})
    assert "# TYPE request_metrics_ttft_ms_average gauge" in strfmt
    assert 'request_metrics_ttft_ms_average{engine="vllm",feature="all"}' in strfmt
    assert "# TYPE total_requests counter" in strfmt

    strfmt_from_helper = prometheus_strfmt(factory, base_labels={"engine": "vllm"})
    assert _strip_volatile_inflight_mean_samples(strfmt_from_helper) == (
        _strip_volatile_inflight_mean_samples(strfmt)
    )


def test_first_to_fifth_inflight_phase_totals_and_prometheus() -> None:
    factory = RequestMetricsFactory(request_log_enabled=False)
    req = factory.new_request(list(range(16)))

    req.record_tokens([16], cached_tokens=8, is_diff=True)
    totals = factory.snapshot().totals()
    assert totals.current_inflight_first_to_fifth_token_requests == 1
    assert totals.current_inflight_first_to_fifth_token_input_tokens == 16

    samples = {sample.name: sample.value for sample in factory.prometheus_samples()}
    assert samples["current_inflight_first_to_fifth_token_requests"] == 1
    assert samples["current_inflight_first_to_fifth_token_input_tokens"] == 16

    req.record_tokens([17, 18, 19], cached_tokens=None, is_diff=True)
    totals = factory.snapshot().totals()
    assert totals.current_inflight_first_to_fifth_token_requests == 1
    assert totals.current_inflight_first_to_fifth_token_input_tokens == 16

    req.record_tokens([20], cached_tokens=None, is_diff=True)
    totals = factory.snapshot().totals()
    assert totals.current_inflight_first_to_fifth_token_requests == 0
    assert totals.current_inflight_first_to_fifth_token_input_tokens == 0
    req.success()


def test_prometheus_strfmt_sanitizes_metric_names() -> None:
    factory = RequestMetricsFactory(
        metric_prefix="req-metrics", request_log_enabled=False
    )
    req = factory.new_request(list(range(16)))
    req.record_tokens([16, 17], cached_tokens=8, is_diff=True)
    req.success()

    strfmt = factory.prometheus_strfmt()
    assert "req_metrics_ttft_ms_average" in strfmt
    assert "req-metrics" not in strfmt


def test_prometheus_strfmt_escapes_label_values() -> None:
    factory = RequestMetricsFactory(request_log_enabled=False)
    req = factory.new_request(list(range(16)))
    req.record_tokens([16, 17], cached_tokens=8, is_diff=True)
    req.success()

    strfmt = factory.prometheus_strfmt({"tag": 'a"b\nc\\d'})
    assert 'tag="a\\"b\\nc\\\\d"' in strfmt


def test_prometheus_strfmt_default_cardinality_budget_and_kv_percentiles() -> None:
    factory = RequestMetricsFactory(request_log_enabled=False)
    req = factory.new_request(list(range(16)))
    req.record_tokens([16, 17], cached_tokens=8, is_diff=True)
    req.success()

    strfmt = factory.prometheus_strfmt()
    sample_lines = [
        line for line in strfmt.splitlines() if line and not line.startswith("#")
    ]
    assert len(sample_lines) <= 240

    kv_quantile_lines = [
        line
        for line in sample_lines
        if line.startswith("request_metrics_kv_cache_hit_rate_quantile{")
    ]
    assert len(kv_quantile_lines) == 10
    for quantile in [
        "0.01",
        "0.05",
        "0.1",
        "0.25",
        "0.5",
        "0.75",
        "0.9",
        "0.95",
        "0.99",
        "0.999",
    ]:
        assert any(f'quantile="{quantile}"' in line for line in kv_quantile_lines)


def test_prometheus_strfmt_omits_empty_observation_samples() -> None:
    factory = RequestMetricsFactory(request_log_enabled=False)

    kv_snapshot = find_metric_snapshot(
        factory.snapshot().series(), "request_metrics_kv_cache_hit_rate"
    )
    assert kv_snapshot.numerator_sum is None
    assert kv_snapshot.denominator_sum is None
    assert kv_snapshot.ratio is None
    assert kv_snapshot.quantiles == {}

    ttft_snapshot = find_metric_snapshot(
        factory.snapshot().series(), "request_metrics_ttft_ms", feature="all"
    )
    assert ttft_snapshot.average is None
    assert ttft_snapshot.quantiles == {}

    strfmt = factory.prometheus_strfmt()
    assert "request_metrics_request_rate_per_second" in strfmt
    assert "request_metrics_request_quantile" not in strfmt
    assert "request_metrics_ttft_ms_average" not in strfmt
    assert "request_metrics_ttft_ms_quantile" not in strfmt
    assert "request_metrics_kv_cache_hit_rate_numerator_sum" not in strfmt
    assert "request_metrics_kv_cache_hit_rate_denominator_sum" not in strfmt
    assert "request_metrics_kv_cache_hit_rate_ratio" not in strfmt
    assert "request_metrics_kv_cache_hit_rate_quantile" not in strfmt


def test_request_metrics_feature_bitmask_records_presence_series() -> None:
    factory = RequestMetricsFactory(request_log_enabled=False)
    req = factory.new_request(
        list(range(16)),
        features=REQUEST_FEATURE_TOOLS | REQUEST_FEATURE_IMAGE,
    )
    req.record_tokens([16, 17], cached_tokens=8, is_diff=True)
    req.record_tokens([18], cached_tokens=None, is_diff=True)
    req.success()

    assert req.features() == REQUEST_FEATURE_TOOLS | REQUEST_FEATURE_IMAGE

    series = factory.snapshot().series()
    assert (
        find_metric_snapshot(series, "request_metrics_ttft_ms", feature="tools").average
        is not None
    )
    assert (
        find_metric_snapshot(series, "request_metrics_ttft_ms", feature="image").average
        is not None
    )
    assert (
        find_metric_snapshot(
            series, "request_metrics_ttft_ms", feature="xgrammar"
        ).average
        is None
    )
    # ITL records only when elapsed time between token batches is > 0; fast
    # synchronous test runs may legitimately leave these series empty.
    assert_optional_nonnegative(
        find_metric_snapshot(series, "request_metrics_itl_ms", feature="tools").average
    )
    assert_optional_nonnegative(
        find_metric_snapshot(series, "request_metrics_itl_ms", feature="image").average
    )
    assert (
        find_metric_snapshot(
            series, "request_metrics_itl_ms", feature="xgrammar"
        ).average
        is None
    )
    assert (
        find_metric_snapshot(
            series, "request_metrics_request", feature="all"
        ).rate_per_second
        is not None
    )
    assert (
        find_metric_snapshot(
            series, "request_metrics_request", feature="tools"
        ).rate_per_second
        is not None
    )
    assert (
        find_metric_snapshot(
            series, "request_metrics_request", feature="image"
        ).rate_per_second
        is not None
    )
    assert find_metric_snapshot(
        series, "request_metrics_request", feature="none"
    ).rate_per_second == pytest.approx(0.0)


@pytest.mark.parametrize(
    "input_ids, output_ids",
    [
        (list(range(128)), [128, 129, 130]),
        (np.arange(128, dtype=np.uint32), np.array([128, 129, 130], dtype=np.uint32)),
    ],
)
def test_request_metrics_accepts_list_and_numpy_inputs(input_ids, output_ids) -> None:
    factory = RequestMetricsFactory(request_log_enabled=False)
    req = factory.new_request(input_ids)
    req.record_tokens(output_ids, cached_tokens=64, is_diff=True)
    req.success()

    kv = find_metric_snapshot(
        factory.snapshot().series(), "request_metrics_kv_cache_hit_rate"
    )
    assert kv.ratio == pytest.approx(0.5, rel=1e-9)


def find_metric_snapshot(series, name: str, *, feature: str | None = None):
    """Find the plain (unlabeled) series for ``name``. Excludes any series
    carrying ``service_tier`` or ``dp_rank`` labels so it never accidentally
    matches a labeled series."""
    for item in series:
        if item.name != name:
            continue
        if "service_tier" in item.labels or "dp_rank" in item.labels:
            continue
        if feature is None and "feature" in item.labels:
            continue
        if feature is not None and item.labels.get("feature") != feature:
            continue
        return item.snapshot()
    raise AssertionError(f"missing metric series {name!r} feature={feature!r}")


def find_metric_snapshot_with_tier(
    series, name: str, *, service_tier: str, feature: str | None = None
):
    """Find the tier-only series for ``name``. Excludes any series carrying
    ``dp_rank`` so it never matches a combined series."""
    for item in series:
        if item.name != name:
            continue
        if item.labels.get("service_tier") != service_tier:
            continue
        if "dp_rank" in item.labels:
            continue
        if feature is None and "feature" in item.labels:
            continue
        if feature is not None and item.labels.get("feature") != feature:
            continue
        return item.snapshot()
    raise AssertionError(
        f"missing metric series {name!r} feature={feature!r} service_tier={service_tier!r}"
    )


def find_metric_snapshot_with_labels(
    series,
    name: str,
    *,
    labels: dict[str, str],
    feature: str | None = None,
):
    """Find a snapshot whose series carries every ``labels`` entry (and the
    optional ``feature`` label).  Other labels are ignored."""
    for item in series:
        if item.name != name:
            continue
        if feature is None and "feature" in item.labels:
            continue
        if feature is not None and item.labels.get("feature") != feature:
            continue
        if not all(item.labels.get(k) == v for k, v in labels.items()):
            continue
        return item.snapshot()
    raise AssertionError(
        f"missing metric series {name!r} feature={feature!r} labels={labels!r}"
    )


def assert_optional_nonnegative(value: float | None) -> None:
    if value is not None:
        assert value >= 0.0


def test_service_tier_fast_produces_tier_labeled_prometheus_samples() -> None:
    factory = RequestMetricsFactory(request_log_enabled=False)
    req = factory.new_request(list(range(16)), service_tier="fast")
    req.record_tokens([16, 17], cached_tokens=8, is_diff=True)
    req.record_tokens([18, 19], cached_tokens=None, is_diff=True)
    req.success()

    series = factory.snapshot().series()

    # Tier-labeled series must exist for the major metrics.
    tiered_ttft = find_metric_snapshot_with_tier(
        series, "request_metrics_ttft_ms", feature="all", service_tier="fast"
    )
    assert tiered_ttft.average is not None

    tiered_request = find_metric_snapshot_with_tier(
        series, "request_metrics_request", service_tier="fast"
    )
    assert tiered_request.rate_per_second is not None

    tiered_successful = find_metric_snapshot_with_tier(
        series,
        "request_metrics_successful_request",
        service_tier="fast",
    )
    assert tiered_successful.rate_per_second is not None

    tiered_kv = find_metric_snapshot_with_tier(
        series, "request_metrics_kv_cache_hit_rate", service_tier="fast"
    )
    assert tiered_kv.ratio is not None

    # The plain unlabelled feature="all" series must still exist (no service_tier).
    plain_ttft = find_metric_snapshot(series, "request_metrics_ttft_ms", feature="all")
    assert plain_ttft.average is not None

    # Prometheus text output must carry the service_tier label.
    strfmt = factory.prometheus_strfmt({"engine": "vllm"})
    assert 'service_tier="fast"' in strfmt


def test_service_tier_none_omits_tier_label() -> None:
    factory = RequestMetricsFactory(request_log_enabled=False)
    req = factory.new_request(list(range(16)))
    req.record_tokens([16, 17], cached_tokens=8, is_diff=True)
    req.success()

    series = factory.snapshot().series()
    assert not any(
        "service_tier" in item.labels for item in series
    ), "no series should carry a service_tier label when tier is None"

    strfmt = factory.prometheus_strfmt()
    assert "service_tier" not in strfmt


def test_service_tier_unknown_raises_value_error() -> None:
    factory = RequestMetricsFactory(request_log_enabled=False)
    with pytest.raises(ValueError):
        factory.new_request(list(range(16)), service_tier="unknown_tier")


def test_dp_rank_only_produces_rank_labeled_prometheus_samples() -> None:
    factory = RequestMetricsFactory(request_log_enabled=False)
    req = factory.new_request(list(range(16)), dp_rank=3)
    req.record_tokens([16, 17], cached_tokens=8, is_diff=True)
    req.record_tokens([18, 19], cached_tokens=None, is_diff=True)
    req.success()

    series = factory.snapshot().series()

    # Rank-labeled distribution series must exist.
    rank_ttft = find_metric_snapshot_with_labels(
        series,
        "request_metrics_ttft_ms",
        labels={"dp_rank": "3"},
        feature="all",
    )
    assert rank_ttft.average is not None

    # Rank-labeled rate/ratio series must exist.
    rank_request = find_metric_snapshot_with_labels(
        series, "request_metrics_request", labels={"dp_rank": "3"}
    )
    assert rank_request.rate_per_second is not None

    rank_successful = find_metric_snapshot_with_labels(
        series, "request_metrics_successful_request", labels={"dp_rank": "3"}
    )
    assert rank_successful.rate_per_second is not None

    rank_kv = find_metric_snapshot_with_labels(
        series, "request_metrics_kv_cache_hit_rate", labels={"dp_rank": "3"}
    )
    assert rank_kv.ratio is not None

    # The plain unlabelled feature="all" series must still exist (no dp_rank).
    plain_ttft = find_metric_snapshot(series, "request_metrics_ttft_ms", feature="all")
    assert plain_ttft.average is not None

    # Prometheus text output must carry the dp_rank label.
    strfmt = factory.prometheus_strfmt({"engine": "vllm"})
    assert 'dp_rank="3"' in strfmt


def test_dp_rank_and_service_tier_produce_independent_series() -> None:
    factory = RequestMetricsFactory(request_log_enabled=False)
    req = factory.new_request(list(range(16)), service_tier="priority", dp_rank=7)
    req.record_tokens([16, 17], cached_tokens=8, is_diff=True)
    req.success()

    series = factory.snapshot().series()

    # A tier-only series (service_tier="priority", NO dp_rank) must exist.
    tier_ttft = find_metric_snapshot_with_labels(
        series,
        "request_metrics_ttft_ms",
        labels={"service_tier": "priority"},
        feature="all",
    )
    assert tier_ttft.average is not None

    tier_request = find_metric_snapshot_with_labels(
        series,
        "request_metrics_request",
        labels={"service_tier": "priority"},
    )
    assert tier_request.rate_per_second is not None

    # A rank-only series (dp_rank="7", NO service_tier) must exist.
    rank_ttft = find_metric_snapshot_with_labels(
        series,
        "request_metrics_ttft_ms",
        labels={"dp_rank": "7"},
        feature="all",
    )
    assert rank_ttft.average is not None

    rank_request = find_metric_snapshot_with_labels(
        series,
        "request_metrics_request",
        labels={"dp_rank": "7"},
    )
    assert rank_request.rate_per_second is not None

    # No combined series (BOTH service_tier and dp_rank) should exist.
    assert not any(
        "service_tier" in item.labels and "dp_rank" in item.labels for item in series
    ), "no series should carry both service_tier and dp_rank labels"


def test_dp_rank_none_omits_rank_label() -> None:
    factory = RequestMetricsFactory(request_log_enabled=False)
    req = factory.new_request(list(range(16)))
    req.record_tokens([16, 17], cached_tokens=8, is_diff=True)
    req.success()

    series = factory.snapshot().series()
    assert not any(
        "dp_rank" in item.labels for item in series
    ), "no series should carry a dp_rank label when rank is None"

    strfmt = factory.prometheus_strfmt()
    assert "dp_rank" not in strfmt


def test_dp_rank_unused_ranks_never_appear() -> None:
    factory = RequestMetricsFactory(request_log_enabled=False)
    req = factory.new_request(list(range(16)), dp_rank=0)
    req.record_tokens([16, 17], cached_tokens=8, is_diff=True)
    req.success()

    series = factory.snapshot().series()
    # Only rank "0" should appear; no other rank value.
    used_ranks = {
        item.labels.get("dp_rank") for item in series if "dp_rank" in item.labels
    }
    assert used_ranks == {"0"}, f"only rank 0 should appear, got {used_ranks}"


def test_dp_rank_above_max_omits_rank_label() -> None:
    factory = RequestMetricsFactory(request_log_enabled=False)
    for rank in (32, 33):
        req = factory.new_request(list(range(16)), dp_rank=rank)
        req.record_tokens([1, 2], cached_tokens=2, is_diff=True)
        req.success()
    series = factory.snapshot().series()
    assert not any(
        "dp_rank" in item.labels for item in series
    ), "ranks >= 32 should not produce dp_rank-labeled series"


def test_dp_rank_negative_omits_rank_label() -> None:
    factory = RequestMetricsFactory(request_log_enabled=False)
    req = factory.new_request(list(range(16)), dp_rank=-1)
    req.record_tokens([1, 2], cached_tokens=2, is_diff=True)
    req.success()
    series = factory.snapshot().series()
    assert not any(
        "dp_rank" in item.labels for item in series
    ), "negative dp_rank should not produce dp_rank-labeled series"


def test_anonymize_parquet_defaults_accept_old_schema_files(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    hash_type = pa.list_(pa.binary(32))
    # The pre-0.0.20 request-log columns that the default lists touch.
    old = pa.table(
        {
            "timestamp_start_unix_ms": pa.array([10, 20], pa.uint64()),
            "total_hashes": pa.array([[b"a" * 32], [b"a" * 32, b"b" * 32]], hash_type),
            "provided_session_id": pa.array(["s1", None], pa.string()),
            "org_id": pa.array(["o", "o"], pa.string()),
            "__version__": pa.array(["0.0.19", "0.0.19"], pa.string()),
        }
    )
    source = tmp_path / "old.parquet"
    pq.write_table(old, source)
    output = tmp_path / "anonymized.parquet"

    llm_runtime_metrics.anonymize_parquet(str(source), str(output))

    table = pq.read_table(output)
    assert table.column_names == old.column_names
    assert table.column("provided_session_id").to_pylist() == ["1", None]
    assert table.column("org_id").to_pylist() == ["1", "1"]
    assert table.column("total_hashes").to_pylist() == [[1], [1, 2]]

    # The 0.0.24 format records the per-request org as request_org_id.
    new = pa.table(
        {
            "timestamp_start_unix_ms": pa.array([10, 20], pa.uint64()),
            "total_hashes": pa.array([[b"a" * 32], [b"a" * 32, b"b" * 32]], hash_type),
            "provided_session_id": pa.array(["s1", None], pa.string()),
            "request_org_id": pa.array(["o1", "o2"], pa.string()),
            "__version__": pa.array(["0.0.24", "0.0.24"], pa.string()),
        }
    )
    source = tmp_path / "new.parquet"
    pq.write_table(new, source)
    output = tmp_path / "anonymized-new.parquet"

    llm_runtime_metrics.anonymize_parquet(str(source), str(output))

    table = pq.read_table(output)
    assert table.column("provided_session_id").to_pylist() == ["1", None]
    # Distinct request orgs stay distinct after pseudonymization.
    assert table.column("request_org_id").to_pylist() != ["o1", "o2"]
    assert len(set(table.column("request_org_id").to_pylist())) == 2
