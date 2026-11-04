from __future__ import annotations

import statistics
import time

from llm_runtime_metrics import RequestMetricsFactory


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    idx = int(0.95 * (len(ordered) - 1))
    return ordered[idx]


def test_new_request_latency_smoke() -> None:
    factory = RequestMetricsFactory(request_log_enabled=False)
    input_ids = list(range(128))

    latencies_ms: list[float] = []
    for _ in range(400):
        start = time.perf_counter()
        req = factory.new_request(input_ids)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        req.cancel()

    assert statistics.mean(latencies_ms) < 0.20
    assert _p95(latencies_ms) < 0.50


def test_record_tokens_latency_smoke() -> None:
    factory = RequestMetricsFactory(request_log_enabled=False)
    req = factory.new_request(list(range(128)))
    diff = [128, 129, 130, 131]

    latencies_ms: list[float] = []
    for _ in range(800):
        start = time.perf_counter()
        req.record_tokens(diff, cached_tokens=64, is_diff=True)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    req.success()

    assert statistics.mean(latencies_ms) < 0.05
    assert _p95(latencies_ms) < 0.20


def test_new_request_labeled_latency_smoke() -> None:
    """Labeled new_request (tier-only and tier+rank) should stay within
    the same latency budget as the unlabeled path, since handles are cached
    after first use."""
    factory = RequestMetricsFactory(request_log_enabled=False)
    input_ids = list(range(128))

    for service_tier, dp_rank, label in [
        ("fast", None, "tier-only"),
        ("fast", 7, "tier+rank"),
    ]:
        # Warm the cache with one call so steady-state is measured.
        factory.new_request(
            input_ids, service_tier=service_tier, dp_rank=dp_rank
        ).cancel()

        latencies_ms: list[float] = []
        for _ in range(400):
            start = time.perf_counter()
            req = factory.new_request(
                input_ids, service_tier=service_tier, dp_rank=dp_rank
            )
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            req.cancel()

        mean_ms = statistics.mean(latencies_ms)
        p95_ms = _p95(latencies_ms)
        assert mean_ms < 0.20, f"{label}: mean {mean_ms:.4f}ms exceeds 0.20ms"
        assert p95_ms < 0.50, f"{label}: p95 {p95_ms:.4f}ms exceeds 0.50ms"


def test_record_tokens_labeled_latency_smoke() -> None:
    """Labeled record_tokens (tier-only and tier+rank) should stay within
    the same latency budget as the unlabeled path."""
    factory = RequestMetricsFactory(request_log_enabled=False)
    diff = [128, 129, 130, 131]

    for service_tier, dp_rank, label in [
        ("fast", None, "tier-only"),
        ("fast", 7, "tier+rank"),
    ]:
        req = factory.new_request(
            list(range(128)), service_tier=service_tier, dp_rank=dp_rank
        )

        latencies_ms: list[float] = []
        for _ in range(800):
            start = time.perf_counter()
            req.record_tokens(diff, cached_tokens=64, is_diff=True)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

        req.success()

        mean_ms = statistics.mean(latencies_ms)
        p95_ms = _p95(latencies_ms)
        assert mean_ms < 0.05, f"{label}: mean {mean_ms:.4f}ms exceeds 0.05ms"
        assert p95_ms < 0.20, f"{label}: p95 {p95_ms:.4f}ms exceeds 0.20ms"
