import gc
import math

import pytest
from llm_runtime_metrics import (
    AtomicCounter,
    AtomicCounterCollector,
    AtomicHistogram,
    StreamChunkMetrics,
)
from prometheus_client import CollectorRegistry, Counter, generate_latest


def hist():
    return AtomicHistogram([0.001, 0.01, 1.0, math.inf])


def test_fused_updates_match_individual_observations():
    ttft, tpot, itl = hist(), hist(), hist()
    expected_ttft, expected_tpot, expected_itl = hist(), hist(), hist()
    counter = AtomicCounter()
    stream = StreamChunkMetrics(ttft, tpot, itl, counter)
    stream.record(4, 0.5)
    expected_ttft.observe(0.5)
    for seconds, tokens in [(0.4, 4), (0.01, 1), (0.1, 8)]:
        stream.record(tokens, chunk_seconds=seconds)
        expected_tpot.observe(seconds / tokens)
        expected_itl.observe_many(seconds / tokens, tokens)
    assert ttft.snapshot() == expected_ttft.snapshot()
    assert tpot.snapshot() == expected_tpot.snapshot()
    assert itl.snapshot() == expected_itl.snapshot()
    assert tpot.snapshot()[1] == 3
    assert itl.snapshot()[1] == 13
    assert counter.get() == 17


@pytest.mark.parametrize("seconds", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_invalid_durations_do_not_record_latency(seconds):
    ttft, tpot, itl = hist(), hist(), hist()
    counter = AtomicCounter()
    stream = StreamChunkMetrics(ttft, tpot, itl, counter)
    stream.record(4, seconds, seconds)
    assert [h.snapshot()[1] for h in (ttft, tpot, itl)] == [0, 0, 0]
    assert counter.get() == 4


def test_zero_tokens_and_absent_duration():
    ttft, tpot, itl = hist(), hist(), hist()
    counter = AtomicCounter()
    stream = StreamChunkMetrics(ttft, tpot, itl, counter)
    stream.record(0, 0.5, 0.2)
    stream.record(4)
    assert ttft.snapshot()[1] == 1
    assert tpot.snapshot()[1] == itl.snapshot()[1] == 0
    assert counter.get() == 4
    with pytest.raises(OverflowError):
        stream.record(-1)
    assert counter.get() == 4


def test_handles_share_storage_and_release_histograms_on_close_or_drop():
    ttft, tpot, itl = hist(), hist(), hist()
    counter = AtomicCounter()
    assert not ttft.is_shared()
    first = StreamChunkMetrics(ttft, tpot, itl, counter)
    second = StreamChunkMetrics(ttft, tpot, itl, counter)
    first.record(3, 0.1)
    second.record(7, 0.2)
    assert counter.get() == 10
    assert ttft.snapshot()[1] == 2
    first.close()
    first.close()
    assert ttft.is_shared()
    with pytest.raises(RuntimeError, match="closed"):
        first.record(1)
    del second
    gc.collect()
    assert not any(h.is_shared() for h in (ttft, tpot, itl))


def test_atomic_counter_collector_matches_prometheus_counter():
    native_registry, reference_registry = CollectorRegistry(), CollectorRegistry()
    native = AtomicCounterCollector(
        "output_tokens_total", "Tokens", ["tier"], registry=native_registry
    )
    reference = Counter(
        "output_tokens_total", "Tokens", ["tier"], registry=reference_registry
    )
    for tier, tokens in [("", 0), ("flex", 4), ("flex", 8), ("fast", 2)]:
        native.labels(tier=tier).inc(tokens)
        reference.labels(tier=tier).inc(tokens)
    assert native.labels(tier="flex") is native.labels(tier="flex")

    def samples(registry):
        return [
            line
            for line in generate_latest(registry).decode().splitlines()
            if "_created" not in line
        ]

    assert samples(native_registry) == samples(reference_registry)
    with pytest.raises(ValueError):
        native.labels(other="flex")
    with pytest.raises(ValueError):
        native.labels()


def test_fused_counter_updates_are_exported_with_labels():
    registry = CollectorRegistry()
    counter = AtomicCounterCollector(
        "llm_output_tokens_total", "Tokens", ["service_tier"], registry=registry
    )
    stream = StreamChunkMetrics(
        hist(), hist(), hist(), counter.labels(service_tier="flex")
    )
    stream.record(4)
    stream.record(3)
    assert (
        'llm_output_tokens_total{service_tier="flex"} 7.0'
        in generate_latest(registry).decode()
    )
