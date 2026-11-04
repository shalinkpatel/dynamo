import math

from llm_runtime_metrics import AtomicHistogram


def vm_bucket_index(
    amount: float, min_exponent: int, max_exponent: int, buckets_per_decimal: int
) -> int:
    if amount == 0:
        return 0
    bucket_count = (max_exponent - min_exponent) * buckets_per_decimal
    bucket_idx = (math.log10(amount) - min_exponent) * buckets_per_decimal
    if bucket_idx < 0:
        return 0
    if bucket_idx >= bucket_count:
        return bucket_count + 1

    idx = int(bucket_idx)
    if bucket_idx == float(idx) and idx > 0:
        idx -= 1
    return idx + 1


def linear_range_index(amount: float, ranges: list[tuple[float, float]]) -> int:
    for idx, (_, upper) in enumerate(ranges):
        if amount <= upper:
            return idx
    return len(ranges) - 1


def test_atomic_histogram_from_vm_matches_python_logic() -> None:
    min_exponent = -4
    max_exponent = 10
    buckets_per_decimal = 12
    hist = AtomicHistogram.from_vm(min_exponent, max_exponent, buckets_per_decimal)

    values: list[float] = [0.0]
    for exponent in range(-8, 18):
        base = 10.0**exponent
        values.extend(
            [base * factor for factor in (1 - 1e-12, 1.0, 1 + 1e-12, 1.2, 3.3, 9.9)]
        )
    values = [v for v in values if v >= 0 and math.isfinite(v)]

    for value in values:
        hist.observe(value)

    buckets, count, _ = hist.snapshot()
    assert count == len(values)

    expected = [0] * len(buckets)
    for value in values:
        index = vm_bucket_index(value, min_exponent, max_exponent, buckets_per_decimal)
        expected[index] += 1

    assert buckets == expected


def test_atomic_histogram_from_ranges_matches_linear_scan() -> None:
    ranges = [
        (0.0, 1e-4),
        (1e-4, 1e-3),
        (1e-3, 1e-2),
        (1e-2, 1.0),
        (1.0, float("inf")),
    ]
    upper_bounds = [upper for (_, upper) in ranges]
    hist = AtomicHistogram(upper_bounds)
    values = [
        0.0,
        1e-9,
        5e-5,
        1e-4,
        1.0000000001e-4,
        7e-4,
        1e-3,
        4e-3,
        1e-2,
        0.5,
        1.0,
        1.5,
        1e6,
    ]

    for value in values:
        hist.observe(value)

    buckets, count, _ = hist.snapshot()
    assert count == len(values)

    expected = [0] * len(ranges)
    for value in values:
        expected[linear_range_index(value, ranges)] += 1

    assert buckets == expected


def test_atomic_histogram_observe_many_matches_repeated_observe() -> None:
    ranges = [(0.0, 1e-3), (1e-3, 1e-2), (1e-2, float("inf"))]
    many = AtomicHistogram.from_ranges(ranges)
    repeated = AtomicHistogram.from_ranges(ranges)

    for amount, count in [(0.0, 3), (5e-4, 7), (2e-3, 5), (2.0, 11)]:
        many.observe_many(amount, count)
        for _ in range(count):
            repeated.observe(amount)

    assert many.snapshot() == repeated.snapshot()


def test_atomic_histogram_preserves_large_native_unit_sums() -> None:
    hist = AtomicHistogram([1e9, 1e12, float("inf")])

    hist.observe_many(25_000_000_000.0, 2)

    buckets, count, total = hist.snapshot()
    assert buckets == [0, 2, 0]
    assert count == 2
    assert total == 50_000_000_000.0


def test_atomic_histogram_observe_many_ignores_invalid_inputs() -> None:
    hist = AtomicHistogram([1e-3, float("inf")])

    hist.observe_many(5e-4, 0)
    hist.observe_many(5e-4, -1)
    hist.observe_many(-1.0, 3)
    hist.observe_many(math.nan, 3)

    buckets, count, total = hist.snapshot()
    assert buckets == [0, 0]
    assert count == 0
    assert total == 0.0
