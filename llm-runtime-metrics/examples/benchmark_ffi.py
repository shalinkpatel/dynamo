#!/usr/bin/env python3
"""
Benchmark to verify if NumPy array passing is zero-copy or not.
The 2ms overhead you're seeing is likely from the .to_vec() copy in Rust.
"""

import os
import sys
import time

import numpy as np

# Add the Python bindings to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "bindings/python/src"))

from llm_runtime_metrics import REQUEST_FEATURE_NONE, RequestMetricsFactory

print("✓ Successfully imported llm_runtime_metrics")


def benchmark_numpy_vs_list():
    """Compare NumPy array vs Python list[int] performance."""
    for enabled_log in [True, False]:
        print(
            f"\n{'=' * 40} Logging {'Enabled' if enabled_log else 'Disabled'} {'=' * 40}"
        )
        factory = RequestMetricsFactory(
            request_log_enabled=enabled_log,
            request_log_output_dir="/tmp/test_metrics",
            request_log_batch_size=10000,
            model_name="test-model",
            block_size=32,
        )

        sizes = [1000, 10000, 50000, 100000, 200000, 500000]

        print("\n" + "=" * 80)
        print("NumPy vs list[int] Performance Comparison")
        print("=" * 80)

        for size in sizes:
            # Create NumPy array
            np_tokens = np.arange(size, dtype=np.uint32)
            assert np_tokens.flags["C_CONTIGUOUS"]

            # Create Python list
            list_tokens = list(range(size))

            # Warmup
            for _ in range(5):
                factory.new_request(np_tokens, REQUEST_FEATURE_NONE)
                factory.new_request(list_tokens, REQUEST_FEATURE_NONE)

            # Benchmark NumPy
            np_times = []
            for _ in range(100):
                start = time.perf_counter()
                np_tokens = np.array(list_tokens, dtype=np.uint32)
                factory.new_request(np_tokens, REQUEST_FEATURE_NONE)
                end = time.perf_counter()
                np_times.append((end - start) * 1000)

            # Benchmark list (include creation time for fair comparison)
            list_times = []
            list_tokens = list(range(size))
            for _ in range(100):
                start = time.perf_counter()
                factory.new_request(list_tokens, REQUEST_FEATURE_NONE)
                end = time.perf_counter()
                list_times.append((end - start) * 1000)

            np_avg = np.mean(np_times)
            list_avg = np.mean(list_times)
            mb = (size * 4) / (1024 * 1024)

            print(f"\nSize: {size:>8,} ({mb:>6.2f} MB)")
            print(f"  NumPy array enabled_log={enabled_log}:  {np_avg:>6.3f} ms")
            print(f"  Python list enabled_log={enabled_log}:  {list_avg:>6.3f} ms")
            print(
                f"  Difference:   {abs(np_avg - list_avg):>6.3f} ms ({'NumPy' if np_avg < list_avg else 'list'} faster)"
            )


if __name__ == "__main__":
    benchmark_numpy_vs_list()
