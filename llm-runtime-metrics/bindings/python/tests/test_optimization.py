#!/usr/bin/env python3
"""Tests for the automatic optimization in new_request().

When request_log_enabled=False, new_request() should automatically
use the fast path (just length) instead of copying the full array.
"""

import time

import numpy as np
import pytest
from llm_runtime_metrics import REQUEST_FEATURE_NONE, RequestMetricsFactory


class TestAutomaticOptimization:
    """Test that new_request() automatically optimizes when logging is disabled."""

    def test_fast_path_with_logging_disabled(self):
        """When logging is disabled, should use fast path (no array copy)."""
        factory = RequestMetricsFactory(
            request_log_enabled=False,
        )

        # Create a large array
        size = 100_000
        tokens = np.arange(size, dtype=np.uint32)

        # Warmup
        for _ in range(10):
            factory.new_request(tokens, REQUEST_FEATURE_NONE)

        # Benchmark - should be very fast (<0.1 ms)
        times = []
        for _ in range(100):
            start = time.perf_counter()
            factory.new_request(tokens, REQUEST_FEATURE_NONE)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        avg_time = np.mean(times)
        # Fast path should be <0.2 ms, slow path would be ~1-2 ms
        assert avg_time < 0.2, f"Fast path not working! Time: {avg_time:.3f} ms"

    def test_full_path_with_logging_enabled(self):
        """When logging is enabled, should use full path (array copy + hash)."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            factory = RequestMetricsFactory(
                request_log_enabled=True,
                request_log_output_dir=tmpdir,
                block_size=64,  # Small block size to ensure hashing happens
            )

            size = 100_000
            tokens = np.arange(size, dtype=np.uint32)

            # Warmup
            for _ in range(5):
                factory.new_request(tokens, REQUEST_FEATURE_NONE)

            # Benchmark - should be slower (~1-2 ms)
            times = []
            for _ in range(50):
                start = time.perf_counter()
                factory.new_request(tokens, REQUEST_FEATURE_NONE)
                end = time.perf_counter()
                times.append((end - start) * 1000)

            avg_time = np.mean(times)
            # Full path should be >0.5 ms (much slower than fast path)
            assert avg_time > 0.5, f"Full path not working! Time: {avg_time:.3f} ms"

    def test_with_list_vs_numpy(self):
        """Both list and NumPy should work with the fast path."""
        factory = RequestMetricsFactory(
            request_log_enabled=False,
            block_size=100_000_000,
        )

        size = 10_000
        np_tokens = np.arange(size, dtype=np.uint32)
        list_tokens = list(range(size))

        # Both should work
        req1 = factory.new_request(np_tokens, REQUEST_FEATURE_NONE)
        req2 = factory.new_request(list_tokens, REQUEST_FEATURE_NONE)

        # Both should record success without errors
        req1.success()
        req2.success()

    def test_metrics_still_work_with_fast_path(self):
        """Ensure metrics are still recorded correctly with fast path."""
        factory = RequestMetricsFactory(
            request_log_enabled=False,
            block_size=100_000_000,
        )

        size = 1000
        tokens = np.arange(size, dtype=np.uint32)

        # Create and complete a request
        req = factory.new_request(tokens, REQUEST_FEATURE_NONE)
        req.success()

        # Check metrics were recorded
        snapshot = factory.snapshot()
        totals = snapshot.totals()

        assert totals.total_requests == 1
        assert totals.total_successful_requests == 1
        assert totals.total_input_tokens == size


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_array(self):
        """Empty array should work."""
        factory = RequestMetricsFactory(
            request_log_enabled=False,
            block_size=100_000_000,
        )

        tokens = np.array([], dtype=np.uint32)
        req = factory.new_request(tokens, REQUEST_FEATURE_NONE)
        req.success()

        snapshot = factory.snapshot()
        assert snapshot.totals().total_input_tokens == 0

    def test_large_block_size_no_hashing(self):
        """With large block_size, no hashing should occur."""
        factory = RequestMetricsFactory(
            request_log_enabled=False,
            block_size=100_000_000,  # Larger than any request
        )

        size = 100_000
        tokens = np.arange(size, dtype=np.uint32)
        req = factory.new_request(tokens, REQUEST_FEATURE_NONE)

        # With large block size, no hashes should be generated
        hashes = req.total_hashes()
        assert (
            len(hashes) == 0
        ), f"Expected 0 hashes with large block_size, got {len(hashes)}"

    def test_small_block_size_with_hashing(self):
        """With small block_size, hashing should occur."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            factory = RequestMetricsFactory(
                request_log_enabled=True,
                request_log_output_dir=tmpdir,
                block_size=100,  # Small blocks
            )

            size = 1000
            tokens = np.arange(size, dtype=np.uint32)
            req = factory.new_request(tokens, REQUEST_FEATURE_NONE)

            # With small block size, hashes should be generated
            hashes = req.total_hashes()
            expected_hashes = (size + 99) // 100  # Ceiling division
            assert (
                len(hashes) >= expected_hashes - 1
            ), f"Expected ~{expected_hashes} hashes, got {len(hashes)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
