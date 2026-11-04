from __future__ import annotations

import time
from pathlib import Path

import pytest
from llm_runtime_metrics import RequestMetricsFactory


def _rows_persisted(factory: RequestMetricsFactory) -> int:
    return factory.snapshot().totals().total_request_log_rows_persisted


def _finished_request(factory: RequestMetricsFactory) -> list[bytes]:
    req = factory.new_request(list(range(130)))
    req.record_tokens([1, 2, 3])
    req.success()
    return req.total_hashes()


def test_configure_request_log_toggles_without_resetting_metrics(
    tmp_path: Path,
) -> None:
    factory = RequestMetricsFactory()
    assert not factory.request_log_enabled
    assert _finished_request(factory) == []

    factory.configure_request_log(
        True, request_log_batch_size=1, request_log_output_dir=str(tmp_path)
    )
    with pytest.raises(ValueError, match="s3_bucket"):
        factory.configure_request_log(True, s3=True)
    with pytest.raises(ValueError, match="file_format"):
        factory.configure_request_log(True, file_format="csv")
    assert factory.request_log_enabled  # the rejected settings kept the current one
    assert len(_finished_request(factory)) == 3
    deadline = time.monotonic() + 5
    while _rows_persisted(factory) < 1:
        assert time.monotonic() < deadline, "timed out waiting for the row"
        time.sleep(0.01)

    factory.configure_request_log(False)
    assert not factory.request_log_enabled
    assert _finished_request(factory) == []
    time.sleep(0.1)
    assert _rows_persisted(factory) == 1
    # Request metrics kept counting through every swap.
    assert factory.snapshot().totals().total_requests == 3
