"""Compare cached Python observations with one fused native call (no task overhead)."""

import statistics
import timeit

from llm_runtime_metrics import AtomicCounter, AtomicHistogram, StreamChunkMetrics
from prometheus_client import CollectorRegistry, Counter


def main():
    ttft, tpot, itl = [AtomicHistogram.from_vm(-4, 10, 12) for _ in range(3)]
    counter = Counter("bench_tokens", "Tokens", registry=CollectorRegistry())
    stream = StreamChunkMetrics(ttft, tpot, itl, AtomicCounter())

    def cached_python():
        seconds_per_token = 0.01 / 4
        tpot.observe(seconds_per_token)
        itl.observe_many(seconds_per_token, 4)
        counter.inc(4)

    def fused_rust():
        stream.record(4, None, 0.01)

    for fn in (cached_python, fused_rust):
        elapsed = statistics.median(timeit.repeat(fn, number=200_000, repeat=7))
        print(f"{fn.__name__}: {elapsed / 200_000 * 1e6:.3f} us/chunk")


if __name__ == "__main__":
    main()
