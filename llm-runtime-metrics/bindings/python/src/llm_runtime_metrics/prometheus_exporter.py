from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._core import AtomicCounter, RequestMetricsFactory


class RequestMetricsCollector:
    """Prometheus collector that pulls typed samples from RequestMetricsFactory."""

    def __init__(
        self,
        factory: RequestMetricsFactory,
        *,
        base_labels: Mapping[str, str] | None = None,
        registry: Any | None = None,
    ) -> None:
        from prometheus_client import REGISTRY

        self.factory = factory
        self.base_labels = dict(base_labels or {})
        self.registry = registry or REGISTRY
        self.registry.register(self)

    def collect(self):
        from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily

        families: dict[tuple[str, tuple[str, ...], str], object] = {}
        for sample in self.factory.prometheus_samples(self.base_labels):
            label_names = tuple(sorted(sample.labels.keys()))
            key = (sample.name, label_names, sample.metric_type)
            family = families.get(key)
            if family is None:
                if sample.metric_type == "counter":
                    family = CounterMetricFamily(
                        sample.name, sample.name, labels=list(label_names)
                    )
                else:
                    family = GaugeMetricFamily(
                        sample.name, sample.name, labels=list(label_names)
                    )
                families[key] = family
            label_values = [sample.labels[name] for name in label_names]
            family.add_metric(label_values, sample.value)

        yield from families.values()


def prometheus_strfmt(
    factory: RequestMetricsFactory,
    *,
    base_labels: Mapping[str, str] | None = None,
) -> str:
    """Return Prometheus text exposition from the Rust encoder path."""
    return factory.prometheus_strfmt(dict(base_labels or {}))


class AtomicCounterCollector:
    """Labeled integer counters backed by Rust atomics, exported at scrape time.

    Resolve labels once per request and pass the returned handle to
    StreamChunkMetrics. As with prometheus_client.Counter, series are retained.
    """

    def __init__(
        self,
        name: str,
        documentation: str,
        labelnames: list[str],
        *,
        registry: Any,
    ) -> None:
        import threading

        self.name = name.removesuffix("_total")
        self.documentation = documentation
        self.labelnames = tuple(labelnames)
        self._labelname_set = frozenset(labelnames)
        self._series: dict[tuple[str, ...], tuple[AtomicCounter, float]] = {}
        self._lock = threading.Lock()
        registry.register(self)

    def labels(self, **labels: str) -> AtomicCounter:
        import time

        if labels.keys() != self._labelname_set:
            raise ValueError(f"Incorrect labels for {self.name}")
        key = tuple(str(labels[name]) for name in self.labelnames)
        with self._lock:
            entry = self._series.get(key)
            if entry is None:
                entry = (AtomicCounter(), time.time())
                self._series[key] = entry
            return entry[0]

    def describe(self):
        from prometheus_client.core import CounterMetricFamily

        yield CounterMetricFamily(self.name, self.documentation, labels=self.labelnames)

    def collect(self):
        from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily
        from prometheus_client.metrics import _use_created

        counter = CounterMetricFamily(
            self.name, self.documentation, labels=self.labelnames
        )
        created = GaugeMetricFamily(
            f"{self.name}_created", self.documentation, labels=self.labelnames
        )
        with self._lock:
            series = list(self._series.items())
        for labels, (value, timestamp) in series:
            counter.add_metric(labels, value.get())
            created.add_metric(labels, timestamp)
        yield counter
        if _use_created:
            yield created
