# llm-runtime-metrics

Snapshot-first performance metrics and request tracing primitives.

Cargo package name: `llm-runtime-metrics`.

This crate is the Rust core for this repository. It keeps metrics collection
independent from any specific exporter and can be paired with the Python
bindings when Prometheus exposition is needed.

The supported root API is the request-metrics flow.
