# llm-runtime-metrics

Snapshot-first performance metrics and request tracing for LLM serving.

Packages:

- Rust crate: `llm-runtime-metrics`
- Python package: `llm-runtime-metrics`

The Rust crate is snapshot-first and Prometheus-agnostic.
The Python package exports snapshots to `prometheus_client` and provides
synchronous atomic histograms, counters, and fused stream-chunk recording.

Supported top-level API:

- Rust: `RequestMetricsFactory`, `RequestMetricsRequest`, snapshot types, request feature flags
- Python: `RequestMetricsFactory`, `RequestMetrics`, snapshot types, feature flags, and Prometheus helpers

Install:

- Rust (from a Dynamo crate): use a path dependency on
  `llm-runtime-metrics/crates/performance-metrics-core`.
- Python: `pip install llm-runtime-metrics`

Python import path:

- `import llm_runtime_metrics`

## Development in Dynamo

This directory is the source of truth for the package, migrated from
`basetenlabs/llm-runtime-metrics` at `1a47427fd388d5c2e17f842847e9588ec22902b4`.
It is an independent Cargo workspace with its own lockfile and version. The
Dynamo workspace consumes the Rust core by path; the Python extension and its
build dependencies stay in this workspace. Run these commands from this directory:

```sh
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked -p llm-runtime-metrics
uv venv
. .venv/bin/activate
uv pip install maturin pytest numpy pyarrow
maturin develop --locked --manifest-path bindings/python/Cargo.toml
python -m pytest -q -c bindings/python/pyproject.toml --confcutdir=bindings/python bindings/python/tests/
```

The `b10-metrics-*.yml` workflows in Dynamo's `.github/workflows` run the
Rust checks, Python tests, and existing platform wheel matrix on package changes.
Python publication is manual: dispatch `b10-metrics-build-release.yml` with
`publish_pypi=true` only for an approved release. It requires a `PYPI_TOKEN`
secret in Dynamo; normal PR/build jobs do not require that secret. Dynamo tags
and merges do not publish this package. No Rust publication is configured.
