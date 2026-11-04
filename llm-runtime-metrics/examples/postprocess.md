# Post-process a request trace dataset for Hugging Face

This tutorial shows the path from raw request trace parquet shards to a Hugging Face dataset upload that the Dataset Viewer can preview directly.

The example mirrors the `fld11-c25-single-worker-july` run:

- Single worker.
- Approximately 22 hours of recording.
- k2.6 architecture.
- B200x4 setup.
- Final Hugging Face dataset repo: `baseten/mp-hashed-k25-fld11-july-single-b200x4-agg`.

The final upload contains:

- `data/train-00000-of-00001.parquet`: the compacted anonymized parquet configured as the default `train` split.
- `README.md`: the dataset card and Dataset Viewer config.
- `plots/*.png`: optional quick-look plots.

## Prerequisites

Install the Hugging Face CLI and make sure the Python binding can import `llm_runtime_metrics`:

```bash
pip install -U huggingface_hub llm-runtime-metrics
hf auth login
```

For local development from this repository, use the existing Python environment or build the binding with maturin:

```bash
cd bindings/python
uv sync
uv run maturin develop
cd ../..
```

For the Rust examples:

```bash
cargo build --release -p llm-runtime-metrics --examples
```

## 1. Start from raw parquet shards

Raw recordings are many small parquet files written by the request tracer to S3. The post-processing flow starts from an S3 bucket and prefix, then produces one compacted anonymized parquet for local analysis and Hugging Face upload.

Use the repo helper to download, compact, and anonymize the S3 prefix in one pass:

```bash
S3_BUCKET="mp-hashed-requests"
S3_PREFIX="fld11-c25-single-worker-july"
ANON_PARQUET="data/${S3_PREFIX}.compacted.anonymized.parquet"

python examples/compact_s3_prefix.py \
  --bucket "$S3_BUCKET" \
  --prefix "$S3_PREFIX" \
  --region us-east-1 \
  --access-key-id "$AWS_ACCESS_KEY_ID" \
  --secret-access-key "$AWS_SECRET_ACCESS_KEY" \
  --output-path "$ANON_PARQUET" \
  --max-concurrent-downloads 32 \
  --anonymize
```

For the example run, this produced:

```text
Compacted 88 parquet objects
Wrote 30146 rows
Output: data/fld11-c25-single-worker-july.compacted.anonymized.parquet
```

## 2. Check the compacted trace

Use the Rust example to compute exact historical hash hit-rate metrics:

```bash
cargo run --release -p llm-runtime-metrics \
  --example historical_hash_hit_rate -- \
  data/fld11-c25-single-worker-july.compacted.anonymized.parquet
```

Example output:

```text
Loaded 30146 requests and 33668020 total block hashes

Request-weighted
  Mean: 83.97%
  P50: 96.49%
  P95: 99.65%
  P99: 99.83%

Block-weighted
  Mean: 91.25%
  P50: 98.05%
  P95: 99.75%
  P99: 99.87%
```

## 3. Optionally generate a completion dataset

The compacted trace contains hashed request-trace data. If you need synthetic prompt and completion text for downstream experiments, run `trace_to_completion_dataset`.

This step can produce a much larger parquet because it adds decoded `prompt` and `output_text` columns.

```bash
cargo run --release -p llm-runtime-metrics \
  --example trace_to_completion_dataset -- \
  --trace-parquet data/fld11-c25-single-worker-july.compacted.anonymized.parquet \
  --output-parquet data/fld11-c25-single-worker-july.completions.parquet \
  --tokenizer https://huggingface.co/gpt2/resolve/main/tokenizer.json \
  --tokenizer-backend huggingface \
  --num-files 8 \
  --max-rows-per-file 15000
```

For the example run:

```text
inferred_block_size=64
loaded_corpus_docs=93232
loaded_corpus_blocks=1163305
done: processed_rows=30146
wrote data/fld11-c25-single-worker-july.completions.parquet
```

This generated parquet was about 5.8 GB. It is useful for local analysis, but the public upload below uses the compacted anonymized parquet so the Hugging Face Dataset Viewer stays responsive.

### Choose a corpus in Rust

Without `--corpus-source`, reconstruction uses the existing FineWeb-Edu sample.
Repeat `--corpus-source` to replace it with local files or HTTP(S) URLs. Files
whose path ends in `.parquet` use the `text` column (override with
`--corpus-text-column`); other files are read as UTF-8 plain text, one document
per file. URL query strings do not affect format detection.

```bash
cargo run --release -p llm-runtime-metrics --example trace_to_completion_dataset -- \
  --trace-parquet trace.anonymized.parquet --output-parquet trace.completions.parquet \
  --tokenizer tokenizer.json \
  --corpus-source corpus/code.parquet \
  --corpus-source corpus/prose.txt
```

Custom documents are tokenized in Rust and split into chunks of at most 3,000
tokens, aligned to the trace's block size (or one block if it exceeds 3,000).
Every complete block is retained; only the incomplete tail of each source
document is dropped. Documents shorter than one block are skipped. No Python
preprocessing or character-count heuristic is required. Existing `token_count`
metadata is ignored for custom corpora because it may use a different tokenizer.
FineWeb retains its existing 250–3,000-token document filter.

`--max-rows-per-file` limits Parquet input rows (default 15,000), not generated
chunks. `--num-files` controls the default FineWeb shard count; it does not limit
explicit sources. `--corpus-shuffle-seed` controls corpus shuffling.

Corpus selection changes the text used to reconstruct hashed blocks. It does
not recover the original conversation or guarantee production speculative-decoding
acceptance. This command produces synthetic completion text; it does not apply
chat templates.

### Tokenizer backend

The Rust `fastokens-b10` backend is the default for both corpus encoding and
reconstruction decoding (`--tokenizer-backend fastokens`). Select
`--tokenizer-backend huggingface` for an unsupported legacy configuration, such
as the older GPT-2 tokenizer JSON used in the historical example above. Both
backends read `tokenizer.json`, disable automatic special-token insertion, and
preserve special tokens during decoding. Unsupported configurations return an
error rather than silently selecting another backend.
The fast backend is an example-only development dependency, so it does not add a
runtime dependency to the metrics library or Python bindings.

## 4. Generate optional plots

Plots are useful context in the dataset card, but they should not replace the main parquet split.

This snippet computes exact historical hash hit rates, 15-minute windows, and quick-look plots:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --with polars --with pyarrow --with matplotlib python - <<'PY'
from pathlib import Path
import math

import matplotlib.pyplot as plt
import polars as pl

source = Path("data/fld11-c25-single-worker-july.compacted.anonymized.parquet")
plot_dir = Path("hf_upload/mp-hashed-k25-fld11-july-single-b200x4-parquet/plots")
plot_dir.mkdir(parents=True, exist_ok=True)

base = (
    pl.scan_parquet(str(source))
    .select([
        "timestamp_start_unix_ms",
        "duration_e2e_ms",
        "duration_ttft_ms",
        "input_tokens",
        "output_tokens",
        "total_hashes",
        "total_hashes_len",
        "request_canceled",
    ])
    .with_columns([
        (pl.col("input_tokens") + pl.col("output_tokens")).alias("total_tokens"),
    ])
    .with_columns([
        (pl.col("timestamp_start_unix_ms") - pl.col("timestamp_start_unix_ms").min()).alias("elapsed_ms"),
    ])
    .with_columns([
        ((pl.col("elapsed_ms") // 900_000) * 900_000).alias("window_start_ms"),
    ])
    .sort("timestamp_start_unix_ms")
    .collect()
)

seen_shards = [set() for _ in range(256)]
request_hits = []
request_blocks = []
request_hit_rates = []
for hashes in base["total_hashes"]:
    if hashes is None or len(hashes) == 0:
        request_hits.append(0)
        request_blocks.append(0)
        request_hit_rates.append(None)
        continue

    hits = 0
    blocks = 0
    for h in hashes:
        if h is None:
            continue
        shard = int(h) & 255
        if h in seen_shards[shard]:
            hits += 1
        blocks += 1
    for h in hashes:
        if h is not None:
            seen_shards[int(h) & 255].add(h)

    request_hits.append(hits)
    request_blocks.append(blocks)
    request_hit_rates.append(hits / blocks if blocks else None)

base = base.with_columns([
    pl.Series("historical_hash_hits", request_hits, dtype=pl.UInt64),
    pl.Series("historical_hash_blocks", request_blocks, dtype=pl.UInt64),
    pl.Series("historical_hash_hit_rate", request_hit_rates, dtype=pl.Float64),
])

time_15m = (
    base.group_by("window_start_ms")
    .agg([
        pl.len().alias("requests"),
        pl.col("total_tokens").sum().alias("total_tokens"),
        pl.col("historical_hash_hits").sum().alias("historical_hash_hits"),
        pl.col("historical_hash_blocks").sum().alias("historical_hash_blocks"),
        pl.col("duration_ttft_ms").quantile(0.95).alias("duration_ttft_ms_p95"),
    ])
    .with_columns([
        (pl.col("window_start_ms") / 3_600_000.0).alias("window_start_hour"),
        (pl.col("historical_hash_hits") / pl.col("historical_hash_blocks")).alias("historical_hash_hit_rate"),
    ])
    .sort("window_start_ms")
)

def save_line(y_col, title, ylabel, filename, scale=1.0):
    x = time_15m["window_start_hour"].to_list()
    y = [(v * scale if v is not None else math.nan) for v in time_15m[y_col].to_list()]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(x, y, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Elapsed hours")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / filename, dpi=160)
    plt.close(fig)

save_line("requests", "Requests per 15-minute window", "Requests", "requests_per_15m.png")
save_line("total_tokens", "Token volume per 15-minute window", "Tokens", "tokens_per_15m.png")
save_line("historical_hash_hit_rate", "Historical hash hit rate by 15-minute window", "Hit rate (%)", "historical_hash_hit_rate_per_15m.png", 100.0)
save_line("duration_ttft_ms_p95", "P95 TTFT by 15-minute window", "P95 TTFT (ms)", "p95_ttft_per_15m.png")

rates = [v * 100.0 for v in base["historical_hash_hit_rate"].to_list() if v is not None]
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.hist(rates, bins=40)
ax.set_title("Request historical hash hit-rate distribution")
ax.set_xlabel("Hit rate (%)")
ax.set_ylabel("Requests")
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(plot_dir / "historical_hash_hit_rate_histogram.png", dpi=160)
plt.close(fig)
PY
```

## 5. Create the Hugging Face upload folder

Use the standard Hugging Face dataset layout:

```text
hf_upload/mp-hashed-k25-fld11-july-single-b200x4-parquet/
  README.md
  .gitattributes
  data/
    train-00000-of-00001.parquet
  plots/
    requests_per_15m.png
    tokens_per_15m.png
    historical_hash_hit_rate_per_15m.png
    p95_ttft_per_15m.png
    historical_hash_hit_rate_histogram.png
```

Copy the main parquet into the viewer path:

```bash
UPLOAD_DIR="hf_upload/mp-hashed-k25-fld11-july-single-b200x4-parquet"

mkdir -p "$UPLOAD_DIR/data" "$UPLOAD_DIR/plots"
cp data/fld11-c25-single-worker-july.compacted.anonymized.parquet \
  "$UPLOAD_DIR/data/train-00000-of-00001.parquet"
```

Add `.gitattributes`:

```bash
cat > "$UPLOAD_DIR/.gitattributes" <<'EOF'
*.parquet filter=lfs diff=lfs merge=lfs -text
*.png filter=lfs diff=lfs merge=lfs -text
*.md text
EOF
```

Add a dataset card:

```bash
cat > "$UPLOAD_DIR/README.md" <<'EOF'
---
pretty_name: MP Hashed K25 FLD11 July Single B200x4
language:
- en
license: apache-2.0
tags:
- llm-runtime
- inference
- performance
- parquet
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-00000-of-00001.parquet
---

# MP Hashed K25 FLD11 July Single B200x4

This dataset contains the compacted anonymized parquet trace from a single-worker run over approximately 22 hours. The run used the k2.6 architecture on a B200x4 setup.

The upload is centered on the main parquet file, with plots included for quick visual inspection. The parquet is configured as the default `train` split so the Hugging Face dataset viewer previews it directly.

The primary file is:

`data/train-00000-of-00001.parquet`

## Summary

- Requests: 30,146
- Trace duration: 22.37 hours
- Input tokens: 2,137,804,121
- Output tokens: 15,994,929
- Total tokens: 2,153,799,050
- Historical hash blocks: 33,668,020
- Historical hash hits: 30,722,127
- Block-weighted historical hash hit rate: 91.25%
- Request-weighted historical hash hit rate mean: 83.97%
- Request-weighted historical hash hit rate p50 / p95 / p99: 96.49% / 99.65% / 99.83%
- End-to-end latency p50 / p95 / p99: 1,842 ms / 11,533 ms / 27,326 ms
- TTFT p50 / p95 / p99: 430 ms / 1,869 ms / 8,326 ms
- Canceled requests: 79

## Files

- `data/train-00000-of-00001.parquet`: compacted anonymized trace, configured as the default `train` split for the Hugging Face dataset viewer.
- `plots/*.png`: quick-look plots for request rate, token volume, historical hash hit rate, TTFT, and hit-rate distribution.

## Plots

![Requests per 15m](plots/requests_per_15m.png)

![Tokens per 15m](plots/tokens_per_15m.png)

![Historical hash hit rate per 15m](plots/historical_hash_hit_rate_per_15m.png)

![P95 TTFT per 15m](plots/p95_ttft_per_15m.png)

![Historical hash hit-rate histogram](plots/historical_hash_hit_rate_histogram.png)

## Method

The historical hash hit-rate metrics shown here were computed with an exact timestamp-ordered replay over request `total_hashes`. A hash counts as a hit when it has appeared in an earlier request. The block-weighted metric divides total hits by total blocks; request-weighted metrics aggregate per-request hit rates.

The source S3 prefix contained 88 parquet files recorded on a mostly regular 15-minute cadence from 2026-07-08 16:57:19 UTC through 2026-07-09 15:12:20 UTC. These files were compacted and anonymized into the primary parquet in this dataset.
EOF
```

The `configs` block is the important part for the Hugging Face Dataset Viewer. It tells Hugging Face that `data/train-00000-of-00001.parquet` is the default `train` split.

## 6. Upload to Hugging Face

Upload the prepared folder:

```bash
HF_REPO="baseten/mp-hashed-k25-fld11-july-single-b200x4-agg"
UPLOAD_DIR="hf_upload/mp-hashed-k25-fld11-july-single-b200x4-parquet"

hf upload "$HF_REPO" \
  "$UPLOAD_DIR" . \
  --repo-type dataset \
  --token "$HF_TOKEN" \
  --delete "*" \
  --commit-message "Use compacted parquet as default split with plots"
```

Use `--delete "*"` when replacing an earlier upload so stale aggregate files do not remain in the repo. If you are uploading to a fresh repo, you can omit it.

## 7. Verify the upload

After upload, the repo should contain:

```text
README.md
.gitattributes
data/train-00000-of-00001.parquet
plots/*.png
```

Open the dataset page and check:

- The Dataset Viewer shows a `train` split.
- The preview table is backed by `data/train-00000-of-00001.parquet`.
- The README renders the plots.
- No large local-only analysis file, such as a generated completions parquet, was uploaded by accident.

For this example, the final main parquet was about 83 MB. The optional completions parquet was about 5.8 GB and was kept local.
