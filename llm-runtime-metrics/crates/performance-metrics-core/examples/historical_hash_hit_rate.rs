use std::collections::HashSet;
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use arrow::array::{Array, ListArray, UInt64Array};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

const SHARD_COUNT: usize = 256;

#[derive(Debug, Clone, Copy)]
struct RequestMeta {
    timestamp_ms: u64,
    start: usize,
    len: usize,
}

#[derive(Debug, Clone, Copy)]
struct WeightedRate {
    rate: f64,
    weight: usize,
}

fn shard_index(value: u64) -> usize {
    (value as usize) & (SHARD_COUNT - 1)
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }

    let pos = q.clamp(0.0, 1.0) * ((sorted.len() - 1) as f64);
    let lower = pos.floor() as usize;
    let upper = pos.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let weight = pos - lower as f64;
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}

fn weighted_quantile(sorted: &[WeightedRate], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }

    let total_weight: usize = sorted.iter().map(|item| item.weight).sum();
    if total_weight == 0 {
        return f64::NAN;
    }

    let target = (q.clamp(0.0, 1.0) * (total_weight.saturating_sub(1)) as f64).floor() as usize;
    let mut cumulative = 0usize;
    for item in sorted {
        cumulative += item.weight;
        if cumulative > target {
            return item.rate;
        }
    }

    sorted.last().map(|item| item.rate).unwrap_or(f64::NAN)
}

fn parse_args() -> Result<PathBuf> {
    std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| {
            anyhow!(
                "usage: cargo run -p llm-runtime-metrics --example historical_hash_hit_rate -- <parquet>"
            )
        })
}

fn load_requests(path: &Path) -> Result<(Vec<RequestMeta>, Vec<u64>)> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("failed to create parquet reader for {}", path.display()))?;
    let reader = builder
        .with_batch_size(1024)
        .build()
        .context("failed to build parquet record batch reader")?;

    let mut requests = Vec::new();
    let mut flat_hashes = Vec::new();

    for batch in reader {
        let batch = batch.context("failed to read record batch")?;

        let ts_col = batch
            .column_by_name("timestamp_start_unix_ms")
            .ok_or_else(|| anyhow!("missing timestamp_start_unix_ms column"))?;
        let hashes_col = batch
            .column_by_name("total_hashes")
            .ok_or_else(|| anyhow!("missing total_hashes column"))?;

        let timestamps = ts_col
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| anyhow!("timestamp_start_unix_ms is not UInt64"))?;
        let hash_lists = hashes_col
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| anyhow!("total_hashes is not List"))?;

        for row_idx in 0..batch.num_rows() {
            if timestamps.is_null(row_idx) || hash_lists.is_null(row_idx) {
                continue;
            }

            let values = hash_lists.value(row_idx);
            let hashes = values
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| anyhow!("total_hashes values are not UInt64"))?;

            let start = flat_hashes.len();
            for hash_idx in 0..hashes.len() {
                if hashes.is_null(hash_idx) {
                    continue;
                }
                flat_hashes.push(hashes.value(hash_idx));
            }

            requests.push(RequestMeta {
                timestamp_ms: timestamps.value(row_idx),
                start,
                len: flat_hashes.len() - start,
            });
        }
    }

    Ok((requests, flat_hashes))
}

fn main() -> Result<()> {
    let input_path = parse_args()?;

    println!("Loading requests from {}...", input_path.display());
    let (mut requests, flat_hashes) = load_requests(&input_path)?;
    println!(
        "Loaded {} requests and {} total block hashes",
        requests.len(),
        flat_hashes.len()
    );

    println!("Sorting requests by timestamp...");
    requests.sort_unstable_by_key(|row| row.timestamp_ms);

    let mut seen_shards: Vec<HashSet<u64>> = (0..SHARD_COUNT).map(|_| HashSet::new()).collect();
    let mut request_rates = Vec::with_capacity(requests.len());
    let mut block_weighted_rates = Vec::with_capacity(requests.len());
    let mut total_hits = 0usize;
    let mut total_blocks = 0usize;

    for (idx, request) in requests.iter().enumerate() {
        let request_hashes = &flat_hashes[request.start..request.start + request.len];
        if request_hashes.is_empty() {
            continue;
        }

        let mut hits = 0usize;
        for &hash in request_hashes {
            let shard = shard_index(hash);
            if seen_shards[shard].contains(&hash) {
                hits += 1;
            }
        }

        let rate = hits as f64 / request_hashes.len() as f64;
        request_rates.push(rate);
        block_weighted_rates.push(WeightedRate {
            rate,
            weight: request_hashes.len(),
        });
        total_hits += hits;
        total_blocks += request_hashes.len();

        for &hash in request_hashes {
            let shard = shard_index(hash);
            seen_shards[shard].insert(hash);
        }

        if idx % 10_000 == 0 {
            println!("  processed {idx} requests");
        }
    }

    request_rates.sort_by(|a, b| a.total_cmp(b));
    block_weighted_rates.sort_by(|a, b| a.rate.total_cmp(&b.rate));

    println!("\nHistorical hash hit-rate summary");
    println!("  Requests: {}", request_rates.len());
    println!("  Total blocks: {}", total_blocks);
    println!("  Method: exact request-by-request replay over timestamp-sorted requests");

    println!("\nRequest-weighted");
    println!(
        "  Mean: {:.2}%",
        (request_rates.iter().sum::<f64>() / request_rates.len() as f64) * 100.0
    );
    println!("  P50: {:.2}%", quantile(&request_rates, 0.50) * 100.0);
    println!("  P95: {:.2}%", quantile(&request_rates, 0.95) * 100.0);
    println!("  P99: {:.2}%", quantile(&request_rates, 0.99) * 100.0);

    println!("\nBlock-weighted");
    println!(
        "  Mean: {:.2}%",
        (total_hits as f64 / total_blocks as f64) * 100.0
    );
    println!(
        "  P50: {:.2}%",
        weighted_quantile(&block_weighted_rates, 0.50) * 100.0
    );
    println!(
        "  P95: {:.2}%",
        weighted_quantile(&block_weighted_rates, 0.95) * 100.0
    );
    println!(
        "  P99: {:.2}%",
        weighted_quantile(&block_weighted_rates, 0.99) * 100.0
    );

    Ok(())
}
