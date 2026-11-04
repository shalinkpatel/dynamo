use std::collections::HashMap;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use arrow::array::{Array, ArrayRef, ListArray, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use futures::stream::{self, StreamExt, TryStreamExt};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use parquet::file::reader::FileReader;
use rand::prelude::SliceRandom;
use rand::{SeedableRng, rngs::StdRng};
#[path = "support/reconstruction_tokenizer.rs"]
mod reconstruction_tokenizer;
use reconstruction_tokenizer::{Backend, Tokenizer};

const DEFAULT_CACHE_DIR: &str = ".cache/synthetic-trace";
const DEFAULT_CORPUS_TEXT_COLUMN: &str = "text";
const DEFAULT_CORPUS_TOKEN_COUNT_COLUMN: &str = "token_count";
const DEFAULT_CORPUS_SHUFFLE_SEED: u64 = 0;
const DEFAULT_FINEWEB_SAMPLE_NUM_FILES: usize = 48;
const DEFAULT_FINEWEB_SAMPLE_ROWS_PER_FILE: usize = 15_000;
const DEFAULT_CORPUS_FILE_THREADS: usize = 8;
const DEFAULT_CORPUS_BATCH_QUEUE_CAPACITY: usize = 2;
const DEFAULT_BOUNDARY_SHIFT_TOKENS: usize = 5;
const DEFAULT_TRACE_BATCH_SIZE: usize = 2048;
const DEFAULT_OUTPUT_WRITE_QUEUE_CAPACITY: usize = 4;
const FINEWEB_MIN_TOKENS: usize = 250;
const FINEWEB_MAX_TOKENS: usize = 3000;

#[derive(Debug, Default)]
struct Args {
    trace_parquet: String,
    output_parquet: String,
    tokenizer: String,
    tokenizer_backend: Backend,
    corpus_text_column: String,
    cache_dir: String,
    max_rows: Option<usize>,
    max_rows_per_file: usize,
    num_files: usize,
    corpus_shuffle_seed: u64,
    seed_first_block_text: Option<String>,
    corpus_sources: Vec<String>,
}

#[derive(Debug)]
struct CorpusDocument {
    tokens: Vec<u32>,
    block_count: usize,
}

#[derive(Debug)]
struct CorpusSupply {
    documents: Vec<CorpusDocument>,
    order: Vec<usize>,
    doc_cursor: usize,
    block_cursor: usize,
    rng: StdRng,
    block_size: usize,
    epoch: usize,
    total_blocks: usize,
    epoch_blocks_used: usize,
}

enum CorpusMessage {
    Documents(Vec<CorpusDocument>),
}

fn parse_args() -> Result<Args> {
    let mut args = Args {
        corpus_text_column: DEFAULT_CORPUS_TEXT_COLUMN.to_string(),
        cache_dir: DEFAULT_CACHE_DIR.to_string(),
        max_rows: None,
        max_rows_per_file: DEFAULT_FINEWEB_SAMPLE_ROWS_PER_FILE,
        num_files: DEFAULT_FINEWEB_SAMPLE_NUM_FILES,
        corpus_shuffle_seed: DEFAULT_CORPUS_SHUFFLE_SEED,
        seed_first_block_text: None,
        ..Args::default()
    };

    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--trace-parquet" => {
                args.trace_parquet = iter
                    .next()
                    .ok_or_else(|| anyhow!("missing value for --trace-parquet"))?
            }
            "--output-parquet" => {
                args.output_parquet = iter
                    .next()
                    .ok_or_else(|| anyhow!("missing value for --output-parquet"))?
            }
            "--tokenizer" => {
                args.tokenizer = iter
                    .next()
                    .ok_or_else(|| anyhow!("missing value for --tokenizer"))?
            }
            "--tokenizer-backend" => {
                args.tokenizer_backend = iter
                    .next()
                    .ok_or_else(|| anyhow!("missing value for --tokenizer-backend"))?
                    .parse()?;
            }
            "--corpus-source" => args.corpus_sources.push(
                iter.next()
                    .ok_or_else(|| anyhow!("missing value for --corpus-source"))?,
            ),
            "--corpus-text-column" => {
                args.corpus_text_column = iter
                    .next()
                    .ok_or_else(|| anyhow!("missing value for --corpus-text-column"))?
            }
            "--cache-dir" => {
                args.cache_dir = iter
                    .next()
                    .ok_or_else(|| anyhow!("missing value for --cache-dir"))?
            }
            "--max-rows" => {
                args.max_rows = Some(
                    iter.next()
                        .ok_or_else(|| anyhow!("missing value for --max-rows"))?
                        .parse()
                        .context("invalid --max-rows")?,
                )
            }
            "--max-rows-per-file" => {
                args.max_rows_per_file = iter
                    .next()
                    .ok_or_else(|| anyhow!("missing value for --max-rows-per-file"))?
                    .parse()
                    .context("invalid --max-rows-per-file")?
            }
            "--num-files" => {
                args.num_files = iter
                    .next()
                    .ok_or_else(|| anyhow!("missing value for --num-files"))?
                    .parse()
                    .context("invalid --num-files")?
            }
            "--corpus-shuffle-seed" => {
                args.corpus_shuffle_seed = iter
                    .next()
                    .ok_or_else(|| anyhow!("missing value for --corpus-shuffle-seed"))?
                    .parse()
                    .context("invalid --corpus-shuffle-seed")?
            }
            "--seed-first-block-text" => {
                args.seed_first_block_text = Some(
                    iter.next()
                        .ok_or_else(|| anyhow!("missing value for --seed-first-block-text"))?,
                )
            }
            other => bail!("unknown argument: {other}"),
        }
    }

    if args.trace_parquet.is_empty() || args.output_parquet.is_empty() || args.tokenizer.is_empty()
    {
        bail!(
            "usage: cargo run -p llm-runtime-metrics --example trace_to_completion_dataset -- \
--trace-parquet <path> --output-parquet <path> --tokenizer <path-or-url> \
[--corpus-source <parquet-or-text-path-or-url>]... [--corpus-text-column text] \
[--tokenizer-backend huggingface|fastokens]"
        );
    }

    Ok(args)
}

fn selected_corpus_sources(args: &Args) -> Vec<String> {
    if args.corpus_sources.is_empty() {
        default_fineweb_edu_sample_urls(args.num_files)
    } else {
        args.corpus_sources.clone()
    }
}

fn default_fineweb_edu_sample_urls(num_files: usize) -> Vec<String> {
    let mut urls = Vec::with_capacity(num_files);
    let mut shard_group = 0usize;
    let mut shard_part = 0usize;
    while urls.len() < num_files {
        urls.push(format!(
            "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/100BT/{shard_group:03}_{shard_part:05}.parquet?download=true"
        ));
        shard_part += 1;
        if shard_part == 10 {
            shard_part = 0;
            shard_group += 1;
        }
    }
    urls
}

fn is_url(value: &str) -> bool {
    value.starts_with("http://") || value.starts_with("https://")
}

async fn resolve_source(source: &str, cache_dir: &Path) -> Result<PathBuf> {
    if !is_url(source) {
        return Ok(PathBuf::from(source));
    }

    fs::create_dir_all(cache_dir)
        .with_context(|| format!("failed to create cache dir {}", cache_dir.display()))?;
    let source_path = source
        .split('?')
        .next()
        .unwrap_or(source)
        .split('#')
        .next()
        .unwrap_or(source);
    let ext = Path::new(source_path)
        .extension()
        .and_then(|x| x.to_str())
        .unwrap_or("bin");
    let key = blake3::hash(source.as_bytes()).to_hex().to_string();
    let cached = cache_dir.join(format!("{key}.{ext}"));
    if cached.exists() {
        return Ok(cached);
    }

    println!("downloading {source}");
    let bytes = reqwest::get(source)
        .await
        .with_context(|| format!("failed to GET {source}"))?
        .error_for_status()
        .with_context(|| format!("unexpected status for {source}"))?
        .bytes()
        .await
        .with_context(|| format!("failed to read body from {source}"))?;

    fs::write(&cached, &bytes)
        .with_context(|| format!("failed to write cached file {}", cached.display()))?;
    Ok(cached)
}

fn progress_style(template: &str) -> Result<ProgressStyle> {
    ProgressStyle::with_template(template).map_err(|e| anyhow!("invalid progress template: {e}"))
}

fn encode_text(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    tokenizer.encode(text)
}

fn decode_tokens(tokenizer: &Tokenizer, ids: &[u32]) -> Result<String> {
    tokenizer.decode(ids)
}

fn is_safe_boundary_char(c: char) -> bool {
    c.is_whitespace()
        || matches!(
            c,
            '.' | ',' | ';' | ':' | '!' | '?' | ')' | ']' | '}' | '"' | '\''
        )
}

fn choose_adjusted_split(tokenizer: &Tokenizer, full_ids: &[u32], nominal_split: usize) -> usize {
    let max_shift = DEFAULT_BOUNDARY_SHIFT_TOKENS.min(full_ids.len().saturating_sub(nominal_split));

    for shift in 0..=max_shift {
        let candidate = nominal_split + shift;
        if candidate >= full_ids.len() {
            return nominal_split.min(full_ids.len());
        }

        let left_start = candidate.saturating_sub(8);
        let right_end = (candidate + 8).min(full_ids.len());
        let left = decode_tokens(tokenizer, &full_ids[left_start..candidate]).ok();
        let right = decode_tokens(tokenizer, &full_ids[candidate..right_end]).ok();

        let left_ok = left
            .as_deref()
            .and_then(|s| s.chars().last())
            .map(is_safe_boundary_char)
            .unwrap_or(false);
        let right_ok = right
            .as_deref()
            .and_then(|s| s.chars().next())
            .map(is_safe_boundary_char)
            .unwrap_or(false);

        if left_ok || right_ok {
            return candidate;
        }
    }

    nominal_split.min(full_ids.len())
}

fn infer_trace_block_size(path: &Path) -> Result<usize> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("failed to create parquet reader for {}", path.display()))?;
    let mut reader = builder
        .with_batch_size(1)
        .build()
        .context("failed to build trace reader")?;

    if let Some(batch) = reader.next() {
        let batch = batch.context("failed to read trace batch")?;
        let col = batch
            .column_by_name("block_size")
            .ok_or_else(|| anyhow!("trace parquet missing block_size column"))?;
        let values = col
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| anyhow!("block_size column is not UInt64"))?;
        if values.is_empty() || values.is_null(0) {
            bail!("unable to read block_size from first trace row");
        }
        Ok(values.value(0) as usize)
    } else {
        bail!("trace parquet is empty")
    }
}

fn estimate_corpus_total_entries(sources: &[PathBuf], max_rows_per_file: usize) -> u64 {
    let mut total = 0u64;
    for source in sources {
        let ext = source
            .extension()
            .and_then(|x| x.to_str())
            .unwrap_or_default();
        if ext.eq_ignore_ascii_case("parquet")
            && let Ok(file) = File::open(source)
            && let Ok(reader) = parquet::file::reader::SerializedFileReader::new(file)
        {
            let rows = reader.metadata().file_metadata().num_rows() as u64;
            let capped = rows.min(max_rows_per_file as u64);
            total = total.saturating_add(capped);
            continue;
        }
        total = total.saturating_add(1.min(max_rows_per_file as u64));
    }
    total
}

fn estimate_trace_total_rows(path: &Path, max_rows: Option<usize>) -> u64 {
    let rows = if let Ok(file) = File::open(path) {
        if let Ok(reader) = parquet::file::reader::SerializedFileReader::new(file) {
            reader.metadata().file_metadata().num_rows() as u64
        } else {
            0
        }
    } else {
        0
    };
    match max_rows {
        Some(limit) => rows.min(limit as u64),
        None => rows,
    }
}

// FineWeb retains its sampling filter. Custom inputs use all complete blocks and
// split long documents after tokenization, without a character-count heuristic.
#[derive(Clone, Copy)]
enum CorpusMode {
    FineWeb,
    Custom,
}

fn corpus_documents(
    mut ids: Vec<u32>,
    block_size: usize,
    mode: CorpusMode,
) -> Result<Vec<CorpusDocument>> {
    if block_size == 0 {
        bail!("corpus block size must be positive");
    }
    if matches!(mode, CorpusMode::FineWeb)
        && !(FINEWEB_MIN_TOKENS..=FINEWEB_MAX_TOKENS).contains(&ids.len())
    {
        return Ok(Vec::new());
    }
    ids.truncate(ids.len() / block_size * block_size);
    let chunk_size = match mode {
        CorpusMode::FineWeb => ids.len().max(1),
        CorpusMode::Custom => (FINEWEB_MAX_TOKENS / block_size).max(1) * block_size,
    };
    Ok(ids
        .chunks(chunk_size)
        .map(|tokens| CorpusDocument {
            block_count: tokens.len() / block_size,
            tokens: tokens.to_vec(),
        })
        .collect())
}

fn append_corpus_text(
    tokenizer: &Tokenizer,
    text: &str,
    block_size: usize,
    mode: CorpusMode,
) -> Result<Vec<CorpusDocument>> {
    corpus_documents(encode_text(tokenizer, text)?, block_size, mode)
}

#[allow(clippy::too_many_arguments)]
fn load_corpus_from_text_file(
    tokenizer: &Tokenizer,
    path: &Path,
    block_size: usize,
    mode: CorpusMode,
    docs_tx: &mpsc::Sender<CorpusMessage>,
    scanned_counter: &Arc<AtomicU64>,
    added_counter: &Arc<AtomicU64>,
    skipped_counter: &Arc<AtomicU64>,
) -> Result<usize> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read text corpus {}", path.display()))?;
    scanned_counter.fetch_add(1, Ordering::Relaxed);
    let docs = append_corpus_text(tokenizer, &text, block_size, mode)?;
    if docs.is_empty() {
        skipped_counter.fetch_add(1, Ordering::Relaxed);
        return Ok(0);
    }
    let blocks = docs.iter().map(|doc| doc.block_count).sum();
    docs_tx
        .send(CorpusMessage::Documents(docs))
        .map_err(|_| anyhow!("corpus collector disconnected"))?;
    added_counter.fetch_add(1, Ordering::Relaxed);
    Ok(blocks)
}

#[allow(clippy::too_many_arguments)]
fn load_corpus_from_parquet(
    tokenizer: &Tokenizer,
    path: &Path,
    text_column: &str,
    block_size: usize,
    mode: CorpusMode,
    docs_tx: &mpsc::Sender<CorpusMessage>,
    max_rows_per_file: usize,
    scanned_counter: &Arc<AtomicU64>,
    added_counter: &Arc<AtomicU64>,
    skipped_counter: &Arc<AtomicU64>,
) -> Result<usize> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("failed to create parquet reader for {}", path.display()))?;
    let reader = builder
        .with_batch_size(4096)
        .build()
        .context("failed to build corpus parquet reader")?;

    let (work_tx, work_rx) = mpsc::sync_channel::<Vec<String>>(DEFAULT_CORPUS_BATCH_QUEUE_CAPACITY);
    let (result_tx, result_rx) = mpsc::channel::<Result<(Vec<CorpusDocument>, usize)>>();
    let local_tokenizer = tokenizer.clone();
    let worker_added = Arc::clone(added_counter);
    let worker_skipped = Arc::clone(skipped_counter);

    let worker = thread::spawn(move || {
        while let Ok(texts) = work_rx.recv() {
            let mut local_documents = Vec::with_capacity(texts.len());
            let mut local_total_blocks = 0usize;
            for text in texts {
                match append_corpus_text(&local_tokenizer, &text, block_size, mode) {
                    Ok(mut docs) => {
                        if docs.is_empty() {
                            worker_skipped.fetch_add(1, Ordering::Relaxed);
                        } else {
                            local_total_blocks +=
                                docs.iter().map(|doc| doc.block_count).sum::<usize>();
                            worker_added.fetch_add(1, Ordering::Relaxed);
                            local_documents.append(&mut docs);
                        }
                    }
                    Err(err) => {
                        let _ = result_tx.send(Err(err));
                        return;
                    }
                }
            }
            if result_tx
                .send(Ok((local_documents, local_total_blocks)))
                .is_err()
            {
                return;
            }
        }
    });

    let mut submitted_batches = 0usize;
    let mut rows_seen = 0usize;
    for batch in reader {
        if rows_seen >= max_rows_per_file {
            break;
        }
        let batch = batch.context("failed to read corpus batch")?;
        let col = batch
            .column_by_name(text_column)
            .ok_or_else(|| anyhow!("corpus parquet missing text column '{text_column}'"))?;
        let texts = col
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow!("corpus text column '{text_column}' is not Utf8"))?;
        let token_count_col = batch.column_by_name(DEFAULT_CORPUS_TOKEN_COUNT_COLUMN);
        let token_counts =
            token_count_col.and_then(|col| col.as_any().downcast_ref::<UInt64Array>());

        let mut candidate_texts = Vec::new();
        for row_idx in 0..texts.len() {
            if rows_seen >= max_rows_per_file {
                break;
            }
            rows_seen += 1;
            if texts.is_null(row_idx) {
                continue;
            }
            scanned_counter.fetch_add(1, Ordering::Relaxed);
            if let Some(token_counts) = token_counts.filter(|_| matches!(mode, CorpusMode::FineWeb))
            {
                if token_counts.is_null(row_idx) {
                    skipped_counter.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                let token_count = token_counts.value(row_idx) as usize;
                if !(FINEWEB_MIN_TOKENS..=FINEWEB_MAX_TOKENS).contains(&token_count) {
                    skipped_counter.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
            }
            candidate_texts.push(texts.value(row_idx).to_owned());
        }

        if !candidate_texts.is_empty() {
            work_tx
                .send(candidate_texts)
                .map_err(|_| anyhow!("corpus tokenizer worker disconnected"))?;
            submitted_batches += 1;
        }
    }
    drop(work_tx);

    let mut total_blocks = 0usize;
    for _ in 0..submitted_batches {
        let (mut local_documents, local_total_blocks) = result_rx
            .recv()
            .map_err(|_| anyhow!("corpus tokenizer worker disconnected"))??;
        total_blocks += local_total_blocks;
        docs_tx
            .send(CorpusMessage::Documents(std::mem::take(
                &mut local_documents,
            )))
            .map_err(|_| anyhow!("corpus collector disconnected"))?;
    }

    worker
        .join()
        .map_err(|_| anyhow!("corpus tokenizer worker panicked"))?;

    Ok(total_blocks)
}

fn load_corpus(
    tokenizer: &Tokenizer,
    sources: &[PathBuf],
    text_column: &str,
    block_size: usize,
    max_rows_per_file: usize,
    shuffle_seed: u64,
    mode: CorpusMode,
) -> Result<CorpusSupply> {
    let expected_total_entries = estimate_corpus_total_entries(sources, max_rows_per_file);
    let scanned_counter = Arc::new(AtomicU64::new(0));
    let added_counter = Arc::new(AtomicU64::new(0));
    let skipped_counter = Arc::new(AtomicU64::new(0));
    let progress_done = Arc::new(AtomicBool::new(false));
    let (docs_tx, docs_rx) = mpsc::channel::<CorpusMessage>();

    let reporter_scanned = Arc::clone(&scanned_counter);
    let reporter_added = Arc::clone(&added_counter);
    let reporter_skipped = Arc::clone(&skipped_counter);
    let reporter_done = Arc::clone(&progress_done);
    let corpus_bar = ProgressBar::new(expected_total_entries.max(1));
    corpus_bar.set_style(progress_style(
        "corpus   [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}",
    )?);
    let reporter_bar = corpus_bar.clone();
    let reporter = thread::spawn(move || {
        while !reporter_done.load(Ordering::Relaxed) {
            let scanned = reporter_scanned.load(Ordering::Relaxed);
            let added = reporter_added.load(Ordering::Relaxed);
            let skipped = reporter_skipped.load(Ordering::Relaxed);
            reporter_bar.set_position(scanned.min(expected_total_entries.max(1)));
            reporter_bar.set_message(format!("added={added} skipped={skipped}"));
            thread::sleep(Duration::from_secs(2));
        }
    });

    let collector = thread::spawn(move || {
        let mut documents = Vec::new();
        while let Ok(message) = docs_rx.recv() {
            match message {
                CorpusMessage::Documents(mut docs) => documents.append(&mut docs),
            }
        }
        documents
    });

    let worker_count = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, DEFAULT_CORPUS_FILE_THREADS);
    let chunk_size = sources.len().div_ceil(worker_count).max(1);

    let worker_results = thread::scope(|scope| {
        let mut handles = Vec::new();
        for chunk in sources.chunks(chunk_size) {
            let local_sources = chunk.to_vec();
            let local_tokenizer = tokenizer.clone();
            let local_text_column = text_column.to_string();
            let local_scanned = Arc::clone(&scanned_counter);
            let local_added = Arc::clone(&added_counter);
            let local_skipped = Arc::clone(&skipped_counter);
            let local_docs_tx = docs_tx.clone();
            handles.push(scope.spawn(move || -> Result<usize> {
                let mut local_total_blocks = 0usize;
                for source in &local_sources {
                    let ext = source
                        .extension()
                        .and_then(|x| x.to_str())
                        .unwrap_or_default();
                    let blocks = if ext.eq_ignore_ascii_case("parquet") {
                        load_corpus_from_parquet(
                            &local_tokenizer,
                            source,
                            &local_text_column,
                            block_size,
                            mode,
                            &local_docs_tx,
                            max_rows_per_file,
                            &local_scanned,
                            &local_added,
                            &local_skipped,
                        )?
                    } else {
                        load_corpus_from_text_file(
                            &local_tokenizer,
                            source,
                            block_size,
                            mode,
                            &local_docs_tx,
                            &local_scanned,
                            &local_added,
                            &local_skipped,
                        )?
                    };
                    local_total_blocks += blocks;
                }
                Ok(local_total_blocks)
            }));
        }

        let mut results = Vec::new();
        for handle in handles {
            results.push(
                handle
                    .join()
                    .map_err(|_| anyhow!("corpus worker thread panicked"))?,
            );
        }
        Ok::<Vec<Result<usize>>, anyhow::Error>(results)
    })?;

    let mut total_blocks = 0usize;
    for result in worker_results {
        let local_total_blocks = result?;
        total_blocks += local_total_blocks;
    }
    drop(docs_tx);
    let documents = collector
        .join()
        .map_err(|_| anyhow!("corpus collector thread panicked"))?;

    progress_done.store(true, Ordering::Relaxed);
    let _ = reporter.join();

    if total_blocks == 0 {
        bail!("no corpus blocks were produced");
    }
    let scanned = scanned_counter.load(Ordering::Relaxed);
    let added = added_counter.load(Ordering::Relaxed);
    let skipped = skipped_counter.load(Ordering::Relaxed);
    corpus_bar.set_position(scanned.min(expected_total_entries.max(1)));
    corpus_bar.finish_with_message(format!("added={added} skipped={skipped}"));
    println!("loaded_corpus_docs={}", documents.len());
    println!("loaded_corpus_added_entries={added}");
    println!("loaded_corpus_blocks={total_blocks}");
    Ok(CorpusSupply::new(documents, block_size, shuffle_seed))
}

impl CorpusSupply {
    fn new(documents: Vec<CorpusDocument>, block_size: usize, seed: u64) -> Self {
        let mut supply = Self {
            order: (0..documents.len()).collect(),
            total_blocks: documents.iter().map(|doc| doc.block_count).sum(),
            documents,
            doc_cursor: 0,
            block_cursor: 0,
            rng: StdRng::seed_from_u64(seed),
            block_size,
            epoch: 0,
            epoch_blocks_used: 0,
        };
        supply.order.shuffle(&mut supply.rng);
        supply
    }

    fn reshuffle(&mut self) {
        self.order.shuffle(&mut self.rng);
        self.doc_cursor = 0;
        self.block_cursor = 0;
        self.epoch += 1;
        self.epoch_blocks_used = 0;
    }

    fn next_block(&mut self) -> Vec<u32> {
        loop {
            if self.doc_cursor >= self.order.len() {
                self.reshuffle();
            }
            let doc_idx = self.order[self.doc_cursor];
            let doc = &self.documents[doc_idx];
            if self.block_cursor >= doc.block_count {
                self.doc_cursor += 1;
                self.block_cursor = 0;
                continue;
            }
            let start = self.block_cursor * self.block_size;
            let end = start + self.block_size;
            self.block_cursor += 1;
            self.epoch_blocks_used += 1;
            return doc.tokens[start..end].to_vec();
        }
    }

    fn total_blocks(&self) -> usize {
        self.total_blocks
    }

    fn epoch_blocks_used(&self) -> usize {
        self.epoch_blocks_used
    }

    fn epoch(&self) -> usize {
        self.epoch
    }
}

fn build_seed_block(tokenizer: &Tokenizer, seed_text: &str, block_size: usize) -> Result<Vec<u32>> {
    let mut ids = encode_text(tokenizer, seed_text)?;
    if ids.len() < block_size {
        let needed = block_size - ids.len();
        let filler = encode_text(
            tokenizer,
            " Continue the text naturally and predict the next content.",
        )?;
        ids.extend(filler.into_iter().cycle().take(needed));
    }
    ids.truncate(block_size);
    Ok(ids)
}

#[derive(Debug)]
struct OutputRow {
    total_hashes_len: u64,
    prompt: String,
    output_text: String,
}

#[derive(Clone)]
struct TraceRenderRow {
    row_seq: usize,
    input_tokens: u64,
    output_tokens: u64,
    total_hashes_len: u64,
    block_indices: Vec<usize>,
}

enum ReplayJob {
    Render {
        block_size: usize,
        rows: Vec<TraceRenderRow>,
    },
    Shutdown,
}

struct ReplayResult {
    rows: Vec<OutputRowWithSeq>,
}

struct OutputRowWithSeq {
    row_seq: usize,
    row: OutputRow,
}

enum WriteJob {
    Batch(RecordBatch),
    Shutdown,
}

fn build_output_schema(input_schema: &Schema) -> Arc<Schema> {
    let mut fields: Vec<Field> = input_schema
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect();
    let existing: std::collections::HashSet<String> =
        fields.iter().map(|field| field.name().clone()).collect();
    if !existing.contains("total_hashes_len") {
        fields.push(Field::new("total_hashes_len", DataType::UInt64, false));
    }
    if !existing.contains("prompt") {
        fields.push(Field::new("prompt", DataType::Utf8, false));
    }
    if !existing.contains("output_text") {
        fields.push(Field::new("output_text", DataType::Utf8, false));
    }
    Arc::new(Schema::new(fields))
}

fn build_output_batch(
    input_batch: &RecordBatch,
    selected_rows: usize,
    rendered_rows: Vec<OutputRowWithSeq>,
    output_schema: Arc<Schema>,
) -> Result<RecordBatch> {
    let mut rendered_rows = rendered_rows;
    rendered_rows.sort_by_key(|row| row.row_seq);

    let mut columns: Vec<ArrayRef> = input_batch
        .columns()
        .iter()
        .map(|column| column.slice(0, selected_rows))
        .collect();
    let has_total_hashes_len = input_batch.column_by_name("total_hashes_len").is_some();
    if !has_total_hashes_len {
        let total_hashes_len = UInt64Array::from(
            rendered_rows
                .iter()
                .map(|row| row.row.total_hashes_len)
                .collect::<Vec<_>>(),
        );
        columns.push(Arc::new(total_hashes_len));
    }
    let prompts = StringArray::from(
        rendered_rows
            .iter()
            .map(|row| row.row.prompt.as_str())
            .collect::<Vec<_>>(),
    );
    let outputs = StringArray::from(
        rendered_rows
            .iter()
            .map(|row| row.row.output_text.as_str())
            .collect::<Vec<_>>(),
    );
    columns.push(Arc::new(prompts));
    columns.push(Arc::new(outputs));

    RecordBatch::try_new(output_schema, columns).context("failed to build output batch")
}

fn spawn_replay_workers(
    tokenizer: &Tokenizer,
    assigned_blocks: Arc<RwLock<Vec<Arc<[u32]>>>>,
    worker_count: usize,
) -> (
    Vec<mpsc::Sender<ReplayJob>>,
    mpsc::Receiver<Result<ReplayResult>>,
) {
    let (result_tx, result_rx) = mpsc::channel::<Result<ReplayResult>>();
    let mut job_txs = Vec::with_capacity(worker_count);

    for _ in 0..worker_count {
        let (job_tx, job_rx) = mpsc::channel::<ReplayJob>();
        let worker_tokenizer = tokenizer.clone();
        let worker_blocks = Arc::clone(&assigned_blocks);
        let worker_result_tx = result_tx.clone();
        thread::spawn(move || {
            while let Ok(job) = job_rx.recv() {
                match job {
                    ReplayJob::Render { block_size, rows } => {
                        let blocks_guard = match worker_blocks.read() {
                            Ok(guard) => guard,
                            Err(_) => {
                                let _ = worker_result_tx
                                    .send(Err(anyhow!("assigned block store poisoned")));
                                return;
                            }
                        };

                        let mut rendered = Vec::with_capacity(rows.len());
                        for row in rows {
                            let mut reconstructed =
                                Vec::with_capacity(row.block_indices.len() * block_size);
                            for &block_idx in &row.block_indices {
                                let Some(block) = blocks_guard.get(block_idx) else {
                                    let _ = worker_result_tx.send(Err(anyhow!(
                                        "missing synthetic block at index {block_idx}"
                                    )));
                                    return;
                                };
                                reconstructed.extend_from_slice(block);
                            }

                            let nominal_input_len = row.input_tokens as usize;
                            let nominal_output_len = row.output_tokens as usize;
                            let adjusted_input_len = choose_adjusted_split(
                                &worker_tokenizer,
                                &reconstructed,
                                nominal_input_len.min(reconstructed.len()),
                            );
                            let output_end =
                                (nominal_input_len + nominal_output_len).min(reconstructed.len());
                            let prompt_end = adjusted_input_len.min(output_end);
                            let prompt_ids = &reconstructed[..prompt_end];
                            let completion_ids = if prompt_end < output_end {
                                &reconstructed[prompt_end..output_end]
                            } else {
                                &[]
                            };

                            let prompt = match decode_tokens(&worker_tokenizer, prompt_ids) {
                                Ok(value) => value,
                                Err(err) => {
                                    let _ = worker_result_tx.send(Err(err));
                                    return;
                                }
                            };
                            let output_text = match decode_tokens(&worker_tokenizer, completion_ids)
                            {
                                Ok(value) => value,
                                Err(err) => {
                                    let _ = worker_result_tx.send(Err(err));
                                    return;
                                }
                            };

                            rendered.push(OutputRowWithSeq {
                                row_seq: row.row_seq,
                                row: OutputRow {
                                    total_hashes_len: row.total_hashes_len,
                                    prompt,
                                    output_text,
                                },
                            });
                        }

                        if worker_result_tx
                            .send(Ok(ReplayResult { rows: rendered }))
                            .is_err()
                        {
                            return;
                        }
                    }
                    ReplayJob::Shutdown => return,
                }
            }
        });
        job_txs.push(job_tx);
    }

    (job_txs, result_rx)
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let args = parse_args()?;
    let cache_dir = PathBuf::from(&args.cache_dir);
    let mode = if args.corpus_sources.is_empty() {
        CorpusMode::FineWeb
    } else {
        CorpusMode::Custom
    };
    let corpus_urls = selected_corpus_sources(&args);
    let progress = MultiProgress::new();
    let download_bar = progress.add(ProgressBar::new(corpus_urls.len() as u64));
    download_bar.set_style(progress_style(
        "download [{elapsed_precise}] [{bar:40.green/white}] {pos}/{len} {msg}",
    )?);
    download_bar.set_message("starting");

    let trace_path = resolve_source(&args.trace_parquet, &cache_dir).await?;
    let tokenizer_path = resolve_source(&args.tokenizer, &cache_dir).await?;
    let corpus_paths = stream::iter(corpus_urls.into_iter())
        .map(|source| {
            let cache_dir = cache_dir.clone();
            let download_bar = download_bar.clone();
            async move {
                let path = resolve_source(&source, &cache_dir).await?;
                download_bar.inc(1);
                download_bar.set_message(
                    path.file_name()
                        .and_then(|x| x.to_str())
                        .unwrap_or("done")
                        .to_string(),
                );
                Ok::<PathBuf, anyhow::Error>(path)
            }
        })
        .buffer_unordered(args.num_files.max(1))
        .try_collect::<Vec<_>>()
        .await?;
    download_bar.finish_with_message(format!("downloaded {}", corpus_paths.len()));

    let block_size = infer_trace_block_size(&trace_path)?;
    println!("inferred_block_size={block_size}");

    let tokenizer = Tokenizer::from_file(&tokenizer_path, args.tokenizer_backend).map_err(|e| {
        anyhow!(
            "failed to load tokenizer from {}: {e}",
            tokenizer_path.display()
        )
    })?;

    let mut corpus_supply = load_corpus(
        &tokenizer,
        &corpus_paths,
        &args.corpus_text_column,
        block_size,
        args.max_rows_per_file,
        args.corpus_shuffle_seed,
        mode,
    )?;

    let seed_block = if let Some(seed_text) = &args.seed_first_block_text {
        Some(build_seed_block(&tokenizer, seed_text, block_size)?)
    } else {
        None
    };

    let trace_file = File::open(&trace_path)
        .with_context(|| format!("failed to open {}", trace_path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(trace_file)
        .with_context(|| format!("failed to create trace reader for {}", trace_path.display()))?;
    let output_schema = build_output_schema(builder.schema().as_ref());
    let outfile = File::create(&args.output_parquet)
        .with_context(|| format!("failed to create {}", args.output_parquet))?;
    let buffered_outfile = BufWriter::new(outfile);
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let writer = ArrowWriter::try_new(buffered_outfile, Arc::clone(&output_schema), Some(props))
        .context("failed to create parquet writer")?;
    let (write_tx, write_rx) = mpsc::sync_channel::<WriteJob>(DEFAULT_OUTPUT_WRITE_QUEUE_CAPACITY);
    let writer_handle = thread::spawn(move || -> Result<()> {
        let mut writer = writer;
        while let Ok(job) = write_rx.recv() {
            match job {
                WriteJob::Batch(batch) => {
                    writer
                        .write(&batch)
                        .context("failed to write output batch")?;
                }
                WriteJob::Shutdown => {
                    writer.close().context("failed to close output parquet")?;
                    return Ok(());
                }
            }
        }
        writer.close().context("failed to close output parquet")?;
        Ok(())
    });

    let reader = builder
        .with_batch_size(DEFAULT_TRACE_BATCH_SIZE)
        .build()
        .context("failed to build trace reader")?;
    let trace_total_rows = estimate_trace_total_rows(&trace_path, args.max_rows);
    let trace_bar = progress.add(ProgressBar::new(trace_total_rows.max(1)));
    trace_bar.set_style(progress_style(
        "trace    [{elapsed_precise}] [{bar:40.magenta/blue}] {pos}/{len} {msg}",
    )?);
    trace_bar.set_message("replaying");
    let corpus_supply_bar = progress.add(ProgressBar::new(corpus_supply.total_blocks() as u64));
    corpus_supply_bar.set_style(progress_style(
        "supply   [{elapsed_precise}] [{bar:40.yellow/blue}] {pos}/{len} {msg}",
    )?);
    corpus_supply_bar.set_message(format!("epoch={}", corpus_supply.epoch()));

    let mut uid_to_block_idx: HashMap<u64, usize> = HashMap::new();
    let mut first_uid_seed_block: HashMap<u64, usize> = HashMap::new();
    let assigned_blocks: Arc<RwLock<Vec<Arc<[u32]>>>> = Arc::new(RwLock::new(Vec::new()));
    let replay_worker_count = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .max(1);
    let (replay_job_txs, replay_result_rx) = spawn_replay_workers(
        &tokenizer,
        Arc::clone(&assigned_blocks),
        replay_worker_count,
    );
    let mut processed_rows = 0usize;
    for batch in reader {
        let batch = batch.context("failed to read trace batch")?;
        let input_tokens = batch
            .column_by_name("input_tokens")
            .ok_or_else(|| anyhow!("trace parquet missing input_tokens"))?
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| anyhow!("input_tokens is not UInt64"))?;
        let output_tokens = batch
            .column_by_name("output_tokens")
            .ok_or_else(|| anyhow!("trace parquet missing output_tokens"))?
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| anyhow!("output_tokens is not UInt64"))?;
        let hashes = batch
            .column_by_name("total_hashes")
            .ok_or_else(|| anyhow!("trace parquet missing total_hashes"))?
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| anyhow!("total_hashes is not List"))?;
        let mut render_rows = Vec::with_capacity(batch.num_rows());
        {
            let mut assigned_blocks_guard = assigned_blocks
                .write()
                .map_err(|_| anyhow!("assigned block store poisoned"))?;

            for row_idx in 0..batch.num_rows() {
                if let Some(limit) = args.max_rows
                    && processed_rows + render_rows.len() >= limit
                {
                    break;
                }

                let row_hashes = hashes.value(row_idx);
                let row_uids = row_hashes
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| anyhow!("total_hashes values are not UInt64"))?;

                let mut block_indices = Vec::with_capacity(row_uids.len());
                for block_pos in 0..row_uids.len() {
                    let uid = row_uids.value(block_pos);
                    let block_idx = if block_pos == 0 {
                        if let Some(existing) = first_uid_seed_block.get(&uid) {
                            *existing
                        } else if let Some(seed) = &seed_block {
                            let synthetic_idx = assigned_blocks_guard.len();
                            assigned_blocks_guard.push(Arc::<[u32]>::from(seed.clone()));
                            first_uid_seed_block.insert(uid, synthetic_idx);
                            uid_to_block_idx.insert(uid, synthetic_idx);
                            synthetic_idx
                        } else if let Some(existing) = uid_to_block_idx.get(&uid) {
                            *existing
                        } else {
                            let synthetic_idx = assigned_blocks_guard.len();
                            let block = corpus_supply.next_block();
                            assigned_blocks_guard.push(Arc::<[u32]>::from(block));
                            uid_to_block_idx.insert(uid, synthetic_idx);
                            synthetic_idx
                        }
                    } else if let Some(existing) = uid_to_block_idx.get(&uid) {
                        *existing
                    } else {
                        let synthetic_idx = assigned_blocks_guard.len();
                        let block = corpus_supply.next_block();
                        assigned_blocks_guard.push(Arc::<[u32]>::from(block));
                        uid_to_block_idx.insert(uid, synthetic_idx);
                        synthetic_idx
                    };
                    block_indices.push(block_idx);
                }

                render_rows.push(TraceRenderRow {
                    row_seq: processed_rows + render_rows.len(),
                    input_tokens: input_tokens.value(row_idx),
                    output_tokens: output_tokens.value(row_idx),
                    total_hashes_len: row_uids.len() as u64,
                    block_indices,
                });
            }
        }
        corpus_supply_bar.set_position(corpus_supply.epoch_blocks_used() as u64);
        corpus_supply_bar.set_message(format!("epoch={}", corpus_supply.epoch()));

        if render_rows.is_empty() {
            if let Some(limit) = args.max_rows
                && processed_rows >= limit
            {
                break;
            }
            continue;
        }

        let chunk_count = render_rows.len().min(replay_job_txs.len()).max(1);
        let chunk_size = render_rows.len().div_ceil(chunk_count).max(1);
        let mut jobs_sent = 0usize;
        for (worker_idx, chunk) in render_rows.chunks(chunk_size).enumerate() {
            replay_job_txs[worker_idx]
                .send(ReplayJob::Render {
                    block_size,
                    rows: chunk.to_vec(),
                })
                .map_err(|_| anyhow!("replay worker disconnected"))?;
            jobs_sent += 1;
        }

        let mut batch_results = Vec::with_capacity(render_rows.len());
        for _ in 0..jobs_sent {
            let result = replay_result_rx
                .recv()
                .map_err(|_| anyhow!("replay worker disconnected"))??;
            batch_results.extend(result.rows);
        }
        let output_batch = build_output_batch(
            &batch,
            render_rows.len(),
            batch_results,
            Arc::clone(&output_schema),
        )?;
        processed_rows += render_rows.len();
        write_tx
            .send(WriteJob::Batch(output_batch))
            .map_err(|_| anyhow!("output writer disconnected"))?;
        trace_bar.set_position((processed_rows as u64).min(trace_total_rows.max(1)));

        if let Some(limit) = args.max_rows
            && processed_rows >= limit
        {
            break;
        }
    }

    for job_tx in replay_job_txs {
        let _ = job_tx.send(ReplayJob::Shutdown);
    }

    write_tx
        .send(WriteJob::Shutdown)
        .map_err(|_| anyhow!("output writer disconnected"))?;
    writer_handle
        .join()
        .map_err(|_| anyhow!("output writer thread panicked"))??;

    corpus_supply_bar.finish_with_message(format!(
        "epoch={} used={}/{}",
        corpus_supply.epoch(),
        corpus_supply.epoch_blocks_used(),
        corpus_supply.total_blocks()
    ));
    trace_bar.finish_with_message(format!("processed {}", processed_rows));
    println!("done: processed_rows={processed_rows}");
    println!("wrote {}", args.output_parquet);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_documents_keep_every_complete_block() {
        let ids: Vec<u32> = (0..6501).collect();
        let docs = corpus_documents(ids.clone(), 64, CorpusMode::Custom).unwrap();
        assert_eq!(docs.len(), 3);
        assert!(
            docs.iter()
                .all(|doc| doc.tokens.len() <= 3000 && doc.tokens.len() == doc.block_count * 64)
        );
        let actual: Vec<_> = docs.into_iter().flat_map(|doc| doc.tokens).collect();
        assert_eq!(actual, ids[..6464]);
        assert_eq!(
            corpus_documents(vec![0; 64], 64, CorpusMode::Custom)
                .unwrap()
                .len(),
            1
        );
        assert!(
            corpus_documents(vec![0; 63], 64, CorpusMode::Custom)
                .unwrap()
                .is_empty()
        );
    }
}
