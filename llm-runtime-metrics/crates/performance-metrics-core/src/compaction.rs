use crate::__private::S3SinkConfig;
use crate::request_trace::{validate_s3_proxy_url, with_s3_credentials};
use anyhow::Context;
use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use futures::TryStreamExt;
use object_store::aws::AmazonS3Builder;
use object_store::path::Path as ObjectPath;
use object_store::{ObjectStore, ObjectStoreExt};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::file::statistics::Statistics;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::runtime::Builder as RuntimeBuilder;
use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tokio::time::Duration;

const DEFAULT_MAX_CONCURRENT_DOWNLOADS: usize = 96;
const PREPARED_QUEUE_FACTOR: usize = 2;
const COMPACTION_MAX_ROW_GROUP_SIZE: usize = 512 * 1024;
const COMPACTION_WRITE_BATCH_SIZE: usize = 32 * 1024;
const COMPACTION_OUTPUT_BUFFER_SIZE: usize = 64 * 1024 * 1024;
const SLOW_APPEND_WARN_SECS: f32 = 5.0;

#[derive(Debug, Clone, Default)]
pub struct AnonymizeConfig {
    pub columns_to_pseudonymize: Vec<String>,
    pub columns_to_drop: Vec<String>,
}

/// Default columns pseudonymized in anonymized outputs. Listed columns a file
/// lacks (e.g. older schemas) are skipped.
const DEFAULT_ANONYMIZE_PSEUDONYMIZE_COLUMNS: [&str; 6] = [
    "provided_session_id",
    "request_org_id",
    "org_id",
    "total_hashes",
    "prefill_worker_id",
    "decode_worker_id",
];
/// Default columns dropped from anonymized outputs. A column in both lists is
/// pseudonymized, so b10_request_id must stay off the pseudonymize list.
const DEFAULT_ANONYMIZE_DROP_COLUMNS: [&str; 3] = ["request_org_id", "org_id", "b10_request_id"];

impl AnonymizeConfig {
    /// The default lists for published (anonymized) outputs.
    pub fn published() -> Self {
        Self {
            columns_to_pseudonymize: DEFAULT_ANONYMIZE_PSEUDONYMIZE_COLUMNS
                .map(String::from)
                .to_vec(),
            columns_to_drop: DEFAULT_ANONYMIZE_DROP_COLUMNS.map(String::from).to_vec(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct S3ParquetCompactConfig {
    pub s3: S3SinkConfig,
    pub output_path: PathBuf,
    pub max_concurrent_downloads: usize,
    pub anonymize: Option<AnonymizeConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParquetCompactResult {
    pub source_objects: usize,
    pub rows_written: usize,
    pub output_path: PathBuf,
    pub min_timestamp: u64,
}

pub fn compact_s3_parquet_to_local(
    config: S3ParquetCompactConfig,
) -> anyhow::Result<ParquetCompactResult> {
    let rt = RuntimeBuilder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to build compaction runtime")?;
    rt.block_on(compact_s3_parquet_to_local_async(config))
}

pub async fn compact_s3_parquet_to_local_async(
    config: S3ParquetCompactConfig,
) -> anyhow::Result<ParquetCompactResult> {
    tokio::select! {
        result = compact_s3_parquet_to_local_inner_async(config) => result,
        _ = tokio::signal::ctrl_c() => {
            anyhow::bail!("interrupted by Ctrl+C");
        }
    }
}

async fn compact_s3_parquet_to_local_inner_async(
    config: S3ParquetCompactConfig,
) -> anyhow::Result<ParquetCompactResult> {
    let start = std::time::Instant::now();
    let store = build_s3_store(&config.s3)?;
    let max_concurrent_downloads = match config.max_concurrent_downloads {
        0 => DEFAULT_MAX_CONCURRENT_DOWNLOADS,
        value => value,
    };

    if let Some(ref anon_config) = config.anonymize {
        let temp_path = config.output_path.with_extension("tmp.parquet");
        let result = compact_store_parquet_prefix_to_local(
            store,
            config.s3.prefix.as_str(),
            temp_path.as_path(),
            max_concurrent_downloads,
        )
        .await?;

        let columns_to_pseudonymize_refs: Vec<&str> = anon_config
            .columns_to_pseudonymize
            .iter()
            .map(|s| s.as_str())
            .collect();
        let columns_to_drop_refs: Vec<&str> = anon_config
            .columns_to_drop
            .iter()
            .map(|s| s.as_str())
            .collect();

        let min_ts = crate::anonymize_parquet_file(
            &temp_path,
            &config.output_path,
            &columns_to_pseudonymize_refs,
            &columns_to_drop_refs,
            result.min_timestamp,
        )?;

        std::fs::remove_file(&temp_path)?;

        let duration_ms = start.elapsed().as_millis();
        println!("Duration: {}ms", duration_ms);
        Ok(ParquetCompactResult {
            source_objects: result.source_objects,
            rows_written: result.rows_written,
            output_path: config.output_path,
            min_timestamp: min_ts,
        })
    } else {
        let result = compact_store_parquet_prefix_to_local(
            store,
            config.s3.prefix.as_str(),
            config.output_path.as_path(),
            max_concurrent_downloads,
        )
        .await?;
        let duration_ms = start.elapsed().as_millis();
        println!("Duration: {}ms", duration_ms);
        Ok(result)
    }
}

async fn compact_store_parquet_prefix_to_local(
    store: Arc<dyn ObjectStore>,
    prefix: &str,
    output_path: &Path,
    max_concurrent_downloads: usize,
) -> anyhow::Result<ParquetCompactResult> {
    let prefix = prefix.trim_matches('/');
    let prefix_path = if prefix.is_empty() {
        None
    } else {
        Some(ObjectPath::from(prefix))
    };

    let mut objects = store
        .list(prefix_path.as_ref())
        .try_collect::<Vec<_>>()
        .await
        .context("failed to list parquet objects")?;
    objects.sort_by(|a, b| a.location.cmp(&b.location));
    objects.retain(|meta| meta.location.as_ref().ends_with(".parquet"));

    if objects.is_empty() {
        anyhow::bail!("no parquet objects found under prefix '{prefix}'");
    }

    if let Some(parent) = output_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create output directory {}", parent.display()))?;
    }

    let locations: Arc<Vec<ObjectPath>> =
        Arc::new(objects.iter().map(|obj| obj.location.clone()).collect());
    let worker_count = max_concurrent_downloads.min(locations.len());
    let queue_capacity = (worker_count.max(1) * PREPARED_QUEUE_FACTOR).max(1);
    let (prepared_tx, mut prepared_rx) =
        mpsc::channel::<anyhow::Result<PreparedObject>>(queue_capacity);
    let next_index = Arc::new(AtomicUsize::new(0));
    let mut worker_set = JoinSet::new();

    for _ in 0..worker_count {
        let store = Arc::clone(&store);
        let locations = Arc::clone(&locations);
        let next_index = Arc::clone(&next_index);
        let prepared_tx = prepared_tx.clone();
        worker_set.spawn(async move {
            loop {
                let idx = next_index.fetch_add(1, Ordering::Relaxed);
                if idx >= locations.len() {
                    break;
                }
                let result =
                    fetch_and_prepare_object(Arc::clone(&store), locations[idx].clone()).await;
                if prepared_tx.send(result).await.is_err() {
                    break;
                }
            }
        });
    }
    drop(prepared_tx);

    let mut prepared_payloads: Vec<PreparedObject> = Vec::with_capacity(objects.len());
    let mut download_errors: Option<anyhow::Error> = None;

    while prepared_payloads.len() < objects.len() {
        let Some(next_prepared) = prepared_rx.recv().await else {
            worker_set.abort_all();
            anyhow::bail!("prepared queue terminated before all parquet objects were processed");
        };

        match next_prepared {
            Ok(fetched) => prepared_payloads.push(fetched),
            Err(err) => {
                worker_set.abort_all();
                download_errors = Some(err);
                break;
            }
        }
    }

    while let Some(result) = worker_set.join_next().await {
        if let Err(err) = result {
            return Err(anyhow::anyhow!("compaction worker task failed: {err}"));
        }
    }

    if let Some(err) = download_errors {
        return Err(err);
    }

    // Compute the union schema from all input files so that columns present
    // in some files but not others (e.g. `service_tier` added during a rolling
    // deployment) are preserved in the output regardless of file processing
    // order.
    let schemas: Vec<_> = prepared_payloads
        .iter()
        .map(|p| p.prepared.schema.clone())
        .collect();
    let union_schema = compute_union_schema(&schemas)?;

    let start_time = std::time::Instant::now();
    let mut processed_files = 0usize;

    let mut writer = None;
    let mut rows_written = 0usize;
    let mut min_timestamp: Option<u64> = None;

    for fetched in prepared_payloads {
        let file_min_ts = append_prepared_payload(
            fetched.location.as_ref(),
            fetched.prepared,
            &union_schema,
            output_path,
            &mut writer,
            &mut rows_written,
        )?;
        if let Some(ts) = file_min_ts {
            min_timestamp = Some(min_timestamp.map_or(ts, |min| min.min(ts)));
        }
        processed_files += 1;
        if processed_files.is_multiple_of(10) {
            let elapsed = start_time.elapsed().as_secs_f32();
            eprintln!(
                "Processed {}/{} files ({:.1}s)",
                processed_files,
                objects.len(),
                elapsed
            );
        }
    }

    println!(
        "All workers completed. Finalizing output file... ({:.1}s)",
        start_time.elapsed().as_secs_f32()
    );

    writer
        .context("writer should be initialized when objects are present")?
        .close()
        .with_context(|| format!("failed to finalize {}", output_path.display()))?;

    Ok(ParquetCompactResult {
        source_objects: objects.len(),
        rows_written,
        output_path: output_path.to_path_buf(),
        min_timestamp: min_timestamp.unwrap_or(0),
    })
}

struct PreparedObject {
    location: ObjectPath,
    prepared: PreparedPayload,
}

struct PreparedPayload {
    schema: arrow::datatypes::SchemaRef,
    batches: Vec<RecordBatch>,
    min_timestamp: Option<u64>,
}

async fn fetch_and_prepare_object(
    store: Arc<dyn ObjectStore>,
    location: ObjectPath,
) -> anyhow::Result<PreparedObject> {
    let fetch_start = std::time::Instant::now();
    let location_str = location.to_string();
    let location_str_clone = location_str.clone();

    // Spawn a warning task that prints if download takes >60s
    let warning_task = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(60)).await;
        eprintln!(
            "SLOW: Download of {} is taking longer than 60s",
            location_str_clone
        );
    });

    let payload = store
        .get(&location)
        .await
        .with_context(|| format!("failed to fetch {location}"))?
        .bytes()
        .await
        .with_context(|| format!("failed to read {location}"))?;

    warning_task.abort();
    let elapsed = fetch_start.elapsed().as_secs_f32();
    if elapsed >= SLOW_APPEND_WARN_SECS {
        eprintln!(
            "SLOW: download of {} took {:.1}s ({} bytes)",
            location,
            elapsed,
            payload.len()
        );
    }
    let prepared = prepare_parquet_payload(location.as_ref(), payload)?;
    Ok(PreparedObject { location, prepared })
}

fn prepare_parquet_payload(location: &str, payload: Bytes) -> anyhow::Result<PreparedPayload> {
    use arrow::array::UInt64Array;
    let decode_start = std::time::Instant::now();
    let mut min_timestamp = min_timestamp_from_row_group_stats(&payload);
    let needs_value_scan = min_timestamp.is_none();

    let builder = ParquetRecordBatchReaderBuilder::try_new(payload)
        .with_context(|| format!("failed to open parquet {location}"))?;
    let schema = builder.schema().clone();
    let record_reader = builder
        .build()
        .with_context(|| format!("failed to build batch reader for {location}"))?;

    let mut batches = Vec::new();
    for batch in record_reader {
        let batch = batch.with_context(|| format!("failed to read batch from {location}"))?;
        if needs_value_scan
            && let Ok(idx) = batch.schema().index_of("timestamp_start_unix_ms")
            && let Some(arr) = batch.column(idx).as_any().downcast_ref::<UInt64Array>()
        {
            for ts in arr.iter().flatten() {
                min_timestamp = Some(min_timestamp.map_or(ts, |min| min.min(ts)));
            }
        }
        batches.push(batch);
    }
    let elapsed = decode_start.elapsed().as_secs_f32();
    if elapsed >= SLOW_APPEND_WARN_SECS {
        eprintln!("SLOW: decode of {location} took {elapsed:.1}s");
    }
    Ok(PreparedPayload {
        schema,
        batches,
        min_timestamp,
    })
}

fn min_timestamp_from_row_group_stats(payload: &Bytes) -> Option<u64> {
    let reader = SerializedFileReader::new(payload.clone()).ok()?;
    let metadata = reader.metadata();
    let ts_column_idx = metadata
        .file_metadata()
        .schema_descr()
        .columns()
        .iter()
        .position(|column| {
            column
                .path()
                .parts()
                .last()
                .is_some_and(|name| name == "timestamp_start_unix_ms")
        })?;

    let mut min_timestamp: Option<u64> = None;
    for row_group in metadata.row_groups() {
        let column = row_group.column(ts_column_idx);
        let Some(stats) = column.statistics() else {
            continue;
        };
        let value = match stats {
            Statistics::Int64(stats) => {
                stats.min_opt().copied().and_then(|v| u64::try_from(v).ok())
            }
            _ => None,
        };
        if let Some(value) = value {
            min_timestamp = Some(min_timestamp.map_or(value, |min| min.min(value)));
        }
    }
    min_timestamp
}

/// Compute the union of multiple schemas by merging all fields. Fields that
/// exist in only some schemas are included in the union as nullable so that
/// batches missing those fields can be aligned with null columns. Fields
/// present in all schemas keep their original nullability. This ensures the
/// compaction output preserves all columns from all input files regardless
/// of file processing order.
fn compute_union_schema(
    schemas: &[arrow::datatypes::SchemaRef],
) -> anyhow::Result<arrow::datatypes::SchemaRef> {
    use arrow::datatypes::{DataType, Field, Schema};

    if schemas.is_empty() {
        anyhow::bail!("no schemas to compute union from");
    }
    if schemas.len() == 1 {
        return Ok(schemas[0].clone());
    }

    // Collect fields in order of first appearance across all schemas.
    let mut fields: Vec<(String, DataType, bool)> = Vec::new();

    for schema in schemas {
        for field in schema.fields() {
            match fields.iter().position(|(name, _, _)| name == field.name()) {
                Some(idx) => {
                    // Field already seen — verify type compatibility and
                    // widen nullability.
                    let (_, existing_type, nullable) = &mut fields[idx];
                    if *existing_type != *field.data_type() {
                        anyhow::bail!(
                            "incompatible types for field '{}': {:?} vs {:?}",
                            field.name(),
                            *existing_type,
                            field.data_type()
                        );
                    }
                    *nullable = *nullable || field.is_nullable();
                }
                None => {
                    fields.push((
                        field.name().clone(),
                        field.data_type().clone(),
                        field.is_nullable(),
                    ));
                }
            }
        }
    }

    let merged_fields: Vec<Field> = fields
        .into_iter()
        .map(|(name, dtype, nullable)| Field::new(name, dtype, nullable))
        .collect();
    Ok(Arc::new(Schema::new(merged_fields)))
}

/// Project a `RecordBatch` to match `expected_schema`. For fields present in
/// the expected schema but missing from the batch, a null column is added
/// (the field must be nullable). Extra fields in the batch that are not in
/// the expected schema are dropped. This allows compaction to proceed when
/// input files have different schemas due to additive schema changes (e.g.,
/// a new nullable `service_tier` column added during a rolling deployment).
fn align_batch_to_schema(
    batch: RecordBatch,
    expected_schema: &arrow::datatypes::SchemaRef,
) -> anyhow::Result<RecordBatch> {
    use arrow::array::new_null_array;

    let batch_schema = batch.schema();
    if batch_schema.as_ref() == expected_schema.as_ref() {
        return Ok(batch);
    }

    let mut columns = Vec::with_capacity(expected_schema.fields().len());
    for field in expected_schema.fields() {
        match batch_schema.column_with_name(field.name()) {
            Some((idx, _)) => {
                columns.push(batch.column(idx).clone());
            }
            None => {
                if !field.is_nullable() {
                    anyhow::bail!(
                        "non-nullable field '{}' missing from batch with schema {:?}",
                        field.name(),
                        batch_schema
                    );
                }
                columns.push(new_null_array(field.data_type(), batch.num_rows()));
            }
        }
    }

    RecordBatch::try_new(expected_schema.clone(), columns)
        .map_err(|e| anyhow::anyhow!("failed to align batch to schema: {e}"))
}

fn append_prepared_payload(
    location: &str,
    prepared: PreparedPayload,
    union_schema: &arrow::datatypes::SchemaRef,
    output_path: &Path,
    writer: &mut Option<ArrowWriter<BufWriter<File>>>,
    rows_written: &mut usize,
) -> anyhow::Result<Option<u64>> {
    let append_start = std::time::Instant::now();
    let PreparedPayload {
        schema,
        batches,
        min_timestamp,
    } = prepared;

    if writer.is_none() {
        let file = File::create(output_path)
            .with_context(|| format!("failed to create {}", output_path.display()))?;
        let buffered_file = BufWriter::with_capacity(COMPACTION_OUTPUT_BUFFER_SIZE, file);
        let props = WriterProperties::builder()
            .set_compression(Compression::UNCOMPRESSED)
            .set_max_row_group_size(COMPACTION_MAX_ROW_GROUP_SIZE)
            .set_write_batch_size(COMPACTION_WRITE_BATCH_SIZE)
            .build();
        *writer = Some(
            ArrowWriter::try_new(buffered_file, union_schema.clone(), Some(props))
                .with_context(|| format!("failed to open {}", output_path.display()))?,
        );
    }

    if schema.as_ref() != union_schema.as_ref() {
        eprintln!(
            "WARN: schema mismatch for {} while compacting prefix '{}' — aligning batches to union schema",
            location,
            output_path.display()
        );
    }

    for batch in batches {
        let batch = align_batch_to_schema(batch, union_schema)?;
        *rows_written += batch.num_rows();
        writer
            .as_mut()
            .expect("writer should be initialized")
            .write(&batch)
            .with_context(|| format!("failed to write batch from {location}"))?;
    }
    let elapsed = append_start.elapsed().as_secs_f32();
    if elapsed >= SLOW_APPEND_WARN_SECS {
        eprintln!("SLOW: append of {location} took {elapsed:.1}s");
    }
    Ok(min_timestamp)
}

fn build_s3_store(config: &S3SinkConfig) -> anyhow::Result<Arc<dyn ObjectStore>> {
    let mut builder = AmazonS3Builder::new()
        .with_bucket_name(config.bucket.clone())
        .with_region(config.region.clone());
    if let Some(endpoint) = &config.endpoint {
        builder = builder.with_endpoint(endpoint.clone());
    }
    if let Some(proxy_url) = &config.proxy_url {
        validate_s3_proxy_url(proxy_url)?;
        builder = builder.with_proxy_url(proxy_url.clone());
    }
    builder = with_s3_credentials(
        builder,
        config.access_key_id.as_deref(),
        config.secret_access_key.as_deref(),
    );
    let store = builder
        .build()
        .map_err(|e| anyhow::anyhow!("failed to build s3 store: {e}"))?;
    Ok(Arc::new(store))
}

/// Anonymize a parquet file by offsetting timestamps and pseudonymizing sensitive columns.
///
/// # Arguments
/// * `input_path` - Path to input parquet file
/// * `output_path` - Path to output parquet file
/// * `columns_to_pseudonymize` - List of column names to pseudonymize (e.g., ["provided_session_id", "org_id", "total_hashes"])
/// * `columns_to_drop` - List of column names to drop entirely
/// * `min_timestamp` - Minimum timestamp used for offsetting. If 0, timestamps are not shifted.
///
/// # Behavior
/// - Offsets timestamps by min_timestamp when min_timestamp > 0
/// - Leaves timestamps unchanged when min_timestamp == 0
/// - Assigns unique IDs to specified columns using a HashMap
/// - Drops specified columns (unless they're also in columns_to_pseudonymize)
/// - total_hashes column is treated specially as List<FixedSizeBinary(32)>
/// - UInt64 columns are pseudonymized to dense UInt64 IDs; Utf8 columns to dense ID strings
pub fn anonymize_parquet_file(
    input_path: &Path,
    output_path: &Path,
    columns_to_pseudonymize: &[&str],
    columns_to_drop: &[&str],
    min_timestamp: u64,
) -> anyhow::Result<u64> {
    use arrow::array::{
        Array, BinaryArray, FixedSizeBinaryArray, ListArray, StringArray, UInt64Array,
    };
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::collections::{HashMap, HashSet};

    // HashMaps for assigning unique IDs - one per column
    let mut id_maps: HashMap<String, HashMap<Vec<u8>, u64>> = HashMap::new();
    for col in columns_to_pseudonymize {
        id_maps.insert(col.to_string(), HashMap::new());
    }

    // Convert to HashSets for faster lookup
    let columns_to_pseudonymize_set: HashSet<&str> =
        columns_to_pseudonymize.iter().copied().collect();
    let columns_to_drop_set: HashSet<&str> = columns_to_drop.iter().copied().collect();
    println!("Starting anonymization to {}", output_path.display());
    // Read input parquet
    let file = std::fs::File::open(input_path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

    // Use provided min_timestamp (0 means no offsetting)
    let min_ts = min_timestamp;

    // Process and write in streaming fashion
    let out_file = File::create(output_path)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_max_row_group_size(COMPACTION_MAX_ROW_GROUP_SIZE)
        .set_write_batch_size(COMPACTION_WRITE_BATCH_SIZE)
        .build();

    let mut writer: Option<ArrowWriter<BufWriter<File>>> = None;

    for batch in reader {
        let batch = batch?;
        let schema = batch.schema();
        let mut columns: Vec<std::sync::Arc<dyn Array>> = Vec::with_capacity(batch.num_columns());
        let mut fields: Vec<arrow::datatypes::Field> = Vec::new();

        for (idx, field) in schema.fields().iter().enumerate() {
            let column = batch.column(idx);
            let col_name = field.name();

            // Handle timestamp offset
            if col_name == "timestamp_start_unix_ms" && min_ts > 0 {
                let arr = column.as_any().downcast_ref::<UInt64Array>().unwrap();
                let offset_values: Vec<u64> = arr
                    .iter()
                    .map(|v| v.unwrap_or(0).saturating_sub(min_ts))
                    .collect();
                columns.push(std::sync::Arc::new(UInt64Array::from(offset_values)));
                fields.push((**field).clone());
            }
            // Handle columns to pseudonymize
            else if columns_to_pseudonymize_set.contains(col_name.as_str()) {
                if col_name == "total_hashes" {
                    // Handle list of binary hashes - assign unique IDs
                    let list_arr = column.as_any().downcast_ref::<ListArray>().unwrap();
                    let id_map = id_maps.get_mut("total_hashes").unwrap();
                    let mut new_lists: Vec<Option<Vec<u64>>> = Vec::new();

                    for i in 0..list_arr.len() {
                        if list_arr.is_null(i) {
                            new_lists.push(None);
                        } else {
                            let values = list_arr.value(i);
                            let mut ids = Vec::new();

                            // Try FixedSizeBinaryArray first (32-byte fixed), then BinaryArray
                            if let Some(fixed_arr) =
                                values.as_any().downcast_ref::<FixedSizeBinaryArray>()
                            {
                                for j in 0..fixed_arr.len() {
                                    if !fixed_arr.is_null(j) {
                                        let hash_bytes = fixed_arr.value(j).to_vec();
                                        let len = id_map.len() as u64;
                                        let id =
                                            *id_map.entry(hash_bytes).or_insert_with(|| len + 1);
                                        ids.push(id);
                                    }
                                }
                            } else if let Some(binary_arr) =
                                values.as_any().downcast_ref::<BinaryArray>()
                            {
                                for j in 0..binary_arr.len() {
                                    if !binary_arr.is_null(j) {
                                        let hash_bytes = binary_arr.value(j).to_vec();
                                        let len = id_map.len() as u64;
                                        let id =
                                            *id_map.entry(hash_bytes).or_insert_with(|| len + 1);
                                        ids.push(id);
                                    }
                                }
                            }
                            new_lists.push(Some(ids));
                        }
                    }

                    // Build new List<UInt64> for the values
                    let values_builder = arrow::array::builder::UInt64Builder::new();
                    let mut list_builder = arrow::array::builder::ListBuilder::new(values_builder);

                    for list in new_lists {
                        if let Some(ids) = list {
                            for id in ids {
                                list_builder.values().append_value(id);
                            }
                            list_builder.append(true);
                        } else {
                            list_builder.append(false);
                        }
                    }

                    columns.push(std::sync::Arc::new(list_builder.finish()));
                    // Update field type to List<UInt64>
                    let new_field = arrow::datatypes::Field::new(
                        field.name().clone(),
                        arrow::datatypes::DataType::List(std::sync::Arc::new(
                            arrow::datatypes::Field::new(
                                "item",
                                arrow::datatypes::DataType::UInt64,
                                true,
                            ),
                        )),
                        field.is_nullable(),
                    );
                    fields.push(new_field);
                } else if let Some(arr) = column.as_any().downcast_ref::<UInt64Array>() {
                    // Integer IDs (e.g. worker ids) become dense per-file UInt64 IDs.
                    let id_map = id_maps.get_mut(col_name.as_str()).unwrap();
                    let id_values: Vec<Option<u64>> = arr
                        .iter()
                        .map(|value| {
                            value.map(|value| {
                                let len = id_map.len() as u64;
                                *id_map
                                    .entry(value.to_le_bytes().to_vec())
                                    .or_insert_with(|| len + 1)
                            })
                        })
                        .collect();
                    columns.push(std::sync::Arc::new(UInt64Array::from(id_values)));
                    fields.push((**field).clone());
                } else {
                    // Handle string columns - assign unique IDs
                    let arr = column
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| anyhow::anyhow!("{} is not StringArray", col_name))?;
                    let id_map = id_maps.get_mut(col_name.as_str()).unwrap();
                    let id_values: Vec<Option<String>> = (0..arr.len())
                        .map(|i| {
                            if arr.is_null(i) {
                                None
                            } else {
                                let value = arr.value(i);
                                let bytes = value.as_bytes().to_vec();
                                let len = id_map.len() as u64;
                                let id = *id_map.entry(bytes).or_insert_with(|| len + 1);
                                Some(id.to_string())
                            }
                        })
                        .collect();
                    columns.push(std::sync::Arc::new(StringArray::from(id_values)));
                    fields.push((**field).clone());
                }
            }
            // Handle columns to drop (only those not pseudonymized)
            else if columns_to_drop_set.contains(col_name.as_str()) {
                // Skip this column
                continue;
            }
            // Pass through other columns unchanged
            else {
                columns.push(column.clone());
                fields.push((**field).clone());
            }
        }

        // Create new schema with only the fields we kept
        let new_schema = arrow::datatypes::Schema::new(fields);
        let new_batch = RecordBatch::try_new(std::sync::Arc::new(new_schema), columns)?;

        if writer.is_none() {
            let buffered_file =
                BufWriter::with_capacity(COMPACTION_OUTPUT_BUFFER_SIZE, out_file.try_clone()?);
            writer = Some(ArrowWriter::try_new(
                buffered_file,
                new_batch.schema(),
                Some(props.clone()),
            )?);
        }

        writer
            .as_mut()
            .expect("writer should be initialized")
            .write(&new_batch)?;
    }

    if let Some(w) = writer {
        w.close()?;
    }

    println!("Anonymization complete.");
    Ok(min_ts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Int64Array, StringArray, UInt64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use bytes::Bytes;
    use object_store::memory::InMemory;
    use object_store::{PutOptions, PutPayload};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn make_test_parquet(rows: &[(&str, i64)]) -> Vec<u8> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8, false),
            Field::new("value", DataType::Int64, false),
        ]));
        let names = rows
            .iter()
            .map(|(name, _)| (*name).to_string())
            .collect::<Vec<_>>();
        let values = rows.iter().map(|(_, value)| *value).collect::<Vec<_>>();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(names)),
                Arc::new(Int64Array::from(values)),
            ],
        )
        .unwrap();

        let mut buf = Vec::new();
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let mut writer = ArrowWriter::try_new(&mut buf, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        buf
    }

    fn temp_output_path(name: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("pm-compact-{name}-{suffix}.parquet"))
    }

    #[tokio::test]
    async fn compact_store_parquet_prefix_to_local_merges_multiple_files() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let payload_a = make_test_parquet(&[("a", 1), ("b", 2)]);
        let payload_b = make_test_parquet(&[("c", 3)]);

        store
            .put_opts(
                &ObjectPath::from("prefix/part-0001.parquet"),
                PutPayload::from(Bytes::from(payload_a)),
                PutOptions::default(),
            )
            .await
            .unwrap();
        store
            .put_opts(
                &ObjectPath::from("prefix/part-0002.parquet"),
                PutPayload::from(Bytes::from(payload_b)),
                PutOptions::default(),
            )
            .await
            .unwrap();

        let output_path = temp_output_path("merge");
        let result =
            compact_store_parquet_prefix_to_local(store, "prefix", output_path.as_path(), 2)
                .await
                .unwrap();

        assert_eq!(result.source_objects, 2);
        assert_eq!(result.rows_written, 3);
        assert_eq!(result.output_path, output_path);

        let output_bytes = std::fs::read(output_path.as_path()).unwrap();
        let record_reader = ParquetRecordBatchReaderBuilder::try_new(Bytes::from(output_bytes))
            .unwrap()
            .build()
            .unwrap();
        let mut names = Vec::new();
        let mut values = Vec::new();
        for batch in record_reader {
            let batch = batch.unwrap();
            let name_col = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let value_col = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            for i in 0..batch.num_rows() {
                names.push(name_col.value(i).to_string());
                values.push(value_col.value(i));
            }
        }
        assert_eq!(names, vec!["a", "b", "c"]);
        assert_eq!(values, vec![1, 2, 3]);

        std::fs::remove_file(output_path).unwrap();
    }

    #[tokio::test]
    async fn compact_store_parquet_prefix_to_local_ignores_non_parquet_objects() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let payload = make_test_parquet(&[("a", 1)]);

        store
            .put_opts(
                &ObjectPath::from("prefix/part-0001.parquet"),
                PutPayload::from(Bytes::from(payload)),
                PutOptions::default(),
            )
            .await
            .unwrap();
        store
            .put_opts(
                &ObjectPath::from("prefix/notes.txt"),
                PutPayload::from(Bytes::from_static(b"ignored")),
                PutOptions::default(),
            )
            .await
            .unwrap();

        let output_path = temp_output_path("filter");
        let result =
            compact_store_parquet_prefix_to_local(store, "prefix", output_path.as_path(), 2)
                .await
                .unwrap();
        assert_eq!(result.source_objects, 1);
        assert_eq!(result.rows_written, 1);

        std::fs::remove_file(output_path).unwrap();
    }

    #[test]
    fn anonymize_parquet_file_offsets_timestamps_and_hashes_columns() {
        use arrow::array::{Int64Array, StringArray, UInt64Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        // Create test data with timestamps, session IDs, and total_hashes
        let schema = Arc::new(Schema::new(vec![
            Field::new("timestamp_start_unix_ms", DataType::UInt64, false),
            Field::new("provided_session_id", DataType::Utf8, true),
            Field::new("org_id", DataType::Utf8, true),
            Field::new("value", DataType::Int64, false),
        ]));

        // Create test data: timestamps [1000, 2000, 500, 3000], session IDs, org IDs
        let timestamps = vec![1000u64, 2000, 500, 3000];
        let sessions: Vec<Option<String>> = vec![
            Some("session-1".to_string()),
            Some("session-2".to_string()),
            Some("session-1".to_string()),
            None,
        ];
        let orgs: Vec<Option<String>> = vec![
            Some("org-1".to_string()),
            Some("org-2".to_string()),
            Some("org-1".to_string()),
            Some("org-3".to_string()),
        ];
        let values = vec![1i64, 2, 3, 4];

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(timestamps)),
                Arc::new(StringArray::from(sessions)),
                Arc::new(StringArray::from(orgs)),
                Arc::new(Int64Array::from(values)),
            ],
        )
        .unwrap();

        // Write test input file
        let input_path = temp_output_path("anonymize_input");
        let output_path = temp_output_path("anonymize_output");

        let mut buf = Vec::new();
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let mut writer = ArrowWriter::try_new(&mut buf, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        std::fs::write(&input_path, &buf).unwrap();

        // Run anonymization with min_timestamp=500 (the actual minimum in the data)
        anonymize_parquet_file(
            &input_path,
            &output_path,
            &["provided_session_id", "org_id"],
            &["org_id"], // org_id is in both, so it will be hashed not dropped
            500,         // min timestamp to offset by
        )
        .unwrap();

        // Verify output
        let output_bytes = std::fs::read(&output_path).unwrap();
        let record_reader = ParquetRecordBatchReaderBuilder::try_new(Bytes::from(output_bytes))
            .unwrap()
            .build()
            .unwrap();

        let mut timestamps = Vec::new();
        let mut sessions = Vec::new();
        let mut orgs = Vec::new();
        let mut values = Vec::new();

        for batch in record_reader {
            let batch = batch.unwrap();
            let schema = batch.schema();

            // Get column indices by name
            let ts_idx = schema.index_of("timestamp_start_unix_ms").unwrap();
            let session_idx = schema.index_of("provided_session_id").unwrap();
            let org_idx = schema.index_of("org_id").unwrap();
            let value_idx = schema.index_of("value").unwrap();

            let ts_col = batch
                .column(ts_idx)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            let session_col = batch
                .column(session_idx)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let org_col = batch
                .column(org_idx)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let value_col = batch
                .column(value_idx)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();

            for i in 0..batch.num_rows() {
                timestamps.push(ts_col.value(i));
                sessions.push(session_col.value(i).to_string());
                orgs.push(org_col.value(i).to_string());
                values.push(value_col.value(i));
            }
        }

        // Verify timestamps are offset (minimum should be 0)
        assert!(timestamps.contains(&0), "Minimum timestamp should be 0");
        assert!(timestamps.contains(&500), "Should have timestamp 500");
        assert!(timestamps.contains(&1500), "Should have timestamp 1500");
        assert!(timestamps.contains(&2500), "Should have timestamp 2500");

        // Verify session IDs are replaced with unique IDs
        // Note: sessions[3] is null, so it will be an empty string
        assert_eq!(sessions.len(), 4);
        for (i, session) in sessions.iter().enumerate() {
            if i == 3 {
                // Null value - should be empty string
                assert_eq!(session.len(), 0, "Null session should be empty string");
            } else {
                // Non-null values - should be unique IDs (numeric strings)
                assert_ne!(session, "session-1", "Session should be anonymized");
                assert_ne!(session, "session-2", "Session should be anonymized");
                assert!(
                    session.parse::<u64>().is_ok(),
                    "Session should be a numeric ID"
                );
            }
        }
        // Verify same session values get same IDs
        assert_eq!(
            sessions[0], sessions[2],
            "Same session values should get same ID"
        );

        // Verify org IDs are replaced with unique IDs (not dropped, since they're in both lists)
        assert_eq!(orgs.len(), 4);
        for org in &orgs {
            assert_ne!(org, "org-1", "Org should be anonymized");
            assert_ne!(org, "org-2", "Org should be anonymized");
            assert_ne!(org, "org-3", "Org should be anonymized");
            assert!(org.parse::<u64>().is_ok(), "Org should be a numeric ID");
        }
        // Verify same org values get same IDs
        assert_eq!(orgs[0], orgs[2], "Same org values should get same ID");

        // Verify values are unchanged
        assert_eq!(values, vec![1, 2, 3, 4]);

        // Cleanup
        std::fs::remove_file(input_path).unwrap();
        std::fs::remove_file(output_path).unwrap();
    }

    #[test]
    fn anonymize_parquet_file_zero_min_timestamp_does_not_shift_timestamps() {
        use arrow::array::{Int64Array, UInt64Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let schema = Arc::new(Schema::new(vec![
            Field::new("timestamp_start_unix_ms", DataType::UInt64, false),
            Field::new("value", DataType::Int64, false),
        ]));

        let input_timestamps = vec![1000u64, 2000u64, 3500u64];
        let values = vec![1i64, 2, 3];

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(input_timestamps.clone())),
                Arc::new(Int64Array::from(values)),
            ],
        )
        .unwrap();

        let input_path = temp_output_path("anonymize_zero_shift_input");
        let output_path = temp_output_path("anonymize_zero_shift_output");

        let mut buf = Vec::new();
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let mut writer = ArrowWriter::try_new(&mut buf, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        std::fs::write(&input_path, &buf).unwrap();

        let returned_min_ts =
            anonymize_parquet_file(&input_path, &output_path, &[], &[], 0).unwrap();
        assert_eq!(returned_min_ts, 0);

        let output_bytes = std::fs::read(&output_path).unwrap();
        let record_reader = ParquetRecordBatchReaderBuilder::try_new(Bytes::from(output_bytes))
            .unwrap()
            .build()
            .unwrap();

        let mut output_timestamps = Vec::new();
        for batch in record_reader {
            let batch = batch.unwrap();
            let ts_col = batch
                .column(batch.schema().index_of("timestamp_start_unix_ms").unwrap())
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            for i in 0..batch.num_rows() {
                output_timestamps.push(ts_col.value(i));
            }
        }

        assert_eq!(output_timestamps, input_timestamps);

        std::fs::remove_file(input_path).unwrap();
        std::fs::remove_file(output_path).unwrap();
    }

    const NEW_COLUMNS: [&str; 5] = [
        "b10_request_id",
        "prefill_worker_id",
        "decode_worker_id",
        "prefill_dp_rank",
        "decode_dp_rank",
    ];

    /// A recorder-schema parquet payload; `routed` rows carry ids `(prefill, decode)`.
    fn make_request_log_parquet(routed: &[(Option<u64>, Option<u64>)]) -> Bytes {
        use crate::__private::{RequestLogEntry, RequestTraceFactory, RequestTraceOptions};

        let factory = RequestTraceFactory::new(RequestTraceOptions::default()).unwrap();
        let block: Vec<u32> = (0..64).collect();
        let entries: Vec<RequestLogEntry> = routed
            .iter()
            .map(|&(prefill, decode)| {
                let mut entry = factory
                    .new_trace(&block, None, vec![], None)
                    .build_log_entry(false);
                entry.b10_request_id = Some("b10-digest".to_string());
                entry.prefill_worker_id = prefill;
                entry.prefill_dp_rank = prefill.map(|_| 0);
                entry.decode_worker_id = decode;
                entry.decode_dp_rank = decode.map(|_| 2);
                entry
            })
            .collect();
        Bytes::from(crate::request_trace::serialize_to_parquet(&entries).unwrap())
    }

    /// The same payload written with the pre-0.0.20 schema (no routing columns).
    fn without_new_columns(payload: Bytes) -> Vec<u8> {
        let batch = ParquetRecordBatchReaderBuilder::try_new(payload)
            .unwrap()
            .build()
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        let schema = batch.schema();
        let keep: Vec<usize> = (0..schema.fields().len())
            .filter(|&i| !NEW_COLUMNS.contains(&schema.field(i).name().as_str()))
            .collect();
        let old = batch.project(&keep).unwrap();
        let mut buf = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut buf, old.schema(), None).unwrap();
        writer.write(&old).unwrap();
        writer.close().unwrap();
        buf
    }

    fn read_parquet(path: &Path) -> RecordBatch {
        let batches: Vec<RecordBatch> =
            ParquetRecordBatchReaderBuilder::try_new(Bytes::from(std::fs::read(path).unwrap()))
                .unwrap()
                .build()
                .unwrap()
                .map(Result::unwrap)
                .collect();
        arrow::compute::concat_batches(&batches[0].schema(), &batches).unwrap()
    }

    #[tokio::test]
    async fn compact_mixed_old_and_new_request_log_schemas_null_fills_new_columns() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let old = without_new_columns(make_request_log_parquet(&[(Some(1), Some(2))]));
        let new = make_request_log_parquet(&[(Some(7), Some(8))]);
        for (name, payload) in [("part-0001", Bytes::from(old)), ("part-0002", new)] {
            store
                .put_opts(
                    &ObjectPath::from(format!("prefix/{name}.parquet")),
                    PutPayload::from(payload),
                    PutOptions::default(),
                )
                .await
                .unwrap();
        }

        let output_path = temp_output_path("mixed-schema");
        let result =
            compact_store_parquet_prefix_to_local(store, "prefix", output_path.as_path(), 2)
                .await
                .unwrap();
        assert_eq!(result.rows_written, 2);

        let batch = read_parquet(&output_path);
        std::fs::remove_file(&output_path).unwrap();
        for name in NEW_COLUMNS {
            let column = batch.column_by_name(name).unwrap();
            assert_eq!(
                column.null_count(),
                1,
                "{name}: only the old-schema row is null"
            );
        }
        let prefill = batch.column_by_name("prefill_worker_id").unwrap();
        let prefill = prefill.as_any().downcast_ref::<UInt64Array>().unwrap();
        let values: Vec<Option<u64>> = prefill.iter().collect();
        assert!(values.contains(&Some(7)) && values.contains(&None));
    }

    #[test]
    fn anonymize_drops_b10_request_id_and_densifies_worker_ids() {
        use arrow::array::UInt32Array;

        let input_path = temp_output_path("anonymize-routing-input");
        let output_path = temp_output_path("anonymize-routing-output");
        let payload = make_request_log_parquet(&[
            (Some(9_000_000_000_000_000_001), Some(42)),
            (Some(5), None),
            (Some(9_000_000_000_000_000_001), Some(42)),
            (None, None),
        ]);
        std::fs::write(&input_path, &payload).unwrap();

        anonymize_parquet_file(
            &input_path,
            &output_path,
            &["prefill_worker_id", "decode_worker_id"],
            &["b10_request_id"],
            0,
        )
        .unwrap();
        let batch = read_parquet(&output_path);
        std::fs::remove_file(&input_path).unwrap();
        std::fs::remove_file(&output_path).unwrap();

        assert!(batch.column_by_name("b10_request_id").is_none());
        let ids = |name: &str| -> Vec<Option<u64>> {
            let column = batch.column_by_name(name).unwrap();
            column
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .iter()
                .collect()
        };
        assert_eq!(
            ids("prefill_worker_id"),
            vec![Some(1), Some(2), Some(1), None]
        );
        assert_eq!(ids("decode_worker_id"), vec![Some(1), None, Some(1), None]);
        let ranks = |name: &str| -> Vec<Option<u32>> {
            let column = batch.column_by_name(name).unwrap();
            column
                .as_any()
                .downcast_ref::<UInt32Array>()
                .unwrap()
                .iter()
                .collect()
        };
        assert_eq!(
            ranks("prefill_dp_rank"),
            vec![Some(0), Some(0), Some(0), None]
        );
        assert_eq!(ranks("decode_dp_rank"), vec![Some(2), None, Some(2), None]);
    }

    fn published_lists() -> (Vec<String>, Vec<String>) {
        let config = AnonymizeConfig::published();
        (config.columns_to_pseudonymize, config.columns_to_drop)
    }

    fn as_refs(columns: &[String]) -> Vec<&str> {
        columns.iter().map(String::as_str).collect()
    }

    #[test]
    fn anonymize_with_published_defaults_accepts_old_schema_files() {
        let input_path = temp_output_path("anonymize-old-schema-input");
        let output_path = temp_output_path("anonymize-old-schema-output");
        let old = without_new_columns(make_request_log_parquet(&[(None, None), (None, None)]));
        std::fs::write(&input_path, &old).unwrap();
        let (pseudonymize, drop) = published_lists();

        anonymize_parquet_file(
            &input_path,
            &output_path,
            &as_refs(&pseudonymize),
            &as_refs(&drop),
            0,
        )
        .unwrap();
        let batch = read_parquet(&output_path);
        std::fs::remove_file(&input_path).unwrap();
        std::fs::remove_file(&output_path).unwrap();

        assert_eq!(batch.num_rows(), 2);
        for name in NEW_COLUMNS {
            assert!(
                batch.column_by_name(name).is_none(),
                "{name} must not be invented"
            );
        }
        // request_org_id is in both lists, so it is pseudonymized rather than dropped.
        let org = batch.column_by_name("request_org_id").unwrap();
        let org = org.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!((org.value(0), org.value(1)), ("1", "1"));
    }

    #[tokio::test]
    async fn compact_then_anonymize_with_published_defaults_handles_mixed_schemas() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let old = without_new_columns(make_request_log_parquet(&[(None, None)]));
        let new = make_request_log_parquet(&[(Some(901), Some(902)), (Some(905), None)]);
        for (name, payload) in [("part-0001", Bytes::from(old)), ("part-0002", new)] {
            store
                .put_opts(
                    &ObjectPath::from(format!("prefix/{name}.parquet")),
                    PutPayload::from(payload),
                    PutOptions::default(),
                )
                .await
                .unwrap();
        }
        let compacted_path = temp_output_path("mixed-anonymize-compacted");
        let output_path = temp_output_path("mixed-anonymize-output");
        // The same two steps compact_s3_parquet_to_local runs with anonymize set.
        compact_store_parquet_prefix_to_local(store, "prefix", compacted_path.as_path(), 2)
            .await
            .unwrap();
        let (pseudonymize, drop) = published_lists();
        anonymize_parquet_file(
            &compacted_path,
            &output_path,
            &as_refs(&pseudonymize),
            &as_refs(&drop),
            0,
        )
        .unwrap();
        let batch = read_parquet(&output_path);
        std::fs::remove_file(&compacted_path).unwrap();
        std::fs::remove_file(&output_path).unwrap();

        assert_eq!(batch.num_rows(), 3);
        assert!(batch.column_by_name("b10_request_id").is_none());
        let mut prefill: Vec<Option<u64>> = batch
            .column_by_name("prefill_worker_id")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .iter()
            .collect();
        prefill.sort();
        // Old row stays null; the two raw ids become dense ids 1 and 2.
        assert_eq!(prefill, vec![None, Some(1), Some(2)]);
    }
}
