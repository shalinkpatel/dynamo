use crate::metrics::{
    DistributionMetricHandle, MetricLabels, MetricSeriesSnapshot, PerformanceMetricsRegistry,
    RateMetricHandle, RatioMetricHandle,
};
use arc_swap::ArcSwap;
use blake3::Hasher;
use bytes::Bytes;
use object_store::aws::{AmazonS3Builder, AwsCredential};
use object_store::path::Path as ObjectPath;
use object_store::{CredentialProvider, ObjectStore, PutOptions, PutPayload};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::runtime::Builder as RuntimeBuilder;
use tokio::sync::mpsc;

const REQUEST_TRACE_HASH_DOMAIN: &[u8] = b"pm-request-trace-v2";
const REQUEST_SESSION_HASH_DOMAIN: &[u8] = b"pm-request-session-v1";
const B10_REQUEST_ID_HASH_DOMAIN: &[u8] = b"pm-b10-request-id-v1";

/// Blake3 hash output size in bytes (256 bits = 32 bytes)
const HASH_SIZE: usize = 32;
pub const DEFAULT_REQUEST_LOG_VERSION: &str = env!("CARGO_PKG_VERSION");
pub const DEFAULT_BLOCK_SIZE: usize = 64;
pub const DEFAULT_REQUEST_LOG_BATCH_SIZE: usize = 2048;
pub const DEFAULT_REQUEST_LOG_QUEUE_CAPACITY: usize = 4096;
pub const DEFAULT_LOCAL_OUTPUT_DIR: &str = "/tmp/records";
static COMPONENT_RECORDER_ID: OnceLock<String> = OnceLock::new();

fn component_recorder_id() -> &'static str {
    COMPONENT_RECORDER_ID
        .get_or_init(|| format!("{:032x}", rand::random::<u128>()))
        .as_str()
}

/// Fixed-size array for a 32-byte Blake3 hash.
pub type HashBytes = [u8; 32];

#[derive(Deserialize)]
#[serde(untagged)]
enum JsonHashBytes {
    Hex(String),
    Bytes(HashBytes),
}

fn encode_hash_hex(hash: &HashBytes) -> String {
    let mut out = String::with_capacity(HASH_SIZE * 2);
    for byte in hash {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

fn decode_hash_hex(value: &str) -> Result<HashBytes, String> {
    if value.len() != HASH_SIZE * 2 {
        return Err(format!(
            "expected {} hex chars, got {}",
            HASH_SIZE * 2,
            value.len()
        ));
    }

    let mut out = [0u8; HASH_SIZE];
    let (pairs, _) = value.as_bytes().as_chunks::<2>();
    for (i, chunk) in pairs.iter().enumerate() {
        let piece = std::str::from_utf8(chunk).map_err(|_| "hash contains non-utf8 bytes")?;
        out[i] = u8::from_str_radix(piece, 16)
            .map_err(|_| format!("invalid hex at byte offset {}", i * 2))?;
    }
    Ok(out)
}

fn serialize_hash_bytes_as_hex<S>(hashes: &[HashBytes], serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let hex_hashes = hashes.iter().map(encode_hash_hex).collect::<Vec<_>>();
    hex_hashes.serialize(serializer)
}

fn deserialize_hash_bytes_from_json<'de, D>(deserializer: D) -> Result<Vec<HashBytes>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let hashes = Vec::<JsonHashBytes>::deserialize(deserializer)?;
    hashes
        .into_iter()
        .map(|hash| match hash {
            JsonHashBytes::Hex(value) => decode_hash_hex(&value).map_err(serde::de::Error::custom),
            JsonHashBytes::Bytes(value) => Ok(value),
        })
        .collect()
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct RequestTraceOptions {
    pub block_size: usize,
    pub request_log_enabled: bool,
    pub request_log_hash_salt: Option<String>,
    pub request_log_batch_size: usize,
    pub request_log_queue_capacity: usize,
    pub local_output_dir: Option<PathBuf>,
    pub s3_config: Option<S3SinkConfig>,
    pub model_name: String,
    pub org_id: String,
    pub component_name: Option<String>,
    pub file_format: FileFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FileFormat {
    #[default]
    Json,
    Parquet,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct S3SinkConfig {
    pub bucket: String,
    pub region: String,
    pub endpoint: Option<String>,
    pub proxy_url: Option<String>,
    pub access_key_id: Option<String>,
    pub secret_access_key: Option<String>,
    pub prefix: String,
}

pub(crate) fn validate_s3_proxy_url(proxy_url: &str) -> anyhow::Result<()> {
    let url = reqwest::Url::parse(proxy_url)
        .map_err(|_| anyhow::anyhow!("invalid S3 proxy URL; expected http(s)://host[:port]"))?;
    anyhow::ensure!(
        matches!(url.scheme(), "http" | "https") && url.has_host(),
        "invalid S3 proxy URL; expected http(s)://host[:port]"
    );
    Ok(())
}

impl Default for RequestTraceOptions {
    fn default() -> Self {
        Self {
            block_size: DEFAULT_BLOCK_SIZE,
            request_log_enabled: false,
            request_log_hash_salt: None,
            request_log_batch_size: DEFAULT_REQUEST_LOG_BATCH_SIZE,
            request_log_queue_capacity: DEFAULT_REQUEST_LOG_QUEUE_CAPACITY,
            local_output_dir: Some(PathBuf::from(DEFAULT_LOCAL_OUTPUT_DIR)),
            s3_config: None,
            model_name: "unset_model_name".to_string(),
            org_id: String::new(),
            component_name: None,
            file_format: FileFormat::Json,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct RequestMetricsOptions {
    pub trace: RequestTraceOptions,
    pub metric_prefix: String,
    pub request_quantiles: Vec<f64>,
    pub request_sample_period_seconds: Option<f64>,
    pub request_window_seconds: Option<f64>,
    pub input_tokens_quantiles: Vec<f64>,
    pub input_tokens_sample_period_seconds: Option<f64>,
    pub input_tokens_window_seconds: Option<f64>,
    pub ttft_quantiles: Vec<f64>,
    pub ttft_window_seconds: Option<f64>,
    pub time_to_fifth_token_quantiles: Vec<f64>,
    pub time_to_fifth_token_window_seconds: Option<f64>,
    pub first_to_fifth_token_itl_quantiles: Vec<f64>,
    pub first_to_fifth_token_itl_window_seconds: Option<f64>,
    pub ttft_per_input_token_quantiles: Vec<f64>,
    pub ttft_per_input_token_window_seconds: Option<f64>,
    pub itl_quantiles: Vec<f64>,
    pub itl_window_seconds: Option<f64>,
    pub pre_first_token_cancellation_quantiles: Vec<f64>,
    pub pre_first_token_cancellation_sample_period_seconds: Option<f64>,
    pub pre_first_token_cancellation_window_seconds: Option<f64>,
    pub pre_20_token_cancellation_quantiles: Vec<f64>,
    pub pre_20_token_cancellation_sample_period_seconds: Option<f64>,
    pub pre_20_token_cancellation_window_seconds: Option<f64>,
    pub mid_stream_cancellation_quantiles: Vec<f64>,
    pub mid_stream_cancellation_sample_period_seconds: Option<f64>,
    pub mid_stream_cancellation_window_seconds: Option<f64>,
    pub successful_request_quantiles: Vec<f64>,
    pub successful_request_sample_period_seconds: Option<f64>,
    pub successful_request_window_seconds: Option<f64>,
    pub kv_cache_hit_rate_quantiles: Vec<f64>,
    pub kv_cache_hit_rate_window_seconds: Option<f64>,
    pub itl_sample_rate: f64,
}

impl Default for RequestMetricsOptions {
    fn default() -> Self {
        Self {
            trace: RequestTraceOptions::default(),
            metric_prefix: "request_metrics".to_string(),
            request_quantiles: vec![0.1, 0.5, 0.9],
            request_sample_period_seconds: Some(1.0),
            request_window_seconds: None,
            input_tokens_quantiles: vec![0.01, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.99],
            input_tokens_sample_period_seconds: Some(1.0),
            input_tokens_window_seconds: None,
            ttft_quantiles: vec![0.01, 0.1, 0.5, 0.9, 0.99],
            ttft_window_seconds: None,
            time_to_fifth_token_quantiles: vec![0.01, 0.1, 0.5, 0.9, 0.99],
            time_to_fifth_token_window_seconds: None,
            first_to_fifth_token_itl_quantiles: vec![0.01, 0.1, 0.5, 0.9, 0.99],
            first_to_fifth_token_itl_window_seconds: None,
            ttft_per_input_token_quantiles: vec![0.01, 0.1, 0.5, 0.8, 0.9, 0.99],
            ttft_per_input_token_window_seconds: None,
            itl_quantiles: vec![0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999],
            itl_window_seconds: None,
            pre_first_token_cancellation_quantiles: vec![],
            pre_first_token_cancellation_sample_period_seconds: Some(1.0),
            pre_first_token_cancellation_window_seconds: None,
            pre_20_token_cancellation_quantiles: vec![],
            pre_20_token_cancellation_sample_period_seconds: Some(1.0),
            pre_20_token_cancellation_window_seconds: None,
            mid_stream_cancellation_quantiles: vec![],
            mid_stream_cancellation_sample_period_seconds: Some(1.0),
            mid_stream_cancellation_window_seconds: None,
            successful_request_quantiles: vec![],
            successful_request_sample_period_seconds: Some(1.0),
            successful_request_window_seconds: None,
            kv_cache_hit_rate_quantiles: vec![
                0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999,
            ],
            kv_cache_hit_rate_window_seconds: None,
            itl_sample_rate: 0.05,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestLogEntry {
    pub timestamp_start_unix_ms: u64,
    pub duration_e2e_ms: u64,
    pub duration_ttft_ms: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(
        serialize_with = "serialize_hash_bytes_as_hex",
        deserialize_with = "deserialize_hash_bytes_from_json"
    )]
    pub total_hashes: Vec<HashBytes>,
    pub provided_session_id: Option<String>,
    pub request_canceled: bool,
    pub cached_tokens_reference: Option<u64>,
    pub model_name: String,
    pub org_id: String,
    #[serde(default)]
    pub component_name: Option<String>,
    #[serde(default)]
    pub component_id: Option<String>,
    pub block_size: usize,
    pub features: Vec<String>,
    pub speculation_ratio: Option<f64>,
    pub mean_output_tokens_per_step: Option<f64>,
    pub service_tier: Option<String>,
    #[serde(default)]
    pub b10_request_id: Option<String>,
    #[serde(default)]
    pub prefill_worker_id: Option<u64>,
    #[serde(default)]
    pub decode_worker_id: Option<u64>,
    #[serde(default)]
    pub prefill_dp_rank: Option<u32>,
    #[serde(default)]
    pub decode_dp_rank: Option<u32>,
    pub __version__: String,
}

#[derive(Clone)]
pub struct RequestTraceFactory {
    recorder: Arc<EventRecorder>,
    hash_key: Option<[u8; 32]>,
    block_size: usize,
    model_name: String,
    org_id: String,
    component_name: Option<String>,
    component_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct RequestFeatures(u8);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum RequestFeatureKind {
    XGrammar,
    Tools,
    Image,
}

impl RequestFeatureKind {
    const ALL: [Self; 3] = [Self::XGrammar, Self::Tools, Self::Image];

    const fn bit(self) -> u8 {
        match self {
            Self::XGrammar => 1 << 0,
            Self::Tools => 1 << 1,
            Self::Image => 1 << 2,
        }
    }

    const fn metric_suffix(self) -> &'static str {
        match self {
            Self::XGrammar => "xgrammar",
            Self::Tools => "tools",
            Self::Image => "image",
        }
    }
}

impl RequestFeatures {
    pub const NONE: Self = Self(0);
    pub const XGRAMMAR: Self = Self(RequestFeatureKind::XGrammar.bit());
    pub const TOOLS: Self = Self(RequestFeatureKind::Tools.bit());
    pub const IMAGE: Self = Self(RequestFeatureKind::Image.bit());

    pub const fn empty() -> Self {
        Self::NONE
    }

    pub const fn bits(self) -> u8 {
        self.0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    pub const fn from_bools(xgrammar: bool, tools: bool, image: bool) -> Self {
        let mut bits = 0;
        if xgrammar {
            bits |= Self::XGRAMMAR.0;
        }
        if tools {
            bits |= Self::TOOLS.0;
        }
        if image {
            bits |= Self::IMAGE.0;
        }
        Self(bits)
    }

    fn includes(self, feature: RequestFeatureKind) -> bool {
        self.0 & feature.bit() != 0
    }

    pub fn to_vec(self) -> Vec<String> {
        RequestFeatureKind::ALL
            .iter()
            .filter(|&&f| self.includes(f))
            .map(|f| f.metric_suffix().to_string())
            .collect()
    }
}

impl fmt::Display for RequestFeatures {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return f.write_str("none");
        }
        let mut first = true;
        for feature in RequestFeatureKind::ALL {
            if !self.includes(feature) {
                continue;
            }
            if !first {
                f.write_str(",")?;
            }
            f.write_str(feature.metric_suffix())?;
            first = false;
        }
        Ok(())
    }
}

impl std::ops::BitOr for RequestFeatures {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for RequestFeatures {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl From<u8> for RequestFeatures {
    fn from(bits: u8) -> Self {
        Self(bits & (Self::XGRAMMAR.0 | Self::TOOLS.0 | Self::IMAGE.0))
    }
}

/// Service tier label applied to the major metrics on their `feature="all"`
/// series. When `None`, no `service_tier` label is emitted (backward compatible).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServiceTier {
    Fast,
    Priority,
    Default,
    Flex,
}

impl ServiceTier {
    pub const fn label_value(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Priority => "priority",
            Self::Default => "default",
            Self::Flex => "flex",
        }
    }
}

impl std::str::FromStr for ServiceTier {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> anyhow::Result<Self> {
        match value {
            "fast" => Ok(Self::Fast),
            "priority" => Ok(Self::Priority),
            "default" => Ok(Self::Default),
            "flex" => Ok(Self::Flex),
            other => anyhow::bail!(
                "unknown service tier '{other}', expected one of: fast, priority, default, flex"
            ),
        }
    }
}

impl fmt::Display for ServiceTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label_value())
    }
}

/// Config for the major metrics that can carry a `service_tier` and/or
/// `dp_rank` label. Reuses the same quantiles/window/sample-period as the
/// corresponding plain metric.
///
/// `service_tier` and `dp_rank` are kept **independent** — a request carrying
/// both produces a tier-only series and a rank-only series, never a combined
/// `{service_tier, dp_rank}` series. This keeps cardinality linear in the
/// number of tiers plus ranks, rather than the product.
#[derive(Clone)]
struct TieredMetricConfig {
    metric_prefix: String,
    request_quantiles: Vec<f64>,
    request_sample_period_seconds: Option<f64>,
    request_window_seconds: Option<f64>,
    ttft_quantiles: Vec<f64>,
    ttft_window_seconds: Option<f64>,
    time_to_fifth_token_quantiles: Vec<f64>,
    time_to_fifth_token_window_seconds: Option<f64>,
    itl_quantiles: Vec<f64>,
    itl_window_seconds: Option<f64>,
    kv_cache_hit_rate_quantiles: Vec<f64>,
    kv_cache_hit_rate_window_seconds: Option<f64>,
    successful_request_quantiles: Vec<f64>,
    successful_request_sample_period_seconds: Option<f64>,
    successful_request_window_seconds: Option<f64>,
    /// Cache of resolved handles per `ServiceTier`, so only the first request
    /// for a given tier touches the registry. Stores `Arc<TieredMetricHandles>`
    /// so a cache hit is a single `Arc::clone` (one atomic op) instead of
    /// cloning six individual handles.
    ///
    /// # Deadlock safety
    ///
    /// The cache mutex is released before calling `resolve_one` (which acquires
    /// the registry mutex). The two cache mutexes are never held simultaneously.
    /// This ensures no lock-ordering deadlock can occur, even under concurrent
    /// `new_request` calls with mixed tiers and ranks.
    tier_cache: Arc<Mutex<HashMap<ServiceTier, Arc<TieredMetricHandles>>>>,
    /// Cache of resolved handles per DP rank, so only the first request for a
    /// given rank touches the registry. See `tier_cache` for deadlock-safety
    /// notes.
    rank_cache: Arc<Mutex<HashMap<u8, Arc<TieredMetricHandles>>>>,
}

/// Lazily-resolved handles for the major metrics carrying a single extra
/// dimension (`service_tier` **or** `dp_rank`, never both on the same series).
/// Each records in parallel with the corresponding unlabelled handle so that
/// callers passing `None` for both see no change.
#[derive(Clone)]
struct TieredMetricHandles {
    request: RateMetricHandle,
    ttft_ms: DistributionMetricHandle,
    time_to_fifth_token_ms: DistributionMetricHandle,
    itl_ms: DistributionMetricHandle,
    kv_cache_hit_rate: RatioMetricHandle,
    successful_request: RateMetricHandle,
}

/// Maximum valid DP rank for labeling. Ranks `0..=DP_RANK_MAX` produce a
/// `dp_rank`-labeled series; ranks above `DP_RANK_MAX` are normalized to
/// `None` (no rank-labeled series, no error).
const DP_RANK_MAX: u8 = 31;

impl TieredMetricConfig {
    /// Resolve labeled handles for the given extra dimensions. Returns a
    /// fixed-size array `[tier, rank]` where each slot is `None` when that
    /// dimension is absent. This avoids a heap allocation per `new_request`.
    ///
    /// Ranks above `DP_RANK_MAX` are silently normalized to `None` (no
    /// rank-labeled series, no error).
    fn resolve(
        &self,
        registry: &PerformanceMetricsRegistry,
        tier: Option<ServiceTier>,
        dp_rank: Option<u8>,
    ) -> anyhow::Result<[Option<Arc<TieredMetricHandles>>; 2]> {
        let dp_rank = dp_rank.filter(|&r| r <= DP_RANK_MAX);
        Ok([
            tier.map(|t| self.resolve_tier(registry, t)).transpose()?,
            dp_rank
                .map(|r| self.resolve_rank(registry, r))
                .transpose()?,
        ])
    }

    /// Look up (or create and cache) the handles for a `service_tier`.
    fn resolve_tier(
        &self,
        registry: &PerformanceMetricsRegistry,
        tier: ServiceTier,
    ) -> anyhow::Result<Arc<TieredMetricHandles>> {
        if let Some(cached) = self.tier_cache.lock().unwrap().get(&tier) {
            return Ok(Arc::clone(cached));
        }
        let mut extra = MetricLabels::new();
        extra.insert("service_tier".to_string(), tier.label_value().to_string());
        let handles = Arc::new(self.resolve_one(registry, &extra)?);
        self.tier_cache
            .lock()
            .unwrap()
            .insert(tier, Arc::clone(&handles));
        Ok(handles)
    }

    /// Look up (or create and cache) the handles for a `dp_rank`.
    fn resolve_rank(
        &self,
        registry: &PerformanceMetricsRegistry,
        rank: u8,
    ) -> anyhow::Result<Arc<TieredMetricHandles>> {
        if let Some(cached) = self.rank_cache.lock().unwrap().get(&rank) {
            return Ok(Arc::clone(cached));
        }
        let mut extra = MetricLabels::new();
        extra.insert("dp_rank".to_string(), rank.to_string());
        let handles = Arc::new(self.resolve_one(registry, &extra)?);
        self.rank_cache
            .lock()
            .unwrap()
            .insert(rank, Arc::clone(&handles));
        Ok(handles)
    }

    /// Build a single `TieredMetricHandles` for the given extra label set.
    ///
    /// Distribution metrics (`ttft_ms`, `time_to_fifth_token_ms`, `itl_ms`)
    /// carry `feature="all"` plus the extra labels. Rate/ratio metrics
    /// (`request`, `successful_request`, `kv_cache_hit_rate`) carry only the
    /// extra labels (no `feature` label, matching their plain counterparts).
    fn resolve_one(
        &self,
        registry: &PerformanceMetricsRegistry,
        extra: &MetricLabels,
    ) -> anyhow::Result<TieredMetricHandles> {
        let prefix = self.metric_prefix.as_str();
        let mut all_labels = feature_metric_labels("all");
        for (k, v) in extra {
            all_labels.insert(k.clone(), v.clone());
        }
        Ok(TieredMetricHandles {
            request: registry.get_or_create_rate_metric_with_labels(
                format!("{prefix}_request"),
                extra.clone(),
                self.request_quantiles.clone(),
                self.request_sample_period_seconds,
                self.request_window_seconds,
            )?,
            ttft_ms: registry.get_or_create_distribution_metric_with_labels(
                format!("{prefix}_ttft_ms"),
                all_labels.clone(),
                self.ttft_quantiles.clone(),
                self.ttft_window_seconds,
            )?,
            time_to_fifth_token_ms: registry.get_or_create_distribution_metric_with_labels(
                format!("{prefix}_time_to_fifth_token_ms"),
                all_labels.clone(),
                self.time_to_fifth_token_quantiles.clone(),
                self.time_to_fifth_token_window_seconds,
            )?,
            itl_ms: registry.get_or_create_distribution_metric_with_labels(
                format!("{prefix}_itl_ms"),
                all_labels.clone(),
                self.itl_quantiles.clone(),
                self.itl_window_seconds,
            )?,
            kv_cache_hit_rate: registry.get_or_create_ratio_metric_with_labels(
                format!("{prefix}_kv_cache_hit_rate"),
                extra.clone(),
                self.kv_cache_hit_rate_quantiles.clone(),
                self.kv_cache_hit_rate_window_seconds,
            )?,
            successful_request: registry.get_or_create_rate_metric_with_labels(
                format!("{prefix}_successful_request"),
                extra.clone(),
                self.successful_request_quantiles.clone(),
                self.successful_request_sample_period_seconds,
                self.successful_request_window_seconds,
            )?,
        })
    }
}

#[derive(Clone)]
struct RequestFeatureMetricConfig {
    metric_prefix: String,
    request_quantiles: Vec<f64>,
    request_sample_period_seconds: Option<f64>,
    request_window_seconds: Option<f64>,
    ttft_quantiles: Vec<f64>,
    ttft_window_seconds: Option<f64>,
    time_to_fifth_token_quantiles: Vec<f64>,
    time_to_fifth_token_window_seconds: Option<f64>,
    first_to_fifth_token_itl_quantiles: Vec<f64>,
    first_to_fifth_token_itl_window_seconds: Option<f64>,
    itl_quantiles: Vec<f64>,
    itl_window_seconds: Option<f64>,
}

#[derive(Clone)]
struct RequestFeatureMetricHandles {
    request_all: RateMetricHandle,
    request_none: RateMetricHandle,
    request_by_feature: Vec<(RequestFeatureKind, RateMetricHandle)>,
    ttft_none: DistributionMetricHandle,
    ttft_by_feature: Vec<(RequestFeatureKind, DistributionMetricHandle)>,
    time_to_fifth_token_none: DistributionMetricHandle,
    time_to_fifth_token_by_feature: Vec<(RequestFeatureKind, DistributionMetricHandle)>,
    first_to_fifth_token_itl_none: DistributionMetricHandle,
    first_to_fifth_token_itl_by_feature: Vec<(RequestFeatureKind, DistributionMetricHandle)>,
    itl_none: DistributionMetricHandle,
    itl_by_feature: Vec<(RequestFeatureKind, DistributionMetricHandle)>,
}

#[derive(Clone)]
pub struct RequestMetricsFactory {
    trace_factory: RequestTraceFactory,
    metrics: PerformanceMetricsRegistry,
    request: RateMetricHandle,
    input_tokens: RateMetricHandle,
    input_length_tokens: DistributionMetricHandle,
    net_new_input_tokens: RateMetricHandle,
    output_tokens: RateMetricHandle,
    output_length_tokens: DistributionMetricHandle,
    ttft_ms: DistributionMetricHandle,
    time_to_fifth_token_ms: DistributionMetricHandle,
    first_to_fifth_token_itl_ms: DistributionMetricHandle,
    ttft_ms_per_input_token: DistributionMetricHandle,
    ttft_ms_per_net_new_input_token: DistributionMetricHandle,
    itl_ms: DistributionMetricHandle,
    pre_first_token_cancellation: RateMetricHandle,
    pre_20_token_cancellation: RateMetricHandle,
    mid_stream_cancellation: RateMetricHandle,
    successful_request: RateMetricHandle,
    kv_cache_hit_rate: RatioMetricHandle,
    itl_sample_rate: f64,
    totals: Arc<RequestMetricsTotals>,
    feature_metric_config: RequestFeatureMetricConfig,
    feature_metrics: Arc<OnceLock<RequestFeatureMetricHandles>>,
    tiered_metric_config: TieredMetricConfig,
}

pub struct RequestTrace {
    factory: RequestTraceFactory,
    // Sink snapshot taken when the request started; hashing and writing follow it.
    worker: Option<Arc<RecorderWorker>>,
    started_at_wallclock: SystemTime,
    first_token_at: Option<SystemTime>,
    input_tokens: usize,
    output_tokens_len: usize,
    output_steps: usize,
    total_hashes: Vec<HashBytes>,
    provided_session_id: Option<String>,
    previous_hash: Option<[u8; 32]>,
    pending: Vec<u32>,
    terminal: bool,
    cached_tokens: Option<u64>,
    model_name: String,
    org_id: String,
    block_size: usize,
    features: Vec<String>,
    speculation_ratio: Option<f64>,
    service_tier: Option<ServiceTier>,
    b10_request_id: Option<String>,
    routed_workers: RoutedWorkers,
}

/// Workers the router chose for a request, recorded on its log row.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RoutedWorkers {
    pub prefill_worker_id: Option<u64>,
    pub prefill_dp_rank: Option<u32>,
    pub decode_worker_id: Option<u64>,
    pub decode_dp_rank: Option<u32>,
}

pub struct RequestMetricsRequest {
    trace: RequestTrace,
    started_at: Instant,
    input_tokens: u64,
    features: RequestFeatures,
    last_token_at: Option<Instant>,
    last_total_tokens: u64,
    first_token_at: Option<Instant>,
    time_to_fifth_token_recorded: bool,
    first_to_fifth_inflight_active: bool,
    cached_tokens_hint: Option<u64>,
    rng: SmallRng,
    input_tokens_rate: RateMetricHandle,
    net_new_input_tokens_rate: RateMetricHandle,
    output_tokens_rate: RateMetricHandle,
    output_length_tokens: DistributionMetricHandle,
    ttft_ms: DistributionMetricHandle,
    time_to_fifth_token_ms: DistributionMetricHandle,
    first_to_fifth_token_itl_ms: DistributionMetricHandle,
    ttft_ms_per_input_token: DistributionMetricHandle,
    ttft_ms_per_net_new_input_token: DistributionMetricHandle,
    itl_ms: DistributionMetricHandle,
    pre_first_token_cancellation: RateMetricHandle,
    pre_20_token_cancellation: RateMetricHandle,
    mid_stream_cancellation: RateMetricHandle,
    successful_request: RateMetricHandle,
    kv_cache_hit_rate: RatioMetricHandle,
    itl_sample_rate: f64,
    totals: Arc<RequestMetricsTotals>,
    inflight_finished: bool,
    ttft_feature_metrics: Vec<DistributionMetricHandle>,
    time_to_fifth_token_feature_metrics: Vec<DistributionMetricHandle>,
    first_to_fifth_token_itl_feature_metrics: Vec<DistributionMetricHandle>,
    itl_feature_metrics: Vec<DistributionMetricHandle>,
    tiered: [Option<Arc<TieredMetricHandles>>; 2],
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RequestMetricsTotalsSnapshot {
    pub total_requests: u64,
    pub total_successful_requests: u64,
    pub total_pre_first_token_cancellations: u64,
    pub total_pre_20_token_cancellations: u64,
    pub total_mid_stream_cancellations: u64,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub total_net_new_input_tokens: u64,
    pub total_kv_cache_hits_tokens: u64,
    pub total_kv_cache_lookup_tokens: u64,
    pub total_itl_samples_recorded: u64,
    pub total_request_log_rows_persisted: u64,
    pub total_request_log_rows_dropped_queue_full: u64,
    pub total_request_log_rows_dropped_flush_error: u64,
    pub kv_cache_hit_rate_lifetime: Option<f64>,
    pub current_inflight_requests: u64,
    pub mean_inflight_requests: f64,
    pub current_inflight_input_tokens: u64,
    pub mean_inflight_input_tokens: f64,
    pub current_inflight_output_tokens: u64,
    pub mean_inflight_output_tokens: f64,
    pub current_inflight_first_to_fifth_token_requests: u64,
    pub mean_inflight_first_to_fifth_token_requests: f64,
    pub current_inflight_first_to_fifth_token_input_tokens: u64,
    pub mean_inflight_first_to_fifth_token_input_tokens: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RequestMetricsSnapshot {
    pub series: Vec<MetricSeriesSnapshot>,
    pub totals: RequestMetricsTotalsSnapshot,
}

#[derive(Debug, Default)]
struct RequestMetricsTotals {
    total_requests: AtomicU64,
    total_successful_requests: AtomicU64,
    total_pre_first_token_cancellations: AtomicU64,
    total_pre_20_token_cancellations: AtomicU64,
    total_mid_stream_cancellations: AtomicU64,
    total_input_tokens: AtomicU64,
    total_output_tokens: AtomicU64,
    total_net_new_input_tokens: AtomicU64,
    total_kv_cache_hits_tokens: AtomicU64,
    total_kv_cache_lookup_tokens: AtomicU64,
    total_itl_samples_recorded: AtomicU64,
    inflight: InflightTracker,
    first_to_fifth_inflight: InflightTracker,
}

const PRE_20_TOKEN_CANCELLATION_EXCLUSIVE_MAX_OUTPUT_TOKENS: usize = 20;

#[derive(Debug, Clone, Copy)]
struct InflightSnapshot {
    current_requests: u64,
    mean_requests: f64,
    current_input_tokens: u64,
    mean_input_tokens: f64,
    current_output_tokens: u64,
    mean_output_tokens: f64,
}

#[derive(Debug)]
struct InflightTracker {
    started_at: Instant,
    last_request_update_us: AtomicU64,
    last_input_update_us: AtomicU64,
    last_output_update_us: AtomicU64,
    current_requests: AtomicU64,
    current_input_tokens: AtomicU64,
    current_output_tokens: AtomicU64,
    request_micros: AtomicU64,
    input_token_micros: AtomicU64,
    output_token_micros: AtomicU64,
}

impl Default for InflightTracker {
    fn default() -> Self {
        Self {
            started_at: Instant::now(),
            last_request_update_us: AtomicU64::new(0),
            last_input_update_us: AtomicU64::new(0),
            last_output_update_us: AtomicU64::new(0),
            current_requests: AtomicU64::new(0),
            current_input_tokens: AtomicU64::new(0),
            current_output_tokens: AtomicU64::new(0),
            request_micros: AtomicU64::new(0),
            input_token_micros: AtomicU64::new(0),
            output_token_micros: AtomicU64::new(0),
        }
    }
}

impl InflightTracker {
    fn now_us(&self) -> u64 {
        self.started_at.elapsed().as_micros() as u64
    }

    // `fetch_update` is deprecated in Rust 1.99 in favor of `try_update`, which older
    // toolchains lack.
    #[allow(deprecated)]
    fn fetch_add_saturating_get_old(target: &AtomicU64, delta: u64) -> u64 {
        target
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |v| {
                Some(v.saturating_add(delta))
            })
            .unwrap_or_else(|old| old)
    }

    #[allow(deprecated)]
    fn fetch_sub_saturating_get_old(target: &AtomicU64, delta: u64) -> u64 {
        target
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |v| {
                Some(v.saturating_sub(delta))
            })
            .unwrap_or_else(|old| old)
    }

    #[allow(deprecated)]
    fn account_elapsed(area: &AtomicU64, dt_us: u64, value_before: u64) {
        let delta = value_before.saturating_mul(dt_us);
        let _ = area.fetch_update(Ordering::AcqRel, Ordering::Acquire, |v| {
            Some(v.saturating_add(delta))
        });
    }

    fn claim_elapsed_us(&self, last_update_us: &AtomicU64) -> u64 {
        loop {
            let prev = last_update_us.load(Ordering::Acquire);
            let now = self.now_us();
            if now <= prev {
                return 0;
            }
            if last_update_us
                .compare_exchange(prev, now, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return now.saturating_sub(prev);
            }
        }
    }

    fn advance_requests_with_add(&self, add: u64) {
        let dt_us = self.claim_elapsed_us(&self.last_request_update_us);
        let prev = Self::fetch_add_saturating_get_old(&self.current_requests, add);
        Self::account_elapsed(&self.request_micros, dt_us, prev);
    }

    fn advance_requests_with_sub(&self, sub: u64) {
        let dt_us = self.claim_elapsed_us(&self.last_request_update_us);
        let prev = Self::fetch_sub_saturating_get_old(&self.current_requests, sub);
        Self::account_elapsed(&self.request_micros, dt_us, prev);
    }

    fn advance_requests_snapshot(&self) {
        let dt_us = self.claim_elapsed_us(&self.last_request_update_us);
        let current = self.current_requests.load(Ordering::Acquire);
        Self::account_elapsed(&self.request_micros, dt_us, current);
    }

    fn advance_input_with_add(&self, add: u64) {
        let dt_us = self.claim_elapsed_us(&self.last_input_update_us);
        let prev = Self::fetch_add_saturating_get_old(&self.current_input_tokens, add);
        Self::account_elapsed(&self.input_token_micros, dt_us, prev);
    }

    fn advance_input_with_sub(&self, sub: u64) {
        let dt_us = self.claim_elapsed_us(&self.last_input_update_us);
        let prev = Self::fetch_sub_saturating_get_old(&self.current_input_tokens, sub);
        Self::account_elapsed(&self.input_token_micros, dt_us, prev);
    }

    fn advance_input_snapshot(&self) {
        let dt_us = self.claim_elapsed_us(&self.last_input_update_us);
        let current = self.current_input_tokens.load(Ordering::Acquire);
        Self::account_elapsed(&self.input_token_micros, dt_us, current);
    }

    fn advance_output_with_add(&self, add: u64) {
        let dt_us = self.claim_elapsed_us(&self.last_output_update_us);
        let prev = Self::fetch_add_saturating_get_old(&self.current_output_tokens, add);
        Self::account_elapsed(&self.output_token_micros, dt_us, prev);
    }

    fn advance_output_with_sub(&self, sub: u64) {
        let dt_us = self.claim_elapsed_us(&self.last_output_update_us);
        let prev = Self::fetch_sub_saturating_get_old(&self.current_output_tokens, sub);
        Self::account_elapsed(&self.output_token_micros, dt_us, prev);
    }

    fn advance_output_snapshot(&self) {
        let dt_us = self.claim_elapsed_us(&self.last_output_update_us);
        let current = self.current_output_tokens.load(Ordering::Acquire);
        Self::account_elapsed(&self.output_token_micros, dt_us, current);
    }

    fn on_start(&self, input_tokens: u64) {
        self.advance_requests_with_add(1);
        self.advance_input_with_add(input_tokens);
    }

    fn add_output_tokens(&self, delta_tokens: u64) {
        if delta_tokens == 0 {
            return;
        }
        self.advance_output_with_add(delta_tokens);
    }

    fn on_finish(&self, input_tokens: u64, output_tokens: u64) {
        self.advance_requests_with_sub(1);
        self.advance_input_with_sub(input_tokens);
        self.advance_output_with_sub(output_tokens);
    }

    fn snapshot(&self) -> InflightSnapshot {
        let now_us = self.now_us();
        self.advance_requests_snapshot();
        self.advance_input_snapshot();
        self.advance_output_snapshot();
        let elapsed_us = now_us.max(1);
        let denom = elapsed_us as f64;
        let req_micros = self.request_micros.load(Ordering::Acquire) as f64;
        let in_tok_micros = self.input_token_micros.load(Ordering::Acquire) as f64;
        let out_tok_micros = self.output_token_micros.load(Ordering::Acquire) as f64;
        InflightSnapshot {
            current_requests: self.current_requests.load(Ordering::Acquire),
            mean_requests: req_micros / denom,
            current_input_tokens: self.current_input_tokens.load(Ordering::Acquire),
            mean_input_tokens: in_tok_micros / denom,
            current_output_tokens: self.current_output_tokens.load(Ordering::Acquire),
            mean_output_tokens: out_tok_micros / denom,
        }
    }
}

/// Hash a string using unkeyed blake3 to avoid exposing semi sensitive information.
fn hash_unkeyed_string(value: &str) -> String {
    blake3::hash(value.as_bytes()).to_hex().to_string()
}

impl RequestTraceFactory {
    pub fn new(options: RequestTraceOptions) -> anyhow::Result<Self> {
        // Keep log filenames stable without using the model name directly in object keys.
        let model_name_hash = hash_unkeyed_string(&options.model_name);

        // Hash the org_id using unkeyed blake3 to avoid exposing sensitive organization identifiers
        let org_id_hash = hash_unkeyed_string(&options.org_id);

        println!(
            "Initializing RequestTraceFactory, request_log_enabled={}, s3={:?}",
            options.request_log_enabled,
            options.s3_config.is_some(),
        );

        let recorder = Arc::new(EventRecorder::new(model_name_hash));
        recorder.configure(&options)?;

        Ok(Self {
            recorder,
            hash_key: options
                .request_log_hash_salt
                .as_deref()
                .map(|salt| *blake3::hash(salt.as_bytes()).as_bytes()),
            block_size: options.block_size.max(1),
            model_name: options.model_name,
            org_id: org_id_hash,
            component_name: options.component_name,
            component_id: component_recorder_id().to_string(),
        })
    }

    /// Turn request logging on or off, or move it to a new sink, from the
    /// request-log fields of `options` (`request_log_enabled`, batch size,
    /// queue capacity, local dir, S3, file format). The other fields are fixed
    /// at construction and ignored. See [`EventRecorder::configure`].
    pub fn configure_request_log(&self, options: &RequestTraceOptions) -> anyhow::Result<()> {
        self.recorder.configure(options)
    }

    pub fn request_log_enabled(&self) -> bool {
        self.recorder.is_enabled()
    }

    /// Create a trace with token array (includes hashing if block_size permits).
    pub fn new_trace(
        &self,
        input_token_ids: &[u32],
        provided_session_id: Option<&str>,
        features: Vec<String>,
        service_tier: Option<ServiceTier>,
    ) -> RequestTrace {
        let mut trace = self.create_trace_base(
            input_token_ids.len(),
            provided_session_id,
            None,
            features,
            service_tier,
        );
        trace.append_tokens(input_token_ids);
        trace
    }

    /// Create a trace with just the token count (no hashing, no token array needed).
    pub fn new_trace_with_count(
        &self,
        input_token_count: usize,
        provided_session_id: Option<&str>,
        features: Vec<String>,
        service_tier: Option<ServiceTier>,
    ) -> RequestTrace {
        // Skip append_tokens - no hashing when using count-only
        self.create_trace_base(
            input_token_count,
            provided_session_id,
            None,
            features,
            service_tier,
        )
    }

    /// Common initialization for both trace creation methods.
    fn create_trace_base(
        &self,
        input_token_count: usize,
        provided_session_id: Option<&str>,
        b10_request_id: Option<&str>,
        features: Vec<String>,
        service_tier: Option<ServiceTier>,
    ) -> RequestTrace {
        let worker = self.recorder.current_worker();
        RequestTrace {
            factory: self.clone(),
            worker,
            started_at_wallclock: SystemTime::now(),
            first_token_at: None,
            input_tokens: input_token_count,
            output_tokens_len: 0,
            output_steps: 0,
            total_hashes: Vec::new(),
            provided_session_id: provided_session_id.map(hash_session_id),
            previous_hash: None,
            pending: Vec::with_capacity(self.block_size.min(input_token_count)),
            terminal: false,
            cached_tokens: None,
            model_name: self.model_name.clone(),
            org_id: self.org_id.clone(),
            block_size: self.block_size,
            features,
            speculation_ratio: None,
            service_tier,
            b10_request_id: b10_request_id
                .filter(|id| !id.is_empty())
                .map(hash_b10_request_id),
            routed_workers: RoutedWorkers::default(),
        }
    }
}

impl RequestMetricsFactory {
    pub fn new(mut options: RequestMetricsOptions) -> anyhow::Result<Self> {
        options.itl_sample_rate = options.itl_sample_rate.clamp(0.0, 1.0);
        let trace_factory = RequestTraceFactory::new(options.trace)?;
        let metrics = PerformanceMetricsRegistry::default();
        let prefix = options.metric_prefix.as_str();

        let request = metrics.new_rate_metric(
            format!("{prefix}_request"),
            options.request_quantiles.clone(),
            options.request_sample_period_seconds,
            options.request_window_seconds,
        )?;
        let input_tokens = metrics.new_rate_metric(
            format!("{prefix}_input_tokens"),
            options.input_tokens_quantiles.clone(),
            options.input_tokens_sample_period_seconds,
            options.input_tokens_window_seconds,
        )?;
        let input_length_tokens = metrics.new_distribution_metric(
            format!("{prefix}_input_length_tokens"),
            options.input_tokens_quantiles.clone(),
            options.input_tokens_window_seconds,
        )?;
        let net_new_input_tokens = metrics.new_rate_metric(
            format!("{prefix}_net_new_input_tokens"),
            options.input_tokens_quantiles.clone(),
            options.input_tokens_sample_period_seconds,
            options.input_tokens_window_seconds,
        )?;
        let output_tokens = metrics.new_rate_metric(
            format!("{prefix}_output_tokens"),
            options.input_tokens_quantiles.clone(),
            options.input_tokens_sample_period_seconds,
            options.input_tokens_window_seconds,
        )?;
        let output_length_tokens = metrics.new_distribution_metric(
            format!("{prefix}_output_length_tokens"),
            options.input_tokens_quantiles.clone(),
            options.input_tokens_window_seconds,
        )?;
        let ttft_ms = metrics.new_distribution_metric_with_labels(
            format!("{prefix}_ttft_ms"),
            feature_metric_labels("all"),
            options.ttft_quantiles.clone(),
            options.ttft_window_seconds,
        )?;
        let time_to_fifth_token_ms = metrics.new_distribution_metric_with_labels(
            format!("{prefix}_time_to_fifth_token_ms"),
            feature_metric_labels("all"),
            options.time_to_fifth_token_quantiles.clone(),
            options.time_to_fifth_token_window_seconds,
        )?;
        let first_to_fifth_token_itl_ms = metrics.new_distribution_metric_with_labels(
            format!("{prefix}_first_to_fifth_token_itl_ms"),
            feature_metric_labels("all"),
            options.first_to_fifth_token_itl_quantiles.clone(),
            options.first_to_fifth_token_itl_window_seconds,
        )?;
        let ttft_ms_per_input_token = metrics.new_distribution_metric(
            format!("{prefix}_ttft_ms_per_input_token"),
            options.ttft_per_input_token_quantiles.clone(),
            options.ttft_per_input_token_window_seconds,
        )?;
        let ttft_ms_per_net_new_input_token = metrics.new_distribution_metric(
            format!("{prefix}_ttft_ms_per_net_new_input_token"),
            options.ttft_per_input_token_quantiles.clone(),
            options.ttft_per_input_token_window_seconds,
        )?;
        let itl_ms = metrics.new_distribution_metric_with_labels(
            format!("{prefix}_itl_ms"),
            feature_metric_labels("all"),
            options.itl_quantiles.clone(),
            options.itl_window_seconds,
        )?;
        let pre_first_token_cancellation = metrics.new_rate_metric(
            format!("{prefix}_pre_first_token_cancellation"),
            options.pre_first_token_cancellation_quantiles.clone(),
            options.pre_first_token_cancellation_sample_period_seconds,
            options.pre_first_token_cancellation_window_seconds,
        )?;
        let pre_20_token_cancellation = metrics.new_rate_metric(
            format!("{prefix}_pre_20_token_cancellation"),
            options.pre_20_token_cancellation_quantiles.clone(),
            options.pre_20_token_cancellation_sample_period_seconds,
            options.pre_20_token_cancellation_window_seconds,
        )?;
        let mid_stream_cancellation = metrics.new_rate_metric(
            format!("{prefix}_mid_stream_cancellation"),
            options.mid_stream_cancellation_quantiles.clone(),
            options.mid_stream_cancellation_sample_period_seconds,
            options.mid_stream_cancellation_window_seconds,
        )?;
        let successful_request = metrics.new_rate_metric(
            format!("{prefix}_successful_request"),
            options.successful_request_quantiles.clone(),
            options.successful_request_sample_period_seconds,
            options.successful_request_window_seconds,
        )?;
        let kv_cache_hit_rate_quantiles = options.kv_cache_hit_rate_quantiles.clone();
        let kv_cache_hit_rate = metrics.new_ratio_metric(
            format!("{prefix}_kv_cache_hit_rate"),
            kv_cache_hit_rate_quantiles.clone(),
            options.kv_cache_hit_rate_window_seconds,
        )?;

        println!(
            "Registered RequestMetricsFactory with prefix '{}'",
            options.metric_prefix
        );

        Ok(Self {
            trace_factory,
            metrics,
            request,
            input_tokens,
            input_length_tokens,
            net_new_input_tokens,
            output_tokens,
            output_length_tokens,
            ttft_ms,
            time_to_fifth_token_ms,
            first_to_fifth_token_itl_ms,
            ttft_ms_per_input_token,
            ttft_ms_per_net_new_input_token,
            itl_ms,
            pre_first_token_cancellation,
            pre_20_token_cancellation,
            mid_stream_cancellation,
            successful_request,
            kv_cache_hit_rate,
            itl_sample_rate: options.itl_sample_rate,
            totals: Arc::new(RequestMetricsTotals::default()),
            feature_metric_config: RequestFeatureMetricConfig {
                metric_prefix: options.metric_prefix.clone(),
                request_quantiles: options.request_quantiles.clone(),
                request_sample_period_seconds: options.request_sample_period_seconds,
                request_window_seconds: options.request_window_seconds,
                ttft_quantiles: options.ttft_quantiles.clone(),
                ttft_window_seconds: options.ttft_window_seconds,
                time_to_fifth_token_quantiles: options.time_to_fifth_token_quantiles.clone(),
                time_to_fifth_token_window_seconds: options.time_to_fifth_token_window_seconds,
                first_to_fifth_token_itl_quantiles: options.first_to_fifth_token_itl_quantiles,
                first_to_fifth_token_itl_window_seconds: options
                    .first_to_fifth_token_itl_window_seconds,
                itl_quantiles: options.itl_quantiles.clone(),
                itl_window_seconds: options.itl_window_seconds,
            },
            feature_metrics: Arc::new(OnceLock::new()),
            tiered_metric_config: TieredMetricConfig {
                metric_prefix: options.metric_prefix,
                request_quantiles: options.request_quantiles,
                request_sample_period_seconds: options.request_sample_period_seconds,
                request_window_seconds: options.request_window_seconds,
                ttft_quantiles: options.ttft_quantiles,
                ttft_window_seconds: options.ttft_window_seconds,
                time_to_fifth_token_quantiles: options.time_to_fifth_token_quantiles,
                time_to_fifth_token_window_seconds: options.time_to_fifth_token_window_seconds,
                itl_quantiles: options.itl_quantiles,
                itl_window_seconds: options.itl_window_seconds,
                kv_cache_hit_rate_quantiles,
                kv_cache_hit_rate_window_seconds: options.kv_cache_hit_rate_window_seconds,
                successful_request_quantiles: options.successful_request_quantiles,
                successful_request_sample_period_seconds: options
                    .successful_request_sample_period_seconds,
                successful_request_window_seconds: options.successful_request_window_seconds,
                tier_cache: Arc::new(Mutex::new(HashMap::new())),
                rank_cache: Arc::new(Mutex::new(HashMap::new())),
            },
        })
    }

    /// Create a new request, hashing the token array for trace logging.
    ///
    /// This is the backward-compatible constructor: it records only the plain
    /// (unlabeled) metric series and returns `RequestMetricsRequest` directly
    /// (no `Result`) because the unlabeled path cannot fail.
    pub fn new_request(
        &self,
        input_token_ids: &[u32],
        features: RequestFeatures,
        provided_session_id: Option<&str>,
    ) -> RequestMetricsRequest {
        self.new_request_labeled(
            Some(input_token_ids),
            input_token_ids.len(),
            features,
            provided_session_id,
            None,
            None,
            None,
        )
        .expect("unlabeled new_request cannot fail")
    }

    /// Create a new request with just the token count (no hashing, no token
    /// array needed).
    ///
    /// This is the backward-compatible constructor: it records only the plain
    /// (unlabeled) metric series and returns `RequestMetricsRequest` directly
    /// (no `Result`) because the unlabeled path cannot fail.
    pub fn new_request_with_count(
        &self,
        input_token_count: usize,
        features: RequestFeatures,
        provided_session_id: Option<&str>,
    ) -> RequestMetricsRequest {
        self.new_request_labeled(
            None,
            input_token_count,
            features,
            provided_session_id,
            None,
            None,
            None,
        )
        .expect("unlabeled new_request_with_count cannot fail")
    }

    /// Create a new request with optional extra labels.
    ///
    /// `input_token_ids` controls the trace path: `Some(ids)` hashes the token
    /// array (full path, needed when request logging is enabled); `None` skips
    /// hashing and uses only `input_token_count` (fast path, when logging is
    /// disabled — avoids copying the token array).
    ///
    /// In addition to the plain series, this records the major metrics into
    /// labeled series when `service_tier` and/or `dp_rank` are `Some`.
    /// `service_tier` and `dp_rank` are kept independent: a request carrying
    /// both produces a tier-only series set and a rank-only series set, never
    /// a combined cross-product. The label scheme matches each metric's plain
    /// counterpart:
    ///
    /// - Distribution metrics (`ttft_ms`, `time_to_fifth_token_ms`, `itl_ms`)
    ///   carry `feature="all"` plus `service_tier=<tier>` or `dp_rank=<rank>`.
    /// - Rate/ratio metrics (`request`, `successful_request`,
    ///   `kv_cache_hit_rate`) carry only the extra dimension (no `feature`
    ///   label, matching their plain series).
    ///
    /// When both `service_tier` and `dp_rank` are `None` no labeled series are
    /// produced (plain path only). Ranks above `DP_RANK_MAX` (31) are silently
    /// normalized to `None`.
    ///
    /// `b10_request_id` is hashed with [`hash_b10_request_id`] before it is
    /// stored; the raw value is never kept. An empty id is stored as null.
    ///
    /// Returns `Err` when the registry cardinality cap is exceeded or a metric
    /// kind mismatch is detected.
    #[allow(clippy::too_many_arguments)]
    pub fn new_request_labeled(
        &self,
        input_token_ids: Option<&[u32]>,
        input_token_count: usize,
        features: RequestFeatures,
        provided_session_id: Option<&str>,
        b10_request_id: Option<&str>,
        service_tier: Option<ServiceTier>,
        dp_rank: Option<u8>,
    ) -> anyhow::Result<RequestMetricsRequest> {
        // When the token slice is provided, derive the count from it to avoid
        // a mismatch between the two arguments.
        let input_token_count = match input_token_ids {
            Some(ids) => ids.len(),
            None => input_token_count,
        };
        self.new_request_internal(
            input_token_count,
            features,
            provided_session_id,
            b10_request_id,
            service_tier,
            dp_rank,
            |count, sid, b10_id, feats, tier| {
                let mut trace = self
                    .trace_factory
                    .create_trace_base(count, sid, b10_id, feats, tier);
                // Skip hashing on the count-only path.
                if let Some(ids) = input_token_ids {
                    trace.append_tokens(ids);
                }
                trace
            },
        )
    }

    /// Common implementation for both request creation methods.
    #[allow(clippy::too_many_arguments)]
    fn new_request_internal<F>(
        &self,
        input_token_count: usize,
        features: RequestFeatures,
        provided_session_id: Option<&str>,
        b10_request_id: Option<&str>,
        service_tier: Option<ServiceTier>,
        dp_rank: Option<u8>,
        create_trace: F,
    ) -> anyhow::Result<RequestMetricsRequest>
    where
        F: FnOnce(
            usize,
            Option<&str>,
            Option<&str>,
            Vec<String>,
            Option<ServiceTier>,
        ) -> RequestTrace,
    {
        // Resolve labeled handles first so any registry error (cardinality cap,
        // kind mismatch) fails before side effects (counter increments,
        // inflight tracking, etc.). dp_rank out of range is normalized to None
        // inside resolve(), not an error.
        let tiered = self
            .tiered_metric_config
            .resolve(&self.metrics, service_tier, dp_rank)?;
        for handles in tiered.iter().flatten() {
            let _ = handles.request.record_count(1);
        }

        let _ = self.request.record_count(1);
        let _ = self
            .input_length_tokens
            .record_value(input_token_count as f64);
        self.totals.total_requests.fetch_add(1, Ordering::Relaxed);
        self.totals
            .total_input_tokens
            .fetch_add(input_token_count as u64, Ordering::Relaxed);
        self.totals.inflight.on_start(input_token_count as u64);
        let (
            ttft_feature_metrics,
            time_to_fifth_token_feature_metrics,
            first_to_fifth_token_itl_feature_metrics,
            itl_feature_metrics,
        ) = self.feature_metric_handles(features);
        let features_vec = features.to_vec();

        Ok(RequestMetricsRequest {
            trace: create_trace(
                input_token_count,
                provided_session_id,
                b10_request_id,
                features_vec,
                service_tier,
            ),
            started_at: Instant::now(),
            input_tokens: input_token_count as u64,
            features,
            last_token_at: None,
            last_total_tokens: 0,
            first_token_at: None,
            time_to_fifth_token_recorded: false,
            first_to_fifth_inflight_active: false,
            cached_tokens_hint: None,
            rng: SmallRng::seed_from_u64(unix_ms(SystemTime::now())),
            input_tokens_rate: self.input_tokens.clone(),
            net_new_input_tokens_rate: self.net_new_input_tokens.clone(),
            output_tokens_rate: self.output_tokens.clone(),
            output_length_tokens: self.output_length_tokens.clone(),
            ttft_ms: self.ttft_ms.clone(),
            time_to_fifth_token_ms: self.time_to_fifth_token_ms.clone(),
            first_to_fifth_token_itl_ms: self.first_to_fifth_token_itl_ms.clone(),
            ttft_ms_per_input_token: self.ttft_ms_per_input_token.clone(),
            ttft_ms_per_net_new_input_token: self.ttft_ms_per_net_new_input_token.clone(),
            itl_ms: self.itl_ms.clone(),
            pre_first_token_cancellation: self.pre_first_token_cancellation.clone(),
            pre_20_token_cancellation: self.pre_20_token_cancellation.clone(),
            mid_stream_cancellation: self.mid_stream_cancellation.clone(),
            successful_request: self.successful_request.clone(),
            kv_cache_hit_rate: self.kv_cache_hit_rate.clone(),
            itl_sample_rate: self.itl_sample_rate,
            totals: Arc::clone(&self.totals),
            inflight_finished: false,
            ttft_feature_metrics,
            time_to_fifth_token_feature_metrics,
            first_to_fifth_token_itl_feature_metrics,
            itl_feature_metrics,
            tiered,
        })
    }

    fn feature_metric_handles(
        &self,
        features: RequestFeatures,
    ) -> (
        Vec<DistributionMetricHandle>,
        Vec<DistributionMetricHandle>,
        Vec<DistributionMetricHandle>,
        Vec<DistributionMetricHandle>,
    ) {
        let handles = self.feature_metrics.get_or_init(|| {
            RequestFeatureMetricHandles::new(&self.metrics, &self.feature_metric_config)
                .expect("failed to register request feature metrics")
        });
        handles.record_request(features);
        handles.for_features(features)
    }

    /// See [`RequestTraceFactory::configure_request_log`]; metrics are unaffected.
    pub fn configure_request_log(&self, options: &RequestTraceOptions) -> anyhow::Result<()> {
        self.trace_factory.configure_request_log(options)
    }

    pub fn request_log_enabled(&self) -> bool {
        self.trace_factory.request_log_enabled()
    }

    pub fn snapshot(&self) -> RequestMetricsSnapshot {
        let mut totals = self.totals.snapshot();
        let c = &self.trace_factory.recorder.counters;
        totals.total_request_log_rows_persisted = c.rows_persisted.load(Ordering::Relaxed);
        totals.total_request_log_rows_dropped_queue_full =
            c.rows_dropped_queue_full.load(Ordering::Relaxed);
        totals.total_request_log_rows_dropped_flush_error =
            c.rows_dropped_flush_error.load(Ordering::Relaxed);
        RequestMetricsSnapshot {
            series: self.metrics.snapshot_all_series(),
            totals,
        }
    }
}

impl RequestMetricsRequest {
    /// Record output tokens as they are generated, along with optional metadata like cached tokens and speculation ratio.
    /// This method must be lock-free, not-blocking and complete within ~20us even under large tokens, to avoid blocking the llm response generation.
    pub fn record_tokens(
        &mut self,
        token_ids: &[u32],
        cached_tokens: Option<u64>,
        is_diff: bool,
        speculation_ratio: Option<f64>,
    ) {
        if let Some(cached) = cached_tokens {
            self.cached_tokens_hint.get_or_insert(cached);
            self.trace.set_cached_tokens(cached);
        }
        if let Some(ratio) = speculation_ratio {
            self.trace.set_speculation_ratio(ratio);
        }

        let appended = self
            .trace
            .record_output_tokens(token_ids, is_diff, speculation_ratio);
        if appended == 0 {
            return;
        }
        let _ = self.output_tokens_rate.record_count(appended as u64);
        self.totals
            .total_output_tokens
            .fetch_add(appended as u64, Ordering::Relaxed);
        self.totals.inflight.add_output_tokens(appended as u64);

        let total_tokens = self.trace.output_tokens() as u64;
        if total_tokens <= self.last_total_tokens {
            return;
        }
        let now = Instant::now();

        if self.first_token_at.is_none() && self.last_total_tokens < 1 && total_tokens >= 1 {
            self.first_token_at = Some(now);
            self.start_first_to_fifth_inflight();
        }

        if !self.time_to_fifth_token_recorded && self.last_total_tokens < 5 && total_tokens >= 5 {
            let time_to_fifth_token_ms = now.duration_since(self.started_at).as_secs_f64() * 1000.0;
            let _ = self
                .time_to_fifth_token_ms
                .record_value(time_to_fifth_token_ms);
            for metric in &self.time_to_fifth_token_feature_metrics {
                let _ = metric.record_value(time_to_fifth_token_ms);
            }
            for tiered in self.tiered.iter().flatten() {
                let _ = tiered
                    .time_to_fifth_token_ms
                    .record_value(time_to_fifth_token_ms);
            }
            if let Some(first_token_at) = self.first_token_at {
                let first_to_fifth_token_itl_ms =
                    now.duration_since(first_token_at).as_secs_f64() * 1000.0;
                let _ = self
                    .first_to_fifth_token_itl_ms
                    .record_value(first_to_fifth_token_itl_ms);
                for metric in &self.first_to_fifth_token_itl_feature_metrics {
                    let _ = metric.record_value(first_to_fifth_token_itl_ms);
                }
            }
            self.time_to_fifth_token_recorded = true;
            self.finish_first_to_fifth_inflight();
        }

        if self.last_token_at.is_none() {
            if self.input_tokens > 0 {
                let _ = self.input_tokens_rate.record_count(self.input_tokens);
            }

            let ttft_ms = now.duration_since(self.started_at).as_secs_f64() * 1000.0;
            let _ = self.ttft_ms.record_value(ttft_ms);
            for metric in &self.ttft_feature_metrics {
                let _ = metric.record_value(ttft_ms);
            }
            for tiered in self.tiered.iter().flatten() {
                let _ = tiered.ttft_ms.record_value(ttft_ms);
            }
            if self.input_tokens > 0 {
                let _ = self
                    .ttft_ms_per_input_token
                    .record_value(ttft_ms / self.input_tokens as f64);
            }
            if let Some(cached) = self.cached_tokens_hint.take() {
                let total = self.input_tokens as f64;
                if total > 0.0 {
                    let hits = (cached as f64).min(total);
                    let _ = self.kv_cache_hit_rate.record_ratio(hits, total);
                    for tiered in self.tiered.iter().flatten() {
                        let _ = tiered.kv_cache_hit_rate.record_ratio(hits, total);
                    }
                    self.totals
                        .total_kv_cache_hits_tokens
                        .fetch_add(hits as u64, Ordering::Relaxed);
                    self.totals
                        .total_kv_cache_lookup_tokens
                        .fetch_add(total as u64, Ordering::Relaxed);
                }
                let net_new = self.input_tokens.saturating_sub(cached);
                if net_new > 0 {
                    let _ = self.net_new_input_tokens_rate.record_count(net_new);
                    self.totals
                        .total_net_new_input_tokens
                        .fetch_add(net_new, Ordering::Relaxed);
                    let _ = self
                        .ttft_ms_per_net_new_input_token
                        .record_value(ttft_ms / net_new as f64);
                }
            }
            self.last_token_at = Some(now);
            self.last_total_tokens = total_tokens;
            return;
        }

        let delta_tokens = total_tokens.saturating_sub(self.last_total_tokens);
        if delta_tokens > 0 {
            let elapsed_ms = now
                .duration_since(self.last_token_at.unwrap_or(now))
                .as_secs_f64()
                * 1000.0;
            if elapsed_ms > 0.0 {
                let itl_ms = elapsed_ms / delta_tokens as f64;
                let samples = sampled_count(delta_tokens, self.itl_sample_rate, &mut self.rng);
                for _ in 0..samples {
                    let _ = self.itl_ms.record_value(itl_ms);
                    for metric in &self.itl_feature_metrics {
                        let _ = metric.record_value(itl_ms);
                    }
                    for tiered in self.tiered.iter().flatten() {
                        let _ = tiered.itl_ms.record_value(itl_ms);
                    }
                }
                self.totals
                    .total_itl_samples_recorded
                    .fetch_add(samples, Ordering::Relaxed);
            }
        }
        self.last_token_at = Some(now);
        self.last_total_tokens = total_tokens;
    }

    /// Account for a terminal usage total when token events were not observable.
    /// No token IDs, latency samples, or output steps can be inferred from usage.
    pub fn reconcile_output_usage(&mut self, total: usize, cached_tokens: Option<u64>) {
        if self.trace.is_terminal() {
            return;
        }
        if let Some(cached) = cached_tokens {
            self.trace.set_cached_tokens(cached);
        }
        let delta = total.saturating_sub(self.trace.output_tokens_len) as u64;
        if delta == 0 {
            return;
        }
        self.trace.output_tokens_len = total;
        let _ = self.output_tokens_rate.record_count(delta);
        self.totals
            .total_output_tokens
            .fetch_add(delta, Ordering::Relaxed);
        self.totals.inflight.add_output_tokens(delta);
    }

    pub fn success(&mut self) {
        if self.trace.is_terminal() {
            return;
        }
        self.finish_inflight();
        let _ = self
            .output_length_tokens
            .record_value(self.trace.output_tokens() as f64);
        let _ = self.successful_request.record_count(1);
        for tiered in self.tiered.iter().flatten() {
            let _ = tiered.successful_request.record_count(1);
        }
        self.totals
            .total_successful_requests
            .fetch_add(1, Ordering::Relaxed);
        self.trace.success();
    }

    pub fn cancel(&mut self) {
        if self.trace.is_terminal() {
            return;
        }
        self.finish_inflight();
        let _ = self
            .output_length_tokens
            .record_value(self.trace.output_tokens() as f64);
        let output_tokens = self.trace.output_tokens();
        if output_tokens == 0 {
            let _ = self.pre_first_token_cancellation.record_count(1);
            self.totals
                .total_pre_first_token_cancellations
                .fetch_add(1, Ordering::Relaxed);
        } else if output_tokens < PRE_20_TOKEN_CANCELLATION_EXCLUSIVE_MAX_OUTPUT_TOKENS {
            let _ = self.pre_20_token_cancellation.record_count(1);
            self.totals
                .total_pre_20_token_cancellations
                .fetch_add(1, Ordering::Relaxed);
        } else {
            let _ = self.mid_stream_cancellation.record_count(1);
            self.totals
                .total_mid_stream_cancellations
                .fetch_add(1, Ordering::Relaxed);
        }
        self.trace.cancel();
    }

    /// Record the workers the router chose. Only the log row changes; metric
    /// labels were fixed when the request was created. Each call replaces all
    /// four values, so `None` fields are stored as null.
    pub fn set_routed_workers(&mut self, routed_workers: RoutedWorkers) {
        self.trace.set_routed_workers(routed_workers);
    }

    pub fn total_hashes(&self) -> &[HashBytes] {
        self.trace.total_hashes()
    }

    pub fn features(&self) -> RequestFeatures {
        self.features
    }

    fn finish_inflight(&mut self) {
        if self.inflight_finished {
            return;
        }
        self.finish_first_to_fifth_inflight();
        self.totals
            .inflight
            .on_finish(self.input_tokens, self.trace.output_tokens() as u64);
        self.inflight_finished = true;
    }

    fn start_first_to_fifth_inflight(&mut self) {
        if self.first_to_fifth_inflight_active {
            return;
        }
        self.totals
            .first_to_fifth_inflight
            .on_start(self.input_tokens);
        self.first_to_fifth_inflight_active = true;
    }

    fn finish_first_to_fifth_inflight(&mut self) {
        if !self.first_to_fifth_inflight_active {
            return;
        }
        self.totals
            .first_to_fifth_inflight
            .on_finish(self.input_tokens, 0);
        self.first_to_fifth_inflight_active = false;
    }
}

impl Drop for RequestMetricsRequest {
    fn drop(&mut self) {
        self.cancel();
    }
}

impl RequestFeatureMetricHandles {
    fn new(
        metrics: &PerformanceMetricsRegistry,
        config: &RequestFeatureMetricConfig,
    ) -> anyhow::Result<Self> {
        let request_all = metrics.new_rate_metric_with_labels(
            format!("{}_request", config.metric_prefix),
            feature_metric_labels("all"),
            config.request_quantiles.clone(),
            config.request_sample_period_seconds,
            config.request_window_seconds,
        )?;
        let request_none = metrics.new_rate_metric_with_labels(
            format!("{}_request", config.metric_prefix),
            feature_metric_labels("none"),
            config.request_quantiles.clone(),
            config.request_sample_period_seconds,
            config.request_window_seconds,
        )?;
        let mut request_by_feature = Vec::with_capacity(RequestFeatureKind::ALL.len());
        for feature in RequestFeatureKind::ALL {
            request_by_feature.push((
                feature,
                metrics.new_rate_metric_with_labels(
                    format!("{}_request", config.metric_prefix),
                    feature_metric_labels(feature.metric_suffix()),
                    config.request_quantiles.clone(),
                    config.request_sample_period_seconds,
                    config.request_window_seconds,
                )?,
            ));
        }

        let ttft_none = metrics.new_distribution_metric_with_labels(
            format!("{}_ttft_ms", config.metric_prefix),
            feature_metric_labels("none"),
            config.ttft_quantiles.clone(),
            config.ttft_window_seconds,
        )?;
        let mut ttft_by_feature = Vec::with_capacity(RequestFeatureKind::ALL.len());
        let time_to_fifth_token_none = metrics.new_distribution_metric_with_labels(
            format!("{}_time_to_fifth_token_ms", config.metric_prefix),
            feature_metric_labels("none"),
            config.time_to_fifth_token_quantiles.clone(),
            config.time_to_fifth_token_window_seconds,
        )?;
        let mut time_to_fifth_token_by_feature = Vec::with_capacity(RequestFeatureKind::ALL.len());
        let first_to_fifth_token_itl_none = metrics.new_distribution_metric_with_labels(
            format!("{}_first_to_fifth_token_itl_ms", config.metric_prefix),
            feature_metric_labels("none"),
            config.first_to_fifth_token_itl_quantiles.clone(),
            config.first_to_fifth_token_itl_window_seconds,
        )?;
        let mut first_to_fifth_token_itl_by_feature =
            Vec::with_capacity(RequestFeatureKind::ALL.len());
        let itl_none = metrics.new_distribution_metric_with_labels(
            format!("{}_itl_ms", config.metric_prefix),
            feature_metric_labels("none"),
            config.itl_quantiles.clone(),
            config.itl_window_seconds,
        )?;
        let mut itl_by_feature = Vec::with_capacity(RequestFeatureKind::ALL.len());

        for feature in RequestFeatureKind::ALL {
            ttft_by_feature.push((
                feature,
                metrics.new_distribution_metric_with_labels(
                    format!("{}_ttft_ms", config.metric_prefix),
                    feature_metric_labels(feature.metric_suffix()),
                    config.ttft_quantiles.clone(),
                    config.ttft_window_seconds,
                )?,
            ));
            time_to_fifth_token_by_feature.push((
                feature,
                metrics.new_distribution_metric_with_labels(
                    format!("{}_time_to_fifth_token_ms", config.metric_prefix),
                    feature_metric_labels(feature.metric_suffix()),
                    config.time_to_fifth_token_quantiles.clone(),
                    config.time_to_fifth_token_window_seconds,
                )?,
            ));
            first_to_fifth_token_itl_by_feature.push((
                feature,
                metrics.new_distribution_metric_with_labels(
                    format!("{}_first_to_fifth_token_itl_ms", config.metric_prefix),
                    feature_metric_labels(feature.metric_suffix()),
                    config.first_to_fifth_token_itl_quantiles.clone(),
                    config.first_to_fifth_token_itl_window_seconds,
                )?,
            ));
            itl_by_feature.push((
                feature,
                metrics.new_distribution_metric_with_labels(
                    format!("{}_itl_ms", config.metric_prefix),
                    feature_metric_labels(feature.metric_suffix()),
                    config.itl_quantiles.clone(),
                    config.itl_window_seconds,
                )?,
            ));
        }

        Ok(Self {
            request_all,
            request_none,
            request_by_feature,
            ttft_none,
            ttft_by_feature,
            time_to_fifth_token_none,
            time_to_fifth_token_by_feature,
            first_to_fifth_token_itl_none,
            first_to_fifth_token_itl_by_feature,
            itl_none,
            itl_by_feature,
        })
    }

    fn record_request(&self, features: RequestFeatures) {
        let _ = self.request_all.record_count(1);
        if features.is_empty() {
            let _ = self.request_none.record_count(1);
            return;
        }
        for (_, handle) in self
            .request_by_feature
            .iter()
            .filter(|(feature, _)| features.includes(*feature))
        {
            let _ = handle.record_count(1);
        }
    }

    fn for_features(
        &self,
        features: RequestFeatures,
    ) -> (
        Vec<DistributionMetricHandle>,
        Vec<DistributionMetricHandle>,
        Vec<DistributionMetricHandle>,
        Vec<DistributionMetricHandle>,
    ) {
        let ttft = if features.is_empty() {
            vec![self.ttft_none.clone()]
        } else {
            self.ttft_by_feature
                .iter()
                .filter(|(feature, _)| features.includes(*feature))
                .map(|(_, handle)| handle.clone())
                .collect()
        };
        let time_to_fifth_token = if features.is_empty() {
            vec![self.time_to_fifth_token_none.clone()]
        } else {
            self.time_to_fifth_token_by_feature
                .iter()
                .filter(|(feature, _)| features.includes(*feature))
                .map(|(_, handle)| handle.clone())
                .collect()
        };
        let first_to_fifth_token_itl = if features.is_empty() {
            vec![self.first_to_fifth_token_itl_none.clone()]
        } else {
            self.first_to_fifth_token_itl_by_feature
                .iter()
                .filter(|(feature, _)| features.includes(*feature))
                .map(|(_, handle)| handle.clone())
                .collect()
        };
        let itl = if features.is_empty() {
            vec![self.itl_none.clone()]
        } else {
            self.itl_by_feature
                .iter()
                .filter(|(feature, _)| features.includes(*feature))
                .map(|(_, handle)| handle.clone())
                .collect()
        };
        (ttft, time_to_fifth_token, first_to_fifth_token_itl, itl)
    }
}

fn feature_metric_labels(feature: &str) -> MetricLabels {
    let mut labels = MetricLabels::new();
    labels.insert("feature".to_string(), feature.to_string());
    labels
}

impl RequestTrace {
    pub fn record_output_tokens(
        &mut self,
        output_token_ids: &[u32],
        is_diff: bool,
        speculation_ratio: Option<f64>,
    ) -> usize {
        if self.terminal {
            return 0;
        }
        // Store the last speculation_ratio if provided
        if let Some(ratio) = speculation_ratio {
            self.speculation_ratio = Some(ratio);
        }
        let delta = if is_diff {
            output_token_ids
        } else {
            if output_token_ids.len() <= self.output_tokens_len {
                return 0;
            }
            &output_token_ids[self.output_tokens_len..]
        };
        // Track first token time if this is the first output
        if self.output_tokens_len == 0 && !delta.is_empty() {
            self.first_token_at = Some(SystemTime::now());
        }
        self.append_tokens(delta);
        self.output_tokens_len += delta.len();
        if !delta.is_empty() {
            self.output_steps += 1;
        }
        delta.len()
    }

    pub fn success(&mut self) {
        if self.terminal {
            return;
        }
        self.flush_full_block();
        self.emit(false);
        self.terminal = true;
    }

    pub fn cancel(&mut self) {
        if self.terminal {
            return;
        }
        self.flush_full_block();
        self.emit(true);
        self.terminal = true;
    }

    pub fn total_hashes(&self) -> &[HashBytes] {
        self.total_hashes.as_slice()
    }

    pub fn input_tokens(&self) -> usize {
        self.input_tokens
    }

    pub fn output_tokens(&self) -> usize {
        self.output_tokens_len
    }

    pub fn is_terminal(&self) -> bool {
        self.terminal
    }

    pub fn set_cached_tokens(&mut self, cached_tokens: u64) {
        self.cached_tokens = Some(cached_tokens);
    }

    pub fn set_speculation_ratio(&mut self, speculation_ratio: f64) {
        self.speculation_ratio = Some(speculation_ratio);
    }

    /// Ignored once the row has been written (after `success`, `cancel` or drop).
    pub fn set_routed_workers(&mut self, routed_workers: RoutedWorkers) {
        if self.terminal {
            return;
        }
        self.routed_workers = routed_workers;
    }

    fn append_tokens(&mut self, token_ids: &[u32]) {
        // Hashes only feed the request log; skip the work when this request has no worker.
        if self.worker.is_none() {
            return;
        }
        let mut remaining = token_ids;
        while !remaining.is_empty() {
            let needed = self.factory.block_size - self.pending.len();
            let take = needed.min(remaining.len());
            self.pending.extend_from_slice(&remaining[..take]);
            remaining = &remaining[take..];
            if self.pending.len() == self.factory.block_size {
                self.flush_full_block();
            }
        }
    }

    fn flush_full_block(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let digest = hash_token_block(
            self.pending.as_slice(),
            self.previous_hash.as_ref(),
            self.factory.hash_key.as_ref(),
        );
        self.previous_hash = Some(*digest.as_bytes());
        self.total_hashes.push(*digest.as_bytes());
        self.pending.clear();
    }

    fn emit(&self, canceled: bool) {
        if let Some(worker) = &self.worker {
            worker.record(self.build_log_entry(canceled));
        }
    }

    pub(crate) fn build_log_entry(&self, canceled: bool) -> RequestLogEntry {
        let now = SystemTime::now();
        let start_ms = unix_ms(self.started_at_wallclock);
        let now_ms = unix_ms(now);
        let duration_e2e_ms = now_ms.saturating_sub(start_ms);
        let duration_ttft_ms = self
            .first_token_at
            .map(|t| unix_ms(t).saturating_sub(start_ms))
            .unwrap_or(0);
        RequestLogEntry {
            timestamp_start_unix_ms: start_ms,
            duration_e2e_ms,
            duration_ttft_ms,
            input_tokens: self.input_tokens as u64,
            output_tokens: self.output_tokens_len as u64,
            total_hashes: self.total_hashes.clone(),
            provided_session_id: self.provided_session_id.clone(),
            request_canceled: canceled,
            cached_tokens_reference: self.cached_tokens,
            model_name: self.model_name.clone(),
            org_id: self.org_id.clone(),
            component_name: self.factory.component_name.clone(),
            component_id: Some(self.factory.component_id.clone()),
            block_size: self.block_size,
            features: self.features.clone(),
            speculation_ratio: self.speculation_ratio,
            mean_output_tokens_per_step: (self.output_steps > 0)
                .then(|| self.output_tokens_len as f64 / self.output_steps as f64),
            service_tier: self.service_tier.map(|t| t.label_value().to_string()),
            b10_request_id: self.b10_request_id.clone(),
            prefill_worker_id: self.routed_workers.prefill_worker_id,
            decode_worker_id: self.routed_workers.decode_worker_id,
            prefill_dp_rank: self.routed_workers.prefill_dp_rank,
            decode_dp_rank: self.routed_workers.decode_dp_rank,
            __version__: DEFAULT_REQUEST_LOG_VERSION.to_string(),
        }
    }
}

impl RequestMetricsTotals {
    fn snapshot(&self) -> RequestMetricsTotalsSnapshot {
        let total_requests = self.total_requests.load(Ordering::Relaxed);
        let total_successful_requests = self.total_successful_requests.load(Ordering::Relaxed);
        let total_pre_first_token_cancellations = self
            .total_pre_first_token_cancellations
            .load(Ordering::Relaxed);
        let total_pre_20_token_cancellations = self
            .total_pre_20_token_cancellations
            .load(Ordering::Relaxed);
        let total_mid_stream_cancellations =
            self.total_mid_stream_cancellations.load(Ordering::Relaxed);
        let total_input_tokens = self.total_input_tokens.load(Ordering::Relaxed);
        let total_output_tokens = self.total_output_tokens.load(Ordering::Relaxed);
        let total_net_new_input_tokens = self.total_net_new_input_tokens.load(Ordering::Relaxed);
        let total_kv_cache_hits_tokens = self.total_kv_cache_hits_tokens.load(Ordering::Relaxed);
        let total_kv_cache_lookup_tokens =
            self.total_kv_cache_lookup_tokens.load(Ordering::Relaxed);
        let total_itl_samples_recorded = self.total_itl_samples_recorded.load(Ordering::Relaxed);
        let kv_cache_hit_rate_lifetime = if total_kv_cache_lookup_tokens > 0 {
            Some(total_kv_cache_hits_tokens as f64 / total_kv_cache_lookup_tokens as f64)
        } else {
            None
        };
        let inflight = self.inflight.snapshot();
        let first_to_fifth_inflight = self.first_to_fifth_inflight.snapshot();
        RequestMetricsTotalsSnapshot {
            total_requests,
            total_successful_requests,
            total_pre_first_token_cancellations,
            total_pre_20_token_cancellations,
            total_mid_stream_cancellations,
            total_input_tokens,
            total_output_tokens,
            total_net_new_input_tokens,
            total_kv_cache_hits_tokens,
            total_kv_cache_lookup_tokens,
            total_itl_samples_recorded,
            total_request_log_rows_persisted: 0,
            total_request_log_rows_dropped_queue_full: 0,
            total_request_log_rows_dropped_flush_error: 0,
            kv_cache_hit_rate_lifetime,
            current_inflight_requests: inflight.current_requests,
            mean_inflight_requests: inflight.mean_requests,
            current_inflight_input_tokens: inflight.current_input_tokens,
            mean_inflight_input_tokens: inflight.mean_input_tokens,
            current_inflight_output_tokens: inflight.current_output_tokens,
            mean_inflight_output_tokens: inflight.mean_output_tokens,
            current_inflight_first_to_fifth_token_requests: first_to_fifth_inflight
                .current_requests,
            mean_inflight_first_to_fifth_token_requests: first_to_fifth_inflight.mean_requests,
            current_inflight_first_to_fifth_token_input_tokens: first_to_fifth_inflight
                .current_input_tokens,
            mean_inflight_first_to_fifth_token_input_tokens: first_to_fifth_inflight
                .mean_input_tokens,
        }
    }
}

impl Drop for RequestTrace {
    fn drop(&mut self) {
        if !self.terminal {
            self.cancel();
        }
    }
}

/// The factory's request-log handle, created with the factory and kept for its
/// lifetime. With logging off it has no worker: no thread, queue or sink.
struct EventRecorder {
    state: ArcSwap<RecorderState>,
    // Live as long as the factory, so request-log totals keep counting across workers.
    counters: Arc<RecorderCounters>,
    model_name_hash: String,
}

struct RecorderState {
    config: Option<RequestLogConfig>,
    worker: Option<Arc<RecorderWorker>>,
}

#[derive(Clone, PartialEq, Eq)]
struct RequestLogConfig {
    request_log_enabled: bool,
    request_log_batch_size: usize,
    request_log_queue_capacity: usize,
    local_output_dir: Option<PathBuf>,
    s3_config: Option<S3SinkConfig>,
    file_format: FileFormat,
}

impl RequestLogConfig {
    fn from_options(options: &RequestTraceOptions) -> Self {
        Self {
            request_log_enabled: options.request_log_enabled,
            request_log_batch_size: options.request_log_batch_size,
            request_log_queue_capacity: options.request_log_queue_capacity,
            local_output_dir: options.local_output_dir.clone(),
            s3_config: options.s3_config.clone(),
            file_format: options.file_format,
        }
    }
}

/// The sending end of a request-log worker thread. The last reference closes
/// the queue and waits for the worker to drain and flush what is queued.
#[derive(Debug)]
struct RecorderWorker {
    tx: Option<mpsc::Sender<Box<RequestLogEntry>>>,
    handle: Option<std::thread::JoinHandle<()>>,
    counters: Arc<RecorderCounters>,
}

#[derive(Debug, Default)]
struct RecorderCounters {
    rows_persisted: AtomicU64,
    rows_dropped_queue_full: AtomicU64,
    rows_dropped_flush_error: AtomicU64,
}

#[derive(Debug)]
struct EventRecorderSink {
    local_output_dir: Option<PathBuf>,
    s3: Option<S3Writer>,
    file_format: FileFormat,
}

#[derive(Debug)]
struct S3Writer {
    store: Arc<dyn ObjectStore>,
    prefix: String,
}

/// Time-based flush configuration - flush every 15 minutes if there's at least 1 entry
const TIME_BASED_FLUSH_INTERVAL: std::time::Duration = std::time::Duration::from_secs(15 * 60); // 15 minutes
const RECORDER_SHUTDOWN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

#[derive(Debug)]
struct EventRecorderInner {
    entries: Vec<RequestLogEntry>,
    sink: EventRecorderSink,
    batch_size: usize,
    model_name_hash: String,
    last_flush_time: std::time::Instant,
    counters: Arc<RecorderCounters>,
}

impl EventRecorder {
    fn new(model_name_hash: String) -> Self {
        Self {
            state: ArcSwap::from_pointee(RecorderState {
                config: None,
                worker: None,
            }),
            counters: Arc::new(RecorderCounters::default()),
            model_name_hash,
        }
    }

    fn is_enabled(&self) -> bool {
        self.state.load().worker.is_some()
    }

    fn current_worker(&self) -> Option<Arc<RecorderWorker>> {
        self.state.load().worker.clone()
    }

    /// Start, stop, or replace the worker from `options`' request-log fields.
    /// The new worker is built first, so on error the current one keeps running.
    /// Requests keep the worker they captured when they started.
    fn configure(&self, options: &RequestTraceOptions) -> anyhow::Result<()> {
        let desired = RequestLogConfig::from_options(options);
        loop {
            let current = self.state.load_full();
            if current.config.as_ref() == Some(&desired)
                && current
                    .worker
                    .as_ref()
                    .is_none_or(|worker| worker.tx.as_ref().is_some_and(|tx| !tx.is_closed()))
            {
                return Ok(());
            }
            let worker = if desired.request_log_enabled {
                let sink = build_event_recorder_sink(
                    desired.local_output_dir.clone(),
                    desired.s3_config.clone(),
                    desired.file_format,
                )?;
                Some(Arc::new(RecorderWorker::new(
                    sink,
                    desired.request_log_batch_size,
                    desired.request_log_queue_capacity,
                    self.model_name_hash.clone(),
                    Arc::clone(&self.counters),
                )))
            } else {
                None
            };
            let next = Arc::new(RecorderState {
                config: Some(desired.clone()),
                worker,
            });
            let previous = self.state.compare_and_swap(&current, next);
            if Arc::ptr_eq(&current, &previous) {
                return Ok(());
            }
        }
    }
}

impl RecorderWorker {
    fn new(
        sink: EventRecorderSink,
        batch_size: usize,
        queue_capacity: usize,
        model_name_hash: String,
        counters: Arc<RecorderCounters>,
    ) -> Self {
        let (tx, mut rx) = mpsc::channel::<Box<RequestLogEntry>>(queue_capacity.max(1));
        let worker_counters = Arc::clone(&counters);
        let handle = std::thread::spawn(move || {
            let rt = RuntimeBuilder::new_current_thread()
                .enable_all()
                .build()
                .expect("failed to build recorder runtime");
            rt.block_on(async move {
                let mut inner = EventRecorderInner {
                    entries: Vec::new(),
                    sink,
                    batch_size: batch_size.max(1),
                    model_name_hash: model_name_hash.chars().take(8).collect(),
                    last_flush_time: std::time::Instant::now(),
                    counters: worker_counters,
                };

                loop {
                    // Calculate next flush deadline based on last flush time
                    let next_flush_deadline = tokio::time::Instant::from_std(
                        inner.last_flush_time + TIME_BASED_FLUSH_INTERVAL,
                    );

                    // Use tokio::select! to wait for either a message or timeout
                    let msg = tokio::select! {
                        msg = rx.recv() => msg,
                        _ = tokio::time::sleep_until(next_flush_deadline) => {
                            // Always advance the timer cursor, even when idle, so we don't spin on a past deadline.
                            inner.last_flush_time = std::time::Instant::now();

                            // Timeout: flush if we have entries.
                            if !inner.entries.is_empty() {
                                println!("performance-metrics: time-based flush triggered, {} events since {} minutes ago.", inner.entries.len(), TIME_BASED_FLUSH_INTERVAL.as_secs() / 60);
                                if let Err(err) = inner.flush_next_batch().await {
                                    eprintln!(
                                        "performance-metrics: dropping request-log batch after time-based flush failure: {err}"
                                    );
                                }
                            }
                            continue;
                        }
                    };

                    // None: every sender is gone and the queue is drained.
                    let Some(entry) = msg else { break };
                    inner.entries.push(*entry);

                    // Size-based flush
                    if inner.entries.len() >= inner.batch_size
                        && let Err(err) = inner.flush_next_batch().await {
                            eprintln!(
                                "performance-metrics: dropping request-log batch after sink failure: {err}"
                            );
                        }
                }
                while !inner.entries.is_empty() {
                    if let Err(err) = inner.flush_next_batch().await {
                        eprintln!(
                            "performance-metrics: dropping request-log batch after sink failure: {err}"
                        );
                    }
                }
            });
        });

        Self {
            tx: Some(tx),
            handle: Some(handle),
            counters,
        }
    }

    fn record(&self, entry: RequestLogEntry) {
        // Non-blocking: try to send, drop entry if channel is full or closed
        // This ensures telemetry never blocks the main request path
        let Some(tx) = self.tx.as_ref() else {
            return;
        };
        match tx.try_send(Box::new(entry)) {
            Ok(()) => {}
            Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
                // Channel full, drop entry to avoid blocking main thread
                self.counters
                    .rows_dropped_queue_full
                    .fetch_add(1, Ordering::Relaxed);
                eprintln!("performance-metrics: dropping request-log entry, channel full");
            }
            Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                // Channel closed (shutting down), drop entry
            }
        }
    }
}

impl Drop for RecorderWorker {
    fn drop(&mut self) {
        drop(self.tx.take());
        let Some(handle) = self.handle.take() else {
            return;
        };
        let started = Instant::now();
        while !handle.is_finished() {
            if started.elapsed() >= RECORDER_SHUTDOWN_TIMEOUT {
                eprintln!(
                    "performance-metrics: request-log worker flush timed out after {}s, abandoning remaining entries",
                    RECORDER_SHUTDOWN_TIMEOUT.as_secs()
                );
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let _ = handle.join();
    }
}

impl EventRecorderInner {
    async fn flush_next_batch(&mut self) -> anyhow::Result<()> {
        let chunk_len = self.batch_size.min(self.entries.len());
        if chunk_len == 0 {
            return Ok(());
        }

        // Update last flush time before flushing
        self.last_flush_time = std::time::Instant::now();

        let batch = self.entries.drain(..chunk_len).collect::<Vec<_>>();
        let result = self.write(&batch).await;
        let counter = if result.is_ok() {
            &self.counters.rows_persisted
        } else {
            &self.counters.rows_dropped_flush_error
        };
        counter.fetch_add(batch.len() as u64, Ordering::Relaxed);
        result
    }

    async fn write(&self, batch: &[RequestLogEntry]) -> anyhow::Result<()> {
        let (payload, extension) = match self.sink.file_format {
            FileFormat::Json => {
                let payload = serde_json::to_vec(&batch)?;
                (payload, "json")
            }
            FileFormat::Parquet => {
                let payload = serialize_to_parquet(batch)?;
                (payload, "parquet")
            }
        };

        // Generate random 4-character hex suffix for distributed uniqueness
        let random_suffix: String = (0..4)
            .map(|_| format!("{:02x}", rand::random::<u8>()))
            .collect();
        let filename = format!(
            "{}-{}-{}.{}",
            self.model_name_hash,
            unix_ms(SystemTime::now()),
            random_suffix,
            extension
        );

        self.sink
            .write_batch(filename.as_str(), payload.as_slice())
            .await
    }
}

pub(crate) fn serialize_to_parquet(entries: &[RequestLogEntry]) -> anyhow::Result<Vec<u8>> {
    use arrow::array::{
        FixedSizeBinaryArray, Float64Array, ListArray, StringArray, UInt8Array, UInt32Array,
        UInt64Array,
    };
    use arrow::buffer::{Buffer, OffsetBuffer};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use parquet::basic::Compression;
    use parquet::file::properties::WriterProperties;
    use std::sync::Arc;

    let schema = Schema::new(vec![
        Field::new("timestamp_start_unix_ms", DataType::UInt64, false),
        Field::new("duration_e2e_ms", DataType::UInt64, false),
        Field::new("duration_ttft_ms", DataType::UInt64, false),
        Field::new("input_tokens", DataType::UInt64, false),
        Field::new("output_tokens", DataType::UInt64, false),
        Field::new(
            "total_hashes",
            DataType::List(Arc::new(Field::new(
                "item",
                DataType::FixedSizeBinary(32),
                false,
            ))),
            true,
        ),
        Field::new("provided_session_id", DataType::Utf8, true),
        Field::new("request_canceled", DataType::UInt8, false),
        Field::new("cached_tokens_reference", DataType::UInt64, true),
        Field::new("model_name", DataType::Utf8, false),
        Field::new("org_id", DataType::Utf8, false),
        Field::new("component_name", DataType::Utf8, true),
        Field::new("component_id", DataType::Utf8, true),
        Field::new("block_size", DataType::UInt64, false),
        Field::new("features", DataType::Utf8, true),
        Field::new("speculation_ratio", DataType::Float64, true),
        Field::new("mean_output_tokens_per_step", DataType::Float64, true),
        Field::new("service_tier", DataType::Utf8, true),
        Field::new("b10_request_id", DataType::Utf8, true),
        Field::new("prefill_worker_id", DataType::UInt64, true),
        Field::new("decode_worker_id", DataType::UInt64, true),
        Field::new("prefill_dp_rank", DataType::UInt32, true),
        Field::new("decode_dp_rank", DataType::UInt32, true),
        Field::new("__version__", DataType::Utf8, false),
    ]);

    let timestamp_start: Vec<u64> = entries.iter().map(|e| e.timestamp_start_unix_ms).collect();
    let duration_e2e: Vec<u64> = entries.iter().map(|e| e.duration_e2e_ms).collect();
    let duration_ttft: Vec<u64> = entries.iter().map(|e| e.duration_ttft_ms).collect();
    let input_tokens: Vec<u64> = entries.iter().map(|e| e.input_tokens).collect();
    let output_tokens: Vec<u64> = entries.iter().map(|e| e.output_tokens).collect();

    // Build ListArray for total_hashes (each entry is a list of FixedSizeBinary(HASH_SIZE))
    // Pre-allocate buffer: sum all hash lengths (each hash is HASH_SIZE bytes)
    let total_hashes_count: usize = entries.iter().map(|e| e.total_hashes.len()).sum();
    let mut total_hashes_values: Vec<u8> = Vec::with_capacity(total_hashes_count * HASH_SIZE);
    for entry in entries {
        for hash in &entry.total_hashes {
            // Hash is already raw bytes (32 bytes)
            total_hashes_values.extend_from_slice(hash);
        }
    }

    // Build offsets: [0, len1, len1+len2, len1+len2+len3, ...]
    // Pre-allocate: entries.len() + 1 offsets
    let mut total_hashes_offsets: Vec<i32> = Vec::with_capacity(entries.len() + 1);
    total_hashes_offsets.push(0);
    for entry in entries {
        let next_offset = total_hashes_offsets.last().unwrap() + entry.total_hashes.len() as i32;
        total_hashes_offsets.push(next_offset);
    }

    let total_hashes_offsets = OffsetBuffer::new(total_hashes_offsets.into());

    // Build FixedSizeBinaryArray from the raw bytes with an explicit width, so a
    // batch with no hashes at all still yields a FixedSizeBinary(HASH_SIZE) array
    let fixed_size_binary = FixedSizeBinaryArray::try_new(
        HASH_SIZE as i32,
        Buffer::from_vec(total_hashes_values),
        None,
    )?;

    let total_hashes = ListArray::new(
        Arc::new(Field::new(
            "item",
            DataType::FixedSizeBinary(HASH_SIZE as i32),
            false,
        )),
        total_hashes_offsets,
        Arc::new(fixed_size_binary),
        None,
    );
    let provided_session_id: Vec<Option<String>> = entries
        .iter()
        .map(|e| e.provided_session_id.clone())
        .collect();
    let request_canceled: Vec<u8> = entries
        .iter()
        .map(|e| if e.request_canceled { 1 } else { 0 })
        .collect();
    let cached_tokens_ref: Vec<Option<u64>> =
        entries.iter().map(|e| e.cached_tokens_reference).collect();
    let model_name: Vec<String> = entries.iter().map(|e| e.model_name.clone()).collect();
    let org_id: Vec<String> = entries.iter().map(|e| e.org_id.clone()).collect();
    let component_name: Vec<Option<String>> =
        entries.iter().map(|e| e.component_name.clone()).collect();
    let component_id: Vec<Option<String>> =
        entries.iter().map(|e| e.component_id.clone()).collect();
    let block_size: Vec<u64> = entries.iter().map(|e| e.block_size as u64).collect();
    let features: Vec<String> = entries.iter().map(|e| e.features.join(",")).collect();
    let speculation_ratio: Vec<Option<f64>> = entries.iter().map(|e| e.speculation_ratio).collect();
    let mean_output_tokens_per_step: Vec<Option<f64>> = entries
        .iter()
        .map(|e| e.mean_output_tokens_per_step)
        .collect();
    let service_tier: Vec<Option<String>> =
        entries.iter().map(|e| e.service_tier.clone()).collect();
    let b10_request_id: Vec<Option<String>> =
        entries.iter().map(|e| e.b10_request_id.clone()).collect();
    let prefill_worker_id: Vec<Option<u64>> = entries.iter().map(|e| e.prefill_worker_id).collect();
    let decode_worker_id: Vec<Option<u64>> = entries.iter().map(|e| e.decode_worker_id).collect();
    let prefill_dp_rank: Vec<Option<u32>> = entries.iter().map(|e| e.prefill_dp_rank).collect();
    let decode_dp_rank: Vec<Option<u32>> = entries.iter().map(|e| e.decode_dp_rank).collect();
    let version: Vec<String> = entries.iter().map(|e| e.__version__.clone()).collect();

    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(UInt64Array::from(timestamp_start)),
            Arc::new(UInt64Array::from(duration_e2e)),
            Arc::new(UInt64Array::from(duration_ttft)),
            Arc::new(UInt64Array::from(input_tokens)),
            Arc::new(UInt64Array::from(output_tokens)),
            Arc::new(total_hashes),
            Arc::new(StringArray::from(provided_session_id)),
            Arc::new(UInt8Array::from(request_canceled)),
            Arc::new(UInt64Array::from(cached_tokens_ref)),
            Arc::new(StringArray::from(model_name)),
            Arc::new(StringArray::from(org_id)),
            Arc::new(StringArray::from(component_name)),
            Arc::new(StringArray::from(component_id)),
            Arc::new(UInt64Array::from(block_size)),
            Arc::new(StringArray::from(features)),
            Arc::new(Float64Array::from(speculation_ratio)),
            Arc::new(Float64Array::from(mean_output_tokens_per_step)),
            Arc::new(StringArray::from(service_tier)),
            Arc::new(StringArray::from(b10_request_id)),
            Arc::new(UInt64Array::from(prefill_worker_id)),
            Arc::new(UInt64Array::from(decode_worker_id)),
            Arc::new(UInt32Array::from(prefill_dp_rank)),
            Arc::new(UInt32Array::from(decode_dp_rank)),
            Arc::new(StringArray::from(version)),
        ],
    )?;

    let mut buf = Vec::new();
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer = ArrowWriter::try_new(&mut buf, batch.schema(), Some(props))?;
    writer.write(&batch)?;
    writer.close()?;

    Ok(buf)
}

impl EventRecorderSink {
    async fn write_batch(&self, filename: &str, payload: &[u8]) -> anyhow::Result<()> {
        let mut attempted = 0usize;
        let mut succeeded = 0usize;
        let mut errors = Vec::new();

        if let Some(output_dir) = &self.local_output_dir {
            attempted += 1;
            let path = output_dir.join(filename);
            match fs::create_dir_all(output_dir.as_path()).and_then(|()| fs::write(path, payload)) {
                Ok(()) => {
                    succeeded += 1;
                }
                Err(err) => {
                    errors.push(format!("local sink error: {err}"));
                }
            }
        }

        if let Some(s3) = &self.s3 {
            attempted += 1;
            let key = if s3.prefix.is_empty() {
                filename.to_string()
            } else {
                format!("{}/{}", s3.prefix.trim_matches('/'), filename)
            };
            match upload_bytes_to_object_store(&s3.store, key.as_str(), payload).await {
                Ok(()) => {
                    succeeded += 1;
                }
                Err(err) => {
                    errors.push(format!("s3 sink error: {err}"));
                }
            }
        }

        if attempted == succeeded {
            return Ok(());
        }
        if succeeded > 0 {
            eprintln!(
                "performance-metrics: request-log batch only persisted to {succeeded}/{attempted} sinks: {}",
                errors.join("; ")
            );
            return Ok(());
        }
        anyhow::bail!("{}", errors.join("; "))
    }
}

// A path-valued key ID selects mounted credentials; secret keys may themselves start with `/`.
pub(crate) fn with_s3_credentials(
    mut builder: AmazonS3Builder,
    access_key_id: Option<&str>,
    secret_access_key: Option<&str>,
) -> AmazonS3Builder {
    if let Some(id) = access_key_id.filter(|id| id.starts_with('/')) {
        let provider = FileCredentialProvider {
            key_id_path: id.into(),
            secret_key_path: secret_access_key.unwrap_or_default().into(),
        };
        if let Err(e) = provider.read() {
            eprintln!("performance-metrics: warning: request-log uploads will fail: {e}");
        }
        return builder.with_credentials(Arc::new(provider));
    }
    if let Some(id) = access_key_id {
        builder = builder.with_access_key_id(id);
    }
    if let Some(secret) = secret_access_key {
        builder = builder.with_secret_access_key(secret);
    }
    builder
}

#[derive(Debug)]
struct FileCredentialProvider {
    key_id_path: PathBuf,
    secret_key_path: PathBuf,
}

impl FileCredentialProvider {
    fn read(&self) -> anyhow::Result<AwsCredential> {
        let read = |path: &PathBuf| -> anyhow::Result<String> {
            let value = fs::read_to_string(path)
                .map_err(|e| anyhow::anyhow!("credential file {}: {e}", path.display()))?;
            let value = value.trim();
            anyhow::ensure!(
                !value.is_empty(),
                "credential file {} is empty",
                path.display()
            );
            Ok(value.to_string())
        };
        Ok(AwsCredential {
            key_id: read(&self.key_id_path)?,
            secret_key: read(&self.secret_key_path)?,
            token: None,
        })
    }
}

#[async_trait::async_trait]
impl CredentialProvider for FileCredentialProvider {
    type Credential = AwsCredential;

    async fn get_credential(&self) -> object_store::Result<Arc<AwsCredential>> {
        self.read()
            .map(Arc::new)
            .map_err(|e| object_store::Error::Generic {
                store: "S3",
                source: e.into(),
            })
    }
}

fn build_event_recorder_sink(
    local_output_dir: Option<PathBuf>,
    s3_config: Option<S3SinkConfig>,
    file_format: FileFormat,
) -> anyhow::Result<EventRecorderSink> {
    let s3 = if let Some(config) = s3_config {
        let mut builder = AmazonS3Builder::new()
            .with_bucket_name(config.bucket)
            .with_region(config.region);
        if let Some(endpoint) = config.endpoint {
            builder = builder.with_endpoint(endpoint);
        }
        if let Some(proxy_url) = config.proxy_url {
            validate_s3_proxy_url(&proxy_url)?;
            builder = builder.with_proxy_url(proxy_url);
        }
        builder = with_s3_credentials(
            builder,
            config.access_key_id.as_deref(),
            config.secret_access_key.as_deref(),
        );
        let store = builder
            .build()
            .map_err(|e| anyhow::anyhow!("failed to build s3 store: {e}"))?;
        Some(S3Writer {
            store: Arc::new(store),
            prefix: config.prefix.trim_matches('/').to_string(),
        })
    } else {
        None
    };

    if local_output_dir.is_none() && s3.is_none() {
        anyhow::bail!("request_log_enabled requires local_output_dir and/or s3_config");
    }

    Ok(EventRecorderSink {
        local_output_dir,
        s3,
        file_format,
    })
}

async fn upload_bytes_to_object_store(
    store: &Arc<dyn ObjectStore>,
    key: &str,
    payload: &[u8],
) -> anyhow::Result<()> {
    let path = ObjectPath::from(key);
    store
        .put_opts(
            &path,
            PutPayload::from(Bytes::copy_from_slice(payload)),
            PutOptions::default(),
        )
        .await
        .map_err(|e| anyhow::anyhow!("failed to upload request-log chunk to object store: {e}"))?;
    Ok(())
}

fn unix_ms(ts: SystemTime) -> u64 {
    ts.duration_since(UNIX_EPOCH)
        .ok()
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn hash_token_block(
    token_ids: &[u32],
    previous_hash: Option<&[u8; 32]>,
    key: Option<&[u8; 32]>,
) -> blake3::Hash {
    let mut hasher = if let Some(key) = key {
        Hasher::new_keyed(key)
    } else {
        Hasher::new()
    };
    hasher.update(REQUEST_TRACE_HASH_DOMAIN);
    hasher.update(&[u8::from(previous_hash.is_some())]);
    if let Some(previous_hash) = previous_hash {
        hasher.update(previous_hash);
    }
    for token_id in token_ids {
        hasher.update(&token_id.to_le_bytes());
    }
    hasher.finalize()
}

fn hash_session_id(session_id: &str) -> String {
    let mut hasher = Hasher::new();
    hasher.update(REQUEST_SESSION_HASH_DOMAIN);
    hasher.update(session_id.as_bytes());
    hasher.finalize().to_hex().to_string()
}

/// Unkeyed, domain-separated blake3 digest (64 hex chars) of a Baseten request
/// id. Exposed so log joins can hash the id from other sources the same way.
pub fn hash_b10_request_id(b10_request_id: &str) -> String {
    let mut hasher = Hasher::new();
    hasher.update(B10_REQUEST_ID_HASH_DOMAIN);
    hasher.update(b10_request_id.as_bytes());
    hasher.finalize().to_hex().to_string()
}

fn sampled_count(tokens: u64, sample_rate: f64, rng: &mut SmallRng) -> u64 {
    if tokens == 0 || sample_rate <= 0.0 {
        return 0;
    }
    let expected = (tokens as f64 * sample_rate).min(50.0);
    let base = expected.floor() as u64;
    let frac = expected - base as f64;
    if frac > 0.0 && rng.random::<f64>() < frac {
        base + 1
    } else {
        base
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hashes are only computed for logged requests; log to a scratch dir.
    fn logged(options: RequestTraceOptions) -> RequestTraceOptions {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "pm-core-hash-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        RequestTraceOptions {
            request_log_enabled: true,
            local_output_dir: Some(dir),
            ..options
        }
    }
    use std::hint::black_box;
    use std::io::Read;
    use std::net::TcpListener;
    use std::sync::mpsc::channel;
    use std::thread;
    use std::time::{Duration, Instant};

    #[test]
    fn terminal_usage_preserves_counts_without_inventing_token_times() {
        let factory = RequestMetricsFactory::new(RequestMetricsOptions::default()).unwrap();
        let mut request = factory.new_request_with_count(8, RequestFeatures::NONE, None);
        request.reconcile_output_usage(3, Some(4));
        request.reconcile_output_usage(2, Some(4));
        assert_eq!(factory.snapshot().totals.total_output_tokens, 3);
        assert_eq!(factory.snapshot().totals.current_inflight_output_tokens, 3);
        assert!(request.first_token_at.is_none());
        assert!(request.trace.first_token_at.is_none());
        request.cancel();
        assert_eq!(factory.snapshot().totals.current_inflight_requests, 0);
        assert_eq!(
            factory.snapshot().totals.total_pre_20_token_cancellations,
            1
        );
    }

    #[test]
    fn request_trace_hashes_are_consecutive_for_cumulative_output() {
        let factory = RequestTraceFactory::new(logged(RequestTraceOptions::default())).unwrap();
        let mut trace = factory.new_trace(&[], None, vec![], None);
        // With DEFAULT_BLOCK_SIZE=64, we expect:
        // - 140 tokens / 64 = 2.18 -> 2 hashes (first call)
        // - 260 tokens / 64 = 4.06 -> 4 hashes (second call)
        // Total: 6 hashes
        trace.record_output_tokens(&(0..140).collect::<Vec<u32>>(), false, None);
        trace.record_output_tokens(&(0..260).collect::<Vec<u32>>(), false, None);
        trace.success();
        assert_eq!(trace.total_hashes.len(), 5);
    }

    #[test]
    fn request_log_mean_output_tokens_per_step_counts_nonempty_deltas() {
        let factory = RequestTraceFactory::new(RequestTraceOptions::default()).unwrap();
        let mut trace = factory.new_trace(&[], None, vec![], None);
        assert_eq!(
            trace.build_log_entry(false).mean_output_tokens_per_step,
            None
        );

        trace.record_output_tokens(&[1, 2, 3], false, None);
        trace.record_output_tokens(&[1, 2, 3], false, None);
        trace.record_output_tokens(&[1, 2, 3, 4, 5], false, None);
        trace.record_output_tokens(&[], true, None);
        trace.record_output_tokens(&[6], true, None);
        trace.success();

        let entry = trace.build_log_entry(false);
        assert_eq!(entry.output_tokens, 6);
        assert_eq!(entry.mean_output_tokens_per_step, Some(2.0));
        assert_eq!(
            serde_json::to_value(&entry).unwrap()["mean_output_tokens_per_step"],
            2.0
        );
    }

    #[test]
    fn request_trace_hash_salt_changes_hash() {
        let unsalted = RequestTraceFactory::new(logged(RequestTraceOptions::default())).unwrap();
        let salted = RequestTraceFactory::new(logged(RequestTraceOptions {
            request_log_hash_salt: Some("salt".to_string()),
            ..Default::default()
        }))
        .unwrap();

        let mut t1 = unsalted.new_trace(&[], None, vec![], None);
        let mut t2 = salted.new_trace(&[], None, vec![], None);
        t1.record_output_tokens(&[1, 2, 3, 4, 5], true, None);
        t2.record_output_tokens(&[1, 2, 3, 4, 5], true, None);
        t1.success();
        t2.success();
        assert_ne!(t1.total_hashes(), t2.total_hashes());
    }

    #[test]
    fn request_trace_same_ids_same_salt_same_hashes() {
        let factory = RequestTraceFactory::new(logged(RequestTraceOptions {
            request_log_hash_salt: Some("fixed-salt".to_string()),
            ..Default::default()
        }))
        .unwrap();

        let input_ids = vec![10_u32, 11, 12, 13];
        let output_ids = vec![101_u32, 102, 103, 104, 105];

        let mut a = factory.new_trace(input_ids.as_slice(), None, vec![], None);
        a.record_output_tokens(output_ids.as_slice(), true, None);
        a.success();

        let mut b = factory.new_trace(input_ids.as_slice(), None, vec![], None);
        b.record_output_tokens(output_ids.as_slice(), true, None);
        b.success();

        assert_eq!(a.total_hashes(), b.total_hashes());
    }

    #[test]
    fn request_trace_hashes_depend_on_previous_block_history() {
        let factory = RequestTraceFactory::new(logged(RequestTraceOptions {
            block_size: 2,
            ..Default::default()
        }))
        .unwrap();

        let mut a = factory.new_trace(&[1, 2], None, vec![], None);
        a.record_output_tokens(&[7, 8], true, None);
        a.success();

        let mut b = factory.new_trace(&[3, 4], None, vec![], None);
        b.record_output_tokens(&[7, 8], true, None);
        b.success();

        assert_eq!(a.total_hashes().len(), 2);
        assert_eq!(b.total_hashes().len(), 2);
        assert_ne!(a.total_hashes()[1], b.total_hashes()[1]);
    }

    #[test]
    fn request_trace_recorder_writes_batch() {
        let tmp = std::env::temp_dir().join(format!("pm-core-test-{}", unix_ms(SystemTime::now())));
        let factory = RequestTraceFactory::new(RequestTraceOptions {
            request_log_enabled: true,
            local_output_dir: Some(tmp.clone()),
            request_log_batch_size: 2,
            ..Default::default()
        })
        .unwrap();

        let mut a = factory.new_trace(&[], None, vec![], None);
        a.record_output_tokens(&[1], true, None);
        a.success();

        let mut b = factory.new_trace(&[], None, vec![], None);
        b.record_output_tokens(&[2], true, None);
        b.cancel();

        thread::sleep(std::time::Duration::from_millis(20));
        let file_count = fs::read_dir(tmp.as_path()).unwrap().count();
        assert!(file_count >= 1);
    }

    #[test]
    fn request_trace_diff_and_cumulative_modes_produce_same_hashes() {
        let factory = RequestTraceFactory::new(logged(RequestTraceOptions::default())).unwrap();

        let mut cumulative = factory.new_trace(&[], None, vec![], None);
        cumulative.record_output_tokens(&[1, 2, 3], false, None);
        cumulative.record_output_tokens(&[1, 2, 3, 4, 5], false, None);
        cumulative.record_output_tokens(&[1, 2, 3, 4, 5, 6], false, None);
        cumulative.success();

        let mut diff = factory.new_trace(&[], None, vec![], None);
        diff.record_output_tokens(&[1, 2, 3], true, None);
        diff.record_output_tokens(&[4, 5], true, None);
        diff.record_output_tokens(&[6], true, None);
        diff.success();

        assert_eq!(cumulative.total_hashes(), diff.total_hashes());
    }

    #[test]
    fn request_trace_drop_flushes_partial_batch() {
        let tmp =
            std::env::temp_dir().join(format!("pm-core-drop-test-{}", unix_ms(SystemTime::now())));
        let counters;
        {
            let factory = RequestTraceFactory::new(RequestTraceOptions {
                request_log_enabled: true,
                local_output_dir: Some(tmp.clone()),
                request_log_batch_size: 2,
                ..Default::default()
            })
            .unwrap();
            counters = Arc::clone(&factory.recorder.counters);
            for _ in 0..2 {
                factory.new_trace(&[1, 2, 3], None, vec![], None).success();
            }
            let mut trace = factory.new_trace(&[1, 2, 3], None, vec![], None);
            trace.record_output_tokens(&[4, 5], true, None);
            // no success/cancel, should flush on drop
        }

        wait_for(&counters.rows_persisted, 3);
        let file_count = fs::read_dir(tmp.as_path()).unwrap().count();
        assert!(file_count >= 1);
        assert_eq!(counters.rows_persisted.load(Ordering::Relaxed), 3);
        assert_eq!(counters.rows_dropped_flush_error.load(Ordering::Relaxed), 0);
        fs::remove_dir_all(tmp).unwrap();
    }

    fn local_log_config(dir: PathBuf) -> RequestTraceOptions {
        RequestTraceOptions {
            request_log_enabled: true,
            request_log_batch_size: 1,
            request_log_queue_capacity: 16,
            local_output_dir: Some(dir),
            ..Default::default()
        }
    }

    /// Workers flush on their own thread; wait for a counter to reach `n`.
    fn wait_for(counter: &AtomicU64, n: u64) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while counter.load(Ordering::Relaxed) < n {
            assert!(Instant::now() < deadline, "timed out waiting for {n}");
            thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn configure_request_log_toggles_logging_without_resetting_counters() {
        let tmp = std::env::temp_dir().join(format!(
            "pm-core-toggle-test-{}",
            unix_ms(SystemTime::now())
        ));
        let factory = RequestTraceFactory::new(RequestTraceOptions::default()).unwrap();
        let rows = &factory.recorder.counters.rows_persisted;
        let tokens: Vec<u32> = (0..130).collect();
        let trace = || factory.new_trace(&tokens, None, vec![], None);

        let mut off = trace();
        off.success();
        assert!(
            off.total_hashes().is_empty(),
            "inert requests must not hash"
        );
        let mut off_then_on = trace();

        factory
            .configure_request_log(&local_log_config(tmp.join("a")))
            .unwrap();
        assert!(factory.request_log_enabled());
        off_then_on.success(); // started unhashed: not written
        let mut bad = local_log_config(tmp.join("bad"));
        bad.s3_config = Some(S3SinkConfig {
            bucket: "b".to_string(),
            region: "us-east-1".to_string(),
            endpoint: None,
            proxy_url: Some("ftp://not-a-proxy".to_string()),
            access_key_id: None,
            secret_access_key: None,
            prefix: "p".to_string(),
        });
        assert!(factory.configure_request_log(&bad).is_err()); // keeps "a"
        let before = factory.recorder.current_worker().unwrap();
        factory
            .configure_request_log(&local_log_config(tmp.join("a")))
            .unwrap();
        let after = factory.recorder.current_worker().unwrap();
        assert!(
            Arc::ptr_eq(&before, &after),
            "unchanged settings must keep the current worker"
        );
        let mut on = trace();
        on.success();
        assert_eq!(on.total_hashes().len(), 3);
        wait_for(rows, 1);

        let mut on_then_off = trace();
        factory
            .configure_request_log(&RequestTraceOptions::default())
            .unwrap();
        assert!(!factory.request_log_enabled());
        on_then_off.success(); // started on: still written to the captured worker
        trace().success(); // started off: not written

        // A second worker keeps counting from where the first stopped.
        factory
            .configure_request_log(&local_log_config(tmp.join("b")))
            .unwrap();
        trace().success();
        wait_for(rows, 3);
        thread::sleep(Duration::from_millis(100));
        assert_eq!(rows.load(Ordering::Relaxed), 3);
        let _ = fs::remove_dir_all(tmp);
    }

    #[test]
    fn replacing_a_stuck_recorder_does_not_block_the_swap() {
        // A proxy that accepts connections and never answers keeps the old
        // worker's final flush stuck; the swap itself must not wait for it.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let proxy_url = format!("http://{}", listener.local_addr().unwrap());
        let _server = thread::spawn(move || {
            let _held: Vec<_> = listener.incoming().take(8).collect();
        });
        let factory = RequestTraceFactory::new(RequestTraceOptions::default()).unwrap();
        factory
            .configure_request_log(&RequestTraceOptions {
                request_log_enabled: true,
                request_log_batch_size: 1024,
                request_log_queue_capacity: 16,
                local_output_dir: None,
                s3_config: Some(S3SinkConfig {
                    bucket: "b".to_string(),
                    region: "us-east-1".to_string(),
                    endpoint: Some("https://s3.example.invalid".to_string()),
                    proxy_url: Some(proxy_url),
                    access_key_id: Some("x".to_string()),
                    secret_access_key: Some("x".to_string()),
                    prefix: "p".to_string(),
                }),
                ..Default::default()
            })
            .unwrap();
        factory.new_trace(&[1], None, vec![], None).success();
        let in_flight = factory.new_trace(&[1], None, vec![], None);

        let started = Instant::now();
        factory
            .configure_request_log(&RequestTraceOptions::default())
            .unwrap();
        assert!(
            started.elapsed() < Duration::from_millis(500),
            "swap waited on the old recorder: {:?}",
            started.elapsed()
        );
        drop(in_flight); // the last reference waits for its bounded flush
    }

    #[test]
    fn last_worker_reference_flushes_queued_rows() {
        let tmp = std::env::temp_dir().join(format!(
            "pm-core-last-worker-test-{}",
            unix_ms(SystemTime::now())
        ));
        let factory = RequestTraceFactory::new(RequestTraceOptions::default()).unwrap();
        let rows = &factory.recorder.counters.rows_persisted;
        factory
            .configure_request_log(&RequestTraceOptions {
                request_log_enabled: true,
                request_log_batch_size: 1024,
                request_log_queue_capacity: 16,
                local_output_dir: Some(tmp.clone()),
                ..Default::default()
            })
            .unwrap();
        let mut trace = factory.new_trace(&[1, 2, 3], None, vec![], None);
        trace.success();
        factory
            .configure_request_log(&RequestTraceOptions::default())
            .unwrap();
        assert_eq!(rows.load(Ordering::Relaxed), 0);
        drop(trace);
        assert_eq!(rows.load(Ordering::Relaxed), 1);
        let _ = fs::remove_dir_all(tmp);
    }

    #[test]
    fn file_credentials_fail_while_missing_and_follow_rotation() {
        let dir =
            std::env::temp_dir().join(format!("pm-core-creds-{}", unix_ms(SystemTime::now())));
        let provider = FileCredentialProvider {
            key_id_path: dir.join("access_key_id"),
            secret_key_path: dir.join("secret_access_key"),
        };
        let get = || futures::executor::block_on(provider.get_credential());
        let err = get().unwrap_err().to_string();
        assert!(err.contains("access_key_id"), "{err}");

        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("access_key_id"), "AKIA1\n").unwrap();
        fs::write(dir.join("secret_access_key"), "s/ecret1\n").unwrap();
        let c = get().unwrap();
        assert_eq!(
            (c.key_id.as_str(), c.secret_key.as_str()),
            ("AKIA1", "s/ecret1")
        );

        fs::write(dir.join("access_key_id"), "AKIA2").unwrap();
        assert_eq!(get().unwrap().key_id, "AKIA2");

        fs::write(dir.join("secret_access_key"), "").unwrap();
        assert!(get().unwrap_err().to_string().contains("empty"));
        fs::remove_dir_all(&dir).unwrap();
        assert!(get().is_err());
    }

    #[test]
    fn s3_sink_with_missing_credential_files_builds_and_fails_uploads_clearly() {
        let sink = build_event_recorder_sink(
            None,
            Some(S3SinkConfig {
                bucket: "example-bucket".to_string(),
                region: "us-east-1".to_string(),
                endpoint: Some("http://127.0.0.1:1".to_string()),
                proxy_url: None,
                access_key_id: Some("/nonexistent/request-recording/access_key_id".to_string()),
                secret_access_key: Some("/nonexistent/request-recording/secret".to_string()),
                prefix: "records".to_string(),
            }),
            FileFormat::Json,
        )
        .unwrap();
        let err = RuntimeBuilder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(sink.write_batch("test.json", b"{}"))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("credential file /nonexistent/request-recording/access_key_id"),
            "{err}"
        );
    }

    #[test]
    fn request_trace_sink_failure_does_not_block_shutdown() {
        let tmp = std::env::temp_dir().join(format!(
            "pm-core-invalid-sink-{}",
            unix_ms(SystemTime::now())
        ));
        fs::write(tmp.as_path(), b"not-a-directory").unwrap();

        let (done_tx, done_rx) = channel();
        thread::spawn(move || {
            let counters;
            {
                let factory = RequestTraceFactory::new(RequestTraceOptions {
                    request_log_enabled: true,
                    local_output_dir: Some(tmp),
                    request_log_batch_size: 2,
                    ..Default::default()
                })
                .unwrap();
                counters = Arc::clone(&factory.recorder.counters);
                for _ in 0..2 {
                    factory.new_trace(&[1, 2, 3], None, vec![], None).success();
                }
                let mut trace = factory.new_trace(&[1, 2, 3], None, vec![], None);
                trace.record_output_tokens(&[4], true, None);
                trace.success();
            }
            wait_for(&counters.rows_dropped_flush_error, 3);
            assert_eq!(counters.rows_persisted.load(Ordering::Relaxed), 0);
            let _ = done_tx.send(());
        });

        assert!(done_rx.recv_timeout(Duration::from_secs(6)).is_ok());
    }

    #[test]
    fn request_trace_s3_config_builds_without_network_io() {
        let factory = RequestTraceFactory::new(RequestTraceOptions {
            request_log_enabled: true,
            local_output_dir: None,
            s3_config: Some(S3SinkConfig {
                bucket: "example-bucket".to_string(),
                region: "us-east-1".to_string(),
                endpoint: Some("http://localhost:4566".to_string()),
                proxy_url: Some("http://localhost:3128".to_string()),
                access_key_id: Some("x".to_string()),
                secret_access_key: Some("x".to_string()),
                prefix: "records".to_string(),
            }),
            ..Default::default()
        });
        assert!(factory.is_ok());
    }

    #[test]
    fn request_trace_rejects_invalid_s3_proxy_url() {
        let factory = RequestTraceFactory::new(RequestTraceOptions {
            request_log_enabled: true,
            local_output_dir: None,
            s3_config: Some(S3SinkConfig {
                bucket: "example-bucket".to_string(),
                region: "us-east-1".to_string(),
                endpoint: None,
                proxy_url: Some("not-a-url".to_string()),
                access_key_id: Some("x".to_string()),
                secret_access_key: Some("x".to_string()),
                prefix: "records".to_string(),
            }),
            ..Default::default()
        });
        assert!(factory.is_err());
    }

    #[test]
    fn request_trace_s3_upload_connects_through_proxy() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let proxy_url = format!("http://{}", listener.local_addr().unwrap());
        let (request_tx, request_rx) = channel();
        let proxy = thread::spawn(move || {
            listener.set_nonblocking(true).unwrap();
            let deadline = Instant::now() + Duration::from_secs(5);
            while Instant::now() < deadline {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        stream
                            .set_read_timeout(Some(Duration::from_secs(2)))
                            .unwrap();
                        let mut buf = [0_u8; 1024];
                        let n = stream.read(&mut buf).unwrap();
                        request_tx
                            .send(String::from_utf8_lossy(&buf[..n]).into_owned())
                            .unwrap();
                        let _ = std::io::Write::write_all(
                            &mut stream,
                            b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                        );
                        return;
                    }
                    Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(10));
                    }
                    Err(err) => panic!("proxy accept failed: {err}"),
                }
            }
        });

        let sink = build_event_recorder_sink(
            None,
            Some(S3SinkConfig {
                bucket: "example-bucket".to_string(),
                region: "us-east-1".to_string(),
                endpoint: Some("https://s3.example.invalid".to_string()),
                proxy_url: Some(proxy_url),
                access_key_id: Some("x".to_string()),
                secret_access_key: Some("x".to_string()),
                prefix: "records".to_string(),
            }),
            FileFormat::Json,
        )
        .unwrap();
        let runtime = RuntimeBuilder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            let _ =
                tokio::time::timeout(Duration::from_secs(3), sink.write_batch("test.json", b"{}"))
                    .await;
        });
        let request = request_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(request.starts_with("CONNECT s3.example.invalid:443 HTTP/1.1"));
        proxy.join().unwrap();
    }

    #[test]
    fn request_metrics_factory_tracks_ttft_and_kv_ratio() {
        let factory = RequestMetricsFactory::new(RequestMetricsOptions::default()).unwrap();
        let mut req = factory
            .new_request_labeled(
                Some(&[1, 2, 3, 4]),
                4,
                RequestFeatures::NONE,
                None,
                None,
                None,
                None,
            )
            .unwrap();
        req.record_tokens(&[100, 101, 102, 103, 104], Some(2), true, None);
        req.success();

        let snapshot = factory.snapshot();
        let ttft = find_metric_snapshot(&snapshot, "request_metrics_ttft_ms", Some("all"));
        let time_to_fifth_token = find_metric_snapshot(
            &snapshot,
            "request_metrics_time_to_fifth_token_ms",
            Some("all"),
        );
        let first_to_fifth_token_itl = find_metric_snapshot(
            &snapshot,
            "request_metrics_first_to_fifth_token_itl_ms",
            Some("all"),
        );
        let kv = find_metric_snapshot(&snapshot, "request_metrics_kv_cache_hit_rate", None);
        let itl = find_metric_snapshot(&snapshot, "request_metrics_itl_ms", Some("all"));
        let input_len =
            find_metric_snapshot(&snapshot, "request_metrics_input_length_tokens", None);
        let output_len =
            find_metric_snapshot(&snapshot, "request_metrics_output_length_tokens", None);
        let successful =
            find_metric_snapshot(&snapshot, "request_metrics_successful_request", None);
        assert!(ttft.average.unwrap_or(0.0) >= 0.0);
        assert!(time_to_fifth_token.average.unwrap_or(0.0) >= 0.0);
        assert!(first_to_fifth_token_itl.average.unwrap_or(0.0) >= 0.0);
        assert!((kv.ratio.unwrap_or(0.0) - 0.5).abs() < 1e-9);
        assert!(itl.average.unwrap_or(0.0) >= 0.0);
        assert!(input_len.average.unwrap_or(0.0) >= 0.0);
        assert!(output_len.average.unwrap_or(0.0) >= 0.0);
        assert!(successful.rate_per_second.unwrap_or(0.0) >= 0.0);
    }

    #[test]
    fn request_metrics_feature_presence_series_overlap() {
        let factory = RequestMetricsFactory::new(RequestMetricsOptions::default()).unwrap();
        let mut req = factory
            .new_request_labeled(
                Some(&[1, 2, 3, 4]),
                4,
                RequestFeatures::TOOLS | RequestFeatures::IMAGE,
                None,
                None,
                None,
                None,
            )
            .unwrap();
        req.record_tokens(&[100, 101, 102], Some(2), true, None);
        req.record_tokens(&[103, 104], None, true, None);
        req.success();

        let snapshot = factory.snapshot();
        let ttft_tools = find_metric_snapshot(&snapshot, "request_metrics_ttft_ms", Some("tools"));
        let ttft_image = find_metric_snapshot(&snapshot, "request_metrics_ttft_ms", Some("image"));
        let ttft_none = find_metric_snapshot(&snapshot, "request_metrics_ttft_ms", Some("none"));
        let ttft_xgrammar =
            find_metric_snapshot(&snapshot, "request_metrics_ttft_ms", Some("xgrammar"));
        let fifth_token_tools = find_metric_snapshot(
            &snapshot,
            "request_metrics_time_to_fifth_token_ms",
            Some("tools"),
        );
        let fifth_token_image = find_metric_snapshot(
            &snapshot,
            "request_metrics_time_to_fifth_token_ms",
            Some("image"),
        );
        let fifth_token_none = find_metric_snapshot(
            &snapshot,
            "request_metrics_time_to_fifth_token_ms",
            Some("none"),
        );
        let fifth_token_xgrammar = find_metric_snapshot(
            &snapshot,
            "request_metrics_time_to_fifth_token_ms",
            Some("xgrammar"),
        );
        let first_to_fifth_itl_tools = find_metric_snapshot(
            &snapshot,
            "request_metrics_first_to_fifth_token_itl_ms",
            Some("tools"),
        );
        let first_to_fifth_itl_image = find_metric_snapshot(
            &snapshot,
            "request_metrics_first_to_fifth_token_itl_ms",
            Some("image"),
        );
        let first_to_fifth_itl_none = find_metric_snapshot(
            &snapshot,
            "request_metrics_first_to_fifth_token_itl_ms",
            Some("none"),
        );
        let first_to_fifth_itl_xgrammar = find_metric_snapshot(
            &snapshot,
            "request_metrics_first_to_fifth_token_itl_ms",
            Some("xgrammar"),
        );
        let itl_tools = find_metric_snapshot(&snapshot, "request_metrics_itl_ms", Some("tools"));
        let itl_image = find_metric_snapshot(&snapshot, "request_metrics_itl_ms", Some("image"));
        let itl_none = find_metric_snapshot(&snapshot, "request_metrics_itl_ms", Some("none"));
        let itl_xgrammar =
            find_metric_snapshot(&snapshot, "request_metrics_itl_ms", Some("xgrammar"));
        let request_all = find_metric_snapshot(&snapshot, "request_metrics_request", Some("all"));
        let request_tools =
            find_metric_snapshot(&snapshot, "request_metrics_request", Some("tools"));
        let request_image =
            find_metric_snapshot(&snapshot, "request_metrics_request", Some("image"));
        let request_none = find_metric_snapshot(&snapshot, "request_metrics_request", Some("none"));
        let request_xgrammar =
            find_metric_snapshot(&snapshot, "request_metrics_request", Some("xgrammar"));

        assert!(ttft_tools.average.unwrap_or(0.0) >= 0.0);
        assert!(ttft_image.average.unwrap_or(0.0) >= 0.0);
        assert_eq!(ttft_none.average, None);
        assert_eq!(ttft_xgrammar.average, None);
        assert!(fifth_token_tools.average.unwrap_or(0.0) >= 0.0);
        assert!(fifth_token_image.average.unwrap_or(0.0) >= 0.0);
        assert_eq!(fifth_token_none.average, None);
        assert_eq!(fifth_token_xgrammar.average, None);
        assert!(first_to_fifth_itl_tools.average.unwrap_or(0.0) >= 0.0);
        assert!(first_to_fifth_itl_image.average.unwrap_or(0.0) >= 0.0);
        assert_eq!(first_to_fifth_itl_none.average, None);
        assert_eq!(first_to_fifth_itl_xgrammar.average, None);
        assert!(itl_tools.average.unwrap_or(0.0) >= 0.0);
        assert!(itl_image.average.unwrap_or(0.0) >= 0.0);
        assert_eq!(itl_none.average, None);
        assert_eq!(itl_xgrammar.average, None);
        assert!(request_all.rate_per_second.unwrap_or(0.0) >= 0.0);
        assert!(request_tools.rate_per_second.unwrap_or(0.0) >= 0.0);
        assert!(request_image.rate_per_second.unwrap_or(0.0) >= 0.0);
        assert_eq!(request_none.rate_per_second.unwrap_or(0.0), 0.0);
        assert_eq!(request_xgrammar.rate_per_second.unwrap_or(0.0), 0.0);
    }

    #[test]
    fn request_metrics_none_feature_series_records_for_plain_requests() {
        let factory = RequestMetricsFactory::new(RequestMetricsOptions::default()).unwrap();
        let mut req = factory
            .new_request_labeled(
                Some(&[1, 2, 3, 4]),
                4,
                RequestFeatures::NONE,
                None,
                None,
                None,
                None,
            )
            .unwrap();
        req.record_tokens(&[100, 101, 102, 103, 104], Some(2), true, None);
        req.success();

        let snapshot = factory.snapshot();
        let ttft_none = find_metric_snapshot(&snapshot, "request_metrics_ttft_ms", Some("none"));
        let fifth_token_none = find_metric_snapshot(
            &snapshot,
            "request_metrics_time_to_fifth_token_ms",
            Some("none"),
        );
        let first_to_fifth_itl_none = find_metric_snapshot(
            &snapshot,
            "request_metrics_first_to_fifth_token_itl_ms",
            Some("none"),
        );
        let itl_none = find_metric_snapshot(&snapshot, "request_metrics_itl_ms", Some("none"));
        let request_none = find_metric_snapshot(&snapshot, "request_metrics_request", Some("none"));
        let request_all = find_metric_snapshot(&snapshot, "request_metrics_request", Some("all"));

        assert!(ttft_none.average.unwrap_or(0.0) >= 0.0);
        assert!(fifth_token_none.average.unwrap_or(0.0) >= 0.0);
        assert!(first_to_fifth_itl_none.average.unwrap_or(0.0) >= 0.0);
        assert!(itl_none.average.unwrap_or(0.0) >= 0.0);
        assert!(request_none.rate_per_second.unwrap_or(0.0) >= 0.0);
        assert!(request_all.rate_per_second.unwrap_or(0.0) >= 0.0);
    }

    #[test]
    fn request_metrics_factory_tracks_lifetime_totals() {
        let factory = RequestMetricsFactory::new(RequestMetricsOptions::default()).unwrap();
        let mut req = factory
            .new_request_labeled(
                Some(&[1, 2, 3, 4]),
                4,
                RequestFeatures::NONE,
                None,
                None,
                None,
                None,
            )
            .unwrap();
        req.record_tokens(&[10, 11], Some(2), true, None);
        req.record_tokens(&[12], None, true, None);
        req.success();

        let snapshot = factory.snapshot();
        let totals = &snapshot.totals;
        assert_eq!(totals.total_requests, 1);
        assert_eq!(totals.total_successful_requests, 1);
        assert_eq!(totals.total_pre_first_token_cancellations, 0);
        assert_eq!(totals.total_pre_20_token_cancellations, 0);
        assert_eq!(totals.total_mid_stream_cancellations, 0);
        assert_eq!(totals.total_input_tokens, 4);
        assert_eq!(totals.total_output_tokens, 3);
        assert_eq!(totals.total_kv_cache_hits_tokens, 2);
        assert_eq!(totals.total_kv_cache_lookup_tokens, 4);
        assert_eq!(totals.kv_cache_hit_rate_lifetime.unwrap_or(0.0), 0.5);
        assert_eq!(totals.current_inflight_requests, 0);
        assert_eq!(totals.current_inflight_input_tokens, 0);
        assert_eq!(totals.current_inflight_output_tokens, 0);
        assert_eq!(totals.current_inflight_first_to_fifth_token_requests, 0);
        assert_eq!(totals.current_inflight_first_to_fifth_token_input_tokens, 0);
        assert!(totals.mean_inflight_requests >= 0.0);
        assert!(totals.mean_inflight_input_tokens >= 0.0);
        assert!(totals.mean_inflight_output_tokens >= 0.0);
        assert!(totals.mean_inflight_first_to_fifth_token_requests >= 0.0);
        assert!(totals.mean_inflight_first_to_fifth_token_input_tokens >= 0.0);
    }

    #[test]
    fn inflight_snapshot_tracks_current_request_and_tokens() {
        let factory = RequestMetricsFactory::new(RequestMetricsOptions::default()).unwrap();
        let mut req = factory
            .new_request_labeled(
                Some(&[1, 2, 3, 4]),
                4,
                RequestFeatures::NONE,
                None,
                None,
                None,
                None,
            )
            .unwrap();
        let snapshot = factory.snapshot();
        assert_eq!(snapshot.totals.current_inflight_requests, 1);
        assert_eq!(snapshot.totals.current_inflight_input_tokens, 4);
        assert_eq!(snapshot.totals.current_inflight_output_tokens, 0);

        req.record_tokens(&[10, 11], Some(2), true, None);
        let snapshot = factory.snapshot();
        assert_eq!(snapshot.totals.current_inflight_requests, 1);
        assert_eq!(snapshot.totals.current_inflight_input_tokens, 4);
        assert_eq!(snapshot.totals.current_inflight_output_tokens, 2);

        req.success();
        let snapshot = factory.snapshot();
        assert_eq!(snapshot.totals.current_inflight_requests, 0);
        assert_eq!(snapshot.totals.current_inflight_input_tokens, 0);
        assert_eq!(snapshot.totals.current_inflight_output_tokens, 0);
    }

    #[test]
    fn first_to_fifth_inflight_tracks_current_request_and_input_tokens() {
        let factory = RequestMetricsFactory::new(RequestMetricsOptions::default()).unwrap();
        let mut req = factory
            .new_request_labeled(
                Some(&[1, 2, 3, 4]),
                4,
                RequestFeatures::NONE,
                None,
                None,
                None,
                None,
            )
            .unwrap();

        let snapshot = factory.snapshot();
        assert_eq!(
            snapshot
                .totals
                .current_inflight_first_to_fifth_token_requests,
            0
        );
        assert_eq!(
            snapshot
                .totals
                .current_inflight_first_to_fifth_token_input_tokens,
            0
        );

        req.record_tokens(&[10], Some(2), true, None);
        let snapshot = factory.snapshot();
        assert_eq!(
            snapshot
                .totals
                .current_inflight_first_to_fifth_token_requests,
            1
        );
        assert_eq!(
            snapshot
                .totals
                .current_inflight_first_to_fifth_token_input_tokens,
            4
        );

        req.record_tokens(&[11, 12, 13], None, true, None);
        let snapshot = factory.snapshot();
        assert_eq!(
            snapshot
                .totals
                .current_inflight_first_to_fifth_token_requests,
            1
        );
        assert_eq!(
            snapshot
                .totals
                .current_inflight_first_to_fifth_token_input_tokens,
            4
        );

        req.record_tokens(&[14], None, true, None);
        let snapshot = factory.snapshot();
        assert_eq!(
            snapshot
                .totals
                .current_inflight_first_to_fifth_token_requests,
            0
        );
        assert_eq!(
            snapshot
                .totals
                .current_inflight_first_to_fifth_token_input_tokens,
            0
        );
    }

    #[test]
    fn first_to_fifth_inflight_clears_on_cancel() {
        let factory = RequestMetricsFactory::new(RequestMetricsOptions::default()).unwrap();
        let mut req = factory
            .new_request_labeled(
                Some(&[1, 2, 3, 4]),
                4,
                RequestFeatures::NONE,
                None,
                None,
                None,
                None,
            )
            .unwrap();
        req.record_tokens(&[10], Some(2), true, None);
        req.cancel();

        let snapshot = factory.snapshot();
        assert_eq!(
            snapshot
                .totals
                .current_inflight_first_to_fifth_token_requests,
            0
        );
        assert_eq!(
            snapshot
                .totals
                .current_inflight_first_to_fifth_token_input_tokens,
            0
        );
    }

    #[test]
    fn first_to_fifth_inflight_clears_on_drop() {
        let factory = RequestMetricsFactory::new(RequestMetricsOptions::default()).unwrap();
        {
            let mut req = factory
                .new_request_labeled(
                    Some(&[1, 2, 3, 4]),
                    4,
                    RequestFeatures::NONE,
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap();
            req.record_tokens(&[10], Some(2), true, None);
        }

        let snapshot = factory.snapshot();
        assert_eq!(
            snapshot
                .totals
                .current_inflight_first_to_fifth_token_requests,
            0
        );
        assert_eq!(
            snapshot
                .totals
                .current_inflight_first_to_fifth_token_input_tokens,
            0
        );
    }

    #[test]
    fn first_to_fifth_inflight_single_batch_five_tokens_is_not_current() {
        let factory = RequestMetricsFactory::new(RequestMetricsOptions::default()).unwrap();
        let mut req = factory
            .new_request_labeled(
                Some(&[1, 2, 3, 4]),
                4,
                RequestFeatures::NONE,
                None,
                None,
                None,
                None,
            )
            .unwrap();
        req.record_tokens(&[10, 11, 12, 13, 14], Some(2), true, None);

        let snapshot = factory.snapshot();
        assert_eq!(
            snapshot
                .totals
                .current_inflight_first_to_fifth_token_requests,
            0
        );
        assert_eq!(
            snapshot
                .totals
                .current_inflight_first_to_fifth_token_input_tokens,
            0
        );
    }

    #[test]
    fn dropped_request_counts_as_cancellation() {
        let factory = RequestMetricsFactory::new(RequestMetricsOptions::default()).unwrap();
        {
            let _req = factory
                .new_request_labeled(
                    Some(&[1, 2, 3, 4]),
                    4,
                    RequestFeatures::NONE,
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap();
        }

        let snapshot = factory.snapshot();
        let totals = &snapshot.totals;
        assert_eq!(totals.total_requests, 1);
        assert_eq!(totals.total_pre_first_token_cancellations, 1);
        assert_eq!(totals.total_pre_20_token_cancellations, 0);
        assert_eq!(totals.total_mid_stream_cancellations, 0);

        let cancellation = find_metric_snapshot(
            &snapshot,
            "request_metrics_pre_first_token_cancellation",
            None,
        );
        assert!(cancellation.rate_per_second.unwrap_or(0.0) >= 0.0);
    }

    #[test]
    fn cancellation_before_20_tokens_tracks_pre_20_bucket() {
        let factory = RequestMetricsFactory::new(RequestMetricsOptions::default()).unwrap();
        let mut req = factory
            .new_request_labeled(
                Some(&[1, 2, 3, 4]),
                4,
                RequestFeatures::NONE,
                None,
                None,
                None,
                None,
            )
            .unwrap();
        req.record_tokens(&[10, 11, 12], Some(2), true, None);
        req.cancel();

        let snapshot = factory.snapshot();
        let totals = &snapshot.totals;
        assert_eq!(totals.total_requests, 1);
        assert_eq!(totals.total_pre_first_token_cancellations, 0);
        assert_eq!(totals.total_pre_20_token_cancellations, 1);
        assert_eq!(totals.total_mid_stream_cancellations, 0);

        let cancellation =
            find_metric_snapshot(&snapshot, "request_metrics_pre_20_token_cancellation", None);
        assert!(cancellation.rate_per_second.unwrap_or(0.0) >= 0.0);
    }

    #[test]
    fn request_metrics_default_has_dense_kv_cache_quantiles() {
        let options = RequestMetricsOptions::default();
        assert_eq!(
            options.kv_cache_hit_rate_quantiles,
            vec![0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]
        );
    }

    #[test]
    fn request_log_entry_includes_null_session_id_when_absent() {
        let entry = RequestLogEntry {
            timestamp_start_unix_ms: 1,
            duration_e2e_ms: 100,
            duration_ttft_ms: 50,
            input_tokens: 3,
            output_tokens: 4,
            total_hashes: vec![[
                0xab, 0xcd, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00,
            ]],
            provided_session_id: None,
            request_canceled: false,
            cached_tokens_reference: None,
            model_name: "unset_model_name".to_string(),
            org_id: String::new(),
            component_name: None,
            component_id: None,
            block_size: DEFAULT_BLOCK_SIZE,
            features: vec![],
            speculation_ratio: None,
            mean_output_tokens_per_step: None,
            service_tier: None,
            b10_request_id: None,
            prefill_worker_id: None,
            decode_worker_id: None,
            prefill_dp_rank: None,
            decode_dp_rank: None,
            __version__: DEFAULT_REQUEST_LOG_VERSION.to_string(),
        };

        let value = serde_json::to_value(entry).unwrap();
        // Now that skip_serializing_if is removed, null values should be included
        assert!(value.get("provided_session_id").is_some());
        assert!(value.get("provided_session_id").unwrap().is_null());
        assert_eq!(
            value
                .get("total_hashes")
                .and_then(|v| v.as_array())
                .and_then(|v| v.first())
                .and_then(|v| v.as_str()),
            Some("abcd000000000000000000000000000000000000000000000000000000000000")
        );
    }

    #[test]
    fn request_trace_hashes_session_id_when_present() {
        let factory = RequestTraceFactory::new(RequestTraceOptions::default()).unwrap();
        let trace = factory.new_trace(&[1, 2, 3], Some("session-123"), vec![], None);
        let entry = trace.build_log_entry(false);
        let expected = hash_session_id("session-123");

        assert_eq!(
            entry.provided_session_id.as_deref(),
            Some(expected.as_str())
        );
    }

    #[test]
    fn request_trace_hashes_session_id_independently_of_salt() {
        let salted_factory = RequestTraceFactory::new(RequestTraceOptions {
            request_log_hash_salt: Some("fixed-salt".to_string()),
            ..Default::default()
        })
        .unwrap();
        let unsalted_factory = RequestTraceFactory::new(RequestTraceOptions::default()).unwrap();
        let salted_entry = salted_factory
            .new_trace(&[1, 2, 3], Some("session-123"), vec![], None)
            .build_log_entry(false);
        let unsalted_entry = unsalted_factory
            .new_trace(&[1, 2, 3], Some("session-123"), vec![], None)
            .build_log_entry(false);

        assert_eq!(
            salted_entry.provided_session_id,
            unsalted_entry.provided_session_id
        );
    }

    #[test]
    fn request_log_entry_contains_service_tier() {
        let factory = RequestTraceFactory::new(RequestTraceOptions::default()).unwrap();
        let trace = factory.new_trace(&[1, 2, 3], None, vec![], Some(ServiceTier::Fast));
        let entry = trace.build_log_entry(false);
        assert_eq!(entry.service_tier.as_deref(), Some("fast"));
    }

    #[test]
    fn request_log_entry_omits_service_tier_when_none() {
        let factory = RequestTraceFactory::new(RequestTraceOptions::default()).unwrap();
        let trace = factory.new_trace(&[1, 2, 3], None, vec![], None);
        let entry = trace.build_log_entry(false);
        assert!(entry.service_tier.is_none());
    }

    #[test]
    fn request_trace_preserves_model_name_in_log_entry() {
        let model_name = "my-model-v1";
        let factory = RequestTraceFactory::new(RequestTraceOptions {
            model_name: model_name.to_string(),
            component_name: Some("frontend".to_string()),
            ..Default::default()
        })
        .unwrap();

        let trace = factory.new_trace(&[1, 2, 3], None, vec![], None);
        let entry = trace.build_log_entry(false);
        assert_eq!(entry.model_name, model_name);
        assert_eq!(entry.component_name.as_deref(), Some("frontend"));
        let component_id = entry.component_id.as_deref().unwrap();
        assert_eq!(component_id.len(), 32);
        assert!(component_id.bytes().all(|byte| byte.is_ascii_hexdigit()));

        let second_factory = RequestTraceFactory::new(RequestTraceOptions::default()).unwrap();
        let second_entry = second_factory
            .new_trace(&[4, 5, 6], None, vec![], None)
            .build_log_entry(false);
        assert_eq!(second_entry.component_id.as_deref(), Some(component_id));
    }

    #[test]
    fn request_trace_hashes_org_id() {
        let org_id = "my-sensitive-org-123";
        let factory = RequestTraceFactory::new(RequestTraceOptions {
            model_name: "test-model".to_string(),
            org_id: org_id.to_string(),
            ..Default::default()
        })
        .unwrap();

        // The org_id should be hashed (64 hex characters for blake3)
        assert_eq!(factory.org_id.len(), 64);
        assert_ne!(factory.org_id, org_id);

        // Verify it's a valid hex string
        assert!(factory.org_id.chars().all(|c| c.is_ascii_hexdigit()));

        // Verify the hash is correct
        let expected_hash = blake3::hash(org_id.as_bytes()).to_hex().to_string();
        assert_eq!(factory.org_id, expected_hash);
    }

    #[test]
    fn perf_record_tokens_mean_latency_is_reasonable_in_dev_profile() {
        let factory = RequestMetricsFactory::new(RequestMetricsOptions::default()).unwrap();

        let mut total_calls = 0u64;
        let mut total_record_ns = 0u128;

        // Keep this bounded for CI/dev while still exercising thousands of tokens.
        let requests = 8usize;
        let chunks_per_request = 32usize;
        let tokens_per_chunk = 128usize; // 4096 output tokens per request

        for req_idx in 0..requests {
            let input = (0..256u32).collect::<Vec<_>>();
            let mut req = factory
                .new_request_labeled(
                    Some(input.as_slice()),
                    input.len(),
                    RequestFeatures::TOOLS,
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap();

            let mut next_token = (req_idx as u32) * 10_000;
            for _ in 0..chunks_per_request {
                let chunk = (0..tokens_per_chunk)
                    .map(|_| {
                        let v = next_token;
                        next_token = next_token.saturating_add(1);
                        v
                    })
                    .collect::<Vec<_>>();

                let t0 = Instant::now();
                req.record_tokens(chunk.as_slice(), Some(64), true, None);
                total_record_ns += t0.elapsed().as_nanos();
                total_calls += 1;
            }
            req.success();
        }

        let mean_us = (total_record_ns as f64 / total_calls as f64) / 1_000.0;

        // Relative baseline: strict atomic adds in a tight loop.
        let strict_ops_per_iter = 64u64;
        let baseline_counter = AtomicU64::new(0);
        let t0 = Instant::now();
        for _ in 0..total_calls {
            for _ in 0..strict_ops_per_iter {
                let _ = black_box(&baseline_counter).fetch_add(1, Ordering::SeqCst);
            }
        }
        let baseline_ns_total = t0.elapsed().as_nanos() as f64;
        let baseline_ns_per_call = baseline_ns_total / total_calls as f64;
        let record_ns_per_call = total_record_ns as f64 / total_calls as f64;
        let ratio = record_ns_per_call / baseline_ns_per_call.max(1.0);
        eprintln!(
            "perf_record_tokens_mean_latency_is_reasonable_in_dev_profile: calls={total_calls}, mean_us={mean_us:.2}, baseline_ns_per_call={baseline_ns_per_call:.2}, ratio={ratio:.2}"
        );

        // Conservative budget for unoptimized dev profile and shared CI hosts.
        assert!(
            mean_us < 250.0,
            "record_tokens mean latency too high: {mean_us:.2}us over {total_calls} calls"
        );

        assert!(
            ratio < 80.0,
            "record_tokens to strict-atomic baseline ratio too high: ratio={ratio:.2}, record_ns_per_call={record_ns_per_call:.2}, baseline_ns_per_call={baseline_ns_per_call:.2}"
        );
    }

    /// Labeled variant of the perf test: exercises `record_tokens` with
    /// `service_tier` only and with both `service_tier` + `dp_rank`, asserting
    /// that the extra labeled handle writes do not blow latency past the same
    /// budget as the plain path.
    #[test]
    fn perf_record_tokens_labeled_mean_latency_is_reasonable_in_dev_profile() {
        let factory = RequestMetricsFactory::new(RequestMetricsOptions::default()).unwrap();

        let requests = 8usize;
        let chunks_per_request = 32usize;
        let tokens_per_chunk = 128usize;

        for (tier, rank, label) in [
            (Some(ServiceTier::Fast), None, "tier-only"),
            (Some(ServiceTier::Fast), Some(7), "tier+rank"),
        ] {
            let mut total_calls = 0u64;
            let mut total_record_ns = 0u128;

            for req_idx in 0..requests {
                let input = (0..256u32).collect::<Vec<_>>();
                let mut req = factory
                    .new_request_labeled(
                        Some(input.as_slice()),
                        input.len(),
                        RequestFeatures::TOOLS,
                        None,
                        None,
                        tier,
                        rank,
                    )
                    .unwrap();

                let mut next_token = (req_idx as u32) * 10_000;
                for _ in 0..chunks_per_request {
                    let chunk = (0..tokens_per_chunk)
                        .map(|_| {
                            let v = next_token;
                            next_token = next_token.saturating_add(1);
                            v
                        })
                        .collect::<Vec<_>>();

                    let t0 = Instant::now();
                    req.record_tokens(chunk.as_slice(), Some(64), true, None);
                    total_record_ns += t0.elapsed().as_nanos();
                    total_calls += 1;
                }
                req.success();
            }

            let mean_us = (total_record_ns as f64 / total_calls as f64) / 1_000.0;

            let strict_ops_per_iter = 64u64;
            let baseline_counter = AtomicU64::new(0);
            let t0 = Instant::now();
            for _ in 0..total_calls {
                for _ in 0..strict_ops_per_iter {
                    let _ = black_box(&baseline_counter).fetch_add(1, Ordering::SeqCst);
                }
            }
            let baseline_ns_total = t0.elapsed().as_nanos() as f64;
            let baseline_ns_per_call = baseline_ns_total / total_calls as f64;
            let record_ns_per_call = total_record_ns as f64 / total_calls as f64;
            let ratio = record_ns_per_call / baseline_ns_per_call.max(1.0);
            eprintln!(
                "perf_record_tokens_labeled ({label}): calls={total_calls}, mean_us={mean_us:.2}, ratio={ratio:.2}"
            );

            assert!(
                mean_us < 250.0,
                "labeled record_tokens mean latency too high ({label}): {mean_us:.2}us over {total_calls} calls"
            );
            assert!(
                ratio < 80.0,
                "labeled record_tokens ratio too high ({label}): ratio={ratio:.2}"
            );
        }
    }

    #[test]
    fn request_trace_org_id_is_empty_when_not_provided() {
        let factory = RequestTraceFactory::new(RequestTraceOptions {
            model_name: "test-model".to_string(),
            org_id: String::new(),
            ..Default::default()
        })
        .unwrap();

        // Empty string should be hashed (not empty)
        assert_eq!(factory.org_id.len(), 64);
        let expected_hash = blake3::hash(b"").to_hex().to_string();
        assert_eq!(factory.org_id, expected_hash);
    }

    #[test]
    fn request_trace_parquet_schema_matches_data() {
        // Test that serialize_to_parquet produces valid parquet with matching schema
        // Use valid 64-character hex strings (representing 32-byte blake3 hashes)
        let entries = vec![RequestLogEntry {
            timestamp_start_unix_ms: 1234567890,
            duration_e2e_ms: 100,
            duration_ttft_ms: 50,
            input_tokens: 10,
            output_tokens: 20,
            total_hashes: vec![
                [
                    0xa1, 0xb2, 0xc3, 0xd4, 0xe5, 0xf6, 0xa7, 0xb8, 0xc9, 0xd0, 0xe1, 0xf2, 0xa3,
                    0xb4, 0xc5, 0xd6, 0xe7, 0xf8, 0xa9, 0xb0, 0xc1, 0xd2, 0xe3, 0xf4, 0xa5, 0xb6,
                    0xc7, 0xd8, 0xe9, 0xf0, 0xa1, 0xb2,
                ],
                [
                    0xe5, 0xf6, 0xa7, 0xb8, 0xc9, 0xd0, 0xe1, 0xf2, 0xa3, 0xb4, 0xc5, 0xd6, 0xe7,
                    0xf8, 0xa9, 0xb0, 0xc1, 0xd2, 0xe3, 0xf4, 0xa5, 0xb6, 0xc7, 0xd8, 0xe9, 0xf0,
                    0xa1, 0xb2, 0xc3, 0xd4, 0xe5, 0xf6,
                ],
                [
                    0xa9, 0xb0, 0xc1, 0xd2, 0xe3, 0xf4, 0xa5, 0xb6, 0xc7, 0xd8, 0xe9, 0xf0, 0xa1,
                    0xb2, 0xc3, 0xd4, 0xe5, 0xf6, 0xa7, 0xb8, 0xc9, 0xd0, 0xe1, 0xf2, 0xa3, 0xb4,
                    0xc5, 0xd6, 0xe7, 0xf8, 0xa9, 0xb0,
                ],
            ],
            provided_session_id: Some("session-123".to_string()),
            request_canceled: false,
            cached_tokens_reference: Some(5),
            model_name: "test-model".to_string(),
            org_id: "test-org-123".to_string(),
            component_name: Some("frontend".to_string()),
            component_id: Some("12345".to_string()),
            block_size: DEFAULT_BLOCK_SIZE,
            features: vec!["feature1".to_string()],
            speculation_ratio: Some(3.5),
            mean_output_tokens_per_step: Some(2.0),
            service_tier: None,
            b10_request_id: None,
            prefill_worker_id: None,
            decode_worker_id: None,
            prefill_dp_rank: None,
            decode_dp_rank: None,
            __version__: DEFAULT_REQUEST_LOG_VERSION.to_string(),
        }];

        let parquet_bytes = serialize_to_parquet(&entries).expect("Should serialize to parquet");
        assert!(
            !parquet_bytes.is_empty(),
            "Parquet bytes should not be empty"
        );

        // Verify it's valid parquet by reading it back
        use parquet::file::reader::{FileReader, SerializedFileReader};
        let reader = SerializedFileReader::new(bytes::Bytes::from(parquet_bytes))
            .expect("Should be valid parquet");
        let schema = reader.metadata().file_metadata().schema();

        // Verify all expected fields are present
        let expected_fields = vec![
            "timestamp_start_unix_ms",
            "duration_e2e_ms",
            "duration_ttft_ms",
            "input_tokens",
            "output_tokens",
            "total_hashes",
            "provided_session_id",
            "request_canceled",
            "cached_tokens_reference",
            "model_name",
            "org_id",
            "component_name",
            "component_id",
            "block_size",
            "features",
            "speculation_ratio",
            "mean_output_tokens_per_step",
            "service_tier",
            "b10_request_id",
            "prefill_worker_id",
            "decode_worker_id",
            "prefill_dp_rank",
            "decode_dp_rank",
            "__version__",
        ];

        assert_eq!(
            schema.get_fields().len(),
            expected_fields.len(),
            "Schema field count should match expected"
        );

        for (i, expected_name) in expected_fields.iter().enumerate() {
            let actual_name = schema.get_fields()[i].name();
            assert_eq!(
                actual_name, *expected_name,
                "Field at position {} should match",
                i
            );
        }

        use arrow::array::Float64Array;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        let parquet_bytes = serialize_to_parquet(&entries).unwrap();
        let mut reader =
            ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::from(parquet_bytes))
                .unwrap()
                .build()
                .unwrap();
        let batch = reader.next().unwrap().unwrap();
        let column = batch
            .column_by_name("mean_output_tokens_per_step")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(column.value(0), 2.0);
    }

    #[test]
    fn request_trace_parquet_hashes_roundtrip() {
        use arrow::array::ListArray;
        use parquet::file::reader::{FileReader, SerializedFileReader};
        use std::io::Write;

        // Create test entries with multiple hashes (using valid 64-char hex strings)
        // These represent blake3-256 hashes (32 bytes = 64 hex chars)
        let entries = vec![
            RequestLogEntry {
                timestamp_start_unix_ms: 1000,
                duration_e2e_ms: 100,
                duration_ttft_ms: 50,
                input_tokens: 10,
                output_tokens: 20,
                total_hashes: vec![
                    [
                        0xa1, 0xb2, 0xc3, 0xd4, 0xe5, 0xf6, 0xa7, 0xb8, 0xc9, 0xd0, 0xe1, 0xf2,
                        0xa3, 0xb4, 0xc5, 0xd6, 0xe7, 0xf8, 0xa9, 0xb0, 0xc1, 0xd2, 0xe3, 0xf4,
                        0xa5, 0xb6, 0xc7, 0xd8, 0xe9, 0xf0, 0xa1, 0xb2,
                    ],
                    [
                        0xe5, 0xf6, 0xa7, 0xb8, 0xc9, 0xd0, 0xe1, 0xf2, 0xa3, 0xb4, 0xc5, 0xd6,
                        0xe7, 0xf8, 0xa9, 0xb0, 0xc1, 0xd2, 0xe3, 0xf4, 0xa5, 0xb6, 0xc7, 0xd8,
                        0xe9, 0xf0, 0xa1, 0xb2, 0xc3, 0xd4, 0xe5, 0xf6,
                    ],
                ],
                provided_session_id: Some("session-123".to_string()),
                request_canceled: false,
                cached_tokens_reference: Some(5),
                model_name: "test-model".to_string(),
                org_id: "test-org".to_string(),
                component_name: None,
                component_id: None,
                block_size: DEFAULT_BLOCK_SIZE,
                features: vec!["feature1".to_string()],
                speculation_ratio: None,
                mean_output_tokens_per_step: None,
                service_tier: None,
                b10_request_id: None,
                prefill_worker_id: None,
                decode_worker_id: None,
                prefill_dp_rank: None,
                decode_dp_rank: None,
                __version__: DEFAULT_REQUEST_LOG_VERSION.to_string(),
            },
            RequestLogEntry {
                timestamp_start_unix_ms: 2000,
                duration_e2e_ms: 200,
                duration_ttft_ms: 100,
                input_tokens: 20,
                output_tokens: 40,
                total_hashes: vec![
                    [
                        0xa1, 0xb2, 0xc3, 0xd4, 0xe5, 0xf6, 0xa7, 0xb8, 0xc9, 0xd0, 0xe1, 0xf2,
                        0xa3, 0xb4, 0xc5, 0xd6, 0xe7, 0xf8, 0xa9, 0xb0, 0xc1, 0xd2, 0xe3, 0xf4,
                        0xa5, 0xb6, 0xc7, 0xd8, 0xe9, 0xf0, 0xa1, 0xb2,
                    ],
                    [
                        0xe5, 0xf6, 0xa7, 0xb8, 0xc9, 0xd0, 0xe1, 0xf2, 0xa3, 0xb4, 0xc5, 0xd6,
                        0xe7, 0xf8, 0xa9, 0xb0, 0xc1, 0xd2, 0xe3, 0xf4, 0xa5, 0xb6, 0xc7, 0xd8,
                        0xe9, 0xf0, 0xa1, 0xb2, 0xc3, 0xd4, 0xe5, 0xf6,
                    ],
                    [
                        0xa9, 0xb0, 0xc1, 0xd2, 0xe3, 0xf4, 0xa5, 0xb6, 0xc7, 0xd8, 0xe9, 0xf0,
                        0xa1, 0xb2, 0xc3, 0xd4, 0xe5, 0xf6, 0xa7, 0xb8, 0xc9, 0xd0, 0xe1, 0xf2,
                        0xa3, 0xb4, 0xc5, 0xd6, 0xe7, 0xf8, 0xa9, 0xb0,
                    ],
                ],
                provided_session_id: None,
                request_canceled: true,
                cached_tokens_reference: None,
                model_name: "test-model-2".to_string(),
                org_id: "test-org-2".to_string(),
                component_name: None,
                component_id: None,
                block_size: DEFAULT_BLOCK_SIZE,
                features: vec![],
                speculation_ratio: Some(2.5),
                mean_output_tokens_per_step: Some(2.0),
                service_tier: None,
                b10_request_id: None,
                prefill_worker_id: None,
                decode_worker_id: None,
                prefill_dp_rank: None,
                decode_dp_rank: None,
                __version__: DEFAULT_REQUEST_LOG_VERSION.to_string(),
            },
        ];

        // Serialize to parquet
        let parquet_bytes = serialize_to_parquet(&entries).expect("Should serialize to parquet");

        // Write to temp file
        let temp_dir =
            std::env::temp_dir().join(format!("pm-parquet-test-{}", unix_ms(SystemTime::now())));
        std::fs::create_dir_all(&temp_dir).unwrap();
        let temp_file = temp_dir.join("test.parquet");
        let mut file = std::fs::File::create(&temp_file).unwrap();
        file.write_all(&parquet_bytes).unwrap();
        file.flush().unwrap();
        drop(file);

        // Read back using parquet reader
        let reader = SerializedFileReader::new(std::fs::File::open(&temp_file).unwrap())
            .expect("Should be valid parquet");

        // Get the row count
        let metadata = reader.metadata().file_metadata();
        assert_eq!(metadata.num_rows(), 2, "Should have 2 rows");

        // Read and validate the data using Arrow
        use arrow::array::{Array, FixedSizeBinaryArray};
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        let parquet_file = std::fs::File::open(&temp_file).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(parquet_file).unwrap();
        let record_reader = builder.build().unwrap();

        // Collect all batches
        let mut batches = Vec::new();
        for batch in record_reader {
            batches.push(batch.unwrap());
        }
        assert!(!batches.is_empty(), "Should have at least one batch");

        // Get the first batch and verify the total_hashes column
        let batch = &batches[0];
        let total_hashes_column = batch.column_by_name("total_hashes").unwrap();
        let total_hashes_list = total_hashes_column
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();

        // Verify first row has 2 hashes
        let first_row_hashes = total_hashes_list.value(0);
        let first_row_binary = first_row_hashes
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .unwrap();
        assert_eq!(first_row_binary.len(), 2, "First row should have 2 hashes");

        // Expected hash values as raw bytes
        let expected_hash1: HashBytes = [
            0xa1, 0xb2, 0xc3, 0xd4, 0xe5, 0xf6, 0xa7, 0xb8, 0xc9, 0xd0, 0xe1, 0xf2, 0xa3, 0xb4,
            0xc5, 0xd6, 0xe7, 0xf8, 0xa9, 0xb0, 0xc1, 0xd2, 0xe3, 0xf4, 0xa5, 0xb6, 0xc7, 0xd8,
            0xe9, 0xf0, 0xa1, 0xb2,
        ];
        let expected_hash2: HashBytes = [
            0xe5, 0xf6, 0xa7, 0xb8, 0xc9, 0xd0, 0xe1, 0xf2, 0xa3, 0xb4, 0xc5, 0xd6, 0xe7, 0xf8,
            0xa9, 0xb0, 0xc1, 0xd2, 0xe3, 0xf4, 0xa5, 0xb6, 0xc7, 0xd8, 0xe9, 0xf0, 0xa1, 0xb2,
            0xc3, 0xd4, 0xe5, 0xf6,
        ];

        // Verify the binary values
        assert_eq!(first_row_binary.value(0), expected_hash1.as_slice());
        assert_eq!(first_row_binary.value(1), expected_hash2.as_slice());

        // Verify second row has 3 hashes
        let second_row_hashes = total_hashes_list.value(1);
        let second_row_binary = second_row_hashes
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .unwrap();
        assert_eq!(
            second_row_binary.len(),
            3,
            "Second row should have 3 hashes"
        );

        // Verify the first hash of second row matches
        assert_eq!(second_row_binary.value(0), expected_hash1.as_slice());

        // Verify the third hash
        let expected_hash3: HashBytes = [
            0xa9, 0xb0, 0xc1, 0xd2, 0xe3, 0xf4, 0xa5, 0xb6, 0xc7, 0xd8, 0xe9, 0xf0, 0xa1, 0xb2,
            0xc3, 0xd4, 0xe5, 0xf6, 0xa7, 0xb8, 0xc9, 0xd0, 0xe1, 0xf2, 0xa3, 0xb4, 0xc5, 0xd6,
            0xe7, 0xf8, 0xa9, 0xb0,
        ];
        assert_eq!(second_row_binary.value(2), expected_hash3.as_slice());

        // Cleanup
        std::fs::remove_dir_all(&temp_dir).unwrap();
    }

    #[test]
    fn request_trace_parquet_batch_without_hashes() {
        // Requests shorter than one block have no hashes. A batch made only of
        // those must still serialize instead of panicking the recorder thread.
        use arrow::array::{Array, ListArray};
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let entry = |timestamp_start_unix_ms| RequestLogEntry {
            timestamp_start_unix_ms,
            duration_e2e_ms: 100,
            duration_ttft_ms: 50,
            input_tokens: 3,
            output_tokens: 5,
            total_hashes: vec![],
            provided_session_id: None,
            request_canceled: false,
            cached_tokens_reference: None,
            model_name: "test-model".to_string(),
            org_id: "test-org".to_string(),
            component_name: None,
            component_id: None,
            block_size: DEFAULT_BLOCK_SIZE,
            features: vec![],
            speculation_ratio: None,
            mean_output_tokens_per_step: None,
            service_tier: None,
            b10_request_id: None,
            prefill_worker_id: None,
            decode_worker_id: None,
            prefill_dp_rank: None,
            decode_dp_rank: None,
            __version__: DEFAULT_REQUEST_LOG_VERSION.to_string(),
        };
        let entries = vec![entry(1000), entry(2000)];

        let parquet_bytes = serialize_to_parquet(&entries).expect("Should serialize to parquet");

        let mut reader =
            ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::from(parquet_bytes))
                .unwrap()
                .build()
                .unwrap();
        let batch = reader.next().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 2);
        let hashes = batch
            .column_by_name("total_hashes")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        assert_eq!(hashes.value(0).len(), 0);
        assert_eq!(hashes.value(1).len(), 0);
        assert_eq!(
            hashes.value_type(),
            arrow::datatypes::DataType::FixedSizeBinary(HASH_SIZE as i32)
        );
    }

    #[test]
    fn request_log_parquet_contains_service_tier() {
        use arrow::array::StringArray;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let entries = vec![RequestLogEntry {
            timestamp_start_unix_ms: 1000,
            duration_e2e_ms: 100,
            duration_ttft_ms: 50,
            input_tokens: 10,
            output_tokens: 20,
            total_hashes: vec![[0u8; 32]],
            provided_session_id: None,
            request_canceled: false,
            cached_tokens_reference: None,
            model_name: "test-model".to_string(),
            org_id: "test-org".to_string(),
            component_name: Some("frontend".to_string()),
            component_id: Some("12345".to_string()),
            block_size: DEFAULT_BLOCK_SIZE,
            features: vec![],
            speculation_ratio: None,
            mean_output_tokens_per_step: None,
            service_tier: Some("fast".to_string()),
            b10_request_id: None,
            prefill_worker_id: None,
            decode_worker_id: None,
            prefill_dp_rank: None,
            decode_dp_rank: None,
            __version__: DEFAULT_REQUEST_LOG_VERSION.to_string(),
        }];

        let parquet_bytes = serialize_to_parquet(&entries).expect("Should serialize to parquet");
        let reader = ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::from(parquet_bytes))
            .unwrap()
            .build()
            .unwrap();
        let batch = reader.into_iter().next().unwrap().unwrap();
        let col = batch
            .column_by_name("service_tier")
            .expect("service_tier column should exist");
        let arr = col
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("service_tier should be a string array");
        assert_eq!(arr.value(0), "fast");
        for (field, expected) in [("component_name", "frontend"), ("component_id", "12345")] {
            let arr = batch
                .column_by_name(field)
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            assert_eq!(arr.value(0), expected);
        }
    }

    fn find_metric_snapshot<'a>(
        snapshot: &'a RequestMetricsSnapshot,
        name: &str,
        feature: Option<&str>,
    ) -> &'a crate::MetricSnapshot {
        &snapshot
            .series
            .iter()
            .find(|series| {
                series.name == name
                    && match feature {
                        Some(feature) => {
                            series.labels.get("feature").map(String::as_str) == Some(feature)
                        }
                        None => !series.labels.contains_key("feature"),
                    }
            })
            .unwrap()
            .snapshot
    }

    /// Find the plain (no `service_tier`) series for `name`. For distribution
    /// metrics this is the `feature="all"` series; for rate/ratio metrics it is
    /// the label-less series. An optional `feature` filter narrows the match
    /// for metrics that carry a `feature` label.
    fn find_plain_series<'a>(
        snapshot: &'a RequestMetricsSnapshot,
        name: &str,
        feature: Option<&str>,
    ) -> Option<&'a crate::MetricSnapshot> {
        snapshot.series.iter().find_map(|s| {
            if s.name != name
                || s.labels.contains_key("service_tier")
                || s.labels.contains_key("dp_rank")
            {
                return None;
            }
            match feature {
                Some(f) => {
                    if s.labels.get("feature").map(String::as_str) == Some(f) {
                        Some(&s.snapshot)
                    } else {
                        None
                    }
                }
                None => {
                    if !s.labels.contains_key("feature") {
                        Some(&s.snapshot)
                    } else {
                        None
                    }
                }
            }
        })
    }

    /// Find the `service_tier=<tier>` series for `name`. Distribution metrics
    /// also carry `feature="all"`; rate/ratio metrics carry only `service_tier`.
    /// An optional `feature` filter narrows the match for distribution metrics.
    fn find_tiered_series<'a>(
        snapshot: &'a RequestMetricsSnapshot,
        name: &str,
        tier: &str,
        feature: Option<&str>,
    ) -> Option<&'a crate::MetricSnapshot> {
        snapshot.series.iter().find_map(|s| {
            if s.name != name
                || s.labels.get("service_tier").map(String::as_str) != Some(tier)
                || s.labels.contains_key("dp_rank")
            {
                return None;
            }
            match feature {
                Some(f) => {
                    if s.labels.get("feature").map(String::as_str) == Some(f) {
                        Some(&s.snapshot)
                    } else {
                        None
                    }
                }
                None => {
                    if !s.labels.contains_key("feature") {
                        Some(&s.snapshot)
                    } else {
                        None
                    }
                }
            }
        })
    }

    /// Find a series for `name` matching all given `(label, value) pairs. An
    /// optional `feature` filter narrows the match for distribution metrics.
    fn find_labeled_series<'a>(
        snapshot: &'a RequestMetricsSnapshot,
        name: &str,
        labels: &[(&str, &str)],
        feature: Option<&str>,
    ) -> Option<&'a crate::MetricSnapshot> {
        snapshot.series.iter().find_map(|s| {
            if s.name != name {
                return None;
            }
            if !labels
                .iter()
                .all(|(k, v)| s.labels.get(*k).map(String::as_str) == Some(*v))
            {
                return None;
            }
            match feature {
                Some(f) => {
                    if s.labels.get("feature").map(String::as_str) == Some(f) {
                        Some(&s.snapshot)
                    } else {
                        None
                    }
                }
                None => {
                    if !s.labels.contains_key("feature") {
                        Some(&s.snapshot)
                    } else {
                        None
                    }
                }
            }
        })
    }

    #[test]
    fn service_tier_fast_produces_tier_labeled_all_series() {
        let factory = RequestMetricsFactory::new(RequestMetricsOptions::default()).unwrap();
        let mut req = factory
            .new_request_labeled(
                Some(&[1, 2, 3, 4]),
                4,
                RequestFeatures::NONE,
                None,
                None,
                Some(ServiceTier::Fast),
                None,
            )
            .unwrap();
        req.record_tokens(&[10, 11, 12], Some(2), true, None);
        // Sleep so ITL (inter-token latency) is recorded as a non-zero value.
        std::thread::sleep(std::time::Duration::from_millis(5));
        req.record_tokens(&[13, 14], None, true, None);
        req.success();

        let snapshot = factory.snapshot();

        // (metric name, feature filter) for each major metric. Distribution
        // metrics carry `feature="all"`; rate/ratio metrics have no feature label.
        let major_metrics: [(&str, Option<&str>); 6] = [
            ("request_metrics_ttft_ms", Some("all")),
            ("request_metrics_time_to_fifth_token_ms", Some("all")),
            ("request_metrics_itl_ms", Some("all")),
            ("request_metrics_kv_cache_hit_rate", None),
            ("request_metrics_request", None),
            ("request_metrics_successful_request", None),
        ];

        // The tier-labeled series must exist for every major metric, and the
        // plain unlabelled series must still exist alongside it.
        for (name, feature) in major_metrics {
            assert!(
                find_tiered_series(&snapshot, name, "fast", feature).is_some(),
                "tier-labeled series for {name} should exist"
            );
            assert!(
                find_plain_series(&snapshot, name, feature).is_some(),
                "plain unlabelled series for {name} should still exist"
            );
        }

        // Tiered series must carry the same recorded values as the unlabelled
        // series for the distribution/ratio major metrics.
        let tiered_ttft =
            find_tiered_series(&snapshot, "request_metrics_ttft_ms", "fast", Some("all")).unwrap();
        let plain_ttft =
            find_plain_series(&snapshot, "request_metrics_ttft_ms", Some("all")).unwrap();
        assert_eq!(
            tiered_ttft.quantiles.get("0.5"),
            plain_ttft.quantiles.get("0.5"),
            "tiered and plain ttft p50 should match"
        );

        let tiered_t5 = find_tiered_series(
            &snapshot,
            "request_metrics_time_to_fifth_token_ms",
            "fast",
            Some("all"),
        )
        .unwrap();
        let plain_t5 = find_plain_series(
            &snapshot,
            "request_metrics_time_to_fifth_token_ms",
            Some("all"),
        )
        .unwrap();
        assert_eq!(
            tiered_t5.quantiles.get("0.5"),
            plain_t5.quantiles.get("0.5"),
            "tiered and plain time_to_fifth_token p50 should match"
        );

        let tiered_itl =
            find_tiered_series(&snapshot, "request_metrics_itl_ms", "fast", Some("all")).unwrap();
        let plain_itl =
            find_plain_series(&snapshot, "request_metrics_itl_ms", Some("all")).unwrap();
        assert_eq!(
            tiered_itl.average, plain_itl.average,
            "tiered and plain itl average should match"
        );

        let tiered_kv =
            find_tiered_series(&snapshot, "request_metrics_kv_cache_hit_rate", "fast", None)
                .unwrap();
        let plain_kv =
            find_plain_series(&snapshot, "request_metrics_kv_cache_hit_rate", None).unwrap();
        assert_eq!(
            tiered_kv.ratio, plain_kv.ratio,
            "tiered and plain kv cache hit rate ratio should match"
        );

        // The tier-labeled request rate series should be positive.
        let tiered_request =
            find_tiered_series(&snapshot, "request_metrics_request", "fast", None).unwrap();
        assert!(
            tiered_request.rate_per_second.unwrap_or(0.0) > 0.0,
            "tiered request rate should be positive"
        );
        let tiered_successful = find_tiered_series(
            &snapshot,
            "request_metrics_successful_request",
            "fast",
            None,
        )
        .unwrap();
        assert!(
            tiered_successful.rate_per_second.unwrap_or(0.0) > 0.0,
            "tiered successful_request rate should be positive"
        );
    }

    #[test]
    fn service_tier_none_produces_no_tier_labeled_series() {
        let factory = RequestMetricsFactory::new(RequestMetricsOptions::default()).unwrap();
        let mut req = factory
            .new_request_labeled(
                Some(&[1, 2, 3, 4]),
                4,
                RequestFeatures::NONE,
                None,
                None,
                None,
                None,
            )
            .unwrap();
        req.record_tokens(&[10, 11, 12], Some(2), true, None);
        req.success();

        let snapshot = factory.snapshot();
        let has_tier_label = snapshot
            .series
            .iter()
            .any(|s| s.labels.contains_key("service_tier"));
        assert!(
            !has_tier_label,
            "no series should carry a service_tier label when tier is None"
        );
    }

    #[test]
    fn service_tier_unused_tiers_never_appear() {
        let factory = RequestMetricsFactory::new(RequestMetricsOptions::default()).unwrap();
        // Only the Fast tier is used; Priority/Default/Flex must never appear.
        let mut req = factory
            .new_request_labeled(
                Some(&[1, 2, 3, 4]),
                4,
                RequestFeatures::NONE,
                None,
                None,
                Some(ServiceTier::Fast),
                None,
            )
            .unwrap();
        req.record_tokens(&[10, 11, 12], Some(2), true, None);
        req.success();

        let snapshot = factory.snapshot();
        let major_metrics: [(&str, Option<&str>); 6] = [
            ("request_metrics_ttft_ms", Some("all")),
            ("request_metrics_time_to_fifth_token_ms", Some("all")),
            ("request_metrics_itl_ms", Some("all")),
            ("request_metrics_kv_cache_hit_rate", None),
            ("request_metrics_request", None),
            ("request_metrics_successful_request", None),
        ];
        for unused_tier in ["priority", "default", "flex"] {
            for (name, feature) in major_metrics {
                assert!(
                    find_tiered_series(&snapshot, name, unused_tier, feature).is_none(),
                    "unused tier {unused_tier:?} should not produce a series for {name}"
                );
            }
        }
    }

    #[test]
    fn dp_rank_only_produces_rank_labeled_series() {
        let factory = RequestMetricsFactory::new(RequestMetricsOptions::default()).unwrap();
        let mut req = factory
            .new_request_labeled(
                Some(&[1, 2, 3, 4]),
                4,
                RequestFeatures::NONE,
                None,
                None,
                None,
                Some(3),
            )
            .unwrap();
        req.record_tokens(&[10, 11, 12], Some(2), true, None);
        std::thread::sleep(std::time::Duration::from_millis(5));
        req.record_tokens(&[13, 14], None, true, None);
        req.success();

        let snapshot = factory.snapshot();

        // (metric name, feature filter) for each major metric.
        let major_metrics: [(&str, Option<&str>); 6] = [
            ("request_metrics_ttft_ms", Some("all")),
            ("request_metrics_time_to_fifth_token_ms", Some("all")),
            ("request_metrics_itl_ms", Some("all")),
            ("request_metrics_kv_cache_hit_rate", None),
            ("request_metrics_request", None),
            ("request_metrics_successful_request", None),
        ];

        // The rank-labeled series must exist for every major metric, and the
        // plain unlabelled series must still exist alongside it.
        for (name, feature) in major_metrics {
            assert!(
                find_labeled_series(&snapshot, name, &[("dp_rank", "3")], feature).is_some(),
                "rank-labeled series for {name} should exist"
            );
            assert!(
                find_plain_series(&snapshot, name, feature).is_some(),
                "plain unlabelled series for {name} should still exist"
            );
        }

        // No series should carry a service_tier label (only dp_rank was given).
        assert!(
            !snapshot
                .series
                .iter()
                .any(|s| s.labels.contains_key("service_tier")),
            "no series should carry a service_tier label when tier is None"
        );

        // Tiered values must match the plain values for distribution/ratio metrics.
        let ranked_ttft = find_labeled_series(
            &snapshot,
            "request_metrics_ttft_ms",
            &[("dp_rank", "3")],
            Some("all"),
        )
        .unwrap();
        let plain_ttft =
            find_plain_series(&snapshot, "request_metrics_ttft_ms", Some("all")).unwrap();
        assert_eq!(
            ranked_ttft.quantiles.get("0.5"),
            plain_ttft.quantiles.get("0.5"),
            "ranked and plain ttft p50 should match"
        );

        let ranked_itl = find_labeled_series(
            &snapshot,
            "request_metrics_itl_ms",
            &[("dp_rank", "3")],
            Some("all"),
        )
        .unwrap();
        let plain_itl =
            find_plain_series(&snapshot, "request_metrics_itl_ms", Some("all")).unwrap();
        assert_eq!(
            ranked_itl.average, plain_itl.average,
            "ranked and plain itl average should match"
        );

        let ranked_request = find_labeled_series(
            &snapshot,
            "request_metrics_request",
            &[("dp_rank", "3")],
            None,
        )
        .unwrap();
        assert!(
            ranked_request.rate_per_second.unwrap_or(0.0) > 0.0,
            "ranked request rate should be positive"
        );
    }

    #[test]
    fn dp_rank_and_service_tier_produce_independent_series() {
        let factory = RequestMetricsFactory::new(RequestMetricsOptions::default()).unwrap();
        let mut req = factory
            .new_request_labeled(
                Some(&[1, 2, 3, 4]),
                4,
                RequestFeatures::NONE,
                None,
                None,
                Some(ServiceTier::Priority),
                Some(7),
            )
            .unwrap();
        req.record_tokens(&[10, 11, 12], Some(2), true, None);
        req.success();

        let snapshot = factory.snapshot();

        let major_metrics: [(&str, Option<&str>); 6] = [
            ("request_metrics_ttft_ms", Some("all")),
            ("request_metrics_time_to_fifth_token_ms", Some("all")),
            ("request_metrics_itl_ms", Some("all")),
            ("request_metrics_kv_cache_hit_rate", None),
            ("request_metrics_request", None),
            ("request_metrics_successful_request", None),
        ];

        // A tier-only series (service_tier="priority", NO dp_rank) must exist.
        for (name, feature) in major_metrics {
            let found = snapshot.series.iter().any(|s| {
                s.name == name
                    && s.labels.get("service_tier").map(String::as_str) == Some("priority")
                    && !s.labels.contains_key("dp_rank")
                    && match feature {
                        Some(f) => s.labels.get("feature").map(String::as_str) == Some(f),
                        None => !s.labels.contains_key("feature"),
                    }
            });
            assert!(found, "tier-only series for {name} should exist");
        }

        // A rank-only series (dp_rank="7", NO service_tier) must exist.
        for (name, feature) in major_metrics {
            let found = snapshot.series.iter().any(|s| {
                s.name == name
                    && s.labels.get("dp_rank").map(String::as_str) == Some("7")
                    && !s.labels.contains_key("service_tier")
                    && match feature {
                        Some(f) => s.labels.get("feature").map(String::as_str) == Some(f),
                        None => !s.labels.contains_key("feature"),
                    }
            });
            assert!(found, "rank-only series for {name} should exist");
        }

        // No combined series (BOTH service_tier and dp_rank) should exist.
        for (name, _feature) in major_metrics {
            let combined = snapshot.series.iter().any(|s| {
                s.name == name
                    && s.labels.contains_key("service_tier")
                    && s.labels.contains_key("dp_rank")
            });
            assert!(
                !combined,
                "no combined service_tier×dp_rank series for {name} should exist"
            );
        }
    }

    #[test]
    fn dp_rank_above_max_normalized_to_none() {
        let factory = RequestMetricsFactory::new(RequestMetricsOptions::default()).unwrap();
        // 31 is the last valid rank; 32 and 33 normalize to None.
        for rank in [32u8, 33] {
            let mut req = factory
                .new_request_labeled(
                    Some(&[1, 2, 3, 4]),
                    4,
                    RequestFeatures::NONE,
                    None,
                    None,
                    None,
                    Some(rank),
                )
                .unwrap();
            req.record_tokens(&[10, 11], Some(2), true, None);
            req.success();
        }
        let snapshot = factory.snapshot();
        assert!(
            !snapshot
                .series
                .iter()
                .any(|s| s.labels.contains_key("dp_rank")),
            "ranks >= 32 should not produce dp_rank-labeled series"
        );
    }

    #[test]
    fn dp_rank_max_valid_creates_series() {
        let factory = RequestMetricsFactory::new(RequestMetricsOptions::default()).unwrap();
        let mut req = factory
            .new_request_labeled(
                Some(&[1, 2, 3, 4]),
                4,
                RequestFeatures::NONE,
                None,
                None,
                None,
                Some(31),
            )
            .unwrap();
        req.record_tokens(&[10, 11], Some(2), true, None);
        req.success();
        let snapshot = factory.snapshot();
        assert!(
            find_labeled_series(
                &snapshot,
                "request_metrics_request",
                &[("dp_rank", "31")],
                None
            )
            .is_some(),
            "dp_rank=31 should produce a rank-labeled series"
        );
    }

    #[test]
    fn dp_rank_invalid_with_tier_still_creates_tier_series() {
        let factory = RequestMetricsFactory::new(RequestMetricsOptions::default()).unwrap();
        let mut req = factory
            .new_request_labeled(
                Some(&[1, 2, 3, 4]),
                4,
                RequestFeatures::NONE,
                None,
                None,
                Some(ServiceTier::Priority),
                Some(33),
            )
            .unwrap();
        req.record_tokens(&[10, 11], Some(2), true, None);
        req.success();
        let snapshot = factory.snapshot();
        // Tier series should exist.
        assert!(
            find_labeled_series(
                &snapshot,
                "request_metrics_request",
                &[("service_tier", "priority")],
                None
            )
            .is_some(),
            "service_tier series should exist even when dp_rank is invalid"
        );
        // No rank series should exist.
        assert!(
            !snapshot
                .series
                .iter()
                .any(|s| s.labels.contains_key("dp_rank")),
            "invalid dp_rank should not produce a rank-labeled series"
        );
    }

    #[test]
    fn dp_rank_none_produces_no_rank_labeled_series() {
        let factory = RequestMetricsFactory::new(RequestMetricsOptions::default()).unwrap();
        let mut req = factory
            .new_request_labeled(
                Some(&[1, 2, 3, 4]),
                4,
                RequestFeatures::NONE,
                None,
                None,
                None,
                None,
            )
            .unwrap();
        req.record_tokens(&[10, 11, 12], Some(2), true, None);
        req.success();

        let snapshot = factory.snapshot();
        assert!(
            !snapshot
                .series
                .iter()
                .any(|s| s.labels.contains_key("dp_rank")),
            "no series should carry a dp_rank label when rank is None"
        );
    }

    #[test]
    fn dp_rank_unused_ranks_never_appear() {
        let factory = RequestMetricsFactory::new(RequestMetricsOptions::default()).unwrap();
        let mut req = factory
            .new_request_labeled(
                Some(&[1, 2, 3, 4]),
                4,
                RequestFeatures::NONE,
                None,
                None,
                None,
                Some(0),
            )
            .unwrap();
        req.record_tokens(&[10, 11, 12], Some(2), true, None);
        req.success();

        let snapshot = factory.snapshot();
        let major_metrics: [(&str, Option<&str>); 6] = [
            ("request_metrics_ttft_ms", Some("all")),
            ("request_metrics_time_to_fifth_token_ms", Some("all")),
            ("request_metrics_itl_ms", Some("all")),
            ("request_metrics_kv_cache_hit_rate", None),
            ("request_metrics_request", None),
            ("request_metrics_successful_request", None),
        ];
        for unused_rank in 1..=31 {
            for (name, feature) in major_metrics {
                assert!(
                    find_labeled_series(
                        &snapshot,
                        name,
                        &[("dp_rank", &unused_rank.to_string())],
                        feature
                    )
                    .is_none(),
                    "unused rank {unused_rank} should not produce a series for {name}"
                );
            }
        }
    }

    /// Stress test: many threads concurrently create requests with mixed
    /// tiers and ranks, record tokens, and call success. This verifies that
    /// the handle caches and registry do not deadlock under contention.
    #[test]
    fn concurrent_mixed_tiers_ranks_no_deadlock() {
        use std::sync::Barrier;
        use std::thread;

        let factory =
            Arc::new(RequestMetricsFactory::new(RequestMetricsOptions::default()).unwrap());
        let threads = 8;
        let requests_per_thread = 50;
        let barrier = Arc::new(Barrier::new(threads));

        let handles: Vec<_> = (0..threads)
            .map(|tid| {
                let factory = factory.clone();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    barrier.wait();
                    for i in 0..requests_per_thread {
                        let tier = match i % 4 {
                            0 => Some(ServiceTier::Fast),
                            1 => Some(ServiceTier::Priority),
                            2 => Some(ServiceTier::Default),
                            _ => Some(ServiceTier::Flex),
                        };
                        let rank = Some((tid as u8 + i as u8) % 32);
                        let input = (0..16u32).collect::<Vec<_>>();
                        let mut req = factory
                            .new_request_labeled(
                                Some(input.as_slice()),
                                input.len(),
                                RequestFeatures::NONE,
                                None,
                                None,
                                tier,
                                rank,
                            )
                            .unwrap();
                        req.record_tokens(&[10, 11, 12, 13, 14], Some(4), true, None);
                        req.record_tokens(&[15, 16], None, true, None);
                        req.success();
                    }
                })
            })
            .collect();

        // If any thread deadlocks, join will hang and the test will time out.
        for h in handles {
            h.join().expect("thread panicked");
        }

        // Verify we got labeled series for multiple tiers and ranks.
        let snapshot = factory.snapshot();
        assert!(
            snapshot
                .series
                .iter()
                .any(|s| s.labels.contains_key("service_tier")),
            "tier-labeled series should exist after concurrent requests"
        );
        assert!(
            snapshot
                .series
                .iter()
                .any(|s| s.labels.contains_key("dp_rank")),
            "rank-labeled series should exist after concurrent requests"
        );
    }

    const RAW_B10_REQUEST_ID: &str = "wp5ab0192f1e8a27d4c5b8e3f2a1d0c9b8e7f6";

    fn routed(prefill: u64, decode: Option<u64>) -> RoutedWorkers {
        RoutedWorkers {
            prefill_worker_id: Some(prefill),
            prefill_dp_rank: Some(0),
            decode_worker_id: decode,
            decode_dp_rank: decode.map(|_| 3),
        }
    }

    #[test]
    fn hash_b10_request_id_is_domain_separated_hex_digest() {
        let digest = hash_b10_request_id(RAW_B10_REQUEST_ID);
        assert_eq!(digest.len(), 64);
        assert!(digest.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(digest, hash_b10_request_id(RAW_B10_REQUEST_ID));
        assert_ne!(digest, hash_session_id(RAW_B10_REQUEST_ID));
        assert_ne!(
            digest,
            blake3::hash(RAW_B10_REQUEST_ID.as_bytes())
                .to_hex()
                .to_string()
        );
    }

    #[test]
    fn hash_b10_request_id_matches_golden_digest() {
        // The digest is a permanent join key: blake3(b"pm-b10-request-id-v1" || raw id).
        // Pinned to a value computed independently; changing the recipe must fail here.
        assert_eq!(
            hash_b10_request_id(RAW_B10_REQUEST_ID),
            "a619fd8ae35a3cd121bfcbd90349aac1588a26034f868965d546e8650f472594"
        );
    }

    #[test]
    fn request_log_rows_carry_hashed_b10_id_and_routed_workers_on_every_terminal_path() {
        let tmp = std::env::temp_dir().join(format!(
            "pm-core-b10-test-{}-{}",
            unix_ms(SystemTime::now()),
            rand::random::<u32>()
        ));
        let counters;
        {
            let factory = RequestMetricsFactory::new(RequestMetricsOptions {
                trace: RequestTraceOptions {
                    request_log_enabled: true,
                    local_output_dir: Some(tmp.clone()),
                    request_log_batch_size: 1,
                    ..Default::default()
                },
                ..Default::default()
            })
            .unwrap();
            let new_request = |input_tokens: usize, b10_id: Option<&str>| {
                factory
                    .new_request_labeled(
                        None,
                        input_tokens,
                        RequestFeatures::NONE,
                        None,
                        b10_id,
                        None,
                        None,
                    )
                    .unwrap()
            };

            let mut succeeded = new_request(1, Some(RAW_B10_REQUEST_ID));
            succeeded.set_routed_workers(routed(11, Some(21)));
            succeeded.success();
            // Ignored: the row was already written.
            succeeded.set_routed_workers(routed(99, Some(99)));

            let mut canceled = new_request(2, Some(RAW_B10_REQUEST_ID));
            canceled.set_routed_workers(routed(12, None));
            canceled.cancel();

            let mut dropped = new_request(3, Some(RAW_B10_REQUEST_ID));
            dropped.set_routed_workers(routed(13, Some(23)));
            drop(dropped);

            new_request(4, None).success();
            new_request(5, Some("")).success();
            counters = Arc::clone(&factory.trace_factory.recorder.counters);
        }
        wait_for(&counters.rows_persisted, 5);

        let mut rows: HashMap<u64, serde_json::Value> = HashMap::new();
        for file in fs::read_dir(&tmp).unwrap() {
            let contents = fs::read_to_string(file.unwrap().path()).unwrap();
            assert!(!contents.contains(RAW_B10_REQUEST_ID));
            let batch: Vec<serde_json::Value> = serde_json::from_str(&contents).unwrap();
            for row in batch {
                rows.insert(row["input_tokens"].as_u64().unwrap(), row);
            }
        }
        let _ = fs::remove_dir_all(&tmp);
        assert_eq!(rows.len(), 5);

        let digest = serde_json::json!(hash_b10_request_id(RAW_B10_REQUEST_ID));
        let worker_columns = |row: &serde_json::Value| {
            [
                "prefill_worker_id",
                "prefill_dp_rank",
                "decode_worker_id",
                "decode_dp_rank",
            ]
            .map(|key| row[key].clone())
        };
        let json = |v: Option<u64>| serde_json::json!(v);

        assert_eq!(rows[&1]["b10_request_id"], digest);
        assert_eq!(rows[&1]["request_canceled"], false);
        assert_eq!(
            worker_columns(&rows[&1]),
            [json(Some(11)), json(Some(0)), json(Some(21)), json(Some(3))]
        );

        assert_eq!(rows[&2]["b10_request_id"], digest);
        assert_eq!(rows[&2]["request_canceled"], true);
        assert_eq!(
            worker_columns(&rows[&2]),
            [json(Some(12)), json(Some(0)), json(None), json(None)]
        );

        assert_eq!(rows[&3]["b10_request_id"], digest);
        assert_eq!(rows[&3]["request_canceled"], true);
        assert_eq!(
            worker_columns(&rows[&3]),
            [json(Some(13)), json(Some(0)), json(Some(23)), json(Some(3))]
        );

        assert!(rows[&4]["b10_request_id"].is_null());
        assert_eq!(
            worker_columns(&rows[&4]),
            [json(None), json(None), json(None), json(None)]
        );

        assert!(rows[&5]["b10_request_id"].is_null());
    }

    #[test]
    fn parquet_rows_carry_hashed_b10_id_and_nullable_worker_columns() {
        use arrow::array::{Array, StringArray, UInt32Array, UInt64Array};
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let factory = RequestTraceFactory::new(RequestTraceOptions::default()).unwrap();
        // One full block each: serialize_to_parquet rejects batches with no hashes at all.
        let block: Vec<u32> = (0..DEFAULT_BLOCK_SIZE as u32).collect();
        let mut routed_trace =
            factory.create_trace_base(block.len(), None, Some(RAW_B10_REQUEST_ID), vec![], None);
        routed_trace.append_tokens(&block);
        routed_trace.set_routed_workers(routed(11, Some(21)));
        let mut unrouted_trace = factory.create_trace_base(block.len(), None, None, vec![], None);
        unrouted_trace.append_tokens(&block);
        let entries = vec![
            routed_trace.build_log_entry(false),
            unrouted_trace.build_log_entry(true),
        ];
        routed_trace.success();

        let bytes = serialize_to_parquet(&entries).unwrap();
        let batch = ParquetRecordBatchReaderBuilder::try_new(Bytes::from(bytes))
            .unwrap()
            .build()
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        let column = |name: &str| batch.column_by_name(name).unwrap().clone();

        let b10 = column("b10_request_id");
        let b10 = b10.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(b10.value(0), hash_b10_request_id(RAW_B10_REQUEST_ID));
        assert!(b10.is_null(1));

        for (name, expected) in [("prefill_worker_id", 11), ("decode_worker_id", 21)] {
            let ids = column(name);
            let ids = ids.as_any().downcast_ref::<UInt64Array>().unwrap();
            assert_eq!(ids.value(0), expected, "{name}");
            assert!(ids.is_null(1), "{name}");
        }
        for (name, expected) in [("prefill_dp_rank", 0), ("decode_dp_rank", 3)] {
            let ranks = column(name);
            let ranks = ranks.as_any().downcast_ref::<UInt32Array>().unwrap();
            assert_eq!(ranks.value(0), expected, "{name}");
            assert!(ranks.is_null(1), "{name}");
        }
        let schema = batch.schema();
        assert_eq!(schema.fields().last().unwrap().name(), "__version__");
    }

    /// The log entry with wall-clock fields zeroed, so two traces compare by content.
    fn comparable_entry(trace: &RequestTrace) -> serde_json::Value {
        let mut entry = serde_json::to_value(trace.build_log_entry(false)).unwrap();
        for key in [
            "timestamp_start_unix_ms",
            "duration_e2e_ms",
            "duration_ttft_ms",
        ] {
            entry[key] = serde_json::json!(0);
        }
        entry
    }

    #[test]
    fn new_request_labeled_trace_matches_new_trace_and_new_trace_with_count() {
        let factory = RequestMetricsFactory::new(RequestMetricsOptions {
            trace: logged(RequestTraceOptions {
                request_log_hash_salt: Some("salt".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        })
        .unwrap();
        let features = RequestFeatures::TOOLS | RequestFeatures::IMAGE;
        // 150 tokens: two full blocks plus a pending partial block that output completes.
        let input: Vec<u32> = (0..150).collect();
        let output: Vec<u32> = (1000..1100).collect();

        let mut labeled = factory
            .new_request_labeled(
                Some(&input),
                0, // ignored when ids are given
                features,
                Some("session"),
                None,
                Some(ServiceTier::Fast),
                None,
            )
            .unwrap();
        let mut direct = factory.trace_factory.new_trace(
            &input,
            Some("session"),
            features.to_vec(),
            Some(ServiceTier::Fast),
        );
        labeled.trace.record_output_tokens(&output, true, None);
        direct.record_output_tokens(&output, true, None);
        assert_eq!(labeled.trace.input_tokens(), 150);
        assert_eq!(labeled.total_hashes(), direct.total_hashes());
        assert_eq!(labeled.total_hashes().len(), 3);
        assert_eq!(comparable_entry(&labeled.trace), comparable_entry(&direct));

        let counted = factory
            .new_request_labeled(None, 150, features, Some("session"), None, None, None)
            .unwrap();
        let direct_count = factory.trace_factory.new_trace_with_count(
            150,
            Some("session"),
            features.to_vec(),
            None,
        );
        assert!(counted.total_hashes().is_empty());
        assert_eq!(
            comparable_entry(&counted.trace),
            comparable_entry(&direct_count)
        );
    }
}
