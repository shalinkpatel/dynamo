use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

const MAX_BUCKETS: usize = 1 << 20;

#[derive(Clone)]
pub struct AtomicHistogram {
    inner: Arc<AtomicHistogramState>,
}

struct AtomicHistogramState {
    indexing: HistogramIndexing,
    buckets: Vec<AtomicU64>,
    count: AtomicU64,
    sum_bits: AtomicU64,
}

enum HistogramIndexing {
    Vm {
        min_exponent: i32,
        buckets_per_decimal: u32,
        buckets_count: usize,
    },
    ExplicitUpperBounds(Vec<f64>),
}

impl AtomicHistogram {
    pub fn new(upper_bounds: Vec<f64>) -> anyhow::Result<Self> {
        Self::build_from_upper_bounds(upper_bounds)
    }

    pub fn from_vm(
        min_exponent: i32,
        max_exponent: i32,
        buckets_per_decimal: u32,
    ) -> anyhow::Result<Self> {
        let (buckets_count, buckets_len) =
            vm_bucket_config(min_exponent, max_exponent, buckets_per_decimal)?;
        let buckets = (0..buckets_len).map(|_| AtomicU64::new(0)).collect();

        Ok(Self {
            inner: Arc::new(AtomicHistogramState {
                indexing: HistogramIndexing::Vm {
                    min_exponent,
                    buckets_per_decimal,
                    buckets_count,
                },
                buckets,
                count: AtomicU64::new(0),
                sum_bits: AtomicU64::new(0.0f64.to_bits()),
            }),
        })
    }

    pub fn from_ranges(bucket_ranges: Vec<(f64, f64)>) -> anyhow::Result<Self> {
        let mut upper_bounds = Vec::with_capacity(bucket_ranges.len());
        let mut prev_max = 0.0;
        if bucket_ranges.is_empty() || bucket_ranges[0].0 != 0.0 {
            return Err(anyhow::Error::msg(
                "bucket_ranges must start at 0 and be contiguous",
            ));
        }
        for (idx, (min, max)) in bucket_ranges.into_iter().enumerate() {
            if min != prev_max {
                return Err(anyhow::Error::msg(format!(
                    "bucket_ranges[{idx}] min ({min}) does not match previous max ({prev_max})"
                )));
            }
            if !(max.is_finite() || max == f64::INFINITY) {
                return Err(anyhow::Error::msg(format!(
                    "bucket_ranges[{idx}] max must be finite or +inf"
                )));
            }
            if max <= min {
                return Err(anyhow::Error::msg(format!(
                    "bucket_ranges[{idx}] must satisfy min < max"
                )));
            }
            upper_bounds.push(max);
            prev_max = max;
        }
        Self::build_from_upper_bounds(upper_bounds)
    }

    pub fn observe(&self, amount: f64) {
        self.observe_count(amount, 1);
    }

    pub fn observe_many(&self, amount: f64, count: i64) {
        if count <= 0 {
            return;
        }
        self.observe_count(amount, count as u64);
    }

    pub fn snapshot(&self) -> (Vec<u64>, u64, f64) {
        let buckets = self
            .inner
            .buckets
            .iter()
            .map(|v| v.load(Ordering::Relaxed))
            .collect::<Vec<_>>();
        let count = self.inner.count.load(Ordering::Relaxed);
        let sum = f64::from_bits(self.inner.sum_bits.load(Ordering::Relaxed));
        (buckets, count, sum)
    }

    pub fn is_shared(&self) -> bool {
        Arc::strong_count(&self.inner) > 1
    }

    pub fn bucket_len(&self) -> usize {
        self.inner.buckets.len()
    }
}

impl AtomicHistogram {
    pub fn observe_count(&self, amount: f64, count: u64) {
        if !amount.is_finite() || amount < 0.0 || count == 0 {
            return;
        }

        let bucket_index = if amount == 0.0 {
            0
        } else {
            match &self.inner.indexing {
                HistogramIndexing::Vm {
                    min_exponent,
                    buckets_per_decimal,
                    buckets_count,
                } => vm_bucket_index(amount, *min_exponent, *buckets_per_decimal, *buckets_count),
                HistogramIndexing::ExplicitUpperBounds(upper_bounds) => {
                    explicit_bucket_index(amount, upper_bounds.as_slice())
                }
            }
        };
        self.inner.buckets[bucket_index].fetch_add(count, Ordering::Relaxed);
        self.inner.count.fetch_add(count, Ordering::Relaxed);
        self.add_sum(amount * count as f64);
    }

    fn add_sum(&self, delta: f64) {
        let mut current = self.inner.sum_bits.load(Ordering::Relaxed);
        loop {
            let next = (f64::from_bits(current) + delta).to_bits();
            match self.inner.sum_bits.compare_exchange_weak(
                current,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
    }

    fn build_from_upper_bounds(upper_bounds: Vec<f64>) -> anyhow::Result<Self> {
        if upper_bounds.is_empty() {
            return Err(anyhow::Error::msg("upper_bounds must not be empty"));
        }
        if upper_bounds[0] <= 0.0 {
            return Err(anyhow::Error::msg(
                "first upper bound must be greater than 0",
            ));
        }

        for (idx, upper) in upper_bounds.iter().copied().enumerate() {
            if !upper.is_finite() && upper != f64::INFINITY {
                return Err(anyhow::Error::msg(format!(
                    "upper_bounds[{idx}] must be finite or +inf"
                )));
            }
            if idx > 0 && upper <= upper_bounds[idx - 1] {
                return Err(anyhow::Error::msg(format!(
                    "upper_bounds must be strictly increasing; index {idx} breaks ordering"
                )));
            }
        }

        if upper_bounds.last().copied() != Some(f64::INFINITY) {
            return Err(anyhow::Error::msg("upper_bounds must end with +inf"));
        }

        let bucket_len = upper_bounds.len();
        if bucket_len > MAX_BUCKETS {
            return Err(anyhow::Error::msg(
                "requested bucket count exceeds allowed maximum",
            ));
        }

        let buckets = (0..bucket_len).map(|_| AtomicU64::new(0)).collect();
        Ok(Self {
            inner: Arc::new(AtomicHistogramState {
                indexing: HistogramIndexing::ExplicitUpperBounds(upper_bounds),
                buckets,
                count: AtomicU64::new(0),
                sum_bits: AtomicU64::new(0.0f64.to_bits()),
            }),
        })
    }
}

/// Integer counter shared by stream recorders and scrape-time exporters.
#[derive(Clone, Default)]
pub struct AtomicCounter {
    value: Arc<AtomicU64>,
}

impl AtomicCounter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn inc(&self, amount: u64) {
        self.value.fetch_add(amount, Ordering::Relaxed);
    }

    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }
}

/// One native call updates all metrics for a caller-timed output chunk.
pub struct StreamChunkMetrics {
    histograms: Option<[AtomicHistogram; 3]>,
    output_tokens: AtomicCounter,
}

impl StreamChunkMetrics {
    pub fn new(
        ttft: &AtomicHistogram,
        tpot: &AtomicHistogram,
        itl: &AtomicHistogram,
        output_tokens: &AtomicCounter,
    ) -> Self {
        Self {
            histograms: Some([ttft.clone(), tpot.clone(), itl.clone()]),
            output_tokens: output_tokens.clone(),
        }
    }

    pub fn record(
        &self,
        token_count: u64,
        ttft_seconds: Option<f64>,
        chunk_seconds: Option<f64>,
    ) -> anyhow::Result<()> {
        let [ttft, tpot, itl] = self
            .histograms
            .as_ref()
            .ok_or_else(|| anyhow::Error::msg("StreamChunkMetrics is closed"))?;
        if let Some(seconds) = ttft_seconds.filter(|seconds| *seconds > 0.0) {
            ttft.observe_count(seconds, 1);
        }
        if let Some(seconds) = chunk_seconds.filter(|seconds| *seconds > 0.0 && token_count > 0) {
            let seconds_per_token = seconds / token_count as f64;
            tpot.observe_count(seconds_per_token, 1);
            itl.observe_count(seconds_per_token, token_count);
        }
        self.output_tokens.inc(token_count);
        Ok(())
    }

    pub fn close(&mut self) {
        self.histograms = None;
    }
}

fn vm_bucket_config(
    min_exponent: i32,
    max_exponent: i32,
    buckets_per_decimal: u32,
) -> anyhow::Result<(usize, usize)> {
    if max_exponent <= min_exponent {
        return Err(anyhow::Error::msg(
            "max_exponent must be greater than min_exponent",
        ));
    }
    if buckets_per_decimal == 0 {
        return Err(anyhow::Error::msg(
            "buckets_per_decimal must be greater than 0",
        ));
    }

    let span = (max_exponent - min_exponent) as u32;
    let raw_count = u64::from(span)
        .checked_mul(u64::from(buckets_per_decimal))
        .ok_or_else(|| anyhow::Error::msg("bucket count overflow"))?;
    let buckets_count =
        usize::try_from(raw_count).map_err(|_| anyhow::Error::msg("bucket count exceeds usize"))?;
    let buckets_len = buckets_count
        .checked_add(2)
        .ok_or_else(|| anyhow::Error::msg("bucket count overflow"))?;

    if buckets_len > MAX_BUCKETS {
        return Err(anyhow::Error::msg(
            "requested bucket count exceeds allowed maximum",
        ));
    }
    Ok((buckets_count, buckets_len))
}

fn vm_bucket_index(
    amount: f64,
    min_exponent: i32,
    buckets_per_decimal: u32,
    buckets_count: usize,
) -> usize {
    let bucket_idx = (amount.log10() - min_exponent as f64) * buckets_per_decimal as f64;
    if bucket_idx < 0.0 {
        return 0;
    }
    if bucket_idx >= buckets_count as f64 {
        return buckets_count + 1;
    }

    let mut idx = bucket_idx as usize;
    if bucket_idx == idx as f64 && idx > 0 {
        idx -= 1;
    }
    idx + 1
}

fn explicit_bucket_index(amount: f64, upper_bounds: &[f64]) -> usize {
    let idx = upper_bounds.partition_point(|upper| *upper < amount);
    idx.min(upper_bounds.len().saturating_sub(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn atomic_histogram_bucket_index_matches_bounds() {
        let min_exp = -4;
        let bpd = 12;
        let buckets_count = (10 - min_exp) as usize * bpd as usize;

        assert_eq!(vm_bucket_index(1e-9, min_exp, bpd, buckets_count), 0);
        assert_eq!(
            vm_bucket_index(1e20, min_exp, bpd, buckets_count),
            buckets_count + 1
        );
        assert_eq!(vm_bucket_index(1e-4, min_exp, bpd, buckets_count), 1);
    }

    #[test]
    fn atomic_histogram_observe_and_snapshot() {
        let hist = AtomicHistogram::from_vm(-4, 10, 12).expect("valid histogram");
        hist.observe(0.001);
        hist.observe(0.0);
        hist.observe(f64::NAN);

        let (buckets, count, sum) = hist.snapshot();
        assert_eq!(buckets.len(), 170);
        assert_eq!(count, 2);
        assert_eq!(buckets[0], 1);
        assert!((sum - 0.001).abs() < 1e-9);
    }

    #[test]
    fn atomic_histogram_from_ranges_indexes_correctly() {
        let hist = AtomicHistogram::new(vec![1e-3, 1e-2, f64::INFINITY]).expect("valid ranges");
        hist.observe(0.0);
        hist.observe(5e-4);
        hist.observe(2e-3);
        hist.observe(2.0);
        let (buckets, count, _) = hist.snapshot();
        assert_eq!(count, 4);
        assert_eq!(buckets, vec![2, 1, 1]);
    }

    #[test]
    fn atomic_histogram_observe_many_records_aggregate_count() {
        let hist = AtomicHistogram::new(vec![1e-3, 1e-2, f64::INFINITY]).expect("valid ranges");
        hist.observe_many(5e-4, 7);
        hist.observe_many(0.0, 3);
        hist.observe_many(2.0, 0);
        hist.observe_many(2.0, -1);
        hist.observe_many(-1.0, 9);

        let (buckets, count, sum) = hist.snapshot();
        assert_eq!(count, 10);
        assert_eq!(buckets, vec![10, 0, 0]);
        assert!((sum - 0.0035).abs() < 1e-9);
    }

    #[test]
    fn atomic_histogram_preserves_large_native_unit_sums() {
        let hist = AtomicHistogram::new(vec![1e9, 1e12, f64::INFINITY]).expect("valid ranges");
        hist.observe_many(25_000_000_000.0, 2);

        let (buckets, count, sum) = hist.snapshot();
        assert_eq!(count, 2);
        assert_eq!(buckets, vec![0, 2, 0]);
        assert_eq!(sum, 50_000_000_000.0);
    }

    fn python_vm_bucket_index(amount: f64, min_exponent: i32, buckets_per_decimal: u32) -> usize {
        let buckets_count = ((10 - min_exponent) as usize) * buckets_per_decimal as usize;
        let bucket_idx = (amount.log10() - min_exponent as f64) * buckets_per_decimal as f64;
        if bucket_idx < 0.0 {
            return 0;
        }
        if bucket_idx >= buckets_count as f64 {
            return buckets_count + 1;
        }
        let mut idx = bucket_idx as usize;
        if bucket_idx == idx as f64 && idx > 0 {
            idx -= 1;
        }
        idx + 1
    }

    #[test]
    fn atomic_histogram_vm_index_matches_vanilla_python_logic() {
        let min_exponent = -4;
        let max_exponent = 10;
        let buckets_per_decimal = 12u32;
        let buckets_count = (max_exponent - min_exponent) as usize * buckets_per_decimal as usize;

        // Deterministic sweep across many magnitudes.
        let mut values = Vec::new();
        for i in -8..18 {
            for frac in [1.0, 1.2, 1.5, 2.0, 3.3, 5.0, 7.7, 9.9] {
                values.push(frac * 10f64.powi(i));
            }
        }
        // Boundary values around exact powers of ten.
        for i in -8..18 {
            let p = 10f64.powi(i);
            values.push(p);
            values.push(p * (1.0 - 1e-12));
            values.push(p * (1.0 + 1e-12));
        }

        for amount in values {
            if amount <= 0.0 || !amount.is_finite() {
                continue;
            }
            let rust_idx =
                vm_bucket_index(amount, min_exponent, buckets_per_decimal, buckets_count);
            let py_idx = python_vm_bucket_index(amount, min_exponent, buckets_per_decimal);
            assert_eq!(
                rust_idx, py_idx,
                "bucket mismatch for amount={amount:e}: rust={rust_idx}, python={py_idx}"
            );
        }
    }

    fn python_range_index(amount: f64, ranges: &[(f64, f64)]) -> usize {
        for (idx, (_, max)) in ranges.iter().enumerate() {
            if amount <= *max {
                return idx;
            }
        }
        ranges.len().saturating_sub(1)
    }

    #[test]
    fn atomic_histogram_explicit_ranges_match_vanilla_range_scan() {
        let ranges = vec![
            (0.0, 1e-4),
            (1e-4, 1e-3),
            (1e-3, 1e-2),
            (1e-2, 1.0),
            (1.0, f64::INFINITY),
        ];
        let upper_bounds = ranges.iter().map(|(_, max)| *max).collect::<Vec<_>>();
        let values = vec![
            1e-9,
            5e-5,
            1e-4,
            1.0000000001e-4,
            7e-4,
            1e-3,
            4e-3,
            1e-2,
            0.5,
            1.0,
            1.5,
            1e6,
        ];
        for amount in values {
            let rust_idx = explicit_bucket_index(amount, &upper_bounds);
            let py_idx = python_range_index(amount, &ranges);
            assert_eq!(
                rust_idx, py_idx,
                "range bucket mismatch for amount={amount:e}: rust={rust_idx}, python={py_idx}"
            );
        }
    }

    #[test]
    fn stream_counts_live_tokens_and_weights_intervals() {
        let histogram = || AtomicHistogram::new(vec![1.0, f64::INFINITY]).unwrap();
        let (ttft, tpot, itl) = (histogram(), histogram(), histogram());
        let counter = AtomicCounter::default();
        let mut stream = StreamChunkMetrics::new(&ttft, &tpot, &itl, &counter);
        assert!(ttft.is_shared());
        stream.record(3, Some(0.5), None).unwrap();
        stream.record(4, None, Some(0.8)).unwrap();
        assert_eq!(counter.get(), 7);
        assert_eq!(ttft.snapshot().1, 1);
        assert_eq!(tpot.snapshot().1, 1);
        assert_eq!(itl.snapshot().1, 4);
        assert_eq!(itl.snapshot().2, 0.8);
        stream.close();
        assert!(!ttft.is_shared());
        assert!(stream.record(1, None, None).is_err());
        assert_eq!(counter.get(), 7);
    }
}
