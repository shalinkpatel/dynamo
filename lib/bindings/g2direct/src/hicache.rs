// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python surface for direct G2-to-G2 sharing of an engine-owned host tier.
//!
//! The consumer is SGLang's `HiCacheStorage` plugin interface: `publish` is
//! called from its backup thread, `lookup_prefix` from its prefetch thread and
//! `fetch` from its prefetch-IO thread, all concurrently. Every method here
//! releases the GIL for the duration of the call so those threads actually run
//! in parallel.

use crate::{get_current_tokio_handle, to_pyerr};

use dynamo_g2direct::hicache_g2::{
    HiCacheG2Config, HiCacheG2Direct, HiCachePageSegments, HiCacheVerify, hicache_g2_cache_key,
    hicache_g2_hash_threads,
};
use dynamo_runtime::{DistributedRuntime, Runtime};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;
use std::time::Duration;

pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHiCacheG2Direct>()?;
    Ok(())
}

/// Split a flat `(ptrs, sizes)` pair, as SGLang's `get_page_buffer_meta`
/// returns it, into one descriptor per page.
fn page_segments(
    ptrs: &[u64],
    sizes: &[u64],
    segments_per_page: usize,
) -> PyResult<Vec<HiCachePageSegments>> {
    HiCachePageSegments::split_flat(ptrs, sizes, segments_per_page)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyclass(name = "HiCacheG2Direct")]
pub struct PyHiCacheG2Direct {
    inner: Arc<HiCacheG2Direct>,
}

impl PyHiCacheG2Direct {
    /// The Python backend hands over fully-qualified keys (model, pool, and the
    /// rank where the pool is sharded), so this is a straight hash.
    fn keys(&self, page_hashes: &[String]) -> Vec<u64> {
        page_hashes
            .iter()
            .map(|key| hicache_g2_cache_key(key))
            .collect()
    }
}

#[pymethods]
impl PyHiCacheG2Direct {
    /// Join the serving-G2 namespace. ETCD and NATS come from the Dynamo runtime
    /// settings (`ETCD_ENDPOINTS`, `NATS_SERVER`) like every other Dynamo process.
    #[new]
    #[pyo3(signature = (
        tp_rank,
        tp_world_size,
        page_size,
        lease_ttl_sec = 30,
        max_transfer_mib = 512,
        ucx_devices = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        tp_rank: u32,
        tp_world_size: u32,
        page_size: u32,
        lease_ttl_sec: u64,
        max_transfer_mib: u64,
        ucx_devices: Option<String>,
    ) -> PyResult<Self> {
        if tp_world_size == 0 || tp_rank >= tp_world_size {
            return Err(PyValueError::new_err(format!(
                "tp_rank {tp_rank} is not valid for tp_world_size {tp_world_size}"
            )));
        }
        if page_size == 0 {
            return Err(PyValueError::new_err("page_size must be positive"));
        }
        if max_transfer_mib == 0 {
            return Err(PyValueError::new_err("max_transfer_mib must be positive"));
        }
        let config = HiCacheG2Config {
            tp_rank,
            tp_world_size,
            page_size,
            lease_ttl: Duration::from_secs(lease_ttl_sec),
            max_transfer_bytes: max_transfer_mib * 1024 * 1024,
            verify: HiCacheVerify::from_env().map_err(to_pyerr)?,
            hash_threads: hicache_g2_hash_threads().map_err(to_pyerr)?,
        };
        let runtime = get_current_tokio_handle();
        let inner = runtime
            .block_on(async move {
                let distributed_runtime =
                    DistributedRuntime::from_settings(Runtime::from_current()?).await?;
                HiCacheG2Direct::new(Arc::new(distributed_runtime), config, ucx_devices).await
            })
            .map_err(to_pyerr)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// `page_bytes` is THIS region's per-page stride; the Rust side needs it to
    /// pick stride-aligned registration chunks, and only the engine knows the
    /// layout.
    fn register_host_region(
        &self,
        py: Python<'_>,
        addr: u64,
        size: usize,
        page_bytes: usize,
    ) -> PyResult<()> {
        py.allow_threads(|| self.inner.register_host_region(addr, size, page_bytes))
            .map_err(to_pyerr)
    }

    /// Advertise pages now resident in this replica's host tier. No data moves.
    fn publish(
        &self,
        py: Python<'_>,
        page_hashes: Vec<String>,
        ptrs: Vec<u64>,
        sizes: Vec<u64>,
        segments_per_page: usize,
    ) -> PyResult<Vec<bool>> {
        let pages = page_segments(&ptrs, &sizes, segments_per_page)?;
        if pages.len() != page_hashes.len() {
            return Err(PyValueError::new_err(format!(
                "{} pages of pointers for {} keys",
                pages.len(),
                page_hashes.len()
            )));
        }
        let keys = self.keys(&page_hashes);
        py.allow_threads(|| self.inner.publish(&keys, pages))
            .map_err(to_pyerr)
    }

    /// Withdraw pages this replica can no longer serve.
    fn unpublish(&self, py: Python<'_>, page_hashes: Vec<String>) -> PyResult<()> {
        let keys = self.keys(&page_hashes);
        py.allow_threads(|| self.inner.unpublish(&keys))
            .map_err(to_pyerr)
    }

    /// Withdraw whatever this replica advertises at the given host slots. Call
    /// it before writing bytes that do not come through G2Direct (an L3 load)
    /// into host slots; `fetch` does this itself. Slots are described as for
    /// `publish`. Returns the number of index entries removed.
    fn invalidate_slots(
        &self,
        py: Python<'_>,
        ptrs: Vec<u64>,
        sizes: Vec<u64>,
        ptrs_per_page: usize,
    ) -> PyResult<usize> {
        // Validate here so malformed arguments raise ValueError, as for `publish`.
        page_segments(&ptrs, &sizes, ptrs_per_page)?;
        py.allow_threads(|| self.inner.invalidate_slots(&ptrs, &sizes, ptrs_per_page))
            .map_err(to_pyerr)
    }

    /// Length of the leading run of `page_hashes` that a peer advertises.
    fn lookup_prefix(&self, py: Python<'_>, page_hashes: Vec<String>) -> PyResult<usize> {
        let keys = self.keys(&page_hashes);
        py.allow_threads(|| self.inner.lookup_prefix(&keys))
            .map_err(to_pyerr)
    }

    /// Per-key availability, for hybrid pools whose restorable set has holes.
    fn lookup_mask(&self, py: Python<'_>, page_hashes: Vec<String>) -> PyResult<Vec<bool>> {
        let keys = self.keys(&page_hashes);
        py.allow_threads(|| self.inner.lookup_mask(&keys))
            .map_err(to_pyerr)
    }

    /// Read the leading pages of `page_hashes` out of peers into the given host
    /// addresses. Returns how many pages, from the front, completed, and a
    /// settlement ticket when reads were still in flight at the deadline: the
    /// pages after the completed prefix must not be reused until
    /// `is_settled(ticket)` returns true.
    fn fetch(
        &self,
        py: Python<'_>,
        page_hashes: Vec<String>,
        ptrs: Vec<u64>,
        sizes: Vec<u64>,
        segments_per_page: usize,
    ) -> PyResult<(usize, Option<u64>)> {
        let destination = page_segments(&ptrs, &sizes, segments_per_page)?;
        if destination.len() != page_hashes.len() {
            return Err(PyValueError::new_err(format!(
                "{} pages of pointers for {} keys",
                destination.len(),
                page_hashes.len()
            )));
        }
        let keys = self.keys(&page_hashes);
        let outcome = py
            .allow_threads(|| self.inner.fetch(&keys, &destination))
            .map_err(to_pyerr)?;
        Ok((outcome.completed, outcome.unsettled_ticket))
    }

    /// Whether the reads left in flight under `ticket` have all finished, so
    /// their destination pages may be reused. True is reported once.
    fn is_settled(&self, ticket: u64) -> PyResult<bool> {
        self.inner.is_settled(ticket).map_err(to_pyerr)
    }

    /// Transport counters. Cheap enough for the engine's scheduler thread.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let snapshot = self.inner.metrics().snapshot();
        let stats = PyDict::new(py);
        for (name, value) in snapshot {
            stats.set_item(name, value)?;
        }
        stats.set_item("instance_id", self.inner.instance_id())?;
        Ok(stats)
    }
}
