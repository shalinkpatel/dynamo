// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Heap bytes the replica directory holds per advertised page, measured with a
//! counting allocator. Ignored by default; run with
//! `cargo test -p dynamo-g2direct --test directory_memory -- --ignored --nocapture`.
// The directory and the L3 directory it is compared with need `transport`.
#![cfg(feature = "transport")]

use dynamo_g2direct::peer_index::{
    ReplicaBlock, ReplicaEvent, ReplicaEventKind, ReplicaIndex, ReplicaSource, ReplicaSourceKey,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering};

struct Counting;
static LIVE: AtomicIsize = AtomicIsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LIVE.fetch_add(layout.size() as isize, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        LIVE.fetch_add(
            new_size as isize - layout.size() as isize,
            Ordering::Relaxed,
        );
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn source(instance_id: u64, row: Option<&str>) -> ReplicaSource {
    ReplicaSource {
        key: ReplicaSourceKey {
            instance_id,
            tp_rank: 0,
        },
        replica_id: Some(instance_id),
        tp_world_size: 4,
        incarnation: uuid::Uuid::from_u128(instance_id as u128),
        block_size: 64,
        row: row.map(str::to_string),
    }
}

/// `sources` publishers advertise `pages` keys each; `shared` of every source's
/// keys are also advertised by every other source (a prefix all replicas hold).
fn measure(sources: u64, pages: u64, shared: u64, row: Option<&str>) -> (usize, f64) {
    let before = LIVE.load(Ordering::Relaxed);
    let index = ReplicaIndex::new();
    for instance in 0..sources {
        let src = source(1_000 + instance, row);
        for chunk in (0..pages).collect::<Vec<_>>().chunks(4096) {
            let blocks = chunk
                .iter()
                .map(|page| {
                    let key = if *page < shared {
                        *page
                    } else {
                        (instance + 1) << 40 | page
                    };
                    ReplicaBlock {
                        key,
                        size_bytes: 2 << 20,
                        checksum: Some(key ^ 0x5555),
                    }
                })
                .collect();
            index.apply(ReplicaEvent {
                source: src.clone(),
                revision: chunk[0] + 1,
                kind: ReplicaEventKind::Upsert(blocks),
            });
        }
    }
    let bytes = (LIVE.load(Ordering::Relaxed) - before) as usize;
    let entries = (sources * pages) as f64;
    drop(index);
    (bytes, bytes as f64 / entries)
}

#[test]
#[ignore = "measurement, prints a table"]
fn bytes_per_directory_entry() {
    println!("sources pages/src shared row    total_MiB  B/entry");
    for (sources, pages, shared, row) in [
        (8, 100_000, 0, Some("e02-sg-s6q4vqmze08")),
        (8, 100_000, 0, None),
        (8, 100_000, 50_000, Some("e02-sg-s6q4vqmze08")),
        (32, 100_000, 0, Some("e02-sg-s6q4vqmze08")),
        (8, 1_000_000, 0, Some("e02-sg-s6q4vqmze08")),
    ] {
        let (bytes, per) = measure(sources, pages, shared, row);
        println!(
            "{sources:>7} {pages:>9} {shared:>6} {:<6} {:>10.1} {per:>8.1}",
            if row.is_some() { "yes" } else { "no" },
            bytes as f64 / (1 << 20) as f64
        );
    }
}

#[test]
#[ignore = "measurement, prints a table"]
fn layout_alternatives() {
    use dynamo_g2direct::peer_index::ReplicaLocation;
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;
    println!(
        "size_of ReplicaLocation={} ReplicaSource={} ReplicaBlock={}",
        std::mem::size_of::<ReplicaLocation>(),
        std::mem::size_of::<ReplicaSource>(),
        std::mem::size_of::<ReplicaBlock>()
    );
    let (sources, pages) = (8u64, 100_000u64);
    let entries = (sources * pages) as f64;
    // (a) key -> Vec of (Arc source, block, revision), shrunk to fit.
    let before = LIVE.load(Ordering::Relaxed);
    let mut by_key: HashMap<u64, Vec<(Arc<ReplicaSource>, ReplicaBlock, u64)>> = HashMap::new();
    for instance in 0..sources {
        let src = Arc::new(source(1_000 + instance, Some("e02-sg-s6q4vqmze08")));
        for page in 0..pages {
            let key = (instance + 1) << 40 | page;
            let v = by_key.entry(key).or_default();
            v.push((
                Arc::clone(&src),
                ReplicaBlock {
                    key,
                    size_bytes: 2 << 20,
                    checksum: Some(key),
                },
                1,
            ));
            v.shrink_to_fit();
        }
    }
    let a = (LIVE.load(Ordering::Relaxed) - before) as f64 / entries;
    drop(by_key);
    // (b) one key set per source; a lookup asks each source's set.
    let before = LIVE.load(Ordering::Relaxed);
    let mut per_source: Vec<HashSet<u64>> = Vec::new();
    for instance in 0..sources {
        let mut set = HashSet::new();
        for page in 0..pages {
            set.insert((instance + 1) << 40 | page);
        }
        per_source.push(set);
    }
    let b = (LIVE.load(Ordering::Relaxed) - before) as f64 / entries;
    println!(
        "(a) Arc source + shrink_to_fit: {a:.1} B/entry; (b) per-source key set: {b:.1} B/entry"
    );
}

#[test]
#[ignore = "measurement, prints a table"]
fn bytes_per_own_l3_directory_entry() {
    use dynamo_g2direct::hicache_staging::L3Directory;
    use std::sync::Arc;
    let pages = 262_144u64;
    let pool: Arc<str> = Arc::from("kv");
    let before = LIVE.load(Ordering::Relaxed);
    let mut directory = L3Directory::new(pages as usize);
    for page in 0..pages {
        // SGLang page keys are 64-hex-digit hashes.
        // One L3 object per page, as for an MLA KV page in Mooncake.
        directory.insert(
            page,
            Arc::clone(&pool),
            format!("{page:064x}"),
            2 << 20,
            vec![page ^ 0x9e37],
        );
    }
    let per = (LIVE.load(Ordering::Relaxed) - before) as f64 / pages as f64;
    println!(
        "own L3Directory: {per:.1} B/entry ({:.1} MiB for {pages})",
        per * pages as f64 / (1 << 20) as f64
    );
}

#[test]
#[ignore = "measurement, prints a table"]
fn key_to_source_index_layout() {
    use std::collections::HashMap;
    // (c) key -> small source index (one u16 inline, a boxed list for the rare
    // key several sources hold): O(1) lookup like today.
    #[allow(dead_code)]
    enum Holders {
        One(u16),
        Many(Box<[u16]>),
    }
    let (sources, pages) = (8u64, 100_000u64);
    let before = LIVE.load(Ordering::Relaxed);
    let mut map: HashMap<u64, Holders> = HashMap::new();
    for instance in 0..sources {
        for page in 0..pages {
            map.insert((instance + 1) << 40 | page, Holders::One(instance as u16));
        }
    }
    let per = (LIVE.load(Ordering::Relaxed) - before) as f64 / (sources * pages) as f64;
    println!("(c) key -> source index: {per:.1} B/entry");
}
