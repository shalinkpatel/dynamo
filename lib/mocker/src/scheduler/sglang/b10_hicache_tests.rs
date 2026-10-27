// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Baseten: scheduler tests for the HiCache host tier and G2PB peers
//! (`baseten_mocker::HiCache`).

use baseten_mocker::HiCacheArgs;

use super::core::SglangCore;
use super::tests::cache_reuse::{assert_slots_accounted, drain};
use super::tests::direct_request as request;
use crate::common::protocols::{EngineType, MockEngineArgs, SglangArgs};

#[test]
fn test_host_tier_serves_prefix_evicted_from_device() {
    // 20 device tokens; 64 host tokens at 1000 B/token; load-back at 1 KB/s.
    let args = MockEngineArgs::builder()
        .engine_type(EngineType::Sglang)
        .num_gpu_blocks(5)
        .block_size(4)
        .speedup_ratio(1.0)
        .kv_bytes_per_token(Some(1000))
        .sglang(Some(SglangArgs {
            page_size: Some(4),
            hicache: Some(HiCacheArgs {
                size: 64.0 * 1000.0 / 1e9,
                bandwidth_gb_per_sec: Some(1e-6),
                ..Default::default()
            }),
            ..Default::default()
        }))
        .build()
        .unwrap();
    let mut core = SglangCore::new(args);
    let mut now_ms = 0.0;
    let a: Vec<u32> = (1..=9).collect();

    core.receive(request(a.clone(), 1));
    drain(&mut core, &mut now_ms);
    // A 16-token prompt pushes A's two cached pages off the device.
    core.receive(request((100..116).collect(), 1));
    drain(&mut core, &mut now_ms);

    core.receive(request(a, 1));
    let admitted = drain(&mut core, &mut now_ms);
    assert_eq!(admitted[0].0, 8, "both pages come back from host");
    assert!(admitted[0].1 >= 8000.0, "8 KB at 1 KB/s");
    assert_eq!(core.cached_tokens_by_tier, [0, 8, 0]);
}

/// 32 device pages of 64 tokens, one 64-token chunk per pass, and a host
/// tier of `hicache_size` GB shared with `group`, read at `peer_gb_per_sec`.
fn peer_args(group: &str, hicache_size: f64, peer_gb_per_sec: Option<f64>) -> MockEngineArgs {
    MockEngineArgs::builder()
        .engine_type(EngineType::Sglang)
        .num_gpu_blocks(32)
        .block_size(64)
        .kv_bytes_per_token(Some(1000))
        .sglang(Some(SglangArgs {
            page_size: Some(64),
            chunked_prefill_size: Some(64),
            max_prefill_tokens: Some(64),
            hicache: Some(HiCacheArgs {
                size: hicache_size,
                peer_group: Some(group.into()),
                peer_bandwidth_gb_per_sec: peer_gb_per_sec,
                ..Default::default()
            }),
            ..Default::default()
        }))
        .build()
        .unwrap()
}

/// Peers advertise more than the local host tier can take: the request
/// recomputes those tokens, still one chunk per pass.
#[test]
fn test_unfetchable_peer_hit_respects_chunk_budget() {
    let mut owner = SglangCore::new(peer_args("unfetchable-peer-test", 0.001, None)); // 1000 host tokens
    let mut reader = SglangCore::new(peer_args("unfetchable-peer-test", 0.000128, None)); // 128 host tokens
    let prompt: Vec<u32> = (1..=520).collect();
    let mut now_ms = 0.0;
    owner.receive(request(prompt.clone(), 1));
    drain(&mut owner, &mut now_ms);

    reader.receive(request(prompt, 1));
    reader.execute_pass_internal(None, now_ms);
    assert_eq!(reader.waiting.len(), 1, "only one 64-token chunk this pass");
    drain(&mut reader, &mut now_ms);
    assert_slots_accounted(&reader);
}

/// A prompt fully cached on a peer still computes its last page (SGLang
/// always computes at least one token).
#[test]
fn test_full_prompt_peer_hit_computes_last_page() {
    let mut owner = SglangCore::new(peer_args("full-peer-test", 0.001, None));
    let mut reader = SglangCore::new(peer_args("full-peer-test", 0.001, None));
    let prompt: Vec<u32> = (1..=512).collect();
    let mut now_ms = 0.0;
    owner.receive(request(prompt.clone(), 1));
    drain(&mut owner, &mut now_ms);

    reader.receive(request(prompt, 1));
    drain(&mut reader, &mut now_ms);
    assert_eq!(reader.cached_tokens_by_tier, [0, 0, 448]);
    assert_slots_accounted(&reader);
}

/// Run `core` to completion; per request, when it was admitted and when its
/// first token came out.
fn admit_and_first_token_ms(
    core: &mut SglangCore,
    mut now_ms: f64,
) -> std::collections::HashMap<uuid::Uuid, (f64, f64)> {
    let mut times = std::collections::HashMap::new();
    while !core.is_empty() {
        let pass = core.execute_pass_internal(None, now_ms);
        for a in &pass.admissions {
            times.entry(a.uuid).or_insert((now_ms, f64::NAN));
        }
        for s in &pass.output_signals {
            let t = times.get_mut(&s.uuid).unwrap();
            if t.1.is_nan() {
                t.1 = pass.end_ms;
            }
        }
        now_ms = pass.end_ms;
    }
    times
}

/// A peer read delays only the request that needs it, by its wire time at
/// the configured bandwidth whatever `speedup_ratio` is; a request in the
/// same step finishes at its compute-only time.
#[test]
fn test_peer_read_delays_only_its_request() {
    let args = |group: &str, speedup_ratio: f64| {
        MockEngineArgs::builder()
            .engine_type(EngineType::Sglang)
            .num_gpu_blocks(64)
            .block_size(64)
            .speedup_ratio(speedup_ratio)
            .kv_bytes_per_token(Some(1000))
            .sglang(Some(SglangArgs {
                page_size: Some(64),
                hicache: Some(HiCacheArgs {
                    size: 0.001, // 1000 host tokens
                    peer_group: Some(group.into()),
                    peer_bandwidth_gb_per_sec: Some(1e-3), // 1 MB/s
                    ..Default::default()
                }),
                ..Default::default()
            }))
            .build()
            .unwrap()
    };
    let shared: Vec<u32> = (1..=512).collect();
    let mut with_peer: Vec<u32> = shared.clone();
    with_peer.extend(1000..1064);
    let other: Vec<u32> = (5000..5256).collect();

    let mut peer_wait = Vec::new();
    let mut compute = Vec::new();
    for (i, speedup) in [1.0, 0.25].into_iter().enumerate() {
        let group = format!("peer-read-isolation-{i}");
        let mut owner = SglangCore::new(args(&group, speedup));
        owner.receive(request(shared.clone(), 1));
        drain(&mut owner, &mut 0.0);

        let mut alone = SglangCore::new(args(&format!("{group}-alone"), speedup));
        let solo = alone.receive(request(other.clone(), 1));
        let solo_ms = admit_and_first_token_ms(&mut alone, 0.0)[&solo].1;

        let mut reader = SglangCore::new(args(&group, speedup));
        let a = reader.receive(request(with_peer.clone(), 1));
        let b = reader.receive(request(other.clone(), 1));
        let times = admit_and_first_token_ms(&mut reader, 0.0);
        assert_eq!(times[&b].1, solo_ms, "the other request is not delayed");
        assert!(times[&a].0 >= 512.0, "512 tokens x 1000 B at 1 MB/s");
        assert_eq!(reader.cached_tokens_by_tier[2], 512);
        assert_slots_accounted(&reader);
        peer_wait.push(times[&a].0);
        compute.push(solo_ms);
    }
    assert_eq!(
        peer_wait[0], peer_wait[1],
        "speedup_ratio does not scale the wire"
    );
    assert!(
        compute[1] > compute[0] * 3.9,
        "speedup_ratio scales compute"
    );
}

/// An arrival at a worker idling on a peer read is not stuck behind it.
#[test]
fn test_arrival_during_peer_read_is_admitted() {
    let args = peer_args("arrival-peer-test", 0.001, Some(1e-3));
    let mut owner = SglangCore::new(args.clone());
    let mut reader = SglangCore::new(args);
    let shared: Vec<u32> = (1..=512).collect();
    owner.receive(request(shared.clone(), 1));
    drain(&mut owner, &mut 0.0);

    let mut with_peer = shared;
    with_peer.extend(1000..1064);
    reader.receive(request(with_peer, 1)); // 512 ms peer read
    let mut now_ms = 0.0;
    while now_ms < 10.0 {
        now_ms = reader.execute_pass_internal(None, now_ms).end_ms;
    }
    let other = reader.receive(request((5000..5064).collect(), 1));
    let admitted_at = now_ms;
    let pass = reader.execute_pass_internal(None, now_ms);
    assert!(pass.admissions.iter().any(|a| a.uuid == other));
    assert!(admitted_at <= 15.0, "admitted at {admitted_at} ms");
}
