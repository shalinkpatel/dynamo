// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::protocol::{GenerationResponseFrameV1, generation_response_frame_v1::Frame};
use prost::Message;
use serde_json::json;

#[derive(Clone, PartialEq, Message)]
struct LegacyChunkFrame {
    #[prost(bytes = "vec", optional, tag = "3")]
    chunk_msgpack: Option<Vec<u8>>,
}

fn b10_observe(tracker: &mut TokenUsageTracker, value: serde_json::Value) -> TokenUsageV1 {
    tracker.observe(Some(&rmpv::ext::to_value(value).unwrap()));
    tracker.snapshot()
}

#[test]
fn b10_usage_accepts_late_counts_and_retains_them() {
    let mut tracker = TokenUsageTracker::new(false);
    assert_eq!(tracker.snapshot(), TokenUsageV1::default());
    let first = b10_observe(&mut tracker, json!({"outputs": [{"token_ids_diff": [11]}]}));
    assert_eq!(first.input_tokens, None);
    assert_eq!(first.cached_tokens, None);
    assert_eq!(first.output_tokens, Some(1));
    let last = b10_observe(
        &mut tracker,
        json!({
            "completion_usage": {"prompt_tokens": 10, "completion_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 7}},
            "outputs": [{"token_ids_diff": [12, 13]}]
        }),
    );
    assert_eq!(
        last,
        TokenUsageV1 {
            input_tokens: Some(10),
            cached_tokens: Some(7),
            output_tokens: Some(3),
        }
    );
    assert_eq!(b10_observe(&mut tracker, json!({"finished": true})), last);
    tracker.observe(None);
    assert_eq!(tracker.snapshot(), last);
}

#[test]
fn b10_disagg_keeps_prefill_counts_and_fills_missing_ones() {
    for (prefill, decode, want) in [
        (
            json!({"token_ids": [], "disaggregated_params": {}}),
            json!({"completion_usage": {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 7}}}),
            (Some(10_u64), Some(7_u64), Some(0_u64)),
        ),
        (
            json!({"input_tokens": 10, "cached_tokens": 0}),
            json!({"completion_usage": {"prompt_tokens": 11, "prompt_tokens_details": {"cached_tokens": 7}},
                "outputs": [{"token_ids_diff": [11]}, {"token_ids_diff": [12, 13]}]}),
            (Some(10_u64), Some(0_u64), Some(3_u64)),
        ),
    ] {
        let mut tracker = TokenUsageTracker::new(true);
        b10_observe(&mut tracker, prefill);
        let usage = b10_observe(&mut tracker, decode);
        assert_eq!(
            (usage.input_tokens, usage.cached_tokens, usage.output_tokens),
            want
        );
    }
}

#[test]
fn b10_usage_preserves_zero_and_ignores_invalid_counters() {
    let mut tracker = TokenUsageTracker::new(false);
    let zero = b10_observe(
        &mut tracker,
        json!({
            "input_tokens": 0, "cached_tokens": 0, "output_tokens": 0
        }),
    );
    assert_eq!(
        zero,
        TokenUsageV1 {
            input_tokens: Some(0),
            cached_tokens: Some(0),
            output_tokens: Some(0),
        }
    );
    assert_eq!(
        b10_observe(
            &mut tracker,
            json!({
                "input_tokens": -1, "cached_tokens": "7", "output_tokens": null
            })
        ),
        zero
    );
}

#[test]
fn b10_decode_denial_usage_retains_known_prefill_work() {
    let response = rmpv::ext::to_value(json!({
        "input_tokens": 10, "cached_tokens": 7,
        "outputs": [{"token_ids_diff": [11]}]
    }))
    .unwrap();
    assert_eq!(
        prefill_usage(&response),
        TokenUsageV1 {
            input_tokens: Some(10),
            cached_tokens: Some(7),
            output_tokens: Some(1),
        }
    );
    assert_eq!(prefill_usage(&Value::Nil), TokenUsageV1::default());
}

#[test]
fn b10_usage_supports_empty_deltas_and_worker_cumulative_counts() {
    let mut tracker = TokenUsageTracker::new(false);
    assert_eq!(
        b10_observe(&mut tracker, json!({"outputs": []})).output_tokens,
        None
    );
    assert_eq!(
        b10_observe(&mut tracker, json!({"token_ids": []})).output_tokens,
        Some(0)
    );
    assert_eq!(
        b10_observe(&mut tracker, json!({"output_tokens": 4})).output_tokens,
        Some(4)
    );
    assert_eq!(
        b10_observe(&mut tracker, json!({"token_ids": [11]})).output_tokens,
        Some(4)
    );
}

#[test]
fn b10_usage_does_not_count_reported_multi_choice_tokens_twice() {
    let mut tracker = TokenUsageTracker::new(false);
    // One worker iteration finishes two choices. Each chunk reports the
    // total for both choices before the second choice's delta is visible.
    for index in [0, 1] {
        let usage = b10_observe(
            &mut tracker,
            json!({
                "index": index, "token_ids": [11], "finish_reason": "stop",
                "completion_usage": {"prompt_tokens": 10, "completion_tokens": 2}
            }),
        );
        assert_eq!(usage.output_tokens, Some(2));
    }
    assert_eq!(tracker.visible_output_tokens, Some(2));
}

#[test]
fn b10_usage_wire_preserves_optional_zero_and_older_frames() {
    // The old wire shape for a chunk has no usage field. Protobuf readers
    // must skip the additive field while preserving the original payload.
    let legacy = LegacyChunkFrame {
        chunk_msgpack: Some(vec![0x80]),
    };
    for count in [None, Some(0), Some(12)] {
        let frame = GenerationResponseFrameV1 {
            usage: Some(TokenUsageV1 {
                input_tokens: count,
                cached_tokens: count,
                output_tokens: count,
            }),
            frame: Some(Frame::ChunkMsgpack(legacy.chunk_msgpack.clone().unwrap())),
        };
        let old_reader = LegacyChunkFrame::decode(frame.encode_to_vec().as_slice()).unwrap();
        assert_eq!(old_reader, legacy);
        let decoded = GenerationResponseFrameV1::decode(frame.encode_to_vec().as_slice()).unwrap();
        assert_eq!(decoded, frame);
    }
    let older = GenerationResponseFrameV1::decode(legacy.encode_to_vec().as_slice()).unwrap();
    assert!(older.usage.is_none());
    // Future usage counters are additive too; existing readers ignore them.
    let future = TokenUsageV1::decode(&[0x08, 10, 0x20, 42][..]).unwrap();
    assert_eq!(future.input_tokens, Some(10));
    assert_eq!(future.cached_tokens, None);
}

#[test]
fn b10_usage_survives_an_old_proxy_via_unchanged_worker_chunks() {
    for prefill_first in [false, true] {
        let mut source = TokenUsageTracker::new(prefill_first);
        let mut after_old_proxy = TokenUsageTracker::new(prefill_first);
        for data in [
            json!({"input_tokens": 10, "cached_tokens": 7,
                "outputs": [{"token_ids_diff": [11]}]}),
            json!({"cached_tokens": 0, "outputs": [{"token_ids_diff": [12, 13]}]}),
        ] {
            let data = rmpv::ext::to_value(data).unwrap();
            source.observe(Some(&data));
            // Annotated worker data is opaque to the old protobuf reader.
            let annotated = Value::Map(vec![("data".into(), data)]);
            let payload = rmp_serde::to_vec_named(&annotated).unwrap();
            let frame = GenerationResponseFrameV1 {
                usage: Some(source.snapshot()),
                frame: Some(Frame::ChunkMsgpack(payload.clone())),
            };
            let legacy = LegacyChunkFrame::decode(frame.encode_to_vec().as_slice()).unwrap();
            // Match RemoteGenerationCoordinator: decode the worker chunk,
            // yield it unchanged, then serialize it again in the service.
            let worker: Value =
                rmp_serde::from_slice(legacy.chunk_msgpack.as_ref().unwrap()).unwrap();
            let relayed = LegacyChunkFrame {
                chunk_msgpack: Some(rmp_serde::to_vec_named(&worker).unwrap()),
            };
            let decoded =
                GenerationResponseFrameV1::decode(relayed.encode_to_vec().as_slice()).unwrap();
            assert!(decoded.usage.is_none());
            let Some(Frame::ChunkMsgpack(bytes)) = decoded.frame else {
                panic!("old proxy must preserve the chunk");
            };
            assert_eq!(bytes, payload);
            let worker: Value = rmp_serde::from_slice(&bytes).unwrap();
            after_old_proxy.observe(Some(&worker["data"]));
            assert_eq!(after_old_proxy.snapshot(), source.snapshot());
        }
    }
}
