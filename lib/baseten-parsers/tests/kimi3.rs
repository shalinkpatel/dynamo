// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Parity at the public adapter boundary, including delegated guided output.

use baseten_parsers::{Event, Tool, UnifiedStream, request_init};
use serde_json::json;

#[test]
fn kimi3_guided_output_matches_upstream_for_each_chunk_and_policy() {
    let tools = [Tool {
        name: "bash".into(),
        description: None,
        parameters: json!({"type": "object", "properties": {"command": {"type": "string"}}}),
        strict: None,
    }];
    for starting_state in ["none", "reasoning", "response"] {
        for policy in ["reject", "recover_as_text", "stream_best_effort"] {
            for named in [false, true] {
                let payload = if named {
                    r#"{"command":"echo hello"}"#
                } else {
                    r#"[{"name":"bash","arguments":{"command":"echo hello"}}]"#
                };
                for payload in [payload, r#"{"command":"unfinished"#] {
                    let input = match starting_state {
                        "reasoning" => format!("private<|close|>think<|sep|>{payload}"),
                        "response" => payload.to_string(),
                        _ => format!("<|open|>think<|sep|>private<|close|>think<|sep|>{payload}"),
                    };
                    let init = request_init(
                        vec![],
                        starting_state,
                        "guided_json",
                        named.then(|| "bash".into()),
                        policy,
                    )
                    .unwrap();
                    let mut local =
                        UnifiedStream::new("baseten_kimi3_streaming", &tools, init.clone())
                            .unwrap();
                    let mut reference = UnifiedStream::new("kimi_k3", &tools, init).unwrap();
                    assert!(local.preserve_special_tokens());
                    for (at, ch) in input.char_indices() {
                        let chunk = Some(&input[at..at + ch.len_utf8()]);
                        assert_same(local.advance(chunk), reference.advance(chunk));
                    }
                    assert_same(local.advance(None), reference.advance(None));
                    assert!(local.advance(Some("late")).is_err());
                }
            }
        }
    }
}

fn assert_same(
    local: Result<Vec<Event>, baseten_parsers::StreamError>,
    reference: Result<Vec<Event>, baseten_parsers::StreamError>,
) {
    match (local, reference) {
        (Ok(local), Ok(reference)) => assert_eq!(local, reference),
        (Err(local), Err(reference)) => {
            assert_eq!(local.events, reference.events);
            assert_eq!(local.error.to_string(), reference.error.to_string());
        }
        (local, reference) => panic!("local/upstream outcome differs: {local:?} vs {reference:?}"),
    }
}

#[test]
fn kimi3_local_native_supports_prefilled_channels() {
    for state in ["none", "reasoning", "response"] {
        let init = request_init(vec![], state, "native", None, "reject").unwrap();
        let mut stream = UnifiedStream::new("baseten_kimi3_streaming", &[], init).unwrap();
        let events = stream.advance(Some("hello")).unwrap();
        let expected = if state == "reasoning" {
            Event::Reasoning("hello".into())
        } else {
            Event::Text("hello".into())
        };
        assert_eq!(events, vec![expected]);
        stream.advance(None).unwrap();
    }
}
