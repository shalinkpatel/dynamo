// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use super::super::KimiK3Parser;
use super::*;
use crate::upstream::tool_calling::traits::ToolParser;
use crate::upstream::unified::{UnifiedEvent, UnifiedParserExt, assemble};
use crate::{Tool, UnifiedParser, UnifiedParserInit};

fn streaming_events(
    chunks: &[&str],
) -> (
    Vec<crate::upstream::UnifiedParserEvent>,
    Box<dyn UnifiedParser>,
) {
    let mut parser = kimi_k3_streaming_unified(&[]);
    let mut events = Vec::new();
    for chunk in chunks {
        events.extend(parser.push(chunk).unwrap());
    }
    events.extend(parser.finish().unwrap().events);
    (events, parser)
}

#[test]
fn streaming_bash_emits_each_value_chunk_before_any_closing_marker() {
    let mut parser = kimi_k3_streaming_unified(&[]);
    let header = concat!(
        "<|open|>tools<|sep|><|open|>call tool=\"bash\" index=\"3\"<|sep|>",
        "<|open|>argument key=\"command\" type=\"string\"<|sep|>"
    );
    let mut events = parser.push(header).unwrap();
    assert_eq!(parser.tool_call_id(0), Some("bash:2"));
    for chunk in ["printf ", "\"héllo 🌍\"", "\\n", "\n", "echo done"] {
        let output = parser.push(chunk).unwrap();
        let emitted: String = output
            .iter()
            .filter_map(|event| match event {
                crate::upstream::UnifiedParserEvent::ToolCall(call) => {
                    assert!(!call.complete);
                    Some(call.arguments.as_str())
                }
                _ => None,
            })
            .collect();
        let encoded = serde_json::to_string(chunk).unwrap();
        assert_eq!(emitted, encoded[1..encoded.len() - 1], "chunk {chunk:?}");
        events.extend(output);
    }
    events.extend(
        parser
            .push("<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|>")
            .unwrap(),
    );
    events.extend(parser.finish().unwrap().events);
    assert_eq!(
        assemble(&events),
        vec![UnifiedEvent::ToolCall {
            name: "bash".into(),
            arguments: serde_json::json!({"command": "printf \"héllo 🌍\"\\n\necho done"}),
        }]
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event,
            crate::upstream::UnifiedParserEvent::ToolCall(call) if call.complete))
            .count(),
        1
    );
}

#[test]
fn streaming_native_matches_buffered_at_every_utf8_split() {
    let bodies = [
            "".to_string(),
            arg("command", "string", "echo \"héllo 🌍\"\n\\path\t\r\u{0001}"),
            format!("{}{}{}{}", arg("n", "number", "1.25e+2"), arg("ok", "boolean", "true"), arg("none", "null", "null"), arg("a", "array", "[1, {\"x\": \" spaced \", \"t\": \"<|close|>argument<|sep|>\"}]")),
            arg("command", "string", "a<|close|>argument<|sep|>b<|open|>call tool=\"literal\"<|sep|>c"),
            arg("command", "string", "a<|close|>argument<|sep|><|close|>call<|sep|>literal"),
            arg("command", "string", "a<|close|>argument<|sep|>"),
            "<|open|>json<|sep|>{ \"command\": \"echo \\\"hi\\\"\", \"array\": [1, 2] }<|close|>json<|sep|>".to_string(),
        ];
    for body in bodies {
        let input = format!(
            "{}private{}{}{}{}{}answer{}",
            THINK_OPEN.canonical,
            THINK_CLOSE.canonical,
            TOOLS_OPEN.canonical,
            call("bash", "1", &body),
            TOOLS_CLOSE.canonical,
            RESPONSE_OPEN.canonical,
            RESPONSE_CLOSE.canonical
        );
        let mut reference =
            crate::upstream::create_unified_parser_for_family("kimi_k3", &[]).unwrap();
        let mut expected = reference.push(&input).unwrap();
        expected.extend(reference.finish().unwrap().events);
        for split in (0..=input.len()).filter(|at| input.is_char_boundary(*at)) {
            let (events, parser) = streaming_events(&[&input[..split], &input[split..]]);
            assert_eq!(
                assemble(&events),
                assemble(&expected),
                "body {body:?}, split {split}"
            );
            assert_eq!(parser.tool_call_id(0), Some("bash:0"));
        }
        let chunks: Vec<_> = input
            .char_indices()
            .map(|(at, ch)| &input[at..at + ch.len_utf8()])
            .collect();
        assert_eq!(assemble(&streaming_events(&chunks).0), assemble(&expected));
        let spaced = input
            .replace("<|open|>", "<|open|> ")
            .replace("<|close|>", "<|close|> ")
            .replace("<|sep|>", " <|sep|>");
        let mut reference =
            crate::upstream::create_unified_parser_for_family("kimi_k3", &[]).unwrap();
        let mut expected = reference.push(&spaced).unwrap();
        expected.extend(reference.finish().unwrap().events);
        let chunks: Vec<_> = spaced
            .char_indices()
            .map(|(at, ch)| &spaced[at..at + ch.len_utf8()])
            .collect();
        assert_eq!(assemble(&streaming_events(&chunks).0), assemble(&expected));
    }
}

#[test]
fn streaming_long_bash_releases_every_chunk_without_reparsing_the_body() {
    let mut native = KimiK3Native::new();
    native.stream_arguments = true;
    let mut output = UnifiedParserOutput::default();
    native
        .push_native(
            concat!(
                "<|open|>tools<|sep|><|open|>call tool=\"bash\" index=\"1\"<|sep|>",
                "<|open|>argument key=\"command\" type=\"string\"<|sep|>"
            ),
            &mut output,
        )
        .unwrap();
    let chunk = "echo \"hello 🌍\"\n".repeat(256);
    let encoded = serde_json::to_string(&chunk).unwrap();
    for _ in 0..128 {
        output.events.clear();
        native.push_native(&chunk, &mut output).unwrap();
        assert_eq!(
            output.events,
            vec![crate::upstream::UnifiedParserEvent::ToolCall(
                ToolCallDelta {
                    tool_index: 0,
                    name: None,
                    arguments: encoded[1..encoded.len() - 1].to_string(),
                    complete: false,
                }
            )]
        );
        assert_eq!(native.call_boundary.body_parse_count, 0);
    }
    output.events.clear();
    native
        .push_native(
            "<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|>",
            &mut output,
        )
        .unwrap();
    assert!(output.events.iter().any(|event| matches!(event,
            crate::upstream::UnifiedParserEvent::ToolCall(call) if call.complete)));
}

#[test]
fn streaming_nested_json_and_raw_json_emit_before_value_closure() {
    for (open, first, second) in [
        (
            "<|open|>argument key=\"items\" type=\"array\"<|sep|>",
            "[1, ",
            "{\"x\": 2",
        ),
        ("<|open|>json<|sep|>", "{ \"command\": \"echo ", "hi"),
    ] {
        let mut parser = kimi_k3_streaming_unified(&[]);
        parser
            .push(&format!(
                "{}<|open|>call tool=\"bash\" index=\"1\"<|sep|>{open}",
                TOOLS_OPEN.canonical
            ))
            .unwrap();
        for chunk in [first, second] {
            let events = parser.push(chunk).unwrap();
            assert!(events.iter().any(|event| matches!(event,
                    crate::upstream::UnifiedParserEvent::ToolCall(call) if !call.arguments.is_empty() && !call.complete)));
        }
    }
}

#[test]
fn streaming_errors_preserve_committed_fragments_and_do_not_complete() {
    let prefix = format!(
        "{}<|open|>call tool=\"bash\" index=\"1\"<|sep|>",
        TOOLS_OPEN.canonical
    );
    let invalid = format!(
        "{prefix}{}{}{}",
        arg("n", "number", "oops"),
        CALL_CLOSE.canonical,
        TOOLS_CLOSE.canonical
    );
    let duplicate = format!(
        "{prefix}{}{}{}{}",
        arg("x", "string", "a"),
        arg("x", "string", "b"),
        CALL_CLOSE.canonical,
        TOOLS_CLOSE.canonical
    );
    for input in [invalid, duplicate] {
        for split in (0..=input.len()).filter(|at| input.is_char_boundary(*at)) {
            let mut parser = kimi_k3_streaming_unified(&[]);
            let mut output = UnifiedParserOutput::default();
            let result = parser
                .parse_into(&input[..split], &mut output)
                .and_then(|()| parser.parse_into(&input[split..], &mut output));
            assert!(result.is_err(), "split {split}, {input}");
            assert!(
                output
                    .events
                    .iter()
                    .any(|event| matches!(event, crate::upstream::UnifiedParserEvent::ToolCall(_)))
            );
            assert!(!output.events.iter().any(|event| matches!(event,
                    crate::upstream::UnifiedParserEvent::ToolCall(call) if call.complete)));
        }
    }
    let mut parser = kimi_k3_streaming_unified(&[]);
    let events = parser
        .push(&format!(
            "{prefix}<|open|>argument key=\"command\" type=\"string\"<|sep|>echo partial"
        ))
        .unwrap();
    assert!(events.iter().any(|event| matches!(event, crate::upstream::UnifiedParserEvent::ToolCall(call) if call.arguments.contains("echo partial"))));
    assert!(parser.finish().is_err());
    parser.reset();
    let events = parser
        .push(&format!(
            "{}{}{}",
            TOOLS_OPEN.canonical,
            call("bash", "2", &arg("command", "string", "done")),
            TOOLS_CLOSE.canonical
        ))
        .unwrap();
    assert_eq!(
        assemble(&events),
        vec![UnifiedEvent::ToolCall {
            name: "bash".into(),
            arguments: serde_json::json!({"command": "done"})
        }]
    );
    assert_eq!(parser.tool_call_id(0), Some("bash:1"));
}

const SEP: &str = "<|sep|>";

fn arg(key: &str, arg_type: &str, value: &str) -> String {
    format!("{OPEN}argument key=\"{key}\" type=\"{arg_type}\"{SEP}{value}{CLOSE}argument{SEP}")
}

fn call(name: &str, index: &str, body: &str) -> String {
    format!("{OPEN}call tool=\"{name}\" index=\"{index}\"{SEP}{body}{CLOSE}call{SEP}")
}

fn run(input: &str, state: UnifiedParserStartingState) -> (Vec<UnifiedEvent>, Vec<String>) {
    let mut parser = kimi_k3_unified(&[]);
    parser
        .initialize_request(UnifiedParserInit {
            starting_state: state,
            ..UnifiedParserInit::default()
        })
        .unwrap();
    let mut events = parser.push(input).unwrap();
    events.extend(parser.finish().unwrap().events);
    let ids = (0..4)
        .filter_map(|index| parser.tool_call_id(index).map(str::to_string))
        .collect();
    (assemble(&events), ids)
}

fn native_events_from_state(
    input: &str,
    chunks: &[usize],
    starting_state: UnifiedParserStartingState,
) -> Vec<crate::upstream::UnifiedParserEvent> {
    let mut parser = kimi_k3_unified(&[]);
    parser
        .initialize_request(UnifiedParserInit {
            starting_state,
            ..UnifiedParserInit::default()
        })
        .unwrap();
    let mut events = Vec::new();
    let mut start = 0;
    for end in chunks.iter().copied().chain(std::iter::once(input.len())) {
        events.extend(parser.push(&input[start..end]).unwrap());
        start = end;
    }
    events.extend(parser.finish().unwrap().events);
    events
}

fn assert_native_fragmentations_from_state(
    input: &str,
    starting_state: UnifiedParserStartingState,
    expected: &[UnifiedEvent],
) {
    let boundaries: Vec<usize> = (1..input.len())
        .filter(|at| input.is_char_boundary(*at))
        .collect();
    assert_eq!(
        assemble(&native_events_from_state(input, &[], starting_state)),
        expected
    );
    for split in &boundaries {
        assert_eq!(
            assemble(&native_events_from_state(input, &[*split], starting_state)),
            expected,
            "split at byte {split}"
        );
    }
    assert_eq!(
        assemble(&native_events_from_state(
            input,
            &boundaries,
            starting_state,
        )),
        expected,
        "one Unicode scalar per push"
    );
}

fn assert_native_fragmentations(input: &str, expected: &[UnifiedEvent]) {
    assert_native_fragmentations_from_state(input, UnifiedParserStartingState::None, expected);
}

#[test]
fn ordered_channels_typed_arguments_ids_and_all_splits() {
    let body = [
        arg("city", "string", "Zürich"),
        arg("days", "number", "1.0"),
        arg("rain", "boolean", "true"),
    ]
    .concat();
    let input = format!(
        "{OPEN}think{SEP}plan{CLOSE}think{SEP}{OPEN}response{SEP}checking{CLOSE}response{SEP}{OPEN}tools{SEP}{}{OPEN}call tool=\"second\" index=\"raw\"{SEP}{CLOSE}call{SEP}{CLOSE}tools{SEP}{CLOSE}message{SEP}{END_OF_MSG}",
        call("weather", "1", &body)
    );
    let expected = vec![
        UnifiedEvent::Reasoning {
            text: "plan".into(),
        },
        UnifiedEvent::Text {
            text: "checking".into(),
        },
        UnifiedEvent::ToolCall {
            name: "weather".into(),
            arguments: serde_json::json!({"city":"Zürich","days":1.0,"rain":true}),
        },
        UnifiedEvent::ToolCall {
            name: "second".into(),
            arguments: serde_json::json!({}),
        },
    ];
    assert_native_fragmentations(&input, &expected);
    let (_, ids) = run(&input, UnifiedParserStartingState::None);
    assert_eq!(ids, ["weather:0", "second:raw"]);
}

#[test]
fn every_closed_channel_returns_to_idle_for_later_channels() {
    let first = call("f", "1", &arg("x", "number", "1"));
    let second = call("g", "2", &arg("y", "string", "Zürich"));
    let input = format!(
        "{}before{}{}{}{}{}mid{}{}{}{}{}after{}{}final{}",
        RESPONSE_OPEN.canonical,
        RESPONSE_CLOSE.canonical,
        TOOLS_OPEN.canonical,
        first,
        TOOLS_CLOSE.canonical,
        THINK_OPEN.canonical,
        THINK_CLOSE.canonical,
        TOOLS_OPEN.canonical,
        second,
        TOOLS_CLOSE.canonical,
        RESPONSE_OPEN.canonical,
        RESPONSE_CLOSE.canonical,
        THINK_OPEN.canonical,
        THINK_CLOSE.canonical,
    );
    assert_native_fragmentations(
        &input,
        &[
            UnifiedEvent::Text {
                text: "before".into(),
            },
            UnifiedEvent::ToolCall {
                name: "f".into(),
                arguments: serde_json::json!({"x":1}),
            },
            UnifiedEvent::Reasoning { text: "mid".into() },
            UnifiedEvent::ToolCall {
                name: "g".into(),
                arguments: serde_json::json!({"y":"Zürich"}),
            },
            UnifiedEvent::Text {
                text: "after".into(),
            },
            UnifiedEvent::Reasoning {
                text: "final".into(),
            },
        ],
    );
}

#[test]
fn spaced_markers_and_prefilled_channels() {
    let input = concat!(
        "private<|close|> think <|sep|>",
        "<|open|> response <|sep|>visible",
        "<|close|> response <|sep|>",
        "<|close|> message <|sep|>"
    );
    assert_eq!(
        run(input, UnifiedParserStartingState::Reasoning).0,
        vec![
            UnifiedEvent::Reasoning {
                text: "private".into()
            },
            UnifiedEvent::Text {
                text: "visible".into()
            }
        ]
    );
    assert_eq!(
        run(
            "visible<|close|>response<|sep|>",
            UnifiedParserStartingState::Response
        )
        .0,
        vec![UnifiedEvent::Text {
            text: "visible".into()
        }]
    );
}

#[test]
fn elided_think_close_hands_off_to_response_or_tools() {
    let response = format!("private{}visible", RESPONSE_OPEN.canonical);
    assert_eq!(
        run(&response, UnifiedParserStartingState::Reasoning).0,
        vec![
            UnifiedEvent::Reasoning {
                text: "private".into()
            },
            UnifiedEvent::Text {
                text: "visible".into()
            }
        ]
    );
    let tool = format!(
        "private{}{}{}",
        TOOLS_OPEN.canonical,
        call("calc", "x", &arg("n", "number", "4")),
        TOOLS_CLOSE.canonical
    );
    assert_eq!(
        run(&tool, UnifiedParserStartingState::Reasoning).0,
        vec![
            UnifiedEvent::Reasoning {
                text: "private".into()
            },
            UnifiedEvent::ToolCall {
                name: "calc".into(),
                arguments: serde_json::json!({"n":4})
            }
        ]
    );
}

#[test]
fn elided_think_close_head_hands_off_to_structure_or_eof() {
    let response = format!(
        "private{}{}visible",
        THINK_CLOSE_HEAD.canonical, RESPONSE_OPEN.canonical
    );
    assert_eq!(
        run(&response, UnifiedParserStartingState::Reasoning).0,
        vec![
            UnifiedEvent::Reasoning {
                text: "private".into()
            },
            UnifiedEvent::Text {
                text: "visible".into()
            }
        ]
    );
    assert_eq!(
        run(
            &format!("private{}", THINK_CLOSE_HEAD.canonical),
            UnifiedParserStartingState::Reasoning
        )
        .0,
        vec![UnifiedEvent::Reasoning {
            text: "private".into()
        }]
    );
}

#[test]
fn raw_json_and_marker_like_argument_data_survive() {
    let raw = r#"{"command":"literal <|close|>call<|sep|>, <|close|>json<|sep|>, and <|open|>call data"}"#;
    let body = format!("{OPEN}json type=\"object\"{SEP}{raw}{CLOSE}json{SEP}");
    let input = format!(
        "{}{}{}",
        TOOLS_OPEN.canonical,
        call("run", "2", &body),
        TOOLS_CLOSE.canonical
    );
    let expected = vec![UnifiedEvent::ToolCall {
        name: "run".into(),
        arguments: serde_json::from_str(raw).unwrap(),
    }];
    assert_native_fragmentations(&input, &expected);
}

#[test]
fn raw_json_must_be_a_valid_object_before_call_commit() {
    for raw in ["{", r#"{"x":}"#, r#""scalar""#, "[]", "null"] {
        let body = format!("{OPEN}json type=\"object\"{SEP}{raw}{CLOSE}json{SEP}");
        let invalid = call("bad", "1", &body);
        let valid = call("good", "2", &arg("x", "number", "7"));
        let input = format!(
            "{}{}{}{}",
            TOOLS_OPEN.canonical, invalid, valid, TOOLS_CLOSE.canonical
        );

        for split in (0..=input.len()).filter(|at| input.is_char_boundary(*at)) {
            let mut parser = kimi_k3_unified(&[]);
            let mut events = parser.push(&input[..split]).unwrap();
            events.extend(parser.push(&input[split..]).unwrap());
            events.extend(parser.finish().unwrap().events);
            assert_eq!(
                events,
                vec![crate::upstream::UnifiedParserEvent::ToolCall(
                    ToolCallDelta {
                        tool_index: 0,
                        name: Some("good".into()),
                        arguments: r#"{"x":7}"#.into(),
                        complete: true,
                    }
                )],
                "raw JSON {raw:?}, split at byte {split}"
            );
            assert_eq!(parser.tool_call_id(0), Some("good:1"));
            assert_eq!(parser.tool_call_id(1), None);
        }
    }
}

#[test]
fn tool_call_id_preserves_non_positive_and_overflowing_indices() {
    let cases = [
        ("0", "weather:0"),
        ("-2", "weather:-2"),
        ("+2", "weather:1"),
        ("9223372036854775807", "weather:9223372036854775806"),
        ("-9223372036854775808", "weather:-9223372036854775808"),
    ];
    for (index, expected) in cases {
        assert_eq!(tool_call_id("weather", Some(index)), expected);
    }
}

#[test]
fn tool_call_id_preserves_non_positive_indices_at_every_split() {
    for index in ["0", "-2", "-9223372036854775808"] {
        let input = format!(
            "{}{OPEN}call tool=\"weather\" index=\"{index}\"{}{}{}{}",
            TOOLS_OPEN.canonical,
            SEP,
            arg("city", "string", "Paris"),
            CALL_CLOSE.canonical,
            TOOLS_CLOSE.canonical,
        );
        for split in input.char_indices().map(|(at, _)| at).chain([input.len()]) {
            let mut parser = kimi_k3_unified(&[]);
            parser
                .initialize_request(UnifiedParserInit::default())
                .unwrap();
            parser.push(&input[..split]).unwrap();
            parser.push(&input[split..]).unwrap();
            parser.finish().unwrap();
            assert_eq!(
                parser.tool_call_id(0),
                Some(tool_call_id("weather", Some(index)).as_str())
            );
        }
    }
}

#[test]
fn typed_string_preserves_argument_close_marker_data() {
    let value = "before<|close|>argument<|sep|>after";
    let input = format!(
        "{}{}{}",
        TOOLS_OPEN.canonical,
        call("echo", "1", &arg("value", "string", value)),
        TOOLS_CLOSE.canonical
    );
    assert_native_fragmentations(
        &input,
        &[UnifiedEvent::ToolCall {
            name: "echo".into(),
            arguments: serde_json::json!({"value": value}),
        }],
    );
}

#[test]
fn typed_string_preserves_argument_close_before_brace_or_bracket() {
    for marker in ARG_CLOSE.variants() {
        for suffix in [
            "{literal}after",
            "[literal]after",
            "{\"city\":\"Zürich\"}",
            "[\"Zürich\"]",
            "{\"city\":\"Zürich\"}after",
            "[\"Zürich\"]after",
        ] {
            let value = format!("before{marker}{suffix}");
            let input = format!(
                "{}{}{}",
                TOOLS_OPEN.canonical,
                call("echo", "1", &arg("value", "string", &value)),
                TOOLS_CLOSE.canonical
            );
            assert_native_fragmentations(
                &input,
                &[UnifiedEvent::ToolCall {
                    name: "echo".into(),
                    arguments: serde_json::json!({"value": value}),
                }],
            );
        }
    }
}

#[test]
fn typed_string_owns_every_marker_like_suffix_until_real_close() {
    let suffixes = [
        TOOLS_CLOSE.canonical,
        MESSAGE_CLOSE.canonical,
        END_OF_MSG,
        THINK_CLOSE.canonical,
        RESPONSE_CLOSE.canonical,
        ARG_OPEN.canonical,
        CALL_CLOSE.canonical,
        "{literal}[bytes]",
        "arbitrary Zürich text",
    ];
    for arg_close in ARG_CLOSE.variants() {
        for suffix in suffixes {
            let value = format!("before{arg_close}{suffix} after");
            let body = [
                arg("first", "number", "1"),
                arg("value", "string", &value),
                arg("last", "boolean", "true"),
            ]
            .concat();
            let input = format!(
                "{}{}{}",
                TOOLS_OPEN.canonical,
                call("echo", "1", &body),
                TOOLS_CLOSE.canonical
            );
            assert_native_fragmentations(
                &input,
                &[UnifiedEvent::ToolCall {
                    name: "echo".into(),
                    arguments: serde_json::json!({
                        "first": 1,
                        "value": value,
                        "last": true,
                    }),
                }],
            );
        }
    }
}

#[test]
fn call_boundary_scan_work_is_linear_for_one_character_chunks() {
    fn work(size: usize) -> (usize, usize, usize) {
        let value = "x".repeat(size);
        let input = call("echo", "1", &arg("value", "string", &value));
        let (_, header_len) = parse_call_header(&input).expect("complete call header");
        let mut boundary = KimiK3CallBoundary::new();
        boundary.begin(header_len, Mode::Tools);
        for end in (1..=input.len()).filter(|at| input.is_char_boundary(*at)) {
            let flush = end == input.len();
            let _ = boundary.advance(&input[..end], flush);
        }
        (
            boundary.scanned_bytes,
            boundary.parsed_body_bytes,
            boundary.body_parse_count,
        )
    }

    let small = work(4_096);
    let large = work(8_192);
    assert!(small.0 <= 4_096 + 160, "small scan work: {small:?}");
    assert!(large.0 <= 8_192 + 160, "large scan work: {large:?}");
    assert!(
        large.0 <= small.0 * 2,
        "scan work did not double: {small:?} -> {large:?}"
    );
    assert_eq!(small.2, 1, "small body parses: {small:?}");
    assert_eq!(large.2, 1, "large body parses: {large:?}");
    assert!(
        large.1 <= small.1 * 2,
        "parse work did not double: {small:?} -> {large:?}"
    );
}

#[test]
fn incomplete_native_call_headers_scan_linearly_and_drop_at_eof() {
    fn work(size: usize) -> (usize, Vec<UnifiedEvent>) {
        let header = format!("{OPEN}call tool=\"echo\" index=\"1\"{}", "x".repeat(size));
        let mut parser = KimiK3Native::new();
        let mut output = UnifiedParserOutput::default();
        for character in header.chars() {
            parser.buffer.push(character);
            parser.drain(false, &mut output);
        }
        let examined = parser
            .call_header_scan
            .as_ref()
            .map_or(0, |scan| scan.examined_bytes);
        parser.drain(true, &mut output);
        (examined, assemble(&output.events))
    }

    let (small_work, small_events) = work(4_096);
    let (large_work, large_events) = work(8_192);
    assert!(small_work > 0, "header scan must examine input");
    assert!(
        large_work <= small_work * 2 + 128,
        "incomplete header scan grew faster than linearly: {small_work} -> {large_work}"
    );
    assert!(
        small_events.is_empty(),
        "incomplete header leaked: {small_events:?}"
    );
    assert!(
        large_events.is_empty(),
        "incomplete header leaked: {large_events:?}"
    );
}

#[test]
fn incomplete_native_call_headers_preserve_separator_spellings_and_reset_state() {
    for separator in SEP_MARKER.variants() {
        let complete = format!(
            "before{OPEN}call tool=\"echo\" index=\"1\"{separator}{}{call_close}{}after",
            arg("value", "string", "ok"),
            TOOLS_CLOSE.canonical,
            call_close = CALL_CLOSE.canonical,
        );
        assert_native_fragmentations(
            &complete,
            &[
                UnifiedEvent::Text {
                    text: "before".into(),
                },
                UnifiedEvent::ToolCall {
                    name: "echo".into(),
                    arguments: serde_json::json!({"value":"ok"}),
                },
                UnifiedEvent::Text {
                    text: "after".into(),
                },
            ],
        );

        for prefix_end in separator
            .char_indices()
            .map(|(at, _)| at)
            .chain([separator.len()])
        {
            let incomplete = format!(
                "prose before{OPEN}call tool=\"echo\" index=\"1\"{}",
                &separator[..prefix_end]
            );
            let expected = vec![UnifiedEvent::Text {
                text: "prose before".into(),
            }];
            assert_native_fragmentations(&incomplete, &expected);
            let (events, _) = run(&incomplete, UnifiedParserStartingState::None);
            assert_eq!(events, expected, "incomplete {separator:?} header");
        }
    }

    let mut parser = kimi_k3_unified(&[]);
    parser
        .push("before<|open|>call tool=\"echo\" index=\"1\"")
        .unwrap();
    parser.finish().unwrap();
    parser.reset();
    let events = parser.push("after").unwrap();
    assert_eq!(
        assemble(&events),
        vec![UnifiedEvent::Text {
            text: "after".into()
        }]
    );
}

#[test]
fn typed_argument_parse_work_is_linear_for_size_doubling() {
    fn work(repetitions: usize) -> (usize, usize, usize) {
        reset_argument_work();
        let false_close = format!("{}literal", ARG_CLOSE.canonical);
        let body = arg("value", "string", &false_close.repeat(repetitions));
        parse_call_body(&body).expect("typed argument body");
        argument_work()
    }

    let small = work(128);
    let large = work(256);
    println!("K3 typed size work: 128={small:?}, 256={large:?}");
    assert!(small.0 > 0 && small.1 > 0, "small work: {small:?}");
    assert!(
        large.0 <= small.0 * 2 + 64,
        "scan bytes grew faster than linearly: {small:?} -> {large:?}"
    );
    assert!(
        large.1 <= small.1 * 2 + 64,
        "marker comparisons grew faster than linearly: {small:?} -> {large:?}"
    );
    assert_eq!(small.2, 1);
    assert_eq!(large.2, 1);
}

#[test]
fn typed_argument_parse_work_is_linear_in_field_count() {
    fn work(field_count: usize) -> (usize, usize, usize) {
        reset_argument_work();
        let body = (0..field_count)
            .map(|index| arg(&format!("field_{index}"), "number", &index.to_string()))
            .collect::<String>();
        let parsed = parse_call_body(&body).expect("many typed arguments");
        let value: Value = serde_json::from_str(&parsed).expect("arguments JSON");
        assert_eq!(
            value.as_object().expect("arguments object").len(),
            field_count
        );
        argument_work()
    }

    let small = work(128);
    let large = work(256);
    println!("K3 typed field work: 128={small:?}, 256={large:?}");
    assert_eq!(small.2, 128, "one map lookup per small field: {small:?}");
    assert_eq!(large.2, 256, "one map lookup per large field: {large:?}");
    assert!(
        large.0 <= small.0 * 2 + 1_024,
        "scan bytes grew faster than bytes plus fields: {small:?} -> {large:?}"
    );
    assert!(
        large.1 <= small.1 * 2 + 1_024,
        "marker comparisons grew faster than bytes plus fields: {small:?} -> {large:?}"
    );
}

#[test]
fn duplicate_typed_arguments_keep_first_position_and_last_value() {
    let body = [
        arg("first", "number", "1"),
        arg("duplicate", "string", "old"),
        arg("last", "boolean", "true"),
        arg("duplicate", "string", "new"),
    ]
    .concat();
    assert_eq!(
        parse_call_body(&body).as_deref(),
        Some(r#"{"first":1,"duplicate":"new","last":true}"#)
    );
}

#[test]
fn malformed_close_candidate_inside_string_preserves_multi_argument_split_parity() {
    let malformed_next = format!("{} broken{SEP}", ARG_OPEN.canonical);
    let value = format!(
        "before{}{}after",
        ARG_CLOSE.spaced.expect("paired marker"),
        malformed_next
    );
    let body = [
        arg("first", "string", &value),
        arg(
            "second",
            "object",
            r#"{"marker":"<|close|>argument<|sep|>"}"#,
        ),
        arg("third", "number", "3"),
    ]
    .concat();
    let input = format!(
        "{}{}{}",
        TOOLS_OPEN.canonical,
        call("echo", "1", &body),
        TOOLS_CLOSE.canonical
    );
    assert_native_fragmentations(
        &input,
        &[UnifiedEvent::ToolCall {
            name: "echo".into(),
            arguments: serde_json::json!({
                "first": value,
                "second": {"marker":"<|close|>argument<|sep|>"},
                "third": 3,
            }),
        }],
    );
}

#[test]
fn mixed_argument_close_spellings_keep_source_order() {
    let first = format!(
        "{OPEN}argument key=\"first\" type=\"string\"{SEP}one{}",
        ARG_CLOSE.spaced.expect("paired marker")
    );
    let body = format!("{first}{}", arg("second", "string", "two"));
    let input = format!(
        "{}{}{}",
        TOOLS_OPEN.canonical,
        call("echo", "1", &body),
        TOOLS_CLOSE.canonical
    );
    assert_native_fragmentations(
        &input,
        &[UnifiedEvent::ToolCall {
            name: "echo".into(),
            arguments: serde_json::json!({"first":"one","second":"two"}),
        }],
    );
}

#[test]
fn non_string_json_keeps_argument_close_inside_quoted_data() {
    for (arg_type, value, expected) in [
        (
            "object",
            r#"{"x":"before<|close|>argument<|sep|>{literal}after"}"#,
            serde_json::json!({"x":"before<|close|>argument<|sep|>{literal}after"}),
        ),
        (
            "array",
            r#"["before<|close|>argument<|sep|>[literal]after"]"#,
            serde_json::json!(["before<|close|>argument<|sep|>[literal]after"]),
        ),
    ] {
        let input = format!(
            "{}{}{}",
            TOOLS_OPEN.canonical,
            call("echo", "1", &arg("value", arg_type, value)),
            TOOLS_CLOSE.canonical
        );
        assert_native_fragmentations(
            &input,
            &[UnifiedEvent::ToolCall {
                name: "echo".into(),
                arguments: serde_json::json!({"value": expected}),
            }],
        );
    }
}

#[test]
fn typed_string_ending_in_argument_close_marker_is_not_truncated() {
    let value = "ends with <|close|>argument<|sep|>";
    let input = format!(
        "{}{}{}",
        TOOLS_OPEN.canonical,
        call("echo", "1", &arg("value", "string", value)),
        TOOLS_CLOSE.canonical
    );
    assert_native_fragmentations(
        &input,
        &[UnifiedEvent::ToolCall {
            name: "echo".into(),
            arguments: serde_json::json!({"value": value}),
        }],
    );
}

#[test]
fn typed_string_preserves_argument_and_call_close_marker_data() {
    for value in [
        "before<|close|>argument<|sep|><|close|>call<|sep|>after",
        "before<|close|> argument <|sep|><|close|> call <|sep|>after",
    ] {
        let input = format!(
            "{}{}{}",
            TOOLS_OPEN.canonical,
            call("echo", "1", &arg("value", "string", value)),
            TOOLS_CLOSE.canonical
        );
        assert_native_fragmentations(
            &input,
            &[UnifiedEvent::ToolCall {
                name: "echo".into(),
                arguments: serde_json::json!({"value": value}),
            }],
        );
    }
}

#[test]
fn typed_string_call_open_is_data_not_a_resync_boundary() {
    let value = format!(
        "before{}after",
        call("quoted", "8", &arg("nested", "string", "literal"))
    );
    let input = format!(
        "{}{}{}",
        TOOLS_OPEN.canonical,
        call("echo", "1", &arg("value", "string", value.as_str())),
        TOOLS_CLOSE.canonical
    );
    assert_native_fragmentations(
        &input,
        &[UnifiedEvent::ToolCall {
            name: "echo".into(),
            arguments: serde_json::json!({"value": value}),
        }],
    );
}

#[test]
fn call_inside_reasoning_restores_reasoning_and_quarantines_reserved_structure() {
    let input = format!(
        "{}before{}after{}",
        THINK_OPEN.canonical,
        call("echo", "1", &arg("value", "string", "ok")),
        THINK_CLOSE.canonical
    );
    assert_native_fragmentations(
        &input,
        &[
            UnifiedEvent::Reasoning {
                text: "before".into(),
            },
            UnifiedEvent::ToolCall {
                name: "echo".into(),
                arguments: serde_json::json!({"value":"ok"}),
            },
            UnifiedEvent::Reasoning {
                text: "after".into(),
            },
        ],
    );

    let quarantined = format!(
        "{}a{}b{}c{} type=\"object\"{}{{\"x\":1}}{}d{}e{}f{}g{}h{}",
        THINK_OPEN.canonical,
        arg("hidden", "string", "secret"),
        ARG_CLOSE.canonical,
        JSON_OPEN.canonical,
        SEP,
        JSON_CLOSE.canonical,
        JSON_CLOSE.canonical,
        RESPONSE_CLOSE.canonical,
        CALL_CLOSE.canonical,
        TOOLS_CLOSE.canonical,
        THINK_CLOSE.canonical,
    );
    assert_native_fragmentations(
        &quarantined,
        &[UnifiedEvent::Reasoning {
            text: "abcdefgh".into(),
        }],
    );
}

#[test]
fn missing_call_close_recovers_at_each_outer_boundary() {
    for boundary in [TOOLS_CLOSE.canonical, MESSAGE_CLOSE.canonical, END_OF_MSG] {
        let input = format!(
            "{}{OPEN}call tool=\"calc\" index=\"1\"{SEP}{}{boundary}",
            TOOLS_OPEN.canonical,
            arg("n", "number", "4")
        );
        assert_native_fragmentations(
            &input,
            &[UnifiedEvent::ToolCall {
                name: "calc".into(),
                arguments: serde_json::json!({"n":4}),
            }],
        );
    }
}

#[test]
fn missing_call_close_recovers_at_active_return_channel_close() {
    for (close, state, leading) in [
        (
            THINK_CLOSE,
            UnifiedParserStartingState::Reasoning,
            UnifiedEvent::Reasoning {
                text: "before Zürich".into(),
            },
        ),
        (
            RESPONSE_CLOSE,
            UnifiedParserStartingState::Response,
            UnifiedEvent::Text {
                text: "before Zürich".into(),
            },
        ),
    ] {
        for close in close.variants() {
            let input = format!(
                "before Zürich{OPEN}call tool=\"calc\" index=\"1\"{SEP}{}{close}after",
                arg("n", "number", "4")
            );
            assert_native_fragmentations_from_state(
                &input,
                state,
                &[
                    leading.clone(),
                    UnifiedEvent::ToolCall {
                        name: "calc".into(),
                        arguments: serde_json::json!({"n":4}),
                    },
                    UnifiedEvent::Text {
                        text: "after".into(),
                    },
                ],
            );
        }
    }
}

#[test]
fn reset_returns_the_full_uncommitted_native_envelope() {
    let input = format!(
        "{}{}{}",
        TOOLS_OPEN.canonical,
        call("calc", "1", &arg("n", "string", "Paris")),
        TOOLS_CLOSE.canonical
    );
    let call_close = input.find(CALL_CLOSE.canonical).expect("call close");
    for split in (1..=call_close).filter(|at| input.is_char_boundary(*at)) {
        let mut parser = kimi_k3_unified(&[]);
        parser.push(&input[..split]).unwrap();
        let recovered = parser.reset();
        let mut reparsed = kimi_k3_unified(&[]);
        let mut events = reparsed.push(&recovered).unwrap();
        events.extend(reparsed.push(&input[split..]).unwrap());
        events.extend(reparsed.finish().unwrap().events);
        assert_eq!(
            assemble(&events),
            vec![UnifiedEvent::ToolCall {
                name: "calc".into(),
                arguments: serde_json::json!({"n":"Paris"}),
            }],
            "reset at byte {split}"
        );
    }
}

#[test]
fn native_stream_commits_name_arguments_index_and_id_together() {
    let header = format!(
        "{}{OPEN}call tool=\"echo\" index=\"1\"{SEP}",
        TOOLS_OPEN.canonical
    );
    let mut parser = kimi_k3_unified(&[]);
    assert!(parser.push(&header).unwrap().is_empty());
    assert_eq!(parser.tool_call_id(0), None);
    let body = arg("value", "string", "Zürich");
    assert!(parser.push(&body).unwrap().is_empty());
    assert!(parser.push(CALL_CLOSE.canonical).unwrap().is_empty());
    assert_eq!(
        parser.push(TOOLS_CLOSE.canonical).unwrap(),
        vec![crate::upstream::UnifiedParserEvent::ToolCall(
            ToolCallDelta {
                tool_index: 0,
                name: Some("echo".into()),
                arguments: r#"{"value":"Zürich"}"#.into(),
                complete: true,
            }
        )]
    );
    assert_eq!(parser.tool_call_id(0), Some("echo:0"));
}

#[test]
fn assistant_message_header_and_renderer_envelope_are_stripped_at_every_split() {
    let visible = "visible Zürich";
    for header in [
        r#"<|open|>message role="assistant"<|sep|>"#,
        r#"<|open|> message role="assistant" <|sep|>"#,
    ] {
        for channel in [
            format!(
                "{}{}{}",
                RESPONSE_OPEN.canonical, visible, RESPONSE_CLOSE.canonical
            ),
            visible.to_string(),
        ] {
            let input = format!("{header}{channel}{}{}", MESSAGE_CLOSE.canonical, END_OF_MSG);
            assert_native_fragmentations(
                &input,
                &[UnifiedEvent::Text {
                    text: visible.into(),
                }],
            );
        }
    }
}

#[test]
fn legacy_and_unified_are_exact_native_projections_at_every_split() {
    let input = format!(
        "{}private{}{}{}after{}",
        THINK_OPEN.canonical,
        call("echo", "1", &arg("value", "string", "Zürich")),
        ARG_CLOSE.canonical,
        "more",
        THINK_CLOSE.canonical
    );
    for split in (0..=input.len()).filter(|at| input.is_char_boundary(*at)) {
        let mut unified = kimi_k3_unified(&[]);
        let mut unified_events = unified.push(&input[..split]).unwrap();
        unified_events.extend(unified.push(&input[split..]).unwrap());
        unified_events.extend(unified.finish().unwrap().events);

        let mut legacy = crate::upstream::KimiK3ToolStreamParser::new(&[]);
        let mut legacy_result = legacy.push(&input[..split]).unwrap();
        legacy_result.append(legacy.push(&input[split..]).unwrap());
        legacy_result.append(legacy.finish().unwrap());
        assert_eq!(
            legacy_result,
            crate::upstream::ToolParseResult::from_deltas(unified_events),
            "split at byte {split}"
        );
    }
}

#[test]
fn eof_recovers_delimiter_terminated_call_but_drops_partial_value() {
    let recovered = format!(
        "{}{OPEN}call tool=\"calc\" index=\"1\"{SEP}{}",
        TOOLS_OPEN.canonical,
        arg("n", "number", "4")
    );
    assert_eq!(
        run(&recovered, UnifiedParserStartingState::None).0,
        vec![UnifiedEvent::ToolCall {
            name: "calc".into(),
            arguments: serde_json::json!({"n":4})
        }]
    );
    let partial = format!(
        "{}{OPEN}call tool=\"calc\" index=\"1\"{SEP}{OPEN}argument key=\"n\" type=\"string\"{SEP}Par",
        TOOLS_OPEN.canonical
    );
    assert!(run(&partial, UnifiedParserStartingState::None).0.is_empty());
}

#[test]
fn malformed_first_call_resynchronizes_to_a_later_complete_call() {
    let input = format!(
        "{}{OPEN}call tool=\"bad\" index=\"1\"{SEP}not-an-argument{}{}",
        TOOLS_OPEN.canonical,
        call("good", "2", &arg("x", "number", "7")),
        TOOLS_CLOSE.canonical
    );
    assert_native_fragmentations(
        &input,
        &[UnifiedEvent::ToolCall {
            name: "good".into(),
            arguments: serde_json::json!({"x":7}),
        }],
    );
}

#[test]
fn malformed_then_valid_call_has_no_ghost_name_index_or_id() {
    let malformed = format!("{OPEN}call tool=\"bad\" index=\"1\"{SEP}not-an-argument");
    let valid = call("good", "2", &arg("x", "number", "7"));
    let input = format!(
        "{}{}{}{}",
        TOOLS_OPEN.canonical, malformed, valid, TOOLS_CLOSE.canonical
    );

    for split in (0..=input.len()).filter(|at| input.is_char_boundary(*at)) {
        let mut parser = kimi_k3_unified(&[]);
        let mut events = parser.push(&input[..split]).unwrap();
        events.extend(parser.push(&input[split..]).unwrap());
        events.extend(parser.finish().unwrap().events);
        assert_eq!(
            events,
            vec![crate::upstream::UnifiedParserEvent::ToolCall(
                ToolCallDelta {
                    tool_index: 0,
                    name: Some("good".into()),
                    arguments: r#"{"x":7}"#.into(),
                    complete: true,
                }
            )],
            "split at byte {split}"
        );
        assert_eq!(parser.tool_call_id(0), Some("good:1"));
        assert_eq!(parser.tool_call_id(1), None);
    }
}

#[test]
fn complete_call_survives_a_truncated_later_call() {
    let input = format!(
        "{}{}{OPEN}call tool=\"later\" index=\"2\"{SEP}{OPEN}argument key=\"x\" type=\"string\"{SEP}unfinished",
        TOOLS_OPEN.canonical,
        call("first", "1", &arg("x", "number", "1"))
    );
    assert_native_fragmentations(
        &input,
        &[UnifiedEvent::ToolCall {
            name: "first".into(),
            arguments: serde_json::json!({"x":1}),
        }],
    );
}

#[test]
fn reset_restarts_channels_indices_and_lifecycle() {
    let first = format!(
        "{}{}{}",
        TOOLS_OPEN.canonical,
        call("first", "1", ""),
        TOOLS_CLOSE.canonical
    );
    let second = format!(
        "{}{}{}",
        TOOLS_OPEN.canonical,
        call("second", "1", ""),
        TOOLS_CLOSE.canonical
    );
    let mut parser = kimi_k3_unified(&[]);
    assert_eq!(parser.push(&first).unwrap().len(), 1);
    assert!(parser.finish().unwrap().events.is_empty());
    assert!(parser.push("later").is_err());
    assert_eq!(parser.reset(), "");
    let events = parser.push(&second).unwrap();
    let crate::upstream::UnifiedParserEvent::ToolCall(call) = &events[0] else {
        panic!("expected a tool call")
    };
    assert_eq!(call.tool_index, 0);
    assert_eq!(parser.tool_call_id(0), Some("second:0"));
}

fn kimi_k3_unified(tools: &[Tool]) -> Box<dyn UnifiedParser> {
    Box::new(KimiK3Parser::new_with_streaming(tools, false))
}

fn kimi_k3_streaming_unified(tools: &[Tool]) -> Box<dyn UnifiedParser> {
    Box::new(KimiK3Parser::new(tools))
}
