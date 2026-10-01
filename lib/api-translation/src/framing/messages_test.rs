use super::*;

#[test]
fn buffered_web_search_uses_anthropic_blocks_and_drops_page_text() {
    use crate::test_utils::FakeMessagesSearchAdapter;

    let name = "baseten__provider__search";
    let adapter = FakeMessagesSearchAdapter { tool_name: name };
    let transcript = vec![
        serde_json::from_value(serde_json::json!({
            "role": "assistant",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": name, "arguments": "{\"search_queries\":[\"rust\"]}"}
            }]
        }))
        .unwrap(),
        serde_json::from_value(serde_json::json!({
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "{\"payload\":{\"results\":[{\"url\":\"https://example.com/rust\",\"title\":\"Rust\",\"text\":\"hidden\"}]}}"
        }))
        .unwrap(),
    ];
    let blocks = serde_json::to_value(transcript_blocks(
        &transcript,
        Some(&adapter),
        &HashSet::new(),
    ))
    .unwrap();
    assert_eq!(blocks[0]["type"], "server_tool_use");
    assert_eq!(blocks[0]["name"], "web_search");
    assert_eq!(blocks[0]["id"], "srvtoolu_call_1");
    assert_eq!(blocks[0]["input"], serde_json::json!({"query": "rust"}));
    assert_eq!(blocks[1]["type"], "web_search_tool_result");
    assert_eq!(blocks[1]["tool_use_id"], blocks[0]["id"]);
    assert_eq!(
        blocks[1]["content"],
        serde_json::json!([{
            "type": "web_search_result",
            "title": "Rust",
            "url": "https://example.com/rust"
        }])
    );
    assert!(blocks[1]["content"][0].get("text").is_none());
}

/// The failed-call path: `is_error` in the transcript reaches both the call's status and the
/// result block, and neither is exercised by the success test above.
#[test]
fn buffered_failed_web_search_renders_the_error_result_block() {
    use crate::test_utils::FakeMessagesSearchAdapter;

    let name = "baseten__provider__search";
    let adapter = FakeMessagesSearchAdapter { tool_name: name };
    let transcript = vec![
        serde_json::from_value(serde_json::json!({
            "role": "assistant",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": name, "arguments": "{\"search_queries\":[\"rust\"]}"}
            }]
        }))
        .unwrap(),
        serde_json::from_value(serde_json::json!({
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "upstream refused"
        }))
        .unwrap(),
    ];
    let failed = HashSet::from(["call_1"]);
    let blocks =
        serde_json::to_value(transcript_blocks(&transcript, Some(&adapter), &failed)).unwrap();
    assert_eq!(blocks[0]["type"], "server_tool_use");
    assert_eq!(blocks[1]["type"], "web_search_tool_result");
    assert_eq!(
        blocks[1]["content"],
        serde_json::json!({
            "type": "web_search_tool_result_error",
            "error_code": "unavailable"
        })
    );
}

/// The generic-block fallback is load-bearing: an adapter is present but this call is not one
/// it expanded, so it must render as a plain `tool_use` under the provider's own name.
#[test]
fn buffered_non_search_call_falls_back_to_the_generic_block() {
    use crate::test_utils::FakeMessagesSearchAdapter;

    let adapter = FakeMessagesSearchAdapter {
        tool_name: "baseten__provider__search",
    };
    let transcript = vec![
        serde_json::from_value(serde_json::json!({
            "role": "assistant",
            "tool_calls": [{
                "id": "call_9",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\":\"berlin\"}"}
            }]
        }))
        .unwrap(),
        serde_json::from_value(serde_json::json!({
            "role": "tool", "tool_call_id": "call_9", "content": "sunny"
        }))
        .unwrap(),
    ];
    let blocks = serde_json::to_value(transcript_blocks(
        &transcript,
        Some(&adapter),
        &HashSet::new(),
    ))
    .unwrap();
    assert_eq!(blocks[0]["type"], "tool_use");
    assert_eq!(blocks[0]["name"], "get_weather");
    assert_eq!(blocks[0]["id"], "call_9");
    assert_eq!(blocks[1]["type"], "tool_result");
    assert_eq!(blocks[1]["tool_use_id"], "call_9");
}

/// Both degrade silently by design, so a test is the only witness: a result that was never JSON
/// yields no citations, and args that do not parse fall back to the generic block.
#[test]
fn buffered_unparseable_content_and_args_degrade_without_inventing_data() {
    use crate::test_utils::FakeMessagesSearchAdapter;

    let name = "baseten__provider__search";
    let adapter = FakeMessagesSearchAdapter { tool_name: name };
    let transcript = vec![
        serde_json::from_value(serde_json::json!({
            "role": "assistant",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": name, "arguments": "{\"search_queries\":[\"rust\"]}"}
            }]
        }))
        .unwrap(),
        serde_json::from_value(serde_json::json!({
            "role": "tool", "tool_call_id": "call_1", "content": "not json at all"
        }))
        .unwrap(),
    ];
    let blocks = serde_json::to_value(transcript_blocks(
        &transcript,
        Some(&adapter),
        &HashSet::new(),
    ))
    .unwrap();
    assert_eq!(blocks[1]["type"], "web_search_tool_result");
    assert_eq!(blocks[1]["content"], serde_json::json!([]));

    // Args that do not parse decline the adapter's block, which is what leaves the generic one
    // standing. Asserted on the adapter directly: reaching it through the transcript would trip
    // `tool_call_input`'s debug assert first, since unparseable history args are a fold bug.
    assert!(
        adapter
            .render_tool_call(&ToolCallToRender {
                tool_name: name,
                id: "call_2",
                args: "{not json",
                status: ToolCallStatus::Completed,
            })
            .is_none()
    );
}

/// The SSE `event:` line and the body's own `type` must agree, or the Anthropic SDK ignores the
/// event. `event_name` restates the vendored enum's serde tags, so this pins the two together:
/// a re-vendor that renames a variant fails here instead of silently shipping a mismatch.
#[test]
fn event_name_matches_the_serialized_type_for_every_variant() {
    let message = AnthropicMessageResponse {
        id: String::new(),
        object_type: MESSAGE_OBJECT.to_string(),
        role: ASSISTANT_ROLE.to_string(),
        content: vec![],
        model: String::new(),
        stop_reason: None,
        stop_sequence: None,
        usage: AnthropicUsage::default(),
    };
    let every_variant = [
        AnthropicStreamEvent::MessageStart { message },
        AnthropicStreamEvent::ContentBlockStart {
            index: 0,
            content_block: AnthropicResponseContentBlock::Text {
                text: String::new(),
                citations: None,
            },
        },
        AnthropicStreamEvent::ContentBlockDelta {
            index: 0,
            delta: AnthropicDelta::TextDelta {
                text: String::new(),
            },
        },
        AnthropicStreamEvent::ContentBlockStop { index: 0 },
        AnthropicStreamEvent::MessageDelta {
            delta: AnthropicMessageDeltaBody {
                stop_reason: None,
                stop_sequence: None,
            },
            usage: AnthropicUsage::default(),
        },
        AnthropicStreamEvent::MessageStop {},
        AnthropicStreamEvent::Ping {},
    ];
    for event in every_variant {
        let body: serde_json::Value = serde_json::from_str(&to_json_string(&event)).unwrap();
        assert_eq!(body["type"], event_name(&event), "body: {body}");
    }
}

use dynamo_protocols::types::PromptTokensDetails;

fn cc_usage(prompt_tokens: u32, cached_tokens: Option<u32>) -> CompletionUsage {
    CompletionUsage {
        prompt_tokens,
        completion_tokens: 7,
        total_tokens: prompt_tokens + 7,
        prompt_tokens_details: cached_tokens.map(|cached| PromptTokensDetails {
            cached_tokens: Some(cached),
            audio_tokens: None,
        }),
        completion_tokens_details: None,
    }
}

#[test]
fn input_tokens_excludes_cache_reads() {
    let usage = anthropic_usage(&cc_usage(1000, Some(900)));
    assert_eq!(usage.input_tokens, 100);
    assert_eq!(usage.cache_read_input_tokens, Some(900));
}

/// Explicit `0`, never absent: downstream metering must not read a missing field as "the whole
/// prompt was written to cache".
#[test]
fn cache_creation_input_tokens_is_explicit_zero() {
    let usage = anthropic_usage(&cc_usage(1000, Some(900)));
    assert_eq!(usage.cache_creation_input_tokens, Some(0));
    let wire = serde_json::to_value(&usage).unwrap();
    assert_eq!(wire["cache_creation_input_tokens"], 0, "{wire}");
}

#[test]
fn input_tokens_is_the_cc_total_when_nothing_was_cached() {
    for cached_tokens in [None, Some(0)] {
        let usage = anthropic_usage(&cc_usage(1000, cached_tokens));
        assert_eq!(usage.input_tokens, 1000, "cached_tokens: {cached_tokens:?}");
    }
}

/// A backend over-reporting cache reads must not wrap the subtraction into a huge `input_tokens`.
#[test]
fn cache_reads_above_the_prompt_total_are_clamped() {
    let usage = anthropic_usage(&cc_usage(100, Some(1000)));
    assert_eq!(usage.input_tokens, 0);
    assert_eq!(usage.cache_read_input_tokens, Some(100));
}
