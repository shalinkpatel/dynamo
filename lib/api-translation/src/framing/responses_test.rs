use super::*;

#[test]
fn react_cap_completes_rather_than_incomplete() {
    let (status, details) = response_status(Termination::ReactCapExhausted);
    assert_eq!(status, Status::Completed);
    assert!(details.is_none());
}

/// A length-truncated model turn is `incomplete` with the spec literal, not `completed`.
#[test]
fn length_termination_is_incomplete_with_spec_reason() {
    let (status, details) = response_status(Termination::Model(FinishReason::Length));
    assert_eq!(status, Status::Incomplete);
    assert_eq!(details.unwrap().reason, "max_output_tokens");
    let (status, _) = response_status(Termination::Model(FinishReason::Stop));
    assert_eq!(status, Status::Completed);
}

#[test]
fn responses_usage_maps_cache_reads_into_input_token_details() {
    let usage = CompletionUsage {
        prompt_tokens: 1000,
        completion_tokens: 50,
        total_tokens: 1050,
        prompt_tokens_details: Some(dynamo_protocols::types::PromptTokensDetails {
            cached_tokens: Some(900),
            audio_tokens: None,
            ..Default::default()
        }),
        completion_tokens_details: None,
    };
    let mapped = responses_usage(&usage);
    assert_eq!(mapped.input_tokens, 1000);
    assert_eq!(mapped.input_tokens_details.cached_tokens, 900);
    assert_eq!(mapped.output_tokens_details.reasoning_tokens, 0);
}

fn frame_body(frame: &str) -> serde_json::Value {
    let (event, data) = frame.split_once('\n').unwrap();
    let name = event.strip_prefix("event: ").unwrap();
    let body: serde_json::Value =
        serde_json::from_str(data.strip_prefix("data: ").unwrap().trim_end()).unwrap();
    assert_eq!(body["type"], name);
    body
}

/// Resolves one hoisted name back to its `(name, namespace)` declaration; every other call
/// passes through as itself (default `resolve_tool_identity`).
struct NamespaceStampAdapter;
impl CodingAdapter for NamespaceStampAdapter {
    fn render_tool_call(&self, _call: &ToolCallToRender<'_>) -> Option<RenderedToolCall> {
        None
    }
    fn resolve_tool_identity(
        &self,
        tool_name: &str,
    ) -> Option<crate::coding_adapter::ToolIdentity> {
        (tool_name == "multi_agent_v1__spawn_agent").then(|| crate::coding_adapter::ToolIdentity {
            name: "spawn_agent".to_string(),
            namespace: "multi_agent_v1".to_string(),
        })
    }
}

/// A namespaced tool's streamed call is stamped back to the client's `(name, namespace)`
/// registry key, not the hoisted wire name codex dispatched it under.
#[test]
fn responses_stamps_a_namespaced_call_with_its_declaration() {
    let mut framing = ResponsesFraming::new(
        "m".to_string(),
        Some(Box::new(NamespaceStampAdapter)),
        ResponsesParams::default(),
    );
    let frames = framing.on_chunk(&SemanticChunk::ToolCall(crate::model::ToolCall {
        id: "call_1".to_string(),
        name: "multi_agent_v1__spawn_agent".to_string(),
        args: json!({}),
        raw_args: "{}".to_string(),
    }));
    let added = frames
        .iter()
        .map(|frame| frame_body(frame))
        .find(|body| body["type"] == "response.output_item.added")
        .expect("an output_item.added frame");
    assert_eq!(added["item"]["type"], "function_call");
    assert_eq!(added["item"]["name"], "spawn_agent");
    assert_eq!(added["item"]["namespace"], "multi_agent_v1");
}

#[test]
fn reasoning_items_carry_the_reasoning_text_discriminator() {
    let mut framing = ResponsesFraming::new("m".to_string(), None, ResponsesParams::default());
    let frames = framing.on_chunk(&SemanticChunk::ThinkingDelta("thinking".into()));
    let added = frames
        .iter()
        .map(|frame| frame_body(frame))
        .find(|body| body["type"] == "response.output_item.added")
        .expect("an output_item.added frame");
    assert_eq!(added["item"]["type"], "reasoning");
    assert_eq!(added["item"]["content"][0]["type"], "reasoning_text");
}

/// The other applier: `finished_body` tags the terminal envelope's `response.output`, which a
/// client parses on replay even when it saw every streaming frame.
#[test]
fn terminal_envelope_reasoning_carries_the_reasoning_text_discriminator() {
    let mut framing = ResponsesFraming::new("m".to_string(), None, ResponsesParams::default());
    let mut frames = framing.on_chunk(&SemanticChunk::ThinkingDelta("thinking".into()));
    frames.extend(framing.finish(
        Termination::Model(dynamo_protocols::types::FinishReason::Stop),
        &CompletionUsage::default(),
        &[],
    ));
    let terminal = frames
        .iter()
        .map(|frame| frame_body(frame))
        .find(|body| body["type"] == "response.completed")
        .expect("a response.completed frame");
    let reasoning = terminal["response"]["output"]
        .as_array()
        .expect("output items")
        .iter()
        .find(|item| item["type"] == "reasoning")
        .expect("a reasoning output item");
    assert_eq!(reasoning["content"][0]["type"], "reasoning_text");
    assert_eq!(reasoning["content"][0]["text"], "thinking");
}

/// One item id per item, on every event of its lifecycle — a client matching deltas to items
/// by `item_id` (not `output_index`) must see them attach.
#[test]
fn item_id_is_stable_across_added_deltas_done_and_envelope() {
    let mut framing = ResponsesFraming::new("m".to_string(), None, ResponsesParams::default());
    let mut frames = Vec::new();
    frames.extend(framing.on_chunk(&SemanticChunk::TextDelta("hi".to_string())));
    frames.extend(framing.finish(
        Termination::Model(dynamo_protocols::types::FinishReason::Stop),
        &CompletionUsage::default(),
        &[],
    ));
    let bodies: Vec<serde_json::Value> = frames.iter().map(|f| frame_body(f)).collect();
    let id_of = |event: &str, field: &str| {
        let body = bodies
            .iter()
            .find(|b| b["type"] == event)
            .unwrap_or_else(|| panic!("no {event}"));
        match field {
            "item" => body["item"]["id"].as_str().unwrap().to_string(),
            _ => body[field].as_str().unwrap().to_string(),
        }
    };
    let added_id = id_of("response.output_item.added", "item");
    assert_eq!(id_of("response.content_part.added", "item_id"), added_id);
    assert_eq!(id_of("response.output_text.delta", "item_id"), added_id);
    assert_eq!(id_of("response.output_text.done", "item_id"), added_id);
    assert_eq!(id_of("response.output_item.done", "item"), added_id);
    let envelope = bodies.last().unwrap();
    assert_eq!(envelope["type"], "response.completed");
    assert_eq!(
        envelope["response"]["output"][0]["id"].as_str().unwrap(),
        added_id
    );
    // The streaming placeholder is in progress; only the close marks it completed.
    let added = bodies
        .iter()
        .find(|b| b["type"] == "response.output_item.added")
        .unwrap();
    assert_eq!(added["item"]["status"], "in_progress");
    let done = bodies
        .iter()
        .find(|b| b["type"] == "response.output_item.done")
        .unwrap();
    assert_eq!(done["item"]["status"], "completed");
}

/// Continues `sequence_number` from the frames already sent (shape rationale on the impl).
#[test]
fn error_sse_frame_is_a_response_failed_envelope_with_a_live_sequence_number() {
    let mut framing = ResponsesFraming::new("m".to_string(), None, ResponsesParams::default());
    let frames_before_error = framing.on_chunk(&SemanticChunk::TextDelta("hi".into()));
    let error_payload = framing.error_sse_frame(
        ErrorClass::Internal,
        Some("model_streamed_error"),
        "upstream unavailable",
    );
    let terminal = error_payload
        .split("\n\n")
        .filter(|frame| !frame.is_empty())
        .last()
        .unwrap()
        .to_string()
        + "\n\n";
    assert!(
        terminal.starts_with("event: response.failed\n"),
        "{terminal}"
    );
    let body = frame_body(&terminal);
    assert_eq!(body["type"], "response.failed");
    assert_eq!(body["response"]["status"], "failed");
    assert_eq!(body["response"]["error"]["code"], "model_streamed_error");
    assert_eq!(body["response"]["error"]["message"], "upstream unavailable");
    assert_eq!(
        body["response"]["output"][0]["content"][0]["text"], "hi",
        "the open item's partial text rides the failed envelope"
    );
    assert!(
        body["sequence_number"].as_u64().unwrap() > frames_before_error.len() as u64,
        "sequence continues past the close frames"
    );
}
mod graft_tests {
    use super::*;

    /// The echo is built from the raw body, so it must accept every effort the ingress accepts.
    /// Parsing it into a type that cannot spell one drops the whole block leniently, taking
    /// `summary` with it and withholding the reasoning item the client asked for.
    #[test]
    fn echoed_reasoning_preserves_max_effort() {
        let params = ResponsesParams::from_body(&json!({
            "reasoning": {"effort": "max", "summary": "auto"}
        }));
        let reasoning = params.reasoning.expect("the block must survive");
        assert_eq!(
            reasoning.effort,
            Some(dynamo_protocols::types::ReasoningEffort::Max),
            "`max` echoes unchanged"
        );
        assert_eq!(
            reasoning.summary,
            Some(dynamo_protocols::types::responses::ReasoningSummary::Auto)
        );

        let off_ladder = ResponsesParams::from_body(&json!({
            "reasoning": {"effort": "turbo", "summary": "auto"}
        }))
        .reasoning
        .expect("the block must survive an unknown effort too");
        assert_eq!(off_ladder.effort, None, "nothing to echo for a non-level");
        assert_eq!(
            off_ladder.summary,
            Some(dynamo_protocols::types::responses::ReasoningSummary::Auto)
        );

        // The halves are independent in both directions: a summary this type
        // cannot spell must not discard the effort either.
        let bad_summary = ResponsesParams::from_body(&json!({
            "reasoning": {"effort": "high", "summary": "verbose"}
        }))
        .reasoning
        .expect("the block must survive an unknown summary");
        assert_eq!(
            bad_summary.effort,
            Some(dynamo_protocols::types::ReasoningEffort::High)
        );
        assert_eq!(bad_summary.summary, None);
    }

    fn params() -> ResponsesParams {
        ResponsesParams::from_body(&json!({
            "model": "m",
            "input": "hi",
            "temperature": 0.5,
            "store": true,
            "presence_penalty": 0.75,
            "frequency_penalty": 0.25,
            "top_logprobs": 5,
            "metadata": {"job": "x"},
            "service_tier": "auto",
            "tools": [{"type": "function", "name": "get_weather", "parameters": {"type": "object"}}],
        }))
    }

    fn frame_body(frame: &str) -> Value {
        let (event, data) = frame.split_once('\n').unwrap();
        let name = event.strip_prefix("event: ").unwrap();
        let body: Value =
            serde_json::from_str(data.strip_prefix("data: ").unwrap().trim_end()).unwrap();
        assert_eq!(body["type"], name);
        body
    }

    /// The terminal envelope echoes request params with the spec's defaults for omitted fields and
    /// the spec patch (present-as-null required fields, `presence_penalty`/`frequency_penalty`/
    /// `store` injection, `usage.input_tokens_details.cache_write_tokens` defaulted to 0 for
    /// openai-python >= 2.53 typed clients).
    #[test]
    fn terminal_envelope_echoes_params_with_spec_defaults_and_patch() {
        let mut framing = ResponsesFraming::new("m".to_string(), None, params());
        let mut frames = framing.on_chunk(&SemanticChunk::TextDelta("hi".into()));
        frames.extend(framing.finish(
            Termination::Model(FinishReason::Stop),
            &CompletionUsage::default(),
            &[],
        ));
        let terminal = frames
            .iter()
            .map(|frame| frame_body(frame))
            .find(|body| body["type"] == "response.completed")
            .expect("a response.completed frame");
        let response = &terminal["response"];
        assert_eq!(response["temperature"], 0.5, "param echoed");
        assert_eq!(response["top_p"], 1.0, "spec default");
        assert_eq!(response["tool_choice"], "auto", "spec default");
        assert_eq!(response["truncation"], "disabled", "spec default");
        assert_eq!(
            response["top_logprobs"], 5,
            "param echoed, not the spec default"
        );
        assert_eq!(
            response["metadata"],
            json!({"job": "x"}),
            "param echoed verbatim"
        );
        assert_eq!(response["service_tier"], "auto", "param echoed");
        assert_eq!(response["store"], true, "injected from the request");
        assert_eq!(
            response["presence_penalty"], 0.75,
            "injected from the request"
        );
        assert_eq!(
            response["frequency_penalty"], 0.25,
            "injected from the request"
        );
        assert_eq!(response["tools"][0]["name"], "get_weather");
        assert_eq!(response["tools"][0]["strict"], true, "normalized");
        for key in dynamo_protocols::types::responses::SPEC_NULLABLE_REQUIRED_RESPONSE_FIELDS {
            assert!(
                response.get(*key).is_some(),
                "spec-nullable-required field {key} must be present"
            );
        }
        assert_eq!(
            response["usage"]["input_tokens_details"]["cache_write_tokens"], 0,
            "cache_write_tokens must be explicit"
        );
    }

    /// A terminal backend error emits `response.failed` carrying whatever output streamed, with the
    /// typed `error.code` string clients classify on.
    #[test]
    fn backend_error_finishes_as_response_failed_with_partial_output_and_typed_code() {
        let cases = [
            (Some(429), "anything", "rate_limit_exceeded"),
            (
                Some(400),
                "Input length 100 exceeds the maximum allowed input length of 10 tokens.",
                "context_length_exceeded",
            ),
            (
                Some(400),
                "maximum context length is 8192 tokens",
                "context_length_exceeded",
            ),
            (Some(500), "boom", "server_error"),
        ];
        for (http_status, message, expected_code) in cases {
            let mut framing =
                ResponsesFraming::new("m".to_string(), None, ResponsesParams::default());
            framing.on_chunk(&SemanticChunk::TextDelta("partial".into()));
            let frames = framing.finish_with_backend_error(&BackendError {
                message: message.to_string(),
                http_status,
                error_code: None,
            });
            let terminal = frames
                .iter()
                .map(|frame| frame_body(frame))
                .find(|body| body["type"] == "response.failed")
                .unwrap_or_else(|| panic!("{message}: no response.failed frame"));
            let response = &terminal["response"];
            assert_eq!(response["status"], "failed");
            assert_eq!(response["error"]["code"], expected_code, "{message}");
            assert_eq!(response["error"]["message"], message);
            assert_eq!(
                response["output"][0]["content"][0]["text"], "partial",
                "the streamed partial output must ride the failure envelope"
            );
            assert_eq!(
                response["output"][0]["status"], "incomplete",
                "the in-flight item closes incomplete, not completed"
            );
        }
    }

    /// A truncation-shaped backend error ("cutoff by max_tokens") is a length finish the worker
    /// mispresented as a failure: re-presented as the spec incomplete shape, never `failed`.
    #[test]
    fn truncation_shaped_backend_error_rescues_to_incomplete() {
        let mut framing = ResponsesFraming::new("m".to_string(), None, ResponsesParams::default());
        framing.on_chunk(&SemanticChunk::TextDelta("partial".into()));
        let frames = framing.finish_with_backend_error(&BackendError {
            message: "model error: Tool calls cutoff by max_tokens.".to_string(),
            http_status: Some(400),
            error_code: None,
        });
        let terminal = frame_body(frames.last().unwrap());
        assert_eq!(terminal["type"], "response.incomplete");
        assert_eq!(
            terminal["response"]["incomplete_details"]["reason"],
            "max_output_tokens"
        );
        assert!(terminal["response"]["error"].is_null(), "not a failure");
        assert_eq!(
            terminal["response"]["output"][0]["content"][0]["text"],
            "partial"
        );
        assert_eq!(terminal["response"]["output"][0]["status"], "incomplete");
    }
}
