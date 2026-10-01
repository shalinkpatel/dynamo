use super::CcRequest;

/// The passthrough contract that used to sit on the wire type: unknown request fields survive
/// a parse/serialize round trip, in the client's order, alongside the modeled fields.
#[test]
fn unmodeled_request_fields_round_trip_in_client_order() {
    let input = serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "chat_template_kwargs": {"enable_thinking": true},
        "guided_json": {"type": "object"},
        "zeta_first": 1,
        "alpha_second": 2
    });
    let request: CcRequest = serde_json::from_value(input.clone()).unwrap();

    assert_eq!(request.model, "test-model");
    assert_eq!(request.messages.len(), 1);
    assert_eq!(
        request.unmodeled.keys().collect::<Vec<_>>(),
        [
            "chat_template_kwargs",
            "guided_json",
            "zeta_first",
            "alpha_second"
        ]
    );

    let output = serde_json::to_value(&request).unwrap();
    assert_eq!(output["model"], "test-model");
    assert_eq!(
        output["chat_template_kwargs"],
        input["chat_template_kwargs"]
    );
    assert_eq!(output["guided_json"], input["guided_json"]);
    assert_eq!(output["zeta_first"], 1);
    assert_eq!(output["alpha_second"], 2);
    // No key is emitted twice: the modeled fields are not mirrored into the catch-all.
    let text = serde_json::to_string(&request).unwrap();
    assert_eq!(text.matches("\"model\"").count(), 1, "{text}");
}
