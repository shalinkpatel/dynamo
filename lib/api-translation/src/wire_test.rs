use super::*;

#[test]
fn cc_frame_is_bare_data() {
    assert_eq!(sse_frame(None, "{\"x\":1}"), "data: {\"x\":1}\n\n");
}

#[test]
fn messages_frame_names_event() {
    assert_eq!(
        sse_frame(Some("message_stop"), "{}"),
        "event: message_stop\ndata: {}\n\n"
    );
}

/// A client that decodes to text before splitting lines breaks on these; escaping keeps the JSON
/// semantically identical and the frame on one line.
#[test]
fn text_line_breaks_are_escaped_not_emitted_raw() {
    let framed = to_json_string(&serde_json::json!({"content": "a\u{2028}b\u{85}c\u{2029}d"}));
    assert_eq!(framed, r#"{"content":"a\u2028b\u0085c\u2029d"}"#);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&framed).unwrap()["content"],
        "a\u{2028}b\u{85}c\u{2029}d",
        "escaping must not change the decoded value"
    );
}

/// Ids are unique within the process (monotonic counter) and carry a process-unique prefix,
/// so two replicas never mint the same `resp_`/`msg_` id.
#[test]
fn id_seq_is_prefixed_and_monotonic() {
    let a = next_id_seq();
    let b = next_id_seq();
    assert_ne!(a, b);
    assert!(a.chars().all(|c| c.is_ascii_hexdigit()), "{a}");
    // Same process prefix (the counter is at most a few trailing hex digits here).
    assert!(a.len() > 8, "{a}");
    assert_eq!(a[..8], b[..8], "{a} vs {b}");
}

#[test]
fn multiline_data_prefixes_each_line() {
    assert_eq!(sse_frame(None, "a\nb"), "data: a\ndata: b\n\n");
}
