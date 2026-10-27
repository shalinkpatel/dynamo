use serde_json::{Map, Value, json};

use super::*;

fn function(value: Value) -> Map<String, Value> {
    serde_json::from_value(value).unwrap()
}

#[test]
fn tool_schema_text_strips_deferral_keys_and_escapes_html() {
    let text = tool_schema_text(&function(json!({
        "name": "f", "description": "<a> & 'b'", "parameters": {"type": "object"},
        "defer_loading": true, "strict": true,
    })));
    assert_eq!(
        text,
        "<tools>\n{\"name\": \"f\", \"description\": \"\\u003ca\\u003e \\u0026 \\u0027b\\u0027\", \
         \"parameters\": {\"type\":\"object\"}}\n</tools>"
    );
}

#[test]
fn may_be_fix_tool_schema_completes_an_object_schema() {
    let mut fixed = function(json!({"name": "f", "parameters": {}}));
    may_be_fix_tool_schema(&mut fixed);
    assert_eq!(
        fixed["parameters"],
        json!({"type": "object", "properties": {}})
    );
}

#[test]
fn may_be_fix_tool_schema_leaves_declared_and_non_object_schemas_alone() {
    let mut declared =
        function(json!({"name": "f", "parameters": {"type": "string", "properties": {"x": {}}}}));
    may_be_fix_tool_schema(&mut declared);
    assert_eq!(
        declared["parameters"],
        json!({"type": "string", "properties": {"x": {}}})
    );
    let mut absent = function(json!({"name": "f"}));
    may_be_fix_tool_schema(&mut absent);
    assert!(!absent.contains_key("parameters"));
}
