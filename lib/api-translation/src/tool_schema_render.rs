//! The canonical tool-schema renderer for deferred-tool references (Messages `tool_reference`
//! blocks here, Responses `tool_search_output` delivery in tool-bank): one compact-JSON line, HTML-significant characters escaped, `defer_loading` and
//! `strict` stripped — the key filter the model's chat template applies, so injected text is
//! byte-consistent with what the template renders. The authoritative envelope is
//! `lib/gwp/vendored_tokenizers/glm5.2/chat_template.jinja`.

use serde_json::Value;

/// dynamo parity: an object `parameters` gains a missing `"type": "object"` and/or
/// `"properties": {}` (a non-object or absent `parameters` is untouched).
pub fn may_be_fix_tool_schema(function: &mut serde_json::Map<String, Value>) {
    let Some(parameters) = function
        .get_mut("parameters")
        .and_then(Value::as_object_mut)
    else {
        return;
    };
    if !parameters.contains_key("type") {
        parameters.insert("type".to_string(), Value::String("object".to_string()));
    }
    if !parameters.contains_key("properties") {
        parameters.insert(
            "properties".to_string(),
            Value::Object(serde_json::Map::new()),
        );
    }
}

/// A function envelope as the `<tools>`-wrapped schema text a deferred-tool render injects.
pub fn tool_schema_text(function: &serde_json::Map<String, Value>) -> String {
    format!("<tools>\n{}\n</tools>", tool_schema_json(function))
}

fn tool_schema_json(function: &serde_json::Map<String, Value>) -> String {
    let mut out = String::from("{");
    let mut first = true;
    for (key, value) in function {
        if key == "defer_loading" || key == "strict" {
            continue;
        }
        if !first {
            out.push_str(", ");
        }
        first = false;
        out.push('"');
        out.push_str(key);
        out.push_str("\": ");
        out.push_str(&tojson_compact(value));
    }
    out.push('}');
    out
}

fn tojson_compact(value: &Value) -> String {
    let mut out = String::new();
    for c in serde_json::to_string(value)
        .expect("a serde_json::Value always serializes")
        .chars()
    {
        match c {
            '<' => out.push_str("\\u003c"),
            '>' => out.push_str("\\u003e"),
            '&' => out.push_str("\\u0026"),
            '\'' => out.push_str("\\u0027"),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
#[path = "tool_schema_render_test.rs"]
mod tests;
