//! Client-executed tool discovery, normalized before protocol deserialization.
//!
//! `defer_loading` markers are honored tool-level and function-nested only; toolset-level markers
//! (`mcp_toolset` `default_config`/`configs`, computer/browser toolset member configs) are
//! unsupported: those entries are Anthropic-native server tools, refused at intake. One policy
//! for every stale pointer — an undeclared `tool_reference` (Messages), an orphan
//! `tool_search_output` (here), an unresolvable replayed namespace: warn and render explicitly,
//! never refuse, never silently pass; one stale pointer must not poison a session replay.

use std::collections::{HashMap, HashSet};

use serde_json::{Value, json};

use super::coding_adapter::{
    CodingAdapter, RenderedToolCall, ToolCallStatus, ToolCallToRender, ToolIdentity,
    ToolResultToRender,
};
use crate::tool_schema_render::{may_be_fix_tool_schema, tool_schema_text};

#[derive(Default)]
pub(crate) struct ToolSearchAdapter {
    native_search: bool,
    namespaced_tools: HashMap<String, ToolIdentity>,
    inner: Option<Box<dyn CodingAdapter>>,
}

impl ToolSearchAdapter {
    pub(crate) fn wrap(
        self,
        inner: Option<Box<dyn CodingAdapter>>,
    ) -> Option<Box<dyn CodingAdapter>> {
        if !self.native_search && self.namespaced_tools.is_empty() {
            return inner;
        }
        Some(Box::new(Self { inner, ..self }))
    }
}

pub(crate) fn rewrite_responses(body: &mut Value) -> Result<ToolSearchAdapter, String> {
    let mut adapter = ToolSearchAdapter::default();
    let mut delivered = Vec::new();
    let mut search_calls = HashSet::new();
    let mut bare_names: HashSet<String> = body
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|tool| tool["type"] == "function")
        .filter_map(|tool| tool["name"].as_str().map(str::to_owned))
        .collect();
    if let Some(items) = body.get_mut("input").and_then(Value::as_array_mut) {
        for item in items {
            match item.get("type").and_then(Value::as_str) {
                Some("tool_search_call") => {
                    validate_execution(item)?;
                    let id = replay_item_id(item)?.to_owned();
                    if item.get("execution").and_then(Value::as_str) == Some("server") {
                        return Err("server-executed tool_search_call is unsupported".into());
                    }
                    let arguments = match item.get("arguments") {
                        Some(Value::String(arguments)) => arguments.clone(),
                        Some(arguments) if arguments.is_object() => arguments.to_string(),
                        Some(Value::Null) | None => json!({}).to_string(),
                        _ => {
                            return Err(
                                "tool_search_call arguments must be an object or string".into()
                            );
                        }
                    };
                    if !search_calls.insert(id.clone()) {
                        return Err(format!("duplicate unresolved tool_search_call {id:?}"));
                    }
                    *item = json!({"type": "function_call", "call_id": id,
                        "name": "tool_search", "arguments": arguments});
                }
                Some("tool_search_output") => {
                    validate_execution(item)?;
                    let id = replay_item_id(item)?.to_owned();
                    if !search_calls.remove(&id) {
                        tracing::warn!(
                            "tool_search_output {id:?} has no preceding tool_search_call"
                        );
                    }
                    if item.get("execution").and_then(Value::as_str) == Some("server") {
                        return Err("server-executed tool_search_output is unsupported".into());
                    }
                    let tools = item
                        .get("tools")
                        .and_then(Value::as_array)
                        .ok_or("tool_search_output requires a tools array")?;
                    let mut schemas = Vec::new();
                    bare_names.extend(
                        tools
                            .iter()
                            .filter(|tool| tool["type"] == "function")
                            .filter_map(|tool| tool["name"].as_str().map(str::to_owned)),
                    );
                    for tool in tools {
                        for mut function in flatten_delivered(tool, &mut adapter.namespaced_tools)?
                        {
                            let mut envelope = delivered_function_envelope(&function)?;
                            may_be_fix_tool_schema(&mut envelope);
                            schemas.push(tool_schema_text(&envelope));
                            function["defer_loading"] = json!(true);
                            delivered.push(function);
                        }
                    }
                    *item = json!({"type": "function_call_output", "call_id": id,
                        "output": schemas.join("\n")});
                }
                _ => {}
            }
        }
    }
    if adapter
        .namespaced_tools
        .keys()
        .any(|name| bare_names.contains(name))
    {
        return Err("discovered namespace collides with a top-level function name".into());
    }
    let fields = body
        .as_object_mut()
        .ok_or("request body must be an object")?;
    // An explicit declaration wins over a replayed discovery of the same function. Both the flag
    // and the dedupe set are judged on the client's original entries, before delivered tools append.
    let explicit_search = fields
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|entries| {
            entries
                .iter()
                .any(|entry| entry["type"] == "function" && entry["name"] == "tool_search")
        });
    let mut names: HashSet<String> = fields
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|entry| entry["type"] == "function")
        .filter_map(|entry| entry["name"].as_str().map(str::to_owned))
        .collect();
    if !delivered.is_empty() {
        let entries = fields
            .entry("tools")
            .or_insert_with(|| json!([]))
            .as_array_mut()
            .ok_or("tools must be an array")?;
        for tool in delivered {
            let name = tool["name"]
                .as_str()
                .expect("flatten_delivered validates the name");
            if names.insert(name.to_owned()) {
                entries.push(tool);
            }
        }
    }
    if let Some(entries) = fields.get_mut("tools").and_then(Value::as_array_mut) {
        for entry in entries
            .iter()
            .filter(|entry| entry["type"] == "tool_search")
        {
            validate_execution(entry)?;
            if entry
                .get("parameters")
                .is_some_and(|parameters| !parameters.is_object())
            {
                return Err("tool_search parameters must be an object".into());
            }
            if entry
                .get("description")
                .is_some_and(|description| !description.is_string())
            {
                return Err("tool_search description must be a string".into());
            }
        }
        let mut synthesized = false;
        entries.retain_mut(|entry| {
            // `execution` absent is hosted per the vendor schema's optional default; only a
            // literal "client" converts to the client-executed function.
            if entry["type"] != "tool_search" || entry["execution"] != "client" {
                return true;
            }
            if explicit_search || synthesized {
                return false;
            }
            synthesized = true;
            adapter.native_search = true;
            // A delivered function already holds the name: drop the declaration, keep native egress.
            if names.contains("tool_search") {
                return false;
            }
            let mut function = tool_search_function_tool();
            if let Some(description) = entry.get("description") {
                function["description"] = description.clone();
            }
            if let Some(parameters) = entry.get("parameters") {
                function["parameters"] = parameters.clone();
            }
            *entry = function;
            true
        });
    }
    // Client-declared namespaces (`{"type":"namespace"}` in `tools[]`) are the coding adapter's
    // to mangle — its ingress rewrite runs after this one — so they are out of scope here.
    let declared_namespaces: HashSet<String> = fields
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|entry| entry["type"] == "namespace")
        .filter_map(|entry| entry["name"].as_str().map(str::to_owned))
        .collect();
    if let Some(items) = fields.get_mut("input").and_then(Value::as_array_mut) {
        for item in items {
            if item["type"] == "function_call"
                && let (Some(namespace), Some(name)) =
                    (item["namespace"].as_str(), item["name"].as_str())
            {
                let prefix = format!("{namespace}__");
                if name.starts_with(&prefix) || declared_namespaces.contains(namespace) {
                    continue;
                }
                let flat = format!("{prefix}{name}");
                // Never silently drop namespace identity: an unresolvable namespace means the
                // discovering tool_search_output was compacted out of the replay, and a bare
                // `name` could misattribute the call to a same-named top-level tool.
                if !adapter.namespaced_tools.contains_key(&flat) {
                    tracing::warn!(
                        "replayed function_call namespace {namespace:?} does not resolve in this \
                         request's discovery (history compaction); flattening to {flat:?} anyway"
                    );
                }
                item["name"] = json!(flat);
            }
        }
    }
    Ok(adapter)
}

pub(crate) fn tool_search_function_tool() -> serde_json::Value {
    json!({"type": "function", "name": "tool_search", "description": "", "parameters": {}})
}

/// Anthropic's tool-search declaration (`tool_search_tool_regex` / `tool_search_tool_bm25`,
/// versioned or not) rewritten in place to the client tool the Responses path synthesizes, so a
/// session can switch protocols mid-discovery. Entry-provided `description`/`input_schema` win
/// over the synthetic defaults; `defer_loading` and `cache_control` carry over. Any other entry is
/// left alone.
pub(crate) fn messages_tool_search_as_client_tool(entry: &mut Value) {
    if !matches!(
        entry.get("type").and_then(Value::as_str),
        Some(
            "tool_search_tool_regex_20251119"
                | "tool_search_tool_bm25_20251119"
                | "tool_search_tool_regex"
                | "tool_search_tool_bm25"
        )
    ) {
        return;
    }
    let synthetic = tool_search_function_tool();
    let mut client_tool = json!({
        "name": synthetic["name"],
        "description": entry.get("description").unwrap_or(&synthetic["description"]),
        "input_schema": entry.get("input_schema").unwrap_or(&synthetic["parameters"]),
    });
    for key in ["defer_loading", "cache_control"] {
        if let Some(value) = entry.get(key) {
            client_tool[key] = value.clone();
        }
    }
    *entry = client_tool;
}

fn validate_execution(value: &Value) -> Result<(), String> {
    match value.get("execution") {
        None => Ok(()),
        Some(Value::String(execution)) if matches!(execution.as_str(), "client" | "server") => {
            Ok(())
        }
        _ => Err("tool_search execution must be client or server".into()),
    }
}

fn replay_item_id(item: &Value) -> Result<&str, String> {
    item.get("call_id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .or_else(|| {
            item.get("id")
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
        })
        .ok_or_else(|| "tool_search replay item requires a non-empty call_id or id".into())
}

fn flatten_delivered(
    tool: &Value,
    identities: &mut HashMap<String, ToolIdentity>,
) -> Result<Vec<Value>, String> {
    match tool["type"].as_str() {
        Some("function") | Some("custom") => {
            validate_function_name(tool)?;
            Ok(vec![tool.clone()])
        }
        Some("namespace") => {
            let namespace = tool["name"]
                .as_str()
                .filter(|name| !name.is_empty())
                .ok_or("tool_search_output namespace requires a name")?;
            tool["tools"]
                .as_array()
                .ok_or("tool_search_output namespace requires tools")?
                .iter()
                .map(|member| {
                    if member["type"] != "function" && member["type"] != "custom" {
                        return Err(
                            "tool_search_output namespace members must be function or custom tools"
                                .into(),
                        );
                    }
                    let name = validate_function_name(member)?;
                    let flat = format!("{namespace}__{name}");
                    let identity = ToolIdentity {
                        name: name.to_owned(),
                        namespace: namespace.to_owned(),
                    };
                    if identities
                        .get(&flat)
                        .is_some_and(|existing| existing != &identity)
                    {
                        return Err(
                            "discovered namespaces flatten to the same function name".into()
                        );
                    }
                    identities.insert(flat.clone(), identity);
                    let mut member = member.clone();
                    member["name"] = json!(flat);
                    Ok(member)
                })
                .collect()
        }
        _ => Err(
            "tool_search_output supports only function or custom tools and namespace groups".into(),
        ),
    }
}

fn validate_function_name(tool: &Value) -> Result<&str, String> {
    let name = tool["name"]
        .as_str()
        .filter(|name| !name.is_empty())
        .ok_or("discovered function tool requires a non-empty name")?;
    Ok(name)
}

/// The delivered wire entry as the function envelope the shared renderer takes: `name`,
/// `description` (default ""), `parameters` (default `{}`).
fn delivered_function_envelope(tool: &Value) -> Result<serde_json::Map<String, Value>, String> {
    let name = tool["name"].as_str().ok_or("function requires name")?;
    let mut function = serde_json::Map::new();
    function.insert("name".to_string(), json!(name));
    function.insert(
        "description".to_string(),
        tool.get("description").cloned().unwrap_or(json!("")),
    );
    function.insert(
        "parameters".to_string(),
        tool.get("parameters").cloned().unwrap_or(json!({})),
    );
    Ok(function)
}

impl CodingAdapter for ToolSearchAdapter {
    fn handles_tool_call(&self, name: &str) -> bool {
        (self.native_search && name == "tool_search")
            || self
                .inner
                .as_ref()
                .is_some_and(|inner| inner.handles_tool_call(name))
    }

    fn render_tool_call(&self, call: &ToolCallToRender<'_>) -> Option<RenderedToolCall> {
        if !self.native_search || call.tool_name != "tool_search" {
            return self
                .inner
                .as_ref()
                .and_then(|inner| inner.render_tool_call(call));
        }
        let arguments: Value = match serde_json::from_str(call.args) {
            Ok(arguments) => arguments,
            Err(error) => {
                tracing::error!("invalid tool_search arguments for {}: {error}", call.id);
                return None;
            }
        };
        Some(RenderedToolCall {
            item: json!({"type": "tool_search_call", "id": call.id, "call_id": call.id,
                "execution": "client", "arguments": arguments,
                "status": match call.status { ToolCallStatus::Failed => "incomplete", ToolCallStatus::InProgress => "in_progress", ToolCallStatus::Completed => "completed" }}),
            lifecycle_events: vec![],
        })
    }

    fn render_tool_result(&self, result: &ToolResultToRender<'_>) -> Option<Value> {
        self.inner
            .as_ref()
            .and_then(|inner| inner.render_tool_result(result))
    }

    fn resolve_tool_identity(&self, name: &str) -> Option<ToolIdentity> {
        self.namespaced_tools.get(name).cloned().or_else(|| {
            self.inner
                .as_ref()
                .and_then(|inner| inner.resolve_tool_identity(name))
        })
    }

    fn splice_rendered_calls(&self, body: Value, rendered: &HashMap<String, Value>) -> Value {
        let mut body = match &self.inner {
            Some(inner) => inner.splice_rendered_calls(body, rendered),
            None => body,
        };
        if let Some(output) = body["output"].as_array_mut() {
            for item in output {
                if let Some(replacement) = item["id"].as_str().and_then(|id| rendered.get(id)) {
                    *item = replacement.clone();
                }
            }
        }
        body
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn replay() -> Value {
        json!({"model": "glm-5.3", "tools": [{"type":"tool_search", "execution":"client"}],
        "input": [
            {"type":"tool_search_call", "call_id":"search1", "arguments":{"query":"weather"}, "execution":"client"},
            {"type":"tool_search_output", "call_id":"search1", "execution":"client", "tools":[
                {"type":"function", "name":"weather", "parameters":{}},
                {"type":"namespace", "name":"crm", "tools":[{"type":"function", "name":"lookup", "description":"<lookup>", "parameters":{}}]}
            ]}
        ]})
    }

    #[test]
    fn discovery_replay_renders_schema_text_and_keeps_namespace_identity() {
        let mut body = replay();
        let adapter = rewrite_responses(&mut body).unwrap();
        assert_eq!(body["input"][0]["type"], "function_call");
        assert_eq!(body["input"][1]["type"], "function_call_output");
        let output = body["input"][1]["output"].as_str().unwrap();
        assert!(output.contains("<tools>\n"));
        assert!(output.contains("\\u003clookup\\u003e"));
        assert!(output.contains(r#""parameters": {"type":"object","properties":{}}"#));
        assert_eq!(body["tools"][1]["defer_loading"], true);
        assert_eq!(body["tools"][2]["name"], "crm__lookup");
        let identity = adapter.resolve_tool_identity("crm__lookup").unwrap();
        assert_eq!(identity.name, "lookup");
        assert_eq!(identity.namespace, "crm");
    }

    #[test]
    fn explicit_search_function_wins_over_native_declaration() {
        for tools in [
            json!([{"type":"tool_search", "execution":"client"}, {"type":"function", "name":"tool_search", "description":"mine"}]),
            json!([{"type":"function", "name":"tool_search", "description":"mine"}, {"type":"tool_search", "execution":"client"}]),
        ] {
            let mut body = json!({"tools":tools});
            let adapter = rewrite_responses(&mut body).unwrap();
            assert_eq!(body["tools"].as_array().unwrap().len(), 1);
            assert_eq!(body["tools"][0]["description"], "mine");
            assert!(adapter.wrap(None).is_none());
        }
    }

    #[test]
    fn invalid_replays_are_refused_before_protocol_parsing() {
        for input in [
            json!([{"type":"tool_search_call", "arguments":{}}]),
            json!([{"type":"tool_search_call", "call_id":"c", "arguments":[1]}]),
            json!([{"type":"tool_search_call", "call_id":"c", "arguments":{}},
                {"type":"tool_search_output", "call_id":"c", "tools":[{"type":"web_search"}]}]),
            json!([{"type":"tool_search_call", "call_id":"c", "arguments":{}},
                {"type":"tool_search_output", "call_id":"c", "tools":[
                    {"type":"namespace", "name":"n", "tools":[{"type":"web_search"}]}]}]),
        ] {
            assert!(rewrite_responses(&mut json!({"input":input})).is_err());
        }
    }

    #[test]
    fn replay_items_fall_back_to_id_when_call_id_is_missing_or_empty() {
        let mut body = json!({"input":[
            {"type":"tool_search_call", "id":"s1", "arguments":{}},
            {"type":"tool_search_output", "id":"s1", "tools":[
                {"type":"function", "name":"weather", "parameters":{}}]},
            {"type":"tool_search_call", "id":"s2", "call_id":"", "arguments":{}}
        ]});
        rewrite_responses(&mut body).unwrap();
        assert_eq!(body["input"][0]["call_id"], "s1");
        assert_eq!(body["input"][1]["call_id"], "s1");
        assert_eq!(body["input"][2]["call_id"], "s2");
        assert_eq!(body["tools"][0]["name"], "weather");
    }

    #[test]
    fn string_arguments_pass_through_and_absent_arguments_default_empty() {
        let mut body = json!({"input":[
            {"type":"tool_search_call", "call_id":"c1", "arguments":"{\"query\":\"x\"}"},
            {"type":"tool_search_call", "call_id":"c2"},
            {"type":"tool_search_call", "call_id":"c3", "arguments":null}
        ]});
        rewrite_responses(&mut body).unwrap();
        assert_eq!(body["input"][0]["arguments"], "{\"query\":\"x\"}");
        assert_eq!(body["input"][1]["arguments"], "{}");
        assert_eq!(body["input"][2]["arguments"], "{}");
    }

    #[test]
    fn orphan_output_warns_and_still_delivers_its_tools() {
        let mut body = json!({"input":[
            {"type":"tool_search_output", "call_id":"orphan", "tools":[
                {"type":"function", "name":"weather", "parameters":{}}
            ]}
        ]});
        rewrite_responses(&mut body).unwrap();
        assert_eq!(body["input"][0]["type"], "function_call_output");
        assert_eq!(body["tools"][0]["name"], "weather");
        assert_eq!(body["tools"][0]["defer_loading"], true);
    }

    #[test]
    fn custom_members_flatten_and_deliver_like_functions() {
        let mut body = json!({"input":[
            {"type":"tool_search_call", "call_id":"s", "arguments":{}},
            {"type":"tool_search_output", "call_id":"s", "tools":[
                {"type":"custom", "name":"freeform", "description":"freeform tool"},
                {"type":"namespace", "name":"crm", "tools":[
                    {"type":"custom", "name":"lookup", "description":"custom lookup"}]}
            ]}
        ]});
        let adapter = rewrite_responses(&mut body).unwrap();
        assert_eq!(body["tools"][0]["name"], "freeform");
        assert_eq!(body["tools"][0]["type"], "custom");
        assert_eq!(body["tools"][0]["defer_loading"], true);
        assert_eq!(body["tools"][1]["name"], "crm__lookup");
        assert_eq!(body["tools"][1]["type"], "custom");
        let identity = adapter.resolve_tool_identity("crm__lookup").unwrap();
        assert_eq!(identity.name, "lookup");
        assert_eq!(identity.namespace, "crm");
    }

    /// The discovering `tool_search_output` was compacted out of the replay: the call keeps its
    /// namespace identity (warned) instead of collapsing onto the same-named top-level tool.
    #[test]
    fn orphaned_replayed_namespace_flattens_instead_of_misattributing() {
        let mut body = json!({
            "tools": [{"type":"function", "name":"read_thread", "parameters":{}}],
            "input": [
                {"type":"function_call", "call_id":"c1", "name":"read_thread",
                 "namespace":"slack", "arguments":"{}"}
            ]
        });
        rewrite_responses(&mut body).unwrap();
        assert_eq!(body["input"][0]["name"], "slack__read_thread");
        assert_eq!(body["tools"][0]["name"], "read_thread");
    }

    /// A replayed call into a discovered namespace flattens through the map as before; an
    /// already-flattened name is not flattened twice.
    #[test]
    fn discovered_namespace_replay_flattens_once() {
        let mut body = json!({"input":[
            {"type":"tool_search_call", "call_id":"s", "arguments":{}},
            {"type":"tool_search_output", "call_id":"s", "tools":[
                {"type":"namespace", "name":"slack", "tools":[
                    {"type":"function", "name":"read_thread", "parameters":{}}]}
            ]},
            {"type":"function_call", "call_id":"c1", "name":"read_thread",
             "namespace":"slack", "arguments":"{}"},
            {"type":"function_call", "call_id":"c2", "name":"slack__read_thread",
             "namespace":"slack", "arguments":"{}"}
        ]});
        rewrite_responses(&mut body).unwrap();
        assert_eq!(body["input"][2]["name"], "slack__read_thread");
        assert_eq!(body["input"][3]["name"], "slack__read_thread");
        assert_eq!(body["tools"][0]["name"], "slack__read_thread");
    }

    /// A client-declared namespace is the coding adapter's to mangle (its rewrite runs after
    /// this one): the call is left untouched here.
    #[test]
    fn client_declared_namespace_replay_is_left_to_the_coding_adapter() {
        let mut body = json!({
            "tools": [{"type":"namespace", "name":"multi_agent_v1",
                       "tools":[{"type":"function", "name":"spawn_agent", "parameters":{}}]}],
            "input": [
                {"type":"function_call", "call_id":"c1", "name":"spawn_agent",
                 "namespace":"multi_agent_v1", "arguments":"{}"}
            ]
        });
        rewrite_responses(&mut body).unwrap();
        assert_eq!(body["input"][0]["name"], "spawn_agent");
    }

    #[test]
    fn top_level_custom_tools_pass_through_untouched() {
        let mut body = json!({"tools":[{"type":"custom", "name":"mine", "description":"d"}]});
        rewrite_responses(&mut body).unwrap();
        assert_eq!(
            body["tools"][0],
            json!({"type":"custom", "name":"mine", "description":"d"})
        );
    }

    #[test]
    fn discovered_namespaces_cannot_shadow_flat_functions() {
        let mut body = replay();
        body["tools"]
            .as_array_mut()
            .unwrap()
            .push(json!({"type":"function", "name":"crm__lookup"}));
        assert!(
            rewrite_responses(&mut body)
                .err()
                .unwrap()
                .contains("collides")
        );
        let mut body = json!({"tools":[{"type":"tool_search","execution":"elsewhere"}]});
        assert!(rewrite_responses(&mut body).is_err());
    }

    #[test]
    fn server_search_is_not_advertised_as_a_client_function() {
        let mut body = json!({"tools":[{"type":"tool_search", "execution":"server"}]});
        let adapter = rewrite_responses(&mut body).unwrap();
        assert_eq!(body["tools"][0]["type"], "tool_search");
        assert!(adapter.wrap(None).is_none());
    }

    #[test]
    fn bare_tool_search_is_hosted_and_passes_through() {
        let mut body = json!({"tools":[{"type":"tool_search"}]});
        let adapter = rewrite_responses(&mut body).unwrap();
        assert_eq!(body["tools"][0]["type"], "tool_search");
        assert!(!adapter.native_search);
        assert!(adapter.wrap(None).is_none());
    }

    #[test]
    fn discovered_tool_search_function_does_not_suppress_native_declaration() {
        let mut body = json!({
            "tools": [{"type":"tool_search", "execution":"client"}],
            "input": [
                {"type":"tool_search_call", "call_id":"s", "arguments":{}},
                {"type":"tool_search_output", "call_id":"s", "tools":[
                    {"type":"function", "name":"tool_search", "description":"discovered", "parameters":{}}
                ]}
            ]
        });
        let adapter = rewrite_responses(&mut body).unwrap();
        assert!(adapter.native_search);
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "tool_search");
        assert_eq!(tools[0]["description"], "discovered");
    }

    #[test]
    fn native_search_egress_carries_object_arguments_and_replay_id() {
        let mut body = json!({"tools":[{"type":"tool_search", "execution":"client"}]});
        let adapter = rewrite_responses(&mut body).unwrap();
        let rendered = adapter
            .render_tool_call(&ToolCallToRender {
                tool_name: "tool_search",
                id: "c1",
                args: r#"{"query":"weather"}"#,
                status: ToolCallStatus::Completed,
            })
            .unwrap();
        assert_eq!(rendered.item["type"], "tool_search_call");
        assert_eq!(rendered.item["call_id"], "c1");
        assert_eq!(rendered.item["execution"], "client");
        assert_eq!(rendered.item["arguments"]["query"], "weather");
    }
}
