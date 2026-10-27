//! API-translation layer. ChatCompletions is the canonical internal hub protocol: the client
//! protocol is translated to CC at the request edge ([`request`]) and back at the response edge
//! ([`sse_emitter`]). The model's CC stream is decoded once at the single boundary source
//! ([`sse_parser::SseParser`]) into [`SemanticChunk`]s that downstream consumers fold — nothing
//! re-derives those boundaries.
//!
//! Ported from tool-bank's `api_translation` module (Apache-2.0). Server tools are represented but
//! never executed here: ingress selection plugs in through [`hooks::IngressHooks`] (the shipped
//! [`hooks::DropServerTools`] is standard-dynamo behavior — drop them), and the egress framings
//! render whatever server-tool activity the caller's loop reports.

pub mod baseten_response_extension;
pub mod client_egress;
pub mod coding_adapter;
pub mod framing;
pub mod history;
pub mod hooks;
pub mod loss;
pub mod model;
pub mod request;
pub mod sse_emitter;
pub mod sse_parser;

pub(crate) mod util;
pub(crate) mod wire;

#[cfg(test)]
mod dynamo_conformance_tests;
#[cfg(test)]
pub(crate) mod test_utils;

use crate::model::ToolCall;

/// What adaptation did not carry through, and the closed set of kinds a counter can be labeled by.
pub use crate::loss::{Loss, LossKind};
/// The client-facing error class, re-exported so callers can select it without reaching into
/// [`model`]. Each protocol renders it as its own `error.type` word on the response edge.
pub use crate::model::ErrorClass;

/// The canonical internal wire types (the fork's `dynamo-protocols`). `Cc*` names track the
/// DoR vocabulary — the client protocol is translated to these at the request edge and back at the
/// response edge; everything between speaks ChatCompletions.
/// The protocol crate this crate's public API is expressed in. Consumers should reach the wire
/// types through this re-export instead of a second `dynamo-protocols` dependency, so the two can
/// never drift onto different revisions.
pub use dynamo_protocols;
pub use dynamo_protocols::types::{ChatCompletionRequestMessage as CcMessage, FinishReason};

/// The canonical ChatCompletions request: the fork's wire type plus the ordered passthrough of
/// every request field it does not model (`chat_template_kwargs`, `guided_*`, provider sampling
/// extras). The catch-all lives HERE, on the translation layer's own type, not on the shared wire
/// type: a `#[serde(flatten)]` map on `CreateChatCompletionRequest` itself would make serde
/// deserialize the struct through its map path, and every wrapper that flattens it alongside
/// its own catch-all (dynamo's `NvCreateChatCompletionRequest`) would then see every key —
/// `model`, `messages`, ... — as unknown.
///
/// `unmodeled` is a `serde_json::Map` (ordered under `preserve_order`), NOT a HashMap, so the
/// re-serialized field order is turn-stable for the KV-cache prefix. Derefs to the wire type so
/// the modeled fields read as its own.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CcRequest {
    #[serde(flatten)]
    pub inner: dynamo_protocols::types::CreateChatCompletionRequest,
    #[serde(flatten)]
    pub unmodeled: serde_json::Map<String, serde_json::Value>,
}

impl std::ops::Deref for CcRequest {
    type Target = dynamo_protocols::types::CreateChatCompletionRequest;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::ops::DerefMut for CcRequest {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl From<dynamo_protocols::types::CreateChatCompletionRequest> for CcRequest {
    fn from(inner: dynamo_protocols::types::CreateChatCompletionRequest) -> Self {
        Self {
            inner,
            unmodeled: serde_json::Map::new(),
        }
    }
}

#[cfg(test)]
#[path = "lib_test.rs"]
mod tests;

/// Anthropic's own id shape for a server-executed tool use, which its clients match results on.
/// Minted on egress by a Messages coding adapter and stripped back off on ingress, so a replayed
/// turn resolves to the same call id the model issued.
pub const SERVER_TOOL_USE_ID_PREFIX: &str = "srvtoolu_";

/// Responses item-id markers: an emitted item's `id` is `<marker><call id>`, and ingress strips the
/// marker exactly once. OpenAI checks replayed item-id prefixes (`fc`, `mcp`, `ws`), so a session can
/// move from a Baseten model to an OpenAI one with ids that carry `_` (Kimi's), which Codex keeps.
pub(crate) const MCP_CALL_ID_PREFIX: &str = "mcp_b10_";
pub(crate) const FUNCTION_CALL_ID_PREFIX: &str = "fc_b10_";
pub const WEB_SEARCH_CALL_ID_PREFIX: &str = "ws_b10_";

pub fn prefixed_item_id(prefix: &str, id: &str) -> String {
    format!("{prefix}{id}")
}

pub fn strip_item_id_prefix<'a>(prefix: &str, id: &'a str) -> &'a str {
    id.strip_prefix(prefix).unwrap_or(id)
}

/// Channel carrying encoded SSE frames from the loop/emitter to the HTTP egress.
pub type SseFrameTx = tokio::sync::mpsc::Sender<String>;

/// External protocol the client speaks. Internal/model edge is always ChatCompletions; `Messages`
/// clients are translated at both edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClientProtocol {
    ChatCompletions,
    Messages,
    Responses,
}

/// Reported per raw model delta by the parser ([`sse_parser::SseDataYield`]); the axis a loop's
/// phase timeline keys off.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentKind {
    Thinking,
    Text,
    ToolCall,
}

/// One finished semantic unit off the model's ChatCompletions stream, drawn once at the single
/// boundary source ([`sse_parser::SseParser`]). Text and thinking stream through as deltas; a tool
/// call is held until its args parse and surfaces once, complete.
#[derive(Debug, Clone)]
pub enum SemanticChunk {
    TextDelta(String),
    ThinkingDelta(String),
    /// A complete tool call: valid-JSON args, assembled by the parser.
    ToolCall(ToolCall),
    /// One model call's usage (we request `stream_options.include_usage`).
    Usage(dynamo_protocols::types::CompletionUsage),
    Stop {
        finish_reason: dynamo_protocols::types::FinishReason,
    },
}
