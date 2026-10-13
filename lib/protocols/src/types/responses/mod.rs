// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Dynamo owns the Responses-API input-side type chain. Upstream async-openai
// is the source for everything else (output-side types, streaming events,
// individual tool-call payloads, etc.).
//
// The input chain is owned because upstream marks fields as required that
// real-world clients (OpenAI Agents SDK, Codex, etc.) routinely omit when
// round-tripping a prior assistant turn as input:
//   - `OutputMessage.id` / `.status` — omitted when echoing a previous output
//   - `OutputTextContent.annotations` — omitted when the part carried none
// Upstream is slow to relax these (see e.g. 64bit/async-openai#535 for the
// sibling `ReasoningItem.id` fix, still open at time of writing); OpenAI's own
// hosted API accepts the relaxed shapes on input regardless.
//
// This mirrors the pattern in `crate::types::chat` where Dynamo owns the
// request types it needs to extend or relax while re-exporting the rest of
// upstream's type library verbatim.
//
// Naming: the relaxed assistant-input message is `InputOutputMessage` (and
// `InputOutputMessageContent` / `InputOutputTextContent` for its content
// parts) to avoid colliding with upstream's `OutputMessage`, which remains the
// canonical type for *output-side* response construction (`OutputItem`,
// `Response.output`). `MessageItem`, `Item`, `InputItem`, `InputParam`, and
// `CreateResponse` are input-only and shadow upstream's same-named types
// without conflict.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// Re-export all upstream response types (shared structures like ResponseUsage,
// tool-call item types, streaming events, etc.). The types we own below
// shadow their upstream counterparts where no dual-side conflict exists.
pub use async_openai::types::responses::*;

// Re-export upstream's pre-shadow `InputContent` under an explicit alias.
// Needed because `FunctionCallOutput::Content` and `EasyInputContent::ContentList`
// are non-owned upstream types that carry upstream's original `InputContent`
// inline, so downstream consumers occasionally need to name it alongside the
// Dynamo-owned shadow defined further down this module.
pub use async_openai::types::responses::InputContent as UpstreamInputContent;

// Re-export from parent module for backward compat.
pub use crate::types::ImageDetail;
pub use crate::types::ResponseFormatJsonSchema;
pub use crate::types::{B10ReasoningEffort, ReasoningEffort};

// Backward-compatible type aliases for Dynamo consumer code migration.
pub type Input = InputParam;
pub type PromptConfig = Prompt;
pub type TextConfig = ResponseTextParam;
pub type TextResponseFormat = TextResponseFormatConfiguration;

/// Input-side MCP errors accept legacy strings as well as structured errors.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum InputMCPToolCallError {
    Legacy(String),
    Structured(MCPToolCallError),
}

impl InputMCPToolCallError {
    /// Text forwarded to the model for a replayed tool failure.
    pub fn to_text(&self) -> String {
        match self {
            Self::Legacy(text) => text.clone(),
            Self::Structured(MCPToolCallError::McpProtocolError(error)) => error.message.clone(),
            Self::Structured(MCPToolCallError::HttpError(error)) => error.message.clone(),
            Self::Structured(MCPToolCallError::McpToolExecutionError(error)) => error
                .content
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| error.content.to_string()),
        }
    }
}

/// Relaxed MCP transcript input; output continues to use upstream `MCPToolCall`.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct InputMCPToolCall {
    pub arguments: String,
    pub id: String,
    pub name: String,
    pub server_label: String,
    pub approval_request_id: Option<String>,
    pub error: Option<InputMCPToolCallError>,
    pub output: Option<String>,
    pub status: Option<MCPToolCallStatus>,
}

/// Stream of response events.
pub type ResponseStream = std::pin::Pin<
    Box<dyn futures::Stream<Item = Result<ResponseStreamEvent, crate::error::OpenAIError>> + Send>,
>;

/// Fields on upstream `Response` that the OpenResponses spec requires as
/// `T | null` but async-openai declares as `Option<T>` with
/// `skip_serializing_if = Option::is_none` — meaning `None` disappears from
/// the wire shape, where the spec wants an explicit `null`.
///
/// Colocated here (next to the upstream `Response` re-export) rather than in
/// `lib/llm/src/protocols/openai/responses/mod.rs` so that when upstream's
/// `Response` gains a new nullable-required field, the reviewer editing this
/// module is looking directly at the authoritative list. Keep sorted
/// alphabetically; entries must match serde field names on `Response` exactly.
///
/// Any field we unconditionally populate ourselves during response
/// construction (e.g. `metadata`, `parallel_tool_calls`, `temperature`,
/// `text`, `tool_choice`, `tools`, `top_p`, `top_logprobs`, `truncation`,
/// `service_tier`, `background`) is deliberately absent — it's always
/// present on the wire, so listing it here would be noise.
pub const SPEC_NULLABLE_REQUIRED_RESPONSE_FIELDS: &[&str] = &[
    "billing",
    "completed_at",
    "conversation",
    "error",
    "incomplete_details",
    "instructions",
    "max_output_tokens",
    "max_tool_calls",
    "previous_response_id",
    "prompt",
    "prompt_cache_key",
    "prompt_cache_retention",
    "reasoning",
    "safety_identifier",
    "usage",
];

// ---------------------------------------------------------------------------
// Input-side assistant message (relaxed vs upstream OutputMessage)
// ---------------------------------------------------------------------------

/// Deserialize `null` or a missing field as the default empty `Vec`. Plain
/// `#[serde(default)]` only fires when the field is absent; explicit `null`
/// would otherwise fail `Vec::deserialize`. Clients (notably some Agents SDK
/// variants) have been observed to send `"annotations": null`, so treat
/// omission and explicit null the same.
fn deserialize_null_as_empty_vec<'de, T, D>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    T: Deserialize<'de>,
    D: serde::Deserializer<'de>,
{
    Option::<Vec<T>>::deserialize(deserializer).map(Option::unwrap_or_default)
}

/// Deserialize `null` or a missing field as `T::default()`. Scalar counterpart
/// to `deserialize_null_as_empty_vec` — plain `#[serde(default)]` rejects
/// explicit `null` because serde tries to deserialize the null into `T` and
/// fails. Real clients emit `null` for unset enum-ish fields (e.g. OpenAI
/// Agents SDK sending `"detail": null` on `input_image` parts).
fn deserialize_null_as_default<'de, T, D>(deserializer: D) -> Result<T, D::Error>
where
    T: Deserialize<'de> + Default,
    D: serde::Deserializer<'de>,
{
    Option::<T>::deserialize(deserializer).map(Option::unwrap_or_default)
}

/// Relaxed counterpart to upstream `OutputTextContent` for input-side content.
/// `annotations` tolerates both missing and explicit `null`; upstream requires
/// it to be a present non-null array.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct InputOutputTextContent {
    #[serde(default, deserialize_with = "deserialize_null_as_empty_vec")]
    pub annotations: Vec<Annotation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Vec<LogProb>>,
    pub text: String,
}

/// Content parts of a prior assistant message presented as input.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputOutputMessageContent {
    OutputText(InputOutputTextContent),
    Refusal(RefusalContent),
}

/// An assistant message echoed back as input for a subsequent turn. Relaxed
/// compared to upstream `OutputMessage`: `id`, `status`, and `content` are all
/// optional. Some clients send a bare assistant shell (`{"type":"message",
/// "role":"assistant"}`) with no `content` at all, usually on pure tool-call
/// turns; treat absent `content` as an empty vec, same way we treat a missing
/// `id`/`status`.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct InputOutputMessage {
    #[serde(default, deserialize_with = "deserialize_null_as_empty_vec")]
    pub content: Vec<InputOutputMessageContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub role: AssistantRole,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<MessagePhase>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<OutputStatus>,
}

// ---------------------------------------------------------------------------
// Input-side image / content / message (shadow upstream, relaxed shapes)
// ---------------------------------------------------------------------------

/// Relaxed counterpart to upstream `InputImageContent`. `detail` defaults to
/// `ImageDetail::Auto` when the client omits it — OpenAI's hosted API and the
/// OpenResponses spec both accept this shape, but upstream's struct marks
/// `detail` as required.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct InputImageContent {
    #[serde(default, deserialize_with = "deserialize_null_as_default")]
    pub detail: ImageDetail,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_url: Option<String>,
}

/// Parts of an input message: text, image, or file. Mirrors upstream
/// `InputContent` but routes `InputImage` through the Dynamo-owned relaxed
/// `InputImageContent` above.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputContent {
    InputText(InputTextContent),
    InputImage(InputImageContent),
    InputFile(InputFileContent),
}

/// User / system / developer input message. Shadows upstream `InputMessage`
/// so we can route through the Dynamo-owned `InputContent` chain.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub struct InputMessage {
    pub content: Vec<InputContent>,
    pub role: InputRole,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<OutputStatus>,
}

/// Content for `EasyInputMessage`. Shadows upstream's same-named enum so the
/// `ContentList` arm carries Dynamo's relaxed `InputContent` (with optional
/// `detail` on `InputImageContent`) instead of upstream's strict variant.
///
/// Without this shadow, the `InputItem::EasyMessage` fallback in the untagged
/// `InputItem` enum is the only path that still routes through upstream's
/// strict types — so any spec-compliant client that omits `type: "message"`
/// on a multimodal message (the documented default) fails with
/// "data did not match any variant of untagged enum InputItem". See issue
/// #9468.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum EasyInputContent {
    /// Plain-text content. Tried first so `"content": "hi"` short-circuits.
    Text(String),
    /// Structured content list (text/image/file parts).
    ContentList(Vec<InputContent>),
}

impl Default for EasyInputContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

/// A simplified message input — the spec-default shape when a client omits the
/// `type` discriminator. Shadows upstream `EasyInputMessage` so the `content`
/// field routes through Dynamo's relaxed `EasyInputContent` (and transitively
/// the relaxed `InputContent` / `InputImageContent`). Field set is identical to
/// upstream for drop-in compatibility with construction sites in lib/llm.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub struct EasyInputMessage {
    /// Type discriminator. Optional with default `MessageType::Message` —
    /// matches the OpenAI Responses spec and `openai-python`'s
    /// `EasyInputMessageParam` (`type: Literal["message"]`, non-Required).
    #[serde(default)]
    pub r#type: MessageType,
    pub role: Role,
    pub content: EasyInputContent,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<MessagePhase>,
}

// ---------------------------------------------------------------------------
// Input-side Item / Message / InputItem / InputParam (shadow upstream)
// ---------------------------------------------------------------------------

/// Message item within `Item`. Untagged; disambiguated by the `role` field:
/// the `Output` variant requires `role: "assistant"` (via `AssistantRole`,
/// which is a single-variant enum) and `Input` requires `role` in
/// `"user" | "system" | "developer"` (via `InputRole`). A payload with an
/// unknown role (e.g. `"tool"`) or a missing `role` produces the generic
/// untagged-enum error — callers are expected to send a valid role. If you
/// see the "data did not match any variant of untagged enum" failure on this
/// type, it is almost always a role mismatch.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum MessageItem {
    /// Prior assistant output echoed back (role: assistant). Tried first — its
    /// `role` constraint excludes user/system/developer inputs.
    Output(InputOutputMessage),
    /// User / system / developer input message.
    Input(InputMessage),
}

/// Content part of an `agent_message` item. Codex drops an agent message
/// whose content is not entirely plaintext when rendering one for a chat
/// bridge (its `plaintext_agent_message_content`), and the same rule holds
/// here: encrypted parts are unrenderable, so the caller skips the item.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentMessageInputContent {
    InputText { text: String },
    EncryptedContent { encrypted_content: String },
}

/// Input-side `agent_message`: a multi-agent transcript item (codex
/// `multi_agent_v1`), host-authored and sent verbatim to every Responses
/// provider.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct AgentMessageItemParam {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub author: String,
    pub recipient: String,
    pub content: Vec<AgentMessageInputContent>,
}

/// Input-side `additional_tools`: codex Responses-Lite framing carries the
/// tool declarations on an input item instead of top-level `tools`.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct AdditionalToolsItemParam {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub role: String,
    pub tools: Vec<serde_json::Value>,
}

/// Structured input item, discriminated by `type`. Mirrors upstream `Item`
/// with local message and MCP overrides and Dynamo's agent-message extension.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Item {
    Message(MessageItem),
    ConfigurationUpdate(ResponseConfigurationUpdateItemParam),
    FileSearchCall(FileSearchToolCall),
    ComputerCall(ComputerToolCall),
    ComputerCallOutput(ComputerCallOutputItemParam),
    WebSearchCall(WebSearchToolCall),
    FunctionCall(FunctionToolCall),
    FunctionCallOutput(FunctionCallOutputItemParam),
    ToolSearchCall(ToolSearchCallItemParam),
    ToolSearchOutput(ToolSearchOutputItemParam),
    Reasoning(ReasoningItem),
    Compaction(CompactionSummaryItemParam),
    ImageGenerationCall(ImageGenToolCall),
    CodeInterpreterCall(CodeInterpreterToolCall),
    LocalShellCall(LocalShellToolCall),
    LocalShellCallOutput(LocalShellToolCallOutput),
    ShellCall(FunctionShellCallItemParam),
    ShellCallOutput(FunctionShellCallOutputItemParam),
    ApplyPatchCall(ApplyPatchToolCallItemParam),
    ApplyPatchCallOutput(ApplyPatchToolCallOutputItemParam),
    McpListTools(MCPListTools),
    McpApprovalRequest(MCPApprovalRequest),
    McpApprovalResponse(MCPApprovalResponse),
    McpCall(InputMCPToolCall),
    CustomToolCallOutput(CustomToolCallOutput),
    CustomToolCall(CustomToolCall),
    /// Codex multi-agent sessions replay sub-agent turns as `agent_message` items (author,
    /// recipient, content); a transcript echo, so parsing it keeps multi-agent replay from failing
    /// the whole request as an unknown variant.
    AgentMessage(AgentMessageItemParam),
    /// Responses-Lite framing: tools declared mid-conversation by a `developer`/`system` item
    /// instead of the top-level `tools[]` (Codex sends these). Declared to the model as tools by
    /// the shared translation crate.
    AdditionalTools(AdditionalToolsItemParam),
}

/// Single input item. Untagged; order matters (most specific first).
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum InputItem {
    ItemReference(ItemReference),
    Item(Item),
    EasyMessage(EasyInputMessage),
}

/// Input to a `POST /v1/responses` request.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum InputParam {
    Text(String),
    Items(Vec<InputItem>),
}

impl Default for InputParam {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

// ---------------------------------------------------------------------------
// CreateResponse (owned, uses Dynamo-owned InputParam)
// ---------------------------------------------------------------------------

/// Responses `reasoning`, request side.
///
/// Structurally the upstream `Reasoning`, but `effort` is the permissive
/// [`B10ReasoningEffort`], so `/v1/responses` and `/v1/chat/completions` accept
/// exactly the same efforts. A value only the serve-side policy can judge
/// must not be refused at deserialization.
///
/// The response object keeps the upstream [`Reasoning`]; [`From`] converts for
/// the echo, where canonical levels stay unchanged and a non-level omits
/// `effort` entirely.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct B10ReasoningParam {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<B10ReasoningEffort>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<ReasoningSummary>,
}

impl From<&B10ReasoningParam> for Reasoning {
    fn from(param: &B10ReasoningParam) -> Self {
        Self {
            effort: param
                .effort
                .as_ref()
                .and_then(B10ReasoningEffort::to_async_openai),
            summary: param.summary.clone(),
            ..Default::default()
        }
    }
}

/// Request body for `POST /v1/responses`. Mirrors upstream `CreateResponse`
/// field-for-field but uses Dynamo-owned `InputParam`, which transitively
/// accepts the relaxed input shapes described in this module's header. All
/// other fields reference upstream types verbatim.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub struct CreateResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation: Option<ConversationParam>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<IncludeEnum>>,
    pub input: InputParam,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tool_calls: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<Prompt>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_retention: Option<PromptCacheRetention>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<B10ReasoningParam>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTierResponses>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<ResponseStreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<ResponseTextParam>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoiceParam>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<Truncation>,
}

// ---------------------------------------------------------------------------
// CountInputTokens (`POST /v1/responses/input_tokens`)
// Backport of upstream ai-dynamo/dynamo#13768 onto the owned input chain.
// ---------------------------------------------------------------------------

/// The `object` discriminator on a [`CountInputTokensResponse`].
pub const RESPONSE_INPUT_TOKENS_OBJECT: &str = "response.input_tokens";

/// Request body for `POST /v1/responses/input_tokens`.
///
/// A subset of [`CreateResponse`] — only the fields that reach the rendered
/// prompt. This mirrors `AnthropicCountTokensRequest`, which is the same
/// subset-of-the-create-request shape for `POST /v1/messages/count_tokens`.
///
/// Two deliberate differences from `CreateResponse`: `input` defaults (the
/// count endpoint accepts a body without one, whereas creating a response
/// requires it), and unknown fields are ignored, so stateful parameters
/// Dynamo does not serve (`conversation`, `previous_response_id`) are accepted
/// and disregarded rather than rejected. This endpoint reports a pre-flight
/// estimate; it never generates, so there is nothing for them to affect.
///
/// Deserialization is forgiving in two further places — an explicit
/// `"input": null` and unrecognized tool shapes — for the same reason
/// `AnthropicTool` keeps every field but `name` optional: a pre-flight
/// estimate that rejects a body it could have scored is strictly worse than
/// one that scores it approximately. See [`deserialize_lenient_tools`].
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub struct CountInputTokensRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// `#[serde(default)]` alone covers an absent `input`, but not an explicit
    /// `"input": null` — serde still hands that null to `InputParam`, whose
    /// deserializer rejects it. Here both mean "nothing to count".
    #[serde(default, deserialize_with = "deserialize_null_default_input")]
    pub input: InputParam,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_lenient_tools"
    )]
    pub tools: Option<Vec<Tool>>,
}

fn deserialize_null_default_input<'de, D>(deserializer: D) -> Result<InputParam, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<InputParam>::deserialize(deserializer)?.unwrap_or_default())
}

/// Drop tool entries that do not deserialize into a known [`Tool`], rather than
/// failing the whole request.
///
/// `Tool` is upstream's `#[serde(tag = "type")]` enum, so it models only the
/// tool types the pinned `async-openai` knows. A caller that forwards a tool in
/// a shape upstream does not model — a Chat-Completions-style `{"type":
/// "custom", "custom": {...}}`, or a tool type newer than the pin — would
/// otherwise get a 400 for a field that contributes almost nothing to the
/// estimate.
///
/// Dropping costs nothing: [`estimate_tool_len`] already scores every
/// non-function tool as 0, because the converter forwards only function tools
/// to the backend. An unparseable tool was going to be worth 0 either way; this
/// only decides whether the rest of the body still gets counted. That is the
/// same trade `estimate_tool_len` documents when it wildcards where
/// `measure_item` is exhaustive — `Tool` is upstream's type, and we carry
/// no obligation to mirror its variants.
fn deserialize_lenient_tools<'de, D>(deserializer: D) -> Result<Option<Vec<Tool>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let Some(raw) = Option::<Vec<serde_json::Value>>::deserialize(deserializer)? else {
        return Ok(None);
    };
    Ok(Some(
        raw.into_iter()
            .filter_map(|tool| serde_json::from_value::<Tool>(tool).ok())
            .collect(),
    ))
}

/// Response body for `POST /v1/responses/input_tokens`.
///
/// `Deserialize` is derived where the Anthropic count response is
/// serialize-only: this body is round-tripped by the frontend's integration
/// tests, which assert on the parsed shape rather than on raw JSON.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct CountInputTokensResponse {
    /// Always [`RESPONSE_INPUT_TOKENS_OBJECT`]. Required by the OpenAI spec.
    pub object: String,
    pub input_tokens: u32,
}

impl CountInputTokensResponse {
    pub fn new(input_tokens: u32) -> Self {
        Self {
            object: RESPONSE_INPUT_TOKENS_OBJECT.to_string(),
            input_tokens,
        }
    }
}

impl CountInputTokensRequest {
    /// Estimate input token count using a `len/3` heuristic.
    ///
    /// Same contract as `AnthropicCountTokensRequest::estimate_tokens`: sum the
    /// character lengths of everything that reaches the prompt, divide by three,
    /// and never report zero for input that carried content.
    ///
    /// This is an estimate, not a tokenization. A frontend serving a
    /// backend that tokenizes for itself has no tokenizer loaded, so this
    /// endpoint has to be able to answer without one.
    pub fn estimate_tokens(&self) -> u32 {
        let mut total_len: usize = 0;

        // `instructions` and a top-level string `input` are not free-floating
        // text: the converter turns them into a system and a user chat message
        // respectively, exactly like the item messages below. Charge them the
        // same role markers, or the identical prompt scores differently
        // depending on which shape the caller used to express it.
        //
        // Both are skipped when empty, because an absent field is not a
        // message. `InputParam::default()` is `Text("")`, so a body with no
        // `input` at all lands here — and it must stay worth zero.
        if let Some(instructions) = &self.instructions.as_ref().filter(|text| !text.is_empty()) {
            total_len += role_len(Role::System) + instructions.len();
        }

        match &self.input {
            InputParam::Text(text) if text.is_empty() => {}
            InputParam::Text(text) => total_len += role_len(Role::User) + text.len(),
            InputParam::Items(items) => total_len += estimate_input_items_len(items),
        }

        if let Some(tools) = &self.tools {
            for tool in tools {
                total_len += estimate_tool_len(tool);
            }
        }

        let tokens = total_len / 3;
        if tokens == 0 && total_len > 0 {
            1
        } else {
            tokens as u32
        }
    }
}

/// Approximate character cost of a role marker, using the same constants
/// `AnthropicCountTokensRequest::estimate_tokens` applies for the same purpose.
fn role_len(role: Role) -> usize {
    match role {
        Role::User => 4,
        Role::Assistant => 9,
        Role::System => 6,
        Role::Developer => 9,
    }
}

fn input_role_len(role: InputRole) -> usize {
    match role {
        InputRole::User => 4,
        InputRole::System => 6,
        InputRole::Developer => 9,
    }
}

/// The `tool` role marker on a tool-result message.
///
/// `role_len` covers only the roles upstream's `Role` enum models; chat
/// completions' `tool` role has no variant there, so it gets its own constant
/// on the same basis the others use — the length of the role word.
const TOOL_ROLE_LEN: usize = 4;

/// What an input item does to the converter's pending assistant message.
enum GroupEffect {
    /// Opens or extends the pending assistant message, emitting no message of
    /// its own.
    Assistant,
    /// Flushes any pending assistant message and emits its own.
    Flush,
    /// Neither emits nor flushes — the converter skips it outright.
    Skip,
}

/// Sum the input items, mirroring `translate_input_items` in the shared
/// api-translation crate — including its coalescing.
///
/// Assistant-side items do not each become a message. An echoed assistant
/// message, a function call, and a reasoning summary all push into one
/// pending turn, which is flushed only by the next non-assistant item or
/// by the end of the list. So the assistant role marker is charged once per
/// flushed group, not once per item: two parallel function calls are one
/// assistant turn and cost one marker between them.
///
/// A per-item sum cannot express that, which is why this walks the list rather
/// than mapping over it.
fn estimate_input_items_len(items: &[InputItem]) -> usize {
    let mut total = 0;
    let mut assistant_open = false;

    for item in items {
        let (effect, len) = measure_input_item(item);
        total += len;
        match effect {
            GroupEffect::Assistant => {
                if !assistant_open {
                    assistant_open = true;
                    total += role_len(Role::Assistant);
                }
            }
            GroupEffect::Flush => assistant_open = false,
            GroupEffect::Skip => {}
        }
    }

    total
}

/// Measure one item and report what it does to the pending assistant group.
///
/// Assistant-side arms return content only: their role marker is the caller's
/// to add, once per group.
fn measure_input_item(item: &InputItem) -> (GroupEffect, usize) {
    match item {
        // A pointer to an item held server-side. The content it names is not in
        // this request, so there is nothing here to measure — and the converter
        // rejects it, so it cannot split an assistant group either.
        InputItem::ItemReference(_) => (GroupEffect::Skip, 0),
        InputItem::EasyMessage(message) => {
            let content = estimate_easy_content_len(&message.content);
            match message.role {
                // A prior assistant turn echoed back; coalesces like the strict
                // `MessageItem::Output` path.
                Role::Assistant => (GroupEffect::Assistant, content),
                role => (GroupEffect::Flush, role_len(role) + content),
            }
        }
        InputItem::Item(item) => measure_item(item),
    }
}

fn estimate_easy_content_len(content: &EasyInputContent) -> usize {
    match content {
        EasyInputContent::Text(text) => text.len(),
        EasyInputContent::ContentList(parts) => parts.iter().map(estimate_input_content_len).sum(),
    }
}

/// Only text parts are measured. An image or file contributes tokens as a
/// function of its decoded form, which a character count cannot model at all —
/// the Anthropic estimator skips non-text blocks for the same reason.
fn estimate_input_content_len(part: &InputContent) -> usize {
    match part {
        InputContent::InputText(text) => text.text.len(),
        InputContent::InputImage(_) | InputContent::InputFile(_) => 0,
    }
}

fn measure_item(item: &Item) -> (GroupEffect, usize) {
    match item {
        Item::Message(MessageItem::Input(message)) => (
            GroupEffect::Flush,
            input_role_len(message.role)
                + message
                    .content
                    .iter()
                    .map(estimate_input_content_len)
                    .sum::<usize>(),
        ),
        // Assistant-side: pushed into the pending message, so no role marker
        // here. See `estimate_input_items_len`.
        Item::Message(MessageItem::Output(message)) => (
            GroupEffect::Assistant,
            message
                .content
                .iter()
                .map(|part| match part {
                    InputOutputMessageContent::OutputText(text) => text.text.len(),
                    InputOutputMessageContent::Refusal(refusal) => refusal.refusal.len(),
                })
                .sum::<usize>(),
        ),
        // Rendered as a tool call on the pending assistant message. `call_id`
        // is excluded deliberately: it is correlation plumbing, not prompt
        // text, in the templates Dynamo renders.
        Item::FunctionCall(call) => (
            GroupEffect::Assistant,
            call.name.len() + call.arguments.len(),
        ),
        // Its own `tool`-role message, one per output, so it both flushes the
        // assistant group and carries a role marker of its own.
        Item::FunctionCallOutput(output) => (
            GroupEffect::Flush,
            TOOL_ROLE_LEN
                + match &output.output {
                    FunctionCallOutput::Text(text) => text.len(),
                    FunctionCallOutput::Content(parts) => parts
                        .iter()
                        .map(|part| match part {
                            UpstreamInputContent::InputText(text) => text.text.len(),
                            UpstreamInputContent::InputImage(_)
                            | UpstreamInputContent::InputFile(_) => 0,
                        })
                        .sum(),
                },
        ),
        // The converter folds both `content` (reasoning text) and `summary`
        // into the pending assistant turn as thinking deltas, so both are
        // measured. `encrypted_content` is excluded: it is an opaque blob the
        // model never sees as prompt text, and it is routinely far larger than
        // the reasoning it stands for.
        Item::Reasoning(reasoning) => (
            GroupEffect::Assistant,
            reasoning
                .content
                .iter()
                .flatten()
                .map(|part| match part {
                    ReasoningItemContent::ReasoningText(text) => text.text.len(),
                })
                .sum::<usize>()
                + reasoning
                    .summary
                    .iter()
                    .map(|part| match part {
                        SummaryPart::SummaryText(text) => text.text.len(),
                    })
                    .sum::<usize>(),
        ),
        // The converter flushes, then renders the plaintext as an assistant
        // message `"Agent message from {author}:\n{text}"`; encrypted or empty
        // ones are dropped entirely.
        Item::AgentMessage(message) => {
            let mut text_len = 0usize;
            for part in &message.content {
                let AgentMessageInputContent::InputText { text } = part else {
                    return (GroupEffect::Flush, 0);
                };
                text_len += text.len();
            }
            if text_len == 0 {
                return (GroupEffect::Flush, 0);
            }
            (
                GroupEffect::Flush,
                role_len(Role::Assistant) + message.author.len() + text_len,
            )
        }
        // Not transcript: the converter merges these into the declared tool
        // list, so they cost what the tool declarations cost. The values are
        // unvalidated JSON, so the serialized length is the available measure.
        Item::AdditionalTools(additional) => (
            GroupEffect::Flush,
            additional
                .tools
                .iter()
                .map(|tool| tool.to_string().len())
                .sum(),
        ),
        // The converter renders the call on the pending assistant turn and the
        // result as its own tool message; approximate the pair as one flushed
        // group carrying both contents.
        Item::McpCall(call) => (
            GroupEffect::Flush,
            role_len(Role::Assistant)
                + call.name.len()
                + call.arguments.len()
                + TOOL_ROLE_LEN
                + call
                    .error
                    .as_ref()
                    .map(|error| error.to_text().len())
                    .or_else(|| call.output.as_ref().map(String::len))
                    .unwrap_or(0),
        ),
        // Everything below contributes nothing, because nothing below reaches
        // the prompt: the converter flushes and skips every one of these
        // ("we do not have a faithful Chat Completions mapping"). The arms
        // above are exactly its handled set. Measuring the serialized form
        // instead would bill callers for JSON scaffolding the model never
        // sees: a bare `web_search_call` is 59 characters, or 19 phantom
        // tokens, and agentic clients echo many such items per turn.
        //
        // Listed out rather than wildcarded on purpose. `Item` is a shadow
        // enum that mirrors upstream variant-for-variant and has to be
        // extended by hand whenever upstream grows a variant. A `_` arm would
        // let that new variant default to zero silently; an exhaustive match
        // turns it into a compile error that forces a render-or-not decision
        // here, the same mechanism the `From` impls rely on to catch drift.
        Item::ConfigurationUpdate(_)
        | Item::Compaction(_)
        | Item::FileSearchCall(_)
        | Item::ComputerCall(_)
        | Item::ComputerCallOutput(_)
        | Item::WebSearchCall(_)
        | Item::ToolSearchCall(_)
        | Item::ToolSearchOutput(_)
        | Item::ImageGenerationCall(_)
        | Item::CodeInterpreterCall(_)
        | Item::LocalShellCall(_)
        | Item::LocalShellCallOutput(_)
        | Item::ShellCall(_)
        | Item::ShellCallOutput(_)
        | Item::ApplyPatchCall(_)
        | Item::ApplyPatchCallOutput(_)
        | Item::McpListTools(_)
        | Item::McpApprovalRequest(_)
        | Item::McpApprovalResponse(_)
        | Item::CustomToolCallOutput(_)
        | Item::CustomToolCall(_) => (GroupEffect::Flush, 0),
    }
}

/// Mirrors the converter's tool handling: only function tools are forwarded to
/// the backend, namespaced ones flattened to their bare function members.
/// Hosted tools (web search, file search, computer use) are dropped there and
/// so cost nothing here.
///
/// Wildcarded where `measure_item` is exhaustive, and deliberately so:
/// `Tool` is upstream's type, not one of our shadows, so we carry no
/// obligation to mirror its variants. Pinning it exhaustively would only
/// break the build every time async-openai adds a hosted tool we would
/// score as zero anyway.
fn estimate_tool_len(tool: &Tool) -> usize {
    match tool {
        Tool::Function(function) => function_tool_len(
            &function.name,
            function.description.as_ref(),
            function.parameters.as_ref(),
        ),
        Tool::Namespace(namespace) => namespace
            .tools
            .iter()
            .map(|tool| match tool {
                // The namespace name is an origin marker used to detect
                // collisions, not prompt text — the converter forwards the
                // bare function name.
                NamespaceToolParamTool::Function(function) => function_tool_len(
                    &function.name,
                    function.description.as_ref(),
                    function.parameters.as_ref(),
                ),
                NamespaceToolParamTool::Custom(_) => 0,
            })
            .sum(),
        _ => 0,
    }
}

fn function_tool_len(
    name: &str,
    description: Option<&String>,
    parameters: Option<&serde_json::Value>,
) -> usize {
    name.len()
        + description.map_or(0, |description| description.len())
        + parameters.map_or(0, |schema| schema.to_string().len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasoning_effort_max_survives_to_the_worker() {
        let json = serde_json::json!({
            "input": "hi",
            "reasoning": {"effort": "max"}
        });
        let req: CreateResponse = serde_json::from_value(json).unwrap();
        assert_eq!(
            req.reasoning.unwrap().effort,
            Some(B10ReasoningEffort::Max),
            "the serve-side policy must see the client's word, not an alias of it"
        );
    }

    #[test]
    fn reasoning_effort_max_echoes_unchanged() {
        let param = B10ReasoningParam {
            effort: Some(B10ReasoningEffort::Max),
            summary: None,
        };
        assert_eq!(
            Reasoning::from(&param).effort,
            Some(ReasoningEffort::Max),
            "the upstream echo preserves `max`"
        );
    }

    #[test]
    fn reasoning_effort_off_ladder_omits_the_echo() {
        let param = B10ReasoningParam {
            effort: Some(B10ReasoningEffort::Other(serde_json::json!("turbo"))),
            summary: None,
        };
        assert_eq!(Reasoning::from(&param).effort, None);
    }

    #[test]
    fn reasoning_effort_canonical_values_pass_through() {
        for (raw, expected) in [
            ("none", B10ReasoningEffort::None),
            ("minimal", B10ReasoningEffort::Minimal),
            ("low", B10ReasoningEffort::Low),
            ("medium", B10ReasoningEffort::Medium),
            ("high", B10ReasoningEffort::High),
            ("xhigh", B10ReasoningEffort::Xhigh),
            ("max", B10ReasoningEffort::Max),
        ] {
            let json = serde_json::json!({
                "input": "hi",
                "reasoning": {"effort": raw}
            });
            let req: CreateResponse = serde_json::from_value(json).unwrap();
            assert_eq!(
                req.reasoning.unwrap().effort,
                Some(expected),
                "effort {raw}"
            );
        }
    }

    #[test]
    fn reasoning_block_without_effort_still_parses() {
        let json = serde_json::json!({
            "input": "hi",
            "reasoning": {"summary": "auto"}
        });
        let req: CreateResponse = serde_json::from_value(json).unwrap();
        let reasoning = req.reasoning.unwrap();
        assert_eq!(reasoning.effort, None);
    }

    #[test]
    fn reasoning_effort_off_ladder_reaches_the_worker_verbatim() {
        let json = serde_json::json!({
            "input": "hi",
            "reasoning": {"effort": "turbo"}
        });
        let req: CreateResponse = serde_json::from_value(json).unwrap();
        assert_eq!(
            req.reasoning.unwrap().effort,
            Some(B10ReasoningEffort::Other(serde_json::json!("turbo")))
        );
    }

    #[test]
    fn omitted_reasoning_defaults_to_none() {
        let json = serde_json::json!({"input": "hi"});
        let req: CreateResponse = serde_json::from_value(json).unwrap();
        assert!(req.reasoning.is_none());
    }

    #[test]
    fn relaxed_assistant_message_without_id_or_status() {
        let json = serde_json::json!({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hi"}]
        });
        let item: InputItem = serde_json::from_value(json).unwrap();
        match item {
            InputItem::Item(Item::Message(MessageItem::Output(out))) => {
                assert_eq!(out.role, AssistantRole::Assistant);
                assert!(out.id.is_none());
                assert!(out.status.is_none());
            }
            other => panic!("expected Item::Message(Output), got {other:?}"),
        }
    }

    #[test]
    fn input_image_without_detail_defaults_to_auto() {
        let json = serde_json::json!({
            "type": "input_image",
            "image_url": "https://example.com/cat.jpg"
        });
        let content: InputContent = serde_json::from_value(json).unwrap();
        match content {
            InputContent::InputImage(img) => assert_eq!(img.detail, ImageDetail::Auto),
            other => panic!("expected InputImage, got {other:?}"),
        }
    }

    #[test]
    fn input_image_with_explicit_null_detail_defaults_to_auto() {
        let json = serde_json::json!({
            "type": "input_image",
            "image_url": "https://example.com/cat.jpg",
            "detail": null
        });
        let content: InputContent = serde_json::from_value(json).unwrap();
        match content {
            InputContent::InputImage(img) => assert_eq!(img.detail, ImageDetail::Auto),
            other => panic!("expected InputImage, got {other:?}"),
        }
    }

    #[test]
    fn assistant_message_without_content_field_deserializes() {
        // Bare assistant shell — no `content` field at all. Seen in real
        // Codex/Agents-SDK traffic on pure tool-call turns. `#[serde(default)]`
        // on `content` must accept omission and yield an empty vec.
        let json = serde_json::json!({
            "type": "message",
            "role": "assistant"
        });
        let item: InputItem = serde_json::from_value(json).unwrap();
        match item {
            InputItem::Item(Item::Message(MessageItem::Output(out))) => {
                assert_eq!(out.role, AssistantRole::Assistant);
                assert!(out.content.is_empty());
                assert!(out.id.is_none());
                assert!(out.status.is_none());
            }
            other => panic!("expected Item::Message(Output), got {other:?}"),
        }
    }

    #[test]
    fn assistant_message_with_explicit_null_content_deserializes() {
        // Mirrors the `annotations: null` case: some serializers emit JSON null
        // for absent fields instead of omitting them. `Vec::deserialize` rejects
        // null, so `content` also needs `deserialize_null_as_empty_vec`.
        let json = serde_json::json!({
            "type": "message",
            "role": "assistant",
            "content": null
        });
        let item: InputItem = serde_json::from_value(json).unwrap();
        match item {
            InputItem::Item(Item::Message(MessageItem::Output(out))) => {
                assert!(out.content.is_empty());
            }
            other => panic!("expected Item::Message(Output), got {other:?}"),
        }
    }

    #[test]
    fn mcp_call_item_deserializes() {
        // Guards against Item variant drift vs upstream — MCP item types were
        // added after the initial owned `Item` chain landed.
        let json = serde_json::json!({
            "type": "mcp_call",
            "id": "mcp_1",
            "server_label": "srv",
            "name": "t",
            "arguments": "{}"
        });
        let item: InputItem = serde_json::from_value(json).unwrap();
        assert!(matches!(item, InputItem::Item(Item::McpCall(_))));
    }

    #[test]
    fn strict_assistant_message_still_deserializes() {
        let json = serde_json::json!({
            "type": "message",
            "role": "assistant",
            "id": "msg_1",
            "status": "completed",
            "content": [{"type": "output_text", "text": "hi", "annotations": []}]
        });
        let item: InputItem = serde_json::from_value(json).unwrap();
        match item {
            InputItem::Item(Item::Message(MessageItem::Output(out))) => {
                assert_eq!(out.id.as_deref(), Some("msg_1"));
                assert_eq!(out.status, Some(OutputStatus::Completed));
            }
            other => panic!("expected Item::Message(Output), got {other:?}"),
        }
    }

    #[test]
    fn user_message_routes_to_input_variant() {
        let json = serde_json::json!({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hi"}]
        });
        let item: InputItem = serde_json::from_value(json).unwrap();
        assert!(matches!(
            item,
            InputItem::Item(Item::Message(MessageItem::Input(_)))
        ));
    }

    #[test]
    fn function_call_item_still_deserializes() {
        let json = serde_json::json!({
            "type": "function_call",
            "call_id": "c",
            "name": "f",
            "arguments": "{}"
        });
        let item: InputItem = serde_json::from_value(json).unwrap();
        assert!(matches!(item, InputItem::Item(Item::FunctionCall(_))));
    }

    #[test]
    fn easy_message_string_content_routes_to_easymessage() {
        let json = serde_json::json!({"role": "assistant", "content": "x"});
        let item: InputItem = serde_json::from_value(json).unwrap();
        assert!(matches!(item, InputItem::EasyMessage(_)));
    }

    #[test]
    fn output_text_without_annotations_defaults_empty() {
        let json = serde_json::json!({"type": "output_text", "text": "hi"});
        let part: InputOutputMessageContent = serde_json::from_value(json).unwrap();
        match part {
            InputOutputMessageContent::OutputText(t) => {
                assert!(t.annotations.is_empty());
            }
            _ => panic!("expected OutputText"),
        }
    }

    #[test]
    fn output_text_with_explicit_null_annotations_deserializes_as_empty() {
        // Some clients serialize absent fields as JSON null instead of omitting
        // them. `Vec::deserialize` would reject null; the custom deserializer
        // treats explicit null identically to a missing field.
        let json = serde_json::json!({"type": "output_text", "text": "hi", "annotations": null});
        let part: InputOutputMessageContent = serde_json::from_value(json).unwrap();
        match part {
            InputOutputMessageContent::OutputText(t) => {
                assert!(t.annotations.is_empty());
            }
            _ => panic!("expected OutputText"),
        }
    }

    #[test]
    fn assistant_message_with_explicit_null_id_and_status_deserializes() {
        // `Option<T>` natively accepts null as `None`, so these explicit-null
        // fields should flow through without a custom deserializer. This test
        // pins that behavior against accidental regressions (e.g. if someone
        // switches the field type away from `Option<_>`).
        let json = serde_json::json!({
            "type": "message",
            "role": "assistant",
            "id": null,
            "status": null,
            "content": [{"type": "output_text", "text": "hi", "annotations": null}]
        });
        let item: InputItem = serde_json::from_value(json).unwrap();
        match item {
            InputItem::Item(Item::Message(MessageItem::Output(out))) => {
                assert!(out.id.is_none());
                assert!(out.status.is_none());
                assert_eq!(out.content.len(), 1);
            }
            other => panic!("expected Item::Message(Output), got {other:?}"),
        }
    }

    #[test]
    fn create_response_roundtrip_with_relaxed_input() {
        let body = serde_json::json!({
            "model": "m",
            "input": [
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "hi"}
                ]},
                {"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"},
                {"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": "\n\n"}
                ]},
                {"type": "function_call_output", "call_id": "c", "output": "x"}
            ]
        });

        let req: CreateResponse = serde_json::from_value(body).unwrap();
        let items = match &req.input {
            InputParam::Items(items) => items,
            _ => panic!("expected Items"),
        };
        assert_eq!(items.len(), 4);
        assert!(matches!(
            items[2],
            InputItem::Item(Item::Message(MessageItem::Output(_)))
        ));
    }

    // ---- EasyInputMessage / multimodal-without-`type` regression coverage ----
    // See issue #9468. Before the EasyInputMessage/EasyInputContent shadow
    // landed, the `InputItem::EasyMessage` fallback still routed through
    // upstream's strict `InputImageContent` (required `detail`), so any
    // multimodal message that omitted the spec-default `type: "message"` would
    // fail with "data did not match any variant of untagged enum InputItem".

    #[test]
    fn easy_message_multimodal_without_type_routes_to_easymessage() {
        // AIPerf's pre-PR-931 payload shape: no top-level `type`, content is a
        // list containing an `input_image` part with no `detail`.
        let json = serde_json::json!({
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,abc"}
            ]
        });
        let item: InputItem = serde_json::from_value(json).unwrap();
        match item {
            InputItem::EasyMessage(easy) => {
                assert_eq!(easy.role, Role::User);
                assert_eq!(easy.r#type, MessageType::Message);
                match easy.content {
                    EasyInputContent::ContentList(parts) => {
                        assert_eq!(parts.len(), 1);
                        match &parts[0] {
                            InputContent::InputImage(img) => {
                                assert_eq!(img.detail, ImageDetail::Auto);
                                assert_eq!(
                                    img.image_url.as_deref(),
                                    Some("data:image/png;base64,abc")
                                );
                            }
                            other => panic!("expected InputImage, got {other:?}"),
                        }
                    }
                    other => panic!("expected ContentList, got {other:?}"),
                }
            }
            other => panic!("expected EasyMessage, got {other:?}"),
        }
    }

    #[test]
    fn easy_message_multimodal_with_explicit_null_detail() {
        // Same shape as above but with `detail: null` — exercises the
        // null-as-default path on the relaxed `InputImageContent` reached via
        // the EasyMessage variant.
        let json = serde_json::json!({
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,abc", "detail": null}
            ]
        });
        let item: InputItem = serde_json::from_value(json).unwrap();
        assert!(matches!(item, InputItem::EasyMessage(_)));
    }

    #[test]
    fn easy_message_assistant_multimodal_without_type() {
        // Mixed-turn shape AIPerf emits when the prior assistant turn carried
        // structured (non-string) content: role=assistant, content list, no
        // top-level `type`.
        let json = serde_json::json!({
            "role": "assistant",
            "content": [
                {"type": "input_text", "text": "ok"}
            ]
        });
        let item: InputItem = serde_json::from_value(json).unwrap();
        match item {
            InputItem::EasyMessage(easy) => {
                assert_eq!(easy.role, Role::Assistant);
            }
            other => panic!("expected EasyMessage(assistant), got {other:?}"),
        }
    }

    #[test]
    fn easy_message_text_only_without_type_unchanged() {
        // Regression guard: the pre-existing text-only path was already
        // working (no multimodal content -> never hit upstream's strict
        // `InputImageContent`). Pin it so a future glob-shadow change can't
        // break it.
        let json = serde_json::json!({"role": "user", "content": "Hello"});
        let item: InputItem = serde_json::from_value(json).unwrap();
        match item {
            InputItem::EasyMessage(easy) => {
                assert_eq!(easy.role, Role::User);
                assert!(matches!(easy.content, EasyInputContent::Text(ref s) if s == "Hello"));
            }
            other => panic!("expected EasyMessage(Text), got {other:?}"),
        }
    }

    #[test]
    fn easy_message_with_explicit_type_still_routes_to_item_message() {
        // AIPerf's post-PR-931 payload (with `type: "message"`) should still
        // hit the structured `Item::Message` path first — proving the existing
        // strict path didn't regress when EasyMessage was shadowed.
        let json = serde_json::json!({
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,abc"}
            ]
        });
        let item: InputItem = serde_json::from_value(json).unwrap();
        match item {
            InputItem::Item(Item::Message(MessageItem::Input(msg))) => {
                assert_eq!(msg.role, InputRole::User);
                assert_eq!(msg.content.len(), 1);
            }
            other => panic!("expected Item::Message(Input), got {other:?}"),
        }
    }

    #[test]
    fn create_response_roundtrip_aiperf_pre_pr931_payload() {
        // End-to-end shape: the exact request body AIPerf was emitting before
        // PR-931 for a multi-turn multimodal conversation. Mirrors what the
        // HTTP frontend receives. Must deserialize without error and preserve
        // turn ordering.
        let body = serde_json::json!({
            "model": "Qwen/Qwen2-VL-2B-Instruct",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Describe"},
                        {"type": "input_image", "image_url": "data:image/png;base64,abc"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "input_text", "text": "ok"}]
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Now describe a different one."}]
                }
            ]
        });
        let req: CreateResponse = serde_json::from_value(body).unwrap();
        let items = match &req.input {
            InputParam::Items(items) => items,
            _ => panic!("expected Items"),
        };
        assert_eq!(items.len(), 3);
        // All three turns must land as EasyMessage (no top-level `type`).
        for (idx, item) in items.iter().enumerate() {
            assert!(
                matches!(item, InputItem::EasyMessage(_)),
                "turn {idx} did not route to EasyMessage: {item:?}",
            );
        }
    }

    // ---- CountInputTokens (`POST /v1/responses/input_tokens`) ----

    fn count_req(json: serde_json::Value) -> CountInputTokensRequest {
        serde_json::from_value(json).expect("CountInputTokensRequest should deserialize")
    }

    #[test]
    fn count_string_input_and_instructions() {
        let req = count_req(serde_json::json!({
            "model": "m",
            "instructions": "be terse",
            "input": "Hello, world!",
        }));
        // (role 6 + 8) + (role 4 + 13) = 31 chars -> 10 tokens.
        assert_eq!(req.estimate_tokens(), (6 + 8 + 4 + 13) / 3);
    }

    #[test]
    fn count_absent_or_null_input_is_zero() {
        assert_eq!(
            count_req(serde_json::json!({"model": "m"})).estimate_tokens(),
            0
        );
        assert_eq!(
            count_req(serde_json::json!({"model": "m", "input": null})).estimate_tokens(),
            0
        );
    }

    #[test]
    fn count_never_reports_zero_for_nonempty_input() {
        let req = count_req(serde_json::json!({"input": "hi"}));
        assert_eq!(req.estimate_tokens(), (4 + 2) / 3);
        // Below the /3 floor: a one-character function name is 1 char total.
        let req = count_req(serde_json::json!({
            "tools": [{"type": "function", "name": "a"}]
        }));
        assert_eq!(req.estimate_tokens(), 1);
    }

    #[test]
    fn count_items_coalesces_assistant_group() {
        let req = count_req(serde_json::json!({
            "input": [
                {"role": "user", "content": "twelve chars"},
                {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
                {"type": "function_call", "call_id": "c2", "name": "g", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "done"}
            ]
        }));
        // user: 4 + 12; assistant group: one role marker (9) + 2 * (1 + 2);
        // tool message: 4 + 4.
        let want = (4 + 12) + (9 + 2 * 3) + (4 + 4);
        assert_eq!(req.estimate_tokens(), (want / 3) as u32);
    }

    #[test]
    fn count_reasoning_scores_content_and_summary_not_encrypted() {
        let req = count_req(serde_json::json!({
            "input": [
                {"type": "reasoning",
                 "summary": [{"type": "summary_text", "text": "abc"}],
                 "content": [{"type": "reasoning_text", "text": "abcdef"}],
                 "encrypted_content": "AAAAAAAABBBBBBBB"}
            ]
        }));
        // assistant group marker (9) + 6 + 3; encrypted blob excluded.
        assert_eq!(req.estimate_tokens(), (9 + 6 + 3) / 3);
    }

    #[test]
    fn count_unknown_tool_shapes_are_dropped_not_rejected() {
        let req = count_req(serde_json::json!({
            "input": "hi",
            "tools": [
                {"type": "definitely-not-a-tool", "blob": 1},
                {"type": "function", "name": "get_weather", "description": "w", "parameters": {"type": "object"}}
            ]
        }));
        let tools = req.tools.as_ref().expect("tools survive");
        assert_eq!(tools.len(), 1, "unknown tool shape must be dropped");
        // input: 4 + 2; function tool: 11 + 1 + len("{\"type\":\"object\"}")=16.
        let want = (4 + 2) + (11 + 1 + 16);
        assert_eq!(req.estimate_tokens(), (want / 3) as u32);
    }

    #[test]
    fn count_agent_message_plaintext_scored_encrypted_zero() {
        let req = count_req(serde_json::json!({
            "input": [
                {"type": "agent_message", "author": "/root", "recipient": "/root/w",
                 "content": [{"type": "input_text", "text": "hello agent"}]},
                {"type": "agent_message", "author": "/root", "recipient": "/root/w",
                 "content": [{"type": "encrypted_content", "encrypted_content": "AAAA"}]}
            ]
        }));
        // Only the plaintext one counts: role 9 + author 5 + text 11.
        assert_eq!(req.estimate_tokens(), (9 + 5 + 11) / 3);
    }

    #[test]
    fn count_additional_tools_scored_as_tool_declarations() {
        let tool = serde_json::json!({"name": "x", "description": "y"});
        let serialized_len = tool.to_string().len();
        let req = count_req(serde_json::json!({
            "input": [
                {"type": "additional_tools", "role": "developer", "tools": [tool]}
            ]
        }));
        assert_eq!(req.estimate_tokens(), (serialized_len / 3) as u32);
    }

    #[test]
    fn count_response_shape() {
        let resp = CountInputTokensResponse::new(7);
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["object"], "response.input_tokens");
        assert_eq!(json["input_tokens"], 7);
    }
}
