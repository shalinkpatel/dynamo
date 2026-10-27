//! Per-client-protocol rendering of TB's output. The loop and the egress speak one protocol-neutral
//! vocabulary ([`SemanticChunk`], [`CompletedIteration`], [`BufferedResponse`]); each protocol's
//! module turns that into its own bytes.
//!
//! [`ClientProtocol::envelope`] is the single place a protocol becomes an implementation, so adding
//! one is a new module plus one arm there. (Ingress adaptation dispatches separately, in
//! [`super::request`] — it is the other edge, with its own per-protocol signature.)
//!
//! Both traits are entirely synchronous: the async edge is the frame channel, owned by
//! [`super::sse_emitter::SseEmitter`], so `dyn` dispatch here costs no boxed future per chunk.

mod cc;
mod messages;
mod responses;

pub use responses::ResponsesParams;

use std::collections::{HashMap, HashSet};

use dynamo_protocols::error::{ApiError, WrappedError};
use dynamo_protocols::types::CompletionUsage;
use http::StatusCode;

use crate::baseten_response_extension::{
    IterationScope, ServerToolCallOutcome, ServerToolCallRecord,
};
use crate::coding_adapter::CodingAdapter;
use crate::model::{
    BackendError, ErrorClass, ServerToolCall, Termination, ToolCall, ToolInvocation,
};
use crate::wire::{sse_frame, to_json_string};
use crate::{CcMessage, ClientProtocol, SemanticChunk};

/// OpenAI's error shape (`{"error": {...}}`), identical on CC and Responses — shared rather than
/// duplicated per envelope.
fn openai_error_json(class: ErrorClass, error_code: Option<&str>, message: &str) -> String {
    to_json_string(&WrappedError {
        error: ApiError {
            message: message.to_string(),
            r#type: Some(openai_error_type(class).to_string()),
            param: None,
            code: error_code.map(str::to_owned),
        },
    })
}

/// OpenAI's `error.type` vocabulary. SDKs read it as the retryability signal (`api_error`
/// conventionally retries), so a caller fault must not carry it.
fn openai_error_type(class: ErrorClass) -> &'static str {
    match class {
        ErrorClass::RateLimited => "rate_limit_error",
        ErrorClass::InvalidRequest
        | ErrorClass::Authentication
        | ErrorClass::Permission
        | ErrorClass::NotFound
        | ErrorClass::RequestTooLarge => "invalid_request_error",
        ErrorClass::Overloaded | ErrorClass::Internal => "api_error",
    }
}

fn openai_error_sse_frame(class: ErrorClass, error_code: Option<&str>, message: &str) -> String {
    sse_frame(None, &openai_error_json(class, error_code, message))
}

/// The status a buffered path would have answered for this backend failure, used only to pick the
/// error-type word on the in-band error frame. Falls back to 500 when the upstream reported none.
fn backend_error_status(error: &BackendError) -> StatusCode {
    error
        .http_status
        .and_then(|status| StatusCode::from_u16(status).ok())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
}

/// One iteration's `baseten` scope before a protocol shapes it, so [`StreamFraming`] needs one
/// staging method rather than one per [`IterationScope`] field.
pub struct StagedIteration<'a> {
    pub index: u32,
    pub usage: Option<&'a CompletionUsage>,
    pub debug_msg: Option<&'a str>,
}

impl StagedIteration<'_> {
    fn scope<Usage: serde::Serialize>(
        &self,
        convert_usage: impl Fn(&CompletionUsage) -> Usage,
    ) -> IterationScope<Usage> {
        IterationScope {
            usage: self.usage.map(convert_usage),
            debug_msg: self.debug_msg.map(str::to_owned).into_iter().collect(),
            ..IterationScope::at(self.index)
        }
    }
}

/// Holds the in-flight iteration's scope out of the flushable extension until its index advances
/// or the stream finishes. One iteration's parts arrive in different frames (a steering
/// `debug_msg` before the model call, `usage` after it), and flushing them as they come hands a
/// streamed client two `iterations[]` entries at one index where the buffered body merges them.
pub(crate) struct OpenIterationScope<Usage: serde::Serialize> {
    open: Option<IterationScope<Usage>>,
    /// The last index [`Self::close`] released. Staging into it again would build a second entry at
    /// one index, which the open scope can no longer catch.
    released: Option<u32>,
}

impl<Usage: serde::Serialize> Default for OpenIterationScope<Usage> {
    fn default() -> Self {
        Self {
            open: None,
            released: None,
        }
    }
}

impl<Usage: serde::Serialize> OpenIterationScope<Usage> {
    /// Merges into the open scope on a matching index; a new index closes the open scope and
    /// returns it, complete and ready to flush.
    pub(crate) fn stage(&mut self, scope: IterationScope<Usage>) -> Option<IterationScope<Usage>> {
        debug_assert!(
            self.released.is_none_or(|released| scope.index > released),
            "iteration {} staged after its scope was released",
            scope.index
        );
        match &mut self.open {
            Some(open) if open.index == scope.index => {
                debug_assert!(
                    open.usage.is_none() || scope.usage.is_none(),
                    "two usage stagings for iteration {}",
                    scope.index
                );
                if open.usage.is_none() {
                    open.usage = scope.usage;
                }
                open.server_tool_calls.extend(scope.server_tool_calls);
                open.continuation_messages
                    .extend(scope.continuation_messages);
                open.debug_msg.extend(scope.debug_msg);
                None
            }
            _ => {
                if let Some(open) = &self.open {
                    debug_assert!(
                        open.index < scope.index,
                        "iteration scopes staged out of order: {} after {}",
                        scope.index,
                        open.index
                    );
                }
                self.open.replace(scope)
            }
        }
    }

    pub(crate) fn close(&mut self) -> Option<IterationScope<Usage>> {
        let closed = self.open.take();
        if let Some(scope) = &closed {
            self.released = Some(scope.index);
        }
        closed
    }
}

/// Rendering state for one streamed client response: every method returns the frames to send.
pub trait StreamFraming: Send {
    fn on_chunk(&mut self, chunk: &SemanticChunk) -> Vec<String>;
    /// One iteration's end: its finished server-tool calls and the messages a client appends to
    /// continue from it. Messages renders the `tool_result` blocks (their `tool_use` already
    /// streamed) and needs no continuation history; CC has no native shape for either.
    fn emit_completed_iteration(&mut self, iteration: &CompletedIteration<'_>) -> Vec<String>;
    /// One server tool the moment it is dispatched, so a CC client can show the arguments while the
    /// call runs. Messages already streamed the same call as a native `tool_use` block.
    fn emit_dispatched_server_tool(
        &mut self,
        iteration: u32,
        server_call: &ServerToolCall,
    ) -> Vec<String>;
    /// One server tool the host refuses to run (budget, fan-out cap), at the point it decides; its
    /// error result still arrives with the iteration. Responses opens the call's item here so it
    /// keeps the model's order; the other protocols already streamed it or record nothing.
    fn emit_refused_server_tool(
        &mut self,
        _iteration: u32,
        _server_call: &ServerToolCall,
    ) -> Vec<String> {
        Vec::new()
    }
    /// Client-executed tool calls on the terminal iteration. CC emits them natively; Messages
    /// already streamed them as `tool_use` blocks.
    fn emit_client_tool_calls(&mut self, calls: &[ToolCall]) -> Vec<String>;
    /// Where an iteration scope lands: a frame of its own (CC) or the next frame that preserves
    /// extras (Messages, Responses).
    fn stage_iteration(&mut self, staged: StagedIteration<'_>) -> Vec<String>;

    fn emit_iteration_usage(&mut self, iteration: u32, usage: &CompletionUsage) -> Vec<String> {
        self.stage_iteration(StagedIteration {
            index: iteration,
            usage: Some(usage),
            debug_msg: None,
        })
    }

    fn emit_iteration_debug_msg(&mut self, iteration: u32, message: &str) -> Vec<String> {
        self.stage_iteration(StagedIteration {
            index: iteration,
            usage: None,
            debug_msg: Some(message),
        })
    }

    /// Every part of the iteration is staged, so its scope can ride the next extras-preserving
    /// frame instead of waiting for the next iteration to displace it. CC keeps the default: it
    /// frames each scope on its own chunk, so nothing is ever left open.
    fn close_iteration(&mut self) {}
    /// Close the response with terminal stop, cumulative usage, and the request-level server-tool
    /// transcript (every completed call, in dispatch order across iterations).
    fn finish(
        &mut self,
        termination: Termination,
        usage: &CompletionUsage,
        server_tool_calls: &[ServerToolCallOutcome],
    ) -> Vec<String>;
    /// The terminal error frame, mid-stream. On the live framing (`&mut self`), not the stateless
    /// envelope: Responses' `sequence_number` must continue from the frames already sent.
    fn error_sse_frame(
        &mut self,
        class: ErrorClass,
        error_code: Option<&str>,
        message: &str,
    ) -> String;
    /// Terminal backend failure: the stream is over and this error is why. CC and Messages render
    /// their single protocol error frame; Responses overrides with the spec failure envelope
    /// (`response.failed` carrying the partial output) and the truncation rescue.
    fn finish_with_backend_error(&mut self, error: &BackendError) -> Vec<String> {
        vec![self.error_sse_frame(
            ErrorClass::from_status(backend_error_status(error)),
            error.error_code.as_deref(),
            &error.message,
        )]
    }
}

/// One client protocol's response shapes: a streaming renderer and the buffered body. Holds the
/// request's [`CodingAdapter`], if any, so neither the loop nor the egress has to carry it.
pub trait ProtocolEnvelope: Send + Sync {
    fn stream_framing(self: Box<Self>, model: String) -> Box<dyn StreamFraming>;
    /// The whole non-streaming response body, already encoded.
    fn buffered_body(&self, response: &BufferedResponse<'_>) -> String;
}

impl ClientProtocol {
    /// `responses_params` are the Responses request's echo-back params
    /// ([`AdaptedIngress::responses_params`](crate::request::AdaptedIngress)); the other
    /// protocols ignore them, pass `Default::default()`.
    pub fn envelope(
        self,
        coding_adapter: Option<Box<dyn CodingAdapter>>,
        responses_params: ResponsesParams,
    ) -> Box<dyn ProtocolEnvelope> {
        match self {
            Self::ChatCompletions => Box::new(cc::CcEnvelope),
            Self::Messages => Box::new(messages::MessagesEnvelope::new(coding_adapter)),
            Self::Responses => Box::new(responses::ResponsesEnvelope::new(
                coding_adapter,
                responses_params,
            )),
        }
    }

    /// The protocol's JSON error object: OpenAI `{"error": {...}}` / Anthropic error body. A
    /// function of the protocol alone, so the pre-flight path reaches it with no request in flight.
    /// Messages has no code slot: Anthropic's `error.type` is a closed vocabulary, not ours to
    /// extend.
    pub fn error_json(self, class: ErrorClass, error_code: Option<&str>, message: &str) -> String {
        match self {
            Self::ChatCompletions | Self::Responses => {
                openai_error_json(class, error_code, message)
            }
            Self::Messages => messages::error_json(class, message),
        }
    }
}

/// One iteration that has finished, tool calls included. Both edges take the same borrow, so they
/// cannot disagree about which iteration a result belongs to.
pub struct CompletedIteration<'a> {
    pub index: u32,
    /// Empty when the model called no server tool in this iteration.
    pub invocations: &'a [ToolInvocation],
    /// This iteration's slice of the loop transcript: its assistant message plus the `tool` results
    /// answering it.
    pub continuation_messages: &'a [CcMessage],
}

impl CompletedIteration<'_> {
    pub(crate) fn server_tool_call_outcomes(&self) -> Vec<ServerToolCallOutcome> {
        self.invocations
            .iter()
            .map(
                |ToolInvocation {
                     server_call,
                     output,
                 }| { ServerToolCallOutcome::completed(server_call, output) },
            )
            .collect()
    }

    pub(crate) fn server_tool_calls(&self) -> Vec<ServerToolCallRecord> {
        self.invocations
            .iter()
            .map(
                |ToolInvocation {
                     server_call,
                     output,
                 }| { ServerToolCallRecord::completed(server_call, output) },
            )
            .collect()
    }
}

/// A finished non-streaming response, protocol-neutral. Each [`ProtocolEnvelope`] renders it into its
/// own body shape.
pub struct BufferedResponse<'a> {
    pub model: &'a str,
    /// Everything the loop appended, in order: one assistant message per iteration, each followed by
    /// the `tool` results answering it. The single source for the body's content — a second ordered
    /// accumulation could disagree with it about what the model saw.
    pub transcript: &'a [CcMessage],
    /// The calls the client must execute, which the transcript declares but does not answer.
    pub client_tool_calls: &'a [ToolCall],
    /// Recorded in canonical CC form, as everything between the two edges is; each protocol maps it
    /// into its own usage shape when rendering.
    pub iterations: &'a [IterationScope<CompletionUsage>],
    /// The request-level server-tool transcript, for the `request` scope.
    pub request_server_tool_calls: &'a [ServerToolCallOutcome],
    pub termination: Termination,
    pub usage: &'a CompletionUsage,
}

impl BufferedResponse<'_> {
    /// The ids of the server-tool calls that failed. The CC transcript's `tool` messages carry no
    /// error flag, so buffered renderers read it from the iteration records.
    /// Provider label per dispatched server-tool name, from the loop's own call records.
    pub(crate) fn server_tool_providers(&self) -> HashMap<&str, &str> {
        self.iterations
            .iter()
            .flat_map(|iteration| &iteration.server_tool_calls)
            .map(|record| {
                (
                    record.identity.name.as_str(),
                    record.identity.provider.as_str(),
                )
            })
            .collect()
    }

    fn failed_server_tool_call_ids(&self) -> HashSet<&str> {
        self.iterations
            .iter()
            .flat_map(|iteration| &iteration.server_tool_calls)
            .filter(|record| record.is_error == Some(true))
            .map(|record| record.identity.id.as_str())
            .collect()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
#[path = "mod_test.rs"]
mod tests;
