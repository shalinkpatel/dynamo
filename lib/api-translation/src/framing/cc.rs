//! ChatCompletions framing — the canonical internal protocol, so this edge is near-identity.
//!
//! Every wire shape here is a vendored `dynamo_protocols` type (patch 0002 gives the response types
//! the serialization differences an emitter needs). Nothing here is restated locally.
//!
//! Server-tool activity has no native CC shape, so it rides `baseten.iterations` (see
//! `docs/architecture.md` "Protocol capability split"). `continuation_messages` carries the loop
//! transcript as CC request-messages a client appends, which Messages gets natively from its blocks.

use std::collections::{HashMap, HashSet, hash_map::Entry};

use dynamo_protocols::types::{
    ChatChoice, ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallChunk, ChatCompletionRequestAssistantMessageContent,
    ChatCompletionResponseMessage, ChatCompletionStreamResponseDelta, CompletionUsage,
    CreateChatCompletionResponse, CreateChatCompletionStreamResponse, FinishReason, FunctionCall,
    FunctionCallStream, FunctionType, ReasoningContent, Role,
};

use super::{
    BufferedResponse, CompletedIteration, ProtocolEnvelope, StagedIteration, StreamFraming,
    openai_error_sse_frame,
};
use crate::baseten_response_extension::{
    BasetenFrame, BasetenResponseExtension, IterationScope, ServerToolCallOutcome,
    ServerToolCallRecord,
};
use crate::model::{ErrorClass, ServerToolCall, Termination, ToolCall, ToolCallDelta};
use crate::util::unix_secs;
use crate::wire::{next_id_seq, sse_frame, to_json_string};
use crate::{CcMessage, SemanticChunk};

const CHUNK_OBJECT: &str = "chat.completion.chunk";

/// Monomorphizes [`BasetenResponseExtension`] on CC's own usage shape — confined here since nothing outside
/// this module renders CC.
type CcExtension = BasetenResponseExtension<CompletionUsage>;

pub(super) struct CcEnvelope;

impl ProtocolEnvelope for CcEnvelope {
    fn stream_framing(self: Box<Self>, model: String) -> Box<dyn StreamFraming> {
        Box::new(CcFraming::new(model))
    }

    fn buffered_body(&self, response: &BufferedResponse<'_>) -> String {
        let created = created_now();
        // One `content` field, so every iteration's text concatenates: CC cannot separate them.
        let text = concatenated(response.transcript, assistant_text);
        let reasoning = concatenated(response.transcript, assistant_reasoning);
        let tool_calls: Vec<ChatCompletionMessageToolCall> = response
            .client_tool_calls
            .iter()
            .map(cc_tool_call)
            .collect();
        #[expect(
            deprecated,
            reason = "vendored struct literal must name `function_call`"
        )]
        let message = ChatCompletionResponseMessage {
            role: Role::Assistant,
            // `null`, not omitted, on a tool-only answer — the OpenAI shape.
            content: (!text.is_empty() || tool_calls.is_empty())
                .then_some(ChatCompletionMessageContent::Text(text)),
            reasoning_content: (!reasoning.is_empty()).then_some(reasoning),
            tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
            refusal: None,
            function_call: None,
            audio: None,
        };
        let body = CreateChatCompletionResponse {
            id: cc_id(created),
            object: "chat.completion".to_string(),
            created,
            model: response.model.to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message,
                finish_reason: Some(cc_finish_reason(response.termination)),
                logprobs: None,
            }],
            usage: Some(response.usage.clone()),
            service_tier: None,
            system_fingerprint: None,
        };
        let extension = CcExtension {
            iterations: response.iterations.to_vec(),
            request: response
                .termination
                .request_scope(response.request_server_tool_calls),
        };
        to_json_string(&BasetenFrame {
            body: &body,
            baseten: (!extension.is_empty()).then_some(&extension),
        })
    }
}

/// The cap's closest fit is `length` — same contract for the client, an answer cut short with no tool
/// calls to run — where `tool_calls` would demand a round TB already ran and `stop` would claim a
/// complete answer.
fn cc_finish_reason(termination: Termination) -> FinishReason {
    match termination {
        Termination::Model(stop) => stop,
        Termination::ReactCapExhausted => FinishReason::Length,
    }
}

fn concatenated(transcript: &[CcMessage], pick: impl Fn(&CcMessage) -> Option<&str>) -> String {
    transcript.iter().filter_map(pick).collect()
}

fn assistant_text(message: &CcMessage) -> Option<&str> {
    match message {
        CcMessage::Assistant(assistant) => match assistant.content.as_ref()? {
            ChatCompletionRequestAssistantMessageContent::Text(text) => Some(text),
            // The loop only ever builds `Text`; an `Array` here would be a history-fold bug.
            ChatCompletionRequestAssistantMessageContent::Array(_) => None,
        },
        _ => None,
    }
}

fn assistant_reasoning(message: &CcMessage) -> Option<&str> {
    match message {
        CcMessage::Assistant(assistant) => match assistant.reasoning_content.as_ref()? {
            ReasoningContent::Text(reasoning) => Some(reasoning),
            ReasoningContent::Segments(_) => None,
        },
        _ => None,
    }
}

/// The model's verbatim argument bytes, never a reserialize — byte-exact cache prefix.
fn cc_tool_call(call: &ToolCall) -> ChatCompletionMessageToolCall {
    ChatCompletionMessageToolCall {
        id: call.id.clone(),
        r#type: FunctionType::Function,
        function: FunctionCall {
            name: call.name.clone(),
            arguments: call.raw_args.clone(),
        },
    }
}

/// Seconds since the epoch as the OpenAI `created` field types it. Wraps in 2106; matching the
/// schema's width beats widening it here.
fn created_now() -> u32 {
    unix_secs() as u32
}

/// Synthetic ChatCompletions response id (`chatcmpl-<created>-<seq>`). The client SDK surfaces it
/// but does not parse the format.
fn cc_id(created: u32) -> String {
    format!("chatcmpl-{created}-{}", next_id_seq())
}

/// One call's position in the `tool_calls` array SDKs accumulate streamed fragments into.
#[derive(Clone, Copy)]
struct ToolCallSlot {
    index: u32,
    /// Set on the fragment that must carry id and name.
    is_opener: bool,
}

struct CcFraming {
    id: String,
    created: u32,
    model: String,
    role_sent: bool,
    /// Index of each call still streaming, retired once the call completes. A model that restarts
    /// its id numbering per iteration would otherwise have the repeat read as a continuation of the
    /// finished call.
    tool_call_index_by_id: HashMap<String, u32>,
    /// SDKs accumulate streamed `tool_calls` by their integer `index`, which must stay distinct
    /// across the whole response — not just within one iteration.
    next_tool_call_index: u32,
    /// Streaming ids whose only fragment so far carried empty args. `finalize`'s empty-args ->
    /// `"{}"` substitution never reaches a client mid-stream, so a call that stays in this set
    /// through completion needs one more fragment correcting it, or the client's accumulated
    /// `arguments` stays `""` forever.
    streamed_with_empty_args: HashSet<String>,
}

impl CcFraming {
    fn new(model: String) -> Self {
        let created = created_now();
        Self {
            id: cc_id(created),
            created,
            model,
            role_sent: false,
            tool_call_index_by_id: HashMap::new(),
            next_tool_call_index: 0,
            streamed_with_empty_args: HashSet::new(),
        }
    }

    fn streaming_slot(&mut self, call_id: &str) -> ToolCallSlot {
        let next_index = self.next_tool_call_index;
        let slot = match self.tool_call_index_by_id.entry(call_id.to_string()) {
            Entry::Occupied(entry) => ToolCallSlot {
                index: *entry.get(),
                is_opener: false,
            },
            Entry::Vacant(entry) => ToolCallSlot {
                index: *entry.insert(next_index),
                is_opener: true,
            },
        };
        if slot.is_opener {
            self.next_tool_call_index += 1;
        }
        slot
    }

    /// The OpenAI streaming shape: the opener carries id/name, later fragments only the args.
    fn tool_call_frame(
        &mut self,
        slot: ToolCallSlot,
        call_id: &str,
        name: &str,
        arguments: String,
    ) -> String {
        let chunk = ChatCompletionMessageToolCallChunk {
            index: slot.index,
            id: slot.is_opener.then(|| call_id.to_string()),
            r#type: slot.is_opener.then_some(FunctionType::Function),
            function: Some(FunctionCallStream {
                name: slot.is_opener.then(|| name.to_string()),
                arguments: Some(arguments),
            }),
        };
        let base = self.empty_delta();
        self.delta_frame(ChatCompletionStreamResponseDelta {
            tool_calls: Some(vec![chunk]),
            ..base
        })
    }

    fn tool_call_delta_frame(&mut self, delta: &ToolCallDelta) -> String {
        let slot = self.streaming_slot(&delta.id);
        if delta.args_delta.is_empty() {
            if slot.is_opener {
                self.streamed_with_empty_args.insert(delta.id.clone());
            }
        } else {
            self.streamed_with_empty_args.remove(&delta.id);
        }
        self.tool_call_frame(slot, &delta.id, &delta.name, delta.args_delta.clone())
    }

    /// `role` rides the response's first delta only; every later one omits it.
    fn take_role(&mut self) -> Option<Role> {
        (!std::mem::replace(&mut self.role_sent, true)).then_some(Role::Assistant)
    }

    fn empty_delta(&mut self) -> ChatCompletionStreamResponseDelta {
        ChatCompletionStreamResponseDelta {
            role: self.take_role(),
            content: None,
            reasoning_content: None,
            tool_calls: None,
            refusal: None,
            function_call: None,
        }
    }

    /// The terminal chunk's delta, which OpenAI leaves empty. `role` never rides it: a client that
    /// opens its assistant message on `delta.role` would open one at the very end.
    fn terminal_delta(&self) -> ChatCompletionStreamResponseDelta {
        ChatCompletionStreamResponseDelta {
            role: None,
            content: None,
            reasoning_content: None,
            tool_calls: None,
            refusal: None,
            function_call: None,
        }
    }

    fn delta_frame(&mut self, delta: ChatCompletionStreamResponseDelta) -> String {
        self.chunk_frame(
            vec![ChatChoiceStream {
                index: 0,
                delta,
                finish_reason: None,
                logprobs: None,
            }],
            None,
            None,
        )
    }

    /// One iteration scope on a frame of its own, which carries `choices: []` so no client SDK folds
    /// it into the message.
    fn iteration_frame(&self, iteration: IterationScope<CompletionUsage>) -> String {
        self.chunk_frame(
            Vec::new(),
            None,
            Some(&CcExtension {
                iterations: vec![iteration],
                request: None,
            }),
        )
    }

    fn chunk_frame(
        &self,
        choices: Vec<ChatChoiceStream>,
        usage: Option<&CompletionUsage>,
        baseten: Option<&CcExtension>,
    ) -> String {
        let chunk = CreateChatCompletionStreamResponse {
            id: self.id.clone(),
            object: CHUNK_OBJECT.to_string(),
            created: self.created,
            model: self.model.clone(),
            choices,
            usage: usage.cloned(),
            service_tier: None,
            system_fingerprint: None,
        };
        sse_frame(
            None,
            &to_json_string(&BasetenFrame {
                body: &chunk,
                baseten,
            }),
        )
    }
}

impl StreamFraming for CcFraming {
    fn on_chunk(&mut self, chunk: &SemanticChunk) -> Vec<String> {
        let (content, reasoning_content) = match chunk {
            SemanticChunk::TextDelta(text) => (Some(text.clone()), None),
            SemanticChunk::ThinkingDelta(thinking) => (None, Some(thinking.clone())),
            SemanticChunk::ToolCallDelta(tool_call_delta) => {
                return vec![self.tool_call_delta_frame(tool_call_delta)];
            }
            // Surfaced by emit_client_tool_calls / emit_completed_iteration / finish instead.
            SemanticChunk::ToolCall(_) | SemanticChunk::Usage(_) | SemanticChunk::Stop { .. } => {
                return Vec::new();
            }
        };
        let delta = ChatCompletionStreamResponseDelta {
            content: content.map(ChatCompletionMessageContent::Text),
            reasoning_content,
            ..self.empty_delta()
        };
        vec![self.delta_frame(delta)]
    }

    /// One frame for the whole iteration, never one per call: a client appending
    /// `continuation_messages` in arrival order must never receive a call without its result.
    fn emit_completed_iteration(&mut self, iteration: &CompletedIteration<'_>) -> Vec<String> {
        vec![self.iteration_frame(IterationScope {
            server_tool_calls: iteration.server_tool_calls(),
            continuation_messages: iteration.continuation_messages.to_vec(),
            ..IterationScope::at(iteration.index)
        })]
    }

    /// One frame carrying every call, never one frame each — the SDK reads `index` per entry.
    fn emit_client_tool_calls(&mut self, calls: &[ToolCall]) -> Vec<String> {
        let chunks: Vec<ChatCompletionMessageToolCallChunk> = calls
            .iter()
            .filter_map(|call| {
                let Some(index) = self.tool_call_index_by_id.remove(&call.id) else {
                    // Linear emission opens every client call with a streamed fragment, which is
                    // what registers its index.
                    debug_assert!(false, "client tool call `{}` never streamed", call.id);
                    tracing::error!(
                        event_name = "cc.unstreamed_client_tool_call",
                        "Client tool call `{}` completed without streaming a fragment",
                        call.id
                    );
                    return None;
                };
                // A stream of only-empty deltas never saw `finalize`'s `"{}"` substitution.
                self.streamed_with_empty_args.remove(&call.id).then_some(
                    ChatCompletionMessageToolCallChunk {
                        index,
                        id: None,
                        r#type: None,
                        function: Some(FunctionCallStream {
                            name: None,
                            arguments: Some(call.raw_args.clone()),
                        }),
                    },
                )
            })
            .collect();
        if chunks.is_empty() {
            return Vec::new();
        }
        let delta = ChatCompletionStreamResponseDelta {
            tool_calls: Some(chunks),
            ..self.empty_delta()
        };
        vec![self.delta_frame(delta)]
    }

    fn emit_dispatched_server_tool(
        &mut self,
        iteration: u32,
        server_call: &ServerToolCall,
    ) -> Vec<String> {
        vec![self.iteration_frame(IterationScope {
            server_tool_calls: vec![ServerToolCallRecord::dispatched(server_call)],
            ..IterationScope::at(iteration)
        })]
    }

    fn stage_iteration(&mut self, staged: StagedIteration<'_>) -> Vec<String> {
        vec![self.iteration_frame(staged.scope(CompletionUsage::clone))]
    }

    fn finish(
        &mut self,
        termination: Termination,
        usage: &CompletionUsage,
        server_tool_calls: &[ServerToolCallOutcome],
    ) -> Vec<String> {
        let mut frames = Vec::new();
        // A response with no content at all (the ReAct cap can end one) still owes the client the
        // opening role delta OpenAI always sends, on its own chunk ahead of the terminal one.
        if !self.role_sent {
            let opening = self.empty_delta();
            frames.push(self.delta_frame(opening));
        }
        let extension = CcExtension {
            request: termination.request_scope(server_tool_calls),
            ..CcExtension::default()
        };
        frames.push(self.chunk_frame(
            vec![ChatChoiceStream {
                index: 0,
                delta: self.terminal_delta(),
                finish_reason: Some(cc_finish_reason(termination)),
                logprobs: None,
            }],
            Some(usage),
            (!extension.is_empty()).then_some(&extension),
        ));
        frames.push(sse_frame(None, "[DONE]"));
        frames
    }

    fn error_sse_frame(
        &mut self,
        class: ErrorClass,
        error_code: Option<&str>,
        message: &str,
    ) -> String {
        openai_error_sse_frame(class, error_code, message)
    }
}
