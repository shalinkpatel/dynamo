// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Vendored Kimi K3 native grammar, with local early argument projection.
//!
//! Native source: ai-dynamo/frontend-crates, parsers/v2/src/unified/kimi_k3.rs
//! at 79b3c206fc7af040e64572f5a630d6348d05a5b5 (dynamo-parsers-v2 0.7.7).
//! `find_first_outside_strings` comes from tool_calling/scan.rs at that revision.
//! The guided grammar/router is excluded; the adapter delegates guided JSON to
//! upstream through its public interface. Native boundary/recovery behavior is
//! retained. Streaming projection and its partial-error contract are local.
//! Review upstream K3 changes when upgrading dependencies and run the native
//! parity and streaming tests before porting fixes into this snapshot.

use std::collections::HashMap;

use super::streaming::StreamingCall;

use serde_json::Value;

use crate::{UnifiedParserOutput, UnifiedParserStartingState, upstream::ToolCallDelta};

const OPEN: &str = "<|open|>";
const CLOSE: &str = "<|close|>";
pub(super) const END_OF_MSG: &str = "<|end_of_msg|>";

const THINK_OPEN: Marker = Marker::pair("<|open|>think<|sep|>", "<|open|> think <|sep|>");
const THINK_CLOSE: Marker = Marker::pair("<|close|>think<|sep|>", "<|close|> think <|sep|>");
const THINK_CLOSE_HEAD: Marker = Marker::pair("<|close|>think", "<|close|> think");
const RESPONSE_OPEN: Marker = Marker::pair("<|open|>response<|sep|>", "<|open|> response <|sep|>");
const RESPONSE_CLOSE: Marker =
    Marker::pair("<|close|>response<|sep|>", "<|close|> response <|sep|>");
const TOOLS_OPEN: Marker = Marker::pair("<|open|>tools<|sep|>", "<|open|> tools <|sep|>");
pub(super) const TOOLS_CLOSE: Marker =
    Marker::pair("<|close|>tools<|sep|>", "<|close|> tools <|sep|>");
const MESSAGE_OPEN: Marker = Marker::pair("<|open|>message", "<|open|> message");
const ASSISTANT_MESSAGE_OPEN: Marker = Marker::pair(
    "<|open|>message role=\"assistant\"<|sep|>",
    "<|open|> message role=\"assistant\" <|sep|>",
);
pub(super) const MESSAGE_CLOSE: Marker =
    Marker::pair("<|close|>message<|sep|>", "<|close|> message <|sep|>");
const CALL_OPEN: Marker = Marker::pair("<|open|>call", "<|open|> call");
pub(super) const CALL_CLOSE: Marker =
    Marker::pair("<|close|>call<|sep|>", "<|close|> call <|sep|>");
pub(super) const ARG_OPEN: Marker = Marker::pair("<|open|>argument", "<|open|> argument");
pub(super) const ARG_CLOSE: Marker =
    Marker::pair("<|close|>argument<|sep|>", "<|close|> argument <|sep|>");
pub(super) const JSON_OPEN: Marker = Marker::pair("<|open|>json", "<|open|> json");
pub(super) const JSON_CLOSE: Marker =
    Marker::pair("<|close|>json<|sep|>", "<|close|> json <|sep|>");

const ALL_MARKERS: &[Marker] = &[
    THINK_OPEN,
    THINK_CLOSE,
    RESPONSE_OPEN,
    RESPONSE_CLOSE,
    TOOLS_OPEN,
    TOOLS_CLOSE,
    ASSISTANT_MESSAGE_OPEN,
    MESSAGE_CLOSE,
    CALL_OPEN,
    CALL_CLOSE,
    ARG_OPEN,
    ARG_CLOSE,
    JSON_OPEN,
    JSON_CLOSE,
    Marker::single(END_OF_MSG),
];

const IDLE_MARKERS: &[Marker] = ALL_MARKERS;
const SEP_MARKER: Marker = Marker::pair("<|sep|>", " <|sep|>");
const REASONING_MARKERS: &[Marker] = &[
    THINK_CLOSE_HEAD,
    THINK_OPEN,
    THINK_CLOSE,
    RESPONSE_OPEN,
    RESPONSE_CLOSE,
    TOOLS_OPEN,
    TOOLS_CLOSE,
    CALL_OPEN,
    CALL_CLOSE,
    ARG_OPEN,
    ARG_CLOSE,
    JSON_OPEN,
    JSON_CLOSE,
    MESSAGE_CLOSE,
    Marker::single(END_OF_MSG),
];
const RESPONSE_MARKERS: &[Marker] = &[
    RESPONSE_CLOSE,
    TOOLS_OPEN,
    CALL_OPEN,
    MESSAGE_CLOSE,
    Marker::single(END_OF_MSG),
];
const TOOLS_MARKERS: &[Marker] = &[
    CALL_OPEN,
    TOOLS_CLOSE,
    MESSAGE_CLOSE,
    Marker::single(END_OF_MSG),
];
#[derive(Clone, Copy)]
pub(super) struct Marker {
    canonical: &'static str,
    spaced: Option<&'static str>,
}

impl Marker {
    const fn single(canonical: &'static str) -> Self {
        Self {
            canonical,
            spaced: None,
        }
    }

    const fn pair(canonical: &'static str, spaced: &'static str) -> Self {
        Self {
            canonical,
            spaced: Some(spaced),
        }
    }

    pub(super) fn variants(self) -> impl Iterator<Item = &'static str> {
        std::iter::once(self.canonical).chain(self.spaced)
    }

    fn match_at(self, text: &str) -> Option<(usize, usize)> {
        self.variants().find_map(|variant| {
            text.starts_with(variant)
                .then_some((0, variant.len()))
                .or_else(|| {
                    variant
                        .strip_prefix(' ')
                        .filter(|variant| text.starts_with(variant))
                        .map(|variant| (0, variant.len()))
                })
        })
    }

    pub(super) fn prefix_len(self, text: &str) -> Option<usize> {
        self.match_at(text).map(|(_, len)| len)
    }

    fn find(self, text: &str) -> Option<(usize, usize)> {
        self.variants()
            .filter_map(|variant| text.find(variant).map(|at| (at, variant.len())))
            .min_by_key(|(at, _)| *at)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum Mode {
    #[default]
    Idle,
    Reasoning,
    Response,
    Tools,
    Call,
    Done,
}

#[derive(Debug)]
struct ActiveCall {
    name: String,
    id: String,
    return_mode: Mode,
}

#[derive(Debug, Clone, Copy)]
struct KimiK3HeaderScan {
    scanned: usize,
    #[cfg(test)]
    examined_bytes: usize,
}

impl KimiK3HeaderScan {
    fn new() -> Self {
        Self {
            scanned: 0,
            #[cfg(test)]
            examined_bytes: 0,
        }
    }

    fn advance(&mut self, text: &str, flush: bool) -> Option<(usize, usize)> {
        if text.len() < self.scanned {
            self.scanned = 0;
        }
        while self.scanned < text.len() {
            let suffix = &text[self.scanned..];
            #[cfg(test)]
            {
                self.examined_bytes += suffix.chars().next().map_or(0, char::len_utf8);
            }
            if let Some(len) = header_separator_at(suffix) {
                return Some((self.scanned, len));
            }
            if !flush && header_separator_is_partial(suffix) {
                break;
            }
            let character = suffix
                .chars()
                .next()
                .expect("non-empty Kimi K3 header suffix");
            self.scanned += character.len_utf8();
        }
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CallBoundary {
    Complete { body_end: usize, consumed: usize },
    Recover { body_end: usize },
    Resync { at: usize },
    Pending,
    Malformed,
}

/// Authoritative native K3 call-boundary owner.
struct KimiK3CallBoundary {
    return_channel: Mode,
    scanned: usize,
    header_len: Option<usize>,
    body_kind: CallBodyKind,
    root_call_open: Option<usize>,
    pending_call_open: Option<usize>,
    complete_call_open: Option<usize>,
    call_closes: Vec<TokenHit>,
    arg_opens: Vec<TokenHit>,
    arg_closes: Vec<TokenHit>,
    json_closes: Vec<TokenHit>,
    outer_closes: Vec<TokenHit>,
    completed_arguments: Option<String>,
    #[cfg(test)]
    scanned_bytes: usize,
    #[cfg(test)]
    parsed_body_bytes: usize,
    #[cfg(test)]
    body_parse_count: usize,
}

impl KimiK3CallBoundary {
    fn reset(&mut self) {
        *self = Self::new();
    }

    fn new() -> Self {
        Self {
            return_channel: Mode::Idle,
            scanned: 0,
            header_len: None,
            body_kind: CallBodyKind::Unknown,
            root_call_open: None,
            pending_call_open: None,
            complete_call_open: None,
            call_closes: Vec::new(),
            arg_opens: Vec::new(),
            arg_closes: Vec::new(),
            json_closes: Vec::new(),
            outer_closes: Vec::new(),
            completed_arguments: None,
            #[cfg(test)]
            scanned_bytes: 0,
            #[cfg(test)]
            parsed_body_bytes: 0,
            #[cfg(test)]
            body_parse_count: 0,
        }
    }

    fn begin(&mut self, header_len: usize, return_channel: Mode) {
        self.reset();
        self.header_len = Some(header_len);
        self.return_channel = return_channel;
    }

    fn advance(&mut self, text: &str, flush: bool) -> CallBoundary {
        self.scan_appended(text, flush);
        let Some(header_len) = self.header_len else {
            return if flush {
                CallBoundary::Malformed
            } else {
                CallBoundary::Pending
            };
        };

        if let Some(next_call) = self.complete_call_open.filter(|at| *at > header_len) {
            if let Some(call_close) = self
                .call_closes
                .iter()
                .copied()
                .filter(|close| close.end() <= next_call)
                .find(|close| {
                    let rest = text[close.end()..next_call].trim_start();
                    TOOLS_CLOSE.prefix_len(rest).is_some() || rest.is_empty()
                })
            {
                let arguments = self.parse_body(&text[header_len..call_close.at]);
                if arguments.is_none() {
                    return CallBoundary::Resync { at: next_call };
                }
                if text[call_close.end()..next_call].trim().is_empty() {
                    self.completed_arguments = arguments;
                    return CallBoundary::Complete {
                        body_end: call_close.at,
                        consumed: call_close.end(),
                    };
                }
            }
            if let Some(body_end) = self.recovery_body_end(text, header_len, next_call) {
                return self.recover_at(text, header_len, body_end);
            }
            if let Some(call_close) = self
                .call_closes
                .iter()
                .copied()
                .find(|close| close.end() <= next_call)
                && let Some(arguments) = self.parse_body(&text[header_len..call_close.at])
            {
                self.completed_arguments = Some(arguments);
                return CallBoundary::Complete {
                    body_end: call_close.at,
                    consumed: call_close.end(),
                };
            }
            if self.body_kind == CallBodyKind::Malformed {
                return CallBoundary::Resync { at: next_call };
            }
        }

        if let Some((call_close, arguments)) = self.structural_call_close(text, header_len, flush) {
            self.completed_arguments = Some(arguments);
            return CallBoundary::Complete {
                body_end: call_close.at,
                consumed: call_close.end(),
            };
        }

        if !flush {
            return CallBoundary::Pending;
        }
        let recovery_limit = self
            .outer_closes
            .iter()
            .copied()
            .find(|outer| {
                !self.outer_boundary_is_typed_data(
                    text,
                    header_len,
                    TokenHit {
                        at: outer.end(),
                        len: 0,
                    },
                )
            })
            .map_or(text.len(), |outer| outer.at);
        if let Some(body_end) = self.recovery_body_end(text, header_len, recovery_limit) {
            return self.recover_at(text, header_len, body_end);
        }
        CallBoundary::Malformed
    }

    fn structural_call_close(
        &mut self,
        text: &str,
        header_len: usize,
        flush: bool,
    ) -> Option<(TokenHit, String)> {
        let call_closes = self.call_closes.clone();
        for call_close in call_closes.into_iter().rev() {
            let boundary = self
                .outer_closes
                .iter()
                .copied()
                .any(|outer| outer.at >= call_close.end());
            if !boundary && (!flush || !text[call_close.end()..].trim().is_empty()) {
                continue;
            }
            let typed_data = self.outer_boundary_is_typed_data(text, header_len, call_close);
            let arguments = (!typed_data)
                .then(|| self.parse_body(&text[header_len..call_close.at]))
                .flatten();
            if let Some(arguments) = arguments {
                return Some((call_close, arguments));
            }
        }
        None
    }

    fn outer_boundary_is_typed_data(
        &self,
        text: &str,
        header_len: usize,
        call_close: TokenHit,
    ) -> bool {
        let Some((outer, arg_close, real_arg_open)) = self
            .outer_closes
            .iter()
            .copied()
            .filter(|outer| outer.at >= header_len && outer.at < call_close.at)
            .find_map(|outer| {
                let arg_close = self
                    .arg_closes
                    .iter()
                    .copied()
                    .rfind(|close| close.end() <= outer.at)?;
                let real_arg_open = self
                    .arg_opens
                    .iter()
                    .copied()
                    .rfind(|open| open.at < arg_close.at)?;
                let arg_type = parse_tag_header(&text[real_arg_open.at..], ARG_OPEN)
                    .and_then(|(attrs, _)| attr_value(&attrs, "type").map(str::to_string));
                let string_argument = arg_type.is_none_or(|arg_type| arg_type == "string");
                let later_arg_close = self
                    .arg_closes
                    .iter()
                    .copied()
                    .find(|close| close.at > outer.at)?;
                let later_arg_open = self
                    .arg_opens
                    .iter()
                    .copied()
                    .rfind(|open| open.at < later_arg_close.at)?;
                (string_argument && later_arg_open.at == real_arg_open.at).then_some((
                    outer,
                    arg_close,
                    real_arg_open,
                ))
            })
        else {
            return false;
        };
        if outer.end() < call_close.at {
            return self.arg_closes.iter().any(|close| close.at > outer.at)
                && self
                    .arg_opens
                    .iter()
                    .copied()
                    .rfind(|open| open.at < call_close.at)
                    .is_some_and(|open| open.at == real_arg_open.at);
        }
        let value_start = parse_tag_header(&text[real_arg_open.at..], ARG_OPEN)
            .map(|(_, len)| real_arg_open.at + len);
        value_start.is_some_and(|value_start| value_start <= arg_close.at)
    }

    fn recover_at(&mut self, text: &str, header_len: usize, body_end: usize) -> CallBoundary {
        match self.parse_body(&text[header_len..body_end]) {
            Some(arguments) => {
                self.completed_arguments = Some(arguments);
                CallBoundary::Recover { body_end }
            }
            None => CallBoundary::Malformed,
        }
    }

    fn parse_body(&mut self, body: &str) -> Option<String> {
        #[cfg(test)]
        {
            self.body_parse_count += 1;
            self.parsed_body_bytes += body.len();
        }
        parse_call_body(body)
    }

    fn recovery_body_end(&self, text: &str, header_len: usize, limit: usize) -> Option<usize> {
        if !self.call_closes.is_empty() {
            return None;
        }
        let candidate = match self.body_kind {
            CallBodyKind::Empty => Some(header_len),
            CallBodyKind::Arguments => self
                .arg_closes
                .iter()
                .copied()
                .rev()
                .find(|close| {
                    close.end() <= limit
                        && text[close.end()..limit].trim().is_empty()
                        && !self.outer_boundary_is_typed_data(
                            text,
                            header_len,
                            TokenHit { at: limit, len: 0 },
                        )
                })
                .map(TokenHit::end),
            CallBodyKind::RawJson => self
                .json_closes
                .iter()
                .copied()
                .rev()
                .find(|close| close.end() <= limit && text[close.end()..limit].trim().is_empty())
                .map(TokenHit::end),
            CallBodyKind::Unknown | CallBodyKind::Malformed => None,
        }?;
        (candidate <= limit && text[candidate..limit].trim().is_empty()).then_some(candidate)
    }

    fn take_arguments(&mut self) -> Option<String> {
        self.completed_arguments.take()
    }

    fn scan_appended(&mut self, text: &str, flush: bool) {
        if text.len() < self.scanned {
            self.reset_progress();
        }
        let ends_at_boundary = SCANNER_MARKERS
            .iter()
            .any(|(_, marker)| marker.variants().any(|variant| text.ends_with(variant)));
        let mut scan_limit = text
            .len()
            .saturating_sub((!flush && !ends_at_boundary) as usize * SCANNER_HOLDBACK);
        while !text.is_char_boundary(scan_limit) {
            scan_limit -= 1;
        }
        while self.scanned < scan_limit {
            let suffix = &text[self.scanned..scan_limit];
            if !flush && scanner_marker_is_partial(suffix) {
                break;
            }
            if let Some((kind, len)) = scanner_marker_at(suffix) {
                let hit = TokenHit {
                    at: self.scanned,
                    len,
                };
                self.note_token(text, kind, hit);
                self.scanned += len;
                #[cfg(test)]
                {
                    self.scanned_bytes += len;
                }
                continue;
            }

            let character = suffix.chars().next().expect("non-empty scanner suffix");
            if self
                .header_len
                .is_some_and(|header_len| self.scanned >= header_len)
                && self.body_kind == CallBodyKind::Unknown
                && !character.is_whitespace()
            {
                self.body_kind = CallBodyKind::Malformed;
            }
            self.scanned += character.len_utf8();
            #[cfg(test)]
            {
                self.scanned_bytes += character.len_utf8();
            }
        }
    }

    fn reset_progress(&mut self) {
        let return_channel = self.return_channel;
        let header_len = self.header_len;
        *self = Self::new();
        self.return_channel = return_channel;
        self.header_len = header_len;
    }

    fn note_token(&mut self, text: &str, kind: ScannerToken, hit: TokenHit) {
        if kind == ScannerToken::CallOpen {
            if self.root_call_open.is_none() {
                self.root_call_open = Some(hit.at);
            } else {
                self.pending_call_open = Some(hit.at);
            }
        }
        if kind == ScannerToken::Sep {
            if self.header_len.is_none()
                && let Some(at) = self.root_call_open
                && let Some((_, len)) = parse_call_header(&text[at..])
                && at + len == hit.end()
            {
                self.header_len = Some(hit.end());
            }
            if let Some(at) = self.pending_call_open
                && let Some((_, len)) = parse_call_header(&text[at..])
                && at + len == hit.end()
            {
                self.complete_call_open = Some(at);
                self.pending_call_open = None;
            }
        }

        let Some(header_len) = self.header_len else {
            return;
        };
        if hit.at >= header_len && self.body_kind == CallBodyKind::Unknown {
            self.body_kind = match kind {
                ScannerToken::ArgumentOpen => CallBodyKind::Arguments,
                ScannerToken::JsonOpen => CallBodyKind::RawJson,
                ScannerToken::CallClose => CallBodyKind::Empty,
                ScannerToken::Sep => return,
                _ => CallBodyKind::Malformed,
            };
        }
        if hit.at < header_len {
            return;
        }
        match kind {
            ScannerToken::CallClose => self.call_closes.push(hit),
            ScannerToken::ArgumentOpen => self.arg_opens.push(hit),
            ScannerToken::ArgumentClose => self.arg_closes.push(hit),
            ScannerToken::JsonClose => self.json_closes.push(hit),
            ScannerToken::ToolsClose | ScannerToken::MessageClose | ScannerToken::EndOfMessage => {
                self.outer_closes.push(hit)
            }
            ScannerToken::ThinkClose if self.return_channel == Mode::Reasoning => {
                self.outer_closes.push(hit)
            }
            ScannerToken::ResponseClose if self.return_channel == Mode::Response => {
                self.outer_closes.push(hit)
            }
            ScannerToken::CallOpen
            | ScannerToken::JsonOpen
            | ScannerToken::Sep
            | ScannerToken::ThinkClose
            | ScannerToken::ResponseClose => {}
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum CallBodyKind {
    #[default]
    Unknown,
    Empty,
    Arguments,
    RawJson,
    Malformed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TokenHit {
    at: usize,
    len: usize,
}

impl TokenHit {
    fn end(self) -> usize {
        self.at + self.len
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScannerToken {
    CallOpen,
    CallClose,
    ArgumentOpen,
    ArgumentClose,
    JsonOpen,
    JsonClose,
    ToolsClose,
    MessageClose,
    ThinkClose,
    ResponseClose,
    EndOfMessage,
    Sep,
}

const SCANNER_MARKERS: &[(ScannerToken, Marker)] = &[
    (ScannerToken::CallOpen, CALL_OPEN),
    (ScannerToken::CallClose, CALL_CLOSE),
    (ScannerToken::ArgumentOpen, ARG_OPEN),
    (ScannerToken::ArgumentClose, ARG_CLOSE),
    (ScannerToken::JsonOpen, JSON_OPEN),
    (ScannerToken::JsonClose, JSON_CLOSE),
    (ScannerToken::ToolsClose, TOOLS_CLOSE),
    (ScannerToken::MessageClose, MESSAGE_CLOSE),
    (ScannerToken::ThinkClose, THINK_CLOSE),
    (ScannerToken::ResponseClose, RESPONSE_CLOSE),
    (ScannerToken::EndOfMessage, Marker::single(END_OF_MSG)),
    (ScannerToken::Sep, SEP_MARKER),
];

const SCANNER_HOLDBACK: usize = 32;

fn scanner_marker_at(text: &str) -> Option<(ScannerToken, usize)> {
    SCANNER_MARKERS
        .iter()
        .flat_map(|(kind, marker)| marker.variants().map(|variant| (*kind, variant)))
        .filter_map(|(kind, marker)| text.starts_with(marker).then_some((kind, marker.len())))
        .max_by_key(|(_, len)| *len)
}

fn scanner_marker_is_partial(text: &str) -> bool {
    SCANNER_MARKERS
        .iter()
        .flat_map(|(_, marker)| marker.variants())
        .any(|marker| marker.len() > text.len() && marker.starts_with(text))
}

fn header_separator_at(text: &str) -> Option<usize> {
    SEP_MARKER
        .variants()
        .find(|variant| text.starts_with(variant))
        .map(str::len)
}

fn header_separator_is_partial(text: &str) -> bool {
    SEP_MARKER
        .variants()
        .any(|variant| variant.len() > text.len() && variant.starts_with(text))
}

/// K3-owned native state; guided parsing is delegated by the enclosing adapter.
pub(crate) struct KimiK3Native {
    buffer: String,
    pub(super) stream_arguments: bool,
    streaming_call: Option<StreamingCall>,
    streaming_error: Option<anyhow::Error>,
    mode: Mode,
    active_call: Option<ActiveCall>,
    call_header_scan: Option<KimiK3HeaderScan>,
    call_boundary: KimiK3CallBoundary,
    tools_open: String,
    tools_return_mode: Option<Mode>,
    next_tool_index: usize,
    pub(super) call_ids: Vec<String>,
}

impl KimiK3Native {
    pub(super) fn new() -> Self {
        Self {
            buffer: String::new(),
            stream_arguments: false,
            streaming_call: None,
            streaming_error: None,
            mode: Mode::Idle,
            active_call: None,
            call_header_scan: None,
            call_boundary: KimiK3CallBoundary::new(),
            tools_open: String::new(),
            tools_return_mode: None,
            next_tool_index: 0,
            call_ids: Vec::new(),
        }
    }

    fn drain(&mut self, flush: bool, output: &mut UnifiedParserOutput) {
        loop {
            let progressed = match self.mode {
                Mode::Idle => self.drain_idle(flush, output),
                Mode::Reasoning => self.drain_reasoning(flush, output),
                Mode::Response => self.drain_response(flush, output),
                Mode::Tools => self.drain_tools(flush, output),
                Mode::Call => self.drain_call(flush, output),
                Mode::Done => {
                    self.buffer.clear();
                    false
                }
            };
            if !progressed || self.streaming_error.is_some() {
                break;
            }
        }
    }

    fn drain_idle(&mut self, flush: bool, output: &mut UnifiedParserOutput) -> bool {
        if self.consume_assistant_message_open() {
            return true;
        }
        if self.consume(THINK_OPEN) {
            self.mode = Mode::Reasoning;
            return true;
        }
        if self.consume(RESPONSE_OPEN) {
            self.mode = Mode::Response;
            return true;
        }
        if self.consume_tools_open(Mode::Idle) {
            self.mode = Mode::Tools;
            return true;
        }
        if self.consume_call_open(Mode::Tools, flush, output) {
            if self.active_call.is_some() {
                self.mode = Mode::Call;
            }
            return true;
        }
        if self.consume_message_end() {
            self.mode = Mode::Done;
            return true;
        }
        if let Some(len) = self.consume_any_at_start(&[
            THINK_CLOSE,
            RESPONSE_CLOSE,
            TOOLS_CLOSE,
            CALL_CLOSE,
            ARG_OPEN,
            ARG_CLOSE,
            JSON_OPEN,
            JSON_CLOSE,
        ]) {
            tracing::warn!(
                why = "kimi_k3_orphan_marker",
                skipped_bytes = len,
                "stripping orphan Kimi K3 XTML marker"
            );
            return true;
        }
        self.emit_safe(flush, IDLE_MARKERS, output, |output, text| {
            output.push_text(text)
        })
    }

    fn drain_reasoning(&mut self, flush: bool, output: &mut UnifiedParserOutput) -> bool {
        if self.consume_reasoning_close(flush) {
            self.mode = Mode::Idle;
            return true;
        }
        if self.consume(RESPONSE_OPEN) {
            tracing::warn!(
                why = "kimi_k3_elided_think_close",
                "recovering Kimi K3 response channel without a think close"
            );
            self.mode = Mode::Response;
            return true;
        }
        if self.consume_tools_open(Mode::Reasoning) {
            tracing::warn!(
                why = "kimi_k3_elided_think_close",
                "recovering Kimi K3 tools channel without a think close"
            );
            self.mode = Mode::Tools;
            return true;
        }
        if self.consume_call_open(Mode::Reasoning, flush, output) {
            tracing::warn!(
                why = "kimi_k3_elided_think_close",
                "recovering Kimi K3 call without a think close"
            );
            if self.active_call.is_some() {
                self.mode = Mode::Call;
            }
            return true;
        }
        if self.consume_message_end() {
            self.mode = Mode::Done;
            return true;
        }
        if self.consume_reasoning_structure(flush) {
            return true;
        }
        if let Some(len) = self.consume_any_at_start(&[
            THINK_OPEN,
            RESPONSE_CLOSE,
            TOOLS_CLOSE,
            CALL_CLOSE,
            ARG_CLOSE,
            JSON_CLOSE,
        ]) {
            tracing::warn!(
                why = "kimi_k3_orphan_marker_in_reasoning",
                skipped_bytes = len,
                "quarantining orphan Kimi K3 structure inside reasoning"
            );
            return true;
        }
        self.emit_safe(flush, REASONING_MARKERS, output, |output, text| {
            output.push_reasoning(text)
        })
    }

    fn drain_response(&mut self, flush: bool, output: &mut UnifiedParserOutput) -> bool {
        if self.consume(RESPONSE_CLOSE) {
            self.mode = Mode::Idle;
            return true;
        }
        if self.consume_tools_open(Mode::Idle) {
            self.mode = Mode::Tools;
            return true;
        }
        if self.consume_call_open(Mode::Response, flush, output) {
            if self.active_call.is_some() {
                self.mode = Mode::Call;
            }
            return true;
        }
        if self.consume_message_end() {
            self.mode = Mode::Done;
            return true;
        }
        self.emit_safe(flush, RESPONSE_MARKERS, output, |output, text| {
            output.push_text(text)
        })
    }

    fn drain_tools(&mut self, flush: bool, output: &mut UnifiedParserOutput) -> bool {
        if self.consume_call_open(Mode::Tools, flush, output) {
            if self.active_call.is_some() {
                self.mode = Mode::Call;
            }
            return true;
        }
        if self.consume(TOOLS_CLOSE) {
            self.tools_open.clear();
            self.mode = self.tools_return_mode.take().unwrap_or(Mode::Idle);
            return true;
        }
        if self.consume_message_end() {
            self.mode = Mode::Done;
            return true;
        }
        self.drop_safe(flush, TOOLS_MARKERS)
    }

    fn drain_call(&mut self, flush: bool, output: &mut UnifiedParserOutput) -> bool {
        let boundary = self.call_boundary.advance(&self.buffer, flush);
        match boundary {
            CallBoundary::Complete { body_end, consumed } => {
                self.complete_call(body_end, consumed, output);
                true
            }
            CallBoundary::Recover { body_end } => {
                tracing::warn!(
                    why = "kimi_k3_recovered_missing_call_close",
                    recovered_bytes = body_end,
                    "recovering delimiter-terminated Kimi K3 call before an outer boundary"
                );
                self.complete_call(body_end, body_end, output);
                true
            }
            CallBoundary::Resync { .. } if self.streaming_call.is_some() => {
                self.streaming_error = Some(anyhow::anyhow!("malformed streaming Kimi K3 call"));
                false
            }
            CallBoundary::Resync { at } => {
                tracing::warn!(
                    why = "kimi_k3_resynchronized_after_incomplete_call",
                    skipped_bytes = at,
                    "dropping malformed Kimi K3 call and resuming at the next call"
                );
                self.buffer.drain(..at);
                self.active_call = None;
                self.call_header_scan = None;
                self.call_boundary.reset();
                self.mode = Mode::Tools;
                true
            }
            CallBoundary::Pending if !flush => {
                if let Some(call) = &mut self.streaming_call
                    && let Err(error) = call.advance(&self.buffer, output)
                {
                    self.streaming_error = Some(error);
                }
                false
            }
            CallBoundary::Pending | CallBoundary::Malformed if self.streaming_call.is_some() => {
                self.streaming_error = Some(anyhow::anyhow!("incomplete streaming Kimi K3 call"));
                false
            }
            CallBoundary::Pending | CallBoundary::Malformed => {
                tracing::warn!(
                    why = "kimi_k3_incomplete_call",
                    buffered_bytes = self.buffer.len(),
                    "dropping incomplete Kimi K3 call at EOF"
                );
                self.buffer.clear();
                self.active_call = None;
                self.call_boundary.reset();
                self.mode = Mode::Tools;
                true
            }
        }
    }

    fn complete_call(
        &mut self,
        body_end: usize,
        consumed: usize,
        output: &mut UnifiedParserOutput,
    ) {
        if let Some(streaming) = &mut self.streaming_call
            && let Err(error) = streaming.advance(&self.buffer[..body_end], output)
        {
            self.streaming_error = Some(error);
            return;
        }
        let arguments = self.call_boundary.take_arguments();
        self.buffer.drain(..consumed);
        self.call_boundary.reset();
        self.finish_call(arguments, output);
    }

    fn finish_call(&mut self, arguments: Option<String>, output: &mut UnifiedParserOutput) {
        let Some(call) = self.active_call.take() else {
            self.mode = Mode::Tools;
            return;
        };
        self.mode = call.return_mode;
        if call.name.is_empty() {
            tracing::warn!(
                why = "kimi_k3_missing_tool_name",
                "dropping Kimi K3 call without a tool name"
            );
            return;
        }
        let Some(arguments) = arguments else {
            tracing::warn!(
                why = "kimi_k3_malformed_call_body",
                "dropping malformed Kimi K3 call"
            );
            return;
        };

        if let Some(streaming) = self.streaming_call.take() {
            if !arguments.starts_with(&streaming.emitted) {
                self.streaming_error = Some(anyhow::anyhow!(
                    "streaming Kimi K3 arguments cannot be revised after emission"
                ));
                return;
            }
            output.push_call(ToolCallDelta {
                tool_index: streaming.tool_index,
                name: None,
                arguments: arguments[streaming.emitted.len()..].to_string(),
                complete: true,
            });
            return;
        }
        let tool_index = self.next_tool_index;
        self.next_tool_index += 1;
        self.call_ids.push(call.id);
        output.push_call(ToolCallDelta {
            tool_index,
            name: Some(call.name),
            arguments,
            complete: true,
        });
    }

    fn consume_call_open(
        &mut self,
        return_mode: Mode,
        flush: bool,
        output: &mut UnifiedParserOutput,
    ) -> bool {
        let Some(open_len) = CALL_OPEN.prefix_len(&self.buffer) else {
            self.call_header_scan = None;
            return false;
        };

        let scan = self
            .call_header_scan
            .get_or_insert_with(KimiK3HeaderScan::new);
        let separator = scan.advance(&self.buffer[open_len..], flush);
        let Some((sep_at, sep_len)) = separator else {
            if flush {
                tracing::warn!(
                    why = "kimi_k3_incomplete_call_header",
                    skipped_bytes = self.buffer.len(),
                    "dropping incomplete Kimi K3 call header at EOF"
                );
                if let Some((_, consumed)) = parse_attrs_prefix(&self.buffer[open_len..]) {
                    self.buffer.drain(..open_len + consumed);
                    self.emit_safe(true, IDLE_MARKERS, output, |output, text| {
                        output.push_text(text)
                    });
                } else {
                    self.buffer.clear();
                }
                self.call_header_scan = None;
                return true;
            }
            return false;
        };

        self.call_header_scan = None;
        let header_end = open_len + sep_at;
        let Some(attrs) = parse_attrs(&self.buffer[open_len..header_end]) else {
            let malformed_len = header_end + sep_len;
            tracing::warn!(
                why = "kimi_k3_malformed_call_header",
                skipped_bytes = malformed_len,
                "dropping malformed Kimi K3 call header"
            );
            self.buffer.drain(..malformed_len);
            return true;
        };
        let name = attr_value(&attrs, "tool").unwrap_or_default().to_string();
        let index = attr_value(&attrs, "index")
            .filter(|index| !index.is_empty())
            .map(str::to_string);
        let id = tool_call_id(&name, index.as_deref());
        self.active_call = Some(ActiveCall {
            name,
            id,
            return_mode,
        });
        self.call_boundary.begin(header_end + sep_len, self.mode);
        if self.stream_arguments && !self.active_call.as_ref().unwrap().name.is_empty() {
            let call = self.active_call.as_ref().unwrap();
            let tool_index = self.next_tool_index;
            self.next_tool_index += 1;
            self.call_ids.push(call.id.clone());
            output.push_call(ToolCallDelta {
                tool_index,
                name: Some(call.name.clone()),
                arguments: String::new(),
                complete: false,
            });
            self.streaming_call = Some(StreamingCall::new(tool_index, header_end + sep_len));
        }
        true
    }

    fn consume_tools_open(&mut self, return_mode: Mode) -> bool {
        let Some(len) = TOOLS_OPEN.prefix_len(&self.buffer) else {
            return false;
        };
        self.tools_open = self.buffer[..len].to_string();
        self.buffer.drain(..len);
        self.tools_return_mode = Some(return_mode);
        true
    }

    fn consume_assistant_message_open(&mut self) -> bool {
        let Some((attrs, header_len)) = parse_tag_header(&self.buffer, MESSAGE_OPEN) else {
            return false;
        };
        if attr_value(&attrs, "role") != Some("assistant") {
            return false;
        }
        self.buffer.drain(..header_len);
        true
    }

    fn consume_reasoning_structure(&mut self, flush: bool) -> bool {
        let Some((open, close)) = [(ARG_OPEN, ARG_CLOSE), (JSON_OPEN, JSON_CLOSE)]
            .into_iter()
            .find(|(open, _)| open.prefix_len(&self.buffer).is_some())
        else {
            return false;
        };
        let open_len = open.prefix_len(&self.buffer).expect("matched opener");
        let Some((sep_at, sep_len)) = SEP_MARKER.find(&self.buffer[open_len..]) else {
            if flush {
                self.buffer.clear();
                return true;
            }
            return false;
        };
        let value_start = open_len + sep_at + sep_len;
        let close_hit = if close.canonical == JSON_CLOSE.canonical {
            find_first_outside_strings(
                &self.buffer[value_start..],
                [close.canonical, close.spaced.expect("paired marker")],
            )
        } else {
            close.find(&self.buffer[value_start..])
        };
        let Some((close_at, close_len)) = close_hit else {
            if flush {
                self.buffer.clear();
                return true;
            }
            return false;
        };
        let consumed = value_start + close_at + close_len;
        self.buffer.drain(..consumed);
        tracing::warn!(
            why = "kimi_k3_structure_in_reasoning",
            skipped_bytes = consumed,
            "quarantining Kimi K3 argument/json structure inside reasoning"
        );
        true
    }

    fn consume_reasoning_close(&mut self, flush: bool) -> bool {
        let Some(head_len) = THINK_CLOSE_HEAD.prefix_len(&self.buffer) else {
            return false;
        };
        if let Some(sep_len) = SEP_MARKER.prefix_len(&self.buffer[head_len..]) {
            self.buffer.drain(..head_len + sep_len);
            return true;
        }
        let tail = &self.buffer[head_len..];
        if tail.is_empty() && flush
            || [OPEN, CLOSE, END_OF_MSG]
                .iter()
                .any(|marker| tail.starts_with(marker))
        {
            self.buffer.drain(..head_len);
            return true;
        }
        false
    }

    fn consume_message_end(&mut self) -> bool {
        self.consume(MESSAGE_CLOSE) || self.consume(Marker::single(END_OF_MSG))
    }

    fn consume(&mut self, marker: Marker) -> bool {
        let Some(len) = marker.prefix_len(&self.buffer) else {
            return false;
        };
        self.buffer.drain(..len);
        true
    }

    fn consume_any_at_start(&mut self, markers: &[Marker]) -> Option<usize> {
        let len = markers
            .iter()
            .filter_map(|marker| marker.prefix_len(&self.buffer))
            .min()?;
        self.buffer.drain(..len);
        Some(len)
    }

    fn emit_safe(
        &mut self,
        flush: bool,
        markers: &[Marker],
        output: &mut UnifiedParserOutput,
        emit: impl FnOnce(&mut UnifiedParserOutput, String),
    ) -> bool {
        let len = safe_len(&self.buffer, markers, flush);
        if len == 0 {
            return false;
        }
        let text = self.buffer.drain(..len).collect();
        emit(output, text);
        true
    }

    fn drop_safe(&mut self, flush: bool, markers: &[Marker]) -> bool {
        let len = safe_len(&self.buffer, markers, flush);
        if len == 0 {
            return false;
        }
        self.buffer.drain(..len);
        true
    }

    fn reset_state(&mut self) {
        self.streaming_call = None;
        self.streaming_error = None;
        self.mode = Mode::Idle;
        self.active_call = None;
        self.call_header_scan = None;
        self.call_boundary.reset();
        self.tools_open.clear();
        self.tools_return_mode = None;
        self.next_tool_index = 0;
        self.call_ids.clear();
    }
}

impl KimiK3Native {
    pub(super) fn apply_native_init(&mut self, starting_state: UnifiedParserStartingState) {
        self.buffer.clear();
        self.reset_state();
        self.restore_native_state(starting_state);
    }

    pub(super) fn restore_native_state(&mut self, starting_state: UnifiedParserStartingState) {
        self.mode = match starting_state {
            UnifiedParserStartingState::None => Mode::Idle,
            UnifiedParserStartingState::Reasoning => Mode::Reasoning,
            UnifiedParserStartingState::Response => Mode::Response,
        };
    }

    pub(super) fn push_native(
        &mut self,
        delta: &str,
        output: &mut UnifiedParserOutput,
    ) -> anyhow::Result<()> {
        self.buffer.push_str(delta);
        self.drain(false, output);
        if let Some(error) = self.streaming_error.take() {
            return Err(error);
        }
        Ok(())
    }

    pub(super) fn finish_native(&mut self, output: &mut UnifiedParserOutput) -> anyhow::Result<()> {
        self.drain(true, output);
        if let Some(error) = self.streaming_error.take() {
            return Err(error);
        }
        match self.mode {
            Mode::Idle | Mode::Response => output.push_text(std::mem::take(&mut self.buffer)),
            Mode::Reasoning => output.push_reasoning(std::mem::take(&mut self.buffer)),
            Mode::Tools | Mode::Call | Mode::Done => self.buffer.clear(),
        }
        self.mode = Mode::Idle;
        self.active_call = None;
        self.call_header_scan = None;
        self.call_boundary.reset();
        Ok(())
    }

    pub(super) fn reset_native(&mut self) -> String {
        let mut buffered = std::mem::take(&mut self.tools_open);
        buffered.push_str(&std::mem::take(&mut self.buffer));
        self.reset_state();
        buffered
    }
}

fn safe_len(text: &str, markers: &[Marker], flush: bool) -> usize {
    if let Some(position) = markers
        .iter()
        .filter_map(|marker| marker.find(text).map(|(at, _)| at))
        .min()
    {
        return position;
    }
    if flush {
        return text.len();
    }
    let holdback = markers
        .iter()
        .flat_map(|marker| marker.variants())
        .filter_map(|marker| {
            marker
                .char_indices()
                .map(|(at, _)| at)
                .filter(|at| *at > 0)
                .rev()
                .find(|at| text.ends_with(&marker[..*at]))
        })
        .max()
        .unwrap_or_default();
    text.len() - holdback
}

pub(super) fn parse_tag_header(text: &str, open: Marker) -> Option<(Vec<(String, String)>, usize)> {
    let open_len = open.prefix_len(text)?;
    let (sep_at, sep_len) = SEP_MARKER.find(&text[open_len..])?;
    let header_end = open_len + sep_at;
    Some((
        parse_attrs(&text[open_len..header_end])?,
        header_end + sep_len,
    ))
}

fn parse_call_header(text: &str) -> Option<(Vec<(String, String)>, usize)> {
    parse_tag_header(text, CALL_OPEN)
}

fn parse_call_body(body: &str) -> Option<String> {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return Some("{}".to_string());
    }

    if let Some(open_len) = JSON_OPEN.prefix_len(trimmed) {
        let (sep_at, sep_len) = SEP_MARKER.find(&trimmed[open_len..])?;
        let value_start = open_len + sep_at + sep_len;
        let (close_at, close_len) = find_first_outside_strings(
            &trimmed[value_start..],
            [
                JSON_CLOSE.canonical,
                JSON_CLOSE.spaced.expect("paired marker"),
            ],
        )?;
        let value_end = value_start + close_at;
        if !trimmed[value_end + close_len..].trim().is_empty() {
            return None;
        }
        let raw = &trimmed[value_start..value_end];
        let Value::Object(_) = serde_json::from_str::<Value>(raw).ok()? else {
            return None;
        };
        return Some(compact_json(raw));
    }

    let mut fields = Vec::<(String, String)>::new();
    let mut field_positions = HashMap::<String, usize>::new();
    let mut cursor = 0;
    while cursor < trimmed.len() {
        cursor += trimmed[cursor..].len() - trimmed[cursor..].trim_start().len();
        if cursor == trimmed.len() {
            break;
        }
        let open_len = ARG_OPEN.prefix_len(&trimmed[cursor..])?;
        let (sep_at, sep_len) = SEP_MARKER.find(&trimmed[cursor + open_len..])?;
        let header_end = cursor + open_len + sep_at;
        let attrs = parse_attrs(&trimmed[cursor + open_len..header_end])?;
        let value_start = header_end + sep_len;
        let key = attr_value(&attrs, "key").unwrap_or_default().to_string();
        let arg_type = attr_value(&attrs, "type").unwrap_or("string");
        let (close_at, close_len) = structural_argument_close(&trimmed[value_start..])?;
        let value_end = value_start + close_at;
        let field_end = value_end + close_len;
        let value = encode_argument_value(arg_type, &trimmed[value_start..value_end]);
        count_argument_field_lookup();
        if let Some(position) = field_positions.get(&key).copied() {
            fields[position].1 = value;
        } else {
            field_positions.insert(key.clone(), fields.len());
            fields.push((key, value));
        }
        cursor = field_end;
    }

    let mut output = String::from("{");
    for (position, (key, value)) in fields.iter().enumerate() {
        if position > 0 {
            output.push(',');
        }
        output.push_str(&serde_json::to_string(key).ok()?);
        output.push(':');
        output.push_str(value);
    }
    output.push('}');
    Some(output)
}

fn structural_argument_close(value_and_rest: &str) -> Option<(usize, usize)> {
    #[derive(Clone, Copy)]
    enum NextArgument {
        None,
        AfterClose(TokenHit),
        Header(TokenHit, HeaderPhase),
    }

    #[derive(Clone, Copy)]
    enum HeaderPhase {
        Whitespace,
        Key,
        Quote,
        Value,
    }

    let mut next_argument = NextArgument::None;
    let mut cursor = 0;
    while cursor < value_and_rest.len() {
        let suffix = &value_and_rest[cursor..];
        if let Some(close_len) = argument_close_len(suffix) {
            count_argument_scan(close_len);
            next_argument = NextArgument::AfterClose(TokenHit {
                at: cursor,
                len: close_len,
            });
            cursor += close_len;
            continue;
        }

        match next_argument {
            NextArgument::AfterClose(close) => {
                let ch = suffix.chars().next()?;
                if ch.is_whitespace() {
                    count_argument_scan(ch.len_utf8());
                    cursor += ch.len_utf8();
                    continue;
                }
                if let Some(open_len) = ARG_OPEN.prefix_len(suffix) {
                    count_argument_scan(open_len);
                    cursor += open_len;
                    next_argument = NextArgument::Header(close, HeaderPhase::Whitespace);
                    continue;
                }
                next_argument = NextArgument::None;
            }
            NextArgument::Header(close, phase) => {
                if let Some(sep_len) = SEP_MARKER.prefix_len(suffix) {
                    count_argument_scan(sep_len);
                    if matches!(phase, HeaderPhase::Whitespace) {
                        return Some((close.at, close.len));
                    }
                    next_argument = NextArgument::None;
                    cursor += sep_len;
                    continue;
                }

                let ch = suffix.chars().next()?;
                let next_phase = match phase {
                    HeaderPhase::Whitespace if ch.is_whitespace() => HeaderPhase::Whitespace,
                    HeaderPhase::Whitespace if ch.is_alphanumeric() || ch == '_' => {
                        HeaderPhase::Key
                    }
                    HeaderPhase::Key if ch.is_alphanumeric() || ch == '_' => HeaderPhase::Key,
                    HeaderPhase::Key if ch == '=' => HeaderPhase::Quote,
                    HeaderPhase::Quote if ch == '"' => HeaderPhase::Value,
                    HeaderPhase::Value if ch == '"' => HeaderPhase::Whitespace,
                    HeaderPhase::Value => HeaderPhase::Value,
                    HeaderPhase::Whitespace | HeaderPhase::Key | HeaderPhase::Quote => {
                        next_argument = NextArgument::None;
                        continue;
                    }
                };
                count_argument_scan(ch.len_utf8());
                cursor += ch.len_utf8();
                next_argument = NextArgument::Header(close, next_phase);
                continue;
            }
            NextArgument::None => {}
        }

        let ch = suffix.chars().next()?;
        count_argument_scan(ch.len_utf8());
        cursor += ch.len_utf8();
    }
    match next_argument {
        NextArgument::AfterClose(close) => Some((close.at, close.len)),
        NextArgument::None | NextArgument::Header(_, _) => None,
    }
}

fn argument_close_len(text: &str) -> Option<usize> {
    ARG_CLOSE
        .variants()
        .find(|marker| marker_matches(text, marker))
        .map(str::len)
}

fn marker_matches(text: &str, marker: &str) -> bool {
    if text.len() < marker.len() {
        return false;
    }
    text.as_bytes()
        .iter()
        .zip(marker.as_bytes())
        .take(marker.len())
        .all(|(actual, expected)| {
            count_argument_comparison();
            actual == expected
        })
}

#[cfg(test)]
std::thread_local! {
    static ARGUMENT_SCAN_BYTES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static ARGUMENT_MARKER_COMPARISONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static ARGUMENT_FIELD_LOOKUPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn count_argument_scan(bytes: usize) {
    #[cfg(test)]
    ARGUMENT_SCAN_BYTES.with(|scanned| scanned.set(scanned.get() + bytes));
    #[cfg(not(test))]
    let _ = bytes;
}

fn count_argument_comparison() {
    #[cfg(test)]
    ARGUMENT_MARKER_COMPARISONS.with(|comparisons| comparisons.set(comparisons.get() + 1));
}

fn count_argument_field_lookup() {
    #[cfg(test)]
    ARGUMENT_FIELD_LOOKUPS.with(|lookups| lookups.set(lookups.get() + 1));
}

#[cfg(test)]
fn reset_argument_work() {
    ARGUMENT_SCAN_BYTES.with(|scanned| scanned.set(0));
    ARGUMENT_MARKER_COMPARISONS.with(|comparisons| comparisons.set(0));
    ARGUMENT_FIELD_LOOKUPS.with(|lookups| lookups.set(0));
}

#[cfg(test)]
fn argument_work() -> (usize, usize, usize) {
    (
        ARGUMENT_SCAN_BYTES.with(std::cell::Cell::get),
        ARGUMENT_MARKER_COMPARISONS.with(std::cell::Cell::get),
        ARGUMENT_FIELD_LOOKUPS.with(std::cell::Cell::get),
    )
}

fn encode_argument_value(arg_type: &str, raw: &str) -> String {
    if arg_type == "string" {
        return serde_json::to_string(raw).expect("serializing a Rust string cannot fail");
    }
    if serde_json::from_str::<Value>(raw).is_ok() {
        compact_json(raw)
    } else {
        serde_json::to_string(raw).expect("serializing a Rust string cannot fail")
    }
}

fn compact_json(raw: &str) -> String {
    let mut output = String::with_capacity(raw.len());
    let mut in_string = false;
    let mut escaped = false;
    for character in raw.chars() {
        if in_string {
            output.push(character);
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                in_string = false;
            }
        } else if character == '"' {
            in_string = true;
            output.push(character);
        } else if !character.is_whitespace() {
            output.push(character);
        }
    }
    output
}

fn parse_attrs(input: &str) -> Option<Vec<(String, String)>> {
    parse_attrs_prefix(input).map(|(attrs, _)| attrs)
}

fn parse_attrs_prefix(input: &str) -> Option<(Vec<(String, String)>, usize)> {
    let mut attrs = Vec::new();
    let mut cursor = 0;
    while cursor < input.len() {
        cursor += input[cursor..].len() - input[cursor..].trim_start().len();
        if cursor == input.len() {
            break;
        }
        let rest = &input[cursor..];
        let key_len = rest
            .char_indices()
            .take_while(|(_, character)| character.is_alphanumeric() || *character == '_')
            .map(|(at, character)| at + character.len_utf8())
            .last()?;
        let key = &rest[..key_len];
        let value = rest[key_len..].strip_prefix("=\"")?;
        let end = value.find('"')?;
        attrs.push((key.to_string(), unescape_attr(&value[..end])));
        cursor += key_len + 2 + end + 1;
    }
    Some((attrs, cursor))
}

fn unescape_attr(value: &str) -> String {
    value.replace("&quot;", "\"").replace("&amp;", "&")
}

pub(super) fn attr_value<'a>(attrs: &'a [(String, String)], key: &str) -> Option<&'a str> {
    attrs
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value.as_str())
}

fn tool_call_id(name: &str, index: Option<&str>) -> String {
    match index.filter(|index| !index.is_empty()) {
        None => name.to_string(),
        Some(index) => index.parse::<i64>().map_or_else(
            |_| format!("{name}:{index}"),
            |index| match index.checked_sub(1) {
                Some(normalized) if index > 0 => format!("{name}:{normalized}"),
                _ => format!("{name}:{index}"),
            },
        ),
    }
}

fn find_first_outside_strings<'a, I>(text: &str, markers: I) -> Option<(usize, usize)>
where
    I: Clone + IntoIterator<Item = &'a str>,
{
    let mut in_string = false;
    let mut escape = false;
    for (idx, c) in text.char_indices() {
        if in_string {
            if escape {
                escape = false;
            } else if c == '\\' {
                escape = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        if c == '"' {
            in_string = true;
            continue;
        }
        if let Some((_, len)) = markers
            .clone()
            .into_iter()
            .find(|m| text[idx..].starts_with(m))
            .map(|m| (idx, m.len()))
        {
            return Some((idx, len));
        }
    }
    None
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
