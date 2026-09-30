// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Early JSON argument projection over the local K3 native boundary owner's buffer.

use crate::{UnifiedParserOutput, upstream::ToolCallDelta};

use super::native::{
    ARG_CLOSE, ARG_OPEN, CALL_CLOSE, END_OF_MSG, JSON_CLOSE, JSON_OPEN, MESSAGE_CLOSE, TOOLS_CLOSE,
    attr_value, parse_tag_header,
};

/// Incremental output cursor over the native parser's call buffer. The existing
/// boundary owner still validates and closes the call; this cursor only projects
/// bytes that cannot be changed by later chunks.
pub(super) struct StreamingCall {
    pub(super) tool_index: usize,
    cursor: usize,
    pub(super) emitted: String,
    state: StreamingValue,
    fields: std::collections::HashSet<String>,
    json_in_string: bool,
    json_escaped: bool,
}

#[derive(Clone, Copy)]
enum StreamingValue {
    Field,
    String,
    Json,
    RawJson,
}

impl StreamingCall {
    pub(super) fn new(tool_index: usize, cursor: usize) -> Self {
        Self {
            tool_index,
            cursor,
            emitted: String::new(),
            state: StreamingValue::Field,
            fields: std::collections::HashSet::new(),
            json_in_string: false,
            json_escaped: false,
        }
    }

    fn emit(&mut self, text: String, output: &mut UnifiedParserOutput) {
        if text.is_empty() {
            return;
        }
        self.emitted.push_str(&text);
        output.push_call(ToolCallDelta {
            tool_index: self.tool_index,
            name: None,
            arguments: text,
            complete: false,
        });
    }

    pub(super) fn advance(
        &mut self,
        buffer: &str,
        output: &mut UnifiedParserOutput,
    ) -> anyhow::Result<()> {
        loop {
            if self.cursor == buffer.len() {
                return Ok(());
            }
            if matches!(self.state, StreamingValue::Field) {
                let rest = &buffer[self.cursor..];
                self.cursor += rest.len() - rest.trim_start().len();
                let rest = &buffer[self.cursor..];
                if let Some((_, len)) = parse_tag_header(rest, JSON_OPEN) {
                    self.cursor += len;
                    self.state = StreamingValue::RawJson;
                    continue;
                }
                let Some((attrs, len)) = parse_tag_header(rest, ARG_OPEN) else {
                    return Ok(());
                };
                let key = attr_value(&attrs, "key").unwrap_or_default();
                anyhow::ensure!(
                    self.fields.insert(key.to_string()),
                    "duplicate streaming Kimi K3 argument key"
                );
                let string = attr_value(&attrs, "type").unwrap_or("string") == "string";
                let prefix = if self.emitted.is_empty() { "{" } else { "," };
                let key = serde_json::to_string(key)?;
                self.emit(
                    format!("{prefix}{key}:{}", if string { "\"" } else { "" }),
                    output,
                );
                self.cursor += len;
                self.state = if string {
                    StreamingValue::String
                } else {
                    StreamingValue::Json
                };
                continue;
            }

            let raw_json = matches!(self.state, StreamingValue::RawJson);
            let string = matches!(self.state, StreamingValue::String);
            let close = if raw_json { JSON_CLOSE } else { ARG_CLOSE };
            let start = self.cursor;
            let mut projected = String::new();
            let mut closed = false;
            while self.cursor < buffer.len() {
                let rest = &buffer[self.cursor..];
                if string || !self.json_in_string {
                    if let Some(len) = close.prefix_len(rest) {
                        if raw_json {
                            break; // The native boundary owner will validate the object.
                        }
                        let tail = rest[len..].trim_start();
                        // A close is structural only when followed by another complete
                        // argument header. At call closure the authoritative owner
                        // supplies the remaining validated JSON instead.
                        if parse_tag_header(tail, ARG_OPEN).is_some() {
                            self.cursor += len;
                            closed = true;
                            break;
                        }
                        if tail.is_empty()
                            || ARG_OPEN.variants().any(|marker| marker.starts_with(tail))
                            || ARG_OPEN.prefix_len(tail).is_some()
                            || CALL_CLOSE
                                .variants()
                                .any(|marker| marker.starts_with(tail) || tail.starts_with(marker))
                            || TOOLS_CLOSE.prefix_len(tail).is_some()
                            || MESSAGE_CLOSE.prefix_len(tail).is_some()
                            || tail.starts_with(END_OF_MSG)
                        {
                            break;
                        }
                        // A marker followed by ordinary data is literal string data.
                    } else if close.variants().any(|marker| marker.starts_with(rest)) {
                        break;
                    }
                }
                let ch = rest.chars().next().expect("non-empty value suffix");
                self.cursor += ch.len_utf8();
                if !string {
                    if self.json_in_string {
                        projected.push(ch);
                        if self.json_escaped {
                            self.json_escaped = false;
                        } else if ch == '\\' {
                            self.json_escaped = true;
                        } else if ch == '"' {
                            self.json_in_string = false;
                        }
                    } else if ch == '"' {
                        self.json_in_string = true;
                        projected.push(ch);
                    } else if !ch.is_whitespace() {
                        projected.push(ch);
                    }
                }
            }
            if string {
                let end = if closed {
                    // The delimiter follows the literal value bytes we just visited.
                    let rest = &buffer[start..self.cursor];
                    close
                        .variants()
                        .find_map(|marker| rest.strip_suffix(marker))
                        .expect("closed argument has a closing marker")
                } else {
                    &buffer[start..self.cursor]
                };
                let encoded = serde_json::to_string(end)?;
                projected.push_str(&encoded[1..encoded.len() - 1]);
                if closed {
                    projected.push('"');
                }
            }
            self.emit(projected, output);
            if closed {
                self.state = StreamingValue::Field;
                self.json_in_string = false;
                self.json_escaped = false;
                continue;
            }
            return Ok(());
        }
    }
}
