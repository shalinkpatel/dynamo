// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Baseten-owned K3 native streaming, using upstream's public event contract.
//! Guided JSON is delegated to the normal upstream K3 parser.

mod native;
mod streaming;

use anyhow::{Result, ensure};

use crate::{
    Tool, UnifiedParser, UnifiedParserInit, UnifiedParserOutput, UnifiedParserStartingState,
    UnifiedToolOutputMode, upstream,
};
use native::KimiK3Native;

pub(crate) struct KimiK3Parser {
    native: KimiK3Native,
    tools: Vec<Tool>,
    guided: Option<Box<dyn UnifiedParser>>,
    starting_state: UnifiedParserStartingState,
    started: bool,
    finished: bool,
}

impl KimiK3Parser {
    pub fn new(tools: &[Tool]) -> Self {
        Self::new_with_streaming(tools, true)
    }

    fn new_with_streaming(tools: &[Tool], streaming: bool) -> Self {
        let mut native = KimiK3Native::new();
        native.stream_arguments = streaming;
        Self {
            native,
            tools: tools.to_vec(),
            guided: None,
            starting_state: UnifiedParserStartingState::None,
            started: false,
            finished: false,
        }
    }
}

impl UnifiedParser for KimiK3Parser {
    fn initialize_request(&mut self, init: UnifiedParserInit) -> Result<()> {
        ensure!(
            !self.started && !self.finished,
            "cannot initialize a unified parser after parsing has started"
        );
        let starting_state = init.starting_state;
        // Build and validate guided initialization before changing native state.
        let guided = if matches!(
            init.tool_output_mode,
            UnifiedToolOutputMode::GuidedJson { .. }
        ) {
            let mut parser = upstream::create_unified_parser_for_family("kimi_k3", &self.tools)?;
            parser.initialize_request(init)?;
            Some(parser)
        } else {
            None
        };
        self.native.apply_native_init(starting_state);
        self.guided = guided;
        self.starting_state = starting_state;
        Ok(())
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn tool_call_id(&self, tool_index: usize) -> Option<&str> {
        match &self.guided {
            Some(parser) => parser.tool_call_id(tool_index),
            None => self.native.call_ids.get(tool_index).map(String::as_str),
        }
    }

    fn parse_into(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> Result<()> {
        ensure!(!self.finished, "cannot push to a finished unified parser");
        self.started = true;
        let result = match &mut self.guided {
            Some(parser) => parser.parse_into(delta, output),
            None => self.native.push_native(delta, output),
        };
        if result.is_err() {
            self.finished = true;
        }
        result
    }

    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        ensure!(!self.finished, "cannot finish a unified parser twice");
        self.started = true;
        self.finished = true;
        match &mut self.guided {
            Some(parser) => parser.finish(),
            None => {
                let mut output = UnifiedParserOutput::default();
                self.native.finish_native(&mut output)?;
                Ok(output)
            }
        }
    }

    fn reset(&mut self) -> String {
        let buffered = match &mut self.guided {
            Some(parser) => parser.reset(),
            None => self.native.reset_native(),
        };
        self.native.restore_native_state(self.starting_state);
        self.started = false;
        self.finished = false;
        buffered
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::upstream::UnifiedParserEvent;

    #[test]
    fn lifecycle_preserves_prefilled_state_on_rejected_init_and_reset() {
        for starting_state in [
            UnifiedParserStartingState::Reasoning,
            UnifiedParserStartingState::Response,
        ] {
            let mut parser = KimiK3Parser::new(&[]);
            parser
                .initialize_request(UnifiedParserInit {
                    starting_state,
                    ..Default::default()
                })
                .unwrap();
            let mut output = UnifiedParserOutput::default();
            parser.parse_into("first", &mut output).unwrap();
            assert!(
                parser
                    .initialize_request(UnifiedParserInit::default())
                    .is_err()
            );
            parser.parse_into("second", &mut output).unwrap();
            let expected = if starting_state == UnifiedParserStartingState::Reasoning {
                UnifiedParserEvent::Reasoning("firstsecond".into())
            } else {
                UnifiedParserEvent::Text("firstsecond".into())
            };
            assert_eq!(output.events, vec![expected]);
            parser.finish().unwrap();
            assert!(parser.parse_into("late", &mut output).is_err());
            assert!(parser.finish().is_err());
            parser.reset();
            output.events.clear();
            parser.parse_into("fresh", &mut output).unwrap();
            let expected = if starting_state == UnifiedParserStartingState::Reasoning {
                UnifiedParserEvent::Reasoning("fresh".into())
            } else {
                UnifiedParserEvent::Text("fresh".into())
            };
            assert_eq!(output.events, vec![expected]);
        }
    }

    #[test]
    fn native_error_is_terminal_until_reset_and_keeps_committed_deltas() {
        let mut parser = KimiK3Parser::new(&[]);
        let mut output = UnifiedParserOutput::default();
        parser
            .parse_into(
                concat!(
                    "<|open|>tools<|sep|><|open|>call tool=\"bash\" index=\"1\"<|sep|>",
                    "<|open|>argument key=\"command\" type=\"string\"<|sep|>echo partial"
                ),
                &mut output,
            )
            .unwrap();
        assert!(output.events.iter().any(|event| matches!(event,
            UnifiedParserEvent::ToolCall(call) if call.arguments.contains("echo partial"))));
        assert!(parser.finish().is_err());
        assert!(parser.parse_into("late", &mut output).is_err());
        parser.reset();
        output.events.clear();
        parser
            .parse_into(
                concat!(
                    "<|open|>tools<|sep|><|open|>call tool=\"bash\" index=\"2\"<|sep|>",
                    "<|open|>argument key=\"command\" type=\"string\"<|sep|>fresh",
                    "<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|>"
                ),
                &mut output,
            )
            .unwrap();
        assert_eq!(parser.tool_call_id(0), Some("bash:1"));
        assert_eq!(parser.tool_call_id(1), None);
        assert!(output.events.iter().any(|event| matches!(event,
            UnifiedParserEvent::ToolCall(call) if call.tool_index == 0 && call.complete)));
        parser.finish().unwrap();
    }
}
