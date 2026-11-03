// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::protocol::TokenUsageV1;
use rmpv::Value;

/// A decode-side denial must retain counters already reported by prefill.
pub(crate) fn prefill_usage(response: &Value) -> TokenUsageV1 {
    let mut tracker = TokenUsageTracker::new(true);
    tracker.observe(Some(response));
    tracker.snapshot()
}

/// Request-level snapshots without waiting for worker counters or changing chunks.
#[derive(Default)]
pub(crate) struct TokenUsageTracker {
    // Worker-reported counters; visible output deltas are tracked separately.
    usage: TokenUsageV1,
    prefill_first: bool,
    visible_output_tokens: Option<u64>,
}

impl TokenUsageTracker {
    pub(crate) fn new(prefill_first: bool) -> Self {
        Self {
            prefill_first,
            ..Self::default()
        }
    }

    pub(crate) fn snapshot(&self) -> TokenUsageV1 {
        TokenUsageV1 {
            output_tokens: self.visible_output_tokens.max(self.usage.output_tokens),
            ..self.usage
        }
    }

    pub(crate) fn observe(&mut self, data: Option<&Value>) {
        let Some(data) = data.filter(|data| data.is_map()) else {
            return;
        };
        let completion = &data["completion_usage"];
        let input = data["input_tokens"]
            .as_u64()
            .or_else(|| data["prompt_tokens"].as_u64())
            .or_else(|| completion["prompt_tokens"].as_u64());
        let cached = data["cached_tokens"]
            .as_u64()
            .or_else(|| completion["prompt_tokens_details"]["cached_tokens"].as_u64());
        // Prefill counts win, but a usage-free prefill handoff (SGLang disagg
        // bootstrap) must not hide the decode leg's counts.
        if !self.prefill_first || self.usage.input_tokens.is_none() {
            self.usage.input_tokens = input.or(self.usage.input_tokens);
        }
        if !self.prefill_first || self.usage.cached_tokens.is_none() {
            self.usage.cached_tokens = cached.or(self.usage.cached_tokens);
        }

        // Count visible token deltas across both legs, including a prefill token.
        // Per-leg cumulative worker counts cannot be safely summed in disagg.
        if let Some(delta) = output_delta(data) {
            self.visible_output_tokens = Some(
                self.visible_output_tokens
                    .unwrap_or(0)
                    .saturating_add(delta),
            );
        }
        if !self.prefill_first
            && let Some(count) = data["output_tokens"]
                .as_u64()
                .or_else(|| data["completion_tokens"].as_u64())
                .or_else(|| completion["completion_tokens"].as_u64())
        {
            // A worker total can include other choices whose deltas have not
            // arrived yet. Never add subsequent deltas to that reported total.
            self.usage.output_tokens = self.usage.output_tokens.max(Some(count));
        }
    }
}

fn output_delta(data: &Value) -> Option<u64> {
    if let Some(outputs) = data["outputs"].as_array() {
        if outputs.is_empty() {
            return None;
        }
        outputs.iter().try_fold(0_u64, |total, output| {
            let tokens = output["token_ids_diff"].as_array()?;
            Some(total.saturating_add(tokens.len() as u64))
        })
    } else {
        // Dynamo's LLMEngineOutput uses a top-level delta instead.
        data["token_ids"]
            .as_array()
            .map(|tokens| tokens.len() as u64)
    }
}

#[cfg(test)]
#[path = "b10_token_usage_tests.rs"]
mod tests;
