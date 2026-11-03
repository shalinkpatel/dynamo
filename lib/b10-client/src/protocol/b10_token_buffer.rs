// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{BidRequestV1, TokensV1};
use std::sync::OnceLock;

pub(super) fn write_legacy_tokens() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| dynamo_runtime::config::env_is_truthy("DYN_COORDINATOR_DUAL_WRITE_TOKENS"))
}

impl BidRequestV1 {
    /// Decode IDs for the local router.
    pub(crate) fn into_routing_tokens(mut self) -> anyhow::Result<Self> {
        self.tokens = dynamo_tokens::b10_token_buffer::decode_owned(
            self.tokens,
            self.token_buffer
                .as_ref()
                .map(|buffer| buffer.payload.as_ref()),
        )?;
        self.token_buffer = None;
        self.validate()?;
        Ok(self)
    }
    /// Outgoing bids use bytes, optionally including legacy IDs for older readers.
    pub fn into_token_buffer(self) -> anyhow::Result<Self> {
        self.into_token_buffer_with_legacy(write_legacy_tokens())
    }

    fn into_token_buffer_with_legacy(mut self, include_legacy: bool) -> anyhow::Result<Self> {
        self.validate()?;
        if include_legacy && let Some(buffer) = &self.token_buffer {
            self.tokens = dynamo_tokens::b10_token_buffer::decode_owned(
                self.tokens,
                Some(buffer.payload.as_ref()),
            )?;
        }
        if self.token_buffer.is_none() {
            self.token_buffer = Some(TokensV1 {
                payload: dynamo_tokens::b10_token_buffer::encode(&self.tokens).into(),
            });
        }
        if !include_legacy {
            self.tokens.clear();
        }
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn b10_bid_dual_write_uses_canonical_buffer_ids() {
        let ids = vec![0, 128, 151_643, u32::MAX];
        let buffer = TokensV1 {
            payload: dynamo_tokens::b10_token_buffer::encode(&ids).into(),
        };
        for include_legacy in [false, true] {
            for token_buffer in [None, Some(buffer.clone())] {
                let bid = BidRequestV1 {
                    tokens: if token_buffer.is_some() {
                        vec![99]
                    } else {
                        ids.clone()
                    },
                    token_buffer,
                    ..Default::default()
                }
                .into_token_buffer_with_legacy(include_legacy)
                .unwrap();
                assert_eq!(bid.token_buffer.unwrap().payload, buffer.payload);
                assert_eq!(
                    bid.tokens,
                    if include_legacy { ids.clone() } else { vec![] }
                );
            }
        }
    }
}
