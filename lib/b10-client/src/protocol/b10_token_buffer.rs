// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{BidRequestV1, TokensV1};

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
    /// All outgoing bids use bytes; retain an already encoded payload unchanged.
    pub fn into_token_buffer(mut self) -> anyhow::Result<Self> {
        self.validate()?;
        if self.token_buffer.is_none() {
            self.token_buffer = Some(TokensV1 {
                payload: dynamo_tokens::b10_token_buffer::encode(&self.tokens).into(),
            });
        }
        self.tokens.clear();
        Ok(self)
    }
}
