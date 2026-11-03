// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

impl crate::pb::BidRequest {
    /// The canonical token sequence, shared by placement lookup and booking.
    pub fn token_ids(
        &self,
    ) -> Result<std::borrow::Cow<'_, [u32]>, dynamo_tokens::b10_token_buffer::TokenBufferError>
    {
        dynamo_tokens::b10_token_buffer::decode(
            &self.tokens,
            self.token_buffer
                .as_ref()
                .map(|buffer| buffer.payload.as_ref()),
        )
    }

    /// Token count for tracing, without allocating or changing attempt replay semantics.
    pub fn token_count(&self) -> usize {
        self.token_buffer
            .as_ref()
            .map_or(self.tokens.len(), |buffer| buffer.payload.len() / 4)
    }
}
