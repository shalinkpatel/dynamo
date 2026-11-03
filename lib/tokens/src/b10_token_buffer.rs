// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared reader-first validation for the HTTP and gRPC token wire contracts.
use std::borrow::Cow;

/// An absent or malformed token representation.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("{0}")]
pub struct TokenBufferError(&'static str);

fn ids(payload: &[u8]) -> impl Iterator<Item = u32> + '_ {
    payload
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// Prefer the byte payload; read legacy IDs only while old writers drain.
/// A present malformed buffer never falls back to the legacy field.
pub fn validate(legacy: &[u32], payload: Option<&[u8]>) -> Result<(), TokenBufferError> {
    match payload {
        None if legacy.is_empty() => Err(TokenBufferError("tokens or token_buffer is required")),
        None => Ok(()),
        Some(bytes) if bytes.is_empty() || !bytes.len().is_multiple_of(4) => Err(TokenBufferError(
            "token_buffer.payload must contain nonempty u32 little-endian values",
        )),
        Some(_) => Ok(()),
    }
}

/// Borrow legacy IDs, or materialize the byte payload once for routing.
pub fn decode<'a>(
    legacy: &'a [u32],
    payload: Option<&[u8]>,
) -> Result<Cow<'a, [u32]>, TokenBufferError> {
    validate(legacy, payload)?;
    Ok(match payload {
        Some(bytes) => Cow::Owned(ids(bytes).collect()),
        None => Cow::Borrowed(legacy),
    })
}

/// Reuse the legacy allocation when reading an old writer.
pub fn decode_owned(
    legacy: Vec<u32>,
    payload: Option<&[u8]>,
) -> Result<Vec<u32>, TokenBufferError> {
    if payload.is_some() {
        Ok(decode(&legacy, payload)?.into_owned())
    } else {
        validate(&legacy, None)?;
        Ok(legacy)
    }
}

/// Encode token IDs as fixed-width little-endian values.
pub fn encode(tokens: &[u32]) -> Vec<u8> {
    tokens.iter().flat_map(|id| id.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn b10_byte_payload_round_trip() {
        let tokens = [0_u32, 127, 128, 16_384, 151_643, u32::MAX];
        assert_eq!(
            decode(&[], Some(&encode(&tokens))).unwrap().as_ref(),
            tokens
        );
    }

    #[test]
    fn b10_rejects_missing_or_malformed_payloads() {
        assert!(validate(&[], None).is_err());
        for bytes in [&[][..], &[1, 0, 0][..], &[1, 0, 0, 0, 0][..]] {
            assert!(validate(&[], Some(bytes)).is_err());
        }
    }
}
