// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Capture HTTP headers in one JSON metadata entry for native remote forwarding.

use axum::http::{HeaderMap, HeaderName, HeaderValue, header};
use std::collections::BTreeMap;

pub const FORWARDED_HEADERS_METADATA_KEY: &str = "baseten.forwarded_headers";
type Headers = Vec<(String, Vec<u8>)>;
const EXCLUDED_HEADERS: &[&str] = &["host", "accept", "content-type"];

pub fn capture(metadata: &mut BTreeMap<String, String>, headers: &HeaderMap) {
    let entries: Vec<_> = headers
        .iter()
        .filter(|(name, _)| !EXCLUDED_HEADERS.contains(&name.as_str()))
        .map(|(name, value)| (name.as_str(), value.as_bytes()))
        .collect();
    metadata.insert(
        FORWARDED_HEADERS_METADATA_KEY.into(),
        serde_json::to_string(&entries).expect("header pairs serialize to JSON"),
    );
}

/// Consume the envelope so it is not sent in the coordinator protobuf metadata.
pub fn take(metadata: &mut BTreeMap<String, String>) -> anyhow::Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    if let Some(encoded) = metadata.remove(FORWARDED_HEADERS_METADATA_KEY) {
        let entries: Headers = serde_json::from_str(&encoded)?;
        for (name, bytes) in entries {
            let name: HeaderName = name.parse()?;
            let mut value = HeaderValue::from_bytes(&bytes)?;
            value.set_sensitive(name == header::AUTHORIZATION);
            headers.append(name, value);
        }
    }
    // The outgoing protobuf body has its own framing; connection-specific fields
    // describe the incoming HTTP connection rather than the coordinator request.
    let connection_headers: Vec<_> = headers
        .get_all(header::CONNECTION)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .map(|name| name.trim().to_owned())
        .collect();
    for name in connection_headers {
        headers.remove(name);
    }
    for name in [
        "accept-encoding",
        "content-encoding",
        "content-length",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "expect",
    ]
    .into_iter()
    .chain(EXCLUDED_HEADERS.iter().copied())
    {
        headers.remove(name);
    }
    Ok(headers)
}
