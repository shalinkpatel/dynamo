// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use reqwest::header::HeaderValue;
use rmpv::Value;

pub(super) const HEADER: &str = "x-baseten-dynamo-served-model-name";

/// Expose the worker's model without decoding its opaque MessagePack on the receiver.
/// This optional attribution header must not prevent generation for an unusable name.
pub(super) fn from_worker_request(request: &Value) -> Option<HeaderValue> {
    let Value::Map(entries) = request else {
        return None;
    };
    let name = entries
        .iter()
        .find(|(key, _)| key.as_str() == Some("model"))?
        .1
        .as_str()
        .filter(|name| !name.is_empty())?;
    HeaderValue::from_str(name).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn b10_optional_served_model_name() {
        for model in [
            Value::Nil,
            Value::from(""),
            Value::from("bad\nname"),
            Value::from(42),
        ] {
            assert!(from_worker_request(&Value::Map(vec![("model".into(), model)])).is_none());
        }
        assert!(from_worker_request(&Value::Map(vec![])).is_none());
        assert_eq!(
            from_worker_request(&Value::Map(vec![("model".into(), "org/model".into())])).unwrap(),
            "org/model"
        );
    }
}
