// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `GET /monitors`: the monitor names in the monitoring TOML at `DYN_MONITOR_CONFIG`.

use std::collections::BTreeMap;

use anyhow::{Context as _, Result, ensure};
use axum::{Json, Router, http::Method, routing::get};
use dynamo_runtime::config::environment_names::llm::monitor::DYN_MONITOR_CONFIG;
use serde::Deserialize;
use serde_json::{Value, json};

use super::RouteDoc;

/// Lenient view of the file: only the fields this endpoint serves are checked.
#[derive(Deserialize)]
struct File {
    monitoring: Monitoring,
}

#[derive(Deserialize)]
struct Monitoring {
    version: i64,
    #[serde(default)]
    monitors: BTreeMap<String, toml::Value>,
}

/// The `GET /monitors` document for the TOML at `path`; names are sorted.
fn document(path: &str) -> Result<Value> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?;
    let monitoring = toml::from_str::<File>(&text)?.monitoring;
    ensure!(
        monitoring.version == 1,
        "unsupported monitoring version {}",
        monitoring.version
    );
    let names: Vec<&String> = monitoring.monitors.keys().collect();
    Ok(json!({"version": 1, "monitors": names, "capture": {"enabled": false}}))
}

/// Registered only when `DYN_MONITOR_CONFIG` is set; a malformed file fails startup.
pub fn monitors_router() -> Result<(Vec<RouteDoc>, Router)> {
    let Some(path) = std::env::var(DYN_MONITOR_CONFIG)
        .ok()
        .filter(|p| !p.is_empty())
    else {
        return Ok((vec![], Router::new()));
    };
    let doc = document(&path).with_context(|| format!("invalid {DYN_MONITOR_CONFIG}"))?;
    Ok((
        vec![RouteDoc::new(Method::GET, "/monitors")],
        Router::new().route("/monitors", get(move || async move { Json(doc) })),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    #[test]
    fn serves_sorted_names_and_rejects_malformed() {
        let fixture = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/monitoring.toml"
        );
        assert_eq!(
            document(fixture).unwrap(),
            json!({"version": 1, "monitors": ["harm", "sustained"], "capture": {"enabled": false}})
        );
        let mut bad = tempfile::NamedTempFile::new().unwrap();
        write!(bad, "[monitoring]\nversion = \"one\"\n").unwrap();
        let bad = bad.path().to_str().unwrap();
        temp_env::with_var(DYN_MONITOR_CONFIG, Some(bad), || {
            assert!(monitors_router().is_err())
        });
        temp_env::with_var(DYN_MONITOR_CONFIG, Some(fixture), || {
            assert_eq!(monitors_router().unwrap().0.len(), 1)
        });
        temp_env::with_var_unset(DYN_MONITOR_CONFIG, || {
            assert!(monitors_router().unwrap().0.is_empty())
        });
    }
}
