// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker lame-duck drain.
//!
//! `POST /engine/drain {"until": "<RFC 3339>"}` on the worker's system port marks the worker as
//! draining: it stays registered and keeps serving the sessions bound to it, while routers with
//! `router_honor_draining` stop giving it new work. The deadline is published as a
//! `b10.drain/until=<RFC 3339>` taint on the worker's base model cards (through upstream's
//! `Discovery::update_model_taints`), so routers and the planner read it from discovery. A worker is draining
//! exactly while a deadline is set; an `until` at or before now (how llm-operator cancels), or
//! `null`, clears it. The generic `/engine/{*path}` handler does not pass the HTTP method, so an
//! empty body (`{}`) reports the current deadline instead of setting one. SIGTERM handling is
//! unchanged.
//!
//! Trust boundary: the system server has no authentication, so this route is open to anything
//! that can reach the pod's system port, exactly like the existing `/engine/sleep`,
//! `/engine/wake_up` and `/v1/loras` routes on the same server. Baseten does not publish that
//! port through a Service; llm-operator calls the pod IP from inside the cluster. A stray
//! drain never removes a worker and degrades to today's routing when every worker is draining.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use anyhow::{Context as _, Result};
use chrono::{DateTime, Utc};
use dynamo_kv_router::scheduling::b10_draining::DRAIN_TAINT_PREFIX;
use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::discovery::{
    Discovery, DiscoveryInstanceId, DiscoveryQuery, ModelCardInstanceId,
};
use dynamo_runtime::engine_routes::EngineRouteCallback;
use dynamo_runtime::metrics::MetricsHierarchy;
use prometheus::IntGauge;

use crate::local_model::runtime_config::TOPOLOGY_TAINT_PREFIX;
use crate::model_card::ModelDeploymentCard;

/// Engine route name, served at `/engine/drain` on the system port.
pub const DRAIN_ROUTE: &str = "drain";

/// Worker system metric: the accepted deadline as unix seconds, 0 while not draining.
pub const DRAINING_UNTIL_GAUGE: &str = "draining_until_unix_seconds";

/// The deadline this worker last accepted; what `GET /engine/drain` reports and what the
/// `draining_until_unix_seconds` gauge mirrors.
struct DrainState {
    until: Mutex<Option<DateTime<Utc>>>,
    /// `None` when the gauge could not be registered; the drain still works without it.
    gauge: Option<IntGauge>,
}

impl DrainState {
    fn current(&self) -> Option<DateTime<Utc>> {
        *self.until.lock().unwrap()
    }

    /// Returns the previous deadline.
    fn set(&self, until: Option<DateTime<Utc>>) -> Option<DateTime<Utc>> {
        let previous = std::mem::replace(&mut *self.until.lock().unwrap(), until);
        if let Some(gauge) = &self.gauge {
            gauge.set(until.map_or(0, |until| until.timestamp()));
        }
        previous
    }
}

/// Serves `/engine/drain` for the base model cards this runtime registered. The route is
/// installed once per runtime; later calls (one per attached model) keep the first state.
/// Never fails: it runs on every worker's model registration, which a drain feature that may
/// never be used must not be able to break.
pub fn register_route(drt: &DistributedRuntime) {
    if drt.engine_routes().get(DRAIN_ROUTE).is_some() {
        return;
    }
    let gauge = drt
        .metrics()
        .create_intgauge(
            DRAINING_UNTIL_GAUGE,
            "Drain deadline accepted by this worker (unix seconds); 0 while not draining",
            &[],
        )
        .inspect_err(|error| {
            tracing::warn!(%error, metric = DRAINING_UNTIL_GAUGE, "Drain route installed without its gauge");
        })
        .ok();
    let state = Arc::new(DrainState {
        until: Mutex::new(None),
        gauge,
    });
    let discovery = drt.discovery();
    let instance_id = drt.connection_id();
    let callback: EngineRouteCallback = Arc::new(move |body| {
        let state = Arc::clone(&state);
        let discovery = Arc::clone(&discovery);
        Box::pin(async move { handle(&state, discovery.as_ref(), instance_id, body).await })
    });
    drt.engine_routes().register(DRAIN_ROUTE, callback);
}

async fn handle(
    state: &DrainState,
    discovery: &dyn Discovery,
    instance_id: u64,
    body: serde_json::Value,
) -> Result<serde_json::Value> {
    let until = match parse_body(&body)? {
        None => state.current(),
        Some(requested) => {
            let until = requested.filter(|until| *until > Utc::now());
            publish_draining_until(discovery, instance_id, until).await?;
            let previous = state.set(until);
            // llm-operator re-sends the same drain on every reconcile; log transitions only.
            if previous != until {
                tracing::info!(
                    previous = ?until_json(previous),
                    until = ?until_json(until),
                    requested = ?until_json(requested),
                    "Drain: deadline changed"
                );
            }
            until
        }
    };
    Ok(serde_json::json!({ "until": until_json(until) }))
}

/// Publishes `until` as the drain taint on each of this worker's base model cards, so routers
/// (through their runtime-config watch) and the planner read the state this worker reports.
/// Other caller taints (e.g. `WORKER_ADDED_TAINTS`) are kept; topology taints are regenerated by
/// discovery. Fails, without changing anything, when no base card is registered; llm-operator
/// retries on its next reconcile.
async fn publish_draining_until(
    discovery: &dyn Discovery,
    instance_id: u64,
    until: Option<DateTime<Utc>>,
) -> Result<()> {
    for (id, card) in base_cards(discovery, instance_id).await? {
        let taints = with_drain_taint(&card.runtime_config.taints, until);
        discovery
            .update_model_taints(id.clone(), taints)
            .await
            .with_context(|| format!("publishing drain taint on model card {}", id.to_path()))?;
    }
    Ok(())
}

/// The caller-managed taints to write: `taints` without topology taints (discovery owns them)
/// and with the drain taint replaced by `until`'s, or removed.
fn with_drain_taint(taints: &HashSet<String>, until: Option<DateTime<Utc>>) -> HashSet<String> {
    taints
        .iter()
        .filter(|taint| {
            !taint.starts_with(TOPOLOGY_TAINT_PREFIX) && !taint.starts_with(DRAIN_TAINT_PREFIX)
        })
        .cloned()
        .chain(until.map(|until| format!("{DRAIN_TAINT_PREFIX}{}", until.to_rfc3339())))
        .collect()
}

fn until_json(until: Option<DateTime<Utc>>) -> Option<String> {
    until.map(|until| until.to_rfc3339())
}

/// `{}` reads (`None`); `{"until": <RFC 3339> | null}` sets (`Some(deadline)` / `Some(None)`).
fn parse_body(body: &serde_json::Value) -> Result<Option<Option<DateTime<Utc>>>> {
    const SHAPE: &str = r#"drain body must be {} or {"until": "<RFC 3339>" | null}"#;
    let Some(object) = body.as_object() else {
        anyhow::bail!(SHAPE);
    };
    match (object.len(), object.get("until")) {
        (0, _) => Ok(None),
        (1, Some(serde_json::Value::Null)) => Ok(Some(None)),
        (1, Some(serde_json::Value::String(until))) => DateTime::parse_from_rfc3339(until)
            .map(|until| Some(Some(until.to_utc())))
            .with_context(|| format!("until {until:?} is not RFC 3339")),
        _ => anyhow::bail!(SHAPE),
    }
}

/// The base model cards registered by `instance_id`, with their discovery ids. LoRA cards
/// (suffixed keys) are skipped: routers read runtime config, taints included, from the base
/// card only, and `update_model_taints` accepts only base cards.
async fn base_cards(
    discovery: &dyn Discovery,
    instance_id: u64,
) -> Result<Vec<(ModelCardInstanceId, ModelDeploymentCard)>> {
    let mut cards = Vec::new();
    for instance in discovery.list(DiscoveryQuery::AllModels).await? {
        let DiscoveryInstanceId::Model(id) = instance.id() else {
            continue;
        };
        if id.instance_id != instance_id || id.model_suffix.is_some() {
            continue;
        }
        let card = instance
            .deserialize_model::<ModelDeploymentCard>()
            .with_context(|| format!("model card {}", id.to_path()))?;
        cards.push((id, card));
    }
    if cards.is_empty() {
        anyhow::bail!("no base model card is registered for worker {instance_id:x}");
    }
    Ok(cards)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::discovery::DiscoverySpec;

    #[tokio::test]
    async fn b10_drain_route_sets_reads_and_clears_the_deadline() {
        let runtime = dynamo_runtime::Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(
            runtime,
            dynamo_runtime::distributed::DistributedConfig::process_local(),
        )
        .await
        .unwrap();
        register_route(&drt);
        let route = drt.engine_routes().get(DRAIN_ROUTE).unwrap();
        let read = || async { route(serde_json::json!({})).await.unwrap() };
        let gauge = || {
            let metrics = drt.metrics().prometheus_expfmt().unwrap();
            metrics
                .lines()
                .find(|line| line.starts_with("dynamo_component_draining_until_unix_seconds"))
                .unwrap_or_else(|| panic!("draining_until gauge missing from:\n{metrics}"))
                .rsplit(' ')
                .next()
                .unwrap()
                .parse::<i64>()
                .unwrap()
        };
        assert_eq!(gauge(), 0);

        // Nothing is registered yet, so a drain cannot take effect and must say so.
        assert!(
            route(serde_json::json!({ "until": "2099-10-01T19:00:00+00:00" }))
                .await
                .is_err()
        );
        assert_eq!(read().await, serde_json::json!({"until": null}));

        let mut card = ModelDeploymentCard::with_name_only("drain-test-model");
        card.runtime_config.taints = HashSet::from(["service_tier/priority".to_string()]);
        let spec = |suffix: Option<&str>| {
            DiscoverySpec::from_model_with_suffix(
                "ns".to_string(),
                "worker".to_string(),
                "generate".to_string(),
                &card,
                suffix.map(str::to_string),
            )
            .unwrap()
        };
        let base = drt.discovery().register(spec(None)).await.unwrap();
        drt.discovery()
            .register(spec(Some("adapter")))
            .await
            .unwrap();
        let instance_id = drt.connection_id();
        let card_taints = |suffix: Option<&'static str>| {
            let discovery = drt.discovery();
            async move {
                let mut taints = discovery
                    .list(DiscoveryQuery::AllModels)
                    .await
                    .unwrap()
                    .into_iter()
                    .find_map(|instance| {
                        let DiscoveryInstanceId::Model(id) = instance.id() else {
                            return None;
                        };
                        (id.instance_id == instance_id && id.model_suffix.as_deref() == suffix)
                            .then(|| {
                                instance
                                    .deserialize_model::<ModelDeploymentCard>()
                                    .unwrap()
                                    .runtime_config
                                    .taints
                            })
                    })
                    .unwrap()
                    .into_iter()
                    .collect::<Vec<_>>();
                taints.sort();
                taints
            }
        };

        // Set, override, and read back; the gauge follows.
        for (until, unix) in [
            ("2099-10-01T19:00:00+00:00", 4_094_564_400),
            ("2099-10-01T19:05:00+00:00", 4_094_564_700),
        ] {
            let response = route(serde_json::json!({ "until": until })).await.unwrap();
            assert_eq!(response, serde_json::json!({ "until": until }));
            assert_eq!(read().await, serde_json::json!({ "until": until }));
            assert_eq!(gauge(), unix);
            // The base card carries the drain taint next to the worker's own; the LoRA card,
            // which routers do not read runtime config from, is left alone.
            assert_eq!(
                card_taints(None).await,
                [
                    format!("b10.drain/until={until}"),
                    "service_tier/priority".to_string()
                ]
            );
            assert_eq!(
                card_taints(Some("adapter")).await,
                ["service_tier/priority"]
            );
        }

        // Attaching another model (a LoRA) registers the route again and keeps the state.
        register_route(&drt);
        assert_eq!(
            read().await,
            serde_json::json!({ "until": "2099-10-01T19:05:00+00:00" })
        );

        // Any offset is accepted and normalized to UTC.
        route(serde_json::json!({ "until": "2099-10-01T12:00:00-07:00" }))
            .await
            .unwrap();
        let normalized = serde_json::json!({ "until": "2099-10-01T19:00:00+00:00" });
        assert_eq!(read().await, normalized);

        for bad in [
            serde_json::json!(null),
            serde_json::json!({"until": "soon"}),
            serde_json::json!({"until": 5}),
            serde_json::json!({"until": null, "extra": 1}),
            serde_json::json!({"deadline": "2099-10-01T12:00:00Z"}),
        ] {
            assert!(route(bad.clone()).await.is_err(), "{bad}");
        }
        assert_eq!(read().await, normalized);

        // null clears, and so does a deadline at or before now (how llm-operator cancels).
        route(serde_json::json!({ "until": null })).await.unwrap();
        assert_eq!(read().await, serde_json::json!({"until": null}));
        assert_eq!(gauge(), 0);
        assert_eq!(card_taints(None).await, ["service_tier/priority"]);
        route(serde_json::json!({ "until": "2099-01-01T00:00:00Z" }))
            .await
            .unwrap();
        let past = (Utc::now() - chrono::Duration::seconds(1)).to_rfc3339();
        assert_eq!(
            route(serde_json::json!({ "until": past })).await.unwrap(),
            serde_json::json!({"until": null})
        );
        assert_eq!(read().await, serde_json::json!({"until": null}));
        assert_eq!(card_taints(None).await, ["service_tier/priority"]);

        // After the base card is unregistered a drain has nothing to act on.
        drt.discovery().unregister(base).await.unwrap();
        assert!(
            route(serde_json::json!({ "until": "2099-10-01T13:00:00Z" }))
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn b10_drain_route_installs_even_when_its_gauge_cannot_register() {
        let runtime = dynamo_runtime::Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(
            runtime,
            dynamo_runtime::distributed::DistributedConfig::process_local(),
        )
        .await
        .unwrap();
        // Taking the gauge's name first makes the route's own registration fail.
        drt.metrics()
            .create_intgauge(DRAINING_UNTIL_GAUGE, "taken", &[])
            .unwrap();
        assert!(
            drt.metrics()
                .create_intgauge(DRAINING_UNTIL_GAUGE, "taken", &[])
                .is_err()
        );
        register_route(&drt);
        let route = drt.engine_routes().get(DRAIN_ROUTE).unwrap();
        assert_eq!(
            route(serde_json::json!({})).await.unwrap(),
            serde_json::json!({ "until": null })
        );
    }
}
