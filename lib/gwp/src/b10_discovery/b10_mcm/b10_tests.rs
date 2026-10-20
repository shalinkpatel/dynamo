// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::b10_discovery::DiscoveryConfig;
use crate::config::{EndpointConfig, GwpConfig, ModelRoute};
use serde_json::json;

fn b10_config(version: &str) -> McmRedisConfig {
    McmRedisConfig {
        url_env: "DYN_GWP_TEST_REDIS".into(),
        endpoints: BTreeMap::from([(
            EndpointId("a".into()),
            McmEndpoint {
                model_deployment_id: version.into(),
                workload_plane: "us-east".into(),
                namespace: Some("tenant".into()),
            },
        )]),
        namespace: None,
        rescan_interval_secs: 30,
        poll_interval_ms: 10,
        timeout_ms: 1000,
        max_sample_age_secs: 10,
        observed_metrics: true,
    }
}
fn b10_routes(version: &str, revision: i64) -> String {
    json!({"modelVersionId": version, "namespace": "tenant", "ruleSetId": revision.to_string(), "weightedRuleSet": {"workloadPlaneWeights": [{"workloadPlaneName": "us-east", "weight": 100, "replicas": 1}]}}).to_string()
}
fn b10_sample(version: &str, now: DateTime<Utc>, capacity: f64) -> String {
    b10_sample_with_workers(version, now, capacity, &["1"])
}
// Alyx persists the response-header sample, whose encoder drops workerIds.
fn b10_observed_sample(version: &str, now: DateTime<Utc>, capacity: f64) -> String {
    let mut sample: serde_json::Value =
        serde_json::from_str(&b10_sample(version, now, capacity)).unwrap();
    sample["llmServiceMetricsV1"]
        .as_object_mut()
        .unwrap()
        .remove("workerIds");
    sample.to_string()
}
fn b10_sample_with_workers(
    version: &str,
    now: DateTime<Utc>,
    capacity: f64,
    ids: &[&str],
) -> String {
    json!({"modelDeploymentId": version, "workloadPlaneName": "us-east", "namespace": "tenant", "timestamp": now.to_rfc3339(), "llmServiceMetricsV1": {"capacity": capacity, "consumption": 3.0, "workerIds": ids}}).to_string()
}
fn b10_records(now: DateTime<Utc>) -> BTreeMap<String, Records> {
    BTreeMap::from([(
        "ov".into(),
        Records {
            routes: Some(b10_routes("ov", 42)),
            relayed: HashMap::from([("us-east".into(), b10_sample("ov", now, 10.0))]),
            observed: HashMap::new(),
        },
    )])
}
fn b10_gwp_config(version: &str) -> GwpConfig {
    let mut config = GwpConfig::default();
    config.endpoints.insert(
        EndpointId("a".into()),
        EndpointConfig {
            ingress_url: Some("http://coordinator:8000/v1".parse().unwrap()),
            planner_url: None,
            api_key: String::new(),
            planner_api_key: None,
            properties: Default::default(),
        },
    );
    config.routes.push(ModelRoute {
        models: vec![version.into()],
        endpoints: vec![EndpointId("a".into())],
    });
    config.discovery = DiscoveryConfig::McmRedis(b10_config(version));
    config
}

#[test]
fn b10_route_membership_and_freshness_publish_mcm_inventory() {
    let now = Utc::now();
    let mut view = View::default();
    let config = b10_config("ov");
    let gwp = b10_gwp_config("ov");
    let mut records = b10_records(now);
    view.refresh(&config, &records, now);
    let update = view.topology(&gwp).unwrap();
    assert_eq!(update.snapshot.workers.len(), 1);
    assert_eq!(update.snapshot.workers[&1].endpoint, EndpointId("a".into()));
    assert!(update.snapshot.workers[&1].observed_load.is_none());
    assert!(update.refreshed_workers.is_empty());
    update.snapshot.validate().unwrap();
    view.refresh(&config, &records, now + chrono::Duration::seconds(11));
    assert!(view.topology(&gwp).unwrap().snapshot.workers.is_empty());
    records.get_mut("ov").unwrap().routes = None;
    view.refresh(&config, &records, now);
    let update = view.topology(&gwp).unwrap();
    assert!(update.snapshot.endpoints.is_empty());
    assert!(update.snapshot.models["ov"].endpoints.is_empty());
    update.snapshot.validate().unwrap();
}

#[test]
fn b10_accepts_both_mcm_rule_sets_and_proto_json_spellings() {
    let now = Utc::now();
    for (route, expected) in [
        (b10_routes("ov", 42), true),
        (json!({"model_version_id":"ov","namespace":"tenant","rule_set_id":43,"fill_first_rule_set":{"workload_plane_states":[{"workload_plane_name":"us-east","replicas":2,"container_concurrency":10}]}}).to_string(), true),
        (json!({"modelVersionId":"ov","namespace":"tenant","ruleSetId":"44","weightedRuleSet":{"workloadPlaneWeights":[{"workloadPlaneName":"us-east","weight":100,"replicas":0}]}}).to_string(), false),
        (json!({"modelVersionId":"ov","namespace":"tenant","ruleSetId":"45","weightedRuleSet":{"workloadPlaneWeights":[{"workloadPlaneName":"us-east","weight":0,"replicas":1}]}}).to_string(), false),
        (json!({"modelVersionId":"ov","namespace":"tenant","ruleSetId":"46","weightedRuleSet":{"workloadPlaneWeights":[{"workloadPlaneName":"us-east","weight":100}]}}).to_string(), true),
        // Mid-migration fill-first route: the serving plane has no replica count yet.
        (json!({"modelVersionId":"ov","namespace":"tenant","ruleSetId":"47","fillFirstRuleSet":{"workloadPlaneStates":[{"workloadPlaneName":"us-east","containerConcurrency":1},{"workloadPlaneName":"us-west","replicas":1,"containerConcurrency":1}]}}).to_string(), true),
        (json!({"modelVersionId":"ov","namespace":"tenant","ruleSetId":"48","fillFirstRuleSet":{"workloadPlaneStates":[{"workloadPlaneName":"us-west","replicas":1,"containerConcurrency":1}]}}).to_string(), false),
    ] {
        let mut records = b10_records(now); records.get_mut("ov").unwrap().routes = Some(route);
        let mut view = View::default(); view.refresh(&b10_config("ov"), &records, now);
        assert_eq!(!view.allowed.is_empty(), expected);
    }
}

#[test]
fn b10_newest_sample_wins_and_old_updates_cannot_restore_capacity() {
    let now = Utc::now();
    let mut records = b10_records(now);
    let mut view = View::default();
    let config = b10_config("ov");
    records.get_mut("ov").unwrap().observed.insert(
        "us-east".into(),
        b10_observed_sample("ov", now + chrono::Duration::seconds(1), 0.0),
    );
    view.refresh(&config, &records, now + chrono::Duration::seconds(1));
    assert!(view.allowed.is_empty());
    records.get_mut("ov").unwrap().observed.clear();
    view.refresh(&config, &records, now + chrono::Duration::seconds(1));
    assert!(view.allowed.is_empty());
    records.get_mut("ov").unwrap().relayed.insert(
        "us-east".into(),
        b10_sample("ov", now + chrono::Duration::seconds(2), 10.0),
    );
    view.refresh(&config, &records, now + chrono::Duration::seconds(2));
    assert_eq!(view.allowed.len(), 1);
    records.get_mut("ov").unwrap().routes = Some(b10_routes("ov", 41));
    view.refresh(&config, &records, now + chrono::Duration::seconds(2));
    assert!(view.allowed.is_empty());
}

#[test]
fn b10_rejects_cross_deployment_samples_and_invalid_times_or_capacity() {
    let now = Utc::now();
    for raw in [
        b10_sample("other", now, 10.0),
        b10_sample("ov", now + chrono::Duration::hours(1), 10.0),
        b10_sample("ov", now, -1.0),
        "{}".into(),
        b10_sample("ov", now, 10.0).replace("tenant", "other-tenant"),
    ] {
        let mut records = b10_records(now);
        records
            .get_mut("ov")
            .unwrap()
            .relayed
            .insert("us-east".into(), raw);
        let mut view = View::default();
        view.refresh(&b10_config("ov"), &records, now);
        assert!(view.allowed.is_empty());
    }
}

#[test]
fn b10_validates_identity_mapping_and_roundtrips_configuration() {
    let example = include_str!("../../../../../deploy/gwp/mcm/gwp.example.yaml");
    let parsed = GwpConfig::load_from_str(example).unwrap();
    assert!(parsed.discovery.discovers_endpoints() && parsed.endpoints.is_empty());
    let mut gwp = b10_gwp_config("ov");
    // Planner discovery still requires every endpoint's planner_url.
    let mut planner = parsed.clone();
    planner.discovery = DiscoveryConfig::Planner;
    planner.endpoints = gwp.endpoints.clone();
    planner.routes = gwp.routes.clone();
    let error = GwpConfig::load_from_str(&serde_yaml::to_string(&planner).unwrap()).unwrap_err();
    assert!(
        error.to_string().contains("requires planner_url"),
        "{error}"
    );
    gwp.discovery = DiscoveryConfig::McmRedis(b10_config("ov"));
    gwp.discovery.validate(&gwp).unwrap();
    let encoded = serde_yaml::to_string(&gwp.discovery).unwrap();
    let decoded: DiscoveryConfig = serde_yaml::from_str(&encoded).unwrap();
    assert_eq!(decoded, gwp.discovery);
    let DiscoveryConfig::McmRedis(ref mut mcm) = gwp.discovery else {
        unreachable!()
    };
    mcm.endpoints
        .get_mut(&EndpointId("a".into()))
        .unwrap()
        .model_deployment_id = "other".into();
    assert!(gwp.discovery.validate(&gwp).is_err());
}

#[test]
fn b10_inventory_replaces_workers_and_newer_empty_samples_prevent_resurrection() {
    let now = Utc::now();
    let mut records = b10_records(now);
    let config = b10_config("ov");
    let mut view = View::default();
    view.refresh(&config, &records, now);
    assert!(view.workers.contains_key(&1));
    let later = now + chrono::Duration::seconds(1);
    records.get_mut("ov").unwrap().relayed.insert(
        "us-east".into(),
        b10_sample_with_workers("ov", later, 10.0, &["2", "18446744073709551615", "2"]),
    );
    view.refresh(&config, &records, later);
    assert_eq!(view.workers.len(), 2);
    assert!(view.workers.contains_key(&u64::MAX));
    assert!(!view.workers.contains_key(&1));
    let newest = later + chrono::Duration::seconds(1);
    records.get_mut("ov").unwrap().relayed.insert(
        "us-east".into(),
        b10_sample_with_workers("ov", newest, 10.0, &[]),
    );
    view.refresh(&config, &records, newest);
    assert!(view.workers.is_empty());
    assert!(view.allowed.is_empty());
    records.get_mut("ov").unwrap().relayed.insert(
        "us-east".into(),
        b10_sample_with_workers("ov", later, 10.0, &["2"]),
    );
    view.refresh(&config, &records, newest);
    assert!(
        view.workers.is_empty(),
        "older relayed inventory must not restore removed workers"
    );
}

#[test]
fn b10_observed_capacity_and_drains_preserve_independent_relayed_inventory() {
    let now = Utc::now();
    let mut records = b10_records(now);
    let config = b10_config("ov");
    let mut view = View::default();
    let newest = now + chrono::Duration::seconds(2);
    let until = (newest + chrono::Duration::minutes(5)).to_rfc3339();
    let mut observed: serde_json::Value =
        serde_json::from_str(&b10_observed_sample("ov", newest, 10.0)).unwrap();
    observed["llmServiceMetricsV1"]["drainingWorkers"] =
        json!([{"workerId":"1","until":until}, {"workerId":"2","until":until}]);
    records
        .get_mut("ov")
        .unwrap()
        .observed
        .insert("us-east".into(), observed.to_string());
    view.refresh(&config, &records, newest);
    assert_eq!(view.workers.len(), 1, "observed samples carry no inventory");
    assert_eq!(
        dynamo_kv_router::scheduling::b10_draining::draining_until(&view.workers[&1].runtime),
        Some(until.as_str())
    );

    // A relayed update can advance inventory while still lagging observed capacity.
    records.get_mut("ov").unwrap().relayed.insert(
        "us-east".into(),
        b10_sample_with_workers("ov", now + chrono::Duration::seconds(1), 10.0, &["2"]),
    );
    view.refresh(&config, &records, newest);
    assert_eq!(view.workers.len(), 1);
    assert!(view.workers.contains_key(&2));
    assert_eq!(
        dynamo_kv_router::scheduling::b10_draining::draining_until(&view.workers[&2].runtime),
        Some(until.as_str())
    );
    records
        .get_mut("ov")
        .unwrap()
        .relayed
        .insert("us-east".into(), b10_sample("ov", now, 10.0));
    view.refresh(&config, &records, newest);
    assert!(
        view.workers.is_empty(),
        "fresh capacity cannot undo a removal"
    );
}

#[test]
fn b10_observed_capacity_requires_fresh_valid_relayed_inventory() {
    let now = Utc::now();
    for relayed in [
        None,
        Some(b10_sample("ov", now - chrono::Duration::seconds(11), 10.0)),
        Some(b10_sample("other", now, 10.0)),
        Some(b10_sample_with_workers("ov", now, 10.0, &[])),
        Some(b10_sample_with_workers("ov", now, 10.0, &["invalid"])),
    ] {
        let mut records = b10_records(now);
        let record = records.get_mut("ov").unwrap();
        record.relayed.clear();
        if let Some(sample) = relayed {
            record.relayed.insert("us-east".into(), sample);
        }
        record.observed.insert(
            "us-east".into(),
            b10_observed_sample("ov", now + chrono::Duration::seconds(1), 10.0),
        );
        let mut view = View::default();
        view.refresh(
            &b10_config("ov"),
            &records,
            now + chrono::Duration::seconds(1),
        );
        assert!(view.workers.is_empty());
        assert!(view.allowed.is_empty());
    }
}

#[test]
fn b10_inventory_removals_are_retained_while_capacity_is_regressed() {
    let now = Utc::now();
    let mut records = b10_records(now);
    let config = b10_config("ov");
    let mut view = View::default();
    let newest = now + chrono::Duration::seconds(2);
    let observed = b10_observed_sample("ov", newest, 10.0);
    records
        .get_mut("ov")
        .unwrap()
        .observed
        .insert("us-east".into(), observed.clone());
    view.refresh(&config, &records, newest);
    assert!(view.workers.contains_key(&1));

    // Losing the observed sample regresses capacity; still remember this removal.
    records.get_mut("ov").unwrap().observed.clear();
    records.get_mut("ov").unwrap().relayed.insert(
        "us-east".into(),
        b10_sample_with_workers("ov", now + chrono::Duration::seconds(1), 10.0, &[]),
    );
    view.refresh(&config, &records, newest);
    assert!(view.workers.is_empty());
    records
        .get_mut("ov")
        .unwrap()
        .observed
        .insert("us-east".into(), observed);
    records
        .get_mut("ov")
        .unwrap()
        .relayed
        .insert("us-east".into(), b10_sample("ov", now, 10.0));
    view.refresh(&config, &records, newest);
    assert!(
        view.workers.is_empty(),
        "capacity recovery cannot restore older inventory"
    );
}

#[test]
fn b10_absent_or_invalid_worker_ids_produce_no_candidates() {
    let now = Utc::now();
    for ids in [
        json!([]),
        json!(["-1"]),
        json!(["+1"]),
        json!(["worker-1"]),
        json!(["18446744073709551616"]),
        json!(["1", "invalid"]),
        serde_json::Value::Null,
    ] {
        let mut sample: serde_json::Value =
            serde_json::from_str(&b10_sample("ov", now, 10.0)).unwrap();
        if ids.is_null() {
            sample["llmServiceMetricsV1"]
                .as_object_mut()
                .unwrap()
                .remove("workerIds");
        } else {
            sample["llmServiceMetricsV1"]["workerIds"] = ids;
        }
        let mut records = b10_records(now);
        records
            .get_mut("ov")
            .unwrap()
            .relayed
            .insert("us-east".into(), sample.to_string());
        let mut view = View::default();
        view.refresh(&b10_config("ov"), &records, now);
        assert!(view.workers.is_empty());
        assert!(view.allowed.is_empty());
    }
}

#[test]
fn b10_inventory_excludes_ambiguous_ids_and_preserves_drain_taints() {
    let now = Utc::now();
    let mut config = b10_config("ov");
    config.endpoints.insert(
        EndpointId("b".into()),
        McmEndpoint {
            workload_plane: "us-west".into(),
            ..config.endpoints[&EndpointId("a".into())].clone()
        },
    );
    let mut records = b10_records(now);
    let record = records.get_mut("ov").unwrap();
    let mut route: serde_json::Value =
        serde_json::from_str(record.routes.as_ref().unwrap()).unwrap();
    route["weightedRuleSet"]["workloadPlaneWeights"]
        .as_array_mut()
        .unwrap()
        .push(json!({"workloadPlaneName":"us-west","weight":50,"replicas":1}));
    record.routes = Some(route.to_string());
    let until = (now - chrono::Duration::seconds(1)).to_rfc3339();
    let east = json!({"model_deployment_id":"ov","workload_plane_name":"us-east","namespace":"tenant",
        "timestamp":now.to_rfc3339(),"llm_service_metrics_v1":{"capacity":10.0,"worker_ids":["1","2"],
        "draining_workers":[{"worker_id":"2","until":until},{"worker_id":"99","until":until}]}});
    record.relayed.insert("us-east".into(), east.to_string());
    record.relayed.insert(
        "us-west".into(),
        b10_sample_with_workers("ov", now, 10.0, &["1", "3"]).replace("us-east", "us-west"),
    );
    let mut view = View::default();
    view.refresh(&config, &records, now);
    assert!(!view.workers.contains_key(&1));
    assert!(!view.workers.contains_key(&99));
    assert_eq!(view.workers.len(), 2);
    assert_eq!(view.allowed.len(), 2);
    assert_eq!(
        dynamo_kv_router::scheduling::b10_draining::draining_until(&view.workers[&2].runtime),
        Some(until.as_str())
    );
    assert!(
        dynamo_kv_router::scheduling::b10_draining::draining_until(&view.workers[&3].runtime)
            .is_none()
    );
}

/// Uses the same Redis endpoint convention as the affinity integration test.
#[test]
fn b10_namespace_discovery_derives_endpoints_from_route_tables() {
    let route = |version: &str, namespace: &str| {
        json!({"modelVersionId": version, "namespace": namespace, "ruleSetId": "1",
            "weightedRuleSet": {"workloadPlaneWeights": [
                {"workloadPlaneName": "us-east", "weight": 50},
                {"workloadPlaneName": "us-west", "weight": 50}]}})
        .to_string()
    };
    let routes: Vec<RouteTable> = vec![serde_json::from_str(&route("ov1", "tenant")).unwrap()];
    let endpoints = discovered_endpoints(&routes);
    assert_eq!(
        endpoints.keys().map(|id| id.0.as_str()).collect::<Vec<_>>(),
        ["ov1/us-east", "ov1/us-west"]
    );

    let mut config = b10_config("ov1");
    config.endpoints = endpoints;
    config.namespace = Some("tenant".into());
    let gwp = GwpConfig {
        discovery: DiscoveryConfig::McmRedis(config.clone()),
        ..Default::default()
    };
    let derived = with_discovered(&gwp, &config.endpoints);
    assert_eq!(derived.routes.len(), 1);
    assert_eq!(derived.routes[0].models, ["ov1"]);
    assert!(derived.endpoints.values().all(|e| e.ingress_url.is_none()));

    let now = Utc::now();
    let mut records = b10_records(now);
    let ov1 = records.remove("ov").unwrap();
    records.insert(
        "ov1".into(),
        Records {
            routes: Some(route("ov1", "tenant")),
            relayed: HashMap::from([("us-east".into(), b10_sample("ov1", now, 10.0))]),
            observed: ov1.observed,
        },
    );
    // Production LSM samples carry no namespace.
    let mut sample: serde_json::Value =
        serde_json::from_str(&records["ov1"].relayed["us-east"]).unwrap();
    sample.as_object_mut().unwrap().remove("namespace");
    records
        .get_mut("ov1")
        .unwrap()
        .relayed
        .insert("us-east".into(), sample.to_string());
    let mut view = View::default();
    view.refresh(&config, &records, now);
    let update = view.topology(&derived).unwrap();
    assert_eq!(
        update.snapshot.workers[&1].endpoint,
        EndpointId("ov1/us-east".into())
    );
    assert_eq!(
        update.snapshot.models["ov1"].endpoints,
        HashSet::from([EndpointId("ov1/us-east".into())]),
        "a plane without fresh inventory is withdrawn"
    );
}

#[test]
fn b10_namespace_discovery_config_validates_without_endpoints() {
    let yaml = r#"
discovery:
  type: mcm_redis
  url_env: DYN_GWP_MCM_REDIS_URL
  namespace: org-tenant
session:
  backend:
    type: pubsub
"#;
    let config = GwpConfig::load_from_str(yaml).unwrap();
    assert!(config.endpoints.is_empty() && config.routes.is_empty());
    let with_endpoints = format!(
        "{yaml}endpoints:\n  a:\n    planner_url: http://p/deep/health\nroutes:\n  - models: [m]\n    endpoints: [a]\n"
    );
    assert!(GwpConfig::load_from_str(&with_endpoints).is_err());
}

#[tokio::test]
async fn b10_empty_tracked_set_polls_nothing() {
    // Nothing listens here: an empty set must not touch Redis or count as an error.
    unsafe { std::env::set_var("DYN_GWP_TEST_REDIS_UNUSED", "redis://127.0.0.1:1/") };
    let mut config = b10_config("ov");
    config.url_env = "DYN_GWP_TEST_REDIS_UNUSED".into();
    config.endpoints.clear();
    let registry = crate::router::GwpRouterRegistry::new_process_local(4, 120)
        .await
        .unwrap();
    let mut source = McmDiscovery::new(config, registry.metrics().clone()).unwrap();
    assert!(source.read().await.unwrap().is_empty());
    assert!(source.connection.is_none());
}

#[tokio::test]
#[ignore = "requires an isolated Redis at DYN_GWP_TEST_REDIS"]
async fn b10_real_redis_drives_topology_and_withdraws_expired_or_unavailable_routes() {
    use crate::router::GwpRouterRegistry;
    let url = std::env::var("DYN_GWP_TEST_REDIS").expect("DYN_GWP_TEST_REDIS");
    let mut connection = redis::Client::open(url)
        .unwrap()
        .get_multiplexed_async_connection()
        .await
        .unwrap();
    let registry = GwpRouterRegistry::new_process_local(4, 120).await.unwrap();
    let version = format!("gwp-test-{}", uuid::Uuid::new_v4());
    let config = b10_config(&version);
    let mut source = McmDiscovery::new(config.clone(), registry.metrics().clone()).unwrap();
    let route_key = format!("rt:{version}");
    let metrics_key = format!("bis:{version}:v1:metrics");
    let observed_key = format!("{metrics_key}:observed");
    let _: () = redis::pipe()
        .cmd("SET")
        .arg(&route_key)
        .arg(b10_routes(&version, 1))
        .ignore()
        .cmd("HSET")
        .arg(&metrics_key)
        .arg("us-east")
        .arg(b10_sample(&version, Utc::now(), 10.0))
        .ignore()
        .query_async(&mut connection)
        .await
        .unwrap();
    source.refresh().await;
    assert_eq!(source.view.allowed.len(), 1);
    source.config.observed_metrics = false;
    source.refresh().await;
    assert_eq!(source.view.allowed.len(), 1);
    source.config.observed_metrics = true;
    let _: () = redis::cmd("HSET")
        .arg(&observed_key)
        .arg("us-east")
        .arg(b10_observed_sample(&version, Utc::now(), 10.0))
        .query_async(&mut connection)
        .await
        .unwrap();
    source.refresh().await;
    assert!(
        source.view.workers.contains_key(&1),
        "observed headers omit worker IDs"
    );
    let _: () = redis::cmd("HSET")
        .arg(&observed_key)
        .arg("us-east")
        .arg(b10_observed_sample(&version, Utc::now(), 0.0))
        .query_async(&mut connection)
        .await
        .unwrap();
    source.refresh().await;
    assert!(source.view.allowed.is_empty());
    let _: () = redis::pipe()
        .cmd("DEL")
        .arg(&observed_key)
        .ignore()
        .cmd("HSET")
        .arg(&metrics_key)
        .arg("us-east")
        .arg(b10_sample(&version, Utc::now(), 10.0))
        .ignore()
        .query_async(&mut connection)
        .await
        .unwrap();
    source.refresh().await;
    assert_eq!(source.view.allowed.len(), 1);
    let mut gwp = b10_gwp_config(&version);
    let planner = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    gwp.endpoints
        .get_mut(&EndpointId("a".into()))
        .unwrap()
        .planner_url = Some(
        format!("http://{}/deep/health", planner.local_addr().unwrap())
            .parse()
            .unwrap(),
    );
    let store = ConfigStore::new(gwp);
    let (output_tx, mut output_rx) = tokio::sync::mpsc::channel(16);
    let handle = crate::b10_discovery::spawn(store.clone(), output_tx, registry.metrics().clone())
        .await
        .unwrap();
    let update = tokio::time::timeout(Duration::from_secs(2), output_rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(update.snapshot.workers.len(), 1);
    assert!(update.snapshot.workers[&1].observed_load.is_none());
    assert!(
        tokio::time::timeout(Duration::from_millis(100), planner.accept())
            .await
            .is_err(),
        "MCM discovery must not contact a configured planner"
    );
    let mut next = (*store.load()).clone();
    next.endpoints
        .get_mut(&EndpointId("a".into()))
        .unwrap()
        .api_key = "reloaded-key".into();
    store.replace(next);
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let update = output_rx.recv().await.unwrap();
            if update.snapshot.endpoints[&EndpointId("a".into())].api_key == "reloaded-key" {
                break;
            }
        }
    })
    .await
    .unwrap();
    let _: () = redis::cmd("PEXPIRE")
        .arg(&metrics_key)
        .arg(1)
        .query_async(&mut connection)
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if output_rx.recv().await.unwrap().snapshot.workers.is_empty() {
                break;
            }
        }
    })
    .await
    .unwrap();
    handle.abort();
    let _: () = redis::cmd("HSET")
        .arg(&metrics_key)
        .arg("us-east")
        .arg(b10_sample(&version, Utc::now(), 10.0))
        .query_async(&mut connection)
        .await
        .unwrap();
    let mut source = McmDiscovery::new(config, registry.metrics().clone()).unwrap();
    source.refresh().await;
    assert_eq!(source.view.workers.len(), 1);
    source.connection = None;
    source.client = redis::Client::open("redis://127.0.0.1:1/").unwrap();
    source.refresh().await;
    assert!(source.view.allowed.is_empty());
    assert!(source.view.workers.is_empty());
    let metrics = registry.prometheus_metrics().unwrap();
    assert!(
        metrics
            .lines()
            .any(|line| line.contains("gwp_discovery_polls_total")
                && line.contains("result=\"error\"")
                && line.ends_with(" 1")),
        "{metrics}"
    );
    let _: () = redis::cmd("DEL")
        .arg(&[route_key, metrics_key, observed_key])
        .query_async(&mut connection)
        .await
        .unwrap();
    registry.shutdown();
}

#[tokio::test]
#[ignore = "requires an isolated Redis at DYN_GWP_TEST_REDIS"]
async fn b10_real_redis_namespace_scan_tracks_deployments() {
    use crate::router::GwpRouterRegistry;
    let url = std::env::var("DYN_GWP_TEST_REDIS").expect("DYN_GWP_TEST_REDIS");
    let mut connection = redis::Client::open(url)
        .unwrap()
        .get_multiplexed_async_connection()
        .await
        .unwrap();
    let namespace = format!("org-{}", uuid::Uuid::new_v4());
    let id = || format!("gwp-scan-{}", uuid::Uuid::new_v4());
    let (ours, starting, theirs, broken) = (id(), id(), id(), id());
    let route = |version: &str, namespace: &str| {
        json!({"modelVersionId": version, "namespace": namespace, "ruleSetId": "1",
            "weightedRuleSet": {"workloadPlaneWeights": [{"workloadPlaneName": "us-east", "weight": 100}]}})
        .to_string()
    };
    let _: () = redis::pipe()
        .cmd("SET")
        .arg(format!("rt:{ours}"))
        .arg(route(&ours, &namespace))
        .ignore()
        .cmd("HSET")
        .arg(format!("bis:{ours}:v1:metrics"))
        .arg("us-east")
        .arg(b10_sample(&ours, Utc::now(), 10.0))
        .ignore()
        .cmd("SET")
        .arg(format!("rt:{starting}"))
        .arg(route(&starting, &namespace))
        .ignore()
        .cmd("SET")
        .arg(format!("rt:{theirs}"))
        .arg(route(&theirs, "org-other"))
        .ignore()
        .cmd("SET")
        .arg(format!("rt:{broken}"))
        .arg("not json")
        .ignore()
        .cmd("HSET")
        .arg(format!("bis:{theirs}:v1:metrics"))
        .arg("us-east")
        .arg(b10_sample(&theirs, Utc::now(), 10.0))
        .ignore()
        .query_async(&mut connection)
        .await
        .unwrap();

    let registry = GwpRouterRegistry::new_process_local(4, 120).await.unwrap();
    let mut config = b10_config(&ours);
    config.endpoints.clear();
    config.namespace = Some(namespace.clone());
    let mut source = McmDiscovery::new(config, registry.metrics().clone()).unwrap();
    source.refresh().await;
    assert_eq!(
        source.config.endpoints.keys().cloned().collect::<Vec<_>>(),
        [EndpointId(format!("{ours}/us-east"))],
        "only deployments in the namespace with LSM samples are tracked"
    );
    assert!(source.foreign_routes.contains(&format!("rt:{theirs}")));
    assert!(
        !source.foreign_routes.contains(&format!("rt:{broken}")),
        "unparsable routes are skipped, not cached"
    );

    // A deployment whose samples appear later joins on the next scan.
    let _: () = redis::cmd("HSET")
        .arg(format!("bis:{starting}:v1:metrics"))
        .arg("us-east")
        .arg(b10_sample(&starting, Utc::now(), 10.0))
        .query_async(&mut connection)
        .await
        .unwrap();
    source.last_scan = None;
    source.refresh().await;
    assert_eq!(source.config.endpoints.len(), 2);

    let _: () = redis::cmd("DEL")
        .arg(format!("rt:{ours}"))
        .arg(format!("rt:{theirs}"))
        .query_async(&mut connection)
        .await
        .unwrap();
    source.last_scan = None;
    source.refresh().await;
    assert_eq!(
        source.config.endpoints.keys().cloned().collect::<Vec<_>>(),
        [EndpointId(format!("{starting}/us-east"))],
        "a removed route drops its deployment"
    );
    assert!(
        !source.foreign_routes.contains(&format!("rt:{theirs}")),
        "cache forgets deleted keys"
    );
    assert!(
        !source.view.revisions.contains_key(&ours),
        "a deleted route drops its revision guard"
    );
    let _: () = redis::cmd("DEL")
        .arg(format!("rt:{broken}"))
        .arg(format!("rt:{starting}"))
        .arg(format!("bis:{ours}:v1:metrics"))
        .arg(format!("bis:{starting}:v1:metrics"))
        .arg(format!("bis:{theirs}:v1:metrics"))
        .query_async(&mut connection)
        .await
        .unwrap();
}

#[tokio::test]
#[ignore = "requires an isolated Redis at DYN_GWP_TEST_REDIS"]
async fn b10_real_redis_publishes_only_changed_topology() {
    use crate::router::GwpRouterRegistry;
    let url = std::env::var("DYN_GWP_TEST_REDIS").expect("DYN_GWP_TEST_REDIS");
    let mut connection = redis::Client::open(url)
        .unwrap()
        .get_multiplexed_async_connection()
        .await
        .unwrap();
    let version = format!("gwp-quiet-{}", uuid::Uuid::new_v4());
    let metrics_key = format!("bis:{version}:v1:metrics");
    let write = |ids: &'static [&'static str]| {
        redis::pipe()
            .cmd("SET")
            .arg(format!("rt:{version}"))
            .arg(b10_routes(&version, 1))
            .ignore()
            .cmd("HSET")
            .arg(&metrics_key)
            .arg("us-east")
            .arg(b10_sample_with_workers(&version, Utc::now(), 10.0, ids))
            .ignore()
            .clone()
    };
    let _: () = write(&["1"]).query_async(&mut connection).await.unwrap();
    let registry = GwpRouterRegistry::new_process_local(4, 120).await.unwrap();
    let store = ConfigStore::new(b10_gwp_config(&version));
    let (tx, mut rx) = tokio::sync::mpsc::channel(16);
    let handle = crate::b10_discovery::spawn(store, tx, registry.metrics().clone())
        .await
        .unwrap();
    let first = tokio::time::timeout(Duration::from_secs(2), rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(first.snapshot.workers.len(), 1);
    // Fresh samples with the same membership publish nothing (poll is 10 ms).
    for _ in 0..5 {
        let _: () = write(&["1"]).query_async(&mut connection).await.unwrap();
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
    assert!(rx.try_recv().is_err(), "unchanged topology was republished");
    let _: () = write(&["1", "2"])
        .query_async(&mut connection)
        .await
        .unwrap();
    let changed = tokio::time::timeout(Duration::from_secs(2), rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(changed.snapshot.workers.len(), 2);
    handle.abort();
    let _: () = redis::cmd("DEL")
        .arg(format!("rt:{version}"))
        .arg(&metrics_key)
        .query_async(&mut connection)
        .await
        .unwrap();
}

#[test]
fn b10_discovered_endpoints_skip_invalid_and_colliding_ids() {
    let route = |version: &str, planes: &[&str]| -> RouteTable {
        let weights: Vec<_> = planes
            .iter()
            .map(|p| json!({"workloadPlaneName": p, "weight": 1}))
            .collect();
        serde_json::from_value(json!({"modelVersionId": version, "namespace": "tenant",
            "ruleSetId": "1", "weightedRuleSet": {"workloadPlaneWeights": weights}}))
        .unwrap()
    };
    // "a" on plane "b/c" and "a/b" on plane "c" would both be "a/b/c".
    let routes = [
        route("a", &["b/c", " "]),
        route("a/b", &["c", "d"]),
        route("  ", &["e"]),
    ];
    let endpoints = discovered_endpoints(&routes);
    assert_eq!(
        endpoints.keys().map(|id| id.0.as_str()).collect::<Vec<_>>(),
        ["a/b/d"],
        "a colliding id is skipped for every claimant"
    );
}

#[test]
fn b10_view_forgets_untracked_deployments() {
    let now = Utc::now();
    let mut view = View::default();
    let mut config = b10_config("ov");
    view.refresh(&config, &b10_records(now), now);
    assert!(!view.sample_times.is_empty() && !view.revisions.is_empty());
    config.endpoints.clear();
    view.refresh(&config, &BTreeMap::new(), now);
    assert!(view.sample_times.is_empty());
    assert!(view.inventory_times.is_empty());
    assert!(
        !view.revisions.is_empty(),
        "the route-revision guard outlives tracking; the scan drops it with the route"
    );
}

#[test]
fn b10_with_discovered_drops_aliases_to_undiscovered_models() {
    let mut endpoints = BTreeMap::new();
    endpoints.insert(
        EndpointId("ov1/us-east".into()),
        McmEndpoint {
            model_deployment_id: "ov1".into(),
            workload_plane: "us-east".into(),
            namespace: None,
        },
    );
    let config = GwpConfig {
        served_alias_model_map: BTreeMap::from([
            ("live".to_string(), "ov1".to_string()),
            ("gone".to_string(), "ov2".to_string()),
        ]),
        ..Default::default()
    };
    let derived = with_discovered(&config, &endpoints);
    assert_eq!(
        derived.served_alias_model_map.keys().collect::<Vec<_>>(),
        ["live"]
    );
    TopologySnapshot::from_config(&derived, HashMap::new()).unwrap();
}

#[tokio::test]
#[ignore = "requires an isolated Redis at DYN_GWP_TEST_REDIS"]
async fn b10_real_redis_one_bad_deployment_does_not_withdraw_others() {
    use crate::router::GwpRouterRegistry;
    let url = std::env::var("DYN_GWP_TEST_REDIS").expect("DYN_GWP_TEST_REDIS");
    let mut connection = redis::Client::open(url)
        .unwrap()
        .get_multiplexed_async_connection()
        .await
        .unwrap();
    let good = format!("gwp-good-{}", uuid::Uuid::new_v4());
    let bad = format!("gwp-bad-{}", uuid::Uuid::new_v4());
    let _: () = redis::pipe()
        .cmd("SET")
        .arg(format!("rt:{good}"))
        .arg(b10_routes(&good, 1))
        .ignore()
        .cmd("HSET")
        .arg(format!("bis:{good}:v1:metrics"))
        .arg("us-east")
        .arg(b10_sample(&good, Utc::now(), 10.0))
        .ignore()
        .cmd("SET")
        .arg(format!("rt:{bad}"))
        .arg(b10_routes(&bad, 1))
        .ignore()
        // WRONGTYPE for HGETALL.
        .cmd("SET")
        .arg(format!("bis:{bad}:v1:metrics"))
        .arg("not a hash")
        .ignore()
        .query_async(&mut connection)
        .await
        .unwrap();
    let mut config = b10_config(&good);
    let mut bad_endpoint = config.endpoints[&EndpointId("a".into())].clone();
    bad_endpoint.model_deployment_id = bad.clone();
    config
        .endpoints
        .insert(EndpointId("b".into()), bad_endpoint);
    let registry = GwpRouterRegistry::new_process_local(4, 120).await.unwrap();
    let mut source = McmDiscovery::new(config, registry.metrics().clone()).unwrap();
    source.refresh().await;
    assert!(source.view.allowed.contains(&EndpointId("a".into())));
    assert!(!source.view.allowed.contains(&EndpointId("b".into())));
    let _: () = redis::cmd("DEL")
        .arg(format!("rt:{good}"))
        .arg(format!("bis:{good}:v1:metrics"))
        .arg(format!("rt:{bad}"))
        .arg(format!("bis:{bad}:v1:metrics"))
        .query_async(&mut connection)
        .await
        .unwrap();
}
