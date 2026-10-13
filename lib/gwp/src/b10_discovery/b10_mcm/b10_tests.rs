// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::b10_discovery::DiscoveryConfig;
use crate::config::{EndpointConfig, GwpConfig, ModelRoute};
use crate::topology::TopologyWorker;
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
    // Match LSM -> protojson -> Beefeater Redis. workerIds intentionally empty:
    // it is not an authoritative inventory and must not erase planner workers.
    json!({"modelDeploymentId": version, "workloadPlaneName": "us-east", "namespace": "tenant", "timestamp": now.to_rfc3339(), "llmServiceMetricsV1": {"capacity": capacity, "consumption": 3.0, "workerIds": []}}).to_string()
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
fn b10_topology(version: &str) -> (GwpConfig, TopologySnapshot) {
    let mut config = GwpConfig::default();
    config.endpoints.insert(
        EndpointId("a".into()),
        EndpointConfig {
            ingress_url: "http://coordinator:8000/v1".parse().unwrap(),
            planner_url: "http://planner:8001/deep/health".parse().unwrap(),
            api_key: String::new(),
            planner_api_key: None,
            properties: Default::default(),
        },
    );
    config.routes.push(ModelRoute {
        models: vec![version.into()],
        endpoints: vec![EndpointId("a".into())],
    });
    let workers = HashMap::from([(
        1,
        TopologyWorker {
            endpoint: EndpointId("a".into()),
            runtime: Default::default(),
            observed_load: None,
        },
    )]);
    let snapshot = TopologySnapshot::from_config(&config, workers).unwrap();
    (config, snapshot)
}

#[test]
fn b10_route_membership_and_freshness_filter_planner_inventory() {
    let now = Utc::now();
    let mut view = View::default();
    let config = b10_config("ov");
    let mut records = b10_records(now);
    view.refresh(&config, &records, now);
    let (_, snapshot) = b10_topology("ov");
    let update = view.filter(&snapshot, &HashSet::from([1]));
    assert_eq!(update.snapshot.workers.len(), 1);
    assert!(update.refreshed_workers.contains(&1));
    update.snapshot.validate().unwrap();
    assert!(
        view.filter(&snapshot, &HashSet::new())
            .refreshed_workers
            .is_empty()
    );
    view.refresh(&config, &records, now + chrono::Duration::seconds(11));
    assert!(
        view.filter(&snapshot, &HashSet::new())
            .snapshot
            .workers
            .is_empty()
    );
    records.get_mut("ov").unwrap().routes = None;
    view.refresh(&config, &records, now);
    let update = view.filter(&snapshot, &HashSet::from([1]));
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
        b10_sample("ov", now + chrono::Duration::seconds(1), 0.0),
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
    GwpConfig::load_from_str(include_str!(
        "../../../../../deploy/gwp/mcm/gwp.example.yaml"
    ))
    .unwrap();
    let (mut gwp, _) = b10_topology("ov");
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

/// Uses the same Redis endpoint convention as the affinity integration test.
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
        .arg(b10_sample(&version, Utc::now(), 0.0))
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
    let (_, snapshot) = b10_topology(&version);
    let (planner_tx, planner_rx) = tokio::sync::mpsc::channel(2);
    let (output_tx, mut output_rx) = tokio::sync::mpsc::channel(2);
    let handle = source.spawn(planner_rx, output_tx);
    planner_tx
        .send(TopologyUpdate {
            snapshot,
            refreshed_workers: HashSet::from([1]),
        })
        .await
        .unwrap();
    let update = tokio::time::timeout(Duration::from_secs(2), output_rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(update.snapshot.workers.len(), 1);
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
    let mut source = McmDiscovery::new(config, registry.metrics().clone()).unwrap();
    source.view.allowed.insert(EndpointId("a".into()));
    source.client = redis::Client::open("redis://127.0.0.1:1/").unwrap();
    source.refresh().await;
    assert!(source.view.allowed.is_empty());
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
