// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The bid seam: GwpScorer, the read-only ranking behind Locate.

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use dynamo_kvlocator::pb;
use dynamo_kvlocator::scorer::{InvalidRequest, Scorer};

use crate::config::EndpointId;
use crate::router::{GwpRouterRegistry, OracleVersionId};
use crate::session::AffinityStore;
use crate::topology::{ModelBinding, TopologySnapshot, TopologyStore};
use dynamo_kv_router::protocols::{WorkerId, WorkerWithDpRank};

use crate::router::RankedCandidate;

/// The bid scorer: ranks every eligible placement read-only through GWP's
/// per-model routers. One instance serves any number of models; the model is
/// resolved per request through the topology binding.
pub struct GwpScorer {
    router: Arc<GwpRouterRegistry>,
    topology: TopologyStore,
    affinity: Arc<dyn AffinityStore>,
}

impl GwpScorer {
    pub fn new(
        router: Arc<GwpRouterRegistry>,
        topology: TopologyStore,
        affinity: Arc<dyn AffinityStore>,
    ) -> Self {
        Self {
            router,
            topology,
            affinity,
        }
    }
}

#[async_trait]
impl Scorer for GwpScorer {
    /// Rank every eligible placement, ordered by increasing score, without
    /// booking anything. The stages, in order: validate the placement
    /// versions, resolve the model, narrow the caller's authoritative
    /// destinations down to an allowed worker set, rank, and map to the wire.
    async fn rank(&self, request: &pb::LocateRequest) -> anyhow::Result<pb::LocateResponse> {
        let (tokens, identity) = validate_input(request)?;

        let topology = self.topology.load();
        let (model_router, binding) = resolve_model(&self.router, &topology, &request.model_id)?;
        let Some((candidates, worker_configs)) = eligible_workers(&topology, binding, request)
        else {
            // No eligible destination answers as an empty ranking, not an error.
            return Ok(resolve_placements(
                Vec::new(),
                &topology,
                topology.canonical_model(&request.model_id),
            ));
        };

        let policy = topology
            .profile(binding)
            .ok_or_else(|| anyhow::anyhow!("model has no routing profile"))?
            .stages;
        let affine_worker = match session_id_of(request).filter(|_| policy.affinity) {
            Some(sid) => match self.affinity.peek_binding(sid).await {
                Ok(binding) => binding.map(|binding| binding.worker),
                Err(error) => {
                    tracing::warn!(%error, "affinity lookup failed; ranking without binding");
                    None
                }
            },
            None => None,
        };
        let mut ranked = model_router
            .rank_candidates(&tokens, &identity, &candidates, &worker_configs, policy)
            .await?;
        let preferences = dynamo_kv_router::protocols::RoutingConstraints {
            preferred_taints: request
                .constraints
                .as_ref()
                .into_iter()
                .flat_map(|constraints| &constraints.preferred_taints)
                .map(|taint| (taint.taint.clone(), taint.weight))
                .collect(),
            ..Default::default()
        };
        for candidate in &mut ranked {
            let config = &worker_configs[&candidate.worker.worker_id];
            let multiplier = preferences
                .preferred_taint_multiplier(&config.taints)
                .unwrap_or(1.0);
            candidate.score = crate::scoring::selection_cost(candidate.score, multiplier);
        }
        let mut response = resolve_placements(
            ranked,
            &topology,
            topology.canonical_model(&request.model_id),
        );
        for candidate in &mut response
            .placements_v1
            .as_mut()
            .expect("V1 evaluated")
            .candidates
        {
            let placement = candidate.placement.as_ref().expect("placement resolved");
            candidate
                .estimate
                .as_mut()
                .expect("estimate resolved")
                .affinity = affine_worker
                == Some(WorkerWithDpRank::new(
                    placement.worker_id,
                    placement.dp_rank,
                ));
        }
        Ok(response)
    }
}

/// Validate the same routing input for lookup and commit.
pub(super) fn validate_input(
    request: &pb::LocateRequest,
) -> anyhow::Result<(
    std::borrow::Cow<'_, [u32]>,
    crate::tokens::RoutingTokenIdentity,
)> {
    if request.model_id.is_empty() {
        return Err(InvalidRequest("model_id must not be empty".into()).into());
    }
    validate_placement_versions(request)?;
    if request.constraints.as_ref().is_some_and(|constraints| {
        constraints
            .preferred_taints
            .iter()
            .any(|taint| !taint.weight.is_finite())
    }) {
        return Err(InvalidRequest("preferred taint weights must be finite".into()).into());
    }
    bid_tokens(request)
}

pub(super) fn bid_tokens(
    request: &pb::LocateRequest,
) -> anyhow::Result<(
    std::borrow::Cow<'_, [u32]>,
    crate::tokens::RoutingTokenIdentity,
)> {
    use dynamo_kv_router::protocols::{BlockExtraInfo, BlockMmObjectInfo};
    let bid = request
        .bid
        .as_ref()
        .ok_or_else(|| InvalidRequest("bid is required".into()))?;
    let tokens = bid
        .token_ids()
        .map_err(|error| InvalidRequest(error.to_string()))?;
    let block_mm_infos = bid
        .mm_routing_args
        .as_ref()
        .map(|args| {
            args.blocks
                .iter()
                .map(|block| {
                    if !block.present {
                        if !block.objects.is_empty() {
                            return Err(InvalidRequest("absent MM block contains objects".into()));
                        }
                        return Ok(None);
                    }
                    let mm_objects = block
                        .objects
                        .iter()
                        .map(|object| {
                            let offsets = object
                                .offsets
                                .iter()
                                .map(|range| {
                                    if range.start > range.end {
                                        return Err(InvalidRequest(
                                            "MM offsets must be ordered".into(),
                                        ));
                                    }
                                    Ok((
                                        usize::try_from(range.start).map_err(|_| {
                                            InvalidRequest("MM offset out of range".into())
                                        })?,
                                        usize::try_from(range.end).map_err(|_| {
                                            InvalidRequest("MM offset out of range".into())
                                        })?,
                                    ))
                                })
                                .collect::<Result<Vec<_>, InvalidRequest>>()?;
                            Ok(BlockMmObjectInfo {
                                mm_hash: object.mm_hash,
                                offsets,
                            })
                        })
                        .collect::<Result<Vec<_>, InvalidRequest>>()?;
                    Ok(Some(BlockExtraInfo { mm_objects }))
                })
                .collect::<Result<Vec<_>, InvalidRequest>>()
        })
        .transpose()?;
    Ok((
        tokens,
        crate::tokens::RoutingTokenIdentity {
            block_mm_infos,
            cache_salt: bid.cache_salt.clone(),
        },
    ))
}

/// Placement versions: empty defaults to [V1]; UNSPECIFIED is invalid and a
/// request naming no supported version is an error. M1 supports only V1.
fn validate_placement_versions(request: &pb::LocateRequest) -> anyhow::Result<()> {
    let requested = &request.requested_placement_versions;
    if requested.contains(&(pb::PlacementVersion::Unspecified as i32)) {
        return Err(InvalidRequest(
            "requested_placement_versions must not contain PLACEMENT_VERSION_UNSPECIFIED".into(),
        )
        .into());
    }
    if !requested.is_empty() && !requested.contains(&(pb::PlacementVersion::V1 as i32)) {
        return Err(InvalidRequest("no supported placement version was requested".into()).into());
    }
    Ok(())
}

/// Resolve the model to its router and binding through the topology.
fn resolve_model<'a>(
    router: &GwpRouterRegistry,
    topology: &'a TopologySnapshot,
    model_id: &str,
) -> anyhow::Result<(std::sync::Arc<crate::router::GwpRouter>, &'a ModelBinding)> {
    let Some((canonical_model, binding)) = topology.model_binding(model_id) else {
        return Err(
            InvalidRequest(format!("model {model_id:?} is not known to GWP routing")).into(),
        );
    };
    let oracle_version_id = OracleVersionId::new(canonical_model.clone())?;
    let Some(model_router) = router.model_router(&oracle_version_id) else {
        anyhow::bail!("model {canonical_model:?} has no scheduler");
    };
    Ok((model_router, binding))
}

/// Narrow the caller's authoritative destinations to the workers the bid may
/// rank: the model's binding intersected with the caller's endpoints, then
/// the allowed-worker and required-taint filters, expanded to (worker,
/// dp_rank) tuples with the runtime configs the B10 scorer's dp-rank blend
/// needs — all from the topology snapshot, gwp's own data. None means
/// nothing is eligible.
pub(super) fn eligible_workers(
    topology: &TopologySnapshot,
    binding: &crate::topology::ModelBinding,
    request: &pb::LocateRequest,
) -> Option<(
    HashSet<dynamo_kv_router::protocols::WorkerWithDpRank>,
    std::collections::HashMap<
        WorkerId,
        dynamo_llm::local_model::runtime_config::ModelRuntimeConfig,
    >,
)> {
    // The caller's destinations are authoritative; the model's binding is
    // the other side. Both sides are config-sized (a handful of endpoints),
    // so this is a set intersection: compare by string, clone once for the
    // owned set workers_in_endpoints takes.
    // No requested destinations means every endpoint of the model.
    let requested: HashSet<EndpointId> = request
        .eligible_endpoint_ids
        .iter()
        .map(|id| super::wire_to_endpoint(topology, &request.model_id, id))
        .collect();
    let eligible_endpoints: HashSet<EndpointId> = binding
        .endpoints
        .iter()
        // Namespace discovery owns the destinations: no list means all of them.
        .filter(|endpoint| {
            (requested.is_empty() && topology.discovered_endpoints) || requested.contains(endpoint)
        })
        .cloned()
        .collect();
    if eligible_endpoints.is_empty() {
        return None;
    }

    let mut workers = topology.workers_in_endpoints(&eligible_endpoints);
    if !request.allowed_worker_ids.is_empty() {
        workers.retain(|worker| request.allowed_worker_ids.contains(worker));
    }
    if let Some(constraints) = request.constraints.as_ref()
        && !constraints.required_taints.is_empty()
    {
        workers.retain(|worker| {
            topology.workers.get(worker).is_some_and(|entry| {
                constraints
                    .required_taints
                    .iter()
                    .all(|taint| entry.runtime.taints.contains(taint))
            })
        });
    }
    if workers.is_empty() {
        return None;
    }
    let mut candidates = HashSet::new();
    let mut configs = std::collections::HashMap::new();
    for &worker in &workers {
        let Some(entry) = topology.workers.get(&worker) else {
            continue;
        };
        let runtime = &entry.runtime;
        configs.insert(worker, runtime.clone());
        let start = runtime.data_parallel_start_rank;
        for dp_rank in start..start + runtime.data_parallel_size {
            candidates.insert(WorkerWithDpRank::new(worker, dp_rank));
        }
    }
    Some((candidates, configs))
}

/// The bid's session id, when the frontend's coordinator payload carried one.
fn session_id_of(request: &pb::LocateRequest) -> Option<&str> {
    request
        .bid
        .as_ref()
        .and_then(|bid| bid.session_id.as_deref())
        .filter(|sid| !sid.is_empty())
}

/// Resolve ranked workers into wire placements. The ranking speaks worker
/// identity only; the topology snapshot is the one authority for which
/// endpoint a worker belongs to, so the endpoint_id on every placement
/// comes from here. A ranked worker the snapshot cannot place (a
/// registry/topology race during churn) is skipped rather than named.
fn resolve_placements(
    ranked: Vec<RankedCandidate>,
    topology: &TopologySnapshot,
    model: &str,
) -> pb::LocateResponse {
    let mut candidates: Vec<pb::LocateCandidate> = ranked
        .into_iter()
        .filter_map(|candidate| {
            let endpoint_id = topology.worker_endpoint(candidate.worker.worker_id)?;
            Some(pb::LocateCandidate {
                placement: Some(pb::Placement {
                    endpoint_id: super::endpoint_to_wire(topology, model, endpoint_id),
                    worker_id: candidate.worker.worker_id,
                    dp_rank: candidate.worker.dp_rank,
                }),
                estimate: Some(pb::BidResponse {
                    prefill_tokens: 0,
                    decode_tokens: 0,
                    ..Default::default()
                }),
                score: Some(candidate.score),
                estimated_overlap_tokens: candidate.cached_tokens as u64,
            })
        })
        .collect();
    candidates.sort_by(|a, b| {
        a.score
            .expect("scored")
            .total_cmp(&b.score.expect("scored"))
            .then_with(|| {
                let a = a.placement.as_ref().expect("placement resolved");
                let b = b.placement.as_ref().expect("placement resolved");
                (&a.endpoint_id, a.worker_id, a.dp_rank).cmp(&(
                    &b.endpoint_id,
                    b.worker_id,
                    b.dp_rank,
                ))
            })
    });
    pb::LocateResponse {
        placements_v1: Some(pb::PlacementResultV1 { candidates }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{EndpointConfig, GwpConfig, ModelRoute, RoutingConfig};
    use crate::topology::{TopologySnapshot, TopologyStore, TopologyWorker};
    use std::collections::{BTreeMap, HashMap};

    const ENDPOINT: &str = "kvl-endpoint";
    const WORKER: u64 = 10;

    async fn make_scorer() -> GwpScorer {
        make_scorer_with(ENDPOINT, false).await
    }

    async fn make_scorer_with(endpoint: &str, discovered: bool) -> GwpScorer {
        let config = GwpConfig {
            endpoints: BTreeMap::from([(
                EndpointId(endpoint.into()),
                EndpointConfig {
                    ingress_url: None,
                    api_key: "ck".into(),
                    planner_url: None,
                    planner_api_key: None,
                    properties: Default::default(),
                },
            )]),
            routes: vec![ModelRoute {
                models: vec!["m".into()],
                endpoints: vec![EndpointId(endpoint.into())],
            }],
            routing: RoutingConfig {
                block_size: 4,
                ..Default::default()
            },
            ..Default::default()
        };
        let snapshot = TopologySnapshot::from_config(
            &config,
            HashMap::from([
                (
                    WORKER,
                    TopologyWorker {
                        endpoint: EndpointId(endpoint.into()),
                        runtime:
                            dynamo_llm::local_model::runtime_config::ModelRuntimeConfig::default(),
                        observed_load: None,
                    },
                ),
                (
                    WORKER + 1,
                    TopologyWorker {
                        endpoint: EndpointId(endpoint.into()),
                        runtime:
                            dynamo_llm::local_model::runtime_config::ModelRuntimeConfig::default(),
                        observed_load: None,
                    },
                ),
            ]),
        )
        .unwrap();
        let mut snapshot = snapshot;
        snapshot.discovered_endpoints = discovered;
        let router = GwpRouterRegistry::new_process_local(4, 120).await.unwrap();
        router
            .reconcile_topology(&snapshot, &snapshot.workers.keys().copied().collect())
            .await
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        let affinity = Arc::new(crate::session::InMemoryAffinityStore::new());
        GwpScorer::new(router, TopologyStore::new(snapshot), affinity.clone())
    }

    fn locate_request(endpoint: &str, tokens: Vec<u32>) -> pb::LocateRequest {
        pb::LocateRequest {
            model_id: "m".into(),
            bid: Some(pb::BidRequest {
                tokens,
                ..Default::default()
            }),
            eligible_endpoint_ids: vec![endpoint.into()],
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn discovered_endpoints_travel_as_planes_and_empty_means_all() {
        let scorer = make_scorer_with("m/us-east", true).await;
        let mut request = locate_request("unused", vec![1, 2, 3, 4, 5, 6, 7, 8]);
        request.eligible_endpoint_ids.clear();
        let candidates = scorer
            .rank(&request)
            .await
            .unwrap()
            .placements_v1
            .unwrap()
            .candidates;
        assert_eq!(candidates.len(), 2);
        assert!(
            candidates
                .iter()
                .all(|c| c.placement.as_ref().unwrap().endpoint_id == "us-east")
        );
        request.eligible_endpoint_ids = vec!["us-east".into()];
        assert_eq!(
            scorer
                .rank(&request)
                .await
                .unwrap()
                .placements_v1
                .unwrap()
                .candidates
                .len(),
            2
        );
    }

    #[tokio::test]
    async fn configured_endpoints_are_not_translated_and_need_a_list() {
        let scorer = make_scorer_with("m/x", false).await;
        let mut request = locate_request("m/x", vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let candidates = scorer
            .rank(&request)
            .await
            .unwrap()
            .placements_v1
            .unwrap()
            .candidates;
        assert_eq!(candidates[0].placement.as_ref().unwrap().endpoint_id, "m/x");
        request.eligible_endpoint_ids.clear();
        assert!(
            scorer
                .rank(&request)
                .await
                .unwrap()
                .placements_v1
                .unwrap()
                .candidates
                .is_empty()
        );
    }

    #[tokio::test]
    async fn ranks_eligible_workers_and_books_nothing() {
        let scorer = make_scorer().await;
        let response = scorer
            .rank(&locate_request(ENDPOINT, vec![1, 2, 3, 4, 5, 6, 7, 8]))
            .await
            .expect("rank");
        let candidates = response.placements_v1.expect("v1 present").candidates;
        assert_eq!(candidates.len(), 2);
        let placement = candidates[0].placement.as_ref().unwrap();
        assert_eq!(placement.endpoint_id, ENDPOINT);
        assert_eq!(placement.worker_id, WORKER);
        assert!(candidates[0].score.is_some());

        // Read-only: the ranking left no scheduler slot behind.
        let model = OracleVersionId::new("m").unwrap();
        let router = scorer.router.model_router(&model).unwrap();
        assert_eq!(router.pending_count(), 0);
    }

    #[tokio::test]
    async fn unknown_destination_ranks_empty_and_unknown_model_errors() {
        let scorer = make_scorer().await;
        // Eligible destination outside the model's binding: empty ranking.
        let response = scorer
            .rank(&locate_request("other-endpoint", vec![1, 2, 3, 4]))
            .await
            .expect("no eligible destination is an empty ranking");
        assert!(
            response
                .placements_v1
                .expect("v1 present")
                .candidates
                .is_empty()
        );

        // Unknown model is an error, not an empty ranking.
        let mut request = locate_request(ENDPOINT, vec![1, 2, 3, 4]);
        request.model_id = "nope".into();
        assert!(scorer.rank(&request).await.is_err());
    }

    #[tokio::test]
    async fn b10_session_binding_marks_affinity_without_hiding_candidates() {
        let scorer = make_scorer().await;
        let bound = WorkerWithDpRank::new(WORKER, 0);
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        // Warm the indexer so the overlap estimate is real.
        scorer
            .router
            .model_router(&OracleVersionId::new("m").unwrap())
            .unwrap()
            .record_routing_decision("warmup", &tokens, bound)
            .await;
        scorer
            .affinity
            .put("sid-1", bound, std::time::Duration::from_secs(60))
            .await
            .expect("bind session");

        // Affinity is metadata on the full scored ranking.
        let mut request = locate_request(ENDPOINT, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        request.bid.as_mut().unwrap().session_id = Some("sid-1".into());
        let response = scorer.rank(&request).await.expect("rank");
        let candidates = response.placements_v1.expect("v1 present").candidates;
        assert_eq!(candidates.len(), 2, "every eligible candidate is scored");
        let placement = candidates[0].placement.as_ref().expect("placement");
        assert_eq!(placement.worker_id, WORKER);
        assert_eq!(placement.dp_rank, 0);
        assert!(candidates[0].estimate.as_ref().expect("estimate").affinity);
        assert_eq!(
            candidates[0].estimated_overlap_tokens, 8,
            "overlap estimate preserves the warmed prefix"
        );

        // A binding excluded by the caller's allow-list falls through to the
        // full ranking (and books nothing).
        let mut request = locate_request(ENDPOINT, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        request.bid.as_mut().unwrap().session_id = Some("sid-1".into());
        request.allowed_worker_ids = vec![WORKER + 1];
        let response = scorer.rank(&request).await.expect("rank");
        let candidates = response.placements_v1.unwrap().candidates;
        assert_eq!(candidates.len(), 1);
        assert_eq!(
            candidates[0].placement.as_ref().unwrap().worker_id,
            WORKER + 1
        );
        assert!(!candidates[0].estimate.as_ref().unwrap().affinity);

        // Without a session, the full ranking answers as before.
        let response = scorer
            .rank(&locate_request(ENDPOINT, vec![1, 2, 3, 4, 5, 6, 7, 8]))
            .await
            .expect("rank");
        let candidates = response.placements_v1.expect("v1 present").candidates;
        assert_eq!(candidates.len(), 2);
        assert!(!candidates[0].estimate.as_ref().expect("estimate").affinity);
    }

    #[tokio::test]
    async fn unspecified_placement_version_is_an_error() {
        let scorer = make_scorer().await;
        let mut request = locate_request(ENDPOINT, vec![1, 2, 3, 4]);
        request.requested_placement_versions = vec![pb::PlacementVersion::Unspecified as i32];
        assert!(scorer.rank(&request).await.is_err());
    }
    #[tokio::test]
    async fn b10_multimodal_and_salt_identity_affect_overlap() {
        let scorer = make_scorer().await;
        let router = scorer
            .router
            .model_router(&OracleVersionId::new("m").unwrap())
            .unwrap();
        let mut request = locate_request(ENDPOINT, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        // The wire codec emits one block entry with block-relative ranges.
        request.bid.as_mut().unwrap().mm_routing_args = Some(pb::MmRoutingArgs {
            blocks: vec![pb::MmRoutingBlock {
                present: true,
                objects: vec![pb::MmRoutingObject {
                    mm_hash: 42,
                    offsets: vec![pb::TokenRange { start: 0, end: 4 }],
                }],
            }],
        });
        request.bid.as_mut().unwrap().cache_salt = Some("tenant-a".into());
        let (tokens, identity) = bid_tokens(&request).unwrap();
        router
            .record_routing_decision_with_identity(
                "warm",
                &tokens,
                &identity,
                WorkerWithDpRank::new(WORKER, 0),
            )
            .await;
        drop(tokens);
        let bid = request.bid.as_mut().unwrap();
        bid.token_buffer = Some(pb::Tokens {
            payload: bid
                .tokens
                .iter()
                .flat_map(|id| id.to_le_bytes())
                .collect::<Vec<_>>()
                .into(),
        });
        bid.tokens.clear();
        let response = scorer.rank(&request).await.unwrap();
        let candidates = response.placements_v1.unwrap().candidates;
        assert_eq!(
            candidates
                .iter()
                .find(|c| c.placement.as_ref().unwrap().worker_id == WORKER)
                .unwrap()
                .estimated_overlap_tokens,
            8
        );
        for change_media in [true, false] {
            let mut changed = request.clone();
            let bid = changed.bid.as_mut().unwrap();
            if change_media {
                bid.mm_routing_args.as_mut().unwrap().blocks[0].objects[0].mm_hash += 1;
            } else {
                bid.cache_salt = Some("tenant-b".into());
            }
            let response = scorer.rank(&changed).await.unwrap();
            assert!(
                response
                    .placements_v1
                    .unwrap()
                    .candidates
                    .iter()
                    .all(|c| c.estimated_overlap_tokens == 0)
            );
        }
        assert_eq!(router.pending_count(), 0);
    }

    #[tokio::test]
    async fn b10_required_and_preferred_taints_use_worker_constraints() {
        let scorer = make_scorer().await;
        let mut topology = (*scorer.topology.load()).clone();
        topology
            .workers
            .get_mut(&WORKER)
            .unwrap()
            .runtime
            .taints
            .insert("zone-a".into());
        scorer.topology.replace(topology);
        let mut request = locate_request(ENDPOINT, vec![1, 2, 3, 4]);
        request.constraints = Some(pb::RoutingConstraints {
            preferred_taints: vec![pb::WeightedTaint {
                taint: "zone-a".into(),
                weight: 0.85,
            }],
            ..Default::default()
        });
        let response = scorer.rank(&request).await.unwrap();
        let candidates = response.placements_v1.unwrap().candidates;
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].placement.as_ref().unwrap().worker_id, WORKER);
        assert!(candidates[0].score.unwrap() < candidates[1].score.unwrap());
        request.constraints.as_mut().unwrap().required_taints = vec!["zone-a".into()];
        let response = scorer.rank(&request).await.unwrap();
        assert_eq!(response.placements_v1.unwrap().candidates.len(), 1);
        request.constraints.as_mut().unwrap().preferred_taints[0].weight = f32::NAN;
        assert!(scorer.rank(&request).await.is_err());
    }

    #[tokio::test]
    async fn b10_score_ties_order_by_endpoint_before_worker() {
        let scorer = make_scorer().await;
        let mut topology = (*scorer.topology.load()).clone();
        let endpoint = topology
            .endpoints
            .get(&EndpointId(ENDPOINT.into()))
            .unwrap()
            .clone();
        topology
            .endpoints
            .insert(EndpointId("a-endpoint".into()), endpoint);
        topology.workers.get_mut(&(WORKER + 1)).unwrap().endpoint = EndpointId("a-endpoint".into());
        let response = resolve_placements(
            vec![
                RankedCandidate {
                    worker: WorkerWithDpRank::new(WORKER, 0),
                    cached_tokens: 0,
                    score: 1.0,
                },
                RankedCandidate {
                    worker: WorkerWithDpRank::new(WORKER + 1, 0),
                    cached_tokens: 0,
                    score: 1.0,
                },
            ],
            &topology,
            "m",
        );
        assert_eq!(
            response.placements_v1.unwrap().candidates[0]
                .placement
                .as_ref()
                .unwrap()
                .worker_id,
            WORKER + 1
        );
    }
    #[tokio::test]
    async fn b10_locate_honors_model_policy_and_zero_cost_taint_preferences() {
        use crate::config::{LoadBalancingPolicy, ModelStagePolicy};
        let scorer = make_scorer().await;
        let mut request = locate_request(ENDPOINT, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        request.bid.as_mut().unwrap().session_id = Some("sid-policy".into());
        let worker = WorkerWithDpRank::new(WORKER, 0);
        scorer
            .affinity
            .put("sid-policy", worker, std::time::Duration::from_secs(60))
            .await
            .unwrap();
        let router = scorer
            .router
            .model_router(&OracleVersionId::new("m").unwrap())
            .unwrap();
        router
            .record_routing_decision("warm", &request.bid.as_ref().unwrap().tokens, worker)
            .await;
        let baseline = scorer
            .rank(&request)
            .await
            .unwrap()
            .placements_v1
            .unwrap()
            .candidates;
        assert!(
            baseline
                .iter()
                .any(|candidate| candidate.estimated_overlap_tokens == 8
                    && candidate.estimate.as_ref().unwrap().affinity)
        );
        assert!(
            baseline
                .iter()
                .any(|candidate| candidate.score.unwrap() > 1.0)
        );
        let mut topology = (*scorer.topology.load()).clone();
        let profile = topology.models["m"].profile.clone();
        topology.profiles.get_mut(&profile).unwrap().stages = ModelStagePolicy {
            trie: false,
            affinity: false,
            load_balancing: LoadBalancingPolicy {
                prefill_weight: Some(0.0),
                decode_weight: Some(0.0),
                active_request_weight: Some(0.0),
                cache_miss_weight: Some(0.0),
                ..Default::default()
            },
        };
        topology
            .workers
            .get_mut(&(WORKER + 1))
            .unwrap()
            .runtime
            .taints
            .insert("preferred".into());
        scorer.topology.replace(topology);
        let candidates = scorer
            .rank(&request)
            .await
            .unwrap()
            .placements_v1
            .unwrap()
            .candidates;
        assert!(
            candidates
                .iter()
                .all(|candidate| candidate.estimated_overlap_tokens == 0
                    && !candidate.estimate.as_ref().unwrap().affinity
                    && candidate.score == Some(1.0))
        );
        request.constraints = Some(pb::RoutingConstraints {
            preferred_taints: vec![pb::WeightedTaint {
                taint: "preferred".into(),
                weight: 0.85,
            }],
            ..Default::default()
        });
        let candidates = scorer
            .rank(&request)
            .await
            .unwrap()
            .placements_v1
            .unwrap()
            .candidates;
        assert_eq!(
            candidates[0].placement.as_ref().unwrap().worker_id,
            WORKER + 1,
            "preference wins even when unweighted work costs are zero"
        );
        assert!(candidates[0].score.unwrap() < candidates[1].score.unwrap());
        assert_eq!(router.pending_count(), 0);
    }
}
