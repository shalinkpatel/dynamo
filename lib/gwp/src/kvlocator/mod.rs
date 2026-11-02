// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! dynamo-gwp's implementations of the dynamo-kvlocator seams.
//!
//! The traits live in the contract crate (`dynamo-kvlocator`); the
//! implementations live here, next to the machinery they adapt, so nothing
//! has to be exported. One seam per file: [`scoring`] is the bid
//! (intend+book is one step: committing *is* booking), [`lifecycle`] is
//! intend+book, open, and finish over GwpCore's slot machinery.

use crate::config::EndpointId;
use crate::topology::TopologySnapshot;

pub mod lifecycle;
pub mod scoring;

pub use lifecycle::GwpLifecycle;
pub use scoring::GwpScorer;

/// Native placement service registered on GWP's existing listener.
pub type LocatorServer = dynamo_kvlocator::pb::kv_locator_service_server::KvLocatorServiceServer<
    dynamo_kvlocator::GrpcKVLocator,
>;

/// [`LocatorServer`] behind the interceptor that stamps when each call's headers arrived.
pub type LocatorService = tonic::service::interceptor::InterceptedService<
    LocatorServer,
    fn(tonic::Request<()>) -> Result<tonic::Request<()>, tonic::Status>,
>;

/// Wire Alyx's native RPCs to the same core and readiness used by Envoy.
pub fn grpc_server(
    core: crate::core::GwpCore,
    readiness: crate::lifecycle::Lifecycle,
) -> LocatorService {
    let router = core.router.clone();
    let service = dynamo_kvlocator::GrpcKVLocator::new(move || readiness.is_ready())
        .with_scorer(std::sync::Arc::new(GwpScorer::new(
            core.router.clone(),
            core.topology.clone(),
            core.affinity.clone(),
        )))
        .with_observer(std::sync::Arc::new(move |method, tokens, elapsed| {
            router
                .metrics()
                .observe_kvlocator_call(method, tokens, elapsed)
        }))
        .with_lifecycle(std::sync::Arc::new(GwpLifecycle::new(core)));
    tonic::service::interceptor::InterceptedService::new(
        LocatorServer::new(service)
            .max_decoding_message_size(dynamo_kvlocator::GRPC_MAX_MESSAGE_BYTES),
        dynamo_kvlocator::mark_received as fn(_) -> _,
    )
}

/// Discovered `<model>/<plane>` ids travel as the plane alone.
fn wire_to_endpoint(topology: &TopologySnapshot, model: &str, id: &str) -> EndpointId {
    if topology.discovered_endpoints {
        EndpointId(format!("{}/{id}", topology.canonical_model(model)))
    } else {
        EndpointId(id.to_string())
    }
}

fn endpoint_to_wire(topology: &TopologySnapshot, model: &str, endpoint: &EndpointId) -> String {
    if !topology.discovered_endpoints {
        return endpoint.0.clone();
    }
    endpoint
        .0
        .strip_prefix(model)
        .and_then(|rest| rest.strip_prefix('/'))
        .unwrap_or(&endpoint.0)
        .to_string()
}
