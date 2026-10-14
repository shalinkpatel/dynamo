// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! dynamo-gwp's implementations of the dynamo-kvlocator seams.
//!
//! The traits live in the contract crate (`dynamo-kvlocator`); the
//! implementations live here, next to the machinery they adapt, so nothing
//! has to be exported. One seam per file: [`scoring`] is the bid
//! (intend+book is one step: committing *is* booking), [`lifecycle`] is
//! intend+book, open, and finish over GwpCore's slot machinery.

pub mod lifecycle;
pub mod scoring;

pub use lifecycle::GwpLifecycle;
pub use scoring::GwpScorer;

/// Native placement service registered on GWP's existing listener.
pub type LocatorServer = dynamo_kvlocator::pb::kv_locator_service_server::KvLocatorServiceServer<
    dynamo_kvlocator::GrpcKVLocator,
>;

/// Wire Alyx's native RPCs to the same core and readiness used by Envoy.
pub fn grpc_server(
    core: crate::core::GwpCore,
    readiness: crate::lifecycle::Lifecycle,
) -> LocatorServer {
    let service = dynamo_kvlocator::GrpcKVLocator::new(move || readiness.is_ready())
        .with_scorer(std::sync::Arc::new(GwpScorer::new(
            core.router.clone(),
            core.topology.clone(),
            core.affinity.clone(),
        )))
        .with_lifecycle(std::sync::Arc::new(GwpLifecycle::new(core)));
    LocatorServer::new(service).max_decoding_message_size(dynamo_kvlocator::GRPC_MAX_MESSAGE_BYTES)
}
