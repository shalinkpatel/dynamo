// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The generation coordinator proto is vendored in lib/b10-client (source of
    // truth, byte-for-byte from the baseten monorepo) and carries the
    // KVLocatorService contract. b10-client keeps its own prost-only copy for
    // the generation path.
    println!("cargo:rerun-if-changed=../b10-client/proto/generation_coordinator.proto");
    tonic_prost_build::configure()
        .btree_map(".dynamo.b10.generation.v1.NewRequest.metadata")
        .build_client(true)
        .build_server(true)
        .compile_protos(
            &["../b10-client/proto/generation_coordinator.proto"],
            &["../b10-client/proto"],
        )?;
    Ok(())
}
