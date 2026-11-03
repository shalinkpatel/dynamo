// SPDX-FileCopyrightText: Copyright (c) 2026 Baseten. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The generation coordinator proto is vendored in lib/b10-client (source of
    // truth, byte-for-byte from the baseten monorepo) and carries the
    // KVLocatorService contract. b10-client keeps its own prost-only copy for
    // the generation path.
    println!("cargo:rerun-if-changed=../b10-client/proto/generation_coordinator.proto");
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR")?);
    tonic_prost_build::configure()
        .bytes(".dynamo.b10.generation.v1.Tokens.payload")
        .btree_map(".dynamo.b10.generation.v1.NewRequest.metadata")
        .file_descriptor_set_path(out_dir.join("kvlocator_descriptor.bin"))
        .build_client(true)
        .build_server(true)
        .compile_protos(
            &["../b10-client/proto/generation_coordinator.proto"],
            &["../b10-client/proto"],
        )?;
    Ok(())
}
