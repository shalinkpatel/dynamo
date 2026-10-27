// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Baseten extensions for the Dynamo mocker.
//!
//! [`HiCache`] simulates an SGLang HiCache host tier, optionally shared with
//! peers (G2PB), with page hashes only. It follows the device KV events the
//! SGLang mocker publishes and answers which pages after a device prefix the
//! host tier or a peer can serve; the mocker owns everything else.

#[cfg(feature = "g2direct")]
pub mod g2pb;
pub mod hicache;
pub mod peer;

pub use hicache::{HiCache, HiCacheArgs, PrefixHits, WritePolicy};
pub use peer::{PeerDirectory, PeerDirectoryHandle, PeerIndex};
