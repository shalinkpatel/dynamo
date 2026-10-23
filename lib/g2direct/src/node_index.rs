// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One serving-G2 directory per node instead of one per TP rank.
//!
//! Every directory copy subscribes to the replica-event subject, so NATS
//! delivers each delta once per copy. In node mode only global TP rank 0
//! subscribes; every other process of the same replica (the other TP ranks
//! and, under attention DP, the other DP groups) sends its raw lookups to it
//! over an abstract Unix socket named after the shared replica id. The
//! directory is the same global one in every process, so which rank owns it
//! does not change answers. Each rank still filters the answer itself (own replica,
//! liveness, suspended pools), so the owner serves plain `batch_locations`.
//!
//! The directory is advisory, so a lookup that cannot reach the owner is a
//! miss: a dead or wedged owner costs hits, never a hang.

use std::io::{Read, Write};
use std::os::linux::net::SocketAddrExt;
use std::os::unix::net::{SocketAddr, UnixListener, UnixStream};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use dynamo_tokens::SequenceHash;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use serde::{Deserialize, Serialize};

use crate::peer_index::{
    ReplicaIndex, ReplicaLocation, ReplicaQueryRequest, ReplicaQueryResponse, ReplicaSourceKey,
};

/// One frame from a non-owner rank. Both ends are processes of one replica,
/// so they always run the same build.
#[derive(Serialize, Deserialize)]
enum NodeIndexRequest {
    Lookup(ReplicaQueryRequest),
    /// Entries whose source refused a single-page lease (see
    /// `ReplicaIndex::forget`); acknowledged with an empty `Locations`.
    Forget(Vec<(SequenceHash, ReplicaSourceKey)>),
}

/// `rank` (default): every TP rank subscribes to the replica events and keeps
/// its own directory. `node`: only global TP rank 0 does, and serves every
/// other process of its replica over a local socket.
pub const HICACHE_G2_INDEX_MODE_ENV: &str = "DYN_HICACHE_G2_INDEX_MODE";
/// Bounds one lookup round trip to the owner. A healthy one takes tens of
/// microseconds.
const OWNER_IO_TIMEOUT: Duration = Duration::from_millis(200);
/// After a failed round trip, lookups miss without trying the owner for this
/// long, so a dead or wedged owner does not cost every lookup a timeout.
const OWNER_RETRY_BACKOFF: Duration = Duration::from_secs(1);
/// Frames larger than this are malformed; lookups are a few thousand keys.
const MAX_FRAME_BYTES: usize = 64 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HiCacheIndexMode {
    PerRank,
    PerNode,
}

fn parse_index_mode(value: Option<&str>) -> Result<HiCacheIndexMode> {
    match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        None | Some("") | Some("rank") => Ok(HiCacheIndexMode::PerRank),
        Some("node") => Ok(HiCacheIndexMode::PerNode),
        Some(other) => {
            anyhow::bail!("{HICACHE_G2_INDEX_MODE_ENV} must be `rank` or `node`; got {other}")
        }
    }
}

pub fn index_mode_from_env() -> Result<HiCacheIndexMode> {
    parse_index_mode(std::env::var(HICACHE_G2_INDEX_MODE_ENV).ok().as_deref())
}

/// Abstract socket names live in the network namespace, which the ranks of one
/// pod share; the kernel drops the name when the owner exits, so there is no
/// stale file to clean up and a second live owner cannot steal it.
pub fn owner_socket_name(replica_id: u64) -> String {
    format!("dynamo-g2pb-index-{replica_id:016x}")
}

fn socket_addr(name: &str) -> Result<SocketAddr> {
    SocketAddr::from_abstract_name(name.as_bytes())
        .with_context(|| format!("invalid abstract socket name {name}"))
}

/// Serve `index` to the other ranks of this replica. Returns `Ok(false)` when
/// another live process already owns `name`; the caller then keeps serving
/// only itself.
pub fn serve_node_index(index: Arc<ReplicaIndex>, name: &str) -> Result<bool> {
    let listener = match UnixListener::bind_addr(&socket_addr(name)?) {
        Ok(listener) => listener,
        Err(error) if error.kind() == std::io::ErrorKind::AddrInUse => return Ok(false),
        Err(error) => return Err(error).with_context(|| format!("failed to bind {name}")),
    };
    listener.set_nonblocking(true)?;
    let listener = tokio::net::UnixListener::from_std(listener)?;
    tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((stream, _)) => {
                    tokio::spawn(serve_connection(stream, Arc::clone(&index)));
                }
                Err(error) => {
                    tracing::warn!(%error, "HiCache G2 node index accept failed");
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
        }
    });
    Ok(true)
}

async fn serve_connection(mut stream: tokio::net::UnixStream, index: Arc<ReplicaIndex>) {
    loop {
        let Ok(len) = stream.read_u32_le().await else {
            return;
        };
        let len = len as usize;
        if len > MAX_FRAME_BYTES {
            return;
        }
        let mut frame = vec![0; len];
        if stream.read_exact(&mut frame).await.is_err() {
            return;
        }
        let response = match rmp_serde::from_slice::<NodeIndexRequest>(&frame) {
            Ok(NodeIndexRequest::Lookup(request)) => ReplicaQueryResponse::Locations(
                index.batch_locations(&request.keys, Some(request.block_size)),
            ),
            Ok(NodeIndexRequest::Forget(entries)) => {
                for (key, source) in entries {
                    index.forget(key, source);
                }
                ReplicaQueryResponse::Locations(Vec::new())
            }
            Err(error) => ReplicaQueryResponse::Error(error.to_string()),
        };
        let Ok(body) = rmp_serde::to_vec_named(&response) else {
            return;
        };
        if stream.write_u32_le(body.len() as u32).await.is_err()
            || stream.write_all(&body).await.is_err()
        {
            return;
        }
    }
}

/// A non-owner rank's view of the node directory.
pub struct NodeIndexClient {
    name: String,
    // ponytail: one connection serializes this rank's lookups; a small pool if
    // lookup concurrency per rank ever matters.
    state: Mutex<ClientState>,
}

#[derive(Default)]
struct ClientState {
    stream: Option<UnixStream>,
    retry_after: Option<Instant>,
}

impl NodeIndexClient {
    pub fn new(name: String) -> Self {
        Self {
            name,
            state: Mutex::default(),
        }
    }

    /// The owner's raw `batch_locations`. Errs (and backs off) when the owner
    /// is unreachable, slow, or answers garbage; callers treat that as a miss.
    pub fn batch_locations(
        &self,
        keys: &[SequenceHash],
        block_size: u32,
    ) -> Result<Vec<Vec<ReplicaLocation>>> {
        let locations = self.request(&NodeIndexRequest::Lookup(ReplicaQueryRequest {
            keys: keys.to_vec(),
            block_size,
        }))?;
        anyhow::ensure!(
            locations.len() == keys.len(),
            "node index answered {} keys for {}",
            locations.len(),
            keys.len()
        );
        Ok(locations)
    }

    /// Have the owner drop entries whose source refused to lease them, as a
    /// rank with its own directory does. Best effort, with the same backoff
    /// as lookups.
    pub fn forget(&self, entries: &[(SequenceHash, ReplicaSourceKey)]) -> Result<()> {
        self.request(&NodeIndexRequest::Forget(entries.to_vec()))
            .map(|_| ())
    }

    fn request(&self, request: &NodeIndexRequest) -> Result<Vec<Vec<ReplicaLocation>>> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("node index client lock poisoned"))?;
        if state
            .retry_after
            .is_some_and(|until| Instant::now() < until)
        {
            anyhow::bail!("HiCache G2 node index owner is backing off");
        }
        let result = self.round_trip(&mut state.stream, request);
        match &result {
            Ok(_) => state.retry_after = None,
            Err(_) => {
                state.stream = None;
                state.retry_after = Some(Instant::now() + OWNER_RETRY_BACKOFF);
            }
        }
        result
    }

    fn round_trip(
        &self,
        stream: &mut Option<UnixStream>,
        request: &NodeIndexRequest,
    ) -> Result<Vec<Vec<ReplicaLocation>>> {
        if stream.is_none() {
            let connected =
                UnixStream::connect_addr(&socket_addr(&self.name)?).with_context(|| {
                    format!("HiCache G2 node index owner {} unreachable", self.name)
                })?;
            connected.set_read_timeout(Some(OWNER_IO_TIMEOUT))?;
            connected.set_write_timeout(Some(OWNER_IO_TIMEOUT))?;
            *stream = Some(connected);
        }
        let stream = stream.as_mut().expect("connected above");
        let body = rmp_serde::to_vec_named(request)?;
        stream.write_all(&(body.len() as u32).to_le_bytes())?;
        stream.write_all(&body)?;
        let mut len = [0; 4];
        stream.read_exact(&mut len)?;
        let len = u32::from_le_bytes(len) as usize;
        anyhow::ensure!(len <= MAX_FRAME_BYTES, "node index frame of {len} bytes");
        let mut frame = vec![0; len];
        stream.read_exact(&mut frame)?;
        match rmp_serde::from_slice(&frame)? {
            ReplicaQueryResponse::Locations(locations) => Ok(locations),
            ReplicaQueryResponse::Error(message) => anyhow::bail!(message),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::peer_index::{
        ReplicaBlock, ReplicaEvent, ReplicaEventKind, ReplicaSource, ReplicaSourceKey,
    };

    fn unique_name(test: &str) -> String {
        format!("dynamo-g2pb-test-{test}-{}", uuid::Uuid::new_v4())
    }

    fn index_with(key: SequenceHash) -> Arc<ReplicaIndex> {
        let index = Arc::new(ReplicaIndex::new());
        assert!(index.apply(ReplicaEvent {
            source: ReplicaSource {
                key: ReplicaSourceKey {
                    instance_id: 7,
                    tp_rank: 3,
                },
                replica_id: Some(70),
                tp_world_size: 8,
                incarnation: uuid::Uuid::from_u128(1),
                block_size: 64,
                row: Some("row04".to_string()),
            },
            revision: 1,
            kind: ReplicaEventKind::Upsert(vec![ReplicaBlock {
                key,
                size_bytes: 4096,
                checksum: Some(5),
            }]),
        }));
        index
    }

    #[test]
    fn index_mode_defaults_to_per_rank() {
        assert_eq!(parse_index_mode(None).unwrap(), HiCacheIndexMode::PerRank);
        assert_eq!(
            parse_index_mode(Some(" ")).unwrap(),
            HiCacheIndexMode::PerRank
        );
        assert_eq!(
            parse_index_mode(Some("rank")).unwrap(),
            HiCacheIndexMode::PerRank
        );
        assert_eq!(
            parse_index_mode(Some(" Node ")).unwrap(),
            HiCacheIndexMode::PerNode
        );
        assert!(parse_index_mode(Some("pod")).is_err());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn non_owner_reads_the_owners_directory() {
        let name = unique_name("read");
        let index = index_with(11);
        assert!(serve_node_index(Arc::clone(&index), &name).unwrap());
        let client = NodeIndexClient::new(name);
        let locations = tokio::task::spawn_blocking(move || {
            // Same answer as a local lookup, in order, with the block-size filter.
            let first = client.batch_locations(&[12, 11], 64).unwrap();
            let filtered = client.batch_locations(&[11], 32).unwrap();
            (first, filtered)
        })
        .await
        .unwrap();
        assert_eq!(locations.0, index.batch_locations(&[12, 11], Some(64)));
        assert_eq!(locations.0[1][0].source.key.tp_rank, 3);
        assert!(locations.1[0].is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn non_owner_refusal_removes_the_entry_from_the_owners_directory() {
        let name = unique_name("forget");
        let index = index_with(11);
        assert!(serve_node_index(Arc::clone(&index), &name).unwrap());
        let client = NodeIndexClient::new(name);
        let source = ReplicaSourceKey {
            instance_id: 7,
            tp_rank: 3,
        };
        let after = tokio::task::spawn_blocking(move || {
            client.forget(&[(11, source)]).unwrap();
            client.batch_locations(&[11], 64).unwrap()
        })
        .await
        .unwrap();
        assert!(after[0].is_empty());
        assert!(index.batch_locations(&[11], Some(64))[0].is_empty());
        assert!(index.counters().contains(&("index_refused_entries", 1)));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn second_owner_for_the_same_replica_is_refused() {
        let name = unique_name("dup");
        assert!(serve_node_index(index_with(11), &name).unwrap());
        assert!(!serve_node_index(index_with(12), &name).unwrap());
    }

    #[test]
    fn missing_owner_errs_quickly_then_backs_off() {
        let client = NodeIndexClient::new(unique_name("missing"));
        let started = Instant::now();
        let error = client.batch_locations(&[11], 64).unwrap_err();
        assert!(error.to_string().contains("unreachable"), "{error}");
        let error = client.batch_locations(&[11], 64).unwrap_err();
        assert!(error.to_string().contains("backing off"), "{error}");
        assert!(started.elapsed() < OWNER_IO_TIMEOUT);
    }

    #[test]
    fn wedged_owner_times_out_instead_of_hanging() {
        let name = unique_name("wedged");
        // Accepts connections but never answers.
        let _listener = UnixListener::bind_addr(&socket_addr(&name).unwrap()).unwrap();
        let client = NodeIndexClient::new(name);
        let started = Instant::now();
        assert!(client.batch_locations(&[11], 64).is_err());
        let elapsed = started.elapsed();
        assert!(
            elapsed >= OWNER_IO_TIMEOUT && elapsed < OWNER_IO_TIMEOUT * 5,
            "{elapsed:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn client_recovers_after_owner_restarts() {
        let name = unique_name("restart");
        let client = Arc::new(NodeIndexClient::new(name.clone()));
        let probe = Arc::clone(&client);
        assert!(
            tokio::task::spawn_blocking(move || probe.batch_locations(&[11], 64))
                .await
                .unwrap()
                .is_err()
        );
        assert!(serve_node_index(index_with(11), &name).unwrap());
        tokio::time::sleep(OWNER_RETRY_BACKOFF).await;
        let locations = tokio::task::spawn_blocking(move || client.batch_locations(&[11], 64))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(locations[0].len(), 1);
    }
}
