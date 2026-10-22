// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Publish a replica's index deltas through its node-index owner.
//!
//! In node index mode one process per replica subscribes; this does the same
//! for publishing. Every process still numbers its own deltas (each TP rank is
//! its own source with its own revisions), but instead of publishing them to
//! NATS it hands each finished [`ReplicaEvent`] to the owner over an abstract
//! Unix socket. The owner merges what arrives in one batch window, its own
//! events included, into a single compact-encoded batch message, so NATS
//! carries one message per replica per window instead of one per process.
//! Readers apply the batch's events one by one, exactly as if each had arrived
//! alone, so revision ordering per source is unchanged.
//!
//! The index is advisory: an event the relay cannot deliver (owner not up yet,
//! socket error, full queue) is dropped and counted, as a failed NATS publish
//! would be.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
use dynamo_runtime::component::Component;
use dynamo_runtime::transports::event_plane::EventPublisher;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::mpsc;

use crate::peer_index::{
    ReplicaEvent, ReplicaEventBatch, SERVING_G2_REPLICA_BATCH_TOPIC, encode_compact_replica_batch,
};

/// `1`/`true`: in node index mode, the other processes of a replica publish
/// through the index owner. Default off.
pub const HICACHE_G2_NODE_PUBLISH_ENV: &str = "DYN_HICACHE_G2_NODE_PUBLISH";
/// How long the owner collects events before publishing a batch.
pub const HICACHE_G2_NODE_PUBLISH_BATCH_MS_ENV: &str = "DYN_HICACHE_G2_NODE_PUBLISH_BATCH_MS";
const DEFAULT_BATCH_WINDOW: Duration = Duration::from_millis(50);
/// A batch is published early once it holds this many blocks (~1.5 MiB in v1).
const MAX_BATCH_BLOCKS: usize = 32 * 1024;
/// Events queued in one process for the relay; past this they are dropped.
const RELAY_QUEUE_CAPACITY: usize = 4096;
/// After a failed connect or write, events are dropped for this long before
/// the client tries the owner again.
const RECONNECT_BACKOFF: Duration = Duration::from_secs(1);
/// Frames larger than this are malformed.
const MAX_FRAME_BYTES: usize = 64 << 20;

#[derive(Debug, Default)]
pub struct NodeRelayStats {
    /// Events this process handed to the relay.
    pub relayed_events: AtomicU64,
    /// Events dropped: queue full, owner unreachable, or a socket error.
    pub relay_dropped_events: AtomicU64,
    /// Owner only: batches published and the events they carried.
    pub relay_published_batches: AtomicU64,
    pub relay_published_events: AtomicU64,
}

impl NodeRelayStats {
    fn drop_events(&self, count: usize) {
        self.relay_dropped_events
            .fetch_add(count as u64, Ordering::Relaxed);
    }
}

pub fn node_publish_from_env() -> Result<bool> {
    parse_bool(
        HICACHE_G2_NODE_PUBLISH_ENV,
        std::env::var(HICACHE_G2_NODE_PUBLISH_ENV).ok().as_deref(),
    )
}

fn parse_bool(name: &str, value: Option<&str>) -> Result<bool> {
    match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        None | Some("") | Some("0" | "false" | "no" | "off") => Ok(false),
        Some("1" | "true" | "yes" | "on") => Ok(true),
        Some(other) => anyhow::bail!("{name} must be a boolean; got {other}"),
    }
}

fn batch_window_from_env() -> Result<Duration> {
    match std::env::var(HICACHE_G2_NODE_PUBLISH_BATCH_MS_ENV) {
        Ok(value) => {
            let ms: u64 = value.trim().parse().with_context(|| {
                format!("{HICACHE_G2_NODE_PUBLISH_BATCH_MS_ENV} must be milliseconds; got {value}")
            })?;
            anyhow::ensure!(
                ms > 0,
                "{HICACHE_G2_NODE_PUBLISH_BATCH_MS_ENV} must be positive"
            );
            Ok(Duration::from_millis(ms))
        }
        Err(_) => Ok(DEFAULT_BATCH_WINDOW),
    }
}

pub fn relay_socket_name(replica_id: u64) -> String {
    format!("dynamo-g2pb-relay-{replica_id:016x}")
}

fn socket_addr(name: &str) -> Result<std::os::unix::net::SocketAddr> {
    use std::os::linux::net::SocketAddrExt;
    std::os::unix::net::SocketAddr::from_abstract_name(name.as_bytes())
        .with_context(|| format!("invalid abstract socket name {name}"))
}

/// Where a [`crate::peer_index::ReplicaPublisher`] sends finished events when
/// it does not publish them to NATS itself.
#[derive(Clone)]
pub struct RelaySender {
    tx: mpsc::Sender<ReplicaEvent>,
    stats: Arc<NodeRelayStats>,
}

impl RelaySender {
    /// Never waits: the publisher's batching loop must not stall on the relay.
    pub fn send(&self, event: ReplicaEvent) {
        match self.tx.try_send(event) {
            Ok(()) => {
                self.stats.relayed_events.fetch_add(1, Ordering::Relaxed);
            }
            Err(_) => self.stats.drop_events(1),
        }
    }
}

/// Owner side: bind the relay socket, accept the other processes' events, and
/// publish everything (its own events included, through the returned sender)
/// as one batch per window. `Ok(None)` when another live process already owns
/// `name`; the caller then publishes directly.
pub async fn start_node_relay_owner(
    component: &Component,
    name: &str,
    stats: Arc<NodeRelayStats>,
) -> Result<Option<RelaySender>> {
    let listener = match std::os::unix::net::UnixListener::bind_addr(&socket_addr(name)?) {
        Ok(listener) => listener,
        Err(error) if error.kind() == std::io::ErrorKind::AddrInUse => return Ok(None),
        Err(error) => return Err(error).with_context(|| format!("failed to bind {name}")),
    };
    listener.set_nonblocking(true)?;
    let listener = tokio::net::UnixListener::from_std(listener)?;
    let publisher = EventPublisher::for_component(component, SERVING_G2_REPLICA_BATCH_TOPIC)
        .await
        .context("failed to create replica batch publisher")?;
    let window = batch_window_from_env()?;
    let (tx, rx) = mpsc::channel(RELAY_QUEUE_CAPACITY);

    let accept_tx = tx.clone();
    let accept_stats = Arc::clone(&stats);
    tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((stream, _)) => {
                    tokio::spawn(read_relayed_events(
                        stream,
                        accept_tx.clone(),
                        Arc::clone(&accept_stats),
                    ));
                }
                Err(error) => {
                    tracing::warn!(%error, "HiCache G2 node relay accept failed");
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
        }
    });
    tokio::spawn(publish_batches(
        rx,
        BatchSink::Nats(Box::new(publisher)),
        window,
        Arc::clone(&stats),
    ));
    Ok(Some(RelaySender { tx, stats }))
}

async fn read_relayed_events(
    mut stream: tokio::net::UnixStream,
    tx: mpsc::Sender<ReplicaEvent>,
    stats: Arc<NodeRelayStats>,
) {
    loop {
        let Ok(len) = stream.read_u32_le().await else {
            return;
        };
        let len = len as usize;
        if len > MAX_FRAME_BYTES {
            tracing::warn!(len, "HiCache G2 node relay frame too large; closing");
            return;
        }
        let mut frame = vec![0; len];
        if stream.read_exact(&mut frame).await.is_err() {
            return;
        }
        match rmp_serde::from_slice::<ReplicaEvent>(&frame) {
            // The owner's queue is shared with its own events; a full queue
            // drops, it never back-pressures the sending rank.
            Ok(event) => {
                if tx.try_send(event).is_err() {
                    stats.drop_events(1);
                }
            }
            Err(error) => {
                tracing::warn!(%error, "malformed HiCache G2 node relay frame; closing");
                return;
            }
        }
    }
}

enum BatchSink {
    Nats(Box<EventPublisher>),
    #[cfg(test)]
    Test(mpsc::UnboundedSender<ReplicaEventBatch>),
}

impl BatchSink {
    async fn publish(&self, batch: &ReplicaEventBatch) -> Result<()> {
        match self {
            BatchSink::Nats(publisher) => {
                publisher
                    .publish_bytes(encode_compact_replica_batch(&batch.events)?)
                    .await
            }
            #[cfg(test)]
            BatchSink::Test(tx) => {
                tx.send(batch.clone()).ok();
                Ok(())
            }
        }
    }
}

fn event_blocks(event: &ReplicaEvent) -> usize {
    use crate::peer_index::ReplicaEventKind;
    match &event.kind {
        ReplicaEventKind::Upsert(blocks) => blocks.len(),
        ReplicaEventKind::Remove(keys) => keys.len(),
        ReplicaEventKind::RemoveSource => 1,
    }
}

async fn publish_batches(
    mut rx: mpsc::Receiver<ReplicaEvent>,
    sink: BatchSink,
    window: Duration,
    stats: Arc<NodeRelayStats>,
) {
    while let Some(first) = rx.recv().await {
        let mut blocks = event_blocks(&first);
        let mut events = vec![first];
        let deadline = tokio::time::sleep(window);
        tokio::pin!(deadline);
        while blocks < MAX_BATCH_BLOCKS {
            tokio::select! {
                _ = &mut deadline => break,
                next = rx.recv() => match next {
                    Some(event) => {
                        blocks += event_blocks(&event);
                        events.push(event);
                    }
                    None => break,
                }
            }
        }
        let count = events.len();
        let batch = ReplicaEventBatch { events };
        match sink.publish(&batch).await {
            Ok(()) => {
                stats
                    .relay_published_batches
                    .fetch_add(1, Ordering::Relaxed);
                stats
                    .relay_published_events
                    .fetch_add(count as u64, Ordering::Relaxed);
            }
            Err(error) => {
                stats.drop_events(count);
                tracing::warn!(%error, events = count, "failed to publish replica event batch");
            }
        }
    }
}

/// Non-owner side: a sender whose events a background task writes to the
/// owner's relay socket, reconnecting after [`RECONNECT_BACKOFF`].
pub fn connect_node_relay(name: String, stats: Arc<NodeRelayStats>) -> RelaySender {
    let (tx, mut rx) = mpsc::channel::<ReplicaEvent>(RELAY_QUEUE_CAPACITY);
    let task_stats = Arc::clone(&stats);
    tokio::spawn(async move {
        let mut stream: Option<tokio::net::UnixStream> = None;
        let mut retry_at = tokio::time::Instant::now();
        while let Some(event) = rx.recv().await {
            if stream.is_none() {
                if tokio::time::Instant::now() < retry_at {
                    task_stats.drop_events(1);
                    continue;
                }
                match connect(&name).await {
                    Ok(connected) => stream = Some(connected),
                    Err(error) => {
                        tracing::debug!(%error, socket = %name, "HiCache G2 node relay owner unreachable");
                        retry_at = tokio::time::Instant::now() + RECONNECT_BACKOFF;
                        task_stats.drop_events(1);
                        continue;
                    }
                }
            }
            let Ok(body) = rmp_serde::to_vec(&event) else {
                task_stats.drop_events(1);
                continue;
            };
            let connected = stream.as_mut().expect("connected above");
            let written = async {
                connected.write_u32_le(body.len() as u32).await?;
                connected.write_all(&body).await
            };
            if let Err(error) = written.await {
                tracing::debug!(%error, socket = %name, "HiCache G2 node relay write failed");
                stream = None;
                retry_at = tokio::time::Instant::now() + RECONNECT_BACKOFF;
                task_stats.drop_events(1);
            }
        }
    });
    RelaySender { tx, stats }
}

async fn connect(name: &str) -> Result<tokio::net::UnixStream> {
    let std_stream = std::os::unix::net::UnixStream::connect_addr(&socket_addr(name)?)
        .with_context(|| format!("failed to connect to {name}"))?;
    std_stream.set_nonblocking(true)?;
    Ok(tokio::net::UnixStream::from_std(std_stream)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::peer_index::{ReplicaBlock, ReplicaEventKind, ReplicaSource, ReplicaSourceKey};

    fn unique_name(test: &str) -> String {
        format!("dynamo-g2pb-relay-test-{test}-{}", std::process::id())
    }

    fn event(tp_rank: u32, revision: u64, key: u64) -> ReplicaEvent {
        ReplicaEvent {
            source: ReplicaSource {
                key: ReplicaSourceKey {
                    instance_id: 1000 + u64::from(tp_rank),
                    tp_rank,
                },
                replica_id: Some(7),
                tp_world_size: 8,
                incarnation: uuid::Uuid::nil(),
                block_size: 64,
                row: None,
            },
            revision,
            kind: ReplicaEventKind::Upsert(vec![ReplicaBlock {
                key,
                size_bytes: 1,
                checksum: Some(key),
            }]),
        }
    }

    /// Owner side without NATS: the same accept/batch tasks, publishing into a
    /// channel.
    fn start_test_owner(
        name: &str,
        window: Duration,
    ) -> (RelaySender, mpsc::UnboundedReceiver<ReplicaEventBatch>) {
        let stats = Arc::new(NodeRelayStats::default());
        let listener =
            std::os::unix::net::UnixListener::bind_addr(&socket_addr(name).unwrap()).unwrap();
        listener.set_nonblocking(true).unwrap();
        let listener = tokio::net::UnixListener::from_std(listener).unwrap();
        let (tx, rx) = mpsc::channel(RELAY_QUEUE_CAPACITY);
        let accept_tx = tx.clone();
        let accept_stats = Arc::clone(&stats);
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                tokio::spawn(read_relayed_events(
                    stream,
                    accept_tx.clone(),
                    Arc::clone(&accept_stats),
                ));
            }
        });
        let (batch_tx, batch_rx) = mpsc::unbounded_channel();
        tokio::spawn(publish_batches(
            rx,
            BatchSink::Test(batch_tx),
            window,
            Arc::clone(&stats),
        ));
        (RelaySender { tx, stats }, batch_rx)
    }

    #[tokio::test]
    async fn owner_merges_every_rank_into_one_batch_in_per_source_order() {
        let name = unique_name("merge");
        let (owner, mut batches) = start_test_owner(&name, Duration::from_millis(200));
        let clients = (1..4u32)
            .map(|_| connect_node_relay(name.clone(), Arc::new(NodeRelayStats::default())))
            .collect::<Vec<_>>();
        owner.send(event(0, 1, 100));
        for revision in 1..=3u64 {
            for (index, client) in clients.iter().enumerate() {
                client.send(event(
                    index as u32 + 1,
                    revision,
                    revision * 10 + index as u64,
                ));
            }
        }
        let batch = tokio::time::timeout(Duration::from_secs(5), batches.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            batch.events.len(),
            10,
            "one batch carries every rank's events"
        );
        for tp_rank in 1..4u32 {
            let revisions = batch
                .events
                .iter()
                .filter(|event| event.source.key.tp_rank == tp_rank)
                .map(|event| event.revision)
                .collect::<Vec<_>>();
            assert_eq!(revisions, vec![1, 2, 3], "rank {tp_rank} keeps its order");
        }
    }

    #[tokio::test]
    async fn unreachable_owner_drops_and_counts_then_recovers() {
        let name = unique_name("late-owner");
        let stats = Arc::new(NodeRelayStats::default());
        let client = connect_node_relay(name.clone(), Arc::clone(&stats));
        client.send(event(1, 1, 1));
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(stats.relay_dropped_events.load(Ordering::Relaxed), 1);

        let (_owner, mut batches) = start_test_owner(&name, Duration::from_millis(20));
        tokio::time::sleep(RECONNECT_BACKOFF + Duration::from_millis(100)).await;
        client.send(event(1, 2, 2));
        let batch = tokio::time::timeout(Duration::from_secs(5), batches.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(batch.events[0].revision, 2);
    }

    #[test]
    fn node_publish_flag_parses() {
        assert!(!parse_bool(HICACHE_G2_NODE_PUBLISH_ENV, None).unwrap());
        assert!(parse_bool(HICACHE_G2_NODE_PUBLISH_ENV, Some(" ON ")).unwrap());
        assert!(!parse_bool(HICACHE_G2_NODE_PUBLISH_ENV, Some("0")).unwrap());
        assert!(parse_bool(HICACHE_G2_NODE_PUBLISH_ENV, Some("maybe")).is_err());
    }
}
