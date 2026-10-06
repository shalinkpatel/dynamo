use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Wall-clock bound on one release attempt. A release is fire-and-forget for
/// the caller, so this only bounds how long a retry waits on a stalled source.
const LEASE_RELEASE_ATTEMPT_TIMEOUT: Duration = Duration::from_secs(2);
const LEASE_RELEASE_INITIAL_BACKOFF: Duration = Duration::from_millis(100);
const LEASE_RELEASE_MAX_BACKOFF: Duration = Duration::from_secs(5);

pub fn unix_time_ms() -> Result<u64> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?;
    u64::try_from(elapsed.as_millis()).context("Unix timestamp does not fit u64 milliseconds")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct G2pbFetchLeaseId(uuid::Uuid);

impl G2pbFetchLeaseId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }
}

impl Default for G2pbFetchLeaseId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for G2pbFetchLeaseId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

/// Release a source-side fetch lease without blocking the caller.
///
/// The caller has finished with the pages (or failed) and must not wait on the
/// source: a stalled release would otherwise hold the caller's thread after a
/// successful transfer. Each attempt is time-limited and retried with backoff.
/// The loop ends when the release succeeds, when the owner has left the request
/// plane (its lease memory is gone), or when the lease has expired on the
/// source anyway -- the source's own expiry is the backstop, so the retries stop
/// exactly when they can no longer matter. Spawn this on a Tokio runtime.
pub async fn release_fetch_lease_bounded<Fut>(
    lease_id: G2pbFetchLeaseId,
    instance_id: u64,
    lease_expires_at_unix_ms: u64,
    context: &'static str,
    owner_present: impl Fn() -> bool,
    release: impl Fn() -> Fut,
) where
    Fut: Future<Output = Result<()>>,
{
    let mut retry_delay = LEASE_RELEASE_INITIAL_BACKOFF;
    loop {
        match tokio::time::timeout(LEASE_RELEASE_ATTEMPT_TIMEOUT, release()).await {
            Ok(Ok(())) => break,
            Ok(Err(error)) => tracing::warn!(
                %lease_id,
                instance_id,
                %error,
                context,
                retry_delay_ms = retry_delay.as_millis(),
                "failed to release fetch lease; retrying"
            ),
            Err(_) => tracing::warn!(
                %lease_id,
                instance_id,
                context,
                retry_delay_ms = retry_delay.as_millis(),
                "timed out releasing fetch lease; retrying"
            ),
        }
        if !owner_present() {
            tracing::info!(
                %lease_id,
                instance_id,
                context,
                "lease owner disappeared; its fetch lease memory no longer exists"
            );
            break;
        }
        if unix_time_ms().is_ok_and(|now| now >= lease_expires_at_unix_ms) {
            tracing::info!(
                %lease_id,
                instance_id,
                context,
                "stopping fetch lease release retries after its source-side expiry"
            );
            break;
        }
        tokio::time::sleep(retry_delay).await;
        retry_delay = (retry_delay * 2).min(LEASE_RELEASE_MAX_BACKOFF);
    }
}

#[cfg(test)]
mod lease_release_tests {
    use super::*;
    use std::cell::Cell;

    #[tokio::test(start_paused = true)]
    async fn retries_failed_and_stalled_releases_until_acknowledged() {
        let attempts = Cell::new(0u32);
        release_fetch_lease_bounded(
            G2pbFetchLeaseId::new(),
            7,
            u64::MAX,
            "test",
            || true,
            || {
                attempts.set(attempts.get() + 1);
                let attempt = attempts.get();
                async move {
                    match attempt {
                        1 => anyhow::bail!("source busy"),
                        2 => std::future::pending().await,
                        _ => Ok(()),
                    }
                }
            },
        )
        .await;
        assert_eq!(attempts.get(), 3);
    }

    #[tokio::test(start_paused = true)]
    async fn stops_when_the_owner_leaves_or_the_lease_expires() {
        for (owner_present, expires_at) in [(false, u64::MAX), (true, 0)] {
            let attempts = Cell::new(0u32);
            release_fetch_lease_bounded(
                G2pbFetchLeaseId::new(),
                7,
                expires_at,
                "test",
                || owner_present,
                || {
                    attempts.set(attempts.get() + 1);
                    async { anyhow::bail!("unreachable peer") }
                },
            )
            .await;
            assert_eq!(attempts.get(), 1);
        }
    }
}
