// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Jittered exponential reconnect backoff for clients that opt into
//! `retry_on_initial_connect`. async-nats' default (2^n ms, capped at 4 s, no
//! jitter) makes a fleet of clients reconnect in lockstep after a NATS restart.

use std::time::Duration;

const BASE: Duration = Duration::from_millis(100);
const CAP: Duration = Duration::from_secs(30);

/// Equal jitter: half of `min(CAP, BASE * 2^(attempts-1))`, plus a uniform
/// random share of the other half. The first attempt is immediate.
pub(super) fn reconnect_delay(attempts: usize) -> Duration {
    if attempts <= 1 {
        return Duration::ZERO;
    }
    let exp = u32::try_from(attempts - 2).unwrap_or(u32::MAX).min(16);
    let ceiling = BASE.saturating_mul(1 << exp).min(CAP);
    ceiling / 2 + ceiling.mul_f64(rand::random::<f64>() / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn b10_reconnect_delay_grows_jittered_and_capped() {
        assert_eq!(reconnect_delay(1), Duration::ZERO);
        for attempts in 2..40 {
            let exp = (attempts - 2).min(16) as u32;
            let ceiling = BASE.saturating_mul(1 << exp).min(CAP);
            let delay = reconnect_delay(attempts);
            assert!(
                delay >= ceiling / 2 && delay <= ceiling,
                "{attempts}: {delay:?}"
            );
        }
        // Jittered: many clients at the same attempt do not all pick one delay.
        let delays = (0..64)
            .map(|_| reconnect_delay(10))
            .collect::<std::collections::HashSet<_>>();
        assert!(delays.len() > 1);
    }
}
