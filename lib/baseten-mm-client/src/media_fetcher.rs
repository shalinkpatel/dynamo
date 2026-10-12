// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Vendored URL policy from lib/llm/src/preprocessor/media/loader.rs.
// Keep its blocklists and DNS/redirect validation aligned with Dynamo.

use std::collections::HashSet;
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use anyhow::Result;
use ipnet::IpNet;
use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use reqwest::redirect::Policy;

const DEFAULT_HTTP_USER_AGENT: &str = "dynamo-ai/dynamo";
const DEFAULT_HTTP_TIMEOUT: Duration = Duration::from_secs(30);
pub(crate) const MAX_REDIRECTS: usize = 3;

// IP ranges that must never be reachable from a user-controlled URL.
// Source: RFC1918 (private), RFC6598 (CGNAT), RFC5735 (loopback, link-local,
// 0.0.0.0/8), RFC4193 (ULA), RFC4291 (IPv6 loopback / link-local), RFC6890
// (reserved). Link-local 169.254/16 covers the AWS / OpenStack metadata IP.
//
// Keep this list in sync with the Python counterpart
// (components/src/dynamo/common/multimodal/url_validator.py::_BLOCKED_IP_NETWORKS).
static BLOCKED_IP_NETWORKS: LazyLock<Vec<IpNet>> = LazyLock::new(|| {
    [
        "0.0.0.0/8",
        "10.0.0.0/8",
        "100.64.0.0/10",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "172.16.0.0/12",
        "192.0.0.0/24",
        "192.0.2.0/24",
        "192.168.0.0/16",
        "198.18.0.0/15",
        "198.51.100.0/24",
        "203.0.113.0/24",
        "224.0.0.0/4",
        "240.0.0.0/4",
        "255.255.255.255/32",
        "::/128",
        "::1/128",
        "::ffff:0:0/96",
        "fc00::/7",
        "fe80::/10",
        "ff00::/8",
    ]
    .iter()
    .map(|s| s.parse().expect("invalid CIDR in BLOCKED_IP_NETWORKS"))
    .collect()
});

// Hostnames we reject by literal match without any DNS lookup. Defends
// against /etc/hosts tricks or malicious resolvers that alias metadata /
// internal-service names to attacker IPs. Match is case-insensitive.
//
// Keep this list in sync with the Python counterpart
// (components/src/dynamo/common/multimodal/url_validator.py::_BLOCKED_HOSTS).
static BLOCKED_HOSTS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
        "metadata",
        "metadata.google.internal",
        "metadata.goog",
        "kubernetes.default",
        "kubernetes.default.svc",
    ]
    .iter()
    .copied()
    .collect()
});

/// Return `true` if `ip` falls inside any of the blocked ranges.
pub fn is_blocked_ip(ip: &IpAddr) -> bool {
    BLOCKED_IP_NETWORKS.iter().any(|net| net.contains(ip))
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MediaFetcher {
    pub user_agent: String,
    pub allow_direct_ip: bool,
    pub allow_direct_port: bool,
    /// When `false` (default), reject URLs that target blocked locations:
    /// an IP literal in the RFC-range blocklist (`BLOCKED_IP_NETWORKS`),
    /// a hostname in the literal blocklist (`BLOCKED_HOSTS`, e.g.
    /// `localhost` / `metadata.google.internal`), or — in
    /// `check_if_url_allowed_with_dns` — a hostname that DNS-resolves
    /// to a blocked IP. The name reads "IP" but semantically this is a
    /// single "allow internal / on-prem targets" switch that covers
    /// both IP and hostname blocklists together: real on-prem
    /// deployments need both at once (private CIDRs *and* internal
    /// service names), and splitting them would give no useful config
    /// while doubling the footgun surface. **Never** set on anything
    /// public-facing.
    pub allow_private_ips: bool,
    pub allowed_media_domains: Option<HashSet<String>>,
    pub timeout: Option<Duration>,
}

impl Default for MediaFetcher {
    fn default() -> Self {
        Self {
            user_agent: DEFAULT_HTTP_USER_AGENT.to_string(),
            allow_direct_ip: false,
            allow_direct_port: false,
            allow_private_ips: false,
            allowed_media_domains: None,
            timeout: Some(DEFAULT_HTTP_TIMEOUT),
        }
    }
}

impl MediaFetcher {
    /// Build a `MediaFetcher` whose defaults respect the shared
    /// `DYN_MM_ALLOW_INTERNAL` environment variable. Mirrors the Python
    /// `UrlValidationPolicy.from_env()` behavior so both fetch paths
    /// (frontend decode in Rust, backend decode in Python) honor the
    /// same on-prem opt-in flag.
    ///
    /// `DYN_MM_ALLOW_INTERNAL=1` flips `allow_direct_ip`,
    /// `allow_direct_port`, and `allow_private_ips` all to `true` at
    /// once.
    pub fn from_env() -> Self {
        let allow_internal = std::env::var("DYN_MM_ALLOW_INTERNAL").ok().as_deref() == Some("1");
        Self {
            allow_direct_ip: allow_internal,
            allow_direct_port: allow_internal,
            allow_private_ips: allow_internal,
            ..Self::default()
        }
    }
}

impl MediaFetcher {
    pub fn check_if_url_allowed(&self, url: &url::Url) -> Result<()> {
        if !matches!(url.scheme(), "http" | "https" | "data") {
            anyhow::bail!("Only HTTP(S) and data URLs are allowed");
        }

        if url.scheme() == "data" {
            return Ok(());
        }

        let host = url
            .host()
            .ok_or_else(|| anyhow::anyhow!("URL has no host component"))?;

        if !self.allow_direct_ip && !matches!(host, url::Host::Domain(_)) {
            anyhow::bail!("Direct IP access is not allowed");
        }
        if !self.allow_direct_port && url.port().is_some() {
            anyhow::bail!("Direct port access is not allowed");
        }

        // Host-level checks: blocked hostnames and IP literals in blocked
        // ranges. DNS-resolved IPs are checked in `check_if_url_allowed_with_dns`.
        if !self.allow_private_ips {
            let ip_literal = match host {
                url::Host::Domain(domain) => {
                    let lowered = domain.trim_end_matches('.').to_ascii_lowercase();
                    if BLOCKED_HOSTS.contains(lowered.as_str()) {
                        anyhow::bail!("Host '{domain}' is blocked (resolves to internal service)");
                    }
                    None
                }
                url::Host::Ipv4(ip) => Some(IpAddr::V4(ip)),
                url::Host::Ipv6(ip) => Some(IpAddr::V6(ip)),
            };
            if let Some(ip) = ip_literal
                && is_blocked_ip(&ip)
            {
                anyhow::bail!("IP literal '{ip}' is in a blocked range");
            }
        }

        if let Some(allowed_domains) = &self.allowed_media_domains
            && let Some(host_str) = url.host_str()
            && !allowed_domains.contains(host_str)
        {
            anyhow::bail!("Host '{host_str}' is not in the allowed_media_domains list");
        }

        Ok(())
    }

    /// Full policy check: runs `check_if_url_allowed` and, for hostname
    /// URLs, resolves DNS and checks each resulting IP against the blocked ranges.
    pub async fn check_if_url_allowed_with_dns(&self, url: &url::Url) -> Result<()> {
        self.check_if_url_allowed(url)?;

        // Only hostnames need DNS resolution; IP-literal hosts were already
        // checked against the blocklist above.
        if self.allow_private_ips || url.scheme() == "data" {
            return Ok(());
        }
        let Some(url::Host::Domain(host)) = url.host() else {
            return Ok(());
        };

        let port = url.port_or_known_default().unwrap_or(0);
        let iter = tokio::net::lookup_host((host, port))
            .await
            .map_err(|e| anyhow::anyhow!("Could not resolve host '{host}': {e}"))?;
        for sock_addr in iter {
            let ip = sock_addr.ip();
            if is_blocked_ip(&ip) {
                anyhow::bail!("Host '{host}' resolves to blocked IP '{ip}'");
            }
        }
        Ok(())
    }

    /// Pool connections with filtered DNS. MediaResolver validates each URL
    /// and follows redirects itself, including when BDN resolves the target.
    pub(crate) fn build_http_client(&self, proxy: Option<&str>) -> Result<reqwest::Client> {
        // Configuration is trusted; media URLs are still checked on every hop.
        let trusted_proxy_host = proxy
            .map(url::Url::parse)
            .transpose()?
            .and_then(|url| url.host_str().map(str::to_owned));
        let mut builder = reqwest::Client::builder()
            .no_proxy()
            .user_agent(&self.user_agent)
            // The resolver validates DNS on each hop, including through BDN.
            .redirect(Policy::none())
            .dns_resolver(Arc::new(BlocklistResolver {
                allow_private_ips: self.allow_private_ips,
                trusted_proxy_host,
            }));

        if let Some(timeout) = self.timeout {
            builder = builder.timeout(timeout);
        }

        if let Some(proxy) = proxy {
            builder = builder.proxy(reqwest::Proxy::all(proxy)?);
        }
        Ok(builder.build()?)
    }
}

/// DNS resolver that filters out blocked IP ranges before reqwest sees them.
///
/// Attached to the shared `reqwest::Client` via `ClientBuilder::dns_resolver`.
/// reqwest calls this for every hostname it needs to resolve — including
/// redirect targets — so DNS rebinding can't slip a blocked IP past us:
/// Direct destinations never receive blocked addresses. The trusted configured
/// proxy host is exempt; source URLs still pass preflight validation.
struct BlocklistResolver {
    allow_private_ips: bool,
    trusted_proxy_host: Option<String>,
}

impl Resolve for BlocklistResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let host = name.as_str().to_string();
        let allow_private = self.allow_private_ips
            || self
                .trusted_proxy_host
                .as_deref()
                .is_some_and(|proxy| proxy.eq_ignore_ascii_case(&host));
        Box::pin(async move {
            let iter = tokio::net::lookup_host((host.as_str(), 0_u16)).await?;
            let addrs: Vec<SocketAddr> = if allow_private {
                iter.collect()
            } else {
                iter.filter(|sa| !is_blocked_ip(&sa.ip())).collect()
            };
            if addrs.is_empty() {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::AddrNotAvailable,
                    format!("no non-blocked addresses for host '{host}'"),
                ))
                    as Box<dyn std::error::Error + Send + Sync>);
            }
            Ok(Box::new(addrs.into_iter()) as Addrs)
        })
    }
}
