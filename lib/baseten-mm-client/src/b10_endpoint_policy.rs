// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Result, bail};

use crate::HttpEncoderConfig;

pub(crate) fn is_production(etcd_endpoints: Option<&str>) -> bool {
    etcd_endpoints.is_some_and(|endpoints| {
        endpoints.contains(".svc.localnet") || endpoints.contains("headless.org-")
    })
}

#[derive(Debug)]
pub struct StableEncoderEndpointRequired;

impl std::fmt::Display for StableEncoderEndpointRequired {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(
            "Multimodal encoder and cache URLs must target production or a stable environment. \
             Deployment-pinned (/deployment/<revision>/...) and development URLs are not permitted. \
             Use /environments/production/sync/<endpoint>, /environments/<environment>/sync/<endpoint>, \
             or a stable environment hostname with /sync/<endpoint>. \
             For an emergency bypass, set DYNAMO_ALLOW_UNSTABLE_ENCODER_URL=1; this is not recommended.",
        )
    }
}

impl std::error::Error for StableEncoderEndpointRequired {}

pub(crate) fn validate_endpoints(
    config: &HttpEncoderConfig,
    allow_unstable: Option<&str>,
) -> Result<()> {
    let allow_unstable =
        allow_unstable.is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"));
    // Validate the final encoder even when the HTTP client connects to a cache/proxy.
    // Every cache hop is also a configured dependency and must remain stable.
    for endpoint in std::iter::once(&config.url).chain(&config.cache_urls) {
        // Do not include URLs in errors: they can contain credentials or signed queries.
        let parsed = url::Url::parse(endpoint)
            .map_err(|_| anyhow::anyhow!("Invalid encoder or cache endpoint URL"))?;
        if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
            bail!("Encoder and cache endpoints must be HTTP(S) URLs");
        }
        let path = percent_encoding::percent_decode_str(parsed.path()).decode_utf8_lossy();
        let mut segments = path.split('/').filter(|segment| !segment.is_empty());
        let first = segments.next().unwrap_or_default();
        let unstable = ["deployment", "deployments", "development"]
            .iter()
            .any(|route| first.eq_ignore_ascii_case(route))
            || (first.eq_ignore_ascii_case("environments")
                && segments
                    .next()
                    .is_some_and(|name| name.eq_ignore_ascii_case("development")));
        if unstable {
            if !allow_unstable {
                return Err(StableEncoderEndpointRequired.into());
            }
            tracing::warn!(
                "DYNAMO_ALLOW_UNSTABLE_ENCODER_URL bypasses the stable multimodal endpoint requirement"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn b10_endpoint_policy() -> Result<()> {
        for endpoint in [
            "https://model-q047oly3.api.baseten.co/environments/production/sync/encode",
            "https://model-q047oly3.api.baseten.co/environments/staging/sync/encode",
            "https://model-q047oly3-apac.api.baseten.co/sync/encode",
            "http://localhost:8000/encode",
        ] {
            validate_endpoints(&HttpEncoderConfig::new(endpoint), None)?;
        }
        for path in [
            "/deployment/qj7lpml/sync/encode",
            "/development/sync/encode",
            "/environments/development/sync/encode",
            "/%64eployment/qj7lpml/sync/encode",
        ] {
            let unstable = format!("https://model.api.baseten.co{path}");
            for position in 0..3 {
                let mut config =
                    HttpEncoderConfig::new("https://model-apac.api.baseten.co/sync/encode");
                config.cache_urls = vec![config.url.clone(), config.url.clone()];
                if position == 0 {
                    config.url = unstable.clone();
                } else {
                    config.cache_urls[position - 1] = unstable.clone();
                }
                config.proxy = Some("http://bdn:8080".into());
                for allow in [None, Some("0")] {
                    let error = validate_endpoints(&config, allow).unwrap_err();
                    assert!(
                        error
                            .downcast_ref::<StableEncoderEndpointRequired>()
                            .is_some()
                    );
                }
                for allow in ["1", "TRUE"] {
                    validate_endpoints(&config, Some(allow))?;
                }
            }
        }
        assert!(
            validate_endpoints(&HttpEncoderConfig::new("file:///tmp/encoder"), Some("1")).is_err()
        );
        Ok(())
    }

    #[test]
    fn b10_production_etcd_requires_bdn() -> Result<()> {
        for endpoints in [
            "http://etcd.org.svc.localnet:2379",
            "etcd-dynamo-headless.org-c167caf9787040038b01934dba3089f7:2379",
            "http://localhost:2379,http://etcd-dynamo-headless.org-example:2379",
        ] {
            let error = crate::bdn_proxy(None, None, Some(endpoints)).unwrap_err();
            assert!(
                error
                    .downcast_ref::<crate::ProductionBdnProxyRequired>()
                    .is_some()
            );
            assert!(crate::bdn_proxy(Some("http://bdn:8080"), None, Some(endpoints))?.is_some());
            assert_eq!(crate::bdn_proxy(None, Some("1"), Some(endpoints))?, None);
        }
        assert_eq!(crate::bdn_proxy(None, None, Some("localhost:2379"))?, None);
        Ok(())
    }
}
