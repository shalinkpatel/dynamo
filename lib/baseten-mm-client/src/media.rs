// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use anyhow::{Context, Result, bail, ensure};
use base64::{Engine, engine::general_purpose::STANDARD};
use dynamo_multimodal::processor::{MediaMetadata, MmFamilyProcessor};
use image::ImageDecoder;
use serde::{Deserialize, Serialize};

use crate::{Dsv41ImageConfig, Dsv41ImagePlan, Media, MediaFetcher, bdn_proxy};

pub struct MediaResolverConfig {
    /// Explicit BDN proxy; otherwise use BDN_PROXY and DYNAMO_DISABLE_BDN_PROXY.
    pub proxy: Option<String>,
    pub fetcher: MediaFetcher,
    /// Maximum compressed bytes per media item, enforced while downloading.
    pub max_bytes: usize,
    pub max_total_bytes: usize,
    pub max_media_items: usize,
    pub max_image_pixels: u64,
    pub max_concurrent_resolutions: usize,
}

impl Default for MediaResolverConfig {
    fn default() -> Self {
        Self {
            proxy: None,
            fetcher: MediaFetcher::from_env(),
            max_bytes: 64 << 20,
            max_total_bytes: 256 << 20,
            max_media_items: 256,
            max_image_pixels: 64_000_000,
            max_concurrent_resolutions: 64,
        }
    }
}

/// Compressed media for prefill plus metadata for frontend token planning.
#[derive(Clone, Serialize, Deserialize)]
pub struct ResolvedImage {
    pub data_url: String,
    /// Full BLAKE3 of compressed bytes, independent of URL and base64 spelling.
    /// This is media identity, not an encoder's model-specific mm_hash.
    pub content_hash: String,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImageDimensions {
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ResolvedMedia {
    pub media: Media,
    pub content_hash: String,
    pub image: Option<ImageDimensions>,
}

impl ResolvedImage {
    pub fn metadata(&self) -> MediaMetadata {
        MediaMetadata::Image {
            width: self.width,
            height: self.height,
        }
    }

    pub fn num_media_tokens(&self, processor: &dyn MmFamilyProcessor) -> Result<usize> {
        Ok(processor.num_media_tokens(&self.metadata())?)
    }

    pub fn dsv41_plan(&self, config: &Dsv41ImageConfig) -> Result<Dsv41ImagePlan> {
        config.plan(self.width, self.height)
    }
}

/// Checkpoint tokenizer for model-specific media control text. Must preserve
/// special tokens and omit unrelated BOS/EOS tokens.
pub type MediaTokenizer = Arc<dyn Fn(&str) -> Result<Vec<i32>> + Send + Sync>;

#[derive(Clone)]
pub struct MediaResolver {
    client: reqwest::Client,
    fetcher: MediaFetcher,
    max_bytes: usize,
    pub(crate) max_total_bytes: usize,
    pub(crate) max_media_items: usize,
    max_image_pixels: u64,
    slots: Arc<tokio::sync::Semaphore>,
    pub(crate) media_tokenizer: Option<MediaTokenizer>,
}

pub(crate) struct MediaBudget {
    used: AtomicUsize,
    limit: usize,
}

impl MediaBudget {
    pub(crate) fn new(limit: usize) -> Self {
        Self {
            used: AtomicUsize::new(0),
            limit,
        }
    }
    fn reserve(&self, bytes: usize) -> Result<()> {
        self.used
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |used| {
                used.checked_add(bytes).filter(|&total| total <= self.limit)
            })
            .map_err(|_| anyhow::anyhow!("Media exceeds aggregate request byte limit"))?;
        Ok(())
    }
}

impl MediaResolver {
    pub fn new(config: MediaResolverConfig) -> Result<Self> {
        ensure!(config.max_bytes > 0, "Media byte limit must be positive");
        ensure!(
            config.max_total_bytes > 0
                && config.max_media_items > 0
                && config.max_image_pixels > 0
                && config.max_concurrent_resolutions > 0
                && config.max_concurrent_resolutions <= tokio::sync::Semaphore::MAX_PERMITS,
            "Media resource limits must be positive and resolution slots must fit semaphore limits"
        );
        let proxy = match config.proxy.as_deref().filter(|value| !value.is_empty()) {
            Some(proxy) => bdn_proxy(Some(proxy), None, None)?,
            None => bdn_proxy(
                std::env::var("BDN_PROXY").ok().as_deref(),
                std::env::var("DYNAMO_DISABLE_BDN_PROXY").ok().as_deref(),
                None,
            )?,
        };
        Ok(Self {
            client: config.fetcher.build_http_client(proxy.as_deref())?,
            fetcher: config.fetcher,
            max_bytes: config.max_bytes,
            max_total_bytes: config.max_total_bytes,
            max_media_items: config.max_media_items,
            max_image_pixels: config.max_image_pixels,
            media_tokenizer: None,
            slots: Arc::new(tokio::sync::Semaphore::new(
                config.max_concurrent_resolutions,
            )),
        })
    }

    /// Supply a tokenizer without placing it in serializable media plans.
    pub fn with_media_tokenizer(mut self, tokenizer: MediaTokenizer) -> Self {
        self.media_tokenizer = Some(tokenizer);
        self
    }

    /// Resolve HTTP(S), a base64 data URI, or bare base64 to canonical media.
    /// Does not decode pixels or call the vision encoder.
    pub async fn resolve_image(&self, source: &str) -> Result<ResolvedImage> {
        let _slot = self.slots.acquire().await?;
        self.resolve_image_inner(source, None).await
    }

    async fn resolve_image_inner(
        &self,
        source: &str,
        budget: Option<&MediaBudget>,
    ) -> Result<ResolvedImage> {
        let bytes = self.resolve_bytes(source, budget).await?;
        let (mut height, mut width) = dynamo_multimodal::image::decode::dimensions(&bytes)?;
        ensure!(width > 0 && height > 0, "Image dimensions must be positive");
        ensure!(
            (width as u64)
                .checked_mul(height as u64)
                .is_some_and(|pixels| pixels <= self.max_image_pixels),
            "Image exceeds original pixel limit"
        );
        let format = image::guess_format(&bytes)?;
        // Preserve formats supported by the upstream dimension reader even
        // when image does not enable a corresponding pixel decoder (e.g. GIF).
        let orientation = if matches!(
            format,
            image::ImageFormat::Jpeg | image::ImageFormat::Png | image::ImageFormat::WebP
        ) {
            let mut decoder = image::ImageReader::new(std::io::Cursor::new(&bytes))
                .with_guessed_format()?
                .into_decoder()?;
            // WebP encoders may retain the JPEG-style Exif prefix; Pillow accepts
            // both representations. image's orientation helper only accepts TIFF.
            let exif = decoder.exif_metadata()?;
            exif.as_deref()
                .and_then(|bytes| {
                    image::metadata::Orientation::from_exif_chunk(
                        bytes.strip_prefix(b"Exif\0\0").unwrap_or(bytes),
                    )
                })
                .unwrap_or(image::metadata::Orientation::NoTransforms)
        } else {
            image::metadata::Orientation::NoTransforms
        };
        if matches!(
            orientation,
            image::metadata::Orientation::Rotate90
                | image::metadata::Orientation::Rotate270
                | image::metadata::Orientation::Rotate90FlipH
                | image::metadata::Orientation::Rotate270FlipH
        ) {
            std::mem::swap(&mut width, &mut height);
        }
        let mime = format.to_mime_type();
        Ok(ResolvedImage {
            data_url: format!("data:{mime};base64,{}", STANDARD.encode(&bytes)),
            content_hash: blake3::hash(&bytes).to_hex().to_string(),
            width: width.try_into()?,
            height: height.try_into()?,
        })
    }
    async fn resolve_bytes(&self, source: &str, budget: Option<&MediaBudget>) -> Result<Vec<u8>> {
        let bytes = if source.starts_with("http://") || source.starts_with("https://") {
            let mut url = url::Url::parse(source).context("Invalid media URL")?;
            let mut redirects = 0;
            let mut response = loop {
                ensure!(
                    url.username().is_empty() && url.password().is_none(),
                    "Media URLs cannot contain credentials"
                );
                self.fetcher.check_if_url_allowed_with_dns(&url).await?;
                // Presigned query strings are credentials; keep them out of errors.
                let response = self
                    .client
                    .get(url.clone())
                    .send()
                    .await
                    .map_err(|e| e.without_url())?;
                if matches!(response.status().as_u16(), 301 | 302 | 303 | 307 | 308) {
                    ensure!(
                        redirects < crate::media_fetcher::MAX_REDIRECTS,
                        "Too many media redirects"
                    );
                    let location = response
                        .headers()
                        .get(reqwest::header::LOCATION)
                        .context("Media redirect has no Location")?
                        .to_str()?;
                    url = url.join(location).context("Invalid media redirect URL")?;
                    redirects += 1;
                    continue;
                }
                break response.error_for_status().map_err(|e| e.without_url())?;
            };
            if response
                .content_length()
                .is_some_and(|n| n > self.max_bytes as u64)
            {
                bail!("Media exceeds byte limit");
            }
            let mut bytes = Vec::new();
            while let Some(chunk) = response.chunk().await.map_err(|e| e.without_url())? {
                ensure!(
                    chunk.len() <= self.max_bytes - bytes.len(),
                    "Media exceeds byte limit"
                );
                if let Some(budget) = budget {
                    budget.reserve(chunk.len())?;
                }
                bytes.extend_from_slice(&chunk);
            }
            bytes
        } else {
            let encoded = if source.starts_with("data:") {
                let (header, encoded) = source.split_once(',').context("Invalid media data URI")?;
                ensure!(
                    header.ends_with(";base64"),
                    "Media data URI must be base64 encoded"
                );
                encoded
            } else {
                source
            };
            // Reject oversized inputs before allocating the decoded buffer.
            let max_encoded = self.max_bytes.div_ceil(3).saturating_mul(4);
            ensure!(encoded.len() <= max_encoded, "Media exceeds byte limit");
            if let Some(budget) = budget {
                budget.reserve(base64::decoded_len_estimate(encoded.len()))?;
            }
            let bytes = STANDARD.decode(encoded).context("Invalid media base64")?;
            ensure!(bytes.len() <= self.max_bytes, "Media exceeds byte limit");
            bytes
        };
        ensure!(!bytes.is_empty(), "Media is empty");
        Ok(bytes)
    }

    pub async fn resolve(&self, media: Media) -> Result<ResolvedMedia> {
        self.resolve_for_plan(media, None).await
    }

    pub(crate) async fn resolve_for_plan(
        &self,
        mut media: Media,
        budget: Option<&MediaBudget>,
    ) -> Result<ResolvedMedia> {
        let _slot = self.slots.acquire().await?;
        let (data_url, content_hash, image) = if matches!(media, Media::Image { .. }) {
            let resolved = self.resolve_image_inner(media.url()?, budget).await?;
            (
                resolved.data_url,
                resolved.content_hash,
                Some(ImageDimensions {
                    width: resolved.width,
                    height: resolved.height,
                }),
            )
        } else {
            let bytes = self.resolve_bytes(media.url()?, budget).await?;
            // The decoder selects the container from bytes, not the transport MIME.
            let data_url = format!(
                "data:application/octet-stream;base64,{}",
                STANDARD.encode(&bytes)
            );
            (data_url, blake3::hash(&bytes).to_hex().to_string(), None)
        };
        media.set_url(data_url);
        Ok(ResolvedMedia {
            media,
            content_hash,
            image,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::ImageEncoder;

    use crate::TEST_IMAGE_BASE64 as PNG;

    // Minimal TIFF IFD with one SHORT Orientation entry. Both byte orders are
    // accepted by Pillow's EXIF reader and by image's header-only reader.
    fn orientation_exif(orientation: u16, big_endian: bool) -> Vec<u8> {
        if big_endian {
            let mut exif = b"MM\0*\0\0\0\x08\0\x01\x01\x12\0\x03\0\0\0\x01".to_vec();
            exif.extend_from_slice(&orientation.to_be_bytes());
            exif.extend_from_slice(&[0; 6]);
            exif
        } else {
            let mut exif = b"II*\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0".to_vec();
            exif.extend_from_slice(&orientation.to_le_bytes());
            exif.extend_from_slice(&[0; 6]);
            exif
        }
    }

    fn image_with_exif(format: image::ImageFormat, exif: Vec<u8>) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        let pixels = vec![127; 84 * 56 * 3];
        match format {
            image::ImageFormat::Jpeg => {
                let mut encoder = image::codecs::jpeg::JpegEncoder::new(&mut bytes);
                encoder.set_exif_metadata(exif)?;
                encoder.write_image(&pixels, 84, 56, image::ExtendedColorType::Rgb8)?;
            }
            image::ImageFormat::Png => {
                let mut encoder = image::codecs::png::PngEncoder::new(&mut bytes);
                encoder.set_exif_metadata(exif)?;
                encoder.write_image(&pixels, 84, 56, image::ExtendedColorType::Rgb8)?;
            }
            image::ImageFormat::WebP => {
                let mut encoder = image::codecs::webp::WebPEncoder::new_lossless(&mut bytes);
                encoder.set_exif_metadata(exif)?;
                encoder.write_image(&pixels, 84, 56, image::ExtendedColorType::Rgb8)?;
            }
            _ => unreachable!(),
        }
        Ok(bytes)
    }

    #[tokio::test]
    async fn gif_inputs_do_not_require_an_exif_decoder() -> Result<()> {
        let resolver = local_resolver(64 << 10)?;
        let gif = "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7";
        let resolved = resolver.resolve_image(gif).await?;
        assert_eq!((resolved.width, resolved.height), (1, 1));
        assert_eq!(resolved.data_url, format!("data:image/gif;base64,{gif}"));
        assert_eq!(
            resolved.content_hash,
            blake3::hash(&STANDARD.decode(gif)?).to_hex().to_string()
        );
        Ok(())
    }

    #[tokio::test]
    async fn exif_geometry_matches_worker_transpose_and_preserves_payload() -> Result<()> {
        let resolver = local_resolver(64 << 10)?;
        let config = Dsv41ImageConfig {
            vision_patch_size: 14,
            vision_downsample_ratio: 3,
            vision_max_n_token: 1024,
            vision_min_pixels: 295936,
            vision_max_wh_ratio: None,
        };
        for format in [
            image::ImageFormat::Jpeg,
            image::ImageFormat::Png,
            image::ImageFormat::WebP,
        ] {
            for big_endian in [false, true] {
                // Invalid values are ignored, as in Pillow ImageOps.exif_transpose.
                for orientation in 0..=9 {
                    let bytes = image_with_exif(format, orientation_exif(orientation, big_endian))?;
                    let base64 = STANDARD.encode(&bytes);
                    let resolved = resolver.resolve_image(&base64).await?;
                    let rotated = (5..=8).contains(&orientation);
                    let expected = if rotated { (56, 84) } else { (84, 56) };
                    assert_eq!(
                        (resolved.width, resolved.height),
                        expected,
                        "{format:?}, EXIF={orientation}, big_endian={big_endian}"
                    );
                    assert_eq!(
                        resolved.content_hash,
                        blake3::hash(&bytes).to_hex().as_str()
                    );
                    assert_eq!(
                        resolved.data_url,
                        format!("data:{};base64,{base64}", format.to_mime_type())
                    );
                    // DSV includes row separators: the two views have different
                    // lengths, even though their pixel and feature counts agree.
                    assert_eq!(
                        resolved.dsv41_plan(&config)?.num_tokens,
                        if rotated { 194 } else { 189 }
                    );
                }
            }
        }
        // Pillow writes an Exif\0\0 prefix into WebP's EXIF chunk. It must
        // produce the same geometry as the TIFF-only WebP representation.
        let mut exif = b"Exif\0\0".to_vec();
        exif.extend(orientation_exif(6, false));
        let bytes = image_with_exif(image::ImageFormat::WebP, exif)?;
        let resolved = resolver.resolve_image(&STANDARD.encode(bytes)).await?;
        assert_eq!((resolved.width, resolved.height), (56, 84));
        Ok(())
    }

    #[tokio::test]
    async fn malformed_exif_is_ignored_without_decoding_pixels() -> Result<()> {
        let resolver = local_resolver(64 << 10)?;
        let mut bad_offset = orientation_exif(6, false);
        bad_offset[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        for format in [
            image::ImageFormat::Jpeg,
            image::ImageFormat::Png,
            image::ImageFormat::WebP,
        ] {
            for exif in [vec![], b"not a TIFF header".to_vec(), bad_offset.clone()] {
                let bytes = image_with_exif(format, exif)?;
                let resolved = resolver.resolve_image(&STANDARD.encode(bytes)).await?;
                assert_eq!((resolved.width, resolved.height), (84, 56));
            }
        }
        let mut bytes = image_with_exif(image::ImageFormat::Png, orientation_exif(6, false))?;
        let idat = bytes.windows(4).position(|chunk| chunk == b"IDAT").unwrap();
        // Break the compressed pixel stream. Header probing must still succeed.
        bytes[idat + 4] = 0;
        assert!(image::load_from_memory(&bytes).is_err());
        let resolved = resolver.resolve_image(&STANDARD.encode(&bytes)).await?;
        assert_eq!((resolved.width, resolved.height), (56, 84));
        let limited = MediaResolver::new(MediaResolverConfig {
            max_image_pixels: 4703,
            ..Default::default()
        })?;
        assert!(
            limited
                .resolve_image(&STANDARD.encode(&bytes))
                .await
                .is_err()
        );
        Ok(())
    }

    #[tokio::test]
    async fn rejects_large_original_geometry_and_streaming_aggregate_budget() -> Result<()> {
        let resolver = MediaResolver::new(MediaResolverConfig {
            max_image_pixels: 4703,
            ..Default::default()
        })?;
        assert!(
            resolver
                .resolve_image(PNG)
                .await
                .err()
                .unwrap()
                .to_string()
                .contains("original pixel limit")
        );
        let budget = MediaBudget::new(10);
        budget.reserve(6)?;
        assert!(budget.reserve(5).is_err());
        budget.reserve(4)?;
        assert!(budget.reserve(1).is_err());
        Ok(())
    }

    fn local_resolver(max_bytes: usize) -> Result<MediaResolver> {
        Ok(MediaResolver {
            client: MediaFetcher::default().build_http_client(None)?,
            fetcher: MediaFetcher::default(),
            max_bytes,
            max_total_bytes: 256 << 20,
            max_media_items: 256,
            max_image_pixels: 64_000_000,
            slots: Arc::new(tokio::sync::Semaphore::new(64)),
            media_tokenizer: None,
        })
    }

    #[tokio::test]
    async fn base64_and_data_uris_have_identical_content_identity() -> Result<()> {
        let resolver = local_resolver(1024)?;
        let bare = resolver.resolve_image(PNG).await?;
        let mislabeled = resolver
            .resolve_image(&format!("data:image/jpeg;base64,{PNG}"))
            .await?;
        assert_eq!(bare.data_url, format!("data:image/png;base64,{PNG}"));
        assert_eq!(mislabeled.data_url, bare.data_url);
        assert_eq!(mislabeled.content_hash, bare.content_hash);
        assert_eq!(
            bare.content_hash,
            blake3::hash(&STANDARD.decode(PNG)?).to_hex().as_str()
        );
        assert_eq!((bare.width, bare.height), (84, 56));
        assert!(matches!(
            bare.metadata(),
            MediaMetadata::Image {
                width: 84,
                height: 56
            }
        ));
        let restored: ResolvedImage = serde_json::from_str(&serde_json::to_string(&bare)?)?;
        assert_eq!(restored.data_url, bare.data_url);
        let processor = dynamo_multimodal::registry::processor_from_spec(
            r#"{
            "family": "qwen_vl", "image_token_id": 151655,
            "patch_size": 14, "merge_size": 2, "temporal_patch_size": 2,
            "min_pixels": 3136, "max_pixels": 1003520,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711]
        }"#,
        )?;
        assert_eq!(bare.num_media_tokens(processor.as_ref())?, 6);
        let config: Dsv41ImageConfig = serde_json::from_value(serde_json::json!({
            "vision_patch_size": 14, "vision_downsample_ratio": 3,
            "vision_max_n_token": 1024, "vision_min_pixels": 295936,
            "vision_max_wh_ratio": null
        }))?;
        assert_eq!(bare.dsv41_plan(&config)?.num_tokens, 189);
        Ok(())
    }

    #[tokio::test]
    async fn malformed_unsupported_and_oversized_inputs_fail() -> Result<()> {
        let resolver = local_resolver(1024)?;
        for invalid in [
            "",
            "not base64!",
            "dGV4dA==",
            "data:image/png,abc",
            "data:image/png;base64",
            "file:///tmp/image.png",
            "s3://bucket/image.png",
        ] {
            assert!(
                resolver.resolve_image(invalid).await.is_err(),
                "accepted {invalid}"
            );
        }
        let bytes = STANDARD.decode(PNG)?;
        assert!(
            local_resolver(bytes.len() - 1)?
                .resolve_image(PNG)
                .await
                .is_err()
        );
        assert!(
            local_resolver(bytes.len() - 1)?
                .resolve_image(&format!("data:image/png;base64,{PNG}"))
                .await
                .is_err()
        );
        assert!(
            local_resolver(bytes.len())?
                .resolve_image(PNG)
                .await
                .is_ok()
        );
        for url in [
            "http://127.0.0.1/image",
            "http://localhost/image",
            "http://metadata.google.internal/image",
            "http://user:secret@example.com/image",
        ] {
            assert!(resolver.resolve_image(url).await.is_err());
        }
        Ok(())
    }

    #[tokio::test]
    async fn proxy_get_preserves_presigned_query_and_enforces_limits_and_redirects() -> Result<()> {
        use axum::{
            Router,
            body::Body,
            http::{HeaderMap, StatusCode, Uri},
            response::Response,
        };
        use std::time::Duration;

        tokio::time::timeout(Duration::from_secs(10), async {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
            let proxy = format!("http://localhost:{}", listener.local_addr()?.port());
            let app = Router::new().fallback(|method: axum::http::Method, uri: Uri, headers: HeaderMap| async move {
                assert_eq!(method, axum::http::Method::GET);
                assert_eq!(uri.query(), Some("X-Amz-Signature=preserved%2Fsignature&X-Amz-Expires=60"));
                assert_eq!(headers["proxy-authorization"], "Basic YmRuOmR5bmFtby1pbWFnZS1jYWNoZQ==");
                assert!(!headers.contains_key("authorization"));
                assert!(!headers.contains_key("clamshack-cache-put"));
                match uri.path() {
                    "/image" => Response::new(Body::from(STANDARD.decode(PNG).unwrap())),
                    "/large" => Response::new(Body::from(vec![0; 2048])),
                    "/stream" => {
                        let chunks = [Ok::<_, std::io::Error>(vec![0; 800]), Ok(vec![0; 800])];
                        Response::new(Body::from_stream(futures::stream::iter(chunks)))
                    }
                    "/redirect" => Response::builder().status(StatusCode::FOUND)
                        .header("location", "http://localhost/image").body(Body::empty()).unwrap(),
                    "/loop" => Response::builder().status(StatusCode::FOUND)
                        .header("location", uri.to_string()).body(Body::empty()).unwrap(),
                    _ => Response::builder().status(StatusCode::FORBIDDEN).body(Body::empty()).unwrap(),
                }
            });
            let server = tokio::spawn(async move { axum::serve(listener, app).await });
            let resolver = MediaResolver::new(MediaResolverConfig {
                proxy: Some(proxy),
                fetcher: MediaFetcher { allow_direct_ip: true, ..Default::default() },
                max_bytes: 1024,
                ..Default::default()
            })?;
            let source = |path| format!("http://93.184.216.34/{path}?X-Amz-Signature=preserved%2Fsignature&X-Amz-Expires=60");
            let fetched = resolver.resolve_image(&source("image")).await?;
            let budget = MediaBudget::new(STANDARD.decode(PNG)?.len() - 1);
            let media = Media::Image { image_url: [("url".into(), source("stream"))].into() };
            assert!(resolver.resolve_for_plan(media, Some(&budget)).await.err().unwrap().to_string().contains("aggregate request byte limit"));
            let inline = resolver.resolve_image(PNG).await?;
            assert_eq!(fetched.data_url, inline.data_url);
            assert_eq!(fetched.content_hash, inline.content_hash);
            for path in ["large", "stream", "redirect", "loop", "error"] {
                let error = resolver.resolve_image(&source(path)).await.err().context("Expected failure")?;
                assert!(!format!("{error:#}").contains("preserved%2Fsignature"));
                if path == "redirect" { assert!(error.to_string().contains("blocked")); }
                if path == "loop" { assert!(error.to_string().contains("Too many media redirects")); }
                if path == "error" { assert_eq!(crate::http_error_status(&error), Some(403)); }
            }
            server.abort();
            Ok::<_, anyhow::Error>(())
        }).await?
    }

    #[tokio::test]
    async fn dropping_resolve_cancels_download() -> Result<()> {
        use std::{sync::Arc, time::Duration};
        use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};

        tokio::time::timeout(Duration::from_secs(10), async {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
            let proxy = format!("http://{}", listener.local_addr()?);
            let started = Arc::new(tokio::sync::Notify::new());
            let notified = started.clone();
            let server = tokio::spawn(async move {
                let (stream, _) = listener.accept().await?;
                let mut stream = BufReader::new(stream);
                let mut line = String::new();
                loop {
                    line.clear();
                    stream.read_line(&mut line).await?;
                    if line == "\r\n" {
                        break;
                    }
                    ensure!(!line.is_empty(), "Download closed before request headers");
                }
                notified.notify_one();
                assert_eq!(stream.read(&mut [0]).await?, 0);
                Ok::<_, anyhow::Error>(())
            });
            let resolver = MediaResolver::new(MediaResolverConfig {
                proxy: Some(proxy),
                fetcher: MediaFetcher {
                    allow_direct_ip: true,
                    ..Default::default()
                },
                max_bytes: 1024,
                ..Default::default()
            })?;
            let pending =
                tokio::spawn(
                    async move { resolver.resolve_image("http://93.184.216.34/image").await },
                );
            started.notified().await;
            pending.abort();
            assert!(matches!(pending.await, Err(error) if error.is_cancelled()));
            server.await??;
            Ok::<_, anyhow::Error>(())
        })
        .await?
    }
}
