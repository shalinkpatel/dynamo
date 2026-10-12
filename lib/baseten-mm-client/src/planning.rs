// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use anyhow::{Context, Result, ensure};
use dynamo_multimodal::{
    models::qwen_vl::{MropeItem, mrope_image_only, smart_resize},
    processor::{ExpansionPart, MediaMetadata, Segment, TokenLayout},
    token_layout::apply_layout,
};
use futures::future::try_join_all;
use serde::{Deserialize, Serialize};

use crate::{
    Dsv41ImageConfig, Dsv41ImagePlan, EncodeRequest, EncoderMetadata, KimiImageConfig,
    KimiImagePlan, MediaResolver, MultiModalClient, ReplayRequest,
};

/// Encoder lengths work for every configured encoder model and modality.
/// Native image planners avoid encoding during resolution.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TokenPlannerSpec {
    #[default]
    Encoder,
    Native {
        spec: serde_json::Value,
    },
    Hf {
        config: serde_json::Value,
        preprocessor_config: serde_json::Value,
    },
    Dsv41 {
        config: Dsv41ImageConfig,
    },
    Kimi {
        config: serde_json::Value,
    },
    Kimi3 {
        config: serde_json::Value,
    },
}

impl TokenPlannerSpec {
    fn is_qwen_image_planner(&self) -> bool {
        match self {
            Self::Native { spec } => spec["family"].as_str() == Some("qwen_vl"),
            Self::Hf { config, .. } => matches!(
                config["model_type"].as_str(),
                Some(
                    "qwen2_vl"
                        | "qwen2_5_vl"
                        | "qwen3_vl"
                        | "qwen3_vl_moe"
                        | "qwen3_5"
                        | "qwen3_5_moe"
                )
            ),
            _ => false,
        }
    }

    pub fn from_hf_configs(
        config: serde_json::Value,
        preprocessor_config: serde_json::Value,
    ) -> Result<Self> {
        match config["model_type"].as_str() {
            Some("kimi_k25" | "kimi_k26" | "kimi_k27") => {
                KimiImageConfig::from_preprocessor_config(&preprocessor_config)?;
                Ok(Self::Kimi {
                    config: preprocessor_config,
                })
            }
            Some("kimi_k3" | "kimi_k3_flash") => {
                KimiImageConfig::from_preprocessor_config(&preprocessor_config)?;
                Ok(Self::Kimi3 {
                    config: preprocessor_config,
                })
            }
            Some("deepseek_v4" | "deepseek_v41") => Ok(Self::Dsv41 {
                config: Dsv41ImageConfig::from_hf_config(&config)?,
            }),
            Some(
                "qwen2_vl" | "qwen2_5_vl" | "qwen3_vl" | "qwen3_vl_moe" | "qwen3_5" | "qwen3_5_moe",
            ) => {
                // Unsupported knobs in a known family are errors, never approximate counts.
                dynamo_multimodal::registry::spec_from_hf_configs(
                    &config.to_string(),
                    &preprocessor_config.to_string(),
                )?;
                Ok(Self::Hf {
                    config,
                    preprocessor_config,
                })
            }
            _ => Ok(Self::Encoder),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PlanningRequest {
    #[serde(flatten)]
    pub request: EncodeRequest,
    pub placeholder_token_id: i32,
    #[serde(default)]
    pub planner: TokenPlannerSpec,
}

/// Safe to send to decode: contains no media URLs, base64, or embeddings.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenPlan {
    pub input_ids: Vec<i32>,
    pub mm_positions: Vec<u32>,
    pub mm_lengths: Vec<u32>,
    pub mm_hashes: Vec<String>,
    pub mm_types: Vec<String>,
    /// Half-open ranges receiving features, excluding DSV4.1 boundary/row tokens.
    pub feature_ranges: Vec<Vec<std::ops::Range<u32>>>,
    pub image_geometry: Vec<Option<Dsv41ImagePlan>>,
    pub image_token_types: Vec<Option<Vec<u8>>>,
    pub kimi_geometry: Vec<Option<KimiImagePlan>>,
    pub qwen_geometry: Vec<Option<QwenImagePlan>>,
    pub mrope: Option<MropePlan>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QwenImagePlan {
    pub grid_thw: [u32; 3],
    pub merge_size: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MropePlan {
    /// Row-major [3, input_length], matching the upstream image-only planner.
    pub position_ids: Vec<i64>,
    pub delta: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Kimi3ImageFraming {
    pub prefix_ids: Vec<i32>,
    pub suffix_ids: Vec<i32>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PlannedMedia {
    pub request: EncodeRequest,
    pub content_hash: String,
    pub mm_hash: String,
    pub length: u32,
    pub feature_token_count: usize,
    pub planner: TokenPlannerSpec,
    pub image_geometry: Option<Dsv41ImagePlan>,
    pub kimi_geometry: Option<KimiImagePlan>,
    pub qwen_geometry: Option<QwenImagePlan>,
    #[serde(default)]
    pub kimi3_framing: Option<Kimi3ImageFraming>,
}

impl PlannedMedia {
    pub fn replay_request(&self) -> Result<ReplayRequest> {
        ensure!(
            matches!(self.planner, TokenPlannerSpec::Encoder),
            "Only encoder-derived plans can use encoder hash/length replay"
        );
        Ok(ReplayRequest {
            request: self.request.clone(),
            mm_hash: self.mm_hash.clone(),
            length: u64::from(self.length),
        })
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PreResolvedRequest {
    pub token_plan: TokenPlan,
    /// Send only to prefill. All media sources are canonical base64 data URIs.
    pub prefill_media: Vec<PlannedMedia>,
}

impl MultiModalClient {
    /// Read authoritative metadata without cloning or forwarding embeddings.
    /// Existing encoder endpoints still return the full cached response body.
    pub async fn plan_metadata(
        &self,
        requests: Vec<EncodeRequest>,
    ) -> Result<Vec<EncoderMetadata>> {
        let requests = requests
            .into_iter()
            .map(serde_json::to_value)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let batch = self.call_batch(requests).await?;
        batch
            .responses
            .iter()
            .map(EncoderMetadata::from_envelope)
            .collect()
    }

    pub async fn pre_resolve(
        &self,
        resolver: &MediaResolver,
        input_ids: &[i32],
        requests: Vec<PlanningRequest>,
        max_tokens: usize,
    ) -> Result<PreResolvedRequest> {
        resolver
            .pre_resolve(input_ids, requests, max_tokens, Some(self))
            .await
    }
}

impl MediaResolver {
    pub async fn pre_resolve(
        &self,
        input_ids: &[i32],
        requests: Vec<PlanningRequest>,
        max_tokens: usize,
        encoder: Option<&MultiModalClient>,
    ) -> Result<PreResolvedRequest> {
        ensure!(
            requests.len() <= self.max_media_items,
            "Prompt exceeds media item limit"
        );
        ensure!(
            self.media_tokenizer.is_some()
                || requests
                    .iter()
                    .all(|request| !matches!(request.planner, TokenPlannerSpec::Kimi3 { .. })),
            "Native Kimi K3 planning requires a checkpoint media tokenizer"
        );
        let budget = crate::media::MediaBudget::new(self.max_total_bytes);
        let budget = &budget;
        let positions = placeholder_positions(input_ids, &requests)?;
        ensure!(
            input_ids.len() <= max_tokens,
            "Prompt exceeds context limit before media expansion"
        );
        if requests
            .iter()
            .any(|request| request.planner.is_qwen_image_planner())
        {
            // The pinned upstream M-RoPE API is image-only. Encoder metadata has
            // no video grid/timestamps, so mixed positions cannot be inferred.
            ensure!(
                requests
                    .iter()
                    .all(|request| request.planner.is_qwen_image_planner()
                        && matches!(request.request.media, crate::Media::Image { .. })),
                "Native Qwen planning requires image-only prompts with a Qwen planner for every media item; mixed native/encoder media are unsupported"
            );
        }
        ensure!(
            encoder.is_some()
                || requests
                    .iter()
                    .all(|request| !matches!(request.planner, TokenPlannerSpec::Encoder)),
            "Encoder-derived planning requires an encoder client"
        );
        let resolved = try_join_all(requests.into_iter().map(|mut item| async move {
            let resolved = self
                .resolve_for_plan(item.request.media, Some(budget))
                .await?;
            item.request.media = resolved.media;
            let mut kimi_geometry = None;
            let mut qwen_geometry = None;
            let mut kimi3_framing = None;
            let (length, feature_token_count, image_geometry) = match &item.planner {
                TokenPlannerSpec::Kimi { config } | TokenPlannerSpec::Kimi3 { config } => {
                    let image = resolved
                        .image
                        .context("Kimi native planning requires image media")?;
                    let geometry = KimiImageConfig::from_preprocessor_config(config)?
                        .plan(image.width, image.height)?;
                    if matches!(item.planner, TokenPlannerSpec::Kimi3 { .. }) {
                        let tokenizer = self.media_tokenizer.as_ref().context("Kimi K3 requires a media tokenizer")?;
                        let prefix_ids = tokenizer(&format!("<|media_begin|>image {}x{}<|media_content|>", image.width, image.height))?;
                        let suffix_ids = tokenizer("<|media_end|>")?;
                        ensure!(!prefix_ids.is_empty() && !suffix_ids.is_empty()
                            && prefix_ids.iter().chain(&suffix_ids).all(|&id| id >= 0 && id != item.placeholder_token_id),
                            "Kimi K3 framing must contain nonnegative control/text IDs without image placeholders");
                        kimi3_framing = Some(Kimi3ImageFraming { prefix_ids, suffix_ids });
                    }
                    let length = geometry.num_tokens;
                    kimi_geometry = Some(geometry);
                    (length, length as usize, None)
                }
                TokenPlannerSpec::Encoder => (0, 0, None),
                TokenPlannerSpec::Native { spec } => {
                    let image = resolved
                        .image
                        .context("Native planning requires image media")?;
                    let spec = serde_json::from_value(spec.clone())?;
                    let (length, geometry) = qwen_plan(spec, &image, item.placeholder_token_id)?;
                    qwen_geometry = Some(geometry);
                    (length.try_into()?, length, None)
                }
                TokenPlannerSpec::Hf {
                    config,
                    preprocessor_config,
                } => {
                    let image = resolved
                        .image
                        .context("Native planning requires image media")?;
                    let spec = dynamo_multimodal::registry::spec_from_hf_configs(
                        &config.to_string(),
                        &preprocessor_config.to_string(),
                    )?;
                    let (length, geometry) = qwen_plan(spec, &image, item.placeholder_token_id)?;
                    qwen_geometry = Some(geometry);
                    (length.try_into()?, length, None)
                }
                TokenPlannerSpec::Dsv41 { config } => {
                    let image = resolved
                        .image
                        .context("DSV4.1 planning requires image media")?;
                    let geometry = config.plan(image.width, image.height)?;
                    let features =
                        usize::try_from(u64::from(geometry.n_llm_h) * u64::from(geometry.n_llm_w))?;
                    (geometry.num_tokens, features, Some(geometry))
                }
            };
            let mm_hash = native_hash(&item, &resolved.content_hash)?;
            Ok::<_, anyhow::Error>(PlannedMedia {
                request: item.request,
                content_hash: resolved.content_hash,
                mm_hash,
                length,
                feature_token_count,
                planner: item.planner,
                image_geometry,
                kimi_geometry,
                qwen_geometry,
                kimi3_framing,
            })
        }))
        .await?;
        let mut media = resolved;
        let fallback_indices: Vec<_> = media
            .iter()
            .enumerate()
            .filter_map(|(index, item)| {
                matches!(item.planner, TokenPlannerSpec::Encoder).then_some(index)
            })
            .collect();
        if !fallback_indices.is_empty() {
            let inputs = fallback_indices
                .iter()
                .map(|&index| media[index].request.clone())
                .collect();
            let metadata = encoder
                .context("Encoder-derived planning requires an encoder client")?
                .plan_metadata(inputs)
                .await?;
            ensure!(
                metadata.len() == fallback_indices.len(),
                "Encoder planning response count differs from media count"
            );
            for (index, metadata) in fallback_indices.into_iter().zip(metadata) {
                let item = &mut media[index];
                item.length = metadata
                    .length
                    .try_into()
                    .context("Encoder span length exceeds u32")?;
                item.feature_token_count = metadata.length.try_into()?;
                item.mm_hash = metadata.mm_hash;
            }
        }
        let mut segments = Vec::new();
        let mut cursor = 0;
        for (index, (&position, item)) in positions.iter().zip(&media).enumerate() {
            ensure!(
                item.length > 0,
                "Media span must contain at least one token"
            );
            segments.push(Segment::Text(cursor..position));
            let id = input_ids[position];
            let expansion = if let Some(geometry) = &item.image_geometry {
                // Pinned SGLang repeats image_token_id throughout; token types,
                // not distinct vocabulary IDs, select learned boundary/row features.
                let mut parts = vec![ExpansionPart::Literal(vec![id])];
                for _ in 0..geometry.n_llm_h {
                    parts.push(ExpansionPart::Feature {
                        id,
                        n: geometry.n_llm_w as usize,
                    });
                    parts.push(ExpansionPart::Literal(vec![id]));
                }
                parts.push(ExpansionPart::Literal(vec![id]));
                parts
            } else if let Some(framing) = &item.kimi3_framing {
                vec![
                    ExpansionPart::Literal(framing.prefix_ids.clone()),
                    ExpansionPart::Feature {
                        id,
                        n: item.feature_token_count,
                    },
                    ExpansionPart::Literal(framing.suffix_ids.clone()),
                ]
            } else {
                vec![ExpansionPart::Feature {
                    id,
                    n: item.length as usize,
                }]
            };
            segments.push(Segment::Media {
                item: index,
                src: position..position + 1,
                expansion,
            });
            cursor = position + 1;
        }
        segments.push(Segment::Text(cursor..input_ids.len()));
        let counts: Vec<_> = media.iter().map(|item| item.feature_token_count).collect();
        let expanded = apply_layout(input_ids, &TokenLayout { segments }, &counts, max_tokens)?;
        let mrope = if !media.is_empty() && media.iter().all(|item| item.qwen_geometry.is_some()) {
            let merge_size = media[0].qwen_geometry.as_ref().unwrap().merge_size;
            let items = media
                .iter()
                .zip(&expanded.offsets)
                .map(|(item, &(start, end))| {
                    let geometry = item.qwen_geometry.as_ref().unwrap();
                    ensure!(
                        geometry.merge_size == merge_size,
                        "Qwen images must share a merge size"
                    );
                    Ok(MropeItem {
                        start,
                        end,
                        grid: geometry.grid_thw,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let (position_ids, delta) =
                mrope_image_only(expanded.input_ids.len(), &items, merge_size)?;
            Some(MropePlan {
                position_ids,
                delta,
            })
        } else {
            None
        };
        // K3 control framing is ordinary text; SGLang's image offsets cover
        // only the feature-token run. DSV4.1 still includes its learned types.
        let offsets: Vec<_> = media
            .iter()
            .enumerate()
            .map(|(index, item)| {
                if item.kimi3_framing.is_some() {
                    let range = expanded.feature_ranges[index]
                        .first()
                        .context("Kimi K3 feature range is missing")?;
                    Ok((range.start, range.end - 1))
                } else {
                    Ok(expanded.offsets[index])
                }
            })
            .collect::<Result<_>>()?;
        let token_plan = TokenPlan {
            input_ids: expanded.input_ids,
            mm_positions: offsets.iter().map(|&(start, _)| start).collect(),
            mm_lengths: offsets
                .iter()
                .map(|&(start, end)| end - start + 1)
                .collect(),
            mm_hashes: media.iter().map(|item| item.mm_hash.clone()).collect(),
            mm_types: media
                .iter()
                .map(|item| item.request.media.kind().to_owned())
                .collect(),
            feature_ranges: expanded.feature_ranges,
            image_token_types: media
                .iter()
                .map(|item| {
                    item.image_geometry
                        .as_ref()
                        .map(Dsv41ImagePlan::token_types)
                })
                .collect(),
            qwen_geometry: media
                .iter()
                .map(|item| item.qwen_geometry.clone())
                .collect(),
            mrope,
            image_geometry: media
                .iter()
                .map(|item| item.image_geometry.clone())
                .collect(),
            kimi_geometry: media
                .iter()
                .map(|item| item.kimi_geometry.clone())
                .collect(),
        };
        Ok(PreResolvedRequest {
            token_plan,
            prefill_media: media,
        })
    }
}

fn qwen_plan(
    spec: dynamo_multimodal::registry::ProcessorSpec,
    image: &crate::ImageDimensions,
    placeholder_token_id: i32,
) -> Result<(usize, QwenImagePlan)> {
    let processor = dynamo_multimodal::registry::build_processor(spec.clone())?;
    let dynamo_multimodal::registry::ProcessorSpec::QwenVl(qwen) = spec else {
        anyhow::bail!("Native image family has no geometry adapter");
    };
    ensure!(
        qwen.image_token_id == placeholder_token_id,
        "Placeholder token ID differs from Qwen checkpoint"
    );
    let length = processor.num_media_tokens(&MediaMetadata::Image {
        width: image.width,
        height: image.height,
    })?;
    let (height, width) = smart_resize(
        image.height as usize,
        image.width as usize,
        qwen.patch_size * qwen.merge_size,
        qwen.min_pixels,
        qwen.max_pixels,
    )?;
    Ok((
        length,
        QwenImagePlan {
            grid_thw: [
                1,
                (height / qwen.patch_size).try_into()?,
                (width / qwen.patch_size).try_into()?,
            ],
            merge_size: qwen.merge_size,
        },
    ))
}

fn placeholder_positions(ids: &[i32], requests: &[PlanningRequest]) -> Result<Vec<usize>> {
    ensure!(
        ids.iter().all(|id| *id >= 0),
        "Input token IDs must be nonnegative"
    );
    let mut expected = BTreeMap::<i32, usize>::new();
    for request in requests {
        ensure!(
            request.placeholder_token_id >= 0,
            "Placeholder token ID must be nonnegative"
        );
        *expected.entry(request.placeholder_token_id).or_default() += 1;
    }
    for (id, count) in expected {
        ensure!(
            ids.iter().filter(|&&token| token == id).count() == count,
            "Placeholder token {id} count differs from media count"
        );
    }
    let mut cursor = 0;
    requests
        .iter()
        .map(|item| {
            let position = cursor
                + ids[cursor..]
                    .iter()
                    .position(|id| *id == item.placeholder_token_id)
                    .context("Media placeholders are not in request order")?;
            cursor = position + 1;
            Ok(position)
        })
        .collect()
}

fn native_hash(item: &PlanningRequest, content_hash: &str) -> Result<String> {
    let fields: BTreeMap<_, _> = item
        .request
        .media
        .fields()
        .iter()
        .filter(|(key, _)| key.as_str() != "url")
        .collect();
    // Native workers must honor the supplied identity; it is not an encoder mm_hash.
    let identity = serde_json::to_vec(&(
        content_hash,
        &item.request.model,
        item.request.detail,
        item.request.media.kind(),
        fields,
        &item.planner,
    ))?;
    Ok(blake3::hash(&identity).to_hex().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HttpEncoderConfig, Media, MediaResolverConfig, TEST_IMAGE_BASE64};
    use base64::Engine;

    fn qwen() -> TokenPlannerSpec {
        TokenPlannerSpec::Native {
            spec: serde_json::json!({
                "family": "qwen_vl", "image_token_id": 100, "patch_size": 14,
                "merge_size": 2, "temporal_patch_size": 2, "min_pixels": 3136,
                "max_pixels": 1003520, "image_mean": [0.48145466, 0.4578275, 0.40821073],
                "image_std": [0.26862954, 0.26130258, 0.27577711]
            }),
        }
    }

    #[tokio::test]
    async fn qwen_plan_carries_grid_and_mrope_positions() -> Result<()> {
        let resolver = MediaResolver::new(MediaResolverConfig::default())?;
        let result = resolver
            .pre_resolve(&[1, 100, 2], vec![image("qwen", 100, qwen())], 8, None)
            .await?;
        let plan = result.token_plan;
        assert_eq!(plan.qwen_geometry[0].as_ref().unwrap().grid_thw, [1, 4, 6]);
        let mrope = plan.mrope.unwrap();
        assert_eq!(mrope.delta, -3);
        assert_eq!(
            mrope.position_ids,
            [
                0, 1, 1, 1, 1, 1, 1, 4, 0, 1, 1, 1, 2, 2, 2, 4, 0, 1, 2, 3, 1, 2, 3, 4
            ]
        );
        assert!(
            resolver
                .pre_resolve(&[1, 101, 2], vec![image("qwen", 101, qwen())], 8, None)
                .await
                .is_err()
        );
        Ok(())
    }

    fn kimi() -> TokenPlannerSpec {
        TokenPlannerSpec::Kimi {
            config: serde_json::json!({"media_proc_cfg": {
                "patch_size":14, "merge_kernel_size":2, "in_patch_limit":65536,
                "patch_limit_on_one_side":512, "fixed_output_tokens":null
            }}),
        }
    }

    fn kimi3() -> TokenPlannerSpec {
        let TokenPlannerSpec::Kimi { config } = kimi() else {
            unreachable!()
        };
        TokenPlannerSpec::Kimi3 { config }
    }

    // Synthetic tokenizer: verify the exact framing text, not checkpoint vocab IDs.
    fn k3_tokenizer(text: &str) -> Result<Vec<i32>> {
        if text == "<|media_end|>" {
            return Ok(vec![903]);
        }
        let dimensions = text
            .strip_prefix("<|media_begin|>image ")
            .and_then(|text| text.strip_suffix("<|media_content|>"))
            .context("Unexpected K3 framing text")?;
        let (width, height) = dimensions
            .split_once('x')
            .context("Missing K3 dimensions")?;
        Ok(vec![901, width.parse()?, height.parse()?, 902])
    }

    #[tokio::test]
    async fn kimi3_frames_original_dimensions_and_records_feature_only_spans() -> Result<()> {
        let resolver = MediaResolver::new(MediaResolverConfig::default())?
            .with_media_tokenizer(std::sync::Arc::new(k3_tokenizer));
        let mut planner = kimi3();
        if let TokenPlannerSpec::Kimi3 { config } = &mut planner {
            config["media_proc_cfg"]["in_patch_limit"] = 4.into();
        }
        let inputs = vec![image("k3", 100, planner.clone()), image("k3", 100, planner)];
        let plan = resolver
            .pre_resolve(&[1, 100, 2, 100, 3], inputs.clone(), 17, None)
            .await?;
        assert_eq!(
            plan.token_plan.input_ids,
            [
                1, 901, 84, 56, 902, 100, 100, 903, 2, 901, 84, 56, 902, 100, 100, 903, 3
            ]
        );
        assert_eq!(plan.token_plan.mm_positions, [5, 13]);
        assert_eq!(plan.token_plan.mm_lengths, [2, 2]);
        assert_eq!(plan.token_plan.feature_ranges[0].len(), 1);
        assert_eq!(plan.token_plan.feature_ranges[0][0], 5..7);
        assert_eq!(plan.token_plan.feature_ranges[1].len(), 1);
        assert_eq!(plan.token_plan.feature_ranges[1][0], 13..15);
        assert_eq!(
            plan.prefill_media[0]
                .kimi_geometry
                .as_ref()
                .unwrap()
                .new_width,
            34
        );
        assert_eq!(plan.prefill_media[0].length, 2);
        assert_eq!(
            plan.prefill_media[0]
                .kimi3_framing
                .as_ref()
                .unwrap()
                .prefix_ids,
            [901, 84, 56, 902]
        );
        let roundtrip: PreResolvedRequest = serde_json::from_str(&serde_json::to_string(&plan)?)?;
        assert_eq!(roundtrip.token_plan.input_ids, plan.token_plan.input_ids);
        assert!(
            resolver
                .pre_resolve(&[1, 100, 2, 100, 3], inputs, 16, None)
                .await
                .is_err()
        );
        Ok(())
    }

    #[tokio::test]
    async fn kimi3_requires_a_tokenizer_and_rejects_invalid_framing() -> Result<()> {
        let resolver = MediaResolver::new(MediaResolverConfig::default())?;
        let mut item = image("k3", 100, kimi3());
        item.request
            .media
            .set_url("http://localhost:1/never-fetch".into());
        assert!(
            resolver
                .pre_resolve(&[100], vec![item], 100, None)
                .await
                .err()
                .unwrap()
                .to_string()
                .contains("checkpoint media tokenizer")
        );
        for ids in [vec![], vec![-1], vec![100]] {
            let resolver = resolver
                .clone()
                .with_media_tokenizer(std::sync::Arc::new(move |_| Ok(ids.clone())));
            assert!(
                resolver
                    .pre_resolve(&[100], vec![image("k3", 100, kimi3())], 100, None)
                    .await
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("framing must contain")
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn rejects_mixed_qwen_plans_before_resolving_or_encoding() -> Result<()> {
        let resolver = MediaResolver::new(MediaResolverConfig::default())?;
        for kind in ["image_url", "video_url", "audio_url"] {
            let fallback: PlanningRequest = serde_json::from_value(serde_json::json!({
                "type":kind, kind:{"url":"http://localhost:1/never-fetch"},
                "model":"qwen", "placeholder_token_id":200
            }))?;
            let error = resolver
                .pre_resolve(
                    &[100, 200],
                    vec![image("qwen", 100, qwen()), fallback],
                    100,
                    None,
                )
                .await
                .err()
                .unwrap();
            assert!(error.to_string().contains("image-only prompts"));
        }
        let error = resolver
            .pre_resolve(
                &[100, 200],
                vec![image("qwen", 100, qwen()), image("kimi", 200, kimi())],
                100,
                None,
            )
            .await
            .err()
            .unwrap();
        assert!(error.to_string().contains("image-only prompts"));
        Ok(())
    }

    fn dsv41() -> TokenPlannerSpec {
        TokenPlannerSpec::Dsv41 {
            config: Dsv41ImageConfig {
                vision_patch_size: 14,
                vision_downsample_ratio: 3,
                vision_max_n_token: 1024,
                vision_min_pixels: 295936,
                vision_max_wh_ratio: None,
            },
        }
    }

    fn image(model: &str, token: i32, planner: TokenPlannerSpec) -> PlanningRequest {
        PlanningRequest {
            request: EncodeRequest::new(
                Media::Image {
                    image_url: [("url".into(), TEST_IMAGE_BASE64.into())].into(),
                },
                model,
            ),
            placeholder_token_id: token,
            planner,
        }
    }

    #[tokio::test]
    async fn native_plans_expand_once_and_separate_prefill_payloads() -> Result<()> {
        let resolver = MediaResolver::new(MediaResolverConfig::default())?;
        let inputs = vec![image("kimi", 100, kimi()), image("dsv41", 200, dsv41())];
        let plan = resolver
            .pre_resolve(&[1, 100, 2, 200, 3], inputs.clone(), 198, None)
            .await?;
        assert_eq!(plan.token_plan.input_ids.len(), 198);
        assert_eq!(plan.token_plan.mm_positions, [1, 8]);
        assert_eq!(plan.token_plan.mm_lengths, [6, 189]);
        assert_eq!(&plan.token_plan.input_ids[1..7], &[100; 6]);
        assert_eq!(&plan.token_plan.input_ids[8..197], &[200; 189]);
        assert_eq!(plan.token_plan.feature_ranges[0].len(), 1);
        assert_eq!(plan.token_plan.feature_ranges[0][0], 1..7);
        assert_eq!(plan.token_plan.feature_ranges[1].len(), 11);
        assert_eq!(plan.token_plan.feature_ranges[1][0], 9..25);
        assert_eq!(plan.token_plan.feature_ranges[1][10], 179..195);
        assert_eq!(plan.prefill_media[1].feature_token_count, 176);
        let types = plan.token_plan.image_token_types[1].as_ref().unwrap();
        assert_eq!(types.len(), 189);
        assert_eq!(types[0], 0);
        assert_eq!(types[188], 3);
        assert_eq!(types.iter().filter(|&&kind| kind == 1).count(), 176);
        assert_eq!(types.iter().filter(|&&kind| kind == 2).count(), 11);
        for range in &plan.token_plan.feature_ranges[1] {
            assert!(
                types[(range.start - 8) as usize..(range.end - 8) as usize]
                    .iter()
                    .all(|&kind| kind == 1)
            );
        }
        assert_eq!(
            plan.prefill_media[0].content_hash,
            plan.prefill_media[1].content_hash
        );
        assert_ne!(plan.token_plan.mm_hashes[0], plan.token_plan.mm_hashes[1]);
        let decode = serde_json::to_string(&plan.token_plan)?;
        assert!(!decode.contains("data:"));
        assert!(!decode.contains(TEST_IMAGE_BASE64));
        assert!(!decode.contains("mm_kwargs"));
        assert!(plan.prefill_media.iter().all(|item| {
            item.request
                .media
                .url()
                .unwrap()
                .starts_with("data:image/png;base64,")
        }));
        assert!(plan.prefill_media[0].replay_request().is_err());
        let roundtrip: PreResolvedRequest = serde_json::from_str(&serde_json::to_string(&plan)?)?;
        assert_eq!(roundtrip.token_plan.input_ids, plan.token_plan.input_ids);
        assert!(
            resolver
                .pre_resolve(&[1, 100, 2, 200, 3], inputs, 197, None)
                .await
                .is_err()
        );
        Ok(())
    }

    #[tokio::test]
    async fn rejects_missing_extra_out_of_order_and_negative_placeholders() -> Result<()> {
        let resolver = MediaResolver::new(MediaResolverConfig::default())?;
        let requests = vec![image("kimi", 100, kimi()), image("dsv41", 200, dsv41())];
        for ids in [
            vec![100],
            vec![100, 100, 200],
            vec![200, 100],
            vec![-1, 100, 200],
        ] {
            assert!(
                resolver
                    .pre_resolve(&ids, requests.clone(), 1000, None)
                    .await
                    .is_err()
            );
        }
        assert!(
            resolver
                .pre_resolve(
                    &[100],
                    vec![image("unknown", 100, TokenPlannerSpec::Encoder)],
                    1000,
                    None
                )
                .await
                .is_err()
        );
        let text = resolver
            .pre_resolve(&[1, 2, 3], Vec::new(), 3, None)
            .await?;
        assert_eq!(text.token_plan.input_ids, [1, 2, 3]);
        assert!(text.prefill_media.is_empty());
        Ok(())
    }

    #[test]
    fn planner_selection_is_exact_or_encoder_derived() -> Result<()> {
        for model in ["glm5_next", "muse_spark", "unknown_future_model"] {
            assert!(matches!(
                TokenPlannerSpec::from_hf_configs(
                    serde_json::json!({"model_type": model}),
                    serde_json::json!({})
                )?,
                TokenPlannerSpec::Encoder
            ));
        }
        for model in ["kimi_k25", "kimi_k27", "kimi_k3", "kimi_k3_flash"] {
            let processor = serde_json::json!({"media_proc_cfg":{
                "patch_size":14, "merge_kernel_size":2, "in_patch_limit":65536,
                "patch_limit_on_one_side":512, "fixed_output_tokens":null
            }});
            assert!(matches!(
                TokenPlannerSpec::from_hf_configs(
                    serde_json::json!({"model_type":model}),
                    processor
                )?,
                TokenPlannerSpec::Kimi { .. } | TokenPlannerSpec::Kimi3 { .. }
            ));
            assert!(
                TokenPlannerSpec::from_hf_configs(
                    serde_json::json!({"model_type":model}),
                    serde_json::json!({})
                )
                .is_err()
            );
        }
        let config = serde_json::json!({"model_type": "qwen3_vl", "image_token_id": 100});
        let processor = serde_json::json!({"patch_size":14, "merge_size":2, "temporal_patch_size":2,
            "min_pixels":3136, "max_pixels":1003520, "image_mean":[0.48145466,0.4578275,0.40821073],
            "image_std":[0.26862954,0.26130258,0.27577711]});
        assert!(matches!(
            TokenPlannerSpec::from_hf_configs(config.clone(), processor.clone())?,
            TokenPlannerSpec::Hf { .. }
        ));
        let mut invalid = processor;
        invalid["do_resize"] = false.into();
        assert!(TokenPlannerSpec::from_hf_configs(config, invalid).is_err());
        let request: PlanningRequest = serde_json::from_value(serde_json::json!({
            "type":"image_url", "image_url":{"url":TEST_IMAGE_BASE64},
            "model":"future-model", "placeholder_token_id":100
        }))?;
        assert!(matches!(request.planner, TokenPlannerSpec::Encoder));
        Ok(())
    }

    #[tokio::test]
    async fn encoder_plans_cover_models_and_modalities_and_replay_exact_metadata() -> Result<()> {
        use axum::{Json, Router, routing::post};
        use std::{
            sync::{
                Arc,
                atomic::{AtomicUsize, Ordering},
            },
            time::Duration,
        };

        tokio::time::timeout(Duration::from_secs(10), async {
            let calls = Arc::new(AtomicUsize::new(0));
            let recorded = calls.clone();
            // Deliberately synthetic encoder lengths: exercise the authoritative
            // metadata protocol, not geometry correctness of these model families.
            let app = Router::new().route("/encode", post(move |Json(request): Json<serde_json::Value>| {
                let recorded = recorded.clone();
                async move {
                    recorded.fetch_add(1, Ordering::SeqCst);
                    let kind = request["type"].as_str().unwrap();
                    assert!(request[kind]["url"].as_str().unwrap().starts_with("data:"));
                    assert!(request.get("placeholder_token_id").is_none());
                    let length = match kind { "image_url" => 6, "audio_url" => 7, _ => 8 };
                    let hash = blake3::hash(request.to_string().as_bytes()).to_hex().to_string();
                    Json(serde_json::json!({"success":true, "mm_response":{
                        "mm_hash":hash, "length":length, "mm_kwargs":"opaque-embedding-not-forwarded"
                    }}))
                }
            }));
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
            let mut config = HttpEncoderConfig::new(format!("http://{}/encode", listener.local_addr()?));
            config.api_key = "test-key".into();
            let client = MultiModalClient::new(config)?;
            let server = tokio::spawn(async move { axum::serve(listener, app).await });
            let resolver = MediaResolver::new(MediaResolverConfig::default())?;
            let mut requests = vec![];
            for model in ["kimi-k25", "kimi-k3", "glm-5.3", "muse-spark", "future-model"] {
                requests.push(image(model, 100, TokenPlannerSpec::Encoder));
            }
            // Containers are intentionally opaque at this layer; the encoder owns decoding.
            requests.push(PlanningRequest { request: EncodeRequest::new(Media::Audio {
                audio_url: [("url".into(), "data:audio/wav;base64,UklGRg==".into())].into()
            }, "audio-model"), placeholder_token_id: 200, planner: TokenPlannerSpec::Encoder });
            requests.push(PlanningRequest { request: EncodeRequest::new(Media::Video {
                video_url: [("url".into(), "data:video/mp4;base64,AAAAAGZ0eXA=".into())].into()
            }, "video-model"), placeholder_token_id: 300, planner: TokenPlannerSpec::Encoder });
            let plan = client.pre_resolve(&resolver, &[1,100,100,100,100,100,200,300,2], requests, 100).await?;
            assert_eq!(plan.token_plan.mm_lengths, [6,6,6,6,6,7,8]);
            assert_eq!(plan.token_plan.mm_positions, [1,7,13,19,25,31,38]);
            assert_eq!(plan.token_plan.input_ids.len(), 47);
            assert_eq!(calls.load(Ordering::SeqCst), 7);
            let serialized = serde_json::to_string(&plan)?;
            assert!(!serialized.contains("opaque-embedding-not-forwarded"));
            let replays = plan.prefill_media.iter().map(PlannedMedia::replay_request).collect::<Result<Vec<_>>>()?;
            let metadata = client.plan_metadata(replays.iter().map(|item| item.request.clone()).collect()).await?;
            assert!(metadata.iter().zip(&replays).all(|(actual, expected)| actual.mm_hash == expected.mm_hash && actual.length == expected.length));
            assert_eq!(calls.load(Ordering::SeqCst), 14);
            server.abort();
            Ok::<_, anyhow::Error>(())
        }).await?
    }

    #[tokio::test]
    async fn rejects_request_media_count_and_aggregate_bytes() -> Result<()> {
        let inputs = vec![image("qwen", 100, qwen()), image("qwen", 100, qwen())];
        let count_limited = MediaResolver::new(MediaResolverConfig {
            max_media_items: 1,
            ..Default::default()
        })?;
        assert!(
            count_limited
                .pre_resolve(&[100, 100], inputs.clone(), 100, None)
                .await
                .err()
                .unwrap()
                .to_string()
                .contains("media item limit")
        );
        let bytes = base64::engine::general_purpose::STANDARD.decode(TEST_IMAGE_BASE64)?;
        let byte_limited = MediaResolver::new(MediaResolverConfig {
            max_total_bytes: bytes.len() + 2,
            ..Default::default()
        })?;
        assert!(
            byte_limited
                .pre_resolve(&[100, 100], inputs, 100, None)
                .await
                .err()
                .unwrap()
                .to_string()
                .contains("aggregate request byte limit")
        );
        Ok(())
    }

    #[tokio::test]
    async fn parallel_fetching_preserves_order_for_all_native_families() -> Result<()> {
        use crate::MediaFetcher;
        use axum::{Router, extract::Path, routing::get};
        use base64::Engine;
        use std::{
            io::Cursor,
            sync::{
                Arc,
                atomic::{AtomicUsize, Ordering},
            },
            time::Duration,
        };

        tokio::time::timeout(Duration::from_secs(10), async {
            let images: Arc<Vec<_>> = Arc::new(
                (0..65)
                    .map(|index| {
                        let mut bytes = Cursor::new(Vec::new());
                        image::DynamicImage::new_rgb8(84 + index * 28, 56)
                            .write_to(&mut bytes, image::ImageFormat::Png)
                            .unwrap();
                        bytes.into_inner()
                    })
                    .collect(),
            );
            let barrier = Arc::new(tokio::sync::Barrier::new(64));
            let active = Arc::new(AtomicUsize::new(0));
            let peak = Arc::new(AtomicUsize::new(0));
            let observed_peak = peak.clone();
            let app = Router::new().route(
                "/image/{index}",
                get(move |Path(index): Path<usize>| {
                    let (images, barrier, active, peak) = (
                        images.clone(),
                        barrier.clone(),
                        active.clone(),
                        peak.clone(),
                    );
                    async move {
                        let count = active.fetch_add(1, Ordering::SeqCst) + 1;
                        peak.fetch_max(count, Ordering::SeqCst);
                        if index < 64 {
                            barrier.wait().await;
                        }
                        // The 65th fetch must wait for one of the first 64 slots.
                        if index % 2 == 0 {
                            tokio::time::sleep(Duration::from_millis(5)).await;
                        }
                        active.fetch_sub(1, Ordering::SeqCst);
                        images[index].clone()
                    }
                }),
            );
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
            let base = format!("http://{}", listener.local_addr()?);
            let server = tokio::spawn(async move { axum::serve(listener, app).await });
            let resolver = MediaResolver::new(MediaResolverConfig {
                fetcher: MediaFetcher {
                    allow_direct_ip: true,
                    allow_direct_port: true,
                    allow_private_ips: true,
                    ..Default::default()
                },
                ..Default::default()
            })?
            .with_media_tokenizer(std::sync::Arc::new(k3_tokenizer));
            let kimi = TokenPlannerSpec::Kimi {
                config: serde_json::json!({"media_proc_cfg":{
                    "patch_size":14,"merge_kernel_size":2,"in_patch_limit":65536,
                    "patch_limit_on_one_side":512,"fixed_output_tokens":null
                }}),
            };
            for (model, planner) in [
                ("kimi-k27", kimi.clone()),
                ("kimi-k3", kimi3()),
                ("dsv4.1", dsv41()),
                ("qwen", qwen()),
            ] {
                let requests = (0..65)
                    .map(|index| {
                        let mut item = image(model, 100, planner.clone());
                        item.request.media.set_url(format!("{base}/image/{index}"));
                        item
                    })
                    .collect();
                let plan = resolver
                    .pre_resolve(
                        &[vec![1], vec![100; 65], vec![2]].concat(),
                        requests,
                        100000,
                        None,
                    )
                    .await?;
                for (index, media) in plan.prefill_media.iter().enumerate() {
                    let encoded = media.request.media.url()?.split_once(',').unwrap().1;
                    let bytes = base64::engine::general_purpose::STANDARD.decode(encoded)?;
                    assert_eq!(
                        dynamo_multimodal::image::decode::dimensions(&bytes)?.1,
                        84 + index * 28
                    );
                }
                if model != "dsv4.1" {
                    assert_eq!(
                        plan.token_plan.mm_lengths,
                        (0..65).map(|index| 6 + 2 * index).collect::<Vec<_>>()
                    );
                }
            }
            assert_eq!(observed_peak.load(Ordering::SeqCst), 64);
            server.abort();
            Ok::<_, anyhow::Error>(())
        })
        .await?
    }
}
