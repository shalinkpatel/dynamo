// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CPU-only geometry matching SGLang's `deepseek_v41_image_processing.py` at
//! ffac53d779c08dcdab2d07e5e2a41dba83f0e65c. No pixel decoding or model execution.

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

/// Geometry settings from the checkpoint's config.json; no serving defaults.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Dsv41ImageConfig {
    pub vision_patch_size: u32,
    pub vision_downsample_ratio: u32,
    pub vision_max_n_token: u32,
    pub vision_min_pixels: u64,
    pub vision_max_wh_ratio: Option<f64>,
}

/// Image span geometry, independent of image bytes and feature extraction.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dsv41ImagePlan {
    pub resized_width: u32,
    pub resized_height: u32,
    pub n_vit_w: u32,
    pub n_vit_h: u32,
    pub n_llm_w: u32,
    pub n_llm_h: u32,
    /// Includes IMAGE_START, one IMAGE_NEW_LINE per row, and IMAGE_END.
    pub num_tokens: u32,
}

impl Dsv41ImagePlan {
    /// SGLang token types: START=0, IMAGE=1, NEW_LINE=2, END=3.
    /// These are types, not vocabulary IDs: every span ID is image_token_id.
    pub fn token_types(&self) -> Vec<u8> {
        let mut types = Vec::with_capacity(self.num_tokens as usize);
        types.push(0);
        for _ in 0..self.n_llm_h {
            types.extend(std::iter::repeat_n(1, self.n_llm_w as usize));
            types.push(2);
        }
        types.push(3);
        types
    }
}

impl Dsv41ImageConfig {
    /// Matches SGLang deepseek_v41_image_processing.plan_image_grid.
    pub fn plan(&self, width: u32, height: u32) -> Result<Dsv41ImagePlan> {
        ensure!(width > 0 && height > 0, "Image dimensions must be positive");
        ensure!(
            self.vision_patch_size > 0 && self.vision_downsample_ratio > 0,
            "DSV4.1 patch size and downsample ratio must be positive"
        );
        ensure!(
            self.vision_max_n_token >= 4,
            "DSV4.1 token limit must fit at least one image cell (4 tokens)"
        );
        ensure!(
            self.vision_min_pixels <= 1 << 53,
            "DSV4.1 minimum pixel count exceeds exact floating-point range"
        );
        if let Some(ratio) = self.vision_max_wh_ratio {
            ensure!(
                ratio.is_finite() && ratio > 0.0,
                "DSV4.1 maximum width/height ratio must be finite and positive"
            );
        }

        // Python uses double precision and truncates after scaling small images.
        let (mut width, mut height) = (f64::from(width), f64::from(height));
        if let Some(ratio) = self.vision_max_wh_ratio {
            width = width.min(height * ratio);
        }
        if width * height < self.vision_min_pixels as f64 {
            let scale = (self.vision_min_pixels as f64 / (width * height)).sqrt();
            width = (width * scale).floor();
            height = (height * scale).floor();
        }
        ensure!(
            width > 0.0 && height > 0.0,
            "DSV4.1 image scaling produced an empty dimension"
        );
        let patch = f64::from(self.vision_patch_size);
        let (mut resized_width, mut resized_height) = (
            rounded_dimension((width / patch).ceil() * patch)?,
            rounded_dimension((height / patch).ceil() * patch)?,
        );
        let token_count = |width: u32, height: u32| {
            let (w, h) = (self.llm_dimension(width), self.llm_dimension(height));
            u64::from(h) * (u64::from(w) + 1) + 2
        };
        if token_count(resized_width, resized_height) > u64::from(self.vision_max_n_token) {
            let ratio = height / width;
            let budget = f64::from(self.vision_max_n_token - 2);
            let max_w = (budget / ratio + 0.25).sqrt() - 0.5;
            let max_h = max_w * ratio;
            let cell = patch * f64::from(self.vision_downsample_ratio);
            let (new_width, new_height) = if max_w < 1.0 {
                (cell, f64::from((self.vision_max_n_token - 2) / 2) * cell)
            } else if max_h < 1.0 {
                (f64::from(self.vision_max_n_token - 3) * cell, cell)
            } else {
                let scale = (max_w.floor() * cell / width).min(max_h.floor() * cell / height);
                (
                    (width * scale / patch).floor() * patch,
                    (height * scale / patch).floor() * patch,
                )
            };
            resized_width = rounded_dimension(new_width)?;
            resized_height = rounded_dimension(new_height)?;
        }
        let num_tokens = token_count(resized_width, resized_height);
        ensure!(
            num_tokens <= u64::from(self.vision_max_n_token),
            "DSV4.1 resized image exceeds the token limit"
        );
        Ok(Dsv41ImagePlan {
            resized_width,
            resized_height,
            n_vit_w: resized_width / self.vision_patch_size,
            n_vit_h: resized_height / self.vision_patch_size,
            n_llm_w: self.llm_dimension(resized_width),
            n_llm_h: self.llm_dimension(resized_height),
            num_tokens: num_tokens.try_into()?,
        })
    }

    fn llm_dimension(&self, pixels: u32) -> u32 {
        (pixels / self.vision_patch_size).div_ceil(self.vision_downsample_ratio)
    }
}

fn rounded_dimension(value: f64) -> Result<u32> {
    ensure!(
        value.is_finite() && value >= 1.0 && value <= f64::from(u32::MAX),
        "DSV4.1 resized dimension is outside the positive u32 range"
    );
    Ok(value as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Dsv41ImageConfig {
        // SGLang DeepseekV4Config defaults at ffac53d779c08dcdab2d07e5e2a41dba83f0e65c.
        Dsv41ImageConfig {
            vision_patch_size: 14,
            vision_downsample_ratio: 3,
            vision_max_n_token: 1024,
            vision_min_pixels: 295936,
            vision_max_wh_ratio: None,
        }
    }

    #[test]
    fn matches_pinned_sglang_geometry() -> Result<()> {
        // Outputs of the pinned plan_image_grid, including its float rounding.
        for (width, height, resized_width, resized_height, n_llm_w, n_llm_h, num_tokens) in [
            (84, 56, 672, 448, 16, 11, 189),
            (1, 1, 546, 546, 13, 13, 184),
            (1920, 1080, 1708, 966, 41, 23, 968),
            (1080, 1920, 966, 1708, 23, 41, 986),
            (4096, 4096, 1302, 1302, 31, 31, 994),
            (100000, 1, 42882, 42, 1021, 1, 1024),
            (1, 100000, 42, 21462, 1, 511, 1024),
            (43, 42, 560, 546, 14, 13, 197),
            (1024, 1024, 1036, 1036, 25, 25, 652),
            (21000, 42, 21000, 42, 500, 1, 503),
        ] {
            let want = Dsv41ImagePlan {
                resized_width,
                resized_height,
                n_vit_w: resized_width / 14,
                n_vit_h: resized_height / 14,
                n_llm_w,
                n_llm_h,
                num_tokens,
            };
            assert_eq!(config().plan(width, height)?, want, "{width}x{height}");
        }
        let mut config = config();
        config.vision_max_wh_ratio = Some(4.0);
        let plan = config.plan(10000, 100)?;
        assert_eq!((plan.resized_width, plan.resized_height), (1092, 280));
        assert_eq!((plan.n_llm_w, plan.n_llm_h, plan.num_tokens), (26, 7, 191));
        Ok(())
    }

    #[test]
    fn checkpoint_settings_and_minimum_budget() -> Result<()> {
        let config: Dsv41ImageConfig = serde_json::from_value(serde_json::json!({
            "model_type": "deepseek_v4", "image_token_id": 129264,
            "vision_patch_size": 14, "vision_downsample_ratio": 3,
            "vision_max_n_token": 4, "vision_min_pixels": 0,
            "vision_max_wh_ratio": null
        }))?;
        for (width, height) in [(1, 1), (10000, 1), (1, 10000), (4096, 4096)] {
            let plan = config.plan(width, height)?;
            assert_eq!((plan.n_llm_w, plan.n_llm_h, plan.num_tokens), (1, 1, 4));
            assert_eq!(
                serde_json::from_str::<Dsv41ImagePlan>(&serde_json::to_string(&plan)?)?,
                plan
            );
        }
        Ok(())
    }

    #[test]
    fn matches_sglang_with_custom_checkpoint_settings() -> Result<()> {
        let config = Dsv41ImageConfig {
            vision_patch_size: 16,
            vision_downsample_ratio: 2,
            vision_max_n_token: 113,
            vision_min_pixels: 10000,
            vision_max_wh_ratio: Some(1.5),
        };
        // Deliberately synthetic settings: verify no geometry depends on defaults.
        for (width, height, resized_width, resized_height, n_llm_w, n_llm_h, num_tokens) in [
            (31, 32, 112, 112, 4, 4, 22),
            (1920, 1080, 384, 256, 12, 8, 106),
            (1080, 1920, 224, 384, 7, 12, 98),
            (100000, 1, 128, 96, 4, 3, 17),
            (1, 100000, 32, 1760, 1, 55, 112),
            (320, 320, 320, 320, 10, 10, 112),
            (64, 80, 96, 112, 3, 4, 18),
            (1, 1, 112, 112, 4, 4, 22),
        ] {
            assert_eq!(
                config.plan(width, height)?,
                Dsv41ImagePlan {
                    resized_width,
                    resized_height,
                    n_vit_w: resized_width / 16,
                    n_vit_h: resized_height / 16,
                    n_llm_w,
                    n_llm_h,
                    num_tokens,
                },
                "{width}x{height}"
            );
        }
        Ok(())
    }

    #[test]
    fn rejects_invalid_geometry_and_settings() {
        for (width, height) in [(0, 1), (1, 0)] {
            assert!(config().plan(width, height).is_err());
        }
        for invalid in [
            Dsv41ImageConfig {
                vision_patch_size: 0,
                ..config()
            },
            Dsv41ImageConfig {
                vision_downsample_ratio: 0,
                ..config()
            },
            Dsv41ImageConfig {
                vision_max_n_token: 3,
                ..config()
            },
            Dsv41ImageConfig {
                vision_min_pixels: (1 << 53) + 1,
                ..config()
            },
        ] {
            assert!(invalid.plan(84, 56).is_err());
        }
        for ratio in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let config = Dsv41ImageConfig {
                vision_max_wh_ratio: Some(ratio),
                ..config()
            };
            assert!(config.plan(84, 56).is_err());
        }
        // SGLang's float rounding produces a zero-height grid for these settings.
        let empty_grid = Dsv41ImageConfig {
            vision_patch_size: 16,
            vision_downsample_ratio: 1,
            vision_max_n_token: 5,
            vision_min_pixels: 10000,
            vision_max_wh_ratio: Some(1.5),
        };
        assert!(empty_grid.plan(93112, 50992).is_err());
        assert!(config().plan(u32::MAX, u32::MAX).is_err());
    }
}
