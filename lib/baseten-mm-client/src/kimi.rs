// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NaViT image geometry shared by Kimi K2.5/K2.7 and K3. Matches SGLang
//! navit_resize_config at ffac53d779c08dcdab2d07e5e2a41dba83f0e65c and vLLM
//! navit_resize_image at f84325c48c0acc1e3703103788c5f2976e719762.

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KimiImageConfig {
    pub patch_size: u32,
    pub merge_kernel_size: u32,
    pub in_patch_limit: u32,
    pub patch_limit_on_one_side: u32,
    pub fixed_output_tokens: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> KimiImageConfig {
        KimiImageConfig {
            patch_size: 14,
            merge_kernel_size: 2,
            in_patch_limit: 16384,
            patch_limit_on_one_side: 512,
            fixed_output_tokens: None,
        }
    }

    #[test]
    fn matches_pinned_navit_resize_including_padding_and_float_rounding() -> Result<()> {
        // Outputs of the pinned Python navit_resize_config; no pixel work.
        for (width, height, new_width, new_height, pad_width, pad_height, num_tokens) in [
            (1, 1, 1, 1, 27, 27, 1),
            (84, 56, 84, 56, 0, 0, 6),
            (13, 15, 13, 15, 15, 13, 1),
            (1920, 1080, 1920, 1080, 12, 12, 2691),
            (4096, 4096, 1795, 1795, 25, 25, 4225),
            (100000, 1, 7167, 1, 1, 27, 256),
            (1, 100000, 1, 7167, 27, 1, 256),
            (8193, 8193, 1792, 1792, 0, 0, 4096),
            (u32::MAX, u32::MAX, 1792, 1792, 0, 0, 4096),
        ] {
            assert_eq!(
                config().plan(width, height)?,
                KimiImagePlan {
                    new_width,
                    new_height,
                    pad_width,
                    pad_height,
                    num_tokens,
                    grid_thw: [
                        1,
                        (new_height + pad_height) / 14,
                        (new_width + pad_width) / 14
                    ],
                },
                "{width}x{height}"
            );
        }
        let fixed = KimiImageConfig {
            fixed_output_tokens: Some(128),
            ..config()
        };
        assert_eq!(fixed.plan(84, 56)?.num_tokens, 128);
        Ok(())
    }

    #[test]
    fn uses_checkpoint_settings_and_rejects_invalid_profiles() -> Result<()> {
        let config =
            KimiImageConfig::from_preprocessor_config(&serde_json::json!({"media_proc_cfg":{
                "patch_size":14,"merge_kernel_size":[2,2],"in_patch_limit":65536,
                "patch_limit_on_one_side":512,"fixed_output_tokens":null
            }}))?;
        assert_eq!(config.plan(4096, 4096)?.num_tokens, 16641);
        assert!(KimiImageConfig::from_preprocessor_config(&serde_json::json!({})).is_err());
        for invalid in [
            KimiImageConfig {
                patch_size: 0,
                ..config.clone()
            },
            KimiImageConfig {
                merge_kernel_size: 0,
                ..config.clone()
            },
            KimiImageConfig {
                in_patch_limit: 0,
                ..config.clone()
            },
            KimiImageConfig {
                fixed_output_tokens: Some(0),
                ..config.clone()
            },
            KimiImageConfig {
                patch_limit_on_one_side: u32::MAX,
                ..config.clone()
            },
        ] {
            assert!(invalid.plan(84, 56).is_err());
        }
        assert!(config.plan(0, 1).is_err());
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KimiImagePlan {
    pub new_width: u32,
    pub new_height: u32,
    pub pad_width: u32,
    pub pad_height: u32,
    pub grid_thw: [u32; 3],
    pub num_tokens: u32,
}

impl KimiImageConfig {
    pub fn from_preprocessor_config(value: &serde_json::Value) -> Result<Self> {
        let mut cfg = value
            .get("media_proc_cfg")
            .or_else(|| {
                value
                    .get("image_processor")
                    .and_then(|image| image.get("media_proc_cfg"))
            })
            .unwrap_or(value)
            .clone();
        // Checkpoints may serialize the square spatial merger as [h, w].
        if let Some(pair) = cfg["merge_kernel_size"].as_array() {
            ensure!(
                pair.len() == 2 && pair[0] == pair[1],
                "Kimi image planner requires a square spatial merger"
            );
            cfg["merge_kernel_size"] = pair[0].clone();
        }
        let config: Self = serde_json::from_value(cfg).context("Invalid Kimi media_proc_cfg")?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.patch_size > 0
                && self.merge_kernel_size > 0
                && self.in_patch_limit > 0
                && self.patch_limit_on_one_side > 0,
            "Kimi patch size, merger, and patch limits must be positive"
        );
        ensure!(
            self.fixed_output_tokens.is_none_or(|n| n > 0),
            "Kimi fixed token count must be positive"
        );
        Ok(())
    }

    pub fn plan(&self, width: u32, height: u32) -> Result<KimiImagePlan> {
        self.validate()?;
        ensure!(width > 0 && height > 0, "Image dimensions must be positive");
        let side_limit = self
            .patch_limit_on_one_side
            .checked_mul(self.patch_size)
            .context("Kimi side limit exceeds u32")?;
        let factor = self
            .merge_kernel_size
            .checked_mul(self.patch_size)
            .context("Kimi merger alignment exceeds u32")?;
        let patches_w = f64::from((width / self.patch_size).max(1));
        let patches_h = f64::from((height / self.patch_size).max(1));
        let scale = 1.0_f64
            .min((f64::from(self.in_patch_limit) / (patches_w * patches_h)).sqrt())
            .min(f64::from(side_limit) / f64::from(width))
            .min(f64::from(side_limit) / f64::from(height));
        let new_width = ((f64::from(width) * scale) as u32).max(1).min(side_limit);
        let new_height = ((f64::from(height) * scale) as u32).max(1).min(side_limit);
        let pad_width = (factor - new_width % factor) % factor;
        let pad_height = (factor - new_height % factor) % factor;
        let padded_width = new_width
            .checked_add(pad_width)
            .context("Kimi padded width exceeds u32")?;
        let padded_height = new_height
            .checked_add(pad_height)
            .context("Kimi padded height exceeds u32")?;
        let num_tokens = match self.fixed_output_tokens {
            Some(tokens) => tokens,
            None => (padded_width / factor)
                .checked_mul(padded_height / factor)
                .context("Kimi token count exceeds u32")?,
        };
        Ok(KimiImagePlan {
            new_width,
            new_height,
            pad_width,
            pad_height,
            grid_thw: [
                1,
                padded_height / self.patch_size,
                padded_width / self.patch_size,
            ],
            num_tokens,
        })
    }
}
