//! FLUX.2 autoencoder (KL VAE with 32 latent channels, 2x2 patchified + batch-normalized
//! latents for the transformer).
//!
//! Implemented from the public architecture (Apache-2.0 reference: Hugging Face `diffusers`
//! `AutoencoderKLFlux2`). Weights use diffusers names. All activations are NHWC.

use std::path::Path;

use anyhow::{Context, Result};
use nnc::{ops, DataType, Device, Graph, Var};
use serde::Deserialize;

use crate::weights::{conv_weight_to_nhwc, upload, Params, SafeTensorsFiles};

#[derive(Clone, Debug, Deserialize)]
pub struct Flux2VaeConfig {
    #[serde(default = "d_ch")]
    pub block_out_channels: Vec<usize>,
    #[serde(default)]
    pub decoder_block_out_channels: Option<Vec<usize>>,
    #[serde(default = "d_lpb")]
    pub layers_per_block: usize,
    #[serde(default = "d_latent")]
    pub latent_channels: usize,
    #[serde(default = "d_groups")]
    pub norm_num_groups: usize,
    #[serde(default = "d_true")]
    pub use_quant_conv: bool,
    #[serde(default = "d_true")]
    pub use_post_quant_conv: bool,
    #[serde(default = "d_true")]
    pub mid_block_add_attention: bool,
    #[serde(default = "d_bn_eps")]
    pub batch_norm_eps: f32,
}

fn d_ch() -> Vec<usize> {
    vec![128, 256, 512, 512]
}
fn d_lpb() -> usize {
    2
}
fn d_latent() -> usize {
    32
}
fn d_groups() -> usize {
    32
}
fn d_true() -> bool {
    true
}
fn d_bn_eps() -> f32 {
    1e-4
}

impl Default for Flux2VaeConfig {
    fn default() -> Self {
        serde_json::from_str("{}").unwrap()
    }
}

pub struct Flux2Vae {
    pub config: Flux2VaeConfig,
    params: Params,
    device: Device,
    dtype: DataType,
    /// Batch-norm statistics over the 4*latent patchified channels (f32, host).
    bn_mean: Vec<f32>,
    bn_std: Vec<f32>,
}

impl Flux2Vae {
    pub fn load(graph: &Graph, dir: &Path, device: Device, dtype: DataType) -> Result<Flux2Vae> {
        let config_path = dir.join("config.json");
        let config: Flux2VaeConfig = if config_path.exists() {
            serde_json::from_slice(&std::fs::read(&config_path)?).context("vae config.json")?
        } else {
            Flux2VaeConfig::default()
        };
        let files = if dir.is_dir() { SafeTensorsFiles::open_dir(dir)? } else { SafeTensorsFiles::open(&[dir])? };
        Flux2Vae::load_from(graph, &files, config, device, dtype)
    }

    pub fn load_from(graph: &Graph, files: &SafeTensorsFiles, config: Flux2VaeConfig, device: Device, dtype: DataType) -> Result<Flux2Vae> {
        let prefix = if files.contains("vae.bn.running_mean") { "vae." } else { "" };
        let mut params = Params::new();
        for key in files.keys() {
            let Some(name) = key.strip_prefix(prefix) else { continue };
            if name.starts_with("bn.") {
                continue;
            }
            let info = files.info(key).unwrap();
            let v = if info.shape.len() == 4 {
                if device.is_gpu() {
                    // GPU convolution kernels take PyTorch's OIHW layout (tagged NCHW).
                    let host = files.read(key, dtype)?;
                    graph.from_host_with_format(&host, device, nnc::Format::Nchw)
                } else {
                    let (w, s) = conv_weight_to_nhwc(&files.read_f32(key)?, &info.shape);
                    upload(graph, &nnc::HostTensor::from_f32(&w, &s), device, dtype)
                }
            } else {
                upload(graph, &files.read(key, DataType::F32)?, device, dtype)
            };
            params.insert(name.to_string(), v);
        }
        let mean = files.read_f32(&format!("{prefix}bn.running_mean")).context("vae bn.running_mean")?;
        let var = files.read_f32(&format!("{prefix}bn.running_var")).context("vae bn.running_var")?;
        let bn_std = var.iter().map(|v| (v + config.batch_norm_eps).sqrt()).collect();
        Ok(Flux2Vae { config, params, device, dtype, bn_mean: mean, bn_std })
    }

    fn p(&self, name: &str) -> &Var {
        self.params.get(name)
    }

    fn conv(&self, x: &Var, name: &str, stride: usize, pad: (usize, usize, usize, usize)) -> Var {
        ops::conv2d(x, self.p(&format!("{name}.weight")), self.params.try_get(&format!("{name}.bias")), (stride, stride), pad, 1)
    }

    fn conv3(&self, x: &Var, name: &str) -> Var {
        self.conv(x, name, 1, (1, 1, 1, 1))
    }

    fn norm(&self, x: &Var, name: &str) -> Var {
        ops::group_norm(
            x,
            3,
            self.config.norm_num_groups,
            &[1, 2],
            1e-6,
            Some((self.p(&format!("{name}.weight")), self.p(&format!("{name}.bias")))),
        )
    }

    fn resnet(&self, x: &Var, name: &str) -> Var {
        let h = ops::silu(&self.norm(x, &format!("{name}.norm1")));
        let h = self.conv3(&h, &format!("{name}.conv1"));
        let h = ops::silu(&self.norm(&h, &format!("{name}.norm2")));
        let h = self.conv3(&h, &format!("{name}.conv2"));
        let shortcut = if self.params.contains(&format!("{name}.conv_shortcut.weight")) {
            self.conv(x, &format!("{name}.conv_shortcut"), 1, (0, 0, 0, 0))
        } else {
            x.clone()
        };
        ops::add(&shortcut, &h)
    }

    fn attention(&self, x: &Var, name: &str) -> Var {
        let s = x.shape();
        let (b, h, w, c) = (s[0], s[1], s[2], s[3]);
        let n = self.norm(x, &format!("{name}.group_norm")).reshape(&[b, h * w, c]);
        let lin = |v: &Var, m: &str| ops::linear(v, self.p(&format!("{name}.{m}.weight")), self.params.try_get(&format!("{name}.{m}.bias")));
        let q = lin(&n, "to_q").reshape(&[b, h * w, 1, c]);
        let k = lin(&n, "to_k").reshape(&[b, h * w, 1, c]);
        let v = lin(&n, "to_v").reshape(&[b, h * w, 1, c]);
        let a = ops::scaled_dot_product_attention(&q, &k, &v, 1.0 / (c as f32).sqrt(), false, None).reshape(&[b, h * w, c]);
        let o = lin(&a, "to_out.0").reshape(&[b, h, w, c]);
        ops::add(x, &o)
    }

    fn mid(&self, x: &Var, prefix: &str) -> Var {
        let x = self.resnet(x, &format!("{prefix}.mid_block.resnets.0"));
        let x = if self.config.mid_block_add_attention {
            self.attention(&x, &format!("{prefix}.mid_block.attentions.0"))
        } else {
            x
        };
        self.resnet(&x, &format!("{prefix}.mid_block.resnets.1"))
    }

    /// Decode raw VAE latents `[B, H/8, W/8, latent]` (NHWC) to images in `[-1, 1]`, `[B, H, W, 3]`.
    pub fn decode_raw(&self, z: &Var) -> Var {
        let mut x = z.clone();
        if self.config.use_post_quant_conv {
            x = self.conv(&x, "post_quant_conv", 1, (0, 0, 0, 0));
        }
        x = self.conv3(&x, "decoder.conv_in");
        x = self.mid(&x, "decoder");
        let blocks = self.config.decoder_block_out_channels.as_ref().unwrap_or(&self.config.block_out_channels).len();
        for i in 0..blocks {
            for j in 0..self.config.layers_per_block + 1 {
                x = self.resnet(&x, &format!("decoder.up_blocks.{i}.resnets.{j}"));
            }
            if i + 1 < blocks {
                x = ops::upsample(&x, 2.0, 2.0, false);
                x = self.conv3(&x, &format!("decoder.up_blocks.{i}.upsamplers.0.conv"));
            }
        }
        x = ops::silu(&self.norm(&x, "decoder.conv_norm_out"));
        self.conv3(&x, "decoder.conv_out")
    }

    /// Encode images in `[-1, 1]` (`[B, H, W, 3]`) to raw latent means `[B, H/8, W/8, latent]`.
    pub fn encode_raw(&self, img: &Var) -> Var {
        let mut x = self.conv3(img, "encoder.conv_in");
        let blocks = self.config.block_out_channels.len();
        for i in 0..blocks {
            for j in 0..self.config.layers_per_block {
                x = self.resnet(&x, &format!("encoder.down_blocks.{i}.resnets.{j}"));
            }
            if i + 1 < blocks {
                x = self.conv(&x, &format!("encoder.down_blocks.{i}.downsamplers.0.conv"), 2, (0, 1, 0, 1));
            }
        }
        x = self.mid(&x, "encoder");
        x = ops::silu(&self.norm(&x, "encoder.conv_norm_out"));
        x = self.conv3(&x, "encoder.conv_out");
        if self.config.use_quant_conv {
            x = self.conv(&x, "quant_conv", 1, (0, 0, 0, 0));
        }
        // Posterior mode = the mean half of the moments.
        x.narrow(3, 0, self.config.latent_channels).contiguous()
    }

    /// Patchified, batch-normalized latent (transformer input space) -> raw latent, NHWC.
    /// `packed`: `[B, h, w, 4*latent]` where channel index = `c*4 + dy*2 + dx`.
    pub fn unpack_latents(&self, packed: &Var) -> Var {
        let s = packed.shape();
        let (b, h, w, c4) = (s[0], s[1], s[2], s[3]);
        let lc = c4 / 4;
        let g = packed.graph();
        let mean = g.from_f32(&self.bn_mean, &[1, 1, 1, c4], self.device, self.dtype);
        let std = g.from_f32(&self.bn_std, &[1, 1, 1, c4], self.device, self.dtype);
        let x = ops::add(&ops::mul(packed, &std), &mean);
        // [B, h, w, c, dy, dx] -> [B, h, dy, w, dx, c]; do it in two 4-D permutes.
        let x = x.reshape(&[b * h * w, lc, 2, 2]).permute(&[0, 2, 3, 1]).contiguous(); // [bhw, dy, dx, c]
        let x = x.reshape(&[b * h, w, 2, 2 * lc]).permute(&[0, 2, 1, 3]).contiguous(); // [bh, dy, w, dx*c]
        x.reshape(&[b, 2 * h, 2 * w, lc])
    }

    /// Raw latent `[B, H8, W8, latent]` -> patchified, batch-normalized `[B, H8/2, W8/2, 4*latent]`.
    pub fn pack_latents(&self, raw: &Var) -> Var {
        let s = raw.shape();
        let (b, h2, w2, lc) = (s[0], s[1], s[2], s[3]);
        let (h, w) = (h2 / 2, w2 / 2);
        let x = raw.reshape(&[b * h, 2, w, 2 * lc]).permute(&[0, 2, 1, 3]).contiguous(); // [bh, w, dy, dx*c]
        let x = x.reshape(&[b * h * w, 2, 2, lc]).permute(&[0, 3, 1, 2]).contiguous(); // [bhw, c, dy, dx]
        let x = x.reshape(&[b, h, w, 4 * lc]);
        let g = raw.graph();
        let c4 = 4 * lc;
        let mean = g.from_f32(&self.bn_mean, &[1, 1, 1, c4], self.device, self.dtype);
        let inv: Vec<f32> = self.bn_std.iter().map(|s| 1.0 / s).collect();
        let inv = g.from_f32(&inv, &[1, 1, 1, c4], self.device, self.dtype);
        ops::mul(&ops::sub(&x, &mean), &inv)
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn dtype(&self) -> DataType {
        self.dtype
    }
}
