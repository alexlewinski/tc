//! FLUX.2 autoencoder (KL VAE with 32 latent channels, 2x2 patchified + batch-normalized
//! latents for the transformer).
//!
//! Implemented from the public architecture (Apache-2.0 reference: Hugging Face `diffusers`
//! `AutoencoderKLFlux2`). Weights use diffusers names. All activations are NHWC.

use std::path::Path;

use anyhow::{bail, Context, Result};
use nnc::{ops, DataType, Device, Graph, Var};
use serde::Deserialize;

use crate::ckpt::{CkptFile, MappedCkpt, Recipe};
use crate::weights::{conv_weight_to_nhwc, upload, Params, SafeTensorsFiles, TensorSource};

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

/// Diffusers module names of the encoder and decoder in the order Draw Things numbers them
/// (`__encoder__[t-K-0]` = weight, `[t-K-1]` = bias of the K-th module). The order was derived by
/// matching every tensor of `flux_2_vae_f16.ckpt` against the diffusers release by value.
pub fn dt_module_order(config: &Flux2VaeConfig) -> (Vec<String>, Vec<String>) {
    fn resnet(v: &mut Vec<String>, p: &str, cin: usize, cout: usize) {
        for m in ["norm1", "conv1", "norm2", "conv2"] {
            v.push(format!("{p}.{m}"));
        }
        if cin != cout {
            v.push(format!("{p}.conv_shortcut"));
        }
    }
    fn mid(v: &mut Vec<String>, p: &str, c: usize, attn: bool) {
        resnet(v, &format!("{p}.mid_block.resnets.0"), c, c);
        if attn {
            for m in ["group_norm", "to_q", "to_k", "to_v", "to_out.0"] {
                v.push(format!("{p}.mid_block.attentions.0.{m}"));
            }
        }
        resnet(v, &format!("{p}.mid_block.resnets.1"), c, c);
    }
    let ch = &config.block_out_channels;
    let mut enc = vec!["encoder.conv_in".to_string()];
    let mut c = ch[0];
    for (b, &out) in ch.iter().enumerate() {
        for j in 0..config.layers_per_block {
            resnet(&mut enc, &format!("encoder.down_blocks.{b}.resnets.{j}"), c, out);
            c = out;
        }
        if b + 1 < ch.len() {
            enc.push(format!("encoder.down_blocks.{b}.downsamplers.0.conv"));
        }
    }
    mid(&mut enc, "encoder", c, config.mid_block_add_attention);
    enc.extend(["encoder.conv_norm_out".to_string(), "encoder.conv_out".into()]);
    if config.use_quant_conv {
        enc.push("quant_conv".into());
    }
    let dch: Vec<usize> = config.decoder_block_out_channels.clone().unwrap_or_else(|| ch.clone()).into_iter().rev().collect();
    let mut dec = vec![];
    if config.use_post_quant_conv {
        dec.push("post_quant_conv".to_string());
    }
    dec.push("decoder.conv_in".into());
    let mut c = dch[0];
    mid(&mut dec, "decoder", c, config.mid_block_add_attention);
    for (b, &out) in dch.iter().enumerate() {
        for j in 0..config.layers_per_block + 1 {
            resnet(&mut dec, &format!("decoder.up_blocks.{b}.resnets.{j}"), c, out);
            c = out;
        }
        if b + 1 < dch.len() {
            dec.push(format!("decoder.up_blocks.{b}.upsamplers.0.conv"));
        }
    }
    dec.extend(["decoder.conv_norm_out".to_string(), "decoder.conv_out".into()]);
    (enc, dec)
}

/// Infer the VAE configuration of a Draw Things checkpoint from its tensor shapes: the
/// candidate (blocks, layers per block) whose module order fits every tensor.
pub fn dt_config(file: &CkptFile) -> Result<Flux2VaeConfig> {
    let shape = |scope: &str, k: usize| file.info(&format!("{scope}[t-{k}-0]")).map(|i| i.shape.clone());
    let count = |scope: &str| (0..).take_while(|k| shape(scope, *k).is_some()).count();
    let (ne, ndec) = (count("__encoder__"), count("__decoder__"));
    for lpb in 1..=3 {
        for blocks in 1..=6 {
            // Channels from the first conv of each block's first resnet, walking the encoder.
            let mut cfg = Flux2VaeConfig { layers_per_block: lpb, ..Default::default() };
            let mut channels = vec![];
            let mut k = 1; // after conv_in
            let mut c = match shape("__encoder__", 0) {
                Some(s) => s[0],
                None => bail!("{}: no encoder tensors", file.path().display()),
            };
            let mut ok = true;
            for b in 0..blocks {
                let mut out = c;
                for _ in 0..lpb {
                    let Some(conv1) = shape("__encoder__", k + 1) else {
                        ok = false;
                        break;
                    };
                    out = conv1[0];
                    k += if out != c { 5 } else { 4 };
                    c = out;
                }
                channels.push(out);
                if b + 1 < blocks {
                    k += 1;
                }
            }
            if !ok {
                continue;
            }
            cfg.block_out_channels = channels;
            cfg.latent_channels = shape("__decoder__", 0).map(|s| s[1]).unwrap_or(32);
            // Not visible in the shapes: FLUX.2 uses 32 groups; small test VAEs use c0 / 2.
            cfg.norm_num_groups = cfg.norm_num_groups.min(cfg.block_out_channels[0] / 2).max(1);
            let (enc, dec) = dt_module_order(&cfg);
            if enc.len() == ne && dec.len() == ndec && check_shapes(file, &enc, &dec) {
                return Ok(cfg);
            }
        }
    }
    bail!("{}: tensors do not match a FLUX.2 VAE layout", file.path().display())
}

fn check_shapes(file: &CkptFile, enc: &[String], dec: &[String]) -> bool {
    for (scope, modules) in [("__encoder__", enc), ("__decoder__", dec)] {
        for (k, m) in modules.iter().enumerate() {
            let Some(w) = file.info(&format!("{scope}[t-{k}-0]")) else { return false };
            let conv = m.contains("conv") && !m.ends_with("conv_norm_out");
            if conv != (w.shape.len() == 4 && w.shape[0] > 1 && !m.contains("norm")) && !m.contains(".to_") {
                return false;
            }
            if m.ends_with("conv_shortcut") && (w.shape[2] != 1 || w.shape[3] != 1) {
                return false;
            }
        }
    }
    true
}

/// Map a Draw Things VAE checkpoint (`__encoder__` / `__decoder__`) onto diffusers names.
/// Latent statistics (`bn.running_mean` / `bn.running_var`) are taken from tensors of those names
/// if the checkpoint has them (Draw Things' own files do not; see `flux2_vae_bn`).
pub fn dt_source(file: CkptFile, config: &Flux2VaeConfig) -> Result<MappedCkpt> {
    let (enc, dec) = dt_module_order(config);
    let mut r = std::collections::BTreeMap::new();
    for (scope, modules) in [("__encoder__", &enc), ("__decoder__", &dec)] {
        for (k, m) in modules.iter().enumerate() {
            for (j, param) in ["weight", "bias"].iter().enumerate() {
                let name = format!("{scope}[t-{k}-{j}]");
                let info = file.info(&name).with_context(|| format!("{}: missing {name} ({m}.{param})", file.path().display()))?;
                let mut shape = info.shape.clone();
                let is_norm = m.ends_with("norm1") || m.ends_with("norm2") || m.ends_with("group_norm") || m.ends_with("conv_norm_out");
                let is_linear = m.contains(".attentions.0.to_");
                if is_norm || (j == 1) {
                    shape = vec![shape.iter().product()];
                } else if is_linear {
                    shape = vec![shape[0], shape[1..].iter().product()];
                }
                r.insert(format!("{m}.{param}"), Recipe { parts: vec![name], shape });
            }
        }
    }
    for bn in ["bn.running_mean", "bn.running_var"] {
        if let Some(info) = file.info(bn) {
            r.insert(bn.to_string(), Recipe::direct(bn, &[info.numel()]));
        }
    }
    let m = MappedCkpt::new(file, r)?;
    let unused = m.unused();
    anyhow::ensure!(unused.is_empty(), "{}: unexpected VAE tensors {:?}", m.file.path().display(), &unused[..unused.len().min(5)]);
    Ok(m)
}

/// The tensors of a Draw Things VAE checkpoint (positional names, shapes as in Draw Things'
/// files, f32) from diffusers weights; the inverse of [`dt_source`]. With `with_bn`, the latent
/// statistics are included as `bn.running_mean` / `bn.running_var` (Draw Things omits them).
pub fn dt_tensors(files: &dyn TensorSource, config: &Flux2VaeConfig, with_bn: bool) -> Result<Vec<(String, nnc::HostTensor)>> {
    let (enc, dec) = dt_module_order(config);
    let mut out = vec![];
    for (scope, modules) in [("__encoder__", &enc), ("__decoder__", &dec)] {
        for (k, m) in modules.iter().enumerate() {
            for (j, param) in ["weight", "bias"].iter().enumerate() {
                let t = files.read(&format!("{m}.{param}"), DataType::F32)?;
                let n = t.numel();
                let is_norm = m.ends_with("norm1") || m.ends_with("norm2") || m.ends_with("group_norm") || m.ends_with("conv_norm_out");
                let t = if is_norm {
                    t.reshape(&[1, n, 1, 1])
                } else if j == 0 && t.shape().len() == 2 {
                    let s = t.shape().to_vec();
                    t.reshape(&[s[0], s[1], 1, 1])
                } else {
                    t
                };
                out.push((format!("{scope}[t-{k}-{j}]"), t));
            }
        }
    }
    if with_bn {
        for bn in ["bn.running_mean", "bn.running_var"] {
            out.push((bn.to_string(), files.read(bn, DataType::F32)?));
        }
    }
    Ok(out)
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
        if dir.is_file() && crate::ckpt::is_ckpt(dir) {
            let file = CkptFile::open(dir)?;
            let config = dt_config(&file)?;
            let files = dt_source(file, &config)?;
            return Flux2Vae::load_from(graph, &files, config, device, dtype);
        }
        let files = if dir.is_dir() { SafeTensorsFiles::open_dir(dir)? } else { SafeTensorsFiles::open(&[dir])? };
        Flux2Vae::load_from(graph, &files, config, device, dtype)
    }

    pub fn load_from(graph: &Graph, files: &dyn TensorSource, config: Flux2VaeConfig, device: Device, dtype: DataType) -> Result<Flux2Vae> {
        let prefix = if files.contains("vae.bn.running_mean") { "vae." } else { "" };
        let mut params = Params::new();
        for key in files.names() {
            let Some(name) = key.strip_prefix(prefix) else { continue };
            if name.starts_with("bn.") {
                continue;
            }
            let key = key.as_str();
            let shape = files.shape(key).unwrap();
            let v = if shape.len() == 4 {
                if device.is_gpu() {
                    // GPU convolution kernels take PyTorch's OIHW layout (tagged NCHW).
                    let host = files.read(key, dtype)?;
                    graph.from_host_with_format(&host, device, nnc::Format::Nchw)
                } else {
                    let (w, s) = conv_weight_to_nhwc(&files.read_f32(key)?, &shape);
                    upload(graph, &nnc::HostTensor::from_f32(&w, &s), device, dtype)
                }
            } else {
                upload(graph, &files.read(key, DataType::F32)?, device, dtype)
            };
            params.insert(name.to_string(), v);
        }
        let (mean, var) = if files.contains(&format!("{prefix}bn.running_mean")) {
            (
                files.read_f32(&format!("{prefix}bn.running_mean")).context("vae bn.running_mean")?,
                files.read_f32(&format!("{prefix}bn.running_var")).context("vae bn.running_var")?,
            )
        } else {
            // Draw Things VAE checkpoints do not store the latent statistics.
            anyhow::ensure!(
                4 * config.latent_channels == super::flux2_vae_bn::RUNNING_MEAN.len(),
                "VAE without latent statistics and {} latent channels (FLUX.2 has 32)",
                config.latent_channels
            );
            (super::flux2_vae_bn::RUNNING_MEAN.to_vec(), super::flux2_vae_bn::RUNNING_VAR.to_vec())
        };
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
