//! FLUX.2 rectified-flow transformer (dev / klein variants).
//!
//! Implemented from the public FLUX.2 architecture description and the Apache-2.0 reference
//! implementations (Black Forest Labs `flux2`, Hugging Face `diffusers`).
//!
//! Internally parameters use the original (BFL) names, which fuse Q/K/V projections:
//!
//! * `img_in`, `txt_in`, `time_in.{in,out}_layer`, optional `guidance_in.{in,out}_layer`
//! * `double_stream_modulation_{img,txt}.lin`, `single_stream_modulation.lin`
//! * `double_blocks.N.{img,txt}_attn.{qkv,proj}`, `.{img,txt}_attn.norm.{query,key}_norm.scale`,
//!   `.{img,txt}_mlp.{0,2}`
//! * `single_blocks.N.linear1`, `.linear2_attn` / `.linear2_mlp` (the attention and MLP column
//!   halves of BFL's `linear2`, kept apart so 8-bit Draw Things weights load without
//!   re-quantization), `.norm.{query,key}_norm.scale`
//! * `final_layer.adaLN_modulation.1` (shift, scale), `final_layer.linear`

use std::path::Path;

use anyhow::{bail, Context, Result};
use nnc::{ops, DataType, Device, Graph, HostTensor, Var};
use serde::Deserialize;

use std::collections::HashMap;

use crate::lora::{linear_with_adapters, Adapter, LoraTarget, TargetResolver};
use crate::ckpt::{CkptFile, CkptTensor, MappedCkpt, Recipe as CkptRecipe};
use crate::weights::{upload, Params, SafeTensorsFiles, TensorSource};

/// Architecture hyper-parameters (diffusers `config.json` field names).
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Flux2Config {
    #[serde(default = "d_in")]
    pub in_channels: usize,
    #[serde(default)]
    pub out_channels: Option<usize>,
    #[serde(default = "d_layers")]
    pub num_layers: usize,
    #[serde(default = "d_single")]
    pub num_single_layers: usize,
    #[serde(default = "d_head_dim")]
    pub attention_head_dim: usize,
    #[serde(default = "d_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "d_joint")]
    pub joint_attention_dim: usize,
    #[serde(default = "d_tg")]
    pub timestep_guidance_channels: usize,
    #[serde(default = "d_mlp")]
    pub mlp_ratio: f32,
    #[serde(default = "d_axes")]
    pub axes_dims_rope: Vec<usize>,
    #[serde(default = "d_theta")]
    pub rope_theta: f64,
    #[serde(default = "d_eps")]
    pub eps: f32,
    #[serde(default = "d_true")]
    pub guidance_embeds: bool,
}

fn d_in() -> usize {
    128
}
fn d_layers() -> usize {
    8
}
fn d_single() -> usize {
    48
}
fn d_head_dim() -> usize {
    128
}
fn d_heads() -> usize {
    48
}
fn d_joint() -> usize {
    15360
}
fn d_tg() -> usize {
    256
}
fn d_mlp() -> f32 {
    3.0
}
fn d_axes() -> Vec<usize> {
    vec![32, 32, 32, 32]
}
fn d_theta() -> f64 {
    2000.0
}
fn d_eps() -> f32 {
    1e-6
}
fn d_true() -> bool {
    true
}

impl Flux2Config {
    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }
    pub fn mlp_hidden(&self) -> usize {
        (self.inner_dim() as f32 * self.mlp_ratio) as usize
    }
    pub fn out_channels(&self) -> usize {
        self.out_channels.unwrap_or(self.in_channels)
    }

    /// Infer the configuration from canonical (BFL) tensor shapes.
    pub fn infer(shape: &dyn Fn(&str) -> Option<Vec<usize>>) -> Result<Flux2Config> {
        let img_in = shape("img_in.weight").context("missing img_in.weight")?;
        let txt_in = shape("txt_in.weight").context("missing txt_in.weight")?;
        let inner = img_in[0];
        let mut num_layers = 0;
        while shape(&format!("double_blocks.{num_layers}.img_attn.qkv.weight")).is_some() {
            num_layers += 1;
        }
        let mut num_single = 0;
        while shape(&format!("single_blocks.{num_single}.linear1.weight")).is_some() {
            num_single += 1;
        }
        let head_dim = shape("double_blocks.0.img_attn.norm.query_norm.scale")
            .or_else(|| shape("single_blocks.0.norm.query_norm.scale"))
            .map(|s| s[0])
            .unwrap_or(128);
        let mlp_hidden = shape("double_blocks.0.img_mlp.2.weight").map(|s| s[1]).unwrap_or(inner * 3);
        let out = shape("final_layer.linear.weight").map(|s| s[0]);
        let tg = shape("time_in.in_layer.weight").map(|s| s[1]).unwrap_or(256);
        let rope_dim = head_dim / 4;
        Ok(Flux2Config {
            in_channels: img_in[1],
            out_channels: out,
            num_layers,
            num_single_layers: num_single,
            attention_head_dim: head_dim,
            num_attention_heads: inner / head_dim,
            joint_attention_dim: txt_in[1],
            timestep_guidance_channels: tg,
            mlp_ratio: mlp_hidden as f32 / inner as f32,
            axes_dims_rope: vec![rope_dim; 4],
            rope_theta: 2000.0,
            eps: 1e-6,
            guidance_embeds: shape("guidance_in.in_layer.weight").is_some(),
        })
    }

    /// All canonical parameter names.
    pub fn parameter_names(&self) -> Vec<String> {
        let mut v = vec![
            "img_in.weight".to_string(),
            "txt_in.weight".into(),
            "time_in.in_layer.weight".into(),
            "time_in.out_layer.weight".into(),
            "double_stream_modulation_img.lin.weight".into(),
            "double_stream_modulation_txt.lin.weight".into(),
            "single_stream_modulation.lin.weight".into(),
            "final_layer.adaLN_modulation.1.weight".into(),
            "final_layer.linear.weight".into(),
        ];
        if self.guidance_embeds {
            v.push("guidance_in.in_layer.weight".into());
            v.push("guidance_in.out_layer.weight".into());
        }
        for i in 0..self.num_layers {
            for s in ["img", "txt"] {
                for n in [
                    "attn.qkv.weight",
                    "attn.proj.weight",
                    "attn.norm.query_norm.scale",
                    "attn.norm.key_norm.scale",
                    "mlp.0.weight",
                    "mlp.2.weight",
                ] {
                    v.push(format!("double_blocks.{i}.{s}_{n}"));
                }
            }
        }
        for i in 0..self.num_single_layers {
            for n in ["linear1.weight", "linear2_attn.weight", "linear2_mlp.weight", "norm.query_norm.scale", "norm.key_norm.scale"] {
                v.push(format!("single_blocks.{i}.{n}"));
            }
        }
        v
    }
}

/// How a canonical parameter is assembled from the source checkpoint.
#[derive(Clone, Debug)]
enum Recipe {
    Direct(String),
    ConcatRows(Vec<String>),
    SwapHalves(String),
    /// Columns `start..end` of a 2-D tensor.
    Cols(String, usize, usize),
}

/// Where FLUX.2 weights come from and in which naming scheme.
pub struct Flux2Source {
    files: Box<dyn TensorSource>,
    diffusers: bool,
    prefix: String,
    /// Weights come from a Draw Things checkpoint (already quantized where it chose to).
    draw_things: bool,
}

impl Flux2Source {
    pub fn new(files: SafeTensorsFiles) -> Result<Flux2Source> {
        for prefix in ["", "model.diffusion_model.", "diffusion_model.", "transformer."] {
            if files.contains(&format!("{prefix}img_in.weight")) {
                return Ok(Flux2Source { files: Box::new(files), diffusers: false, prefix: prefix.into(), draw_things: false });
            }
            if files.contains(&format!("{prefix}x_embedder.weight")) {
                return Ok(Flux2Source { files: Box::new(files), diffusers: true, prefix: prefix.into(), draw_things: false });
            }
        }
        bail!("not a FLUX.2 transformer checkpoint (no img_in / x_embedder weight)")
    }

    /// A Draw Things checkpoint (`__dit__[t-...]` tensors), exposed under canonical names.
    pub fn from_ckpt(file: CkptFile) -> Result<(Flux2Source, Flux2Config)> {
        let mapped = dt_mapping(file)?;
        let source = Flux2Source { files: Box::new(mapped), diffusers: false, prefix: String::new(), draw_things: true };
        let config = Flux2Config::infer(&|n| source.shape(n))?;
        Ok((source, config))
    }

    pub fn is_draw_things(&self) -> bool {
        self.draw_things
    }

    /// Whether any weight is stored 8-bit palettized.
    pub fn has_palettized(&self) -> bool {
        self.draw_things && self.files.read_raw("double_blocks.0.img_attn.proj.weight").map(|t| matches!(t, CkptTensor::Palettized { .. })).unwrap_or(false)
    }

    fn recipe(&self, canonical: &str) -> Recipe {
        let p = &self.prefix;
        if let Some((stem, half)) = canonical.strip_suffix(".weight").and_then(|c| {
            c.strip_suffix(".linear2_attn").map(|s| (s, 0)).or_else(|| c.strip_suffix(".linear2_mlp").map(|s| (s, 1)))
        }) {
            if !self.draw_things {
                let n = stem.rsplit('.').next().unwrap_or("0");
                let key = if self.diffusers {
                    format!("{p}single_transformer_blocks.{n}.attn.to_out.weight")
                } else {
                    format!("{p}{stem}.linear2.weight")
                };
                let total = self.files.shape(&key).map(|s| s[1]).unwrap_or(0);
                let inner = self.files.shape(&key).map(|s| s[0]).unwrap_or(0);
                return if half == 0 { Recipe::Cols(key, 0, inner) } else { Recipe::Cols(key, inner, total) };
            }
        }
        if !self.diffusers {
            return Recipe::Direct(format!("{p}{canonical}"));
        }
        let parts: Vec<&str> = canonical.split('.').collect();
        let direct = |s: String| Recipe::Direct(format!("{p}{s}"));
        match parts[0] {
            "img_in" => direct("x_embedder.weight".into()),
            "txt_in" => direct("context_embedder.weight".into()),
            "time_in" | "guidance_in" => {
                let emb = if parts[0] == "time_in" { "timestep_embedder" } else { "guidance_embedder" };
                let lin = if parts[1] == "in_layer" { "linear_1" } else { "linear_2" };
                direct(format!("time_guidance_embed.{emb}.{lin}.{}", parts[2]))
            }
            "double_stream_modulation_img" | "double_stream_modulation_txt" | "single_stream_modulation" => {
                direct(format!("{}.linear.{}", parts[0], parts[2]))
            }
            "final_layer" => {
                if parts[1] == "linear" {
                    direct(format!("proj_out.{}", parts[2]))
                } else {
                    Recipe::SwapHalves(format!("{p}norm_out.linear.{}", parts[3]))
                }
            }
            "double_blocks" => {
                let n = parts[1];
                let rest = parts[2..].join(".");
                let b = format!("{p}transformer_blocks.{n}");
                let param = parts.last().unwrap();
                let param = if *param == "scale" { "weight" } else { param };
                match rest.as_str() {
                    r if r.starts_with("img_attn.qkv") => Recipe::ConcatRows(vec![
                        format!("{b}.attn.to_q.{param}"),
                        format!("{b}.attn.to_k.{param}"),
                        format!("{b}.attn.to_v.{param}"),
                    ]),
                    r if r.starts_with("txt_attn.qkv") => Recipe::ConcatRows(vec![
                        format!("{b}.attn.add_q_proj.{param}"),
                        format!("{b}.attn.add_k_proj.{param}"),
                        format!("{b}.attn.add_v_proj.{param}"),
                    ]),
                    r => {
                        let base = r.rsplit_once('.').map(|x| x.0).unwrap_or(r);
                        let mapped = match base {
                            "img_attn.proj" => "attn.to_out.0",
                            "txt_attn.proj" => "attn.to_add_out",
                            "img_attn.norm.query_norm" => "attn.norm_q",
                            "img_attn.norm.key_norm" => "attn.norm_k",
                            "txt_attn.norm.query_norm" => "attn.norm_added_q",
                            "txt_attn.norm.key_norm" => "attn.norm_added_k",
                            "img_mlp.0" => "ff.linear_in",
                            "img_mlp.2" => "ff.linear_out",
                            "txt_mlp.0" => "ff_context.linear_in",
                            "txt_mlp.2" => "ff_context.linear_out",
                            other => other,
                        };
                        Recipe::Direct(format!("{b}.{mapped}.{param}"))
                    }
                }
            }
            "single_blocks" => {
                let n = parts[1];
                let rest = parts[2..parts.len() - 1].join(".");
                let param = parts.last().unwrap();
                let param = if *param == "scale" { "weight" } else { param };
                let mapped = match rest.as_str() {
                    "linear1" => "attn.to_qkv_mlp_proj",
                    "norm.query_norm" => "attn.norm_q",
                    "norm.key_norm" => "attn.norm_k",
                    other => other,
                };
                direct(format!("single_transformer_blocks.{n}.{mapped}.{param}"))
            }
            _ => direct(canonical.to_string()),
        }
    }

    pub fn shape(&self, canonical: &str) -> Option<Vec<usize>> {
        match self.recipe(canonical) {
            Recipe::Direct(k) | Recipe::SwapHalves(k) => self.files.shape(&k),
            Recipe::Cols(k, a, b) => self.files.shape(&k).map(|s| vec![s[0], b - a]),
            Recipe::ConcatRows(keys) => {
                let mut rows = 0;
                let mut rest = None;
                for k in &keys {
                    let s = self.files.shape(k)?;
                    rows += s[0];
                    rest = Some(s[1..].to_vec());
                }
                let mut s = vec![rows];
                s.extend(rest.unwrap_or_default());
                Some(s)
            }
        }
    }

    /// Read a canonical parameter on the host in `dtype`.
    pub fn read(&self, canonical: &str, dtype: DataType) -> Result<HostTensor> {
        match self.recipe(canonical) {
            Recipe::Direct(k) => self.files.read(&k, dtype),
            Recipe::ConcatRows(keys) => {
                let parts: Vec<HostTensor> = keys.iter().map(|k| self.files.read(k, dtype)).collect::<Result<_>>()?;
                let mut bytes = Vec::new();
                let mut rows = 0;
                for p in &parts {
                    bytes.extend_from_slice(p.bytes());
                    rows += p.shape()[0];
                }
                let mut shape = vec![rows];
                shape.extend_from_slice(&parts[0].shape()[1..]);
                Ok(HostTensor::from_bytes(bytes, dtype, &shape))
            }
            Recipe::SwapHalves(k) => {
                let t = self.files.read(&k, dtype)?;
                let b = t.bytes();
                let half = b.len() / 2;
                let mut out = Vec::with_capacity(b.len());
                out.extend_from_slice(&b[half..]);
                out.extend_from_slice(&b[..half]);
                Ok(HostTensor::from_bytes(out, dtype, t.shape()))
            }
            Recipe::Cols(k, a, b) => Ok(host_cols(&self.files.read(&k, dtype)?, a, b - a)),
        }
    }

    /// Read a canonical parameter in stored form (Draw Things q8p weights stay palettized).
    pub fn read_raw(&self, canonical: &str) -> Result<CkptTensor> {
        match self.recipe(canonical) {
            Recipe::Direct(k) => self.files.read_raw(&k),
            _ => Ok(CkptTensor::Dense(self.read(canonical, DataType::F32)?)),
        }
    }

    pub fn is_diffusers(&self) -> bool {
        self.diffusers
    }
}

/// Load the transformer weights from a diffusers directory (with `config.json`), a
/// directory of safetensors shards, or a single safetensors file.
pub fn open_source(path: &Path) -> Result<(Flux2Source, Flux2Config)> {
    if path.is_file() && crate::ckpt::is_ckpt(path) {
        return Flux2Source::from_ckpt(CkptFile::open(path)?);
    }
    let files = if path.is_dir() {
        SafeTensorsFiles::open_dir(path)?
    } else {
        SafeTensorsFiles::open(&[path])?
    };
    let source = Flux2Source::new(files)?;
    let config_path = if path.is_dir() { Some(path.join("config.json")) } else { None };
    let config = match config_path.filter(|p| p.exists()) {
        Some(p) => serde_json::from_slice(&std::fs::read(&p)?).with_context(|| format!("parsing {}", p.display()))?,
        None => Flux2Config::infer(&|n| source.shape(n))?,
    };
    Ok((source, config))
}

/// Resolves LoRA module names in any common FLUX.2 naming scheme to canonical targets.
pub struct Flux2LoraResolver {
    pub config: Flux2Config,
}

impl Flux2LoraResolver {
    fn target(param: String, row_offset: usize, rows: Option<usize>) -> LoraTarget {
        LoraTarget { param: format!("{param}.weight"), row_offset, rows, col_offset: 0, cols: None, swap_halves: false }
    }

    /// BFL `linear2` (attention and MLP columns) maps onto the two canonical halves.
    fn linear2(&self, n: &str) -> Vec<LoraTarget> {
        let inner = self.config.inner_dim();
        let mut a = Self::target(format!("single_blocks.{n}.linear2_attn"), 0, None);
        a.cols = Some(inner);
        let mut m = Self::target(format!("single_blocks.{n}.linear2_mlp"), 0, None);
        m.col_offset = inner;
        m.cols = Some(self.config.mlp_hidden());
        vec![a, m]
    }

    /// Targets of a Draw Things module name (`__dit__[t-x_q-3-0]`, `x_q-3`, ...).
    pub fn draw_things(&self, module: &str) -> Vec<LoraTarget> {
        let m = module.trim_start_matches("__dit__[").trim_end_matches(']');
        let m = m.strip_prefix("t-").unwrap_or(m);
        let m = m.strip_suffix("-0").filter(|r| r.rsplit_once('-').map(|(_, i)| i.parse::<usize>().is_ok()).unwrap_or(false)).unwrap_or(m);
        let Some((kind, idx)) = m.rsplit_once('-') else { return vec![] };
        let Ok(i) = idx.parse::<usize>() else { return vec![] };
        let (d, h, nd) = (self.config.inner_dim(), self.config.mlp_hidden(), self.config.num_layers);
        let t = |p: String, k: usize, n: usize| vec![Self::target(p, k * n, Some(n))];
        let whole = |p: String| vec![Self::target(p, 0, None)];
        let stream = |pre: &str| if pre == "x" { "img" } else { "txt" };
        match kind {
            "x_embedder" => whole("img_in".into()),
            "context_embedder" => whole("txt_in".into()),
            "t_embedder_0" => whole("time_in.in_layer".into()),
            "t_embedder_1" => whole("time_in.out_layer".into()),
            "linear" => whole("final_layer.linear".into()),
            k if k.starts_with("x_ada_ln_") => t("double_stream_modulation_img.lin".into(), k[9..].parse().unwrap_or(0), d),
            k if k.starts_with("context_ada_ln_") => t("double_stream_modulation_txt.lin".into(), k[15..].parse().unwrap_or(0), d),
            k if k.starts_with("single_ada_ln_") => t("single_stream_modulation.lin".into(), k[14..].parse().unwrap_or(0), d),
            k if k.starts_with("ada_ln_") => t("final_layer.adaLN_modulation.1".into(), 1 - k[7..].parse::<usize>().unwrap_or(0).min(1), d),
            "x_q" | "x_k" | "x_v" | "c_q" | "c_k" | "c_v" => {
                let k = "qkv".find(&kind[2..]).unwrap();
                if kind.starts_with('x') && i >= nd {
                    t(format!("single_blocks.{}.linear1", i - nd), k, d)
                } else {
                    t(format!("double_blocks.{i}.{}_attn.qkv", stream(&kind[..1])), k, d)
                }
            }
            "x_o" if i >= nd => whole(format!("single_blocks.{}.linear2_attn", i - nd)),
            "x_o" | "c_o" => whole(format!("double_blocks.{i}.{}_attn.proj", stream(&kind[..1]))),
            "x_gate_proj" | "c_gate_proj" => t(format!("double_blocks.{i}.{}_mlp.0", stream(&kind[..1])), 0, h),
            "x_up_proj" | "c_up_proj" => t(format!("double_blocks.{i}.{}_mlp.0", stream(&kind[..1])), 1, h),
            "x_down_proj" | "c_down_proj" => whole(format!("double_blocks.{i}.{}_mlp.2", stream(&kind[..1]))),
            "x_w1" => vec![Self::target(format!("single_blocks.{i}.linear1"), 3 * d, Some(h))],
            "x_w3" => vec![Self::target(format!("single_blocks.{i}.linear1"), 3 * d + h, Some(h))],
            "x_w2" => whole(format!("single_blocks.{i}.linear2_mlp")),
            _ => vec![],
        }
    }
}

impl TargetResolver for Flux2LoraResolver {
    fn resolve(&self, module: &str) -> Vec<LoraTarget> {
        let inner = self.config.inner_dim();
        let parts: Vec<&str> = module.split('.').collect();
        let whole = |c: String| vec![Self::target(c, 0, None)];
        let slice = |c: String, i: usize| vec![Self::target(c, i * inner, Some(inner))];
        if module.starts_with("__dit__[") || module.starts_with("t-") {
            return self.draw_things(module);
        }
        match parts.as_slice() {
            ["img_in"] | ["x_embedder"] => whole("img_in".into()),
            ["txt_in"] | ["context_embedder"] => whole("txt_in".into()),
            ["time_in", l] => whole(format!("time_in.{l}")),
            ["guidance_in", l] => whole(format!("guidance_in.{l}")),
            ["time_guidance_embed", e, l] => {
                let base = if *e == "timestep_embedder" { "time_in" } else { "guidance_in" };
                let l = if *l == "linear_1" { "in_layer" } else { "out_layer" };
                whole(format!("{base}.{l}"))
            }
            [m, "lin"] | [m, "linear"] if m.contains("modulation") => whole(format!("{m}.lin")),
            ["final_layer", "linear"] | ["proj_out"] => whole("final_layer.linear".into()),
            ["final_layer", "adaLN_modulation", "1"] => whole("final_layer.adaLN_modulation.1".into()),
            ["norm_out", "linear"] => {
                let mut t = Self::target("final_layer.adaLN_modulation.1".into(), 0, None);
                t.swap_halves = true;
                vec![t]
            }
            ["single_blocks", n, "linear2"] => self.linear2(n),
            ["double_blocks", n, rest @ ..] => whole(format!("double_blocks.{n}.{}", rest.join("."))),
            ["single_blocks", n, rest @ ..] => whole(format!("single_blocks.{n}.{}", rest.join("."))),
            ["transformer_blocks", n, rest @ ..] => {
                let b = format!("double_blocks.{n}");
                match rest.join(".").as_str() {
                    "attn.to_q" => slice(format!("{b}.img_attn.qkv"), 0),
                    "attn.to_k" => slice(format!("{b}.img_attn.qkv"), 1),
                    "attn.to_v" => slice(format!("{b}.img_attn.qkv"), 2),
                    "attn.add_q_proj" => slice(format!("{b}.txt_attn.qkv"), 0),
                    "attn.add_k_proj" => slice(format!("{b}.txt_attn.qkv"), 1),
                    "attn.add_v_proj" => slice(format!("{b}.txt_attn.qkv"), 2),
                    "attn.to_out.0" => whole(format!("{b}.img_attn.proj")),
                    "attn.to_add_out" => whole(format!("{b}.txt_attn.proj")),
                    "ff.linear_in" => whole(format!("{b}.img_mlp.0")),
                    "ff.linear_out" => whole(format!("{b}.img_mlp.2")),
                    "ff_context.linear_in" => whole(format!("{b}.txt_mlp.0")),
                    "ff_context.linear_out" => whole(format!("{b}.txt_mlp.2")),
                    _ => vec![],
                }
            }
            ["single_transformer_blocks", n, rest @ ..] => match rest.join(".").as_str() {
                "attn.to_qkv_mlp_proj" => whole(format!("single_blocks.{n}.linear1")),
                "attn.to_out" => self.linear2(n),
                _ => vec![],
            },
            _ => vec![],
        }
    }

    fn canonical_modules(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .config
            .parameter_names()
            .into_iter()
            .filter_map(|n| n.strip_suffix(".weight").map(|s| s.to_string()))
            .filter(|n| !n.ends_with(".linear2_mlp"))
            .map(|n| n.strip_suffix("_attn").filter(|s| s.ends_with(".linear2")).map(|s| s.to_string()).unwrap_or(n))
            .collect();
        v.dedup();
        v
    }
}

/// Columns `start..start+n` of a 2-D host tensor.
pub fn host_cols(t: &HostTensor, start: usize, n: usize) -> HostTensor {
    let s = t.shape();
    let (rows, cols) = (s[0], s[1..].iter().product::<usize>());
    let es = t.dtype().size();
    let b = t.bytes();
    let mut out = Vec::with_capacity(rows * n * es);
    for r in 0..rows {
        out.extend_from_slice(&b[(r * cols + start) * es..(r * cols + start + n) * es]);
    }
    HostTensor::from_bytes(out, t.dtype(), &[rows, n])
}

fn dt_name(kind: &str, i: usize) -> String {
    format!("__dit__[t-{kind}-{i}-0]")
}

/// Map a Draw Things FLUX.2 checkpoint onto canonical names. The tensor names
/// (`__dit__[t-<kind>-<index>-0]`) were read from the checkpoint index and every tensor was
/// checked against the BFL release (docs/PROVENANCE.md, scripts/ckpt/verify_flux2.py).
pub fn dt_mapping(file: CkptFile) -> Result<MappedCkpt> {
    let shape = |n: &str| file.info(n).map(|i| i.shape.clone()).with_context(|| format!("{}: missing {n}; not a FLUX.2 checkpoint", file.path().display()));
    let count = |k: &str| (0..).take_while(|i| file.contains(&dt_name(k, *i))).count();
    let img_in = shape(&dt_name("x_embedder", 0))?;
    let d = img_in[0];
    let nd = count("c_q");
    let ns = count("x_q") - nd;
    let mut r = std::collections::BTreeMap::new();
    let mut add = |canonical: String, kinds: &[(&str, usize)]| -> Result<()> {
        let parts: Vec<String> = kinds.iter().map(|(k, i)| dt_name(k, *i)).collect();
        let mut rows = 0;
        let mut rest = vec![];
        for p in &parts {
            let s = shape(p)?;
            rows += s[0];
            rest = s[1..].to_vec();
        }
        let mut sh = vec![rows];
        sh.extend(rest);
        r.insert(canonical, CkptRecipe { parts, shape: sh });
        Ok(())
    };
    add("img_in.weight".into(), &[("x_embedder", 0)])?;
    add("txt_in.weight".into(), &[("context_embedder", 0)])?;
    add("time_in.in_layer.weight".into(), &[("t_embedder_0", 0)])?;
    add("time_in.out_layer.weight".into(), &[("t_embedder_1", 0)])?;
    add("final_layer.linear.weight".into(), &[("linear", 0)])?;
    // Draw Things stores the final modulation as (scale, shift); BFL as (shift, scale).
    add("final_layer.adaLN_modulation.1.weight".into(), &[("ada_ln_1", 0), ("ada_ln_0", 0)])?;
    let ada = |p: &str, n: usize| (0..n).map(|k| (format!("{p}_ada_ln_{k}"), 0)).collect::<Vec<_>>();
    for (canonical, p, n) in [
        ("double_stream_modulation_img.lin.weight", "x", 6),
        ("double_stream_modulation_txt.lin.weight", "context", 6),
        ("single_stream_modulation.lin.weight", "single", 3),
    ] {
        let kinds = ada(p, n);
        let kinds: Vec<(&str, usize)> = kinds.iter().map(|(k, i)| (k.as_str(), *i)).collect();
        add(canonical.into(), &kinds)?;
    }
    for i in 0..nd {
        for (s, x) in [("img", "x"), ("txt", "c")] {
            let k = |n: &str| format!("{x}_{n}");
            let b = format!("double_blocks.{i}.{s}");
            add(format!("{b}_attn.qkv.weight"), &[(&k("q"), i), (&k("k"), i), (&k("v"), i)])?;
            add(format!("{b}_attn.proj.weight"), &[(&k("o"), i)])?;
            add(format!("{b}_attn.norm.query_norm.scale"), &[(&k("norm_q"), i)])?;
            add(format!("{b}_attn.norm.key_norm.scale"), &[(&k("norm_k"), i)])?;
            add(format!("{b}_mlp.0.weight"), &[(&k("gate_proj"), i), (&k("up_proj"), i)])?;
            add(format!("{b}_mlp.2.weight"), &[(&k("down_proj"), i)])?;
        }
    }
    for j in 0..ns {
        let i = nd + j;
        let b = format!("single_blocks.{j}");
        add(format!("{b}.linear1.weight"), &[("x_q", i), ("x_k", i), ("x_v", i), ("x_w1", j), ("x_w3", j)])?;
        add(format!("{b}.linear2_attn.weight"), &[("x_o", i)])?;
        add(format!("{b}.linear2_mlp.weight"), &[("x_w2", j)])?;
        add(format!("{b}.norm.query_norm.scale"), &[("x_norm_q", i)])?;
        add(format!("{b}.norm.key_norm.scale"), &[("x_norm_k", i)])?;
    }
    if d == 0 {
        bail!("bad checkpoint");
    }
    let m = MappedCkpt::new(file, r)?;
    let unused = m.unused();
    if !unused.is_empty() {
        log::warn!("{}: {} tensors not used (e.g. {:?})", m.file.path().display(), unused.len(), &unused[..unused.len().min(3)]);
    }
    Ok(m)
}

/// The FLUX.2 transformer with weights resident on a device.
pub struct Flux2 {
    pub config: Flux2Config,
    pub params: Params,
    pub device: Device,
    pub dtype: DataType,
    /// Linear weights stored 8-bit palettized.
    pub quantized: bool,
    /// Runtime (unmerged) LoRA adapters per parameter.
    pub adapters: HashMap<String, Vec<Adapter>>,
}

/// Linear weights at least this large are quantized in 8-bit mode.
const QUANTIZE_MIN_ELEMENTS: usize = 1 << 16;

/// Upload a weight, palettizing large matrices when `quantize` is set. On CPU the palettized
/// values are expanded back to floats (same numerics, no memory saving).
pub fn upload_weight(graph: &Graph, host: &HostTensor, device: Device, dtype: DataType, quantize: bool) -> Var {
    if quantize && host.shape().len() == 2 && host.numel() >= QUANTIZE_MIN_ELEMENTS {
        // Palettes are f16 unless the model runs in f16 already (Metal has no bf16 palette kernel).
        let src_dtype = if dtype == DataType::F32 || dtype == DataType::BF16 { DataType::F16 } else { dtype };
        let h = host.to_dtype(src_dtype);
        let q = nnc::quant::palettize8_fast(&h, nnc::quant::DEFAULT_BLOCK);
        if device.is_gpu() {
            return graph.from_palettized(&q, src_dtype, host.shape(), 8, nnc::quant::DEFAULT_BLOCK, device);
        }
        let back = nnc::quant::depalettize(&q, src_dtype, host.shape(), 8, nnc::quant::DEFAULT_BLOCK);
        return upload(graph, &back, device, dtype);
    }
    upload(graph, host, device, dtype)
}

/// Upload a parameter read from a source. Palettized (Draw Things q8p) weights are used as-is
/// on GPU when `quantize` is set and expanded otherwise; dense weights from Draw Things
/// checkpoints stay dense (the checkpoint chose not to quantize them).
pub fn upload_param(graph: &Graph, t: CkptTensor, device: Device, dtype: DataType, quantize: bool, keep_dense: bool) -> Var {
    match t {
        CkptTensor::Palettized { data, dtype: pd, shape, block } if quantize && device.is_gpu() && shape.len() == 2 => {
            let numel = shape.iter().product();
            let data = if pd == dtype { data } else { crate::ckpt::convert_palettes(&data, pd, dtype, numel, block) };
            graph.from_palettized(&data, dtype, &shape, 8, block, device)
        }
        t @ CkptTensor::Palettized { .. } => upload(graph, &t.to_host(dtype), device, dtype),
        CkptTensor::Dense(h) if keep_dense => upload(graph, &h, device, dtype),
        CkptTensor::Dense(h) => upload_weight(graph, &h, device, dtype, quantize),
    }
}

/// Sinusoidal timestep embedding (`flip_sin_to_cos=True`, `downscale_freq_shift=0`).
pub fn timestep_embedding(t: &[f32], dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let mut out = vec![0f32; t.len() * dim];
    for (b, tv) in t.iter().enumerate() {
        for i in 0..half {
            let freq = (-(10000f64.ln()) * i as f64 / half as f64).exp();
            let arg = *tv as f64 * freq;
            out[b * dim + i] = arg.cos() as f32;
            out[b * dim + half + i] = arg.sin() as f32;
        }
    }
    out
}

/// Interleaved (cos, sin) rotary table for 4-axis position ids, shaped `[1, T, 1, head_dim]`.
pub fn rope_table(ids: &[[f32; 4]], axes: &[usize], theta: f64) -> (Vec<f32>, Vec<usize>) {
    let head_dim: usize = axes.iter().sum();
    let mut out = vec![0f32; ids.len() * head_dim];
    for (t, pos) in ids.iter().enumerate() {
        let mut o = t * head_dim;
        for (a, dim) in axes.iter().enumerate() {
            for k in 0..dim / 2 {
                let freq = 1.0 / theta.powf((2 * k) as f64 / *dim as f64);
                let angle = pos[a] as f64 * freq;
                out[o] = angle.cos() as f32;
                out[o + 1] = angle.sin() as f32;
                o += 2;
            }
        }
    }
    (out, vec![1, ids.len(), 1, head_dim])
}

/// Position ids for text tokens: `(0, 0, 0, l)`.
pub fn text_ids(len: usize) -> Vec<[f32; 4]> {
    (0..len).map(|l| [0.0, 0.0, 0.0, l as f32]).collect()
}

/// Position ids for an `h x w` latent grid at time coordinate `t`: `(t, y, x, 0)`.
pub fn image_ids(h: usize, w: usize, t: f32) -> Vec<[f32; 4]> {
    let mut v = Vec::with_capacity(h * w);
    for y in 0..h {
        for x in 0..w {
            v.push([t, y as f32, x as f32, 0.0]);
        }
    }
    v
}

/// Precomputed per-step conditioning shared by all blocks.
pub struct Flux2Step {
    temb: Var,
    img_mod: Vec<Var>,
    txt_mod: Vec<Var>,
    single_mod: Vec<Var>,
}

impl Flux2Step {
    /// The first image-stream modulation chunk (shift of the first norm), for diagnostics.
    pub fn img_mod_first(&self) -> &Var {
        &self.img_mod[0]
    }
}

impl Flux2 {
    /// Upload all weights from `source`.
    pub fn load(graph: &Graph, source: &Flux2Source, config: Flux2Config, device: Device, dtype: DataType, quantize: bool) -> Result<Flux2> {
        let mut params = Params::new();
        for name in config.parameter_names() {
            let raw = source.read_raw(&name).with_context(|| format!("loading {name}"))?;
            params.insert(name, upload_param(graph, raw, device, dtype, quantize, source.is_draw_things()));
        }
        Ok(Flux2 { config, params, device, dtype, quantized: quantize, adapters: HashMap::new() })
    }

    /// Re-read one parameter from the source (used to undo LoRA merges).
    pub fn reload_param(&mut self, graph: &Graph, source: &Flux2Source, name: &str) -> Result<()> {
        let raw = source.read_raw(name)?;
        self.params.insert(name.to_string(), upload_param(graph, raw, self.device, self.dtype, self.quantized, source.is_draw_things()));
        Ok(())
    }

    /// Matrix product in the compute type (inputs are converted to it).
    fn lin(&self, x: &Var, name: &str) -> Var {
        let x = x.to_dtype(self.dtype);
        linear_with_adapters(&x, self.params.get(name), self.adapters.get(name))
    }

    /// Type of the residual streams, norms, modulation and gating. With f16 compute (8-bit
    /// palettized weights on Metal) this is f32: FLUX.2's text stream outgrows f16's range
    /// (Qwen3 features reach ~2e4 and grow through the double blocks), which saturated the
    /// whole stream from the first single block on. Matrix products and attention stay f16.
    fn stream_dtype(&self) -> DataType {
        if self.dtype == DataType::F16 {
            DataType::F32
        } else {
            self.dtype
        }
    }

    /// Matrix product with the result in the stream type.
    fn lin_s(&self, x: &Var, name: &str) -> Var {
        self.lin(x, name).to_dtype(self.stream_dtype())
    }

    fn graph(&self) -> &Graph {
        self.params.get("img_in.weight").graph()
    }

    fn modulation(&self, temb_act: &Var, name: &str, sets: usize) -> Vec<Var> {
        let m = self.lin(temb_act, &format!("{name}.weight"));
        let dim = self.config.inner_dim();
        let b = m.dim(0);
        let m = m.reshape(&[b, 1, 3 * sets * dim]);
        (0..3 * sets)
            .map(|i| {
                let c = m.narrow(2, i * dim, dim).contiguous().to_dtype(self.stream_dtype());
                // Scale chunks are used as (1 + scale).
                if i % 3 == 1 {
                    ops::add_scalar(&c, 1.0)
                } else {
                    c
                }
            })
            .collect()
    }

    /// Per-step conditioning from the (already ×1000) timestep and optional guidance.
    pub fn prepare_step(&self, timesteps: &[f32], guidance: Option<&[f32]>) -> Flux2Step {
        let g = self.graph();
        let tg = self.config.timestep_guidance_channels;
        let t = timestep_embedding(timesteps, tg);
        let t = g.from_f32(&t, &[timesteps.len(), tg], self.device, self.dtype);
        let mut temb = self.mlp_embed(&t, "time_in");
        if let (Some(gv), true) = (guidance, self.params.contains("guidance_in.in_layer.weight")) {
            let ge = timestep_embedding(gv, tg);
            let ge = g.from_f32(&ge, &[gv.len(), tg], self.device, self.dtype);
            temb = ops::add(&temb, &self.mlp_embed(&ge, "guidance_in"));
        }
        let act = ops::silu(&temb);
        Flux2Step {
            img_mod: self.modulation(&act, "double_stream_modulation_img.lin", 2),
            txt_mod: self.modulation(&act, "double_stream_modulation_txt.lin", 2),
            single_mod: self.modulation(&act, "single_stream_modulation.lin", 1),
            temb,
        }
    }

    fn mlp_embed(&self, x: &Var, name: &str) -> Var {
        let h = self.lin(x, &format!("{name}.in_layer.weight"));
        let h = ops::silu(&h);
        self.lin(&h, &format!("{name}.out_layer.weight"))
    }

    fn modulate(&self, x: &Var, shift: &Var, scale1p: &Var) -> Var {
        let n = ops::layer_norm(&x.to_dtype(self.stream_dtype()), &[2], self.config.eps, None);
        ops::add(&ops::mul(&n, scale1p), shift)
    }

    /// Split fused `[B, T, 3*inner]` into normalized q, k and v, shaped `[B, T, H, D]`.
    fn qkv(&self, qkv: &Var, q_norm: &str, k_norm: &str) -> (Var, Var, Var) {
        let (b, t) = (qkv.dim(0), qkv.dim(1));
        let h = self.config.num_attention_heads;
        let d = self.config.attention_head_dim;
        let inner = h * d;
        // nnc CPU kernels handle at most 4-D tensors, so split along the flat channel axis.
        let take = |i: usize| qkv.narrow(2, i * inner, inner).contiguous().reshape(&[b, t, h, d]);
        // RMS norm squares its input: compute it in the stream type, attend in the compute type.
        let sdt = self.stream_dtype();
        let norm = |x: Var, w: &str| ops::rms_norm(&x.to_dtype(sdt), &[3], 1e-6, Some(&self.params.get(w).to_dtype(sdt))).to_dtype(self.dtype);
        (norm(take(0), q_norm), norm(take(1), k_norm), take(2).to_dtype(self.dtype))
    }

    fn attention(&self, q: &Var, k: &Var, v: &Var, rope: &Var) -> Var {
        let q = ops::cmul(q, rope);
        let k = ops::cmul(k, rope);
        let d = self.config.attention_head_dim;
        let o = ops::scaled_dot_product_attention(&q, &k, v, 1.0 / (d as f32).sqrt(), false, None);
        let (b, t) = (o.dim(0), o.dim(1));
        o.reshape(&[b, t, self.config.inner_dim()])
    }

    /// KV-variant attention over `[txt, ref, img]`: text and image tokens attend to everything,
    /// the `n_ref` reference tokens only to themselves.
    fn attention_kv(&self, q: &Var, k: &Var, v: &Var, rope: &Var, lt: usize, n_ref: usize) -> Var {
        if n_ref == 0 {
            return self.attention(q, k, v, rope);
        }
        let q = ops::cmul(q, rope);
        let k = ops::cmul(k, rope);
        let d = self.config.attention_head_dim;
        let scale = 1.0 / (d as f32).sqrt();
        let t = q.dim(1);
        let part = |x: &Var, a: usize, n: usize| x.narrow(1, a, n).contiguous();
        let q_main = ops::concat(&[&part(&q, 0, lt), &part(&q, lt + n_ref, t - lt - n_ref)], 1);
        let o_main = ops::scaled_dot_product_attention(&q_main, &k, v, scale, false, None);
        let o_ref = ops::scaled_dot_product_attention(&part(&q, lt, n_ref), &part(&k, lt, n_ref), &part(v, lt, n_ref), scale, false, None);
        let o = ops::concat(&[&part(&o_main, 0, lt), &o_ref, &part(&o_main, lt, t - lt - n_ref)], 1);
        let b = o.dim(0);
        o.reshape(&[b, t, self.config.inner_dim()])
    }

    /// Apply `f` to consecutive token segments (dim 1), each with its own modulation.
    fn segmented(x: &Var, segs: &[(usize, &[Var])], f: impl Fn(&Var, &[Var]) -> Var) -> Var {
        if segs.len() == 1 {
            return f(x, segs[0].1);
        }
        let mut start = 0;
        let mut outs = vec![];
        for (n, m) in segs {
            if *n > 0 {
                outs.push(f(&x.narrow(1, start, *n).contiguous(), m));
            }
            start += n;
        }
        let refs: Vec<&Var> = outs.iter().collect();
        ops::concat(&refs, 1)
    }

    fn swiglu_mlp(&self, x: &Var, w_in: &str, w_out: &str) -> Var {
        let h = self.lin(x, w_in);
        let m = h.dim(2) / 2;
        let a = h.narrow(2, 0, m).contiguous();
        let gate = h.narrow(2, m, m).contiguous();
        self.lin_s(&self.silu_gate(&a, &gate), w_out)
    }

    /// `silu(a) * gate`, computed in the stream type and limited to the compute type's range.
    fn silu_gate(&self, a: &Var, gate: &Var) -> Var {
        let sdt = self.stream_dtype();
        let h = ops::mul(&ops::silu(&a.to_dtype(sdt)), &gate.to_dtype(sdt));
        if self.dtype == DataType::F16 {
            ops::clamp(&h, Some(-65504.0), Some(65504.0))
        } else {
            h
        }
    }

    fn double_block(&self, i: usize, img: Var, txt: Var, step: &Flux2Step, rope: &Var, kv: Option<(&Flux2Step, usize)>) -> (Var, Var) {
        let p = format!("double_blocks.{i}");
        let tm = &step.txt_mod;
        let li = img.dim(1);
        let segs: Vec<(usize, &[Var])> = match kv {
            Some((r, n)) if n > 0 => vec![(n, &r.img_mod[..]), (li - n, &step.img_mod[..])],
            _ => vec![(li, &step.img_mod[..])],
        };
        let n_ref = kv.map(|k| k.1).unwrap_or(0);
        let img_n = Self::segmented(&img, &segs, |x, m| self.modulate(x, &m[0], &m[1]));
        let txt_n = self.modulate(&txt, &tm[0], &tm[1]);
        let iqkv = self.lin(&img_n, &format!("{p}.img_attn.qkv.weight"));
        let tqkv = self.lin(&txt_n, &format!("{p}.txt_attn.qkv.weight"));
        let (iq, ik, iv) = self.qkv(
            &iqkv,
            &format!("{p}.img_attn.norm.query_norm.scale"),
            &format!("{p}.img_attn.norm.key_norm.scale"),
        );
        let (tq, tk, tv) = self.qkv(
            &tqkv,
            &format!("{p}.txt_attn.norm.query_norm.scale"),
            &format!("{p}.txt_attn.norm.key_norm.scale"),
        );
        let q = ops::concat(&[&tq, &iq], 1);
        let k = ops::concat(&[&tk, &ik], 1);
        let v = ops::concat(&[&tv, &iv], 1);
        let lt = txt.dim(1);
        let attn = self.attention_kv(&q, &k, &v, rope, lt, n_ref);
        let txt_attn = attn.narrow(1, 0, lt).contiguous();
        let img_attn = attn.narrow(1, lt, li).contiguous();
        let img_attn = self.lin_s(&img_attn, &format!("{p}.img_attn.proj.weight"));
        let txt_attn = self.lin_s(&txt_attn, &format!("{p}.txt_attn.proj.weight"));
        let img = ops::add(&img, &Self::segmented(&img_attn, &segs, |x, m| ops::mul(x, &m[2])));
        let img_n = Self::segmented(&img, &segs, |x, m| self.modulate(x, &m[3], &m[4]));
        let img_ff = self.swiglu_mlp(&img_n, &format!("{p}.img_mlp.0.weight"), &format!("{p}.img_mlp.2.weight"));
        let img = ops::add(&img, &Self::segmented(&img_ff, &segs, |x, m| ops::mul(x, &m[5])));
        let txt = ops::add(&txt, &ops::mul(&txt_attn, &tm[2]));
        let txt_n = self.modulate(&txt, &tm[3], &tm[4]);
        let txt_ff = self.swiglu_mlp(&txt_n, &format!("{p}.txt_mlp.0.weight"), &format!("{p}.txt_mlp.2.weight"));
        let txt = ops::add(&txt, &ops::mul(&txt_ff, &tm[5]));
        (img, txt)
    }

    fn single_block(&self, i: usize, x: Var, step: &Flux2Step, rope: &Var, lt: usize, kv: Option<(&Flux2Step, usize)>) -> Var {
        let p = format!("single_blocks.{i}");
        let l = x.dim(1);
        let n_ref = kv.map(|k| k.1).unwrap_or(0);
        let segs: Vec<(usize, &[Var])> = match kv {
            Some((r, n)) if n > 0 => vec![(lt, &step.single_mod[..]), (n, &r.single_mod[..]), (l - lt - n, &step.single_mod[..])],
            _ => vec![(l, &step.single_mod[..])],
        };
        let xn = Self::segmented(&x, &segs, |x, m| self.modulate(x, &m[0], &m[1]));
        let h = self.lin(&xn, &format!("{p}.linear1.weight"));
        let inner = self.config.inner_dim();
        let mlp = self.config.mlp_hidden();
        let qkv = h.narrow(2, 0, 3 * inner).contiguous();
        let (q, k, v) = self.qkv(&qkv, &format!("{p}.norm.query_norm.scale"), &format!("{p}.norm.key_norm.scale"));
        let attn = self.attention_kv(&q, &k, &v, rope, lt, n_ref);
        let a = h.narrow(2, 3 * inner, mlp).contiguous();
        let gate = h.narrow(2, 3 * inner + mlp, mlp).contiguous();
        let m = self.silu_gate(&a, &gate);
        let out = ops::add(&self.lin_s(&attn, &format!("{p}.linear2_attn.weight")), &self.lin_s(&m, &format!("{p}.linear2_mlp.weight")));
        ops::add(&x, &Self::segmented(&out, &segs, |x, m| ops::mul(x, &m[2])))
    }

    /// Run the transformer.
    ///
    /// * `img`: packed latents `[B, N, in_channels]` (target tokens first, then reference tokens)
    /// * `txt`: text features `[B, L, joint_attention_dim]`
    /// * `rope`: table from [`rope_table`] for the concatenation `[txt ids, img ids]`
    ///
    /// Returns the prediction for all `N` image tokens: `[B, N, out_channels]`.
    pub fn forward(&self, img: &Var, txt: &Var, step: &Flux2Step, rope: &Var) -> Var {
        self.forward_impl(img, txt, step, rope, None)
    }

    /// Run the FLUX.2 [klein] KV variant with `n_ref` reference tokens placed *first* in `img`
    /// (`[ref, target]`, rope table for `[txt, ref, target]`). Reference tokens are modulated with
    /// `ref_step` (the fixed timestep 0) and attend only to themselves, as in diffusers'
    /// `Flux2KleinKVPipeline` (whose KV cache is an optimization of exactly this computation).
    /// Returns the prediction for the target tokens only.
    pub fn forward_kv(&self, img: &Var, txt: &Var, step: &Flux2Step, ref_step: &Flux2Step, n_ref: usize, rope: &Var) -> Var {
        self.forward_impl(img, txt, step, rope, Some((ref_step, n_ref)))
    }

    fn forward_impl(&self, img: &Var, txt: &Var, step: &Flux2Step, rope: &Var, kv: Option<(&Flux2Step, usize)>) -> Var {
        // TC_DEBUG_STATS=1: activation statistics after every block of the first forward pass.
        let stats = crate::debug::stats_enabled() && FORWARD_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 0;
        let mut img = self.lin_s(img, "img_in.weight");
        let mut txt = self.lin_s(txt, "txt_in.weight");
        if stats {
            crate::debug::print_stats("img_in", &img);
            crate::debug::print_stats("txt_in", &txt);
        }
        for i in 0..self.config.num_layers {
            let (a, b) = self.double_block(i, img, txt, step, rope, kv);
            img = a;
            txt = b;
            if stats {
                crate::debug::print_stats(&format!("double {i} img"), &img);
                crate::debug::print_stats(&format!("double {i} txt"), &txt);
            }
        }
        let lt = txt.dim(1);
        let n_ref = kv.map(|k| k.1).unwrap_or(0);
        let li = img.dim(1) - n_ref;
        let mut x = ops::concat(&[&txt, &img], 1);
        drop(txt);
        drop(img);
        for i in 0..self.config.num_single_layers {
            x = self.single_block(i, x, step, rope, lt, kv);
            if stats {
                crate::debug::print_stats(&format!("single {i}"), &x);
            }
        }
        let x = x.narrow(1, lt + n_ref, li).contiguous();
        let emb = self.lin(&ops::silu(&step.temb), "final_layer.adaLN_modulation.1.weight");
        let dim = self.config.inner_dim();
        let b = emb.dim(0);
        let emb = emb.reshape(&[b, 1, 2 * dim]);
        let sdt = self.stream_dtype();
        let shift = emb.narrow(2, 0, dim).contiguous().to_dtype(sdt);
        let scale = ops::add_scalar(&emb.narrow(2, dim, dim).contiguous().to_dtype(sdt), 1.0);
        let x = self.modulate(&x, &shift, &scale);
        let out = self.lin(&x, "final_layer.linear.weight");
        if stats {
            crate::debug::print_stats("final", &out);
        }
        out
    }
}

static FORWARD_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// The tensors of a Draw Things FLUX.2 checkpoint (names and splits as in Draw Things' own
/// files, f32), from any source; the inverse of [`dt_mapping`]. Used to build test models.
pub fn dt_tensors(src: &Flux2Source, config: &Flux2Config) -> Result<Vec<(String, HostTensor)>> {
    let (d, h, nd) = (config.inner_dim(), config.mlp_hidden(), config.num_layers);
    let mut out = vec![];
    let rd = |n: &str| src.read(n, DataType::F32);
    let rows = |t: &HostTensor, a: usize, n: usize| {
        let c = t.shape()[1];
        HostTensor::from_f32(&t.to_f32_vec()[a * c..(a + n) * c], &[n, c])
    };
    out.push((dt_name("x_embedder", 0), rd("img_in.weight")?));
    out.push((dt_name("context_embedder", 0), rd("txt_in.weight")?));
    out.push((dt_name("t_embedder_0", 0), rd("time_in.in_layer.weight")?));
    out.push((dt_name("t_embedder_1", 0), rd("time_in.out_layer.weight")?));
    out.push((dt_name("linear", 0), rd("final_layer.linear.weight")?));
    let ada = rd("final_layer.adaLN_modulation.1.weight")?;
    out.push((dt_name("ada_ln_0", 0), rows(&ada, d, d)));
    out.push((dt_name("ada_ln_1", 0), rows(&ada, 0, d)));
    for (canonical, p, n) in [
        ("double_stream_modulation_img.lin.weight", "x", 6),
        ("double_stream_modulation_txt.lin.weight", "context", 6),
        ("single_stream_modulation.lin.weight", "single", 3),
    ] {
        let t = rd(canonical)?;
        for k in 0..n {
            out.push((dt_name(&format!("{p}_ada_ln_{k}"), 0), rows(&t, k * d, d)));
        }
    }
    for i in 0..nd {
        for (s, x) in [("img", "x"), ("txt", "c")] {
            let b = format!("double_blocks.{i}.{s}");
            let qkv = rd(&format!("{b}_attn.qkv.weight"))?;
            for (k, q) in ["q", "k", "v"].iter().enumerate() {
                out.push((dt_name(&format!("{x}_{q}"), i), rows(&qkv, k * d, d)));
            }
            out.push((dt_name(&format!("{x}_o"), i), rd(&format!("{b}_attn.proj.weight"))?));
            out.push((dt_name(&format!("{x}_norm_q"), i), rd(&format!("{b}_attn.norm.query_norm.scale"))?));
            out.push((dt_name(&format!("{x}_norm_k"), i), rd(&format!("{b}_attn.norm.key_norm.scale"))?));
            let mlp = rd(&format!("{b}_mlp.0.weight"))?;
            out.push((dt_name(&format!("{x}_gate_proj"), i), rows(&mlp, 0, h)));
            out.push((dt_name(&format!("{x}_up_proj"), i), rows(&mlp, h, h)));
            out.push((dt_name(&format!("{x}_down_proj"), i), rd(&format!("{b}_mlp.2.weight"))?));
        }
    }
    for j in 0..config.num_single_layers {
        let i = nd + j;
        let b = format!("single_blocks.{j}");
        let l1 = rd(&format!("{b}.linear1.weight"))?;
        for (k, q) in ["q", "k", "v"].iter().enumerate() {
            out.push((dt_name(&format!("x_{q}"), i), rows(&l1, k * d, d)));
        }
        out.push((dt_name("x_w1", j), rows(&l1, 3 * d, h)));
        out.push((dt_name("x_w3", j), rows(&l1, 3 * d + h, h)));
        out.push((dt_name("x_o", i), rd(&format!("{b}.linear2_attn.weight"))?));
        out.push((dt_name("x_w2", j), rd(&format!("{b}.linear2_mlp.weight"))?));
        out.push((dt_name("x_norm_q", i), rd(&format!("{b}.norm.query_norm.scale"))?));
        out.push((dt_name("x_norm_k", i), rd(&format!("{b}.norm.key_norm.scale"))?));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ckpt::tests::{tmp, Writer};

    fn fixture(p: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(p)
    }

    /// Write the tiny BFL fixture as a Draw Things checkpoint in f32; `q8p` palettizes the
    /// large matrices in f16 like Draw Things, `expand` stores those q8p values expanded to f32.
    fn write_dt(path: &Path, q8p: bool, expand: bool) -> Flux2Config {
        let bfl = SafeTensorsFiles::open(&[fixture("flux2/flux2_bfl.safetensors")]).unwrap();
        let src = Flux2Source::new(bfl).unwrap();
        let config = Flux2Config::infer(&|n| src.shape(n)).unwrap();
        let mut w = Writer::new(path, false);
        for (name, t) in dt_tensors(&src, &config).unwrap() {
            if q8p && t.shape().len() == 2 && t.numel() >= 1024 {
                let h = t.to_dtype(DataType::F16);
                if expand {
                    let data = nnc::quant::palettize8_fast(&h, 1024);
                    w.dense(&name, &crate::ckpt::depalettize8(&data, DataType::F16, h.shape(), 1024).to_dtype(DataType::F32), crate::ckpt::CODEC_RAW);
                } else {
                    w.q8p(&name, &h, 1024);
                }
            } else {
                w.dense(&name, &t, crate::ckpt::CODEC_RAW);
            }
        }
        w.finish();
        config
    }

    fn run(path: &Path, tol: f32) -> Vec<f32> {
        let g = Graph::new();
        let (source, config) = open_source(path).unwrap();
        assert!(source.is_draw_things());
        let model = Flux2::load(&g, &source, config.clone(), Device::Cpu, DataType::F32, false).unwrap();
        let io = SafeTensorsFiles::open(&[fixture("flux2/flux2_io.safetensors")]).unwrap();
        let ld = |n: &str| upload(&g, &io.read(n, DataType::F32).unwrap(), Device::Cpu, DataType::F32);
        let mut ids = text_ids(5);
        ids.extend(image_ids(4, 4, 0.0));
        ids.extend(image_ids(2, 2, 10.0));
        let (rope, shape) = rope_table(&ids, &config.axes_dims_rope, config.rope_theta);
        let rope = g.from_f32(&rope, &shape, Device::Cpu, DataType::F32);
        let y = model.forward(&ld("x"), &ld("ctx"), &model.prepare_step(&[700.0], None), &rope).to_f32_vec();
        let want = io.read_f32("out").unwrap();
        let d: f32 = y.iter().zip(&want).map(|(a, b)| (a - b) * (a - b)).sum();
        let n: f32 = want.iter().map(|v| v * v).sum();
        let rel = (d / n).sqrt();
        eprintln!("{}: relative error {rel}", path.display());
        assert!(rel <= tol, "{}: relative error {rel}", path.display());
        y
    }

    #[test]
    fn draw_things_checkpoint_matches_diffusers() {
        let raw = tmp("dt_flux2_f32.ckpt");
        let config = write_dt(&raw, false, false);
        let (source, c2) = open_source(&raw).unwrap();
        assert_eq!(c2, config);
        assert!(!source.has_palettized());
        // Exact mapping (f32 storage).
        run(&raw, 1e-5);
        let q = tmp("dt_flux2_q8p.ckpt");
        write_dt(&q, true, false);
        let (source, _) = open_source(&q).unwrap();
        assert!(source.has_palettized());

        // Concatenated q8p parts stay palettized.
        assert!(matches!(source.read_raw("double_blocks.0.img_attn.qkv.weight").unwrap(), CkptTensor::Palettized { .. }));
        // 8-bit palettes on random weights (the real 9B checkpoint agrees with BFL's weights
        // to 0.1%, docs/PROVENANCE.md); the palettized path must equal the same values loaded dense.
        let yq = run(&q, 0.1);
        let e = tmp("dt_flux2_q8p_expanded.ckpt");
        write_dt(&e, true, true);
        let ye = run(&e, 0.1);
        let m = yq.iter().zip(&ye).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(m < 1e-4, "palettized vs expanded: {m}");
    }
}
