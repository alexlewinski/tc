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
//! * `single_blocks.N.{linear1,linear2}`, `.norm.{query,key}_norm.scale`
//! * `final_layer.adaLN_modulation.1` (shift, scale), `final_layer.linear`

use std::path::Path;

use anyhow::{bail, Context, Result};
use nnc::{ops, DataType, Device, Graph, HostTensor, Var};
use serde::Deserialize;

use std::collections::HashMap;

use crate::lora::{linear_with_adapters, Adapter, LoraTarget, TargetResolver};
use crate::weights::{upload, Params, SafeTensorsFiles};

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
            for n in ["linear1.weight", "linear2.weight", "norm.query_norm.scale", "norm.key_norm.scale"] {
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
}

/// Where FLUX.2 weights come from and in which naming scheme.
pub struct Flux2Source {
    files: SafeTensorsFiles,
    diffusers: bool,
    prefix: String,
}

impl Flux2Source {
    pub fn new(files: SafeTensorsFiles) -> Result<Flux2Source> {
        for prefix in ["", "model.diffusion_model.", "diffusion_model.", "transformer."] {
            if files.contains(&format!("{prefix}img_in.weight")) {
                return Ok(Flux2Source { files, diffusers: false, prefix: prefix.into() });
            }
            if files.contains(&format!("{prefix}x_embedder.weight")) {
                return Ok(Flux2Source { files, diffusers: true, prefix: prefix.into() });
            }
        }
        bail!("not a FLUX.2 transformer checkpoint (no img_in / x_embedder weight)")
    }

    fn recipe(&self, canonical: &str) -> Recipe {
        let p = &self.prefix;
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
                    "linear2" => "attn.to_out",
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
            Recipe::Direct(k) | Recipe::SwapHalves(k) => self.files.info(&k).map(|i| i.shape.clone()),
            Recipe::ConcatRows(keys) => {
                let mut rows = 0;
                let mut rest = None;
                for k in &keys {
                    let s = self.files.info(k)?.shape.clone();
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
        }
    }

    pub fn is_diffusers(&self) -> bool {
        self.diffusers
    }
}

/// Load the transformer weights from a diffusers directory (with `config.json`), a
/// directory of safetensors shards, or a single safetensors file.
pub fn open_source(path: &Path) -> Result<(Flux2Source, Flux2Config)> {
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

impl TargetResolver for Flux2LoraResolver {
    fn resolve(&self, module: &str) -> Vec<LoraTarget> {
        let inner = self.config.inner_dim();
        let parts: Vec<&str> = module.split('.').collect();
        let whole = |c: String| vec![LoraTarget { param: format!("{c}.weight"), row_offset: 0, rows: None, swap_halves: false }];
        let slice = |c: String, i: usize| {
            vec![LoraTarget { param: format!("{c}.weight"), row_offset: i * inner, rows: Some(inner), swap_halves: false }]
        };
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
            ["norm_out", "linear"] => vec![LoraTarget {
                param: "final_layer.adaLN_modulation.1.weight".into(),
                row_offset: 0,
                rows: None,
                swap_halves: true,
            }],
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
                "attn.to_out" => whole(format!("single_blocks.{n}.linear2")),
                _ => vec![],
            },
            _ => vec![],
        }
    }

    fn canonical_modules(&self) -> Vec<String> {
        self.config
            .parameter_names()
            .into_iter()
            .filter_map(|n| n.strip_suffix(".weight").map(|s| s.to_string()))
            .collect()
    }
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
        let src_dtype = if dtype == DataType::F32 { DataType::F16 } else { dtype };
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

impl Flux2 {
    /// Upload all weights from `source`.
    pub fn load(graph: &Graph, source: &Flux2Source, config: Flux2Config, device: Device, dtype: DataType, quantize: bool) -> Result<Flux2> {
        let mut params = Params::new();
        for name in config.parameter_names() {
            let host = source.read(&name, if dtype == DataType::F32 { DataType::F32 } else { dtype }).with_context(|| format!("loading {name}"))?;
            params.insert(name, upload_weight(graph, &host, device, dtype, quantize));
        }
        Ok(Flux2 { config, params, device, dtype, quantized: quantize, adapters: HashMap::new() })
    }

    /// Re-read one parameter from the source (used to undo LoRA merges).
    pub fn reload_param(&mut self, graph: &Graph, source: &Flux2Source, name: &str) -> Result<()> {
        let host = source.read(name, self.dtype)?;
        self.params.insert(name.to_string(), upload_weight(graph, &host, self.device, self.dtype, self.quantized));
        Ok(())
    }

    fn lin(&self, x: &Var, name: &str) -> Var {
        linear_with_adapters(x, self.params.get(name), self.adapters.get(name))
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
                let c = m.narrow(2, i * dim, dim).contiguous();
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
        let n = ops::layer_norm(x, &[2], self.config.eps, None);
        ops::add(&ops::mul(&n, scale1p), shift)
    }

    fn clamp_f16(&self, x: Var) -> Var {
        if self.dtype == DataType::F16 {
            ops::clamp(&x, Some(-65504.0), Some(65504.0))
        } else {
            x
        }
    }

    /// Split fused `[B, T, 3*inner]` into normalized q, k and v, shaped `[B, T, H, D]`.
    fn qkv(&self, qkv: &Var, q_norm: &str, k_norm: &str) -> (Var, Var, Var) {
        let (b, t) = (qkv.dim(0), qkv.dim(1));
        let h = self.config.num_attention_heads;
        let d = self.config.attention_head_dim;
        let inner = h * d;
        // nnc CPU kernels handle at most 4-D tensors, so split along the flat channel axis.
        let take = |i: usize| qkv.narrow(2, i * inner, inner).contiguous().reshape(&[b, t, h, d]);
        let q = ops::rms_norm(&take(0), &[3], 1e-6, Some(self.params.get(q_norm)));
        let k = ops::rms_norm(&take(1), &[3], 1e-6, Some(self.params.get(k_norm)));
        (q, k, take(2))
    }

    fn attention(&self, q: &Var, k: &Var, v: &Var, rope: &Var) -> Var {
        let q = ops::cmul(q, rope);
        let k = ops::cmul(k, rope);
        let d = self.config.attention_head_dim;
        let o = ops::scaled_dot_product_attention(&q, &k, v, 1.0 / (d as f32).sqrt(), false, None);
        let (b, t) = (o.dim(0), o.dim(1));
        o.reshape(&[b, t, self.config.inner_dim()])
    }

    fn swiglu_mlp(&self, x: &Var, w_in: &str, w_out: &str) -> Var {
        let h = self.lin(x, w_in);
        let m = h.dim(2) / 2;
        let a = h.narrow(2, 0, m).contiguous();
        let gate = h.narrow(2, m, m).contiguous();
        let h = ops::mul(&ops::silu(&a), &gate);
        self.lin(&h, w_out)
    }

    fn double_block(&self, i: usize, img: Var, txt: Var, step: &Flux2Step, rope: &Var) -> (Var, Var) {
        let p = format!("double_blocks.{i}");
        let im = &step.img_mod;
        let tm = &step.txt_mod;
        let img_n = self.modulate(&img, &im[0], &im[1]);
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
        let attn = self.attention(&q, &k, &v, rope);
        let lt = txt.dim(1);
        let li = img.dim(1);
        let txt_attn = attn.narrow(1, 0, lt).contiguous();
        let img_attn = attn.narrow(1, lt, li).contiguous();
        let img_attn = self.lin(&img_attn, &format!("{p}.img_attn.proj.weight"));
        let txt_attn = self.lin(&txt_attn, &format!("{p}.txt_attn.proj.weight"));
        let img = ops::add(&img, &ops::mul(&img_attn, &im[2]));
        let img_n = self.modulate(&img, &im[3], &im[4]);
        let img_ff = self.swiglu_mlp(&img_n, &format!("{p}.img_mlp.0.weight"), &format!("{p}.img_mlp.2.weight"));
        let img = ops::add(&img, &ops::mul(&img_ff, &im[5]));
        let txt = ops::add(&txt, &ops::mul(&txt_attn, &tm[2]));
        let txt_n = self.modulate(&txt, &tm[3], &tm[4]);
        let txt_ff = self.swiglu_mlp(&txt_n, &format!("{p}.txt_mlp.0.weight"), &format!("{p}.txt_mlp.2.weight"));
        let txt = ops::add(&txt, &ops::mul(&txt_ff, &tm[5]));
        (self.clamp_f16(img), self.clamp_f16(txt))
    }

    fn single_block(&self, i: usize, x: Var, step: &Flux2Step, rope: &Var) -> Var {
        let p = format!("single_blocks.{i}");
        let sm = &step.single_mod;
        let xn = self.modulate(&x, &sm[0], &sm[1]);
        let h = self.lin(&xn, &format!("{p}.linear1.weight"));
        let inner = self.config.inner_dim();
        let mlp = self.config.mlp_hidden();
        let qkv = h.narrow(2, 0, 3 * inner).contiguous();
        let (q, k, v) = self.qkv(&qkv, &format!("{p}.norm.query_norm.scale"), &format!("{p}.norm.key_norm.scale"));
        let attn = self.attention(&q, &k, &v, rope);
        let a = h.narrow(2, 3 * inner, mlp).contiguous();
        let gate = h.narrow(2, 3 * inner + mlp, mlp).contiguous();
        let m = ops::mul(&ops::silu(&a), &gate);
        let cat = ops::concat(&[&attn, &m], 2);
        let out = self.lin(&cat, &format!("{p}.linear2.weight"));
        self.clamp_f16(ops::add(&x, &ops::mul(&out, &sm[2])))
    }

    /// Run the transformer.
    ///
    /// * `img`: packed latents `[B, N, in_channels]` (target tokens first, then reference tokens)
    /// * `txt`: text features `[B, L, joint_attention_dim]`
    /// * `rope`: table from [`rope_table`] for the concatenation `[txt ids, img ids]`
    ///
    /// Returns the prediction for all `N` image tokens: `[B, N, out_channels]`.
    pub fn forward(&self, img: &Var, txt: &Var, step: &Flux2Step, rope: &Var) -> Var {
        let mut img = self.lin(img, "img_in.weight");
        let mut txt = self.lin(txt, "txt_in.weight");
        for i in 0..self.config.num_layers {
            let (a, b) = self.double_block(i, img, txt, step, rope);
            img = a;
            txt = b;
        }
        let lt = txt.dim(1);
        let li = img.dim(1);
        let mut x = ops::concat(&[&txt, &img], 1);
        drop(txt);
        drop(img);
        for i in 0..self.config.num_single_layers {
            x = self.single_block(i, x, step, rope);
        }
        let x = x.narrow(1, lt, li).contiguous();
        let emb = self.lin(&ops::silu(&step.temb), "final_layer.adaLN_modulation.1.weight");
        let dim = self.config.inner_dim();
        let b = emb.dim(0);
        let emb = emb.reshape(&[b, 1, 2 * dim]);
        let shift = emb.narrow(2, 0, dim).contiguous();
        let scale = ops::add_scalar(&emb.narrow(2, dim, dim).contiguous(), 1.0);
        let x = self.modulate(&x, &shift, &scale);
        self.lin(&x, "final_layer.linear.weight")
    }
}
