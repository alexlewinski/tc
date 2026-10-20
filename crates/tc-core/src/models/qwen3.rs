//! Qwen3 decoder used as a text encoder (hidden states from intermediate layers).
//!
//! Implemented from the public Qwen3 architecture (Apache-2.0 reference: Hugging Face
//! `transformers`). Only the forward pass needed to extract hidden states is implemented.

use std::path::Path;

use anyhow::{Context, Result};
use nnc::{ops, DataType, Device, Graph, HostTensor, Var};
use serde::Deserialize;

use crate::weights::{upload, Params, SafeTensorsFiles};

#[derive(Clone, Debug, Deserialize)]
pub struct Qwen3Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default = "d_eps")]
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default)]
    pub rope_parameters: Option<serde_json::Value>,
    pub vocab_size: usize,
    #[serde(default)]
    pub attention_bias: bool,
}

fn d_eps() -> f32 {
    1e-6
}

impl Qwen3Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim.unwrap_or(self.hidden_size / self.num_attention_heads)
    }
    pub fn theta(&self) -> f64 {
        self.rope_theta
            .or_else(|| self.rope_parameters.as_ref().and_then(|p| p.get("rope_theta")).and_then(|v| v.as_f64()))
            .unwrap_or(1_000_000.0)
    }
}

pub struct Qwen3 {
    pub config: Qwen3Config,
    params: Params,
    device: Device,
    dtype: DataType,
    layers_loaded: usize,
}

/// Reorder the output rows of a per-head projection from "rotate half" layout to interleaved
/// pairs: row `2i` <- `i`, row `2i+1` <- `i + D/2` within each head.
fn interleave_rows(t: &HostTensor, head_dim: usize) -> HostTensor {
    let shape = t.shape().to_vec();
    let row_len: usize = shape[1..].iter().product::<usize>() * t.dtype().size();
    let rows = shape[0];
    let bytes = t.bytes();
    let mut out = vec![0u8; bytes.len()];
    let half = head_dim / 2;
    for r in 0..rows {
        let head = r / head_dim;
        let j = r % head_dim;
        let src = if j % 2 == 0 { head * head_dim + j / 2 } else { head * head_dim + half + j / 2 };
        out[r * row_len..(r + 1) * row_len].copy_from_slice(&bytes[src * row_len..(src + 1) * row_len]);
    }
    HostTensor::from_bytes(out, t.dtype(), &shape)
}

impl Qwen3 {
    /// Load from a `text_encoder` directory (config.json + safetensors). Only the first
    /// `num_layers` decoder layers are loaded.
    pub fn load(graph: &Graph, dir: &Path, num_layers: usize, device: Device, dtype: DataType) -> Result<Qwen3> {
        let config: Qwen3Config = serde_json::from_slice(&std::fs::read(dir.join("config.json")).context("text encoder config.json")?)?;
        let files = SafeTensorsFiles::open_dir(dir)?;
        Qwen3::load_from(graph, &files, config, num_layers, device, dtype)
    }

    pub fn load_from(
        graph: &Graph,
        files: &SafeTensorsFiles,
        config: Qwen3Config,
        num_layers: usize,
        device: Device,
        dtype: DataType,
    ) -> Result<Qwen3> {
        let prefix = if files.contains("model.embed_tokens.weight") { "model." } else { "" };
        let hd = config.head_dim();
        let mut params = Params::new();
        let mut load = |name: &str, f: &dyn Fn(HostTensor) -> HostTensor| -> Result<()> {
            let host = files.read(&format!("{prefix}{name}"), dtype).with_context(|| format!("loading {name}"))?;
            params.insert(name.to_string(), upload(graph, &f(host), device, dtype));
            Ok(())
        };
        let id = |t: HostTensor| t;
        let il = |t: HostTensor| interleave_rows(&t, hd);
        let il_norm = |t: HostTensor| {
            let n = t.numel();
            interleave_rows(&t.reshape(&[n, 1]), hd).reshape(&[n])
        };
        load("embed_tokens.weight", &id)?;
        let num_layers = num_layers.min(config.num_hidden_layers);
        for i in 0..num_layers {
            let p = format!("layers.{i}");
            load(&format!("{p}.input_layernorm.weight"), &id)?;
            load(&format!("{p}.post_attention_layernorm.weight"), &id)?;
            load(&format!("{p}.self_attn.q_proj.weight"), &il)?;
            load(&format!("{p}.self_attn.k_proj.weight"), &il)?;
            load(&format!("{p}.self_attn.v_proj.weight"), &id)?;
            load(&format!("{p}.self_attn.o_proj.weight"), &id)?;
            load(&format!("{p}.self_attn.q_norm.weight"), &il_norm)?;
            load(&format!("{p}.self_attn.k_norm.weight"), &il_norm)?;
            if config.attention_bias {
                load(&format!("{p}.self_attn.q_proj.bias"), &il_norm)?;
                load(&format!("{p}.self_attn.k_proj.bias"), &il_norm)?;
                load(&format!("{p}.self_attn.v_proj.bias"), &id)?;
            }
            load(&format!("{p}.mlp.gate_proj.weight"), &id)?;
            load(&format!("{p}.mlp.up_proj.weight"), &id)?;
            load(&format!("{p}.mlp.down_proj.weight"), &id)?;
        }
        if num_layers == config.num_hidden_layers {
            // The last hidden state is reported after the final norm.
            load("norm.weight", &id)?;
        }
        Ok(Qwen3 { config, params, device, dtype, layers_loaded: num_layers })
    }

    fn rope(&self, graph: &Graph, len: usize) -> Var {
        let hd = self.config.head_dim();
        let theta = self.config.theta();
        let mut table = vec![0f32; len * hd];
        for t in 0..len {
            for i in 0..hd / 2 {
                // Matches the reference: inverse frequencies and angles in float32.
                let inv = 1.0f32 / (theta as f32).powf((2 * i) as f32 / hd as f32);
                let a = t as f32 * inv;
                table[t * hd + 2 * i] = a.cos();
                table[t * hd + 2 * i + 1] = a.sin();
            }
        }
        graph.from_f32(&table, &[1, len, 1, hd], self.device, self.dtype)
    }

    /// Causal mask that also hides padding keys (positions `>= valid`).
    fn mask(&self, graph: &Graph, len: usize, valid: usize) -> Var {
        let neg = if self.dtype == DataType::F32 { -1e9 } else { -65504.0 };
        let mut m = vec![0f32; len * len];
        for i in 0..len {
            for j in 0..len {
                if j > i || j >= valid {
                    m[i * len + j] = neg;
                }
            }
        }
        graph.from_f32(&m, &[1, 1, len, len], self.device, self.dtype)
    }

    /// Encode `ids` (length `T`, the first `valid` are real tokens, the rest padding) and return
    /// the hidden states after each layer in `layers` (1-based like `output_hidden_states`, so
    /// `0` is the embedding output) concatenated per token: `[1, T, layers.len() * hidden]`.
    pub fn hidden_states(&self, ids: &[i32], valid: usize, layers: &[usize]) -> Result<Var> {
        let max_layer = *layers.iter().max().unwrap_or(&0);
        anyhow::ensure!(max_layer <= self.layers_loaded, "layer {max_layer} not loaded");
        let graph = self.params.get("embed_tokens.weight").graph().clone();
        let len = ids.len();
        let idx = graph.from_host(&HostTensor::from_i32(ids, &[len]), self.device);
        let mut h = ops::index_select(self.params.get("embed_tokens.weight"), &idx).reshape(&[1, len, self.config.hidden_size]);
        let rope = self.rope(&graph, len);
        let mask = self.mask(&graph, len, valid.max(1));
        let mut outs: Vec<(usize, Var)> = vec![];
        if layers.contains(&0) {
            outs.push((0, h.clone()));
        }
        for i in 0..max_layer {
            h = self.layer(i, &h, &rope, &mask);
            if layers.contains(&(i + 1)) {
                if i + 1 == self.config.num_hidden_layers {
                    let n = ops::rms_norm(&h, &[2], self.config.rms_norm_eps, Some(self.params.get("norm.weight")));
                    outs.push((i + 1, n));
                } else {
                    outs.push((i + 1, h.clone()));
                }
            }
        }
        let ordered: Vec<Var> = layers.iter().map(|l| outs.iter().find(|(k, _)| k == l).unwrap().1.clone()).collect();
        let refs: Vec<&Var> = ordered.iter().collect();
        Ok(ops::concat(&refs, 2))
    }

    fn layer(&self, i: usize, x: &Var, rope: &Var, mask: &Var) -> Var {
        let c = &self.config;
        let p = format!("layers.{i}");
        let eps = c.rms_norm_eps;
        let len = x.dim(1);
        let hd = c.head_dim();
        let get = |n: &str| self.params.get(&format!("{p}.{n}"));
        let bias = |n: &str| self.params.try_get(&format!("{p}.{n}"));
        let h = ops::rms_norm(x, &[2], eps, Some(get("input_layernorm.weight")));
        let q = ops::linear(&h, get("self_attn.q_proj.weight"), bias("self_attn.q_proj.bias")).reshape(&[1, len, c.num_attention_heads, hd]);
        let k = ops::linear(&h, get("self_attn.k_proj.weight"), bias("self_attn.k_proj.bias")).reshape(&[1, len, c.num_key_value_heads, hd]);
        let v = ops::linear(&h, get("self_attn.v_proj.weight"), bias("self_attn.v_proj.bias")).reshape(&[1, len, c.num_key_value_heads, hd]);
        let q = ops::cmul(&ops::rms_norm(&q, &[3], eps, Some(get("self_attn.q_norm.weight"))), rope);
        let k = ops::cmul(&ops::rms_norm(&k, &[3], eps, Some(get("self_attn.k_norm.weight"))), rope);
        let a = ops::scaled_dot_product_attention(&q, &k, &v, 1.0 / (hd as f32).sqrt(), false, Some(mask));
        let a = a.reshape(&[1, len, c.num_attention_heads * hd]);
        let x = ops::add(x, &ops::linear(&a, get("self_attn.o_proj.weight"), None));
        let h = ops::rms_norm(&x, &[2], eps, Some(get("post_attention_layernorm.weight")));
        let gate = ops::linear(&h, get("mlp.gate_proj.weight"), None);
        let up = ops::linear(&h, get("mlp.up_proj.weight"), None);
        let m = ops::linear(&ops::mul(&ops::silu(&gate), &up), get("mlp.down_proj.weight"), None);
        ops::add(&x, &m)
    }
}
