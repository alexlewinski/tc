//! Qwen3 decoder used as a text encoder (hidden states from intermediate layers).
//!
//! Implemented from the public Qwen3 architecture (Apache-2.0 reference: Hugging Face
//! `transformers`). Only the forward pass needed to extract hidden states is implemented.

use std::path::Path;

use anyhow::{Context, Result};
use nnc::{ops, DataType, Device, Graph, HostTensor, Var};
use serde::Deserialize;

use crate::ckpt::{CkptFile, MappedCkpt, Recipe};
use crate::weights::{Params, SafeTensorsFiles, TensorSource};

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
pub(crate) fn interleave_rows(t: &HostTensor, head_dim: usize) -> HostTensor {
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

/// Name under which [`dt_source`] marks a Draw Things checkpoint.
const DT_MARKER: &str = "__draw_things__";

fn dt_name(kind: &str, i: usize) -> String {
    format!("__text_model__[t-{kind}-{i}-0]")
}

/// Map a Draw Things Qwen3 checkpoint (`__text_model__[t-...]`) onto Hugging Face names and
/// infer its configuration. Names were read from the checkpoint index and each tensor checked
/// against the Qwen/Qwen3-8B release (docs/PROVENANCE.md).
pub fn dt_source(file: CkptFile) -> Result<(MappedCkpt, Qwen3Config)> {
    let shape = |n: &str| file.info(n).map(|i| i.shape.clone()).with_context(|| format!("{}: missing {n}; not a Qwen3 checkpoint", file.path().display()));
    let emb = shape(&dt_name("tok_embeddings", 0))?;
    let (vocab, hidden) = (emb[0], emb[1]);
    let layers = (0..).take_while(|i| file.contains(&dt_name("q_proj", *i))).count();
    let head_dim: usize = shape(&dt_name("norm_q", 0))?.iter().product();
    let q_rows = shape(&dt_name("q_proj", 0))?[0];
    let kv_rows = shape(&dt_name("k_proj", 0))?[0];
    let inter = shape("__text_model__[t-mlp-0-mlp_gate_proj-0-0]")?[0];
    let mut r = std::collections::BTreeMap::new();
    let mut add = |canonical: String, part: String, sh: Vec<usize>| {
        r.insert(canonical, Recipe { parts: vec![part], shape: sh });
    };
    add("embed_tokens.weight".into(), dt_name("tok_embeddings", 0), vec![vocab, hidden]);
    add("norm.weight".into(), dt_name("norm", 0), vec![hidden]);
    // Present in every Draw Things source so the loader can tell.
    add(DT_MARKER.into(), dt_name("norm", 0), vec![hidden]);
    for i in 0..layers {
        let l = format!("layers.{i}");
        add(format!("{l}.input_layernorm.weight"), dt_name("input_layernorm", i), vec![hidden]);
        add(format!("{l}.post_attention_layernorm.weight"), dt_name("post_attention_layernorm", i), vec![hidden]);
        add(format!("{l}.self_attn.q_proj.weight"), dt_name("q_proj", i), vec![q_rows, hidden]);
        add(format!("{l}.self_attn.k_proj.weight"), dt_name("k_proj", i), vec![kv_rows, hidden]);
        add(format!("{l}.self_attn.v_proj.weight"), dt_name("v_proj", i), vec![kv_rows, hidden]);
        add(format!("{l}.self_attn.o_proj.weight"), dt_name("out_proj", i), vec![hidden, q_rows]);
        add(format!("{l}.self_attn.q_norm.weight"), dt_name("norm_q", i), vec![head_dim]);
        add(format!("{l}.self_attn.k_norm.weight"), dt_name("norm_k", i), vec![head_dim]);
        for p in ["gate", "up", "down"] {
            let sh = if p == "down" { vec![hidden, inter] } else { vec![inter, hidden] };
            add(format!("{l}.mlp.{p}_proj.weight"), format!("__text_model__[t-mlp-{i}-mlp_{p}_proj-0-0]"), sh);
        }
    }
    let config = Qwen3Config {
        hidden_size: hidden,
        intermediate_size: inter,
        num_hidden_layers: layers,
        num_attention_heads: q_rows / head_dim,
        num_key_value_heads: kv_rows / head_dim,
        head_dim: Some(head_dim),
        rms_norm_eps: 1e-6,
        rope_theta: Some(1_000_000.0),
        rope_parameters: None,
        vocab_size: vocab,
        attention_bias: false,
    };
    let m = MappedCkpt::new(file, r)?;
    let unused = m.unused();
    if !unused.is_empty() {
        log::warn!("{}: {} tensors not used (e.g. {:?})", m.file.path().display(), unused.len(), &unused[..unused.len().min(3)]);
    }
    Ok((m, config))
}

impl Qwen3 {
    /// Load from a `text_encoder` directory (config.json + safetensors). Only the first
    /// `num_layers` decoder layers are loaded.
    pub fn load(graph: &Graph, dir: &Path, num_layers: usize, device: Device, dtype: DataType, quantize: bool) -> Result<Qwen3> {
        if dir.is_file() && crate::ckpt::is_ckpt(dir) {
            let (files, config) = dt_source(CkptFile::open(dir)?)?;
            return Qwen3::load_from(graph, &files, config, num_layers, device, dtype, quantize);
        }
        let config: Qwen3Config = serde_json::from_slice(&std::fs::read(dir.join("config.json")).context("text encoder config.json")?)?;
        let files = SafeTensorsFiles::open_dir(dir)?;
        Qwen3::load_from(graph, &files, config, num_layers, device, dtype, quantize)
    }

    /// Load from Hugging Face names (`[model.]layers.N...`). A Draw Things source
    /// ([`dt_source`]) already stores q/k projections in the interleaved rotary layout used here,
    /// so they are not permuted again, and its q8p weights are used as stored.
    pub fn load_from(
        graph: &Graph,
        files: &dyn TensorSource,
        config: Qwen3Config,
        num_layers: usize,
        device: Device,
        dtype: DataType,
        quantize: bool,
    ) -> Result<Qwen3> {
        let prefix = if files.contains("model.embed_tokens.weight") { "model." } else { "" };
        let draw_things = files.contains(DT_MARKER);
        let hd = config.head_dim();
        let mut params = Params::new();
        let mut load = |name: &str, f: &dyn Fn(HostTensor) -> HostTensor| -> Result<()> {
            let q = quantize && name.starts_with("layers.");
            let v = if draw_things {
                let raw = files.read_raw(name).with_context(|| format!("loading {name}"))?;
                crate::models::flux2::upload_param(graph, raw, device, dtype, q, true)
            } else {
                let host = files.read(&format!("{prefix}{name}"), dtype).with_context(|| format!("loading {name}"))?;
                crate::models::flux2::upload_weight(graph, &f(host), device, dtype, q)
            };
            params.insert(name.to_string(), v);
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

/// The tensors of a Draw Things Qwen3 checkpoint (names, shapes and interleaved q/k layout as in
/// Draw Things' own files, f32) from Hugging Face weights; the inverse of [`dt_source`].
pub fn dt_tensors(files: &dyn TensorSource, config: &Qwen3Config) -> Result<Vec<(String, HostTensor)>> {
    let prefix = if files.contains("model.embed_tokens.weight") { "model." } else { "" };
    let rd = |n: &str| files.read(&format!("{prefix}{n}"), DataType::F32);
    let (h, hd) = (config.hidden_size, config.head_dim());
    let mut out = vec![];
    out.push((dt_name("tok_embeddings", 0), rd("embed_tokens.weight")?));
    out.push((dt_name("norm", 0), rd("norm.weight")?.reshape(&[1, h])));
    for i in 0..config.num_hidden_layers {
        let l = format!("layers.{i}");
        out.push((dt_name("input_layernorm", i), rd(&format!("{l}.input_layernorm.weight"))?.reshape(&[1, h])));
        out.push((dt_name("post_attention_layernorm", i), rd(&format!("{l}.post_attention_layernorm.weight"))?.reshape(&[1, h])));
        for (dt, hf) in [("q_proj", "q_proj"), ("k_proj", "k_proj")] {
            out.push((dt_name(dt, i), interleave_rows(&rd(&format!("{l}.self_attn.{hf}.weight"))?, hd)));
        }
        for (dt, hf) in [("norm_q", "q_norm"), ("norm_k", "k_norm")] {
            let t = rd(&format!("{l}.self_attn.{hf}.weight"))?;
            out.push((dt_name(dt, i), interleave_rows(&t.reshape(&[hd, 1]), hd).reshape(&[1, 1, 1, hd])));
        }
        out.push((dt_name("v_proj", i), rd(&format!("{l}.self_attn.v_proj.weight"))?));
        out.push((dt_name("out_proj", i), rd(&format!("{l}.self_attn.o_proj.weight"))?));
        for p in ["gate", "up", "down"] {
            out.push((format!("__text_model__[t-mlp-{i}-mlp_{p}_proj-0-0]"), rd(&format!("{l}.mlp.{p}_proj.weight"))?));
        }
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

    /// The tiny fixture written as a Draw Things Qwen3 checkpoint: q/k projections and norms in
    /// the interleaved rotary layout, norms shaped `[1, n]` / `[1, 1, 1, head_dim]`, f32 storage.
    #[test]
    fn draw_things_checkpoint_matches_transformers() {
        let dir = fixture("qwen3/text_encoder");
        let config: Qwen3Config = serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
        let st = SafeTensorsFiles::open_dir(&dir).unwrap();
        let path = tmp("dt_qwen3.ckpt");
        let mut w = Writer::new(&path, false);
        for (name, t) in dt_tensors(&st, &config).unwrap() {
            w.dense(&name, &t, crate::ckpt::CODEC_RAW);
        }
        w.finish();
        let g = Graph::new();
        let a = Qwen3::load(&g, &path, 4, Device::Cpu, DataType::F32, false).unwrap();
        let b = Qwen3::load(&g, &dir, 4, Device::Cpu, DataType::F32, false).unwrap();
        assert_eq!(a.config.num_attention_heads, config.num_attention_heads);
        assert_eq!(a.config.num_key_value_heads, config.num_key_value_heads);
        let ids: Vec<i32> = (0..12).map(|i| i * 7 % 100).collect();
        let x = a.hidden_states(&ids, 9, &[1, 2, 4]).unwrap().to_f32_vec();
        let y = b.hidden_states(&ids, 9, &[1, 2, 4]).unwrap().to_f32_vec();
        let m = x.iter().zip(&y).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
        assert!(m < 1e-4, "max difference {m}");
    }
}
