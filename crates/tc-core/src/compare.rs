//! Tensor-by-tensor comparison of a Draw Things checkpoint with the official release of the same
//! model, through the same loaders the pipeline uses (parity checking of the `.ckpt` mapping).

use std::path::Path;

use anyhow::{bail, Context, Result};
use nnc::DataType;

use crate::ckpt::{is_ckpt, CkptFile};
use crate::models::{flux2, flux2_vae, qwen3};
use crate::weights::{SafeTensorsFiles, TensorSource};

/// Model component stored in a file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Component {
    Transformer,
    TextEncoder,
    Vae,
    Lora,
}

/// Which component a Draw Things checkpoint holds (from its tensor names).
pub fn ckpt_component(file: &CkptFile) -> Option<Component> {
    let mut keys = file.keys();
    let first = keys.next()?;
    if file.keys().any(|k| k.ends_with("__up__") || k.ends_with("__down__")) {
        return Some(Component::Lora);
    }
    if first.starts_with("__dit__") {
        Some(Component::Transformer)
    } else if first.starts_with("__text_model__") {
        Some(Component::TextEncoder)
    } else if first.starts_with("__encoder__") || first.starts_with("__decoder__") {
        Some(Component::Vae)
    } else {
        None
    }
}

/// One compared tensor.
#[derive(Clone, Debug)]
pub struct Row {
    pub name: String,
    pub shape: Vec<usize>,
    /// `||a - b|| / ||b||`.
    pub rel: f64,
    pub max_abs: f32,
}

fn row(name: &str, a: &[f32], b: &[f32], shape: Vec<usize>) -> Row {
    let (mut d2, mut b2, mut mx) = (0f64, 0f64, 0f32);
    for (x, y) in a.iter().zip(b) {
        let d = x - y;
        d2 += (d as f64) * (d as f64);
        b2 += (*y as f64) * (*y as f64);
        mx = mx.max(d.abs());
    }
    Row { name: name.to_string(), shape, rel: (d2 / b2.max(1e-30)).sqrt(), max_abs: mx }
}

/// Compare every canonical tensor of `ckpt` with `reference` (a diffusers component directory,
/// a safetensors file, or a snapshot directory with `transformer/`, `text_encoder/`, `vae/`).
/// `filter` keeps tensors whose canonical name contains it. `progress` sees each row.
pub fn compare(ckpt: &Path, reference: &Path, filter: Option<&str>, progress: &mut dyn FnMut(&Row)) -> Result<Vec<Row>> {
    if !is_ckpt(ckpt) {
        bail!("{} is not a Draw Things checkpoint", ckpt.display());
    }
    let file = CkptFile::open(ckpt)?;
    let component = ckpt_component(&file).context("unrecognized checkpoint (not a FLUX.2 transformer, Qwen3 or VAE)")?;
    let sub = |d: &str| {
        let p = reference.join(d);
        if p.is_dir() {
            p
        } else {
            reference.to_path_buf()
        }
    };
    let keep = |n: &str| filter.map(|f| n.contains(f)).unwrap_or(true);
    let mut rows = vec![];
    let mut push = |r: Row, rows: &mut Vec<Row>| {
        progress(&r);
        rows.push(r);
    };
    match component {
        Component::Transformer => {
            let (a, config) = flux2::Flux2Source::from_ckpt(file)?;
            let (b, config_b) = flux2::open_source(&sub("transformer"))?;
            if config.num_layers != config_b.num_layers || config.inner_dim() != config_b.inner_dim() {
                bail!("different architectures: {config:?} vs {config_b:?}");
            }
            for n in config.parameter_names().into_iter().filter(|n| keep(n)) {
                let x = a.read(&n, DataType::F32)?;
                let y = b.read(&n, DataType::F32)?;
                push(row(&n, &x.to_f32_vec(), &y.to_f32_vec(), x.shape().to_vec()), &mut rows);
            }
        }
        Component::TextEncoder => {
            let (a, config) = qwen3::dt_source(file)?;
            let dir = sub("text_encoder");
            let b = if dir.is_dir() { SafeTensorsFiles::open_dir(&dir)? } else { SafeTensorsFiles::open(&[&dir])? };
            let prefix = if b.contains("model.embed_tokens.weight") { "model." } else { "" };
            let hd = config.head_dim();
            for n in a.names().into_iter().filter(|n| !n.starts_with("__") && keep(n)) {
                let x = a.read(&n, DataType::F32)?;
                let mut y = TensorSource::read(&b, &format!("{prefix}{n}"), DataType::F32)?;
                // The checkpoint stores q/k in the interleaved rotary layout (see qwen3::dt_source).
                if n.ends_with("q_proj.weight") || n.ends_with("k_proj.weight") {
                    y = qwen3::interleave_rows(&y, hd);
                } else if n.ends_with("q_norm.weight") || n.ends_with("k_norm.weight") {
                    let k = y.numel();
                    y = qwen3::interleave_rows(&y.reshape(&[k, 1]), hd).reshape(&[k]);
                }
                push(row(&n, &x.to_f32_vec(), &y.to_f32_vec(), x.shape().to_vec()), &mut rows);
            }
        }
        Component::Vae => {
            let config = flux2_vae::dt_config(&file)?;
            let a = flux2_vae::dt_source(file, &config)?;
            let dir = sub("vae");
            let b = if dir.is_dir() { SafeTensorsFiles::open_dir(&dir)? } else { SafeTensorsFiles::open(&[&dir])? };
            for n in a.names().into_iter().filter(|n| keep(n)) {
                let x = a.read(&n, DataType::F32)?;
                let y = TensorSource::read(&b, &n, DataType::F32)?;
                push(row(&n, &x.to_f32_vec(), &y.to_f32_vec(), x.shape().to_vec()), &mut rows);
            }
        }
        Component::Lora => bail!("{} is a LoRA; use compare_lora", ckpt.display()),
    }
    Ok(rows)
}

/// Compare the weight updates of a Draw Things LoRA with another LoRA file (typically the
/// original `.safetensors` it was converted from), on the parameters of `transformer`.
/// Rows: `(parameter, relative difference, best-fit scale)`; NaN when only one file adapts it.
pub fn compare_lora(lora: &Path, reference: &Path, transformer: &Path) -> Result<Vec<(String, f64, f64)>> {
    let (source, config) = flux2::open_source(transformer)?;
    let resolver = flux2::Flux2LoraResolver { config };
    let a = crate::lora::LoraFile::open(lora)?;
    let b = crate::lora::LoraFile::open(reference)?;
    crate::lora::compare_deltas(&a, &b, &resolver, &|n| source.shape(n))
}
