//! Low-rank adapter (LoRA) loading and merging.
//!
//! Supported on-disk conventions (keys may carry prefixes such as `diffusion_model.`,
//! `transformer.`, `base_model.model.`):
//!
//! * PEFT / ai-toolkit: `<module>.lora_A.weight`, `<module>.lora_B.weight` (optional `<module>.alpha`)
//! * kohya: `lora_unet_<module_with_underscores>.lora_down.weight`, `.lora_up.weight`, `.alpha`
//! * full weight differences: `<module>.diff`
//!
//! Each model provides a [`TargetResolver`] that maps module names (in any naming scheme it
//! understands) to rows of its canonical parameters.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result};
use nnc::{ops, DataType, HostTensor, Var};

use crate::weights::{upload, Params, SafeTensorsFiles};

/// Where a LoRA delta lands in a canonical parameter.
#[derive(Clone, Debug, PartialEq)]
pub struct LoraTarget {
    /// Canonical parameter name (e.g. `double_blocks.0.img_attn.qkv.weight`).
    pub param: String,
    /// First output row affected.
    pub row_offset: usize,
    /// Number of output rows affected (`None` = the whole parameter).
    pub rows: Option<usize>,
    /// The delta's two output halves are stored swapped relative to the parameter.
    pub swap_halves: bool,
}

pub trait TargetResolver {
    /// Map a module name (without `.weight`) to targets. Empty if unknown.
    fn resolve(&self, module: &str) -> Vec<LoraTarget>;
    /// Canonical module names, used to decode kohya-style underscore names.
    fn canonical_modules(&self) -> Vec<String>;
}

/// One low-rank (or full) update parsed from a file.
pub struct LoraUpdate {
    pub module: String,
    /// `[rank, in]`, `None` for full diffs.
    pub down: Option<HostTensor>,
    /// `[out, rank]`, or the full `[out, in]` diff.
    pub up: HostTensor,
    pub alpha: Option<f32>,
}

/// A parsed LoRA file.
pub struct LoraFile {
    pub updates: Vec<LoraUpdate>,
    pub ignored_keys: Vec<String>,
}

const PREFIXES: &[&str] = &[
    "base_model.model.",
    "model.diffusion_model.",
    "diffusion_model.",
    "transformer.",
    "model.",
];

fn strip_prefixes(mut k: &str) -> &str {
    loop {
        let mut changed = false;
        for p in PREFIXES {
            if let Some(rest) = k.strip_prefix(p) {
                k = rest;
                changed = true;
            }
        }
        if !changed {
            return k;
        }
    }
}

enum Part {
    Down,
    Up,
    Alpha,
    Diff,
}

fn split_key(key: &str) -> Option<(String, Part)> {
    const SUFFIXES: &[(&str, u8)] = &[
        (".lora_A.default.weight", 0),
        (".lora_B.default.weight", 1),
        (".lora_A.weight", 0),
        (".lora_B.weight", 1),
        (".lora_down.weight", 0),
        (".lora_up.weight", 1),
        (".lora.down.weight", 0),
        (".lora.up.weight", 1),
        (".alpha", 2),
        (".diff", 3),
    ];
    for (s, kind) in SUFFIXES {
        if let Some(m) = key.strip_suffix(s) {
            let part = match kind {
                0 => Part::Down,
                1 => Part::Up,
                2 => Part::Alpha,
                _ => Part::Diff,
            };
            return Some((m.to_string(), part));
        }
    }
    None
}

impl LoraFile {
    pub fn open(path: &Path) -> Result<LoraFile> {
        let files = SafeTensorsFiles::open(&[path]).with_context(|| format!("opening LoRA {}", path.display()))?;
        LoraFile::from_files(&files)
    }

    pub fn from_files(files: &SafeTensorsFiles) -> Result<LoraFile> {
        let mut downs: BTreeMap<String, HostTensor> = BTreeMap::new();
        let mut ups: BTreeMap<String, HostTensor> = BTreeMap::new();
        let mut alphas: HashMap<String, f32> = HashMap::new();
        let mut diffs: BTreeMap<String, HostTensor> = BTreeMap::new();
        let mut ignored = vec![];
        for key in files.keys() {
            let Some((module, part)) = split_key(key) else {
                ignored.push(key.clone());
                continue;
            };
            let module = strip_prefixes(&module).to_string();
            match part {
                Part::Down => {
                    downs.insert(module, files.read(key, DataType::F32)?);
                }
                Part::Up => {
                    ups.insert(module, files.read(key, DataType::F32)?);
                }
                Part::Alpha => {
                    let v = files.read_f32(key)?;
                    if let Some(a) = v.first() {
                        alphas.insert(module, *a);
                    }
                }
                Part::Diff => {
                    diffs.insert(module, files.read(key, DataType::F32)?);
                }
            }
        }
        let mut updates = vec![];
        for (module, down) in downs {
            let Some(up) = ups.remove(&module) else {
                ignored.push(format!("{module} (no up/B matrix)"));
                continue;
            };
            // Conv-style LoRA weights [r, in, 1, 1] -> [r, in].
            let down = flatten2(down);
            let up = flatten2(up);
            let alpha = alphas.get(&module).copied();
            updates.push(LoraUpdate { module, down: Some(down), up, alpha });
        }
        for (module, _) in ups {
            ignored.push(format!("{module} (no down/A matrix)"));
        }
        for (module, diff) in diffs {
            updates.push(LoraUpdate { module, down: None, up: diff, alpha: None });
        }
        Ok(LoraFile { updates, ignored_keys: ignored })
    }
}

fn flatten2(t: HostTensor) -> HostTensor {
    let s = t.shape().to_vec();
    if s.len() <= 2 {
        return t;
    }
    let rest: usize = s[1..].iter().product();
    t.reshape(&[s[0], rest])
}

/// Decode kohya `lora_unet_a_b_0_c` names into canonical dotted module names.
fn kohya_map(resolver: &dyn TargetResolver) -> HashMap<String, String> {
    resolver
        .canonical_modules()
        .into_iter()
        .map(|m| (m.replace('.', "_"), m))
        .collect()
}

/// Result of merging adapters.
#[derive(Debug, Default)]
pub struct MergeReport {
    pub applied: usize,
    pub unmatched: Vec<String>,
    pub touched: HashSet<String>,
}

/// Merge `lora` (scaled by `weight`) into `params`.
pub fn merge(params: &mut Params, resolver: &dyn TargetResolver, lora: &LoraFile, weight: f32) -> Result<MergeReport> {
    let kohya = kohya_map(resolver);
    let mut report = MergeReport::default();
    for u in &lora.updates {
        let mut module = u.module.as_str();
        if let Some(rest) = module.strip_prefix("lora_unet_") {
            match kohya.get(rest) {
                Some(m) => module = m.as_str(),
                None => {
                    report.unmatched.push(u.module.clone());
                    continue;
                }
            }
        }
        if module.starts_with("lora_te") || module.starts_with("text_encoder") {
            report.unmatched.push(u.module.clone());
            continue;
        }
        let targets = resolver.resolve(module);
        if targets.is_empty() {
            report.unmatched.push(u.module.clone());
            continue;
        }
        for target in targets {
            let Some(w) = params.try_get(&target.param).cloned() else {
                report.unmatched.push(u.module.clone());
                continue;
            };
            apply_update(&w, u, &target, weight)
                .with_context(|| format!("applying LoRA {} to {}", u.module, target.param))?;
            report.touched.insert(target.param.clone());
            report.applied += 1;
        }
    }
    Ok(report)
}

/// A LoRA kept separate from the base weight (used when weights are quantized):
/// `y[.., row_offset..row_offset+rows] += up @ (down @ x)`, with the strength folded into `up`.
#[derive(Clone)]
pub struct Adapter {
    pub down: Var,
    pub up: Var,
    pub row_offset: usize,
    pub rows: usize,
}

/// Build runtime adapters for `lora` (scaled by `weight`). Returns `(param, adapter)` pairs.
pub fn adapters(
    params: &Params,
    resolver: &dyn TargetResolver,
    lora: &LoraFile,
    weight: f32,
) -> Result<(Vec<(String, Adapter)>, MergeReport)> {
    let kohya = kohya_map(resolver);
    let mut report = MergeReport::default();
    let mut out = vec![];
    for u in &lora.updates {
        let mut module = u.module.as_str();
        if let Some(rest) = module.strip_prefix("lora_unet_") {
            match kohya.get(rest) {
                Some(m) => module = m.as_str(),
                None => {
                    report.unmatched.push(u.module.clone());
                    continue;
                }
            }
        }
        let targets = resolver.resolve(module);
        let Some(down) = &u.down else {
            report.unmatched.push(format!("{} (full diff not supported with quantized weights)", u.module));
            continue;
        };
        if targets.is_empty() {
            report.unmatched.push(u.module.clone());
            continue;
        }
        for target in targets {
            let Some(w) = params.try_get(&target.param) else {
                report.unmatched.push(u.module.clone());
                continue;
            };
            let ws = w.shape();
            let rank = down.shape()[0];
            let rows = target.rows.unwrap_or(ws[0]);
            anyhow::ensure!(u.up.shape()[0] == rows && down.shape()[1] == ws[1], "LoRA {} does not fit {}", u.module, target.param);
            let scale = weight * u.alpha.map(|a| a / rank as f32).unwrap_or(1.0);
            let mut up = u.up.to_f32_vec().iter().map(|v| v * scale).collect::<Vec<f32>>();
            if target.swap_halves {
                let h = rows / 2 * rank;
                up.rotate_left(h);
            }
            let graph = w.graph().clone();
            let device = w.device();
            let dtype = if device.is_gpu() { w.dtype() } else { DataType::F32 };
            let up = graph.from_f32(&up, &[rows, rank], device, dtype);
            let down = upload(&graph, down, device, dtype);
            out.push((target.param.clone(), Adapter { down, up, row_offset: target.row_offset, rows }));
            report.touched.insert(target.param.clone());
            report.applied += 1;
        }
    }
    Ok((out, report))
}

/// `x @ w^T` plus any runtime adapters for this weight.
pub fn linear_with_adapters(x: &Var, w: &Var, adapters: Option<&Vec<Adapter>>) -> Var {
    let out = ops::linear(x, w, None);
    let Some(adapters) = adapters else { return out };
    let last = out.rank() - 1;
    let out_dim = out.dim(last);
    let mut out = out;
    for a in adapters {
        let t = ops::linear(x, &a.down, None);
        let u = ops::linear(&t, &a.up, None);
        if a.rows == out_dim {
            out = ops::add(&out, &u);
        } else {
            let slice = out.narrow(last, a.row_offset, a.rows);
            let sum = ops::add(&slice, &u);
            slice.assign(&sum);
        }
    }
    out
}

fn apply_update(w: &Var, u: &LoraUpdate, target: &LoraTarget, weight: f32) -> Result<()> {
    let graph = w.graph().clone();
    let device = w.device();
    // Compute the delta in f32 on CPU devices and in the weight's type on GPUs.
    let dtype = if device.is_gpu() { w.dtype() } else { DataType::F32 };
    let ws = w.shape();
    let (delta, scale) = match &u.down {
        Some(down) => {
            let rank = down.shape()[0];
            let scale = weight * u.alpha.map(|a| a / rank as f32).unwrap_or(1.0);
            let down = upload(&graph, down, device, dtype);
            let up = upload(&graph, &u.up, device, dtype);
            (ops::matmul(&up, &down, None, None), scale)
        }
        None => (upload(&graph, &u.up, device, dtype), weight),
    };
    let rows = target.rows.unwrap_or(ws[0]);
    let ds = delta.shape();
    anyhow::ensure!(
        ds.len() == 2 && ds[0] == rows && ds[1] == ws[1..].iter().product::<usize>(),
        "LoRA delta shape {ds:?} does not fit {:?} rows {}..{}",
        ws,
        target.row_offset,
        target.row_offset + rows
    );
    let delta = if target.swap_halves {
        let h = rows / 2;
        ops::concat(&[&delta.narrow(0, h, h), &delta.narrow(0, 0, h)], 0)
    } else {
        delta
    };
    let delta = delta.to_dtype(w.dtype()).reshape(&{
        let mut s = vec![rows];
        s.extend_from_slice(&ws[1..]);
        s
    });
    let slice = w.narrow(0, target.row_offset, rows);
    let updated = ops::add_scaled(&slice, &delta, 1.0, scale);
    slice.assign(&updated);
    Ok(())
}
