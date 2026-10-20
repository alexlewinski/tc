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
    /// The parameter holds input columns `col_offset..col_offset+cols` of the module the LoRA
    /// was trained on (`None` = all of them).
    pub col_offset: usize,
    pub cols: Option<usize>,
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
        if crate::ckpt::is_ckpt(path) {
            let f = crate::ckpt::CkptFile::open(path).with_context(|| format!("opening LoRA {}", path.display()))?;
            return LoraFile::from_ckpt(&f);
        }
        let files = SafeTensorsFiles::open(&[path]).with_context(|| format!("opening LoRA {}", path.display()))?;
        LoraFile::from_files(&files)
    }

    /// A Draw Things LoRA checkpoint: `<model tensor>__down__` `[rank, in]` and
    /// `<model tensor>__up__` `[out, rank]` per adapted tensor (e.g. `__dit__[t-x_q-3-0]__up__`),
    /// with any alpha already folded in (no alpha tensors are stored).
    pub fn from_ckpt(file: &crate::ckpt::CkptFile) -> Result<LoraFile> {
        let mut downs: BTreeMap<String, HostTensor> = BTreeMap::new();
        let mut ups: BTreeMap<String, HostTensor> = BTreeMap::new();
        let mut ignored = vec![];
        for key in file.keys() {
            if let Some(m) = key.strip_suffix("__down__") {
                downs.insert(m.to_string(), file.read(key, DataType::F32)?);
            } else if let Some(m) = key.strip_suffix("__up__") {
                ups.insert(m.to_string(), file.read(key, DataType::F32)?);
            } else {
                ignored.push(key.clone());
            }
        }
        let mut updates = vec![];
        for (module, down) in downs {
            let Some(up) = ups.remove(&module) else {
                ignored.push(format!("{module} (no __up__ matrix)"));
                continue;
            };
            updates.push(LoraUpdate { module, down: Some(flatten2(down)), up: flatten2(up), alpha: None });
        }
        for (module, _) in ups {
            ignored.push(format!("{module} (no __down__ matrix)"));
        }
        if updates.is_empty() {
            anyhow::bail!("{} has no LoRA weights (no `__down__` / `__up__` tensors)", file.path().display());
        }
        Ok(LoraFile { updates, ignored_keys: ignored })
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
        if module.starts_with("lora_te") || module.starts_with("text_encoder") || module.starts_with("__text_model__") {
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

/// Check, without touching any weights, that every update of `lora` resolves to a parameter of
/// a matching shape (`shape` gives canonical parameter shapes).
pub fn check(resolver: &dyn TargetResolver, lora: &LoraFile, shape: &dyn Fn(&str) -> Option<Vec<usize>>) -> MergeReport {
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
        let targets = resolver.resolve(module);
        if targets.is_empty() {
            report.unmatched.push(u.module.clone());
            continue;
        }
        for t in targets {
            let Some(ws) = shape(&t.param) else {
                report.unmatched.push(format!("{} ({} missing)", u.module, t.param));
                continue;
            };
            let rows = t.rows.unwrap_or(ws[0]);
            let in_dim = match &u.down {
                Some(d) => d.shape()[1],
                None => u.up.shape()[1],
            };
            let cols_ok = match t.cols {
                Some(n) => in_dim >= t.col_offset + n && n == ws[1],
                None => in_dim == ws[1],
            };
            if u.up.shape()[0] != rows || !cols_ok || t.row_offset + rows > ws[0] {
                report.unmatched.push(format!("{} (shape {:?} does not fit {} {:?})", u.module, u.up.shape(), t.param, ws));
                continue;
            }
            report.touched.insert(t.param.clone());
            report.applied += 1;
        }
    }
    report
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
            let down = match target.cols {
                Some(n) if down.shape()[1] >= target.col_offset + n => crate::models::flux2::host_cols(down, target.col_offset, n),
                _ => down.clone(),
            };
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
            let down = upload(&graph, &down, device, dtype);
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
    let cols = |t: &HostTensor| match target.cols {
        Some(n) if t.shape()[1] >= target.col_offset + n => crate::models::flux2::host_cols(t, target.col_offset, n),
        _ => t.clone(),
    };
    let (delta, scale) = match &u.down {
        Some(down) => {
            let rank = down.shape()[0];
            let scale = weight * u.alpha.map(|a| a / rank as f32).unwrap_or(1.0);
            let down = upload(&graph, &cols(down), device, dtype);
            let up = upload(&graph, &u.up, device, dtype);
            (ops::matmul(&up, &down, None, None), scale)
        }
        None => (upload(&graph, &cols(&u.up), device, dtype), weight),
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

/// Per-parameter comparison of the weight updates of two LoRAs (e.g. a Draw Things conversion
/// and its original): `(parameter, relative difference, best-fit scale of a relative to b)`.
pub fn compare_deltas(
    a: &LoraFile,
    b: &LoraFile,
    resolver: &dyn TargetResolver,
    shape: &dyn Fn(&str) -> Option<Vec<usize>>,
) -> Result<Vec<(String, f64, f64)>> {
    let kohya = kohya_map(resolver);
    // parameter -> [(update index, target)] for each file.
    let group = |l: &LoraFile| -> BTreeMap<String, Vec<(usize, LoraTarget)>> {
        let mut m: BTreeMap<String, Vec<(usize, LoraTarget)>> = BTreeMap::new();
        for (i, u) in l.updates.iter().enumerate() {
            let module = match u.module.strip_prefix("lora_unet_") {
                Some(rest) => match kohya.get(rest) {
                    Some(m) => m.clone(),
                    None => continue,
                },
                None => u.module.clone(),
            };
            for t in resolver.resolve(&module) {
                m.entry(t.param.clone()).or_default().push((i, t));
            }
        }
        m
    };
    let (ga, gb) = (group(a), group(b));
    // Each update is a factor pair: rows `r0..r0+n` of the parameter get `scale * up @ down`,
    // with `down` restricted to the parameter's columns.
    struct Factor {
        up: Vec<f32>,
        down: Vec<f32>,
        rank: usize,
        r0: usize,
        n: usize,
        m: usize,
        scale: f64,
    }
    let factors = |l: &LoraFile, list: &[(usize, LoraTarget)], ws: &[usize]| -> Vec<Factor> {
        list.iter()
            .filter_map(|(i, t)| {
                let u = &l.updates[*i];
                let down = u.down.as_ref()?;
                let rank = down.shape()[0];
                let n = t.rows.unwrap_or(ws[0]);
                let mut up = u.up.to_f32_vec();
                if t.swap_halves {
                    up.rotate_left(n / 2 * rank);
                }
                let m = t.cols.unwrap_or(ws[1]);
                let d = crate::models::flux2::host_cols(down, t.col_offset, m).to_f32_vec();
                Some(Factor { up, down: d, rank, r0: t.row_offset, n, m, scale: u.alpha.map(|a| a as f64 / rank as f64).unwrap_or(1.0) })
            })
            .collect()
    };
    // <s_x U_x V_x, s_y U_y V_y>_F = s_x s_y sum_ab (U_x^T U_y)_ab (V_x V_y^T)_ab over shared rows.
    let dot = |x: &Factor, y: &Factor| -> f64 {
        use rayon::prelude::*;
        let (r0, r1) = (x.r0.max(y.r0), (x.r0 + x.n).min(y.r0 + y.n));
        if r0 >= r1 || x.m != y.m {
            return 0.0;
        }
        let uu = (r0..r1)
            .into_par_iter()
            .fold(
                || vec![0f64; x.rank * y.rank],
                |mut acc, r| {
                    let (ux, uy) = (&x.up[(r - x.r0) * x.rank..][..x.rank], &y.up[(r - y.r0) * y.rank..][..y.rank]);
                    for a in 0..x.rank {
                        let v = ux[a] as f64;
                        for b in 0..y.rank {
                            acc[a * y.rank + b] += v * uy[b] as f64;
                        }
                    }
                    acc
                },
            )
            .reduce(
                || vec![0f64; x.rank * y.rank],
                |mut p, q| {
                    p.iter_mut().zip(q).for_each(|(a, b)| *a += b);
                    p
                },
            );
        let t: f64 = (0..x.rank * y.rank)
            .into_par_iter()
            .map(|ab| {
                let (a, b) = (ab / y.rank, ab % y.rank);
                let (vx, vy) = (&x.down[a * x.m..][..x.m], &y.down[b * y.m..][..y.m]);
                let vv: f64 = vx.chunks(256).zip(vy.chunks(256)).map(|(p, q)| p.iter().zip(q).map(|(a, b)| a * b).sum::<f32>() as f64).sum();
                uu[ab] * vv
            })
            .sum();
        x.scale * y.scale * t
    };
    let gram = |p: &[Factor], q: &[Factor]| -> f64 { p.iter().map(|x| q.iter().map(|y| dot(x, y)).sum::<f64>()).sum() };
    let mut rows = vec![];
    for (param, la) in &ga {
        let Some(lb) = gb.get(param) else {
            rows.push((param.clone(), f64::NAN, f64::NAN));
            continue;
        };
        let ws = shape(param).with_context(|| format!("unknown parameter {param}"))?;
        let ws = vec![ws[0], ws[1..].iter().product()];
        let (fa, fb) = (factors(a, la, &ws), factors(b, lb, &ws));
        let (aa, bb, ab) = (gram(&fa, &fa), gram(&fb, &fb), gram(&fa, &fb));
        let dd = (aa + bb - 2.0 * ab).max(0.0);
        rows.push((param.clone(), (dd / bb.max(1e-30)).sqrt(), ab / bb.max(1e-30)));
    }
    for param in gb.keys().filter(|p| !ga.contains_key(*p)) {
        rows.push((param.clone(), f64::NAN, f64::NAN));
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ckpt::tests::{tmp, Writer};
    use crate::models::flux2::{open_source, Flux2, Flux2LoraResolver};
    use nnc::{DataType, Device, Graph};

    fn fixture(p: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(p)
    }

    /// The fixture's PEFT LoRA written the way Draw Things stores LoRAs (`__dit__[...]__down__` /
    /// `__up__`, one pair per model tensor, fused projections split) must give the same updates.
    #[test]
    fn draw_things_lora_layout_matches_peft() {
        let (source, config) = open_source(&fixture("flux2/transformer")).unwrap();
        let resolver = Flux2LoraResolver { config: config.clone() };
        let peft = LoraFile::open(&fixture("flux2/lora_peft.safetensors")).unwrap();
        let get = |m: &str| peft.updates.iter().find(|u| u.module == m).unwrap();
        let (d, h, nd) = (config.inner_dim(), config.mlp_hidden(), config.num_layers);
        let path = tmp("dt_lora.ckpt");
        let mut w = Writer::new(&path, false);
        let mut pair = |name: &str, down: &HostTensor, up: &HostTensor, rows: std::ops::Range<usize>| {
            let r = up.shape()[1];
            let up = HostTensor::from_f32(&up.to_f32_vec()[rows.start * r..rows.end * r], &[rows.len(), r]);
            w.dense(&format!("__dit__[{name}]__down__"), &down.to_dtype(DataType::F16), crate::ckpt::CODEC_RAW);
            w.dense(&format!("__dit__[{name}]__up__"), &up.to_dtype(DataType::F16), crate::ckpt::CODEC_RAW);
        };
        let u = get("transformer_blocks.0.attn.to_k");
        pair("t-x_k-0-0", u.down.as_ref().unwrap(), &u.up, 0..d);
        let u = get("transformer_blocks.1.ff.linear_out");
        pair("t-x_down_proj-1-0", u.down.as_ref().unwrap(), &u.up, 0..d);
        let u = get("single_transformer_blocks.0.attn.to_qkv_mlp_proj");
        for (k, kind) in ["x_q", "x_k", "x_v"].iter().enumerate() {
            pair(&format!("t-{kind}-{nd}-0"), u.down.as_ref().unwrap(), &u.up, k * d..(k + 1) * d);
        }
        pair("t-x_w1-0-0", u.down.as_ref().unwrap(), &u.up, 3 * d..3 * d + h);
        pair("t-x_w3-0-0", u.down.as_ref().unwrap(), &u.up, 3 * d + h..3 * d + 2 * h);
        // diffusers norm_out.linear is (scale, shift); Draw Things ada_ln_0 = scale, ada_ln_1 = shift.
        let u = get("norm_out.linear");
        pair("t-ada_ln_0-0-0", u.down.as_ref().unwrap(), &u.up, 0..d);
        pair("t-ada_ln_1-0-0", u.down.as_ref().unwrap(), &u.up, d..2 * d);
        w.finish();

        let dt = LoraFile::open(&path).unwrap();
        let rep = check(&resolver, &dt, &|n| source.shape(n));
        assert!(rep.unmatched.is_empty(), "{:?}", rep.unmatched);
        let rows = compare_deltas(&dt, &peft, &resolver, &|n| source.shape(n)).unwrap();
        assert_eq!(rows.len(), 4);
        for (p, rel, scale) in &rows {
            assert!(*rel < 2e-3 && (scale - 1.0).abs() < 2e-3, "{p}: rel {rel} scale {scale}");
        }

        // Merged weights agree.
        let g = Graph::new();
        let mut a = Flux2::load(&g, &source, config.clone(), Device::Cpu, DataType::F32, false).unwrap();
        let mut b = Flux2::load(&g, &source, config.clone(), Device::Cpu, DataType::F32, false).unwrap();
        merge(&mut a.params, &resolver, &dt, 0.8).unwrap();
        merge(&mut b.params, &resolver, &peft, 0.8).unwrap();
        for n in config.parameter_names() {
            let (x, y) = (a.params.get(&n).to_f32_vec(), b.params.get(&n).to_f32_vec());
            let m = x.iter().zip(&y).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
            assert!(m < 2e-3, "{n}: {m}");
        }
    }
}
