//! Device-vs-CPU diagnostic for a model as `generate` loads it: the same weights (8-bit palettized
//! where the checkpoint is, in the same precision) run on the target device and, expanded to f32,
//! on the CPU, on identical inputs. Reports the relative error of each stage, from single
//! palettized matrix products up to a reduced transformer (first blocks + final layer) in plain
//! and KV mode with LoRAs, the first Qwen3 layers and a VAE round trip.

use anyhow::{Context, Result};
use nnc::{ops, DataType, Device, Graph, HostTensor, Var};

use crate::lora::{adapters, merge, LoraFile};
use crate::models::flux2::{image_ids, open_source, rope_table, text_ids, upload_param, Flux2, Flux2LoraResolver};
use crate::models::flux2_vae::Flux2Vae;
use crate::models::qwen3::Qwen3;
use crate::pipeline::{Flux2Files, LoraSpec, Precision};

/// What to check.
#[derive(Clone, Debug)]
pub struct CheckOptions {
    pub device: Device,
    /// Transformer / text encoder precision before palettization forces f16 on GPU.
    pub precision: Precision,
    /// Double and single blocks of the reduced transformer.
    pub double_blocks: usize,
    pub single_blocks: usize,
    pub text_encoder: bool,
    pub vae: bool,
}

/// `||a - b|| / ||b||`, and whether `a` has non-finite values.
pub fn rel(a: &[f32], b: &[f32]) -> (f64, bool) {
    let bad = a.iter().any(|v| !v.is_finite());
    let d: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let n: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    ((d / n.max(1e-30)).sqrt(), bad)
}

struct Rng(u64);

impl Rng {
    fn vec(&mut self, n: usize, amp: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * amp
            })
            .collect()
    }
}

fn verdict(r: f64, bad: bool, ok: f64) -> &'static str {
    if bad {
        "FAIL (non-finite values)"
    } else if r <= ok {
        "ok"
    } else if r <= ok * 10.0 {
        "suspicious"
    } else {
        "FAIL"
    }
}

/// Run the checks, writing one line per stage to `out`.
pub fn run(files: &Flux2Files, loras: &[LoraSpec], o: &CheckOptions, out: &mut dyn FnMut(String)) -> Result<()> {
    let dev = o.device;
    let gg = Graph::new();
    if dev.is_gpu() {
        gg.use_gpu_stream();
    }
    let gc = Graph::new();
    let mut rng = Rng(7);
    // Reduced-precision tolerance (f16 / bf16 activations); the CPU side is f32.
    let tol = |dtype: DataType| if dtype == DataType::F32 { 1e-3 } else { 2e-2 };

    // 1. Transformer: single matrix products, then the reduced model.
    let (source, full) = open_source(&files.transformer)?;
    let quantize = source.has_palettized();
    let dtype = if quantize && dev.is_gpu() && o.precision.transformer == DataType::BF16 { DataType::F16 } else { o.precision.transformer };
    out(format!(
        "transformer {} on {dev:?}: {} weights, compute {dtype:?}, {} double + {} single blocks checked",
        files.transformer.display(),
        if quantize { "8-bit palettized" } else { "dense" },
        o.double_blocks.min(full.num_layers),
        o.single_blocks.min(full.num_single_layers)
    ));
    for name in [
        "txt_in.weight",
        "double_stream_modulation_img.lin.weight",
        "double_blocks.0.img_attn.qkv.weight",
        "double_blocks.0.img_attn.proj.weight",
        "double_blocks.0.img_mlp.0.weight",
        "double_blocks.0.img_mlp.2.weight",
        "single_blocks.0.linear1.weight",
        "single_blocks.0.linear2_attn.weight",
        "single_blocks.0.linear2_mlp.weight",
    ] {
        let raw = source.read_raw(name)?;
        let shape = raw.shape().to_vec();
        let w_dev = upload_param(&gg, raw.clone(), dev, dtype, quantize, source.is_draw_things());
        let w_cpu = gc.from_host(&raw.to_host(DataType::F32), Device::Cpu);
        for t in [1usize, 77] {
            let x = rng.vec(t * shape[1], 1.0);
            let xd = gg.from_f32(&x, &[1, t, shape[1]], dev, dtype);
            let xc = gc.from_f32(&x, &[1, t, shape[1]], Device::Cpu, DataType::F32);
            let yd = ops::linear(&xd, &w_dev, None).to_f32_vec();
            let yc = ops::linear(&xc, &w_cpu, None).to_f32_vec();
            line(out, &format!("matmul {name} {shape:?} x {t} tokens{}", if w_dev.is_quantized() { " (palettized)" } else { "" }), rel(&yd, &yc), tol(dtype));
        }
    }
    gg.garbage_collect();

    let mut config = full.clone();
    config.num_layers = o.double_blocks.min(full.num_layers);
    config.num_single_layers = o.single_blocks.min(full.num_single_layers);
    let mut md = Flux2::load(&gg, &source, config.clone(), dev, dtype, quantize).context("loading the transformer on the device")?;
    let mut mc = Flux2::load(&gc, &source, config.clone(), Device::Cpu, DataType::F32, false).context("loading the transformer on the CPU")?;
    let resolver = Flux2LoraResolver { config: config.clone() };
    for l in loras {
        let f = LoraFile::open(&l.path)?;
        let rc = merge(&mut mc.params, &resolver, &f, l.weight)?;
        if md.quantized {
            let (list, _) = adapters(&md.params, &resolver, &f, l.weight)?;
            for (p, a) in list {
                md.adapters.entry(p).or_default().push(a);
            }
        } else {
            merge(&mut md.params, &resolver, &f, l.weight)?;
        }
        out(format!("LoRA {} (weight {}): {} updates in the checked blocks", l.path.display(), l.weight, rc.applied));
    }
    let c = &config;
    let (lt, h, w, rh, rw) = (32usize, 8usize, 8usize, 6usize, 8usize);
    let txt = rng.vec(lt * c.joint_attention_dim, 3.0);
    let img = rng.vec(h * w * c.in_channels, 1.0);
    let rf = rng.vec(rh * rw * c.in_channels, 1.0);
    let both = |g: &Graph, d: Device, t: DataType| {
        (
            g.from_f32(&txt, &[1, lt, c.joint_attention_dim], d, t),
            g.from_f32(&img, &[1, h * w, c.in_channels], d, t),
            g.from_f32(&rf, &[1, rh * rw, c.in_channels], d, t),
        )
    };
    let (td, id, rd) = both(&gg, dev, dtype);
    let (tc, ic, rc) = both(&gc, Device::Cpu, DataType::F32);
    let rope = |g: &Graph, d: Device, t: DataType, kv: bool| {
        let mut ids = text_ids(lt);
        if kv {
            ids.extend(image_ids(rh, rw, 10.0));
            ids.extend(image_ids(h, w, 0.0));
        } else {
            ids.extend(image_ids(h, w, 0.0));
            ids.extend(image_ids(rh, rw, 10.0));
        }
        let (r, s) = rope_table(&ids, &c.axes_dims_rope, c.rope_theta);
        g.from_f32(&r, &s, d, t)
    };
    let sd = md.prepare_step(&[700.0], None);
    let sc = mc.prepare_step(&[700.0], None);
    line(out, "timestep embedding + modulation (img, first chunk)", rel(&sd_first(&md, &sd), &sd_first(&mc, &sc)), tol(dtype));
    let yd = md.forward(&ops::concat(&[&id, &rd], 1), &td, &sd, &rope(&gg, dev, dtype, false)).to_f32_vec();
    let yc = mc.forward(&ops::concat(&[&ic, &rc], 1), &tc, &sc, &rope(&gc, Device::Cpu, DataType::F32, false)).to_f32_vec();
    line(out, "reduced transformer, plain (image + reference tokens)", rel(&yd, &yc), tol(dtype) * 2.0);
    let (r0d, r0c) = (md.prepare_step(&[0.0], None), mc.prepare_step(&[0.0], None));
    let n = rh * rw;
    let yd = md.forward_kv(&ops::concat(&[&rd, &id], 1), &td, &sd, &r0d, n, &rope(&gg, dev, dtype, true)).to_f32_vec();
    let yc = mc.forward_kv(&ops::concat(&[&rc, &ic], 1), &tc, &sc, &r0c, n, &rope(&gc, Device::Cpu, DataType::F32, true)).to_f32_vec();
    line(out, "reduced transformer, KV variant (reference tokens first)", rel(&yd, &yc), tol(dtype) * 2.0);
    drop(md);
    drop(mc);
    gg.garbage_collect();
    gc.garbage_collect();

    // 2. Text encoder: first layers.
    if o.text_encoder {
        let layers = 2;
        let palettized = crate::ckpt::is_ckpt(&files.text_encoder)
            && crate::ckpt::CkptFile::open(&files.text_encoder)?
                .info("__text_model__[t-q_proj-0-0]")
                .map(|i| i.codec == crate::ckpt::CODEC_Q8P)
                .unwrap_or(false);
        let tdt = if palettized && dev.is_gpu() && o.precision.text_encoder == DataType::BF16 { DataType::F16 } else { o.precision.text_encoder };
        let qd = Qwen3::load(&gg, &files.text_encoder, layers, dev, tdt, palettized)?;
        let qc = Qwen3::load(&gc, &files.text_encoder, layers, Device::Cpu, DataType::F32, false)?;
        let vocab = qc.config.vocab_size;
        let ids: Vec<i32> = (0..32).map(|i| (i * 7919 % vocab) as i32).collect();
        let a = qd.hidden_states(&ids, 28, &[1, 2])?.to_dtype(DataType::F32).to_f32_vec();
        let b = qc.hidden_states(&ids, 28, &[1, 2])?.to_f32_vec();
        line(out, &format!("Qwen3 first {layers} layers ({tdt:?}{})", if palettized { ", palettized" } else { "" }), rel(&a, &b), tol(tdt) * 2.0);
        drop(qd);
        drop(qc);
        gg.garbage_collect();
        gc.garbage_collect();
    }

    // 3. VAE.
    if o.vae {
        let vd = Flux2Vae::load(&gg, &files.vae, dev, o.precision.vae)?;
        let vc = Flux2Vae::load(&gc, &files.vae, Device::Cpu, DataType::F32)?;
        let lc = 4 * vc.config.latent_channels;
        let z = rng.vec(8 * 8 * lc, 1.0);
        let zd = gg.from_f32(&z, &[1, 8, 8, lc], dev, o.precision.vae);
        let zc = gc.from_f32(&z, &[1, 8, 8, lc], Device::Cpu, DataType::F32);
        let a = vd.decode_raw(&vd.unpack_latents(&zd)).to_dtype(DataType::F32).to_f32_vec();
        let b = vc.decode_raw(&vc.unpack_latents(&zc)).to_f32_vec();
        line(out, &format!("VAE decode 128x128 ({:?})", o.precision.vae), rel(&a, &b), tol(o.precision.vae));
        let im: Vec<f32> = b.iter().map(|v| v.clamp(-1.0, 1.0)).collect();
        let xd = gg.from_host(&HostTensor::from_f32(&im, &[1, 128, 128, 3]), dev).to_dtype(o.precision.vae);
        let xc = gc.from_f32(&im, &[1, 128, 128, 3], Device::Cpu, DataType::F32);
        let a = vd.pack_latents(&vd.encode_raw(&xd)).to_dtype(DataType::F32).to_f32_vec();
        let b = vc.pack_latents(&vc.encode_raw(&xc)).to_f32_vec();
        line(out, &format!("VAE encode 128x128 ({:?})", o.precision.vae), rel(&a, &b), tol(o.precision.vae) * 2.0);
    }
    Ok(())
}

fn line(out: &mut dyn FnMut(String), name: &str, r: (f64, bool), ok: f64) {
    out(format!("{name:<62} rel {:>9.5}  {}", r.0, verdict(r.0, r.1, ok)));
}

fn sd_first(_m: &Flux2, s: &crate::models::flux2::Flux2Step) -> Vec<f32> {
    let v: &Var = s.img_mod_first();
    v.to_dtype(DataType::F32).to_f32_vec()
}
