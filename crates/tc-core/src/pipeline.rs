//! End-to-end FLUX.2 generation: text encoding, (optional) reference / init image encoding,
//! flow-matching sampling and decoding, with model and LoRA caching across requests.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use image::RgbImage;
use log::{info, warn};
use nnc::{ops, DataType, Device, Graph, HostTensor, Var};

use crate::image_io;
use crate::lora::{merge, LoraFile};
use crate::models::flux2::{image_ids, open_source, rope_table, text_ids, Flux2, Flux2Config, Flux2LoraResolver, Flux2Source};
use crate::models::flux2_vae::Flux2Vae;
use crate::models::qwen3::Qwen3;
use crate::rng::TorchCpuRng;
use crate::scheduler::{flux2_empirical_mu, img2img_start, sigmas, FlowMatchConfig, Shift};
use crate::tokenizer::QwenTokenizer;

/// Paths of the components of a FLUX.2 model.
#[derive(Clone, Debug, PartialEq)]
pub struct Flux2Files {
    /// Transformer: a diffusers `transformer/` directory, a single safetensors file or a Draw
    /// Things checkpoint (`.ckpt`).
    pub transformer: PathBuf,
    /// Text encoder: a diffusers `text_encoder/` directory or a Draw Things checkpoint.
    pub text_encoder: PathBuf,
    pub tokenizer: PathBuf,
    /// VAE: a diffusers `vae/` directory, a safetensors file or a Draw Things checkpoint.
    pub vae: PathBuf,
    pub scheduler_config: Option<PathBuf>,
    /// Step-distilled models ignore classifier-free guidance.
    pub is_distilled: bool,
    /// FLUX.2 [klein] KV variant: reference images are conditioned "causally" (they attend only
    /// to themselves, at timestep 0) and placed before the target tokens.
    pub kv: bool,
}

impl Flux2Files {
    /// Resolve the components of a Hugging Face (diffusers layout) snapshot directory.
    pub fn from_diffusers_dir(dir: &Path) -> Result<Flux2Files> {
        let need = |p: PathBuf| -> Result<PathBuf> {
            if p.exists() {
                Ok(p)
            } else {
                bail!("missing {}", p.display())
            }
        };
        let index: serde_json::Value = std::fs::read(dir.join("model_index.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or(serde_json::Value::Null);
        let is_distilled = index.get("is_distilled").and_then(|v| v.as_bool()).unwrap_or(false);
        let scheduler = dir.join("scheduler/scheduler_config.json");
        Ok(Flux2Files {
            transformer: need(dir.join("transformer"))?,
            text_encoder: need(dir.join("text_encoder"))?,
            tokenizer: need(dir.join("tokenizer"))?,
            vae: need(dir.join("vae"))?,
            scheduler_config: scheduler.exists().then_some(scheduler),
            is_distilled,
            kv: index.get("_class_name").and_then(|v| v.as_str()) == Some("Flux2KleinKVPipeline"),
        })
    }
}

/// Numeric precision per component.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Precision {
    pub transformer: DataType,
    pub text_encoder: DataType,
    pub vae: DataType,
}

impl Precision {
    pub fn default_for(device: Device) -> Precision {
        if device.is_gpu() {
            // FLUX.2 and Qwen3 are trained in bf16; fp16 activations can overflow. The VAE is
            // fine in f16 (and nnc's group norm has no bf16 Metal kernel).
            Precision { transformer: DataType::BF16, text_encoder: DataType::BF16, vae: DataType::F16 }
        } else {
            // nnc's CPU GEMM kernels are FP32 only.
            Precision { transformer: DataType::F32, text_encoder: DataType::F32, vae: DataType::F32 }
        }
    }
}

#[derive(Clone, Debug)]
pub struct EngineOptions {
    pub device: Device,
    pub precision: Precision,
    /// Keep the text encoder resident between requests (faster, uses more memory).
    pub keep_text_encoder: bool,
    /// Number of prompt embeddings to cache.
    pub prompt_cache_size: usize,
    /// Store transformer and text encoder linear weights 8-bit palettized (about half the
    /// memory; LoRAs then run as separate low-rank branches instead of being merged).
    pub quantize: bool,
    /// Keep weights that a Draw Things checkpoint stores 8-bit palettized (q8p) in that form
    /// (as if `quantize` were set for that component) instead of expanding them.
    pub keep_palettized: bool,
}

impl Default for EngineOptions {
    fn default() -> Self {
        let device = nnc::default_device();
        EngineOptions {
            device,
            precision: Precision::default_for(device),
            keep_text_encoder: false,
            prompt_cache_size: 8,
            quantize: false,
            keep_palettized: true,
        }
    }
}

/// A LoRA to apply, with its strength.
#[derive(Clone, Debug, PartialEq)]
pub struct LoraSpec {
    pub path: PathBuf,
    pub weight: f32,
}

/// Flow-matching sampler variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Sampler {
    /// Deterministic Euler (the reference FLUX.2 sampler).
    #[default]
    Euler,
    /// Re-noises the predicted clean latent with fresh noise at every step.
    EulerAncestral,
}

/// One generation request.
#[derive(Clone, Debug)]
pub struct GenerateParams {
    pub prompt: String,
    pub negative_prompt: String,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    pub guidance_scale: f32,
    pub seed: u64,
    /// Denoising strength for `init_image` (1.0 = ignore the init image's content).
    pub strength: f32,
    pub init_image: Option<RgbImage>,
    /// Conditioning (editing / reference) images.
    pub reference_images: Vec<RgbImage>,
    pub loras: Vec<LoraSpec>,
    /// `None` = FLUX.2's resolution dependent shift; `Some(k)` = constant shift `k`.
    pub shift: Option<f32>,
    pub sampler: Sampler,
    pub num_images: usize,
    /// Maximum text tokens.
    pub max_sequence_length: usize,
    /// Precomputed `(positive, negative)` text embeddings `[1, L, D]`; skips the text encoder.
    pub prompt_embeddings: Option<(HostTensor, Option<HostTensor>)>,
}

impl Default for GenerateParams {
    fn default() -> Self {
        GenerateParams {
            prompt: String::new(),
            negative_prompt: String::new(),
            width: 1024,
            height: 1024,
            steps: 4,
            guidance_scale: 1.0,
            seed: 0,
            strength: 1.0,
            init_image: None,
            reference_images: vec![],
            loras: vec![],
            shift: None,
            sampler: Sampler::Euler,
            num_images: 1,
            max_sequence_length: 512,
            prompt_embeddings: None,
        }
    }
}

/// Progress events reported during generation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Progress {
    TextEncoded,
    ImageEncoded,
    Sampling { step: usize, total: usize },
    ImageDecoded,
}

/// Cooperative cancellation flag.
#[derive(Clone, Default)]
pub struct Cancellation(Arc<AtomicBool>);

impl Cancellation {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// Generated images as `[1, H, W, 3]` f32 tensors in `[-1, 1]`.
pub struct GenerateOutput {
    pub images: Vec<HostTensor>,
}

struct LoadedTransformer {
    files: PathBuf,
    source: Flux2Source,
    model: Flux2,
    /// LoRAs currently merged and the parameters they modified.
    applied: Vec<LoraSpec>,
    touched: Vec<String>,
}

/// Holds models across requests. Not thread-safe: use one engine per worker thread.
pub struct Flux2Engine {
    pub options: EngineOptions,
    graph: Graph,
    files: Option<Flux2Files>,
    tokenizer: Option<QwenTokenizer>,
    text_encoder: Option<Qwen3>,
    transformer: Option<LoadedTransformer>,
    vae: Option<Flux2Vae>,
    prompt_cache: Vec<(String, usize, HostTensor)>,
}

/// Qwen3 layers whose hidden states FLUX.2 consumes.
pub const TEXT_ENCODER_LAYERS: [usize; 3] = [9, 18, 27];

impl Flux2Engine {
    pub fn new(mut options: EngineOptions) -> Flux2Engine {
        if options.quantize && options.device.is_gpu() {
            // Metal's palettized GEMM needs activations in the palette dtype, and its
            // depalettize kernel only supports f32/f16 palettes: quantized models run in f16.
            for dtype in [&mut options.precision.transformer, &mut options.precision.text_encoder] {
                if *dtype == DataType::BF16 {
                    info!("--quantize on GPU: running in f16 (bf16 palettes are not supported by Metal)");
                    *dtype = DataType::F16;
                }
            }
        }
        let graph = Graph::new();
        if options.device.is_gpu() {
            graph.use_gpu_stream();
        }
        Flux2Engine {
            options,
            graph,
            files: None,
            tokenizer: None,
            text_encoder: None,
            transformer: None,
            vae: None,
            prompt_cache: vec![],
        }
    }

    /// Select the model. Components are loaded lazily and kept until a different model is set.
    pub fn set_model(&mut self, files: Flux2Files) {
        if self.files.as_ref() == Some(&files) {
            return;
        }
        let old = self.files.take();
        if old.as_ref().map(|o| &o.text_encoder) != Some(&files.text_encoder) {
            self.text_encoder = None;
            self.tokenizer = None;
            self.prompt_cache.clear();
        }
        if old.as_ref().map(|o| &o.transformer) != Some(&files.transformer) {
            self.transformer = None;
        }
        if old.as_ref().map(|o| &o.vae) != Some(&files.vae) {
            self.vae = None;
        }
        self.files = Some(files);
        self.graph.garbage_collect();
    }

    /// Metal's palettized GEMM needs activations in the palette type and has no bf16 kernels.
    fn effective_dtype(&self, dtype: DataType, quantized: bool) -> DataType {
        if quantized && self.options.device.is_gpu() && dtype == DataType::BF16 {
            info!("8-bit palettized weights on GPU: running in f16 (Metal has no bf16 palette kernels)");
            DataType::F16
        } else {
            dtype
        }
    }

    fn files(&self) -> Result<&Flux2Files> {
        self.files.as_ref().context("no model selected")
    }

    fn ensure_transformer(&mut self) -> Result<()> {
        let files = self.files()?.clone();
        if self.transformer.as_ref().map(|t| t.files == files.transformer).unwrap_or(false) {
            return Ok(());
        }
        self.transformer = None;
        self.graph.garbage_collect();
        let t0 = Instant::now();
        let (source, config) = open_source(&files.transformer)?;
        let quantize = self.options.quantize || (self.options.keep_palettized && source.has_palettized());
        let dtype = self.effective_dtype(self.options.precision.transformer, quantize);
        let model = Flux2::load(&self.graph, &source, config, self.options.device, dtype, quantize)?;
        info!("loaded transformer {} in {:.1}s", files.transformer.display(), t0.elapsed().as_secs_f32());
        self.transformer = Some(LoadedTransformer { files: files.transformer, source, model, applied: vec![], touched: vec![] });
        Ok(())
    }

    /// Merge exactly `loras` into the transformer (restoring touched weights first if the set changed).
    fn apply_loras(&mut self, loras: &[LoraSpec]) -> Result<()> {
        let t = self.transformer.as_mut().context("transformer not loaded")?;
        if t.applied == loras {
            return Ok(());
        }
        for name in std::mem::take(&mut t.touched) {
            t.model.reload_param(&self.graph, &t.source, &name)?;
        }
        t.applied.clear();
        t.model.adapters.clear();
        let resolver = Flux2LoraResolver { config: t.model.config.clone() };
        let mut touched = std::collections::BTreeSet::new();
        for spec in loras {
            let file = LoraFile::open(&spec.path)?;
            let report = if t.model.quantized {
                let (list, report) = crate::lora::adapters(&t.model.params, &resolver, &file, spec.weight)
                    .with_context(|| format!("loading LoRA {}", spec.path.display()))?;
                for (param, a) in list {
                    t.model.adapters.entry(param).or_default().push(a);
                }
                report
            } else {
                merge(&mut t.model.params, &resolver, &file, spec.weight).with_context(|| format!("merging LoRA {}", spec.path.display()))?
            };
            info!("LoRA {} (weight {}): {} updates applied", spec.path.display(), spec.weight, report.applied);
            if !report.unmatched.is_empty() {
                warn!(
                    "LoRA {}: {} modules not matched (e.g. {:?})",
                    spec.path.display(),
                    report.unmatched.len(),
                    &report.unmatched[..report.unmatched.len().min(3)]
                );
            }
            if report.applied == 0 {
                bail!("LoRA {} does not match this model", spec.path.display());
            }
            if !t.model.quantized {
                touched.extend(report.touched);
            }
        }
        t.touched = touched.into_iter().collect();
        t.applied = loras.to_vec();
        // Weights were rewritten in place on the stream; finish before sampling reads them.
        self.graph.synchronize();
        Ok(())
    }

    fn ensure_vae(&mut self) -> Result<&Flux2Vae> {
        if self.vae.is_none() {
            let files = self.files()?.clone();
            self.vae = Some(Flux2Vae::load(&self.graph, &files.vae, self.options.device, self.options.precision.vae)?);
        }
        Ok(self.vae.as_ref().unwrap())
    }

    /// Prompt embedding `[1, L, 3*hidden]` on the host (f32).
    pub fn encode_prompt(&mut self, prompt: &str, max_len: usize) -> Result<HostTensor> {
        if let Some((_, _, t)) = self.prompt_cache.iter().find(|(p, l, _)| p == prompt && *l == max_len) {
            return Ok(t.clone());
        }
        let files = self.files()?.clone();
        if self.tokenizer.is_none() {
            self.tokenizer = Some(QwenTokenizer::load(&files.tokenizer)?);
        }
        if self.text_encoder.is_none() {
            let t0 = Instant::now();
            let layers = *TEXT_ENCODER_LAYERS.iter().max().unwrap();
            let palettized = self.options.keep_palettized && crate::ckpt::is_ckpt(&files.text_encoder) && {
                let f = crate::ckpt::CkptFile::open(&files.text_encoder)?;
                f.info("__text_model__[t-q_proj-0-0]").map(|i| i.codec == crate::ckpt::CODEC_Q8P).unwrap_or(false)
            };
            let quantize = self.options.quantize || palettized;
            let dtype = self.effective_dtype(self.options.precision.text_encoder, quantize);
            self.text_encoder = Some(Qwen3::load(&self.graph, &files.text_encoder, layers, self.options.device, dtype, quantize)?);
            info!("loaded text encoder in {:.1}s", t0.elapsed().as_secs_f32());
        }
        let text = QwenTokenizer::chat_prompt(prompt);
        let (ids, valid) = self.tokenizer.as_ref().unwrap().encode_padded(&text, max_len)?;
        let h = self.text_encoder.as_ref().unwrap().hidden_states(&ids, valid, &TEXT_ENCODER_LAYERS)?;
        let host = h.to_dtype(DataType::F32).to_host();
        if !self.options.keep_text_encoder {
            self.text_encoder = None;
            self.graph.garbage_collect();
        }
        self.prompt_cache.push((prompt.to_string(), max_len, host.clone()));
        if self.prompt_cache.len() > self.options.prompt_cache_size {
            self.prompt_cache.remove(0);
        }
        Ok(host)
    }

    /// Encode an RGB image to packed latents `[1, h, w, 128]` (h, w = size / 16).
    fn encode_image(&mut self, img: &RgbImage) -> Result<Var> {
        let device = self.options.device;
        let dtype = self.options.precision.vae;
        let graph = self.graph.clone();
        let vae = self.ensure_vae()?;
        let x = graph.from_host(&image_io::to_tensor(img), device).to_dtype(dtype);
        let raw = vae.encode_raw(&x);
        Ok(vae.pack_latents(&raw))
    }

    /// Run a generation. `progress` returns `false` to cancel.
    pub fn generate(
        &mut self,
        p: &GenerateParams,
        cancel: &Cancellation,
        progress: &mut dyn FnMut(Progress) -> bool,
    ) -> Result<GenerateOutput> {
        if p.width % 16 != 0 || p.height % 16 != 0 || p.width == 0 || p.height == 0 {
            bail!("width and height must be positive multiples of 16 (got {}x{})", p.width, p.height);
        }
        let check = |c: &Cancellation| -> Result<()> {
            if c.is_cancelled() {
                bail!("cancelled")
            }
            Ok(())
        };
        let t_start = Instant::now();
        let is_distilled = self.files()?.is_distilled;
        let do_cfg = p.guidance_scale > 1.0 && !is_distilled;
        if p.guidance_scale > 1.0 && is_distilled {
            info!("guidance scale {} ignored for a step-distilled model", p.guidance_scale);
        }

        // 1. Text.
        let (cond, uncond) = match &p.prompt_embeddings {
            Some((c, u)) => (c.clone(), if do_cfg { Some(u.clone().context("negative prompt embedding required for CFG")?) } else { None }),
            None => {
                let c = self.encode_prompt(&p.prompt, p.max_sequence_length)?;
                let u = if do_cfg { Some(self.encode_prompt(&p.negative_prompt, p.max_sequence_length)?) } else { None };
                (c, u)
            }
        };
        if !progress(Progress::TextEncoded) {
            cancel.cancel();
        }
        check(cancel)?;

        // 2. Images.
        let (lh, lw) = ((p.height / 16) as usize, (p.width / 16) as usize);
        let mut refs: Vec<Var> = vec![];
        let mut ref_ids: Vec<[f32; 4]> = vec![];
        for (i, img) in p.reference_images.iter().enumerate() {
            let img = prepare_reference(img);
            let lat = self.encode_image(&img)?;
            let s = lat.shape();
            ref_ids.extend(image_ids(s[1], s[2], 10.0 * (i as f32 + 1.0)));
            refs.push(lat.reshape(&[1, s[1] * s[2], s[3]]));
        }
        let init = match &p.init_image {
            Some(img) if p.strength < 1.0 => {
                let img = image_io::resize_cover_crop(img, p.width, p.height);
                Some(self.encode_image(&img)?)
            }
            _ => None,
        };
        if !refs.is_empty() || init.is_some() {
            if !progress(Progress::ImageEncoded) {
                cancel.cancel();
            }
            check(cancel)?;
        }

        // 3. Transformer + LoRAs (and the VAE for decoding).
        self.ensure_vae()?;
        let sched_cfg = self.files()?.scheduler_config.as_ref().map(|p| FlowMatchConfig::load(p)).unwrap_or_default();
        self.ensure_transformer()?;
        self.apply_loras(&p.loras)?;
        let device = self.options.device;
        let graph = self.graph.clone();
        let t = self.transformer.as_ref().unwrap();
        // The loaded model's type, not the requested one: palettized weights force f16 on GPU.
        let dtype = t.model.dtype;
        let model = &t.model;
        let config: &Flux2Config = &model.config;
        let channels = config.in_channels;

        let cond_v = graph.from_host(&cond, device).to_dtype(dtype);
        let uncond_v = uncond.as_ref().map(|u| graph.from_host(u, device).to_dtype(dtype));
        let txt_len = cond.shape()[1];
        let kv = self.files()?.kv && !refs.is_empty();
        let mut ids = text_ids(txt_len);
        if kv {
            // KV variant: reference tokens come before the target tokens.
            ids.extend(ref_ids.iter().copied());
            ids.extend(image_ids(lh, lw, 0.0));
        } else {
            ids.extend(image_ids(lh, lw, 0.0));
            ids.extend(ref_ids.iter().copied());
        }
        let (rope, rope_shape) = rope_table(&ids, &config.axes_dims_rope, config.rope_theta);
        let rope = graph.from_f32(&rope, &rope_shape, device, dtype);
        let refs_cat = if refs.is_empty() {
            None
        } else {
            let r: Vec<Var> = refs.iter().map(|v| v.to_dtype(dtype)).collect();
            let rr: Vec<&Var> = r.iter().collect();
            Some(ops::concat(&rr, 1))
        };

        // 4. Schedule.
        let n_tokens = lh * lw;
        let shift = match p.shift {
            Some(k) => Shift::Constant(k as f64),
            None => Shift::Mu(flux2_empirical_mu(n_tokens, p.steps)),
        };
        let sig = sigmas(p.steps, shift, &sched_cfg);
        let start = if init.is_some() { img2img_start(p.steps, p.strength) } else { 0 };
        if start >= p.steps {
            bail!("strength {} leaves no denoising steps", p.strength);
        }

        let n_ref = refs_cat.as_ref().map(|r| r.dim(1)).unwrap_or(0);
        let ref_step = if kv { Some(model.prepare_step(&[0.0], None)) } else { None };
        let mut images = vec![];
        for n in 0..p.num_images.max(1) {
            let seed = p.seed.wrapping_add(n as u64);
            // Noise in NCHW order (matching reference pipelines for a given seed) -> tokens.
            let mut rng = TorchCpuRng::new(seed);
            let draw_noise = |rng: &mut TorchCpuRng| {
                let noise = rng.randn(channels * n_tokens);
                let mut tokens = vec![0f32; noise.len()];
                for c in 0..channels {
                    for i in 0..n_tokens {
                        tokens[i * channels + c] = noise[c * n_tokens + i];
                    }
                }
                graph.from_f32(&tokens, &[1, n_tokens, channels], device, DataType::F32)
            };
            let noise = draw_noise(&mut rng);
            // Keep the running latent in f32 for accuracy across steps.
            let mut x = match &init {
                Some(init) => {
                    let s = sig[start] as f32;
                    let x0 = init.to_dtype(DataType::F32).reshape(&[1, n_tokens, channels]);
                    ops::add_scaled(&x0, &noise, 1.0 - s, s)
                }
                None => noise,
            };
            for i in start..p.steps {
                check(cancel)?;
                let s = sig[i];
                let s_next = sig[i + 1];
                let step = model.prepare_step(&[(s * 1000.0) as f32], None);
                let xin = x.to_dtype(dtype);
                let run = |txt: &Var| -> Var {
                    match (&refs_cat, &ref_step) {
                        (Some(r), Some(rs)) => model.forward_kv(&ops::concat(&[r, &xin], 1), txt, &step, rs, n_ref, &rope),
                        (Some(r), None) => model.forward(&ops::concat(&[&xin, r], 1), txt, &step, &rope).narrow(1, 0, n_tokens).contiguous(),
                        (None, _) => model.forward(&xin, txt, &step, &rope),
                    }
                };
                let mut v = run(&cond_v);
                if let Some(u) = &uncond_v {
                    let vu = run(u);
                    // uncond + g * (cond - uncond)
                    let d = ops::sub(&v, &vu);
                    v = ops::add_scaled(&vu, &d, 1.0, p.guidance_scale);
                }
                let v = v.to_dtype(DataType::F32);
                if p.sampler == Sampler::EulerAncestral && s_next > 0.0 {
                    let x0 = ops::add_scaled(&x, &v, 1.0, -(s as f32));
                    let fresh = draw_noise(&mut rng);
                    x = ops::add_scaled(&x0, &fresh, 1.0 - s_next as f32, s_next as f32);
                } else {
                    x = ops::add_scaled(&x, &v, 1.0, (s_next - s) as f32);
                }
                if !progress(Progress::Sampling { step: i + 1, total: p.steps }) {
                    cancel.cancel();
                }
            }
            check(cancel)?;
            let vae_dtype = self.options.precision.vae;
            let vae = self.vae.as_ref().unwrap();
            let packed = x.reshape(&[1, lh, lw, channels]).to_dtype(vae_dtype);
            let raw = vae.unpack_latents(&packed);
            let img = vae.decode_raw(&raw).to_dtype(DataType::F32).to_host();
            if !progress(Progress::ImageDecoded) {
                cancel.cancel();
            }
            images.push(img);
        }
        info!("generated {} image(s) in {:.1}s", images.len(), t_start.elapsed().as_secs_f32());
        Ok(GenerateOutput { images })
    }
}

/// Reference images are limited to ~1MP and cropped to multiples of 16 (as in the reference
/// FLUX.2 [klein] pipeline).
fn prepare_reference(img: &RgbImage) -> RgbImage {
    let (mut w, mut h) = (img.width(), img.height());
    let mut img = img.clone();
    if (w as u64) * (h as u64) > 1024 * 1024 {
        let scale = ((1024.0 * 1024.0) / (w as f64 * h as f64)).sqrt();
        w = (w as f64 * scale) as u32;
        h = (h as f64 * scale) as u32;
        img = image_io::resize(&img, w, h);
    }
    let (cw, ch) = ((w / 16) * 16, (h / 16) * 16);
    image_io::resize_cover_crop(&img, cw.max(16), ch.max(16))
}

/// LoRA name -> path resolution helper used by front-ends.
pub fn resolve_lora_path(name: &str, search_dirs: &[PathBuf]) -> Option<PathBuf> {
    let p = Path::new(name);
    if p.is_absolute() && p.exists() {
        return Some(p.to_path_buf());
    }
    for d in search_dirs {
        for candidate in [d.join(name), d.join("loras").join(name), d.join(format!("{name}.safetensors"))] {
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

/// Simple per-key memo (used by front-ends to share engines per model).
pub type EngineMap = HashMap<String, Flux2Engine>;
