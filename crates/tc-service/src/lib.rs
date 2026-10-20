//! Shared front-end logic for `draw-things-cli` and `gRPCServerCLI`: model registry,
//! translation of a [`GenerationConfiguration`] into a pipeline request, and a worker thread
//! that owns the inference engine.

pub mod zoo;

use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread::JoinHandle;

use anyhow::{bail, Context, Result};
use image::RgbImage;
use log::{info, warn};
use nnc::HostTensor;
pub use tc_config::GenerationConfiguration;
use tc_core::pipeline::{Cancellation, EngineOptions, Flux2Engine, GenerateParams, LoraSpec, Progress, Sampler};

pub use zoo::{ModelSpec, ModelStore, ResolvedModel};

/// Everything a generation needs besides the configuration.
#[derive(Clone, Debug, Default)]
pub struct GenerationInput {
    pub config: GenerationConfiguration,
    pub prompt: String,
    pub negative_prompt: String,
    /// The canvas image. With `strength < 1` it initializes the latent (image-to-image);
    /// otherwise it is used as a reference (editing) image.
    pub image: Option<RgbImage>,
    /// Additional reference ("moodboard") images.
    pub references: Vec<RgbImage>,
}

/// Fill unset (zero) configuration fields with the model's recommended settings.
pub fn apply_recommended(config: &mut GenerationConfiguration, spec: Option<&ModelSpec>) {
    let (steps, cfg, size) = spec.map(|s| (s.default_steps, s.default_guidance, s.default_size)).unwrap_or((4, 1.0, 1024));
    if config.steps == 0 {
        config.steps = steps;
    }
    if config.guidance_scale == 0.0 {
        config.guidance_scale = cfg;
    }
    if config.start_width == 0 {
        config.start_width = (size / 64) as u16;
    }
    if config.start_height == 0 {
        config.start_height = (size / 64) as u16;
    }
    if config.strength == 0.0 {
        config.strength = 1.0;
    }
}

/// Recommended configuration for a model (used by the CLI as the base for overrides).
pub fn recommended_configuration(model: &str, spec: Option<&ModelSpec>) -> GenerationConfiguration {
    let mut c = GenerationConfiguration { model: Some(model.to_string()), ..Default::default() };
    apply_recommended(&mut c, spec);
    // Flow-matching models: deterministic Euler with resolution dependent shift.
    c.sampler = tc_config::SamplerType::DDIMTrailing as i8;
    c.resolution_dependent_shift = true;
    c
}

/// Translate a request into pipeline parameters.
pub fn build_params(store: &ModelStore, input: &GenerationInput) -> Result<(ResolvedModel, GenerateParams)> {
    let model_ref = input.config.model.as_deref().filter(|m| !m.is_empty()).context("configuration has no model")?;
    let resolved = store.resolve(model_ref)?;
    let mut c = input.config.clone();
    apply_recommended(&mut c, resolved.spec.as_ref());
    let mut loras = vec![];
    for l in &c.loras {
        let Some(file) = l.file.as_deref().filter(|f| !f.is_empty()) else { continue };
        if l.mode == 2 {
            continue; // refiner-only
        }
        if l.weight == 0.0 {
            continue;
        }
        let path = store.resolve_lora(file).with_context(|| format!("LoRA {file:?} not found in {:?}", store.dirs))?;
        loras.push(LoraSpec { path, weight: l.weight });
    }
    let mut references = vec![];
    let mut init = None;
    if let Some(img) = &input.image {
        if c.strength < 1.0 {
            init = Some(img.clone());
        } else {
            references.push(img.clone());
        }
    }
    references.extend(input.references.iter().cloned());
    let sampler = if c.sampler_type().is_ancestral() { Sampler::EulerAncestral } else { Sampler::Euler };
    let params = GenerateParams {
        prompt: input.prompt.clone(),
        negative_prompt: input.negative_prompt.clone(),
        width: c.width_px(),
        height: c.height_px(),
        steps: c.steps.max(1) as usize,
        guidance_scale: c.guidance_scale,
        seed: c.seed as u64,
        strength: c.strength.clamp(0.0, 1.0),
        init_image: init,
        reference_images: references,
        loras,
        shift: if c.resolution_dependent_shift { None } else { Some(c.shift) },
        sampler,
        num_images: (c.batch_count.max(1) * c.batch_size.max(1)) as usize,
        max_sequence_length: 512,
        prompt_embeddings: None,
    };
    if params.width == 0 || params.height == 0 {
        bail!("invalid size {}x{}", params.width, params.height);
    }
    Ok((resolved, params))
}

/// Events streamed back from the worker.
#[derive(Debug)]
pub enum Event {
    Progress(Progress),
    Finished(Result<Vec<HostTensor>>),
}

struct Job {
    input: GenerationInput,
    cancel: Cancellation,
    events: Sender<Event>,
}

/// Owns the engine on a dedicated thread; jobs run one at a time in submission order.
pub struct Worker {
    jobs: Option<Sender<Job>>,
    handle: Option<JoinHandle<()>>,
    pub store: ModelStore,
}

impl Worker {
    pub fn spawn(store: ModelStore, options: EngineOptions) -> Worker {
        let (tx, rx) = channel::<Job>();
        let worker_store = store.clone();
        let handle = std::thread::Builder::new()
            .name("tc-generation".into())
            .spawn(move || {
                let mut engine = Flux2Engine::new(options);
                while let Ok(job) = rx.recv() {
                    let result = run_job(&mut engine, &worker_store, &job);
                    if let Err(e) = &result {
                        warn!("generation failed: {e:#}");
                    }
                    let _ = job.events.send(Event::Finished(result));
                }
            })
            .expect("spawn worker");
        Worker { jobs: Some(tx), handle: Some(handle), store }
    }

    /// Queue a generation. Progress and the final result arrive on the returned channel.
    pub fn submit(&self, input: GenerationInput, cancel: Cancellation) -> Receiver<Event> {
        let (tx, rx) = channel();
        let job = Job { input, cancel, events: tx };
        if let Some(jobs) = &self.jobs {
            if let Err(e) = jobs.send(job) {
                let _ = e.0.events.send(Event::Finished(Err(anyhow::anyhow!("worker stopped"))));
            }
        }
        rx
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.jobs.take();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn run_job(engine: &mut Flux2Engine, store: &ModelStore, job: &Job) -> Result<Vec<HostTensor>> {
    let (resolved, params) = build_params(store, &job.input)?;
    info!(
        "generating {}x{} steps={} cfg={} seed={} loras={} refs={} model={}",
        params.width,
        params.height,
        params.steps,
        params.guidance_scale,
        params.seed,
        params.loras.len(),
        params.reference_images.len(),
        resolved.dir.display()
    );
    engine.set_model(resolved.files);
    let events = job.events.clone();
    let out = engine.generate(&params, &job.cancel, &mut |p| events.send(Event::Progress(p)).is_ok())?;
    Ok(out.images)
}

/// Paths searched for models and LoRAs.
pub fn model_dirs(primary: PathBuf, secondary: Option<PathBuf>) -> Vec<PathBuf> {
    let mut v = vec![primary];
    if let Some(s) = secondary {
        v.push(s);
    }
    v
}
