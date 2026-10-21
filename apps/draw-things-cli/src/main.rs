//! `draw-things-cli` — local FLUX.2 inference from the command line.

mod terminal;

use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use clap::{Args, CommandFactory, Parser, Subcommand, ValueEnum};
use image::RgbImage;
use tc_config::GenerationConfiguration;
use tc_core::image_io;
use tc_core::pipeline::{Cancellation, EngineOptions, Precision, Progress};
use tc_service::{zoo, Event, GenerationInput, ModelStore, Worker};

const COMMAND: &str = "draw-things-cli";

#[derive(Parser)]
#[command(
    name = COMMAND,
    version,
    about = "Local inference CLI for FLUX.2 models.",
    after_help = "COMMON COMMANDS:\n  draw-things-cli generate --model flux_2_klein_9b --prompt \"a red cube on a table\"\n  draw-things-cli models list --downloaded-only\n  draw-things-cli models ensure --model flux_2_klein_9b\n  draw-things-cli completion zsh\n\nENVIRONMENT:\n  DRAWTHINGS_MODELS_DIR   Default models directory when --models-dir is not provided.\n  HF_TOKEN                Hugging Face token for gated model downloads."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run inference and save generated image(s).
    Generate(Box<GenerateArgs>),
    /// Model utilities.
    #[command(subcommand)]
    Models(ModelsCommand),
    /// Authentication helpers (Draw Things cloud compute is not available in this implementation).
    Auth {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, hide = true)]
        rest: Vec<String>,
    },
    /// Training utilities (not available in this implementation).
    Train {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, hide = true)]
        rest: Vec<String>,
    },
    /// Generate shell completion scripts.
    Completion {
        /// Shell to generate completions for.
        shell: clap_complete::Shell,
        /// Write the completion script to a file instead of stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

#[derive(Args, Clone)]
struct ModelsDir {
    /// Models directory. Resolution order: --models-dir, DRAWTHINGS_MODELS_DIR,
    /// <binary-dir>/Models (if it exists), ~/Documents/Models.
    #[arg(long)]
    models_dir: Option<String>,
}

#[derive(Subcommand)]
enum ModelsCommand {
    /// List known models.
    List {
        #[command(flatten)]
        dir: ModelsDir,
        /// Show downloaded models only.
        #[arg(long)]
        downloaded_only: bool,
        /// Disable network access (no effect: the catalog is built in).
        #[arg(long)]
        offline: bool,
    },
    /// Ensure model files exist locally (download from Hugging Face if missing).
    Ensure {
        #[command(flatten)]
        dir: ModelsDir,
        /// Model id, name, alias or Hugging Face repo.
        #[arg(short, long)]
        model: String,
        /// Include model dependencies (all components are always downloaded).
        #[arg(long = "dependencies", overrides_with = "no_dependencies")]
        _dependencies: bool,
        #[arg(long = "no-dependencies", hide = true)]
        no_dependencies: bool,
    },
    /// Import a fine-tuned transformer (.safetensors, BFL or diffusers naming) as a new model
    /// that reuses the text encoder, VAE and tokenizer of a base model.
    Import {
        #[command(flatten)]
        dir: ModelsDir,
        /// Local transformer checkpoint (.safetensors).
        artifact: PathBuf,
        /// Display name for the imported model.
        #[arg(long)]
        name: Option<String>,
        /// Base model providing the other components.
        #[arg(long, default_value = "flux_2_klein_9b")]
        base: String,
        /// Inspect without writing files.
        #[arg(long)]
        dry_run: bool,
        /// Replace an existing import with the same id.
        #[arg(long)]
        force: bool,
    },
    /// Describe a Draw Things checkpoint (.ckpt): component, encodings, mapping coverage. With
    /// --lora, check that LoRAs (.ckpt or .safetensors) fully map onto the model.
    Inspect {
        #[command(flatten)]
        dir: ModelsDir,
        /// Checkpoint file (path or name in the models directory).
        file: String,
        /// LoRA files to check against this transformer checkpoint.
        #[arg(long)]
        lora: Vec<String>,
    },
    /// Check the GPU against the CPU for a model loaded as `generate` loads it (8-bit palettized
    /// weights, f16/bf16): single matrix products, the first transformer blocks in plain and KV mode
    /// with LoRAs, the first Qwen3 layers and a VAE round trip. Prints the relative error per stage.
    CheckGpu {
        #[command(flatten)]
        dir: ModelsDir,
        /// Model (e.g. flux_2_klein_9b_kv_q8p.ckpt).
        model: String,
        /// LoRA FILE[:WEIGHT] (repeatable).
        #[arg(long)]
        lora: Vec<String>,
        /// Transformer blocks to check: double and single (default 1 1; each double block costs
        /// about 2 GB of CPU memory in f32).
        #[arg(long, num_args = 2, default_values_t = [1, 1])]
        blocks: Vec<usize>,
        /// Transformer / text encoder precision on the GPU (default bf16; palettized weights run in f16).
        #[arg(long)]
        precision: Option<String>,
        /// Skip the text encoder check.
        #[arg(long)]
        no_text_encoder: bool,
        /// Skip the VAE check.
        #[arg(long)]
        no_vae: bool,
        /// Run the "device" side on the CPU too (tests the check itself).
        #[arg(long, hide = true)]
        cpu: bool,
    },
    /// Compare a Draw Things checkpoint tensor by tensor with the official release of the same
    /// component (diffusers snapshot directory, component directory or safetensors file). For a
    /// Draw Things LoRA, compare its weight updates with the original LoRA file.
    Compare {
        #[command(flatten)]
        dir: ModelsDir,
        /// Draw Things checkpoint (transformer, Qwen3 text encoder or VAE).
        ckpt: String,
        /// Reference: e.g. ~/Documents/Models/FLUX.2-klein-9b-kv (or its transformer/ directory).
        reference: PathBuf,
        /// Only tensors whose canonical name contains this.
        #[arg(long)]
        filter: Option<String>,
        /// Print every tensor (default: summary per tensor kind).
        #[arg(long)]
        all: bool,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum TerminalProtocol {
    Auto,
    Iterm2,
    Kitty,
}

#[derive(Args)]
struct GenerateArgs {
    #[command(flatten)]
    dir: ModelsDir,
    /// Model file, model name, alias, or Hugging Face repo/URL.
    #[arg(short, long)]
    model: Option<String>,
    /// Prompt text.
    #[arg(short, long)]
    prompt: Option<String>,
    /// Read prompt text from a file, or `-` for stdin.
    #[arg(long, conflicts_with = "prompt")]
    prompt_file: Option<String>,
    /// Negative prompt text (used with classifier-free guidance on non-distilled models).
    #[arg(long)]
    negative_prompt: Option<String>,
    /// Read negative prompt text from a file, or `-` for stdin.
    #[arg(long, conflicts_with = "negative_prompt")]
    negative_prompt_file: Option<String>,
    /// Sampling steps.
    #[arg(long)]
    steps: Option<u32>,
    /// CFG guidance scale.
    #[arg(long)]
    cfg: Option<f32>,
    /// Output width in pixels (multiple of 64). Defaults to the first --image when given.
    #[arg(long)]
    width: Option<u32>,
    /// Output height in pixels (multiple of 64). Defaults to the first --image when given.
    #[arg(long)]
    height: Option<u32>,
    /// Number of frames for video-capable models (not supported by FLUX.2).
    #[arg(long)]
    frames: Option<u32>,
    /// Denoising strength for img2img (0...1).
    #[arg(long)]
    strength: Option<f32>,
    /// Random seed.
    #[arg(short, long)]
    seed: Option<u32>,
    /// Inline JSON override (camelCase keys, as produced by Draw Things; snake_case accepted).
    #[arg(long)]
    config_json: Option<String>,
    /// Path to a JSON override file.
    #[arg(long)]
    config_file: Option<PathBuf>,
    /// Input image path. Repeat to add reference images. The first image is the canvas, the
    /// rest are reference ("moodboard") images.
    #[arg(long)]
    image: Vec<PathBuf>,
    /// Center-crop every --image to the output size, keeping the first one as the canvas.
    #[arg(long)]
    crop: bool,
    #[arg(long = "init-image", hide = true)]
    init_image: Option<PathBuf>,
    #[arg(long = "input-image", hide = true)]
    input_image: Option<PathBuf>,
    /// Audio input (not supported by FLUX.2).
    #[arg(long)]
    audio: Option<PathBuf>,
    /// LoRA to apply, as FILE or FILE:WEIGHT (repeatable). Files are looked up in the models directory.
    #[arg(long)]
    lora: Vec<String>,
    /// Output path (.png). If omitted, the result is shown inline in supported terminals.
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Render the first generated image inline in supported terminals.
    #[arg(long)]
    terminal_image: bool,
    /// Terminal inline image protocol.
    #[arg(long, value_enum, default_value = "auto")]
    terminal_image_protocol: TerminalProtocol,
    /// Auto-download missing model files.
    #[arg(long, default_value_t = true, overrides_with = "no_download_missing")]
    download_missing: bool,
    #[arg(long, hide = true)]
    no_download_missing: bool,
    /// Disable live progress output.
    #[arg(long)]
    disable_preview: bool,
    /// Never use the network.
    #[arg(long)]
    offline: bool,
    /// Generate locally (default).
    #[arg(long, conflicts_with_all = ["remote", "cloud_compute"])]
    local: bool,
    /// Generate on a remote gRPC server instead of locally.
    #[arg(long)]
    remote: bool,
    /// Remote server host or URL.
    #[arg(long = "remote-url")]
    remote_url: Option<String>,
    /// Remote server port.
    #[arg(long = "remote-port", default_value_t = 7859)]
    remote_port: u16,
    /// Use TLS for remote generation (default on; --no-remote-tls to disable).
    #[arg(long = "remote-tls", default_value_t = true, overrides_with = "no_remote_tls")]
    remote_tls: bool,
    #[arg(long = "no-remote-tls")]
    no_remote_tls: bool,
    /// Shared secret for password-protected remote servers.
    #[arg(long = "remote-shared-secret")]
    remote_shared_secret: Option<String>,
    /// Draw Things cloud compute (not available in this implementation).
    #[arg(long = "cloud-compute")]
    cloud_compute: bool,
    /// Print an estimate and exit (not available).
    #[arg(long)]
    estimate: bool,
    /// Transformer precision on GPU: f16, bf16 or f32.
    #[arg(long)]
    precision: Option<String>,
    /// Text encoder precision on GPU: f16, bf16 or f32.
    #[arg(long)]
    text_encoder_precision: Option<String>,
    /// Run on CPU even if a GPU is available.
    #[arg(long)]
    cpu: bool,
    /// Store transformer / text encoder weights 8-bit palettized (about half the memory).
    #[arg(long)]
    quantize: bool,
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let cli = Cli::parse();
    let r = match cli.command {
        Command::Generate(a) => generate(*a),
        Command::Models(m) => models(m),
        Command::Auth { .. } => Err(anyhow::anyhow!("Draw Things cloud authentication is not available in this implementation")),
        Command::Train { .. } => Err(anyhow::anyhow!("LoRA training is not available in this implementation")),
        Command::Completion { shell, output } => completion(shell, output),
    };
    if let Err(e) = r {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn completion(shell: clap_complete::Shell, output: Option<PathBuf>) -> Result<()> {
    let mut cmd = Cli::command();
    match output {
        Some(p) => {
            let mut f = std::fs::File::create(&p)?;
            clap_complete::generate(shell, &mut cmd, COMMAND, &mut f);
        }
        None => clap_complete::generate(shell, &mut cmd, COMMAND, &mut std::io::stdout()),
    }
    Ok(())
}

fn store_for(dir: &ModelsDir) -> Result<ModelStore> {
    let d = zoo::default_models_dir(dir.models_dir.as_deref())?;
    Ok(ModelStore::new(vec![d]))
}

fn models(cmd: ModelsCommand) -> Result<()> {
    match cmd {
        ModelsCommand::List { dir, downloaded_only, .. } => {
            let store = store_for(&dir)?;
            for m in &store.models {
                let local = store.snapshot_dir(&m.repo);
                if downloaded_only && local.is_none() {
                    continue;
                }
                let status = match &local {
                    Some(p) => format!("downloaded ({})", p.display()),
                    None => "not downloaded".into(),
                };
                println!("{:<24} {:<26} {:<42} {}", m.id, m.name, m.repo, status);
            }
            Ok(())
        }
        ModelsCommand::Ensure { dir, model, .. } => {
            let store = store_for(&dir)?;
            ensure_model(&store, &model)?;
            Ok(())
        }
        ModelsCommand::Import { dir, artifact, name, base, dry_run, force } => import_model(&store_for(&dir)?, &artifact, name, &base, dry_run, force),
        ModelsCommand::Inspect { dir, file, lora } => {
            let store = store_for(&dir)?;
            let path = store.local_file(&file).with_context(|| format!("{file} not found"))?;
            print!("{}", zoo::inspect_ckpt(&store, &path, &lora)?);
            Ok(())
        }
        ModelsCommand::CheckGpu { dir, model, lora, blocks, precision, no_text_encoder, no_vae, cpu } => {
            let store = store_for(&dir)?;
            let resolved = store.resolve(&model)?;
            let device = if cpu { nnc::Device::Cpu } else { nnc::default_device() };
            if !device.is_gpu() && !cpu {
                bail!("no GPU found (nnc sees {} GPUs)", nnc::gpu_count());
            }
            let mut p = Precision::default_for(if cpu { nnc::Device::Cpu } else { nnc::Device::Gpu(0) });
            if let Some(pr) = precision {
                p.transformer = parse_dtype(&pr)?;
                p.text_encoder = p.transformer;
            }
            let mut loras = vec![];
            for l in &lora {
                let (f, w) = match l.rsplit_once(':') {
                    Some((f, w)) if w.parse::<f32>().is_ok() => (f.to_string(), w.parse().unwrap()),
                    _ => (l.clone(), 1.0),
                };
                let path = store.resolve_lora(&f).with_context(|| format!("LoRA {f:?} not found"))?;
                loras.push(tc_core::pipeline::LoraSpec { path, weight: w });
            }
            let o = tc_core::gpucheck::CheckOptions {
                device,
                precision: p,
                double_blocks: blocks[0],
                single_blocks: blocks[1],
                text_encoder: !no_text_encoder,
                vae: !no_vae,
            };
            tc_core::gpucheck::run(&resolved.files, &loras, &o, &mut |l| println!("{l}"))?;
            println!("(ok: within the expected rounding of f16/bf16 against f32; FAIL: the device computes something else)");
            Ok(())
        }
        ModelsCommand::Compare { dir, ckpt, reference, filter, all } => {
            let store = store_for(&dir)?;
            let path = store.local_file(&ckpt).with_context(|| format!("{ckpt} not found"))?;
            let lora_file = tc_core::ckpt::CkptFile::open(&path).ok().and_then(|f| tc_core::compare::ckpt_component(&f)) == Some(tc_core::compare::Component::Lora);
            if lora_file {
                let reference = store.resolve_lora(&reference.to_string_lossy()).unwrap_or(reference);
                let transformer = store
                    .find_ckpts(&store.sibling_dirs(&path), "flux_2_klein")
                    .into_iter()
                    .find(|p| tc_core::ckpt::CkptFile::open(p).ok().and_then(|f| tc_core::compare::ckpt_component(&f)) == Some(tc_core::compare::Component::Transformer))
                    .context("no FLUX.2 transformer checkpoint next to the LoRA to take parameter shapes from")?;
                let rows = tc_core::compare::compare_lora(&path, &reference, &transformer)?;
                let mut worst = 0f64;
                let mut scales = vec![];
                for (p, rel, scale) in &rows {
                    if all || !rel.is_finite() {
                        println!("{p:<50} rel {rel:.5} scale {scale:.4}");
                    }
                    if rel.is_finite() {
                        worst = worst.max(*rel);
                        scales.push(*scale);
                    }
                }
                scales.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = scales.get(scales.len() / 2).copied().unwrap_or(f64::NAN);
                println!(
                    "{} parameters compared, worst relative difference of the LoRA updates {worst:.5}, best-fit scale median {median:.4} (min {:.4}, max {:.4})",
                    scales.len(),
                    scales.first().copied().unwrap_or(f64::NAN),
                    scales.last().copied().unwrap_or(f64::NAN)
                );
                return Ok(());
            }
            let mut kinds: std::collections::BTreeMap<String, (usize, f64, f32)> = Default::default();
            let rows = tc_core::compare::compare(&path, &reference, filter.as_deref(), &mut |r| {
                if all {
                    println!("{:<60} {:>16} rel {:.5} max {:.5}", r.name, format!("{:?}", r.shape), r.rel, r.max_abs);
                }
            })?;
            for r in &rows {
                let kind = r.name.split('.').filter(|p| p.parse::<usize>().is_err()).collect::<Vec<_>>().join(".");
                let e = kinds.entry(kind).or_insert((0, 0.0, 0.0));
                e.0 += 1;
                e.1 = e.1.max(r.rel);
                e.2 = e.2.max(r.max_abs);
            }
            println!("{:<60} {:>6} {:>12} {:>10}", "tensor kind", "count", "worst rel", "max abs");
            for (k, (n, rel, mx)) in &kinds {
                println!("{k:<60} {n:>6} {rel:>12.5} {mx:>10.5}");
            }
            let worst = rows.iter().map(|r| r.rel).fold(0.0, f64::max);
            println!("{} tensors compared, worst relative error {worst:.5}", rows.len());
            Ok(())
        }
    }
}

fn ensure_model(store: &ModelStore, model: &str) -> Result<PathBuf> {
    if let Ok(r) = store.resolve(model) {
        eprintln!("{} is available at {}", model, r.dir.display());
        return Ok(r.dir);
    }
    if model.ends_with(".ckpt") {
        // Draw Things checkpoint: download it and its text encoder / VAE from the public CDN.
        let names: Vec<String> = std::iter::once(model.to_string()).chain(zoo::dt_dependencies(model)).collect();
        let mut last = String::new();
        for name in &names {
            if store.local_file(name).is_some() {
                continue;
            }
            eprintln!("downloading {name} from {} into {}", zoo::DT_CDN, store.primary_dir().display());
            zoo::download_dt_ckpt(name, store.primary_dir(), &mut |file, done, total| {
                let line = format!("  {file}: {:.0}%", if total > 0 { done as f64 * 100.0 / total as f64 } else { 0.0 });
                if line != last {
                    eprint!("\r{line}        ");
                    last = line.clone();
                }
            })?;
            eprintln!();
        }
        store.ensure_tokenizer()?;
        let r = store.resolve(model)?;
        eprintln!("{model} is available at {}", r.dir.display());
        return Ok(r.dir);
    }
    let repo = match store.find_spec(model) {
        Some(s) => s.repo.clone(),
        None if model.contains('/') => model.trim_start_matches("hf://").trim_start_matches("https://huggingface.co/").to_string(),
        None => bail!("unknown model {model:?}; see `{COMMAND} models list`"),
    };
    eprintln!("downloading {repo} into {}", store.primary_dir().display());
    let mut last = String::new();
    let dir = zoo::download_snapshot(&repo, store.primary_dir(), &mut |file, done, total| {
        let line = format!("  {file}: {:.0}%", if total > 0 { done as f64 * 100.0 / total as f64 } else { 0.0 });
        if line != last {
            eprint!("\r{line}        ");
            last = line;
        }
    })?;
    eprintln!("\ndone: {}", dir.display());
    Ok(dir)
}

fn import_model(store: &ModelStore, artifact: &Path, name: Option<String>, base: &str, dry_run: bool, force: bool) -> Result<()> {
    let base = store.resolve(base).with_context(|| format!("base model {base:?} must be downloaded first"))?;
    let (_, config) = tc_core::models::flux2::open_source(artifact).context("not a FLUX.2 transformer checkpoint")?;
    let stem = artifact.file_stem().and_then(|s| s.to_str()).unwrap_or("imported").to_string();
    let id: String = stem.chars().map(|c| if c.is_alphanumeric() { c.to_ascii_lowercase() } else { '_' }).collect();
    let display = name.unwrap_or_else(|| stem.clone());
    println!("model id: {id}\nname: {display}\nlayers: {} double / {} single, hidden {}", config.num_layers, config.num_single_layers, config.inner_dim());
    if dry_run {
        return Ok(());
    }
    let target = store.primary_dir().join(&id);
    if target.exists() {
        if !force {
            bail!("{} exists (use --force to replace)", target.display());
        }
        std::fs::remove_dir_all(&target)?;
    }
    std::fs::create_dir_all(target.join("transformer"))?;
    let dst = target.join("transformer").join("diffusion_pytorch_model.safetensors");
    link_or_copy(artifact, &dst)?;
    for c in ["text_encoder", "tokenizer", "vae", "scheduler"] {
        let src = base.dir.join(c);
        if src.exists() {
            link_or_copy(&src, &target.join(c))?;
        }
    }
    let index = serde_json::json!({"_class_name": "Flux2KleinPipeline", "is_distilled": base.files.is_distilled, "name": display});
    std::fs::write(target.join("model_index.json"), serde_json::to_vec_pretty(&index)?)?;
    println!("imported to {}", target.display());
    Ok(())
}

fn link_or_copy(src: &Path, dst: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        let abs = src.canonicalize()?;
        if std::os::unix::fs::symlink(&abs, dst).is_ok() {
            return Ok(());
        }
    }
    if src.is_dir() {
        std::fs::create_dir_all(dst)?;
        for e in std::fs::read_dir(src)? {
            let e = e?;
            link_or_copy(&e.path(), &dst.join(e.file_name()))?;
        }
    } else {
        std::fs::copy(src, dst)?;
    }
    Ok(())
}

fn read_text(path: &str) -> Result<String> {
    if path == "-" {
        let mut s = String::new();
        std::io::stdin().read_to_string(&mut s)?;
        Ok(s)
    } else {
        std::fs::read_to_string(path).with_context(|| format!("reading {path}"))
    }
}

fn round64(v: f64) -> u32 {
    ((v / 64.0).round().max(1.0) as u32) * 64
}

fn parse_dtype(s: &str) -> Result<nnc::DataType> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "f16" | "fp16" | "float16" | "half" => nnc::DataType::F16,
        "bf16" | "bfloat16" => nnc::DataType::BF16,
        "f32" | "fp32" | "float32" | "float" => nnc::DataType::F32,
        other => bail!("unknown precision {other:?} (use f16, bf16 or f32)"),
    })
}

/// Resolved inputs for a generation (shared by local and remote paths).
struct Prepared {
    config: GenerationConfiguration,
    prompt: String,
    negative_prompt: String,
    canvas: Option<RgbImage>,
    references: Vec<RgbImage>,
}

fn prepare(a: &GenerateArgs, store: &ModelStore) -> Result<Prepared> {
    if a.cloud_compute {
        bail!("--cloud-compute is not available in this implementation; use --remote with a gRPCServerCLI");
    }
    if a.estimate {
        bail!("--estimate is not available in this implementation");
    }
    if a.frames.is_some() || a.audio.is_some() {
        bail!("video/audio options are not supported by FLUX.2 image models");
    }
    let Some(model) = a.model.clone() else {
        let mut msg = String::from("--model is required. Known models:\n");
        for m in &store.models {
            msg.push_str(&format!("  {:<24} {}\n", m.id, m.name));
        }
        bail!(msg);
    };
    let prompt = match (&a.prompt, &a.prompt_file) {
        (Some(p), _) => p.clone(),
        (None, Some(f)) => read_text(f)?,
        (None, None) => bail!("--prompt or --prompt-file is required"),
    };
    let negative_prompt = match (&a.negative_prompt, &a.negative_prompt_file) {
        (Some(p), _) => p.clone(),
        (None, Some(f)) => read_text(f)?,
        _ => String::new(),
    };
    // Image aliases.
    let mut images = a.image.clone();
    for alias in [&a.init_image, &a.input_image].into_iter().flatten() {
        if images.is_empty() {
            images.push(alias.clone());
        } else if &images[0] != alias {
            bail!("--image / --init-image / --input-image disagree");
        }
    }
    let loaded: Vec<RgbImage> = images.iter().map(|p| image_io::load_rgb(p)).collect::<Result<_>>()?;

    // Configuration precedence: recommended < JSON overrides < flags.
    let spec = store.find_spec(&model).cloned();
    let mut config = tc_service::recommended_configuration(&model, spec.as_ref());
    for raw in [a.config_json.clone(), a.config_file.as_ref().map(|p| std::fs::read_to_string(p)).transpose()?].into_iter().flatten() {
        let json: serde_json::Value = serde_json::from_str(&raw).context("invalid configuration JSON")?;
        let unknown = tc_config::apply_json_overrides(&mut config, &json)?;
        if !unknown.is_empty() {
            eprintln!("warning: ignoring unknown configuration keys: {unknown:?}");
        }
    }
    config.model = Some(model);
    if let Some(s) = a.steps {
        config.steps = s;
    }
    if let Some(c) = a.cfg {
        config.guidance_scale = c;
    }
    if let Some(s) = a.seed {
        config.seed = s;
    } else if a.config_json.is_none() && a.config_file.is_none() {
        config.seed = rand_seed();
    }
    if let Some(s) = a.strength {
        if !(0.0..=1.0).contains(&s) {
            bail!("--strength must be within 0...1");
        }
        config.strength = s;
    }
    for l in &a.lora {
        let (file, weight) = match l.rsplit_once(':') {
            Some((f, w)) if w.parse::<f32>().is_ok() => (f.to_string(), w.parse().unwrap()),
            _ => (l.clone(), 1.0),
        };
        config.loras.push(tc_config::LoRA { file: Some(file), weight, mode: 0 });
    }
    // Size: explicit flags, else from the first image, else JSON / recommended.
    let (mut w, mut h) = (a.width, a.height);
    if let Some(first) = loaded.first() {
        let ar = first.width() as f64 / first.height() as f64;
        match (w, h) {
            (None, None) => {
                w = Some(round64(first.width() as f64));
                h = Some(round64(first.height() as f64));
            }
            (Some(ww), None) => h = Some(round64(ww as f64 / ar)),
            (None, Some(hh)) => w = Some(round64(hh as f64 * ar)),
            _ => {}
        }
    }
    if let Some(w) = w {
        if w % 64 != 0 {
            bail!("--width must be a multiple of 64");
        }
        config.start_width = (w / 64) as u16;
    }
    if let Some(h) = h {
        if h % 64 != 0 {
            bail!("--height must be a multiple of 64");
        }
        config.start_height = (h / 64) as u16;
    }
    let (ow, oh) = (config.width_px(), config.height_px());
    let mut canvas = None;
    let mut references = vec![];
    for (i, img) in loaded.into_iter().enumerate() {
        let same_aspect = ((img.width() as f64 / img.height() as f64) - (ow as f64 / oh as f64)).abs() < 0.02;
        if i == 0 && (same_aspect || a.crop) {
            if img.width() < ow || img.height() < oh {
                eprintln!("warning: canvas image is smaller than the output size and will be upscaled");
            }
            canvas = Some(image_io::resize_cover_crop(&img, ow, oh));
        } else if a.crop {
            references.push(image_io::resize_cover_crop(&img, ow, oh));
        } else {
            references.push(img);
        }
    }
    Ok(Prepared { config, prompt, negative_prompt, canvas, references })
}

fn rand_seed() -> u32 {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    (t.as_nanos() as u32) ^ std::process::id().rotate_left(16)
}

fn generate(a: GenerateArgs) -> Result<()> {
    let store = store_for(&a.dir)?;
    let prepared = prepare(&a, &store)?;
    let images = if a.remote {
        generate_remote(&a, &prepared)?
    } else {
        let model = prepared.config.model.clone().unwrap();
        if store.resolve(&model).is_err() {
            if a.download_missing && !a.no_download_missing && !a.offline {
                ensure_model(&store, &model)?;
            } else {
                store.resolve(&model)?;
            }
        }
        generate_local(&a, &store, prepared)?
    };
    let pngs: Vec<Vec<u8>> = images.iter().map(image_io::encode_png).collect::<Result<_>>()?;
    match &a.output {
        Some(out) => {
            for (i, png) in pngs.iter().enumerate() {
                let path = if pngs.len() == 1 { out.clone() } else { numbered(out, i + 1) };
                std::fs::write(&path, png).with_context(|| format!("writing {}", path.display()))?;
                println!("{}", path.display());
            }
            if a.terminal_image {
                terminal::show(&pngs[0], a.terminal_image_protocol)?;
            }
        }
        None => {
            if std::io::stdout().is_terminal() && terminal::supported(a.terminal_image_protocol) {
                terminal::show(&pngs[0], a.terminal_image_protocol)?;
            } else {
                let path = PathBuf::from("output.png");
                std::fs::write(&path, &pngs[0])?;
                println!("{}", path.display());
            }
        }
    }
    Ok(())
}

fn numbered(p: &Path, i: usize) -> PathBuf {
    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
    let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("png");
    p.with_file_name(format!("{stem}-{i}.{ext}"))
}

fn generate_local(a: &GenerateArgs, store: &ModelStore, prepared: Prepared) -> Result<Vec<RgbImage>> {
    let device = if a.cpu || nnc::gpu_count() == 0 { nnc::Device::Cpu } else { nnc::Device::Gpu(0) };
    let mut precision = Precision::default_for(device);
    if let Some(p) = &a.precision {
        precision.transformer = parse_dtype(p)?;
    }
    if let Some(p) = &a.text_encoder_precision {
        precision.text_encoder = parse_dtype(p)?;
    }
    let worker = Arc::new(Worker::spawn(store.clone(), EngineOptions { device, precision, keep_text_encoder: false, prompt_cache_size: 0, quantize: a.quantize, keep_palettized: true }));
    let input = GenerationInput {
        config: prepared.config,
        prompt: prepared.prompt,
        negative_prompt: prepared.negative_prompt,
        image: prepared.canvas,
        references: prepared.references,
    };
    let events = worker.submit(input, Cancellation::default());
    let quiet = a.disable_preview;
    for ev in events {
        match ev {
            Event::Progress(p) => {
                if !quiet {
                    match p {
                        Progress::TextEncoded => eprintln!("text encoded"),
                        Progress::ImageEncoded => eprintln!("images encoded"),
                        Progress::Sampling { step, total } => eprint!("\rsampling {step}/{total}"),
                        Progress::ImageDecoded => eprintln!("\nimage decoded"),
                    }
                    let _ = std::io::stderr().flush();
                }
            }
            Event::Finished(r) => {
                let tensors = r?;
                return Ok(tensors.iter().flat_map(image_io::from_tensor).collect());
            }
        }
    }
    bail!("generation worker stopped unexpectedly")
}

fn generate_remote(a: &GenerateArgs, p: &Prepared) -> Result<Vec<RgbImage>> {
    let url = a.remote_url.clone().context("--remote requires --remote-url")?;
    let (host, port) = match url.trim_start_matches("https://").trim_start_matches("http://").rsplit_once(':') {
        Some((h, port)) if port.parse::<u16>().is_ok() => (h.to_string(), port.parse().unwrap()),
        _ => (url.trim_start_matches("https://").trim_start_matches("http://").to_string(), a.remote_port),
    };
    let tls = a.remote_tls && !a.no_remote_tls;
    let to_wire = |img: &RgbImage| {
        let t = image_io::to_tensor(img);
        tc_wire::encode(&tc_wire::WireTensor::from_f32(&t.to_f32_vec(), t.shape(), tc_wire::WireType::F16), tc_wire::Compression::FpzipOrZip)
    };
    let image = p.canvas.as_ref().map(to_wire);
    let refs: Vec<Vec<u8>> = p.references.iter().map(to_wire).collect();
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    let quiet = a.disable_preview;
    let result = rt.block_on(async {
        let mut client = tc_grpc::client::RemoteClient::connect(&host, port, tls, a.remote_shared_secret.clone()).await?;
        client.echo(COMMAND).await?;
        client
            .generate(p.config.to_flatbuffer(), &p.prompt, &p.negative_prompt, image, refs, |s| {
                if !quiet {
                    if let Some(tc_grpc::proto::image_generation_signpost_proto::Signpost::Sampling(x)) = &s.signpost {
                        eprint!("\rsampling {}", x.step);
                    }
                }
            })
            .await
    })?;
    if !quiet {
        eprintln!();
    }
    result
        .images
        .iter()
        .map(|t| tc_grpc::server::tensor_to_rgb(t))
        .collect::<Result<Vec<_>>>()
}
