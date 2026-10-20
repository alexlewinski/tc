//! Model registry and on-disk resolution.
//!
//! Models are Hugging Face snapshots in diffusers layout. They are looked up, in order, in:
//!
//! 1. `<models-dir>/<repo-name>/` (e.g. `Models/FLUX.2-klein-9B/`)
//! 2. `<models-dir>/<owner>--<repo-name>/`
//! 3. the Hugging Face cache (`$HF_HOME/hub` or `~/.cache/huggingface/hub`)
//!
//! A model reference may be a registry id (`flux_2_klein_9b`), an alias (including Draw Things
//! file names such as `flux_2_klein_9b_q8p.ckpt`), a display name (`FLUX.2 [klein] 9B`), a
//! Hugging Face repo (`black-forest-labs/FLUX.2-klein-9B`, `hf://...`, `https://huggingface.co/...`)
//! or a path to a snapshot directory.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use tc_core::pipeline::Flux2Files;

/// Built-in or user-registered model description.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelSpec {
    pub id: String,
    pub name: String,
    /// Hugging Face repository (`owner/name`).
    pub repo: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    /// Step-distilled (guidance-free) model.
    #[serde(default)]
    pub distilled: bool,
    #[serde(default = "d_steps")]
    pub default_steps: u32,
    #[serde(default = "d_cfg")]
    pub default_guidance: f32,
    #[serde(default = "d_size")]
    pub default_size: u32,
    #[serde(default)]
    pub note: String,
}

fn d_steps() -> u32 {
    4
}
fn d_cfg() -> f32 {
    1.0
}
fn d_size() -> u32 {
    1024
}

impl ModelSpec {
    pub fn dir_name(&self) -> &str {
        self.repo.rsplit('/').next().unwrap_or(&self.repo)
    }
}

fn spec(id: &str, name: &str, repo: &str, distilled: bool, steps: u32, cfg: f32, aliases: &[&str]) -> ModelSpec {
    ModelSpec {
        id: id.into(),
        name: name.into(),
        repo: repo.into(),
        aliases: aliases.iter().map(|s| s.to_string()).collect(),
        distilled,
        default_steps: steps,
        default_guidance: cfg,
        default_size: 1024,
        note: String::new(),
    }
}

/// Models known out of the box.
pub fn builtin_models() -> Vec<ModelSpec> {
    vec![
        spec(
            "flux_2_klein_9b",
            "FLUX.2 [klein] 9B",
            "black-forest-labs/FLUX.2-klein-9B",
            true,
            4,
            1.0,
            &["flux_2_klein_9b_q8p.ckpt", "flux_2_klein_9b_q6p.ckpt", "flux_2_klein_9b_i8x.ckpt", "flux_2_klein_9b_f16.ckpt", "flux2_klein_9b", "klein-9b"],
        ),
        spec(
            "flux_2_klein_base_9b",
            "FLUX.2 [klein] Base 9B",
            "black-forest-labs/FLUX.2-klein-base-9B",
            false,
            50,
            4.0,
            &["flux_2_klein_base_9b_q8p.ckpt", "flux_2_klein_base_9b_q6p.ckpt", "flux_2_klein_base_9b_i8x.ckpt", "klein-base-9b"],
        ),
        spec(
            "flux_2_klein_4b",
            "FLUX.2 [klein] 4B",
            "black-forest-labs/FLUX.2-klein-4B",
            true,
            4,
            1.0,
            &["flux_2_klein_4b_q8p.ckpt", "flux_2_klein_4b_q6p.ckpt", "flux_2_klein_4b_i8x.ckpt", "flux2_klein_4b", "klein-4b"],
        ),
        spec(
            "flux_2_klein_base_4b",
            "FLUX.2 [klein] Base 4B",
            "black-forest-labs/FLUX.2-klein-base-4B",
            false,
            50,
            4.0,
            &["flux_2_klein_base_4b_q8p.ckpt", "flux_2_klein_base_4b_q6p.ckpt", "flux_2_klein_base_4b_i8x.ckpt", "klein-base-4b"],
        ),
    ]
}

/// Name of the optional user registry file inside a models directory.
pub const CUSTOM_MODELS_FILE: &str = "tonic_clap_models.json";

fn normalize(s: &str) -> String {
    s.chars().filter(|c| c.is_alphanumeric()).collect::<String>().to_lowercase()
}

/// A resolved model.
#[derive(Clone, Debug)]
pub struct ResolvedModel {
    pub spec: Option<ModelSpec>,
    pub dir: PathBuf,
    pub files: Flux2Files,
}

/// Lookup of models and LoRAs on disk.
#[derive(Clone, Debug)]
pub struct ModelStore {
    pub dirs: Vec<PathBuf>,
    pub models: Vec<ModelSpec>,
}

impl ModelStore {
    pub fn new(dirs: Vec<PathBuf>) -> ModelStore {
        let mut models = builtin_models();
        for d in &dirs {
            let p = d.join(CUSTOM_MODELS_FILE);
            if let Ok(bytes) = std::fs::read(&p) {
                match serde_json::from_slice::<Vec<ModelSpec>>(&bytes) {
                    Ok(custom) => {
                        for c in custom {
                            models.retain(|m| m.id != c.id);
                            models.push(c);
                        }
                    }
                    Err(e) => log::warn!("ignoring {}: {e}", p.display()),
                }
            }
        }
        ModelStore { dirs, models }
    }

    pub fn primary_dir(&self) -> &Path {
        &self.dirs[0]
    }

    /// Find a registry entry by id / alias / display name / repo.
    pub fn find_spec(&self, reference: &str) -> Option<&ModelSpec> {
        let r = strip_hf(reference);
        let n = normalize(&r);
        self.models.iter().find(|m| {
            normalize(&m.id) == n
                || normalize(&m.name) == n
                || normalize(&m.repo) == n
                || normalize(m.dir_name()) == n
                || m.aliases.iter().any(|a| normalize(a) == n)
        })
    }

    /// Where a repo's snapshot lives, if present.
    pub fn snapshot_dir(&self, repo: &str) -> Option<PathBuf> {
        let name = repo.rsplit('/').next().unwrap_or(repo);
        for d in &self.dirs {
            for c in [d.join(name), d.join(repo.replace('/', "--"))] {
                if c.join("transformer").is_dir() {
                    return Some(c);
                }
            }
        }
        hf_cache_snapshot(repo)
    }

    /// Resolve a model reference to component files.
    pub fn resolve(&self, reference: &str) -> Result<ResolvedModel> {
        let as_path = PathBuf::from(reference);
        if as_path.join("transformer").is_dir() {
            let files = Flux2Files::from_diffusers_dir(&as_path)?;
            return Ok(ResolvedModel { spec: None, dir: as_path, files });
        }
        let spec = self.find_spec(reference).cloned();
        let repo = match &spec {
            Some(s) => s.repo.clone(),
            None => strip_hf(reference),
        };
        let dir = self
            .snapshot_dir(&repo)
            .or_else(|| self.dirs.iter().map(|d| d.join(reference)).find(|p| p.join("transformer").is_dir()))
            .with_context(|| {
                if self.dirs.iter().any(|d| tc_core::weights::is_sqlite_checkpoint(&d.join(reference))) {
                    return format!(
                        "{reference:?} in {:?} is a Draw Things checkpoint (SQLite .ckpt), which is not supported; \
                         download the diffusers release next to it with `draw-things-cli models ensure --model {reference}`",
                        self.dirs
                    );
                }
                format!(
                    "model {reference:?} not found in {:?}; download it with `draw-things-cli models ensure --model {reference}`",
                    self.dirs
                )
            })?;
        let mut files = Flux2Files::from_diffusers_dir(&dir)?;
        if let Some(s) = &spec {
            files.is_distilled = files.is_distilled || s.distilled;
        }
        Ok(ResolvedModel { spec, dir, files })
    }

    /// Resolve a LoRA file name to a path.
    pub fn resolve_lora(&self, file: &str) -> Option<PathBuf> {
        tc_core::pipeline::resolve_lora_path(file, &self.dirs)
    }

    /// Whether `file` (a model reference or a file in a models directory) exists locally.
    pub fn exists(&self, file: &str) -> bool {
        self.local_path(file).is_some()
    }

    /// Local path of a model directory or file.
    pub fn local_path(&self, file: &str) -> Option<PathBuf> {
        if let Ok(m) = self.resolve(file) {
            return Some(m.dir);
        }
        self.dirs.iter().map(|d| d.join(file)).find(|p| p.exists())
    }
}

fn strip_hf(r: &str) -> String {
    let r = r.trim();
    let r = r.strip_prefix("hf://").unwrap_or(r);
    let r = r.strip_prefix("https://huggingface.co/").unwrap_or(r);
    let r = r.strip_prefix("http://huggingface.co/").unwrap_or(r);
    let parts: Vec<&str> = r.split('/').collect();
    if parts.len() >= 2 && !parts[0].is_empty() {
        format!("{}/{}", parts[0], parts[1])
    } else {
        r.to_string()
    }
}

/// `$HF_HOME/hub/models--owner--name/snapshots/<rev>` (most recent).
pub fn hf_cache_snapshot(repo: &str) -> Option<PathBuf> {
    let hub = std::env::var_os("HF_HUB_CACHE")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HF_HOME").map(|h| PathBuf::from(h).join("hub")))
        .or_else(|| dirs::home_dir().map(|h| h.join(".cache/huggingface/hub")))?;
    let base = hub.join(format!("models--{}", repo.replace('/', "--"))).join("snapshots");
    let mut snaps: Vec<PathBuf> = std::fs::read_dir(&base).ok()?.filter_map(|e| e.ok().map(|e| e.path())).collect();
    snaps.sort_by_key(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok());
    snaps.into_iter().rev().find(|p| p.join("transformer").is_dir())
}

/// Default models directory: `DRAWTHINGS_MODELS_DIR`, else `<binary dir>/Models` if present,
/// else `~/Documents/Models` (created if missing).
pub fn default_models_dir(explicit: Option<&str>) -> Result<PathBuf> {
    if let Some(e) = explicit {
        return Ok(PathBuf::from(e));
    }
    if let Some(v) = std::env::var_os("DRAWTHINGS_MODELS_DIR") {
        return Ok(PathBuf::from(v));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(d) = exe.parent().map(|p| p.join("Models")) {
            if d.is_dir() {
                return Ok(d);
            }
        }
    }
    let home = dirs::home_dir().context("cannot determine home directory")?;
    let d = home.join("Documents").join("Models");
    std::fs::create_dir_all(&d).with_context(|| format!("creating {}", d.display()))?;
    Ok(d)
}

/// Download the diffusers components of `repo` from Hugging Face into `<dir>/<repo-name>`.
/// Uses `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) for gated repositories.
pub fn download_snapshot(repo: &str, models_dir: &Path, progress: &mut dyn FnMut(&str, u64, u64)) -> Result<PathBuf> {
    let name = repo.rsplit('/').next().unwrap_or(repo);
    let target = models_dir.join(name);
    std::fs::create_dir_all(&target)?;
    let token = std::env::var("HF_TOKEN").or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN")).ok();
    let endpoint = std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".into());
    let agent = ureq::AgentBuilder::new().try_proxy_from_env(true).build();
    let auth = |r: ureq::Request| match &token {
        Some(t) => r.set("Authorization", &format!("Bearer {t}")),
        None => r,
    };
    let url = format!("{endpoint}/api/models/{repo}/tree/main?recursive=true");
    let listing: serde_json::Value = auth(agent.get(&url))
        .call()
        .with_context(|| format!("listing {repo} (gated repos need HF_TOKEN)"))?
        .into_json()?;
    let wanted = |p: &str| {
        p == "model_index.json"
            || ["transformer/", "text_encoder/", "tokenizer/", "vae/", "scheduler/"].iter().any(|d| p.starts_with(d))
    };
    let files: Vec<(String, u64)> = listing
        .as_array()
        .context("unexpected listing")?
        .iter()
        .filter(|e| e["type"] == "file")
        .filter_map(|e| Some((e["path"].as_str()?.to_string(), e["size"].as_u64().unwrap_or(0))))
        .filter(|(p, _)| wanted(p))
        .collect();
    if files.is_empty() {
        bail!("{repo} has no diffusers components");
    }
    for (path, size) in files {
        let dest = target.join(&path);
        if dest.exists() && std::fs::metadata(&dest).map(|m| m.len() == size).unwrap_or(false) {
            continue;
        }
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let url = format!("{endpoint}/{repo}/resolve/main/{path}");
        let resp = auth(agent.get(&url)).call().with_context(|| format!("downloading {path}"))?;
        let tmp = dest.with_extension("part");
        let mut out = std::fs::File::create(&tmp)?;
        let mut reader = resp.into_reader();
        let mut buf = vec![0u8; 1 << 20];
        let mut done = 0u64;
        loop {
            let n = std::io::Read::read(&mut reader, &mut buf)?;
            if n == 0 {
                break;
            }
            std::io::Write::write_all(&mut out, &buf[..n])?;
            done += n as u64;
            progress(&path, done, size);
        }
        drop(out);
        std::fs::rename(&tmp, &dest)?;
    }
    Ok(target)
}
