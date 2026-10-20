//! Model registry and on-disk resolution.
//!
//! A model is either a Draw Things checkpoint (`<models-dir>/flux_2_klein_9b_kv_q8p.ckpt`, with
//! the matching `qwen_3_*.ckpt` text encoder and `flux_2_vae*.ckpt` next to it), or a Hugging Face
//! snapshot in diffusers layout, looked up in:
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
    /// FLUX.2 [klein] KV variant (see `Flux2Files::kv`).
    #[serde(default)]
    pub kv: bool,
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
        kv: false,
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
        ModelSpec {
            kv: true,
            ..spec(
                "flux_2_klein_9b_kv",
                "FLUX.2 [klein] 9B KV",
                "black-forest-labs/FLUX.2-klein-9b-kv",
                true,
                4,
                1.0,
                &["flux_2_klein_9b_kv_q8p.ckpt", "flux_2_klein_9b_kv_q6p.ckpt", "flux_2_klein_9b_kv_i8x.ckpt", "flux_2_klein_9b_kv_f16.ckpt", "klein-9b-kv"],
            )
        },
        spec(
            "flux_2_klein_base_9b",
            "FLUX.2 [klein] Base 9B",
            "black-forest-labs/FLUX.2-klein-base-9B",
            false,
            50,
            4.0,
            &["flux_2_klein_base_9b_q8p.ckpt", "flux_2_klein_base_9b_q6p.ckpt", "flux_2_klein_base_9b_i8x.ckpt", "flux_2_klein_base_9b_f16.ckpt", "klein-base-9b"],
        ),
        spec(
            "flux_2_klein_4b",
            "FLUX.2 [klein] 4B",
            "black-forest-labs/FLUX.2-klein-4B",
            true,
            4,
            1.0,
            &["flux_2_klein_4b_q8p.ckpt", "flux_2_klein_4b_q6p.ckpt", "flux_2_klein_4b_i8x.ckpt", "flux_2_klein_4b_f16.ckpt", "flux2_klein_4b", "klein-4b"],
        ),
        spec(
            "flux_2_klein_base_4b",
            "FLUX.2 [klein] Base 4B",
            "black-forest-labs/FLUX.2-klein-base-4B",
            false,
            50,
            4.0,
            &["flux_2_klein_base_4b_q8p.ckpt", "flux_2_klein_base_4b_q6p.ckpt", "flux_2_klein_base_4b_i8x.ckpt", "flux_2_klein_base_4b_f16.ckpt", "klein-base-4b"],
        ),
    ]
}

/// Where a downloaded tokenizer for Draw Things checkpoints is kept, inside a models directory.
pub const TOKENIZER_DIR: &str = "flux_2_tokenizer";
const TOKENIZER_REPO: &str = "black-forest-labs/FLUX.2-klein-4B";
/// Public download location of Draw Things checkpoints (`<CDN>/<file name>`).
pub const DT_CDN: &str = "https://static.libnnc.org";
/// SHA-256 of every public Draw Things checkpoint, keyed by file name.
pub const DT_MANIFEST: &str = "https://models.drawthings.ai/models_sha256.json";

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
        // A Draw Things checkpoint named (or given as a path) directly.
        if let Some(path) = self.local_file(reference).filter(|p| tc_core::ckpt::is_ckpt(p)) {
            return self.resolve_ckpt(&path, spec.as_ref());
        }
        let repo = match &spec {
            Some(s) => s.repo.clone(),
            None => strip_hf(reference),
        };
        let dir = self.snapshot_dir(&repo).or_else(|| self.dirs.iter().map(|d| d.join(reference)).find(|p| p.join("transformer").is_dir()));
        let Some(dir) = dir else {
            // A registry id whose Draw Things checkpoint is present.
            if let Some(s) = &spec {
                if let Some(path) = s.aliases.iter().filter(|a| a.ends_with(".ckpt")).find_map(|a| self.local_file(a).filter(|p| tc_core::ckpt::is_ckpt(p))) {
                    return self.resolve_ckpt(&path, Some(s));
                }
            }
            bail!(
                "model {reference:?} not found in {:?}; download it with `draw-things-cli models ensure --model {reference}`",
                self.dirs
            );
        };
        let mut files = Flux2Files::from_diffusers_dir(&dir)?;
        if let Some(s) = &spec {
            files.is_distilled = files.is_distilled || s.distilled;
            files.kv = files.kv || s.kv;
        }
        Ok(ResolvedModel { spec, dir, files })
    }

    /// A file given as a path, or by name in one of the models directories.
    pub fn local_file(&self, name: &str) -> Option<PathBuf> {
        let p = PathBuf::from(name);
        if p.is_file() {
            return Some(p);
        }
        self.dirs.iter().map(|d| d.join(name)).find(|p| p.is_file())
    }

    /// Directories searched for the components that go with a checkpoint: its own directory,
    /// then the models directories.
    pub fn sibling_dirs(&self, ckpt: &Path) -> Vec<PathBuf> {
        let mut v: Vec<PathBuf> = ckpt.parent().map(|p| p.to_path_buf()).into_iter().collect();
        for d in &self.dirs {
            if !v.contains(d) {
                v.push(d.clone());
            }
        }
        v
    }

    /// Draw Things checkpoints whose name starts with `prefix`, best encodings first.
    pub fn find_ckpts(&self, dirs: &[PathBuf], prefix: &str) -> Vec<PathBuf> {
        let mut found = vec![];
        for d in dirs {
            let Ok(rd) = std::fs::read_dir(d) else { continue };
            let mut v: Vec<PathBuf> = rd
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| {
                    let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    n.starts_with(prefix) && n.ends_with(".ckpt") && p.is_file()
                })
                .collect();
            let rank = |p: &PathBuf| {
                let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                ["_q8p", "_f16", "_q6p"].iter().position(|t| n.contains(t)).unwrap_or(9)
            };
            v.sort_by_key(|p| (rank(p), p.clone()));
            found.extend(v);
        }
        found
    }

    /// Components for a Draw Things transformer checkpoint.
    pub fn resolve_ckpt(&self, transformer: &Path, spec: Option<&ModelSpec>) -> Result<ResolvedModel> {
        use tc_core::ckpt::CkptFile;
        let file = CkptFile::open(transformer)?;
        if tc_core::compare::ckpt_component(&file) != Some(tc_core::compare::Component::Transformer) {
            bail!("{} is not a FLUX.2 transformer checkpoint (it may be a LoRA, text encoder or VAE)", transformer.display());
        }
        let (_, config) = tc_core::models::flux2::Flux2Source::from_ckpt(file)?;
        let text_hidden = config.joint_attention_dim / tc_core::pipeline::TEXT_ENCODER_LAYERS.len();
        let dirs = self.sibling_dirs(transformer);
        let qwen = if text_hidden == 2560 { "qwen_3_4b" } else { "qwen_3_8b" };
        let text_encoder = self
            .find_ckpts(&dirs, "qwen_3")
            .into_iter()
            .find(|p| {
                CkptFile::open(p)
                    .ok()
                    .and_then(|f| f.info("__text_model__[t-tok_embeddings-0-0]").map(|i| i.shape.get(1) == Some(&text_hidden)))
                    .unwrap_or(false)
            })
            .with_context(|| {
                format!(
                    "no text encoder for {} (expected {qwen}_q8p.ckpt, Qwen3 with hidden size {text_hidden}) in {dirs:?}; \
                     download it with `draw-things-cli models ensure --model {qwen}_q8p.ckpt`",
                    transformer.display()
                )
            })?;
        let vae = self.find_ckpts(&dirs, "flux_2_vae").into_iter().next().with_context(|| {
            format!("no FLUX.2 VAE (flux_2_vae_f16.ckpt) in {dirs:?}; download it with `draw-things-cli models ensure --model flux_2_vae_f16.ckpt`")
        })?;
        let tokenizer = self.ensure_tokenizer()?;
        let name = transformer.file_name().and_then(|n| n.to_str()).unwrap_or("").to_lowercase();
        let files = Flux2Files {
            transformer: transformer.to_path_buf(),
            text_encoder,
            tokenizer,
            vae,
            scheduler_config: None,
            is_distilled: spec.map(|s| s.distilled).unwrap_or(!name.contains("base")),
            kv: spec.map(|s| s.kv).unwrap_or(false) || name.contains("_kv"),
        };
        Ok(ResolvedModel { spec: spec.cloned(), dir: transformer.to_path_buf(), files })
    }

    /// The Qwen3 tokenizer used by FLUX.2 [klein]: from any local FLUX.2 snapshot, else
    /// `<models-dir>/flux_2_tokenizer/`, downloaded on first use (about 11 MB, from the
    /// Apache-2.0 FLUX.2 [klein] 4B release; Draw Things checkpoints do not include it).
    pub fn ensure_tokenizer(&self) -> Result<PathBuf> {
        for m in &self.models {
            if let Some(d) = self.snapshot_dir(&m.repo) {
                if d.join("tokenizer/tokenizer.json").is_file() {
                    return Ok(d.join("tokenizer"));
                }
            }
        }
        for d in &self.dirs {
            let t = d.join(TOKENIZER_DIR);
            if t.join("tokenizer.json").is_file() {
                return Ok(t);
            }
        }
        let target = self.primary_dir().join(TOKENIZER_DIR);
        log::info!("downloading the Qwen3 tokenizer into {}", target.display());
        std::fs::create_dir_all(&target)?;
        let agent = ureq::AgentBuilder::new().try_proxy_from_env(true).build();
        for f in ["tokenizer_config.json", "tokenizer.json"] {
            let url = format!("https://huggingface.co/{TOKENIZER_REPO}/resolve/main/tokenizer/{f}");
            let resp = agent.get(&url).call().with_context(|| {
                format!("downloading the tokenizer ({url}); place tokenizer.json and tokenizer_config.json in {}", target.display())
            })?;
            let tmp = target.join(format!("{f}.part"));
            std::io::copy(&mut resp.into_reader(), &mut std::fs::File::create(&tmp)?)?;
            std::fs::rename(&tmp, target.join(f))?;
        }
        Ok(target)
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

/// Download a Draw Things checkpoint by file name from the public CDN into `dir`, verifying its
/// SHA-256 against the published manifest. Skips files that already exist.
pub fn download_dt_ckpt(name: &str, dir: &Path, progress: &mut dyn FnMut(&str, u64, u64)) -> Result<PathBuf> {
    use sha2::Digest;
    let target = dir.join(name);
    if target.is_file() {
        return Ok(target);
    }
    let agent = ureq::AgentBuilder::new().try_proxy_from_env(true).build();
    let manifest: serde_json::Value = agent.get(DT_MANIFEST).call().with_context(|| format!("fetching {DT_MANIFEST}"))?.into_json()?;
    let sha = manifest.get(name).and_then(|v| v.as_str()).with_context(|| format!("{name} is not a published Draw Things model (see {DT_MANIFEST})"))?.to_lowercase();
    let url = format!("{DT_CDN}/{name}");
    let resp = agent.get(&url).call().with_context(|| format!("downloading {url}"))?;
    let total: u64 = resp.header("Content-Length").and_then(|v| v.parse().ok()).unwrap_or(0);
    std::fs::create_dir_all(dir)?;
    let tmp = dir.join(format!("{name}.part"));
    let mut out = std::fs::File::create(&tmp)?;
    let mut reader = resp.into_reader();
    let mut hasher = sha2::Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut done = 0u64;
    loop {
        let n = std::io::Read::read(&mut reader, &mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        std::io::Write::write_all(&mut out, &buf[..n])?;
        done += n as u64;
        progress(name, done, total);
    }
    drop(out);
    let got = format!("{:x}", hasher.finalize());
    if got != sha {
        let _ = std::fs::remove_file(&tmp);
        bail!("{name}: SHA-256 mismatch (got {got}, expected {sha})");
    }
    std::fs::rename(&tmp, &target)?;
    Ok(target)
}

/// The Draw Things files a transformer checkpoint needs alongside it.
pub fn dt_dependencies(name: &str) -> Vec<String> {
    let n = name.to_lowercase();
    if !n.starts_with("flux_2_") || n.starts_with("flux_2_vae") {
        return vec![];
    }
    let qwen = if n.contains("_4b") { "qwen_3_4b_f16.ckpt" } else { "qwen_3_8b_q8p.ckpt" };
    vec![qwen.to_string(), "flux_2_vae_f16.ckpt".to_string()]
}

/// Human-readable description of a Draw Things checkpoint; `loras` are checked against it
/// when it is a transformer.
pub fn inspect_ckpt(store: &ModelStore, path: &Path, loras: &[String]) -> Result<String> {
    use std::fmt::Write;
    use tc_core::ckpt::CkptFile;
    use tc_core::compare::{ckpt_component, Component};
    use tc_core::lora::{check, LoraFile};
    use tc_core::models::flux2::{Flux2LoraResolver, Flux2Source};
    let mut o = String::new();
    let file = CkptFile::open(path)?;
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    writeln!(o, "file: {} ({:.1} MB)", path.display(), size as f64 / 1e6)?;
    if let Some(e) = file.external_store() {
        writeln!(o, "external data: {}", e.display())?;
    }
    let mut codecs: std::collections::BTreeMap<String, usize> = Default::default();
    for k in file.keys() {
        let i = file.info(k).unwrap();
        *codecs.entry(format!("{} {:?}", i.codec_name(), i.dtype)).or_default() += 1;
    }
    writeln!(o, "tensors: {} ({})", file.len(), codecs.iter().map(|(k, v)| format!("{v} {k}")).collect::<Vec<_>>().join(", "))?;
    let component = ckpt_component(&file);
    writeln!(o, "component: {}", component.map(|c| format!("{c:?}")).unwrap_or("unknown".into()))?;
    let transformer = match component {
        Some(Component::Transformer) => Some(path.to_path_buf()),
        _ => store.find_ckpts(&store.sibling_dirs(path), "flux_2_klein").into_iter().find(|p| {
            CkptFile::open(p).ok().and_then(|f| ckpt_component(&f)) == Some(Component::Transformer)
        }),
    };
    match component {
        Some(Component::Transformer) => {
            let r = store.resolve_ckpt(path, store.find_spec(&path.file_name().unwrap_or_default().to_string_lossy()));
            let (source, config) = Flux2Source::from_ckpt(file)?;
            writeln!(o, "FLUX.2: {} double + {} single blocks, hidden {}, palettized: {}", config.num_layers, config.num_single_layers, config.inner_dim(), source.has_palettized())?;
            match r {
                Ok(m) => writeln!(
                    o,
                    "kv variant: {}\ndistilled: {}\ntext encoder: {}\nvae: {}\ntokenizer: {}",
                    m.files.kv,
                    m.files.is_distilled,
                    m.files.text_encoder.display(),
                    m.files.vae.display(),
                    m.files.tokenizer.display()
                )?,
                Err(e) => writeln!(o, "components: {e:#}")?,
            }
        }
        Some(Component::TextEncoder) => {
            let (_, c) = tc_core::models::qwen3::dt_source(file)?;
            writeln!(o, "Qwen3: {} layers, hidden {}, {} heads / {} kv heads, vocab {}", c.num_hidden_layers, c.hidden_size, c.num_attention_heads, c.num_key_value_heads, c.vocab_size)?;
        }
        Some(Component::Vae) => {
            let c = tc_core::models::flux2_vae::dt_config(&file)?;
            tc_core::models::flux2_vae::dt_source(file, &c)?;
            writeln!(o, "FLUX.2 VAE: channels {:?}, {} latent channels, all tensors mapped", c.block_out_channels, c.latent_channels)?;
        }
        _ => {}
    }
    let mut loras: Vec<PathBuf> = loras.iter().map(|l| store.resolve_lora(l).unwrap_or_else(|| PathBuf::from(l))).collect();
    if component == Some(Component::Lora) {
        loras.insert(0, path.to_path_buf());
    }
    if !loras.is_empty() {
        let t = transformer.context("no FLUX.2 transformer checkpoint to check the LoRAs against")?;
        let (source, config) = Flux2Source::from_ckpt(CkptFile::open(&t)?)?;
        let resolver = Flux2LoraResolver { config };
        for l in &loras {
            let f = LoraFile::open(l)?;
            let r = check(&resolver, &f, &|n| source.shape(n));
            writeln!(
                o,
                "LoRA {}: {} low-rank updates, {} applied to {} parameters of {}, {} unmatched{}",
                l.display(),
                f.updates.len(),
                r.applied,
                r.touched.len(),
                t.file_name().unwrap_or_default().to_string_lossy(),
                r.unmatched.len(),
                if r.unmatched.is_empty() { String::new() } else { format!(" (e.g. {:?})", &r.unmatched[..r.unmatched.len().min(5)]) }
            )?;
        }
    }
    Ok(o)
}
