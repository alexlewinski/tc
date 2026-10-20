//! Test support: the tiny FLUX.2 [klein] fixture as a Draw Things models folder.

use std::path::Path;

use anyhow::{Context, Result};
use tc_core::ckpt::{CkptWriter, CODEC_RAW};
use tc_core::weights::{SafeTensorsFiles, TensorSource};

/// Write `fixtures/tiny-klein` (see scripts/fixtures/snapshot_tiny.py) into `dir` the way a
/// Draw Things models folder holds a model: `flux_2_klein_tiny_f32.ckpt` (and a palettized
/// `flux_2_klein_tiny_q8p.ckpt`), `qwen_3_tiny_f32.ckpt`,
/// `flux_2_vae_tiny.ckpt` (f32 tensors), the tokenizer in `flux_2_tokenizer/`, `lora_a.safetensors`
/// as is and `lora_b` converted to Draw Things' LoRA layout as `lora_b_dt.ckpt`
/// (`__dit__[...]__down__` / `__up__`, alpha folded in). Does nothing if `dir/done` exists.
pub fn write_tiny_draw_things_models(fixtures: &Path, dir: &Path) -> Result<()> {
    if dir.join("done").exists() {
        return Ok(());
    }
    std::fs::create_dir_all(dir.join(crate::zoo::TOKENIZER_DIR))?;
    let snap = fixtures.join("tiny-klein");
    let write = |file: &str, tensors: Vec<(String, nnc::HostTensor)>| -> Result<()> {
        let mut w = CkptWriter::new(&dir.join(file), false)?;
        for (n, t) in tensors {
            w.dense(&n, &t, CODEC_RAW)?;
        }
        w.finish()
    };
    let (src, config) = tc_core::models::flux2::open_source(&snap.join("transformer"))?;
    let transformer = tc_core::models::flux2::dt_tensors(&src, &config)?;
    // Like Draw Things' `_q8p` files: f16, matrices 8-bit palettized (loaded as such on GPU).
    let mut w = CkptWriter::new(&dir.join("flux_2_klein_tiny_q8p.ckpt"), false)?;
    for (n, t) in &transformer {
        let t = t.to_dtype(nnc::DataType::F16);
        if t.shape().len() == 2 && t.numel() % 1024 == 0 {
            w.q8p(n, &t, 1024)?;
        } else {
            w.dense(n, &t, CODEC_RAW)?;
        }
    }
    w.finish()?;
    write("flux_2_klein_tiny_f32.ckpt", transformer)?;
    let te = SafeTensorsFiles::open_dir(&snap.join("text_encoder"))?;
    let te_config = serde_json::from_slice(&std::fs::read(snap.join("text_encoder/config.json"))?)?;
    write("qwen_3_tiny_f32.ckpt", tc_core::models::qwen3::dt_tensors(&te, &te_config)?)?;
    let vae = SafeTensorsFiles::open_dir(&snap.join("vae"))?;
    let vae_config = serde_json::from_slice(&std::fs::read(snap.join("vae/config.json"))?)?;
    write("flux_2_vae_tiny.ckpt", tc_core::models::flux2_vae::dt_tensors(&vae, &vae_config, true)?)?;
    for f in ["tokenizer.json", "tokenizer_config.json"] {
        std::fs::copy(snap.join("tokenizer").join(f), dir.join(crate::zoo::TOKENIZER_DIR).join(f))?;
    }
    std::fs::copy(fixtures.join("lora_a.safetensors"), dir.join("lora_a.safetensors"))?;
    // lora_b: kohya double_blocks.0.img_mlp.0 (rank 4, alpha 2) -> x_gate_proj-0 / x_up_proj-0.
    let lb = SafeTensorsFiles::open(&[fixtures.join("lora_b.safetensors")])?;
    let key = "lora_unet_double_blocks_0_img_mlp_0";
    let down = TensorSource::read(&lb, &format!("{key}.lora_down.weight"), nnc::DataType::F32)?;
    let up = lb.read_f32(&format!("{key}.lora_up.weight"))?;
    let rank = down.shape()[0];
    let alpha = lb.read_f32(&format!("{key}.alpha"))?.first().copied().context("alpha")?;
    let scale = alpha / rank as f32;
    let half = up.len() / 2;
    let mut w = CkptWriter::new(&dir.join("lora_b_dt.ckpt"), false)?;
    for (kind, part) in [("x_gate_proj", &up[..half]), ("x_up_proj", &up[half..])] {
        let part: Vec<f32> = part.iter().map(|v| v * scale).collect();
        w.dense(&format!("__dit__[t-{kind}-0-0]__down__"), &down, CODEC_RAW)?;
        w.dense(&format!("__dit__[t-{kind}-0-0]__up__"), &nnc::HostTensor::from_f32(&part, &[part.len() / rank, rank]), CODEC_RAW)?;
    }
    w.finish()?;
    std::fs::write(dir.join("done"), b"")?;
    Ok(())
}
