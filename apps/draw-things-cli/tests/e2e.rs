//! End-to-end: tiny FLUX.2 [klein] snapshot + two LoRAs through the real CLI binary, compared
//! with a diffusers reference rendering (see scripts/fixtures/snapshot_tiny.py).

use std::path::PathBuf;
use std::process::Command;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../crates/tc-service/tests/fixtures")
}

fn expected(file: &str) -> Vec<f32> {
    let bytes = std::fs::read(fixtures().join(file)).unwrap();
    let files = safetensors_header(&bytes);
    files
}

/// Minimal reader for the single f32 tensor in expected.safetensors.
fn safetensors_header(bytes: &[u8]) -> Vec<f32> {
    let n = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let offs = header["image"]["data_offsets"].as_array().unwrap();
    let (a, b) = (offs[0].as_u64().unwrap() as usize, offs[1].as_u64().unwrap() as usize);
    bytes[8 + n + a..8 + n + b].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

/// Runs `generate` on the tiny snapshot with two LoRAs plus `extra` arguments; returns the PNG
/// as floats in `[0, 1]`.
fn generate(name: &str, extra: &[&str]) -> Vec<f32> {
    generate_with(name, &fixtures(), "tiny-klein", ["lora_a.safetensors:0.8", "lora_b.safetensors:0.5"], extra)
}

fn generate_with(name: &str, models_dir: &std::path::Path, model: &str, loras: [&str; 2], extra: &[&str]) -> Vec<f32> {
    let out = std::env::temp_dir().join(format!("tc-e2e-{name}-{}.png", std::process::id()));
    let status = Command::new(env!("CARGO_BIN_EXE_draw-things-cli"))
        .args(["generate", "--offline", "--disable-preview", "--model", model])
        .arg("--models-dir")
        .arg(models_dir)
        .args(["--prompt", "a red cube on a table", "--steps", "3", "--width", "128", "--height", "64", "--seed", "7"])
        .args(["--lora", loras[0], "--lora", loras[1]])
        .args(extra)
        .arg("-o")
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success(), "{name}: draw-things-cli failed");
    let img = image::open(&out).unwrap().to_rgb8();
    assert_eq!((img.width(), img.height()), (128, 64));
    let _ = std::fs::remove_file(out);
    img.as_raw().iter().map(|v| *v as f32 / 255.0).collect()
}

/// (max, mean) absolute error against the diffusers rendering in `expected_file`.
fn error(got: &[f32], expected_file: &str) -> (f32, f32) {
    let exp = expected(expected_file);
    assert_eq!(got.len(), exp.len());
    let diffs: Vec<f32> = got.iter().zip(exp.iter()).map(|(a, b)| (a - b).abs()).collect();
    (diffs.iter().cloned().fold(0f32, f32::max), diffs.iter().sum::<f32>() / diffs.len() as f32)
}

#[test]
fn cli_generate_with_two_loras_matches_reference() {
    let (max, _) = error(&generate("loras", &["--cpu"]), "expected.safetensors");
    assert!(max <= 1.5 / 255.0, "max pixel error {max}");
}

#[test]
fn cli_generate_with_two_loras_and_reference_image_matches_reference() {
    let reference = fixtures().join("ref.png");
    let (max, _) = error(&generate("loras-ref", &["--cpu", "--image", reference.to_str().unwrap()]), "expected_ref.safetensors");
    assert!(max <= 1.5 / 255.0, "max pixel error {max}");
}

/// The tiny model as a Draw Things models folder: transformer, `qwen_3_*.ckpt` text encoder and
/// VAE as `.ckpt` files (f32), the tokenizer in `flux_2_tokenizer/`, and the second LoRA converted
/// to Draw Things' LoRA layout (`__dit__[...]__down__` / `__up__`, alpha folded in).
fn draw_things_models_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tc-e2e-dt-models-{}", std::process::id()));
    tc_service::testing::write_tiny_draw_things_models(&fixtures(), &dir).unwrap();
    dir
}

#[test]
fn cli_generate_with_draw_things_checkpoints_matches_reference() {
    let dir = draw_things_models_dir();
    let loras = ["lora_a.safetensors:0.8", "lora_b_dt.ckpt:0.5"];
    let (max, _) = error(&generate_with("dt", &dir, "flux_2_klein_tiny_f32.ckpt", loras, &["--cpu"]), "expected.safetensors");
    assert!(max <= 1.5 / 255.0, "max pixel error {max}");
    let reference = fixtures().join("ref.png");
    let got = generate_with("dt-ref", &dir, "flux_2_klein_tiny_f32.ckpt", loras, &["--cpu", "--image", reference.to_str().unwrap()]);
    let (max, _) = error(&got, "expected_ref.safetensors");
    assert!(max <= 1.5 / 255.0, "max pixel error {max}");
}

/// Metal (`NNC_TEST_DEVICE=gpu`): a palettized (`_q8p`) Draw Things checkpoint at the default
/// precision with a Draw Things LoRA and a reference image. Palettized weights run in f16 on
/// GPU while the default is bf16; every input the transformer sees must follow the model.
#[test]
fn cli_metal_draw_things_q8p_checkpoint() {
    if std::env::var("NNC_TEST_DEVICE").as_deref() != Ok("gpu") {
        eprintln!("skipped: set NNC_TEST_DEVICE=gpu to run on Metal");
        return;
    }
    let dir = draw_things_models_dir();
    let loras = ["lora_a.safetensors:0.8", "lora_b_dt.ckpt:0.5"];
    let reference = fixtures().join("ref.png");
    let got = generate_with("dt-q8p", &dir, "flux_2_klein_tiny_q8p.ckpt", loras, &["--image", reference.to_str().unwrap()]);
    assert!(got.iter().all(|v| v.is_finite()));
    let (max, mean) = error(&got, "expected_ref.safetensors");
    eprintln!("metal q8p checkpoint: max error {max:.4}, mean error {mean:.5}");
    // The same request on CPU (8-bit weights, f32) gives max 0.068 / mean 0.0015; f16 adds a
    // little. A wrong input type or layout is far outside this.
    assert!(max <= 0.2 && mean <= 0.01, "max {max}, mean {mean}");
}

/// Metal smoke test (`NNC_TEST_DEVICE=gpu`, a Mac with a real GPU): the reference image + two
/// LoRAs case through the CLI on the GPU at each precision. Reduced precision gets looser bounds;
/// they catch broken kernels and layouts, not rounding.
#[test]
fn cli_metal_smoke_reference_image_and_two_loras() {
    if std::env::var("NNC_TEST_DEVICE").as_deref() != Ok("gpu") {
        eprintln!("skipped: set NNC_TEST_DEVICE=gpu to run on Metal");
        return;
    }
    assert!(nnc::gpu_count() > 0, "NNC_TEST_DEVICE=gpu but nnc sees no GPU (no usable Metal device)");
    let reference = fixtures().join("ref.png");
    let reference = reference.to_str().unwrap();
    let cases: [(&str, &[&str], f32, f32); 4] = [
        ("f32", &["--precision", "f32", "--text-encoder-precision", "f32"], 3.0 / 255.0, 1.0 / 255.0),
        ("f16", &["--precision", "f16", "--text-encoder-precision", "f16"], 0.1, 0.01),
        ("bf16 (default)", &[], 0.2, 0.03),
        ("quantize", &["--quantize"], 0.2, 0.03),
    ];
    let mut failures = vec![];
    for (name, flags, max_bound, mean_bound) in cases {
        let mut args = vec!["--image", reference];
        args.extend_from_slice(flags);
        let got = generate(&format!("metal-{}", name.split(' ').next().unwrap()), &args);
        let (max, mean) = error(&got, "expected_ref.safetensors");
        eprintln!("metal {name}: max error {max:.4}, mean error {mean:.5}");
        if !(max <= max_bound && mean <= mean_bound) {
            failures.push(format!("{name}: max {max:.4} (bound {max_bound}), mean {mean:.5} (bound {mean_bound})"));
        }
    }
    assert!(failures.is_empty(), "Metal smoke test failed:\n{}", failures.join("\n"));
}
