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
    let out = std::env::temp_dir().join(format!("tc-e2e-{name}-{}.png", std::process::id()));
    let status = Command::new(env!("CARGO_BIN_EXE_draw-things-cli"))
        .args(["generate", "--offline", "--disable-preview", "--model", "tiny-klein"])
        .arg("--models-dir")
        .arg(fixtures())
        .args(["--prompt", "a red cube on a table", "--steps", "3", "--width", "128", "--height", "64", "--seed", "7"])
        .args(["--lora", "lora_a.safetensors:0.8", "--lora", "lora_b.safetensors:0.5"])
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
