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

/// Runs `generate` on the tiny snapshot with two LoRAs plus `extra` arguments and compares the
/// PNG with the diffusers rendering in `expected_file`.
fn generate_and_compare(name: &str, extra: &[&str], expected_file: &str) {
    let out = std::env::temp_dir().join(format!("tc-e2e-{name}-{}.png", std::process::id()));
    let status = Command::new(env!("CARGO_BIN_EXE_draw-things-cli"))
        .args(["generate", "--cpu", "--offline", "--disable-preview", "--model", "tiny-klein"])
        .arg("--models-dir")
        .arg(fixtures())
        .args(["--prompt", "a red cube on a table", "--steps", "3", "--width", "128", "--height", "64", "--seed", "7"])
        .args(["--lora", "lora_a.safetensors:0.8", "--lora", "lora_b.safetensors:0.5"])
        .args(extra)
        .arg("-o")
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success());
    let img = image::open(&out).unwrap().to_rgb8();
    assert_eq!((img.width(), img.height()), (128, 64));
    let exp = expected(expected_file);
    let mut max = 0f32;
    for (a, b) in img.as_raw().iter().zip(exp.iter()) {
        max = max.max((*a as f32 / 255.0 - b).abs());
    }
    eprintln!("{name}: max pixel error {max}");
    assert!(max <= 1.5 / 255.0, "{name}: max pixel error {max}");
    let _ = std::fs::remove_file(out);
}

#[test]
fn cli_generate_with_two_loras_matches_reference() {
    generate_and_compare("loras", &[], "expected.safetensors");
}

#[test]
fn cli_generate_with_two_loras_and_reference_image_matches_reference() {
    let reference = fixtures().join("ref.png");
    generate_and_compare("loras-ref", &["--image", reference.to_str().unwrap()], "expected_ref.safetensors");
}
