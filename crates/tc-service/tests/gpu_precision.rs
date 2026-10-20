//! GPU (`NNC_TEST_DEVICE=gpu`) precision matrix: the tiny FLUX.2 [klein] snapshot with two LoRAs,
//! through the same request mapping as the CLI / gRPC server, at several per-component
//! precisions, against the diffusers renderings. Pins down which component misbehaves on Metal.

use std::path::PathBuf;

use nnc::{DataType, Device};
use tc_config::{GenerationConfiguration, LoRA};
use tc_core::pipeline::{Cancellation, EngineOptions, Flux2Engine, Precision};
use tc_service::{build_params, GenerationInput, ModelStore};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn expected(file: &str) -> Vec<f32> {
    let bytes = std::fs::read(fixtures().join(file)).unwrap();
    let n = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let offs = header["image"]["data_offsets"].as_array().unwrap();
    let (a, b) = (offs[0].as_u64().unwrap() as usize, offs[1].as_u64().unwrap() as usize);
    bytes[8 + n + a..8 + n + b].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

/// (max, mean) error of one generation against `expected_file`.
fn run(device: Device, precision: Precision, with_ref: bool, expected_file: &str) -> (f32, f32) {
    let store = ModelStore::new(vec![fixtures()]);
    let config = GenerationConfiguration {
        model: Some("tiny-klein".into()),
        start_width: 2,
        start_height: 1,
        steps: 3,
        seed: 7,
        guidance_scale: 1.0,
        strength: 1.0,
        sampler: tc_config::SamplerType::DDIMTrailing as i8,
        resolution_dependent_shift: true,
        loras: vec![
            LoRA { file: Some("lora_a.safetensors".into()), weight: 0.8, mode: 0 },
            LoRA { file: Some("lora_b.safetensors".into()), weight: 0.5, mode: 0 },
        ],
        ..Default::default()
    };
    let references = if with_ref { vec![tc_core::image_io::load_rgb(&fixtures().join("ref.png")).unwrap()] } else { vec![] };
    let input = GenerationInput { config, prompt: "a red cube on a table".into(), references, ..Default::default() };
    let (resolved, params) = build_params(&store, &input).unwrap();
    let mut engine = Flux2Engine::new(EngineOptions { device, precision, keep_text_encoder: false, prompt_cache_size: 0, quantize: false });
    engine.set_model(resolved.files);
    let out = engine.generate(&params, &Cancellation::default(), &mut |_| true).unwrap();
    let got: Vec<f32> = out.images[0].to_f32_vec().iter().map(|v| (v / 2.0 + 0.5).clamp(0.0, 1.0)).collect();
    let exp = expected(expected_file);
    assert_eq!(got.len(), exp.len());
    let diffs: Vec<f32> = got.iter().zip(exp.iter()).map(|(a, b)| (a - b).abs()).collect();
    (diffs.iter().cloned().fold(0f32, f32::max), diffs.iter().sum::<f32>() / diffs.len() as f32)
}

/// The same request on CPU matches diffusers (checks this harness).
#[test]
fn cpu_matches_reference() {
    let f32p = Precision::default_for(Device::Cpu);
    for (with_ref, file) in [(false, "expected.safetensors"), (true, "expected_ref.safetensors")] {
        let (max, _) = run(Device::Cpu, f32p, with_ref, file);
        assert!(max <= 1.5 / 255.0, "cpu (ref: {with_ref}): max error {max}");
    }
}

#[test]
fn gpu_precision_matrix() {
    if std::env::var("NNC_TEST_DEVICE").as_deref() != Ok("gpu") {
        eprintln!("skipped: set NNC_TEST_DEVICE=gpu to run on Metal");
        return;
    }
    use DataType::{BF16, F16, F32};
    let p = |transformer, text_encoder, vae| Precision { transformer, text_encoder, vae };
    // (label, precision, reference image, max error bound)
    let cases = [
        ("f32/f32/f32 no-ref", p(F32, F32, F32), false, 3.0 / 255.0),
        ("f32/f32/f32 ref", p(F32, F32, F32), true, 3.0 / 255.0),
        ("f32/f32/f16-vae no-ref", p(F32, F32, F16), false, 0.05),
        ("f32/f32/f16-vae ref", p(F32, F32, F16), true, 0.05),
        ("f32/f16-te/f32 ref", p(F32, F16, F32), true, 0.1),
        ("f16/f32/f32 ref", p(F16, F32, F32), true, 0.1),
        ("bf16/bf16/f16 ref", p(BF16, BF16, F16), true, 0.2),
    ];
    let mut failures = vec![];
    for (label, precision, with_ref, bound) in cases {
        let file = if with_ref { "expected_ref.safetensors" } else { "expected.safetensors" };
        let (max, mean) = run(Device::Gpu(0), precision, with_ref, file);
        eprintln!("gpu {label}: max error {max:.4}, mean error {mean:.5}");
        if !(max <= bound) {
            failures.push(format!("{label}: max {max:.4} > {bound}"));
        }
    }
    assert!(failures.is_empty(), "GPU precision matrix:\n{}", failures.join("\n"));
}
