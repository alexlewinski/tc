//! GPU (`NNC_TEST_DEVICE=gpu`) precision matrix: the tiny FLUX.2 [klein] snapshot with two LoRAs,
//! through the same request mapping as the CLI / gRPC server, at several per-component
//! precisions, against the diffusers renderings. Pins down which component misbehaves on Metal.

use std::path::PathBuf;

use nnc::{DataType, Device, HostTensor};
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

fn engine(device: Device, precision: Precision) -> (Flux2Engine, ModelStore) {
    let store = ModelStore::new(vec![fixtures()]);
    let mut engine = Flux2Engine::new(EngineOptions { device, precision, keep_text_encoder: false, prompt_cache_size: 0, quantize: false, keep_palettized: true });
    engine.set_model(store.resolve("tiny-klein").unwrap().files);
    (engine, store)
}

/// (max, mean, NaN count) of `|a - b|`.
fn error(got: &[f32], exp: &[f32]) -> (f32, f32, usize) {
    assert_eq!(got.len(), exp.len());
    let nans = got.iter().filter(|v| !v.is_finite()).count();
    let diffs: Vec<f32> = got.iter().zip(exp.iter()).filter(|(a, _)| a.is_finite()).map(|(a, b)| (a - b).abs()).collect();
    (diffs.iter().cloned().fold(0f32, f32::max), diffs.iter().sum::<f32>() / diffs.len().max(1) as f32, nans)
}

/// (max, mean, non-finite pixels) of one generation against `expected_file`; `embeddings`
/// replaces the text encoder.
fn run(device: Device, precision: Precision, with_ref: bool, expected_file: &str, embeddings: Option<&HostTensor>) -> (f32, f32, usize) {
    error(&generate(device, precision, with_ref, true, embeddings), &expected(expected_file))
}

/// One generation (pixels in `[0, 1]`, non-finite values kept).
fn generate(device: Device, precision: Precision, with_ref: bool, with_loras: bool, embeddings: Option<&HostTensor>) -> Vec<f32> {
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
        loras: if with_loras {
            vec![
                LoRA { file: Some("lora_a.safetensors".into()), weight: 0.8, mode: 0 },
                LoRA { file: Some("lora_b.safetensors".into()), weight: 0.5, mode: 0 },
            ]
        } else {
            vec![]
        },
        ..Default::default()
    };
    let references = if with_ref { vec![tc_core::image_io::load_rgb(&fixtures().join("ref.png")).unwrap()] } else { vec![] };
    let input = GenerationInput { config, prompt: "a red cube on a table".into(), references, ..Default::default() };
    let (_, mut params) = build_params(&store, &input).unwrap();
    params.prompt_embeddings = embeddings.map(|e| (e.clone(), None));
    let (mut engine, _) = engine(device, precision);
    let out = engine.generate(&params, &Cancellation::default(), &mut |_| true).unwrap();
    // NaN must survive the clamp so it is counted.
    out.images[0].to_f32_vec().iter().map(|v| if v.is_finite() { (v / 2.0 + 0.5).clamp(0.0, 1.0) } else { *v }).collect()
}

/// The same request on CPU matches diffusers (checks this harness).
#[test]
fn cpu_matches_reference() {
    let f32p = Precision::default_for(Device::Cpu);
    for (with_ref, file) in [(false, "expected.safetensors"), (true, "expected_ref.safetensors")] {
        let (max, _, nans) = run(Device::Cpu, f32p, with_ref, file, None);
        assert_eq!(nans, 0);
        assert!(max <= 1.5 / 255.0, "cpu (ref: {with_ref}): max error {max}");
    }
    let emb = engine(Device::Cpu, f32p).0.encode_prompt("a red cube on a table", 512).unwrap();
    let (max, _, _) = run(Device::Cpu, f32p, true, "expected_ref.safetensors", Some(&emb));
    assert!(max <= 1.5 / 255.0, "cpu with precomputed embeddings: max error {max}");
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
    // Stage 1: text embeddings, GPU vs CPU.
    let prompt = "a red cube on a table";
    let cpu_emb = engine(Device::Cpu, Precision::default_for(Device::Cpu)).0.encode_prompt(prompt, 512).unwrap();
    // Bounds relative to the largest |value|, after 27 Qwen3 layers. Measured on Metal (GitHub
    // macos-15 runner): f32 exact, f16 0.6%, bf16 5.1%; bf16 keeps 3 fewer mantissa bits than f16
    // (8x coarser) and its error is ~8.6x f16's, i.e. rounding, not a kernel bug. The images
    // these embeddings produce stay within the stage-3 bounds.
    for (label, te, rel) in [("f32", F32, 1e-3), ("f16", F16, 0.02), ("bf16", BF16, 0.1)] {
        let gpu_emb = engine(Device::Gpu(0), p(F32, te, F32)).0.encode_prompt(prompt, 512).unwrap();
        let (max, mean, nans) = error(&gpu_emb.to_f32_vec(), &cpu_emb.to_f32_vec());
        let scale = cpu_emb.to_f32_vec().iter().fold(0f32, |m, v| m.max(v.abs()));
        eprintln!("gpu text embeddings {label}: max error {max:.4} (max |ref| {scale:.2}), mean error {mean:.5}, non-finite {nans}");
        if nans > 0 || max > rel * scale.max(1.0) {
            failures.push(format!("text embeddings {label}: max {max:.4}, non-finite {nans}"));
        }
    }
    // Stage 2a: no LoRAs, CPU embeddings: GPU vs the same generation on CPU.
    let f32p = p(F32, F32, F32);
    let cpu_img = generate(Device::Cpu, f32p, true, false, Some(&cpu_emb));
    let gpu_img = generate(Device::Gpu(0), f32p, true, false, Some(&cpu_emb));
    let (max, mean, nans) = error(&gpu_img, &cpu_img);
    eprintln!("gpu no-lora vs cpu (f32, ref): max error {max:.4}, mean error {mean:.5}, non-finite {nans}");
    if nans > 0 || max > 3.0 / 255.0 {
        failures.push(format!("no-lora vs cpu: max {max:.4}, non-finite {nans}"));
    }
    // Stage 2b: generation with the CPU embeddings (no text encoder on GPU).
    for (label, precision, with_ref, bound) in [
        ("cpu-emb f32/-/f32 no-ref", p(F32, F32, F32), false, 3.0 / 255.0),
        ("cpu-emb f32/-/f32 ref", p(F32, F32, F32), true, 3.0 / 255.0),
        ("cpu-emb f32/-/f16-vae ref", p(F32, F32, F16), true, 0.05),
    ] {
        let file = if with_ref { "expected_ref.safetensors" } else { "expected.safetensors" };
        let (max, mean, nans) = run(Device::Gpu(0), precision, with_ref, file, Some(&cpu_emb));
        eprintln!("gpu {label}: max error {max:.4}, mean error {mean:.5}, non-finite {nans}");
        if nans > 0 || !(max <= bound) {
            failures.push(format!("{label}: max {max:.4} > {bound}, non-finite {nans}"));
        }
    }
    // Stage 3: the full request path.
    for (label, precision, with_ref, bound) in cases {
        let file = if with_ref { "expected_ref.safetensors" } else { "expected.safetensors" };
        let (max, mean, nans) = run(Device::Gpu(0), precision, with_ref, file, None);
        eprintln!("gpu {label}: max error {max:.4}, mean error {mean:.5}, non-finite {nans}");
        if nans > 0 || !(max <= bound) {
            failures.push(format!("{label}: max {max:.4} > {bound}, non-finite {nans}"));
        }
    }
    assert!(failures.is_empty(), "GPU precision matrix:\n{}", failures.join("\n"));
}
