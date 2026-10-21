//! `models check-gpu` on the tiny model as a Draw Things folder (8-bit palettized transformer):
//! every stage on the device (Metal with `NNC_TEST_DEVICE=gpu`, else the CPU) must match the
//! CPU f32 reference.

use std::path::PathBuf;

use tc_core::gpucheck::{run, CheckOptions};
use tc_core::pipeline::{LoraSpec, Precision};
use tc_service::ModelStore;

#[test]
fn check_gpu_on_draw_things_q8p_model() {
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let dir = std::env::temp_dir().join(format!("tc-gpucheck-dt-models-{}", std::process::id()));
    tc_service::testing::write_tiny_draw_things_models(&fixtures, &dir).unwrap();
    let store = ModelStore::new(vec![dir.clone()]);
    let resolved = store.resolve("flux_2_klein_tiny_q8p.ckpt").unwrap();
    let gpu = std::env::var("NNC_TEST_DEVICE").as_deref() == Ok("gpu");
    let device = if gpu { nnc::Device::Gpu(0) } else { nnc::Device::Cpu };
    let loras: Vec<LoraSpec> = ["lora_a.safetensors", "lora_b_dt.ckpt"]
        .iter()
        .map(|l| LoraSpec { path: store.resolve_lora(l).unwrap(), weight: 0.7 })
        .collect();
    let o = CheckOptions {
        device,
        precision: Precision::default_for(device),
        double_blocks: 1,
        single_blocks: 2,
        text_encoder: true,
        vae: true,
    };
    let mut lines = vec![];
    run(&resolved.files, &loras, &o, &mut |l| {
        eprintln!("{l}");
        lines.push(l)
    })
    .unwrap();
    // Palettized weights stay palettized only on the GPU (the CPU expands them).
    assert!(lines.iter().any(|l| l.contains("8-bit palettized weights")));
    if gpu {
        assert!(lines.iter().any(|l| l.contains("(palettized)")), "no palettized weight reached the GPU");
    }
    let failed: Vec<&String> = lines.iter().filter(|l| l.contains("FAIL") || l.contains("suspicious")).collect();
    assert!(failed.is_empty(), "{failed:#?}");
    let _ = std::fs::remove_dir_all(dir);
}
