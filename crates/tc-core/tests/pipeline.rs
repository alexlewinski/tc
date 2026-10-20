mod common;
use common::*;
use nnc::Device;
use tc_core::image_io;
use tc_core::pipeline::{Cancellation, EngineOptions, Flux2Engine, Flux2Files, GenerateParams, Precision};
use tc_core::weights::SafeTensorsFiles;

fn engine(distilled: bool) -> Flux2Engine {
    let dir = fixture("pipeline");
    let mut e = Flux2Engine::new(EngineOptions {
        device: Device::Cpu,
        precision: Precision::default_for(Device::Cpu),
        keep_text_encoder: false,
        prompt_cache_size: 0,
    });
    e.set_model(Flux2Files {
        transformer: dir.join("transformer"),
        text_encoder: dir.join("missing"),
        tokenizer: dir.join("missing"),
        vae: dir.join("vae"),
        scheduler_config: Some(dir.join("scheduler/scheduler_config.json")),
        is_distilled: distilled,
    });
    e
}

fn check(name: &str, distilled: bool, guidance: f32, steps: usize, with_ref: bool) {
    let io = SafeTensorsFiles::open(&[fixture("pipeline/pipeline_io.safetensors")]).unwrap();
    let pe = io.read("pe", nnc::DataType::F32).unwrap();
    let ne = io.read("ne", nnc::DataType::F32).unwrap();
    let mut e = engine(distilled);
    let params = GenerateParams {
        width: 96,
        height: 64,
        steps,
        guidance_scale: guidance,
        seed: 1234,
        reference_images: if with_ref { vec![image_io::load_rgb(&fixture("pipeline/ref.png")).unwrap()] } else { vec![] },
        prompt_embeddings: Some((pe, Some(ne))),
        ..Default::default()
    };
    let mut events = vec![];
    let out = e.generate(&params, &Cancellation::default(), &mut |p| {
        events.push(p);
        true
    }).unwrap();
    let img: Vec<f32> = out.images[0].to_f32_vec().iter().map(|v| (v / 2.0 + 0.5).clamp(0.0, 1.0)).collect();
    assert_close(&img, &io.read_f32(name).unwrap(), 1e-4, name);
    assert!(events.len() >= steps + 2, "{events:?}");
}

#[test]
fn distilled_text_to_image_matches_diffusers() {
    check("distilled", true, 1.0, 4, false);
}

#[test]
fn cfg_text_to_image_matches_diffusers() {
    check("cfg", false, 3.5, 3, false);
}

#[test]
fn reference_image_matches_diffusers() {
    check("ref", true, 1.0, 2, true);
}
