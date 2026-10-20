mod common;
use common::*;
use nnc::{DataType, Device, Graph};
use tc_core::lora::{merge, LoraFile};
use tc_core::models::flux2::{image_ids, open_source, rope_table, text_ids, Flux2, Flux2LoraResolver};
use tc_core::weights::SafeTensorsFiles;

fn run(dir: &str, source_path: &str, with_lora: bool, guidance: Option<f32>) {
    let g = Graph::new();
    let (source, config) = open_source(&fixture(dir).join(source_path)).unwrap();
    let mut model = Flux2::load(&g, &source, config.clone(), device(), DataType::F32, false).unwrap();
    let io = SafeTensorsFiles::open(&[fixture(dir).join("flux2_io.safetensors")]).unwrap();
    if with_lora {
        let resolver = Flux2LoraResolver { config: config.clone() };
        let a = LoraFile::open(&fixture(dir).join("lora_peft.safetensors")).unwrap();
        let r = merge(&mut model.params, &resolver, &a, 0.8).unwrap();
        assert!(r.unmatched.is_empty(), "{:?}", r.unmatched);
        let b = LoraFile::open(&fixture(dir).join("lora_bfl.safetensors")).unwrap();
        let r = merge(&mut model.params, &resolver, &b, 0.5).unwrap();
        assert!(r.unmatched.is_empty(), "{:?}", r.unmatched);
    }
    let x = load(&g, &io, "x");
    let ctx = load(&g, &io, "ctx");
    let mut ids = text_ids(5);
    ids.extend(image_ids(4, 4, 0.0));
    ids.extend(image_ids(2, 2, 10.0));
    let (rope, shape) = rope_table(&ids, &config.axes_dims_rope, config.rope_theta);
    let rope = g.from_f32(&rope, &shape, device(), DataType::F32);
    let gv = guidance.map(|v| vec![v * 1000.0]);
    let step = model.prepare_step(&[700.0], gv.as_deref());
    let y = model.forward(&x, &ctx, &step, &rope);
    let expected = io.read_f32(if with_lora { "out_lora" } else { "out" }).unwrap();
    assert_close(&y.to_f32_vec(), &expected, 1e-4, &format!("{dir}/{source_path} lora={with_lora}"));
}

#[test]
fn flux2_diffusers_layout() {
    run("flux2", "transformer", false, None);
}

#[test]
fn flux2_bfl_layout() {
    run("flux2", "flux2_bfl.safetensors", false, None);
}

#[test]
fn flux2_with_two_loras() {
    run("flux2", "transformer", true, None);
    run("flux2", "flux2_bfl.safetensors", true, None);
}

#[test]
fn flux2_guidance_embed() {
    run("flux2_guidance", "transformer", false, Some(3.5));
}

#[test]
fn flux2_runtime_lora_adapters_match_merged() {
    let dir = "flux2";
    let g = Graph::new();
    let (source, config) = open_source(&fixture(dir).join("transformer")).unwrap();
    let mut model = Flux2::load(&g, &source, config.clone(), device(), DataType::F32, true).unwrap();
    let resolver = Flux2LoraResolver { config: config.clone() };
    for (file, w) in [("lora_peft.safetensors", 0.8), ("lora_bfl.safetensors", 0.5)] {
        let f = LoraFile::open(&fixture(dir).join(file)).unwrap();
        let (list, report) = tc_core::lora::adapters(&model.params, &resolver, &f, w).unwrap();
        assert!(report.unmatched.is_empty(), "{:?}", report.unmatched);
        for (p, a) in list {
            model.adapters.entry(p).or_default().push(a);
        }
    }
    let io = SafeTensorsFiles::open(&[fixture(dir).join("flux2_io.safetensors")]).unwrap();
    let x = load(&g, &io, "x");
    let ctx = load(&g, &io, "ctx");
    let mut ids = text_ids(5);
    ids.extend(image_ids(4, 4, 0.0));
    ids.extend(image_ids(2, 2, 10.0));
    let (rope, shape) = rope_table(&ids, &config.axes_dims_rope, config.rope_theta);
    let rope = g.from_f32(&rope, &shape, device(), DataType::F32);
    let step = model.prepare_step(&[700.0], None);
    let y = model.forward(&x, &ctx, &step, &rope);
    assert_close(&y.to_f32_vec(), &io.read_f32("out_lora").unwrap(), 1e-4, "runtime adapters");
}

#[test]
fn quantized_weights_stay_close() {
    let g = Graph::new();
    let n = 512 * 256;
    let w: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.37).sin() * 0.05).collect();
    let host = nnc::HostTensor::from_f32(&w, &[512, 256]);
    let q = tc_core::models::flux2::upload_weight(&g, &host, Device::Cpu, DataType::F32, true);
    let back = q.to_f32_vec();
    let rmse = (w.iter().zip(&back).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>() / n as f64).sqrt();
    assert!(rmse < 5e-4, "rmse {rmse}");
    assert!(back.iter().zip(&w).any(|(a, b)| a != b), "weights were not quantized");
}
