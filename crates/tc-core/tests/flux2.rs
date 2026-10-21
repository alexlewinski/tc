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

#[test]
fn flux2_kv_variant_matches_diffusers() {
    let dir = "flux2";
    let g = Graph::new();
    let (source, config) = open_source(&fixture(dir).join("transformer")).unwrap();
    let model = Flux2::load(&g, &source, config.clone(), device(), DataType::F32, false).unwrap();
    let io = SafeTensorsFiles::open(&[fixture(dir).join("flux2_kv_io.safetensors")]).unwrap();
    let x = load(&g, &io, "x");
    let r = load(&g, &io, "ref");
    let ctx = load(&g, &io, "ctx");
    let mut ids = text_ids(5);
    ids.extend(image_ids(2, 2, 10.0));
    ids.extend(image_ids(2, 3, 20.0));
    ids.extend(image_ids(4, 4, 0.0));
    let (rope, shape) = rope_table(&ids, &config.axes_dims_rope, config.rope_theta);
    let rope = g.from_f32(&rope, &shape, device(), DataType::F32);
    let ref_step = model.prepare_step(&[0.0], None);
    let input = nnc::ops::concat(&[&r, &x], 1);
    for (t, key) in [(700.0, "out"), (400.0, "out_cached")] {
        let step = model.prepare_step(&[t], None);
        let y = model.forward_kv(&input, &ctx, &step, &ref_step, 10, &rope);
        assert_eq!(y.shape(), vec![1, 16, 8]);
        assert_close(&y.to_f32_vec(), &io.read_f32(key).unwrap(), 1e-4, key);
    }
}

/// f16 compute (as with 8-bit palettized weights on Metal) with text features as large as
/// Qwen3's real outliers (~2e4): the streams must not saturate. Before the residual streams,
/// norms and gating moved to f32, f16 squared these values in the layer norm and the whole
/// stream ended at the f16 limit (the 9B model then produced pure noise). GPU only: nnc's CPU
/// matrix products are f32.
#[test]
fn flux2_f16_with_large_text_features_gpu() {
    if !device().is_gpu() {
        return;
    }
    let dir = "flux2";
    let (source, config) = open_source(&fixture(dir).join("transformer")).unwrap();
    let io = SafeTensorsFiles::open(&[fixture(dir).join("flux2_io.safetensors")]).unwrap();
    let x = io.read_f32("x").unwrap();
    let x_shape = io.info("x").unwrap().shape.clone();
    let ctx = io.read_f32("ctx").unwrap();
    let ctx_shape = io.info("ctx").unwrap().shape.clone();
    // Scale the text features so their largest value is 2e4.
    let peak = ctx.iter().fold(0f32, |m, v| m.max(v.abs()));
    let ctx: Vec<f32> = ctx.iter().map(|v| v * 2e4 / peak).collect();
    let mut ids = text_ids(5);
    ids.extend(image_ids(4, 4, 0.0));
    ids.extend(image_ids(2, 2, 10.0));
    let (rope, rope_shape) = rope_table(&ids, &config.axes_dims_rope, config.rope_theta);
    let run = |dev: Device, dtype: DataType| -> Vec<f32> {
        let g = Graph::new();
        let model = Flux2::load(&g, &source, config.clone(), dev, dtype, false).unwrap();
        let xv = g.from_f32(&x, &x_shape, dev, dtype);
        let cv = g.from_f32(&ctx, &ctx_shape, dev, dtype);
        let rv = g.from_f32(&rope, &rope_shape, dev, dtype);
        let step = model.prepare_step(&[700.0], None);
        model.forward(&xv, &cv, &step, &rv).to_f32_vec()
    };
    let reference = run(Device::Cpu, DataType::F32);
    let got = run(device(), DataType::F16);
    assert!(got.iter().all(|v| v.is_finite()), "non-finite output");
    let num: f64 = got.iter().zip(&reference).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
    let den: f64 = reference.iter().map(|b| (*b as f64).powi(2)).sum();
    let rel = (num / den).sqrt();
    eprintln!("f16 GPU vs f32 CPU with large text features: rel {rel:.5}");
    assert!(rel < 0.02, "rel {rel}");
}
