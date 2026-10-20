//! `cargo run --release -p tc-core --example flux2_reduced_parity -- <transformer.ckpt> <reference.safetensors> [double] [single]`
//! Runs the first `double`/`single` blocks (default 1/1) of a Draw Things FLUX.2 checkpoint and of
//! a reference (BFL or diffusers names) on the same random inputs, plain and KV variant, on CPU.
use nnc::{ops, DataType, Device, Graph};
use tc_core::models::flux2::{image_ids, open_source, rope_table, text_ids, Flux2};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let nd: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(1);
    let ns: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(1);
    let g = Graph::new();
    let load = |p: &str| -> anyhow::Result<Flux2> {
        let (source, mut config) = open_source(std::path::Path::new(p))?;
        config.num_layers = nd;
        config.num_single_layers = ns;
        Flux2::load(&g, &source, config, Device::Cpu, DataType::F32, false)
    };
    let a = load(&args[0])?;
    let b = load(&args[1])?;
    let c = a.config.clone();
    let mut seed = 1u64;
    let mut rnd = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 3.4
            })
            .collect()
    };
    let (lt, h, w, rh, rw) = (24usize, 8usize, 8usize, 4usize, 6usize);
    let txt = g.from_f32(&rnd(lt * c.joint_attention_dim), &[1, lt, c.joint_attention_dim], Device::Cpu, DataType::F32);
    let img = g.from_f32(&rnd(h * w * c.in_channels), &[1, h * w, c.in_channels], Device::Cpu, DataType::F32);
    let r = g.from_f32(&rnd(rh * rw * c.in_channels), &[1, rh * rw, c.in_channels], Device::Cpu, DataType::F32);
    let rel = |p: &[f32], q: &[f32]| {
        let d: f64 = p.iter().zip(q).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
        let n: f64 = q.iter().map(|y| (*y as f64).powi(2)).sum();
        (d / n).sqrt()
    };
    // Plain: [txt, img, ref].
    let mut ids = text_ids(lt);
    ids.extend(image_ids(h, w, 0.0));
    ids.extend(image_ids(rh, rw, 10.0));
    let (rope, s) = rope_table(&ids, &c.axes_dims_rope, c.rope_theta);
    let rope = g.from_f32(&rope, &s, Device::Cpu, DataType::F32);
    let x = ops::concat(&[&img, &r], 1);
    let ya = a.forward(&x, &txt, &a.prepare_step(&[700.0], None), &rope).to_f32_vec();
    let yb = b.forward(&x, &txt, &b.prepare_step(&[700.0], None), &rope).to_f32_vec();
    println!("plain forward ({nd} double + {ns} single blocks): rel {:.5}", rel(&ya, &yb));
    // KV variant: [txt, ref, img].
    let mut ids = text_ids(lt);
    ids.extend(image_ids(rh, rw, 10.0));
    ids.extend(image_ids(h, w, 0.0));
    let (rope, s) = rope_table(&ids, &c.axes_dims_rope, c.rope_theta);
    let rope = g.from_f32(&rope, &s, Device::Cpu, DataType::F32);
    let x = ops::concat(&[&r, &img], 1);
    let n = rh * rw;
    let ya = a.forward_kv(&x, &txt, &a.prepare_step(&[700.0], None), &a.prepare_step(&[0.0], None), n, &rope).to_f32_vec();
    let yb = b.forward_kv(&x, &txt, &b.prepare_step(&[700.0], None), &b.prepare_step(&[0.0], None), n, &rope).to_f32_vec();
    println!("KV forward: rel {:.5}", rel(&ya, &yb));
    Ok(())
}
