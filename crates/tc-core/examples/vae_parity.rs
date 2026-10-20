//! `cargo run --release -p tc-core --example vae_parity -- <flux_2_vae_f16.ckpt> <diffusers vae dir or safetensors>`
//! Encodes and decodes a test image with both VAEs (CPU, f32) and prints the differences.
use nnc::{DataType, Device, Graph};
use tc_core::models::flux2_vae::Flux2Vae;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let g = Graph::new();
    let a = Flux2Vae::load(&g, std::path::Path::new(&args[0]), Device::Cpu, DataType::F32)?;
    let b = Flux2Vae::load(&g, std::path::Path::new(&args[1]), Device::Cpu, DataType::F32)?;
    let (h, w) = (128usize, 192usize);
    let img: Vec<f32> = (0..h * w * 3).map(|i| (((i / 3) % w) as f32 / w as f32 * 2.0 - 1.0) * (((i / 3 / w) as f32 * 0.07).sin())).collect();
    let x = g.from_f32(&img, &[1, h, w, 3], Device::Cpu, DataType::F32);
    let rel = |p: &[f32], q: &[f32]| {
        let d: f64 = p.iter().zip(q).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
        let n: f64 = q.iter().map(|y| (*y as f64).powi(2)).sum();
        (d / n).sqrt()
    };
    let (ea, eb) = (a.pack_latents(&a.encode_raw(&x)), b.pack_latents(&b.encode_raw(&x)));
    println!("packed latents {:?}: rel {:.5}", ea.shape(), rel(&ea.to_f32_vec(), &eb.to_f32_vec()));
    let (da, db) = (a.decode_raw(&a.unpack_latents(&eb)), b.decode_raw(&b.unpack_latents(&eb)));
    println!("decoded image {:?}: rel {:.5}", da.shape(), rel(&da.to_f32_vec(), &db.to_f32_vec()));
    Ok(())
}
