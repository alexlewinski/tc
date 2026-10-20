use nnc::{DataType, HostTensor};
fn main() {
    let n = 16384 * 8;
    let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0137).sin() * 0.05 + ((i * 7919 % 1000) as f32 - 500.0) * 1e-4).collect();
    let host = HostTensor::from_f32(&data, &[n]).to_dtype(DataType::F16);
    let t = std::time::Instant::now();
    let q = nnc::quant::palettize(&host, 8, 16384);
    println!("kmeans 8 blocks: {:?} ({} bytes)", t.elapsed(), q.len());
    let t = std::time::Instant::now();
    let f = nnc::quant::palettize8_fast(&host, 16384);
    println!("fast 8 blocks: {:?} ({} bytes)", t.elapsed(), f.len());
    let err = |q: &[u8]| {
        let back = nnc::quant::depalettize(q, DataType::F16, &[n], 8, 16384).to_f32_vec();
        let h = host.to_f32_vec();
        let mse: f64 = h.iter().zip(&back).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>() / n as f64;
        mse.sqrt()
    };
    println!("rmse kmeans {:e} fast {:e}", err(&q), err(&f));
}
