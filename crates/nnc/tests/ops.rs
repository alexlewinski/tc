//! Numerical checks of nnc ops against PyTorch references (see scripts/gen_op_fixtures.py).

use nnc::{ops, DataType, Device, Graph, HostTensor, Var};
use safetensors::SafeTensors;

struct Fixtures {
    bytes: Vec<u8>,
}

impl Fixtures {
    fn load() -> Fixtures {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ops.safetensors");
        Fixtures { bytes: std::fs::read(path).unwrap() }
    }
    fn get(&self, name: &str) -> HostTensor {
        let st = SafeTensors::deserialize(&self.bytes).unwrap();
        let t = st.tensor(name).unwrap_or_else(|_| panic!("missing {name}"));
        HostTensor::from_bytes(t.data().to_vec(), DataType::F32, t.shape())
    }
    fn var(&self, g: &Graph, name: &str) -> Var {
        g.from_host(&self.get(name), device())
    }
}

fn device() -> Device {
    match std::env::var("NNC_TEST_DEVICE").as_deref() {
        Ok("gpu") => Device::Gpu(0),
        _ => Device::Cpu,
    }
}

#[track_caller]
fn check(fx: &Fixtures, out: &Var, name: &str, tol: f32) {
    let expected = fx.get(name);
    assert_eq!(out.shape(), expected.shape(), "{name}: shape");
    let got = out.to_f32_vec();
    let exp = expected.to_f32_vec();
    let mut max_err = 0f32;
    for (a, b) in got.iter().zip(exp.iter()) {
        max_err = max_err.max((a - b).abs());
    }
    assert!(max_err <= tol, "{name}: max abs error {max_err} > {tol}");
}

#[test]
fn linear_and_matmul() {
    let fx = Fixtures::load();
    let g = Graph::new();
    let x = fx.var(&g, "linear.x");
    let w = fx.var(&g, "linear.w");
    let b = fx.var(&g, "linear.b");
    check(&fx, &ops::linear(&x, &w, Some(&b)), "linear.out", 1e-4);
    check(&fx, &ops::linear(&x, &w, None), "linear_nobias.out", 1e-4);
    let a = fx.var(&g, "bmm.a");
    let c = fx.var(&g, "bmm.c");
    check(&fx, &ops::matmul(&a, &c, None, Some((2, 3))), "bmm.out", 1e-4);
}

#[test]
fn broadcast_binary() {
    let fx = Fixtures::load();
    let g = Graph::new();
    let p = fx.var(&g, "bc.p");
    let q = fx.var(&g, "bc.q");
    check(&fx, &ops::add(&p, &q), "bc.add", 1e-5);
    check(&fx, &ops::mul(&p, &q), "bc.mul", 1e-5);
    check(&fx, &ops::sub(&p, &q), "bc.sub", 1e-5);
    // Lower rank operand gets broadcast too.
    let q2 = q.reshape(&[6]);
    check(&fx, &ops::add(&p, &q2), "bc.add", 1e-5);
}

#[test]
fn norms() {
    let fx = Fixtures::load();
    let g = Graph::new();
    let x = fx.var(&g, "ln.x");
    check(&fx, &ops::layer_norm(&x, &[2], 1e-6, None), "ln.out", 1e-4);
    let gm = fx.var(&g, "ln.g");
    let bt = fx.var(&g, "ln.b");
    check(&fx, &ops::layer_norm(&x, &[2], 1e-5, Some((&gm, &bt))), "ln_affine.out", 1e-4);
    check(&fx, &ops::rms_norm(&x, &[2], 1e-6, Some(&gm)), "rms.out", 1e-4);
    let xg = fx.var(&g, "gn.x");
    let gg = fx.var(&g, "gn.g");
    let gb = fx.var(&g, "gn.b");
    check(&fx, &ops::group_norm(&xg, 3, 8, &[1, 2], 1e-6, Some((&gg, &gb))), "gn.out", 1e-4);
}

#[test]
fn attention() {
    let fx = Fixtures::load();
    let g = Graph::new();
    let q = fx.var(&g, "attn.q");
    let k = fx.var(&g, "attn.k");
    let v = fx.var(&g, "attn.v");
    check(&fx, &ops::scaled_dot_product_attention(&q, &k, &v, 0.3, false, None), "attn.out", 1e-4);
    let q = fx.var(&g, "attnc.q");
    let k = fx.var(&g, "attnc.k");
    let v = fx.var(&g, "attnc.v");
    let s = 1.0 / (8f32).sqrt();
    check(&fx, &ops::scaled_dot_product_attention(&q, &k, &v, s, true, None), "attnc.out", 1e-4);
    let m = fx.var(&g, "attnm.mask");
    check(&fx, &ops::scaled_dot_product_attention(&q, &k, &v, s, false, Some(&m)), "attnm.out", 1e-4);
}

#[test]
fn activations() {
    let fx = Fixtures::load();
    let g = Graph::new();
    let x = fx.var(&g, "act.x");
    check(&fx, &ops::silu(&x), "act.silu", 1e-5);
    check(&fx, &ops::gelu(&x, false), "act.gelu", 1e-4);
    check(&fx, &ops::gelu(&x, true), "act.gelu_tanh", 1e-4);
    check(&fx, &ops::softmax(&x), "act.softmax", 1e-5);
    check(&fx, &ops::sigmoid(&x), "act.sigmoid", 1e-5);
    check(&fx, &ops::exp(&x), "act.exp", 1e-4);
}

/// Fixtures store CPU-layout weights `[O, kh, kw, I]`; GPU kernels want `[O, I, kh, kw]` (NCHW).
fn conv_weight(fx: &Fixtures, g: &Graph, name: &str) -> Var {
    let h = fx.get(name);
    if device() == Device::Cpu {
        return g.from_host(&h, Device::Cpu);
    }
    let s = h.shape().to_vec();
    let (o, kh, kw, i) = (s[0], s[1], s[2], s[3]);
    let v = h.to_f32_vec();
    let mut out = vec![0f32; v.len()];
    for a in 0..o {
        for b in 0..i {
            for y in 0..kh {
                for x in 0..kw {
                    out[((a * i + b) * kh + y) * kw + x] = v[((a * kh + y) * kw + x) * i + b];
                }
            }
        }
    }
    g.from_host_with_format(&HostTensor::from_f32(&out, &[o, i, kh, kw]), device(), nnc::Format::Nchw)
}

#[test]
fn convolution_and_upsample() {
    let fx = Fixtures::load();
    let g = Graph::new();
    let x = fx.var(&g, "conv.x");
    let w3 = conv_weight(&fx, &g, "conv.w3");
    let b3 = fx.var(&g, "conv.b3");
    check(&fx, &ops::conv2d(&x, &w3, Some(&b3), (1, 1), (1, 1, 1, 1), 1), "conv.out3", 1e-4);
    check(&fx, &ops::conv2d(&x, &w3, Some(&b3), (2, 2), (0, 1, 0, 1), 1), "conv.out3s2", 1e-4);
    let w1 = conv_weight(&fx, &g, "conv.w1");
    let b1 = fx.var(&g, "conv.b1");
    check(&fx, &ops::conv2d(&x, &w1, Some(&b1), (1, 1), (0, 0, 0, 0), 1), "conv.out1", 1e-4);
    check(&fx, &ops::upsample(&x, 2.0, 2.0, false), "up.out", 0.0);
}

#[test]
fn complex_multiply() {
    let fx = Fixtures::load();
    let g = Graph::new();
    let a = fx.var(&g, "cmul.a");
    let b = fx.var(&g, "cmul.b");
    check(&fx, &ops::cmul(&a, &b), "cmul.out", 1e-5);
}

#[test]
fn views_and_reductions() {
    let fx = Fixtures::load();
    let g = Graph::new();
    let x = fx.var(&g, "perm.x");
    check(&fx, &x.permute(&[0, 2, 1, 3]).contiguous(), "perm.out", 0.0);
    check(&fx, &ops::transpose(&x, 1, 2), "perm.out", 0.0);
    check(&fx, &x.narrow(1, 1, 2).contiguous(), "narrow.out", 0.0);
    let x2 = ops::scale(&x, 2.0);
    check(&fx, &ops::concat(&[&x, &x2], 2), "cat.out", 1e-6);
    check(&fx, &ops::reduce_mean(&x, &[3]), "mean.out", 1e-5);
    check(&fx, &ops::reduce_sum(&x, &[1]), "sum.out", 1e-5);
    // narrow of a permuted view, then reshape (forces a copy)
    let y = x.permute(&[0, 2, 1, 3]).reshape(&[2, 4, 15]);
    assert_eq!(y.shape(), vec![2, 4, 15]);
}

#[test]
fn index_select_and_dtypes() {
    let fx = Fixtures::load();
    let g = Graph::new();
    let table = fx.var(&g, "idx.table");
    let idx = g.from_host(&HostTensor::from_i32(&[3, 0, 9, 3], &[4]), device());
    check(&fx, &ops::index_select(&table, &idx), "idx.out", 0.0);
    let half = table.to_dtype(DataType::F16);
    assert_eq!(half.dtype(), DataType::F16);
    check(&fx, &half.to_dtype(DataType::F32), "idx.table", 1e-2);
    let bf = table.to_dtype(DataType::BF16);
    check(&fx, &bf.to_dtype(DataType::F32), "idx.table", 3e-2);
    // f16 compute path for linear (the CPU GEMM kernels are FP32-only).
    if device() == Device::Cpu {
        return;
    }
    let x = fx.var(&g, "linear.x").to_dtype(DataType::F16);
    let w = fx.var(&g, "linear.w").to_dtype(DataType::F16);
    let b = fx.var(&g, "linear.b").to_dtype(DataType::F16);
    check(&fx, &ops::linear(&x, &w, Some(&b)), "linear.out", 2e-2);
}

#[test]
fn palettize_roundtrip() {
    let n = 40000;
    let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();
    let host = HostTensor::from_f32(&data, &[200, 200]).to_dtype(DataType::F16);
    let q = nnc::quant::palettize(&host, 8, nnc::quant::DEFAULT_BLOCK);
    assert_eq!(q.len(), 3 * 256 * 2 + n);
    let back = nnc::quant::depalettize(&q, DataType::F16, &[200, 200], 8, nnc::quant::DEFAULT_BLOCK).to_f32_vec();
    let max = data.iter().zip(back.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(max < 0.01, "palettize error {max}");
    let g = Graph::new();
    let v = g.from_palettized(&q, DataType::F16, &[200, 200], 8, nnc::quant::DEFAULT_BLOCK, Device::Cpu);
    assert!(v.is_quantized());
    assert_eq!(v.shape(), vec![200, 200]);
}

/// The f16 kernels the VAE uses on GPU (group norm with scale/bias, layer / RMS norm).
/// GPU only: nnc's CPU kernels are f32.
#[test]
fn norms_f16_gpu() {
    if !device().is_gpu() {
        return;
    }
    let fx = Fixtures::load();
    let g = Graph::new();
    let h = |name: &str| fx.var(&g, name).to_dtype(DataType::F16);
    let x = h("ln.x");
    let (gm, bt) = (h("ln.g"), h("ln.b"));
    check(&fx, &ops::layer_norm(&x, &[2], 1e-5, Some((&gm, &bt))).to_dtype(DataType::F32), "ln_affine.out", 2e-2);
    check(&fx, &ops::rms_norm(&x, &[2], 1e-6, Some(&gm)).to_dtype(DataType::F32), "rms.out", 2e-2);
    let (xg, gg, gb) = (h("gn.x"), h("gn.g"), h("gn.b"));
    check(&fx, &ops::group_norm(&xg, 3, 8, &[1, 2], 1e-6, Some((&gg, &gb))).to_dtype(DataType::F32), "gn.out", 2e-2);
}
