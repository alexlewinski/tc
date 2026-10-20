#![allow(dead_code)]
use nnc::{DataType, Device, Graph, Var};
use tc_core::weights::SafeTensorsFiles;

pub fn fixture(path: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(path)
}

/// Device under test: `NNC_TEST_DEVICE=gpu` runs the model tests on the GPU (in f32).
pub fn device() -> Device {
    match std::env::var("NNC_TEST_DEVICE").as_deref() {
        Ok("gpu") => Device::Gpu(0),
        _ => Device::Cpu,
    }
}

pub fn load(g: &Graph, files: &SafeTensorsFiles, name: &str) -> Var {
    g.from_host(&files.read(name, DataType::F32).unwrap(), device())
}

#[track_caller]
pub fn assert_close(got: &[f32], exp: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), exp.len(), "{what}: length");
    let mut max_err = 0f32;
    let mut max_ref = 0f32;
    for (a, b) in got.iter().zip(exp) {
        max_err = max_err.max((a - b).abs());
        max_ref = max_ref.max(b.abs());
    }
    eprintln!("{what}: max abs err {max_err:e} (max |ref| {max_ref})");
    assert!(max_err <= tol * max_ref.max(1.0), "{what}: max abs error {max_err} too large");
}
