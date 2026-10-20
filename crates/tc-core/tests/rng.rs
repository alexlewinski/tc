mod common;
use tc_core::rng::TorchCpuRng;
use tc_core::weights::SafeTensorsFiles;

#[test]
fn matches_torch_randn() {
    let f = SafeTensorsFiles::open(&[common::fixture("randn.safetensors")]).unwrap();
    for (seed, n) in [(0u64, 7usize), (42, 16), (1234, 37), (7, 1000), ((1u64 << 31) + 5, 4099)] {
        let exp = f.read_f32(&format!("{seed}_{n}")).unwrap();
        let got = TorchCpuRng::new(seed).randn(n);
        common::assert_close(&got, &exp, 1e-6, &format!("seed {seed} n {n}"));
    }
}
