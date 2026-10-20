mod common;
use common::*;
use nnc::{DataType, Device, Graph};
use tc_core::models::qwen3::Qwen3;
use tc_core::weights::SafeTensorsFiles;

#[test]
fn qwen3_hidden_states_match_reference() {
    let g = Graph::new();
    let model = Qwen3::load(&g, &fixture("qwen3/text_encoder"), 4, Device::Cpu, DataType::F32, false).unwrap();
    let io = SafeTensorsFiles::open(&[fixture("qwen3/qwen3_io.safetensors")]).unwrap();
    let ids: Vec<i32> = io.read_f32("ids").unwrap().iter().map(|v| *v as i32).collect();
    let out = model.hidden_states(&ids, 7, &[1, 2, 4]).unwrap();
    assert_eq!(out.shape(), vec![1, 12, 96]);
    assert_close(&out.to_f32_vec(), &io.read_f32("out").unwrap(), 1e-4, "qwen3");
}
