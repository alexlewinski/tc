mod common;
use common::*;
use nnc::{DataType, Device, Graph};
use tc_core::models::flux2_vae::Flux2Vae;
use tc_core::weights::SafeTensorsFiles;

#[test]
fn vae_decode_encode_pack_match_reference() {
    let g = Graph::new();
    let vae = Flux2Vae::load(&g, &fixture("vae/vae"), device(), DataType::F32).unwrap();
    let io = SafeTensorsFiles::open(&[fixture("vae/vae_io.safetensors")]).unwrap();
    let z = load(&g, &io, "z");
    let dec = vae.decode_raw(&z);
    assert_eq!(dec.shape(), vec![1, 12, 16, 3]);
    assert_close(&dec.to_f32_vec(), &io.read_f32("dec").unwrap(), 1e-4, "decode");
    let img = load(&g, &io, "img");
    let enc = vae.encode_raw(&img);
    assert_close(&enc.to_f32_vec(), &io.read_f32("enc").unwrap(), 1e-4, "encode");
    let packed = vae.pack_latents(&enc);
    assert_eq!(packed.shape(), vec![1, 4, 6, 16]);
    assert_close(&packed.to_f32_vec(), &io.read_f32("packed").unwrap(), 1e-4, "pack");
    let back = vae.unpack_latents(&packed);
    assert_close(&back.to_f32_vec(), &io.read_f32("enc").unwrap(), 1e-4, "unpack");
}
