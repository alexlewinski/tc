fn main() {
    let proto = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("../../proto");
    println!("cargo:rerun-if-changed={}", proto.display());
    let fds = protox::compile([proto.join("image_service.proto")], [&proto]).expect("compiling proto");
    tonic_build::configure().build_server(true).build_client(true).compile_fds(fds).expect("tonic codegen");
}
