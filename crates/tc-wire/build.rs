// Builds the vendored LLNL fpzip library (BSD-3, third_party/fpzip).
fn main() {
    let root = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("../../third_party/fpzip");
    println!("cargo:rerun-if-changed={}", root.display());
    let mut b = cc::Build::new();
    b.cpp(true)
        .include(root.join("include"))
        .include(root.join("src"))
        .define("FPZIP_FP", "FPZIP_FP_FAST")
        .define("FPZIP_BLOCK_SIZE", "0x1000")
        .flag_if_supported("-std=c++11")
        .warnings(false);
    for f in ["error.cpp", "rcdecoder.cpp", "rcencoder.cpp", "rcqsmodel.cpp", "read.cpp", "version.cpp", "write.cpp"] {
        b.file(root.join("src").join(f));
    }
    b.compile("fpzip");
}
