//! Builds the vendored ccv / nnc library (third_party/ccv) with its own makefiles.
//!
//! We do not run ccv's `configure` script: it auto-detects whatever happens to be
//! installed (libpng, FFTW, GSL, ...), and some of those are GPL. Instead we write
//! `config.mk` ourselves with a small, explicit set of features per platform:
//!
//! * macOS (Apple Silicon or Intel): Accelerate + Metal (MPS) + Metal FlashAttention,
//!   libdispatch for CPU parallelism.
//! * Linux: plain CPU build (optionally OpenBLAS, optionally CUDA).

use std::env;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let src_root = manifest_dir.join("../../third_party/ccv/lib");
    let src_root = src_root
        .canonicalize()
        .unwrap_or_else(|_| panic!("vendored ccv not found at {}", src_root.display()));
    println!("cargo:rerun-if-changed={}", src_root.display());
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CC");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=NNC_SYS_DEBUG");

    #[cfg(feature = "regenerate-bindings")]
    regenerate_bindings(&src_root, &manifest_dir);

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let build_root = out_dir.join("ccv/lib");
    sync_tree(&src_root, &build_root);

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let cc = env::var("CC").ok().unwrap_or_else(|| {
        if Command::new("clang").arg("--version").output().is_ok() {
            "clang".to_string()
        } else {
            "cc".to_string()
        }
    });

    let mut defines: Vec<&str> = vec!["-D HAVE_PTHREAD"];
    let mut cflags: Vec<String> = vec![];
    let mut ldflags: Vec<String> = vec!["-lm".into(), "-lpthread".into()];
    let mut mps = false;
    let mut cuda = false;

    if target_os == "macos" || target_os == "ios" {
        mps = true;
        defines.push("-D HAVE_ACCELERATE_FRAMEWORK");
        defines.push("-D HAVE_MPS");
        defines.push("-D USE_DISPATCH");
        cflags.push("-fblocks".into());
        if let Ok(target) = env::var("MACOSX_DEPLOYMENT_TARGET") {
            cflags.push(format!("-mmacosx-version-min={target}"));
        } else if target_os == "macos" {
            // MPSGraph APIs used by nnc need at least macOS 13.
            cflags.push("-mmacosx-version-min=13.0".into());
        }
    } else {
        if target_arch == "x86_64" {
            defines.push("-D HAVE_SSE2");
            cflags.push("-msse2".into());
        }
        if target_os == "linux" {
            ldflags.push("-lrt".into());
        }
        if cfg!(feature = "openblas") {
            defines.push("-D HAVE_CBLAS");
            ldflags.push("-lopenblas".into());
        }
        if cfg!(feature = "cuda") {
            cuda = true;
        }
    }
    if cuda {
        defines.push("-D HAVE_CUDA");
    }
    let debug = env::var("NNC_SYS_DEBUG").map(|v| v == "1").unwrap_or(false);
    if debug {
        cflags.push("-g".into());
    }
    // Position independent code so the static library can be linked into anything.
    cflags.push("-fPIC".into());

    let cuda_prefix = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    let mut config = String::new();
    config.push_str(&format!("CC := {cc}\n"));
    config.push_str("AR := ar\n");
    if cuda {
        config.push_str(&format!("NVCC := {cuda_prefix}/bin/nvcc\n"));
        config.push_str("CUDA_SRCS := \n");
        config.push_str("CUDA_COMPAT_LIB := gpu/libnnc-compat-cuda.o\n");
        config.push_str("CUDA_COMPAT_SM80_LIB := \n");
        config.push_str("CUDA_CMD_LIB := libnnc-cmd-cuda.o\n");
    } else {
        config.push_str("NVCC := \nCUDA_SRCS := \nCUDA_COMPAT_LIB := \nCUDA_COMPAT_SM80_LIB := \nCUDA_CMD_LIB := \n");
    }
    if mps {
        config.push_str("MFA_COMPAT_LIB := mfa/libnnc-compat-mfa.o\n");
        config.push_str("MPS_COMPAT_LIB := mps/libnnc-compat-mps.o\n");
        config.push_str("MPS_CMD_LIB := libnnc-cmd-mps.o\n");
    } else {
        config.push_str("MFA_COMPAT_LIB := \nMPS_COMPAT_LIB := \nMPS_CMD_LIB := \n");
    }
    config.push_str(&format!("DEFINE_MACROS := {}\n", defines.join(" ")));
    config.push_str("prefix := /usr/local\nexec_prefix := ${prefix}\n");
    let mut all_cflags = cflags.join(" ");
    if cuda {
        all_cflags.push_str(&format!(" -I{cuda_prefix}/include"));
    }
    config.push_str(&format!("CFLAGS := {all_cflags} $(DEFINE_MACROS)\n"));
    config.push_str("NVFLAGS := --use_fast_math -arch=sm_70 -std=c++14 $(DEFINE_MACROS)\n");
    config.push_str(&format!("LDFLAGS := {}\n", ldflags.join(" ")));
    write_if_changed(&build_root.join("config.mk"), config.as_bytes());

    let jobs = env::var("NUM_JOBS").unwrap_or_else(|_| "4".into());
    let status = Command::new("make")
        .current_dir(&build_root)
        .arg(format!("-j{jobs}"))
        .arg("libccv.a")
        .env_remove("CFLAGS")
        .env_remove("LDFLAGS")
        .env_remove("MAKEFLAGS")
        .env_remove("MFLAGS")
        .status()
        .expect("failed to run make (is GNU make installed?)");
    if !status.success() {
        panic!("building vendored ccv failed");
    }

    println!("cargo:rustc-link-search=native={}", build_root.display());
    println!("cargo:rustc-link-lib=static=ccv");
    println!("cargo:include={}", build_root.display());
    if mps {
        for fw in [
            "Accelerate",
            "Metal",
            "MetalPerformanceShaders",
            "MetalPerformanceShadersGraph",
            "CoreVideo",
            "CoreML",
            "IOSurface",
            "Foundation",
        ] {
            println!("cargo:rustc-link-lib=framework={fw}");
        }
        if target_os == "macos" {
            println!("cargo:rustc-link-lib=framework=IOKit");
        }
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=objc");
    } else {
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=pthread");
        if target_os == "linux" {
            println!("cargo:rustc-link-lib=rt");
        }
        if cfg!(feature = "openblas") {
            println!("cargo:rustc-link-lib=openblas");
        }
        if cuda {
            println!("cargo:rustc-link-search=native={cuda_prefix}/lib64");
            for lib in ["cudart", "cublas", "curand", "cufile", "stdc++"] {
                println!("cargo:rustc-link-lib={lib}");
            }
        }
    }
}

/// Mirror `src` into `dst`, only touching files whose size or mtime changed so that
/// make's incremental rebuilds keep working. `*.inc.gz` files are decompressed in place
/// (they hold precompiled Metal kernels that are too large to vendor uncompressed).
fn sync_tree(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).unwrap();
    for entry in fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        let name = entry.file_name();
        let ty = entry.file_type().unwrap();
        if ty.is_dir() {
            sync_tree(&path, &dst.join(&name));
            continue;
        }
        let name_str = name.to_string_lossy();
        if let Some(stem) = name_str.strip_suffix(".gz") {
            let target = dst.join(stem);
            if is_stale(&path, &target) {
                let mut data = Vec::new();
                flate2::read::GzDecoder::new(fs::File::open(&path).unwrap())
                    .read_to_end(&mut data)
                    .unwrap();
                fs::File::create(&target).unwrap().write_all(&data).unwrap();
            }
            continue;
        }
        let target = dst.join(&name);
        if is_stale(&path, &target) {
            fs::copy(&path, &target).unwrap();
            // Preserve executability (configure-generated scripts etc.).
        }
    }
}

fn is_stale(src: &Path, dst: &Path) -> bool {
    let Ok(dm) = fs::metadata(dst) else { return true };
    let sm = fs::metadata(src).unwrap();
    match (sm.modified(), dm.modified()) {
        (Ok(s), Ok(d)) => s > d,
        _ => true,
    }
}

fn write_if_changed(path: &Path, contents: &[u8]) {
    if let Ok(existing) = fs::read(path) {
        if existing == contents {
            return;
        }
    }
    fs::write(path, contents).unwrap();
}

#[cfg(feature = "regenerate-bindings")]
fn regenerate_bindings(src_root: &Path, manifest_dir: &Path) {
    let bindings = bindgen::Builder::default()
        .header(manifest_dir.join("wrapper.h").to_string_lossy())
        .clang_arg(format!("-I{}", src_root.display()))
        .allowlist_function("ccv_nnc_.*|ccv_cnnp_.*|ccv_array_.*|ccv_cli_.*|ccv_nnc_hint_auto")
        .allowlist_type("ccv_nnc_.*|ccv_cnnp_.*|ccv_array_t")
        .allowlist_var("CCV_.*")
        .derive_default(true)
        .derive_debug(false)
        .layout_tests(true)
        .prepend_enum_name(false)
        .generate()
        .expect("bindgen failed");
    bindings
        .write_to_file(manifest_dir.join("src/bindings.rs"))
        .unwrap();
}
