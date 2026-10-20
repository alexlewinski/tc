//! Safe Rust interface to the vendored nnc library.
//!
//! The design follows the eager "dynamic graph" style of nnc: a [`Graph`] owns tensor
//! variables ([`Var`]); every op executes immediately on the device the inputs live on
//! (CPU, Metal on macOS, or CUDA when built with it) and returns new variables. Gradients
//! are disabled by default since this crate is used for inference.
//!
//! Parts of this crate are a Rust port of ideas from s4nnc (BSD 3-Clause,
//! Copyright (c) 2020, Liu Liu).

mod graph;
mod host;
pub mod ops;
pub mod quant;
mod types;

pub use graph::{Graph, Var};
pub use host::HostTensor;
pub use types::{DataType, Device, Error, Format, Result};

pub use nnc_sys as sys;

use std::sync::Once;

static INIT: Once = Once::new();

/// Initialize nnc (idempotent). Called automatically by [`Graph::new`].
pub fn init() {
    INIT.call_once(|| unsafe {
        sys::ccv_nnc_init();
    });
}

/// Number of GPUs (Metal devices on macOS, CUDA devices on Linux) visible to nnc.
pub fn gpu_count() -> usize {
    init();
    unsafe { sys::ccv_nnc_device_count(sys::CCV_STREAM_CONTEXT_GPU as i32).max(0) as usize }
}

/// The best available device: GPU 0 when present, CPU otherwise.
pub fn default_device() -> Device {
    if gpu_count() > 0 {
        Device::Gpu(0)
    } else {
        Device::Cpu
    }
}

/// Set nnc's log level (0 = none, 1 = errors, 2 = info, 3 = verbose).
pub fn set_log_level(level: u32) {
    let level = match level {
        0 => sys::CCV_CLI_NONE,
        1 => sys::CCV_CLI_ERROR,
        2 => sys::CCV_CLI_INFO,
        _ => sys::CCV_CLI_VERBOSE,
    };
    unsafe {
        let levels = sys::ccv_cli_output_level_and_above(level as i32);
        sys::ccv_cli_set_output_levels(levels);
    }
}

/// Enable or disable nnc's Metal FlashAttention kernels (macOS). Enabled by default.
pub fn set_flash_attention(enabled: bool) {
    let flag = sys::CCV_NNC_DISABLE_MFA_ATTENTION as u64;
    unsafe {
        if enabled {
            sys::ccv_nnc_disable_flag(flag);
        } else {
            sys::ccv_nnc_enable_flag(flag);
        }
    }
}
