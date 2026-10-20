//! Tonic Clap inference core.
//!
//! Clean-room implementation of FLUX.2 text-to-image inference on top of nnc: weight loading
//! from safetensors, LoRA merging, the FLUX.2 transformer / Qwen3 text encoder / FLUX.2 VAE,
//! flow-matching sampling and the end-to-end generation pipeline.

pub mod lora;
pub mod models;
pub mod weights;
