# Provenance

Tonic Clap is a clean-room reimplementation released under the BSD 3-Clause license.

## What it is based on

| Area | Source |
|------|--------|
| Compute backend | ccv / nnc by Liu Liu (BSD-3), vendored unchanged except as listed in `third_party/ccv/README.md` |
| Safe nnc wrapper (`crates/nnc`) | design ideas from s4nnc (BSD-3, Liu Liu) |
| FLUX.2 transformer, VAE, pipeline, scheduler | public architecture description and Apache-2.0 reference code (Black Forest Labs, Hugging Face diffusers) |
| Qwen3 text encoder | Apache-2.0 reference code (Hugging Face transformers) |
| Seeded noise | PyTorch's (BSD) CPU normal-sampling algorithm, verified against `torch.randn` |
| Tensor compression | LLNL fpzip (BSD-3), vendored |

## Interface compatibility

The command line flags, the gRPC service definition and messages, the
`GenerationConfiguration` FlatBuffers layout and the image tensor wire format are
functional interfaces of the Draw Things software. They were documented in
[INTERFACE.md](INTERFACE.md) from their observable/declared form (flag names, protocol
field names and numbers, schema field order, byte layout) so that existing clients keep
working; no implementation code from Draw Things was used.

## What was deliberately not reused

Draw Things' own model definitions, weight naming, `.ckpt` model files, model zoo,
samplers and generation pipeline are not reproduced. Models are loaded from the official
Hugging Face safetensors releases instead.
