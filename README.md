# Tonic Clap

Fast local **FLUX.2** image generation in Rust, primarily for **Apple Silicon Macs**, with:

* **`draw-things-cli`** — a command line generator (clap), command-compatible with the
  Draw Things CLI for image generation.
* **`gRPCServerCLI`** — a gRPC server (tonic) that speaks the Draw Things image generation
  protocol, so other software (or the Draw Things app) can drive generation remotely.

The compute backend is [ccv / nnc](https://github.com/liuliu/ccv) (vendored C, BSD-3) with its
Metal (MPS + Metal FlashAttention) backend on macOS and a CPU backend elsewhere.

Supported models: **FLUX.2 [klein] 9B / 4B** (distilled and base) from the official Hugging
Face releases, with any number of LoRAs (diffusers/PEFT, ai-toolkit/BFL and kohya formats).

> Tonic Clap is an independent clean-room implementation, released under the BSD 3-Clause
> license. It is not affiliated with Draw Things or Black Forest Labs. See
> [docs/PROVENANCE.md](docs/PROVENANCE.md).

## Quick start (macOS, Apple Silicon)

```sh
# Requirements: a recent Xcode (the macOS 15+ SDK; nnc uses newer Core ML / Metal APIs), Rust (https://rustup.rs)
cargo build --release -p draw-things-cli -p grpc-server-cli

# Download FLUX.2 [klein] 9B (gated on Hugging Face: accept the license and set a token)
export HF_TOKEN=hf_...
./target/release/draw-things-cli models ensure --model flux_2_klein_9b

# Generate with two LoRAs (files placed in the models directory)
./target/release/draw-things-cli generate \
  --model flux_2_klein_9b \
  --prompt "a red cube on a wooden table, studio light" \
  --width 1024 --height 1024 --seed 42 \
  --lora my_style.safetensors:0.8 --lora my_subject.safetensors:0.6 \
  -o cube.png

# Edit / reference-guided generation: the same, plus a reference image
./target/release/draw-things-cli generate \
  --model flux_2_klein_9b \
  --prompt "the subject from the reference image sitting on a wooden table, studio light" \
  --image reference.png \
  --width 1024 --height 1024 --seed 42 \
  --lora my_style.safetensors:0.8 --lora my_subject.safetensors:0.6 \
  -o edit.png
```

`--image` adds a reference image (repeat it for several). With the default `--strength 1` it
conditions FLUX.2's editing path and the output starts from noise; with `--strength` below 1
the first image instead initializes the latent (image-to-image). Without `--width`/`--height`
the output size follows the first image (rounded to multiples of 64).

The models directory defaults to `$DRAWTHINGS_MODELS_DIR`, then `<binary dir>/Models`, then
`~/Documents/Models`; override with `--models-dir`. A model is a Hugging Face snapshot in
diffusers layout (`model_index.json`, `transformer/`, `text_encoder/`, `tokenizer/`, `vae/`,
`scheduler/`), stored as `<models-dir>/FLUX.2-klein-9B/` or found in the Hugging Face cache.

## Serving over gRPC

```sh
./target/release/gRPCServerCLI ~/Documents/Models --port 7859 --shared-secret mysecret
```

The server listens with TLS (self-signed certificate) unless `--no-tls`, advertises itself
over Bonjour as `_dt-grpc._tcp`, and accepts `GenerateImage` requests whose configuration
names the model and LoRAs:

```sh
# Another machine, using this CLI as the client:
draw-things-cli generate --remote --remote-url 192.168.1.20 --no-remote-tls \
  --remote-shared-secret mysecret --model flux_2_klein_9b --prompt "a red cube" \
  --config-json '{"loras":[{"file":"my_style.safetensors","weight":0.8},{"file":"my_subject.safetensors","weight":0.6}]}' \
  --image reference.png \
  -o cube.png
```

The client sends the first `--image` as the request's canvas image when its aspect ratio
matches the output, otherwise as a `shuffle` hint (further images always go as hints); with
`strength` 1 the server uses either as a FLUX.2 reference image. See
[docs/INTERFACE.md](docs/INTERFACE.md).

Writing your own client: see [docs/INTERFACE.md](docs/INTERFACE.md) for the protocol
(`proto/image_service.proto`), the `GenerationConfiguration` FlatBuffer, and the image tensor
format. `crates/tc-grpc/src/client.rs` is a compact reference client.

## Precision and memory

| Component | GPU default | CPU |
|-----------|-------------|-----|
| FLUX.2 transformer | bf16 (`--precision f16` is faster on M1/M2, may overflow) | f32 |
| Qwen3 text encoder | bf16 (`--text-encoder-precision`) | f32 |
| VAE | f16 | f32 |

FLUX.2 [klein] 9B needs roughly 18 GB for the transformer plus ~13 GB while the text encoder
runs (it is unloaded after encoding unless `--keep-text-encoder`). LoRAs are merged into the
weights when a request needs them and un-merged when it doesn't, so switching LoRAs between
requests is cheap.

`--quantize` stores the transformer and text-encoder linear weights 8-bit palettized (about
half the memory, so 9B fits on 16 GB Macs). On GPU it forces f16 for those components, since
Metal's palettized kernels have no bf16 variant; LoRAs then run as separate low-rank branches.
bf16 on Metal needs macOS 15 or later.

## Repository layout

| Path | What |
|------|------|
| `third_party/ccv` | vendored ccv / nnc (BSD-3) |
| `third_party/fpzip` | vendored LLNL fpzip (BSD-3), used by the tensor wire format |
| `crates/nnc-sys` | builds nnc with its makefiles; raw FFI |
| `crates/nnc` | safe tensors / ops over nnc's dynamic graph |
| `crates/tc-core` | weights, FLUX.2 transformer, Qwen3, VAE, LoRA, sampler, pipeline |
| `crates/tc-config` | `GenerationConfiguration` (FlatBuffers + JSON) |
| `crates/tc-wire` | tensor wire codec (raw / deflate / fpzip) |
| `crates/tc-service` | model registry, request mapping, worker thread |
| `crates/tc-grpc` | gRPC server and client |
| `apps/draw-things-cli`, `apps/grpc-server-cli` | the binaries |
| `scripts/` | reference-fixture generators (PyTorch / diffusers) used by the tests |

## Testing

`cargo test --workspace` runs everything on CPU. The tests compare each op, each model (FLUX.2,
Qwen3, VAE), LoRA merging, the scheduler, the seeded noise and the full pipeline against
PyTorch / diffusers / transformers references produced by `scripts/fixtures/*.py` (fixtures are
checked in, so Python isn't needed to run the tests). End-to-end tests drive a tiny FLUX.2
[klein] snapshot with two LoRAs, with and without a reference image, through the real
`draw-things-cli` binary and through a gRPC server and client.

### Checking Metal on your Mac

GitHub's macOS runners are virtual machines without Metal Performance Shaders, so CI checks the
Metal build but not Metal numerics. On an Apple Silicon Mac:

```sh
cargo run -q -p nnc --example devices    # should list one GPU
NNC_TEST_DEVICE=gpu cargo test -p nnc -p tc-core -p draw-things-cli -- --test-threads=1
```

This compares each op and model on Metal with the PyTorch references, and runs the reference
image + two LoRAs case through `draw-things-cli` on the GPU in f32, f16, bf16 and `--quantize`,
printing the error of each against diffusers.

With the same seed, outputs match `diffusers`' `Flux2KleinPipeline` (the initial noise
reproduces `torch.randn` on CPU).

## License

BSD 3-Clause, see [LICENSE](LICENSE). Third-party components keep their own licenses
([NOTICE.md](NOTICE.md)).
