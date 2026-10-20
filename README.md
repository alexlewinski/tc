# Tonic Clap

Fast local **FLUX.2** image generation in Rust, primarily for **Apple Silicon Macs**, with:

* **`draw-things-cli`** — a command line generator (clap), command-compatible with the
  Draw Things CLI for image generation.
* **`gRPCServerCLI`** — a gRPC server (tonic) that speaks the Draw Things image generation
  protocol, so other software (or the Draw Things app) can drive generation remotely.

The compute backend is [ccv / nnc](https://github.com/liuliu/ccv) (vendored C, BSD-3) with its
Metal (MPS + Metal FlashAttention) backend on macOS and a CPU backend elsewhere.

Supported models: **FLUX.2 [klein] 9B / 9B-KV / 4B** (distilled and base), either from the
official Hugging Face releases or as the `.ckpt` files Draw Things keeps in its models folder,
with any number of LoRAs (Draw Things `.ckpt`, diffusers/PEFT, ai-toolkit/BFL and kohya formats).

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
`~/Documents/Models`; override with `--models-dir`. A model is either a Hugging Face snapshot in
diffusers layout (`model_index.json`, `transformer/`, `text_encoder/`, `tokenizer/`, `vae/`,
`scheduler/`), stored as `<models-dir>/FLUX.2-klein-9B/` or found in the Hugging Face cache, or a
Draw Things checkpoint (below).

## Using a Draw Things models folder

Draw Things' own files work directly: name the transformer `.ckpt` as the model and the
LoRA `.ckpt` files as LoRAs. The matching Qwen3 text encoder (`qwen_3_8b_*.ckpt` for 9B, including
a `-tensordata` sidecar) and VAE (`flux_2_vae_f16.ckpt`) are found next to it. The Qwen3 tokenizer
is not part of Draw Things' files; it is downloaded once (~11 MB) into
`<models-dir>/flux_2_tokenizer/` unless a FLUX.2 snapshot is present.

```sh
# Draw Things' usual models folder (Settings shows the actual location if you moved it)
DT="$HOME/Library/Containers/com.liuliu.draw-things/Data/Documents/Models"
./target/release/draw-things-cli generate --models-dir "$DT" \
  --model flux_2_klein_9b_kv_q8p.ckpt \
  --lora nnegrat_ink_9b_2250_lora_f16.ckpt:0.8 \
  --lora replace_outfit_v1_000005750_lora_f16.ckpt:1.0 \
  --image croquis.png --image outfit.png \
  --prompt "Have the croquis on image 1 wear the outfit on image 2, while keeping the croquis unchanged" \
  --seed 42 -o out.png

./target/release/draw-things-cli models inspect --models-dir "$DT" flux_2_klein_9b_kv_q8p.ckpt \
  --lora nnegrat_ink_9b_2250_lora_f16.ckpt --lora replace_outfit_v1_000005750_lora_f16.ckpt
```

* `flux_2_klein_9b_kv_*` is **FLUX.2 [klein] 9B-KV** (black-forest-labs/FLUX.2-klein-9b-kv): the
  9B model trained so that reference images can be KV-cached. Its reference tokens are placed
  before the image, modulated at timestep 0 and attend only to themselves, as in diffusers'
  `Flux2KleinKVPipeline` (Tonic Clap computes this exactly, without the cache).
* 8-bit palettized (`q8p`) weights are used as stored, as with `--quantize` (on the GPU this runs
  in f16), so the 9B model needs about as much memory as in Draw Things.
* `models ensure --model flux_2_klein_9b_kv_q8p.ckpt` downloads a model and its text encoder and
  VAE from the Draw Things CDN (`https://static.libnnc.org/<file>`), verifying the SHA-256 from
  `https://models.drawthings.ai/models_sha256.json`.
* Supported encodings: raw, zip, fpzip, ezm7 and q8p (all FLUX.2 [klein] `_q8p` / `_f16` files and
  the LoRA files Draw Things writes); `_q6p` / `_i8x` files are reported as unsupported.
* `models compare CKPT REFERENCE` checks a checkpoint against the official release tensor by
  tensor, and a Draw Things LoRA against its original `.safetensors` (relative difference of the
  weight updates and best-fit scale).

The gRPC server reads the same files: put the `.ckpt` names in the request configuration
(`"model": "flux_2_klein_9b_kv_q8p.ckpt"`, `"loras": [{"file": "nnegrat_ink_9b_2250_lora_f16.ckpt", ...}]`).

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
| `scripts/` | reference-fixture generators (PyTorch / diffusers) used by the tests; `scripts/ckpt/` verification of the Draw Things checkpoint mapping against official weights |

## Testing

`cargo test --workspace` runs everything on CPU. The tests compare each op, each model (FLUX.2,
Qwen3, VAE), LoRA merging, the scheduler, the seeded noise and the full pipeline against
PyTorch / diffusers / transformers references produced by `scripts/fixtures/*.py` (fixtures are
checked in, so Python isn't needed to run the tests). End-to-end tests drive a tiny FLUX.2
[klein] snapshot with two LoRAs, with and without a reference image, through the real
`draw-things-cli` binary and through a gRPC server and client, both from the diffusers snapshot
and from the same model written as Draw Things `.ckpt` files (with a Draw Things-layout LoRA).
Draw Things checkpoint reading (inline, `-tensordata` sidecar and trailer stores; raw, zip,
fpzip, ezm7 and q8p tensors) and the transformer, Qwen3, VAE and LoRA mappings have their own
unit tests; their verification against the real published files is described in
[docs/PROVENANCE.md](docs/PROVENANCE.md).

### Checking Metal on your Mac

CI runs these GPU tests on GitHub's Apple Silicon runners (virtual machines with a Metal
device). To run them on your own Mac:

```sh
cargo run -q -p nnc --example devices    # should list one GPU
NNC_TEST_DEVICE=gpu cargo test -p nnc -p tc-core -p tc-service -p draw-things-cli -- --test-threads=1
```

This compares each op and model on Metal with the PyTorch references, and runs the reference
image + two LoRAs case through `draw-things-cli` on the GPU in f32, f16, bf16 and `--quantize`,
printing the error of each against diffusers.

With the same seed, outputs match `diffusers`' `Flux2KleinPipeline` (the initial noise
reproduces `torch.randn` on CPU).

## License

BSD 3-Clause, see [LICENSE](LICENSE). Third-party components keep their own licenses
([NOTICE.md](NOTICE.md)).
