# Interfaces

This document specifies the external interfaces Tonic Clap implements: the gRPC protocol,
the `GenerationConfiguration` table, the tensor wire format and the command line.

## 1. gRPC: `ImageGenerationService`

Definition: [`proto/image_service.proto`](../proto/image_service.proto) (proto3, no package,
so method paths are `/ImageGenerationService/<Method>`). Default port **7859**. TLS is on by
default with a self-signed certificate; clients are expected not to verify it (or to pin it).
Maximum message size: 1 GiB.

### Authentication

If the server has a shared secret, every request must carry it in `sharedSecret`:

* `Echo` with a missing/wrong secret returns normally with `sharedSecretMissing = true`.
* All other methods fail with `UNAUTHENTICATED`.

### `Echo(EchoRequest) -> EchoReply`

`message = "HELLO <name>"`, `serverIdentifier` = random 64-bit id fixed for the server's
lifetime. With `--model-browser`, `files` lists locally available models (registry ids) and
`*.safetensors` / `*.ckpt` files in the models directories, and `override.models` holds a JSON array
describing the available models.

### `FilesExist(FileListRequest) -> FileExistenceResponse`

For each name in `files`: `existences[i]` tells whether it resolves to a local model or a file
in a models directory; `hashes[i]` is the SHA-256 of the file when the name is also listed in
`filesWithHash` (empty otherwise).

### `UploadFile(stream FileUploadRequest) -> stream UploadResponse`

1. `initRequest{filename, sha256, totalSize}` — the server creates `<models-dir>/<filename>.part`
   (`filename` is reduced to its base name; names starting with `.` are rejected).
2. `chunk{content, offset}` messages in order; `offset` must equal the bytes received so far.
   Each chunk is acknowledged with `receivedOffset`.
3. When the client half-closes, the size and SHA-256 are verified and the file is renamed to
   `<models-dir>/<filename>`. Mismatches fail with `INVALID_ARGUMENT` / `DATA_LOSS`.

### `GenerateImage(ImageGenerationRequest) -> stream ImageGenerationResponse`

Request fields used:

| Field | Meaning |
|-------|---------|
| `configuration` | `GenerationConfiguration` FlatBuffer (§2). Its `model` names the model. |
| `prompt`, `negativePrompt` | text (negative prompt only matters for CFG on non-distilled models) |
| `image` | canvas image tensor (§3). With `strength < 1` it initializes the latent (image-to-image); otherwise it is a reference image for FLUX.2 editing. |
| `hints` | `hintType = "shuffle"` tensors are additional reference images; other hint types are ignored |
| `contents` | content-addressed blobs: any `image` / `mask` / hint tensor that is exactly 32 bytes is a SHA-256 key into `contents` |
| `chunked` | client accepts results split into 4 MiB chunks |
| `mask` | accepted, currently ignored (no inpainting) |

Response stream:

1. Progress messages with `currentSignpost` (`textEncoded`, `imageEncoded`, `sampling{step}`,
   `imageDecoded`) and `signposts` (the expected sequence).
2. If there are results: one message with `downloadSize` = total bytes of all images.
3. One message per image with `generatedImages = [tensor]`, `scaleFactor = 1`,
   `chunkState = LAST_CHUNK`. If `chunked` and the total exceeds 4 MiB, images larger than
   4 MiB are split over several messages with `chunkState = MORE_CHUNKS` until the last.

Result images are `f16` tensors, shape `[1, H, W, 3]`, NHWC, values in `[-1, 1]`, compressed
with fpzip unless the server runs with `--no-response-compression`. Errors end the stream with
a non-OK status. Cancelling the call (client disconnect) cancels the generation.

`Pubkey` and `Hours` return empty responses.

## 2. `GenerationConfiguration`

A FlatBuffers table (root table, little endian). Field `i` uses vtable slot `4 + 2*i`.
Sizes marked "64px" are in units of 64 pixels.

| # | Field | Type | Default | Used |
|---|-------|------|---------|------|
| 0 | id | int64 | 0 | |
| 1 | start_width | uint16 (64px) | 0 | ✓ (0 → model default) |
| 2 | start_height | uint16 (64px) | 0 | ✓ |
| 3 | seed | uint32 | 0 | ✓ |
| 4 | steps | uint32 | 0 | ✓ (0 → model default) |
| 5 | guidance_scale | float | 0 | ✓ (CFG on non-distilled models) |
| 6 | strength | float | 0 | ✓ (0 → 1.0) |
| 7 | model | string | | ✓ |
| 8 | sampler | int8 | 0 | ✓ (ancestral types → stochastic Euler, others → Euler) |
| 9 | batch_count | uint32 | 1 | ✓ |
| 10 | batch_size | uint32 | 1 | ✓ (images = count × size, seeds seed+i) |
| 11 | hires_fix | bool | false | |
| 12–13 | hires_fix_start_width/height | uint16 | 0 | |
| 14 | hires_fix_strength | float | 0.7 | |
| 15 | upscaler | string | | |
| 16 | image_guidance_scale | float | 1.5 | |
| 17 | seed_mode | int8 | 0 | (noise always matches PyTorch CPU) |
| 18 | clip_skip | uint32 | 1 | |
| 19 | controls | [Control] | | |
| 20 | loras | [LoRA] | | ✓ |
| 21 | mask_blur | float | 0 | |
| 22 | face_restoration | string | | |
| 23–24 | decode_with_attention, hires_fix_decode_with_attention (deprecated) | bool | true | |
| 25 | clip_weight | float | 1 | |
| 26 | negative_prompt_for_image_prior | bool | true | |
| 27 | image_prior_steps | uint32 | 5 | |
| 28 | refiner_model | string | | |
| 29–30 | original_image_height/width | uint32 | 0 | |
| 31–32 | crop_top / crop_left | int32 | 0 | |
| 33–34 | target_image_height/width | uint32 | 0 | |
| 35 | aesthetic_score | float | 6 | |
| 36 | negative_aesthetic_score | float | 2.5 | |
| 37 | zero_negative_prompt | bool | false | |
| 38 | refiner_start | float | 0.7 | |
| 39–40 | negative_original_image_height/width | uint32 | 0 | |
| 41 | name | string | | |
| 42 | fps_id | uint32 | 5 | |
| 43 | motion_bucket_id | uint32 | 127 | |
| 44 | cond_aug | float | 0.02 | |
| 45 | start_frame_cfg | float | 1 | |
| 46 | num_frames | uint32 | 14 | |
| 47 | mask_blur_outset | int32 | 0 | |
| 48 | sharpness | float | 0 | |
| 49 | shift | float | 1 | ✓ when `resolution_dependent_shift` is false |
| 50 | stage_2_steps | uint32 | 10 | |
| 51 | stage_2_cfg | float | 1 | |
| 52 | stage_2_shift | float | 1 | |
| 53 | tiled_decoding | bool | false | |
| 54–56 | decoding_tile_width/height/overlap | uint16 (64px) | 10, 10, 2 | |
| 57 | stochastic_sampling_gamma | float | 0.3 | |
| 58 | preserve_original_after_inpaint | bool | true | |
| 59 | tiled_diffusion | bool | false | |
| 60–62 | diffusion_tile_width/height/overlap | uint16 (64px) | 16, 16, 2 | |
| 63 | upscaler_scale_factor | uint8 | 0 | |
| 64 | t5_text_encoder | bool | true | |
| 65 | separate_clip_l | bool | false | |
| 66 | clip_l_text | string | | |
| 67 | separate_open_clip_g | bool | false | |
| 68 | open_clip_g_text | string | | |
| 69 | speed_up_with_guidance_embed | bool | true | |
| 70 | guidance_embed | float | 3.5 | |
| 71 | resolution_dependent_shift | bool | true | ✓ (FLUX.2 empirical shift) |
| 72 | tea_cache_start | int32 | 5 | |
| 73 | tea_cache_end | int32 | -1 | |
| 74 | tea_cache_threshold | float | 0.06 | |
| 75 | tea_cache | bool | false | |
| 76 | separate_t5 | bool | false | |
| 77 | t5_text | string | | |
| 78 | tea_cache_max_skip_steps | int32 | 3 | |
| 79 | causal_inference_enabled | bool | false | |
| 80 | causal_inference | int32 | 3 | |
| 81 | causal_inference_pad | int32 | 0 | |
| 82 | cfg_zero_star | bool | false | |
| 83 | cfg_zero_init_steps | int32 | 0 | |
| 84 | compression_artifacts | int8 | 0 | |
| 85 | compression_artifacts_quality | float | 43.1 | |
| 86 | color_calibration | int8 | 0 | |
| 87 | expand_prompt_to_json | bool | false | |
| 88 | shift_for_audio | float | 3 | |
| 89 | uses_sol_attention | bool | false | |
| 90 | sol_attention_start | int32 | 2 | |
| 91 | sol_attention_tau | float | 0.5 | |

`LoRA` table: `file: string` (slot 4), `weight: float = 0.6` (slot 6), `mode: int8` (slot 8;
0 all, 1 base, 2 refiner — refiner-only LoRAs are skipped). The file is looked up in the
models directories (also under `loras/`).

`Control` table: `file`, `weight = 1`, `guidance_start = 0`, `guidance_end = 1`, `no_prompt`,
`global_average_pooling = true`, `down_sampling_rate = 1`, `control_mode`, `target_blocks: [string]`,
`input_override` (parsed and preserved, not used).

Sampler values: 0 DPMPP2MKarras, 1 EulerA, 2 DDIM, 3 PLMS, 4 DPMPPSDEKarras, 5 UniPC, 6 LCM,
7 EulerASubstep, 8 DPMPPSDESubstep, 9 TCD, 10 EulerATrailing, 11 DPMPPSDETrailing,
12 DPMPP2MAYS, 13 EulerAAYS, 14 DPMPPSDEAYS, 15 DPMPP2MTrailing, 16 DDIMTrailing,
17 UniPCTrailing, 18 UniPCAYS, 19 TCDTrailing.

### JSON form (CLI `--config-json` / `--config-file`)

A partial object merged onto the model's recommended settings. Keys are camelCase
(snake_case accepted); sizes are in pixels: `width`, `height`, `seed`, `steps`,
`guidanceScale`, `strength`, `model`, `sampler` (number or name such as `"Euler A Trailing"`),
`batchCount`, `batchSize`, `shift`, `resolutionDependentShift`, `loras` (`[{"file", "weight",
"mode": "all"|"base"|"refiner"}]`), `controls`, and the remaining fields under their camelCase
names (`hiresFixWidth`, `decodingTileWidth`, … in pixels; `fps`, `motionScale`,
`guidingFrameNoise`, `startFrameGuidance`, `stage2Guidance` map to `fps_id`,
`motion_bucket_id`, `cond_aug`, `start_frame_cfg`, `stage_2_cfg`).

## 3. Tensor wire format

Images, masks and hints are sent as:

```
offset 0   u32   codec: 0 = raw, 0x217 = raw deflate (RFC 1951, no zlib header), 0xf7217 = fpzip
offset 4   i32   memory type (1 = CPU)
offset 8   i32   format (1 = NCHW, 2 = NHWC)
offset 12  i32   datatype: 0x1000 u8, 0x2000 i32, 0x4000 f32, 0x8000 i64, 0x10000 f64, 0x20000 f16, 0x80000 bf16
offset 16  i32   reserved (0)
offset 20  i32×12 dimensions, zero-terminated
offset 68  ...   payload: little-endian elements (raw), deflated elements, or an fpzip stream
```

fpzip streams store floats: dimensions map to `nx = d[n-1]`, `ny = d[n-2]`, `nz = d[n-3]`,
`nf = product of the rest`; f16 tensors are widened to f32 and encoded with 19 bits of
precision (lossless for f16). Images are NHWC `[1, H, W, 3]` floats in `[-1, 1]`.

## 4. Command line

### `draw-things-cli generate`

`--model` (registry id, alias, display name, `owner/repo`, `hf://owner/repo`, Hugging Face URL,
snapshot path, or a Draw Things checkpoint file name / path such as
`flux_2_klein_9b_kv_q8p.ckpt`), `--prompt` / `--prompt-file` (`-` = stdin), `--negative-prompt` /
`--negative-prompt-file`, `--steps`, `--cfg`, `--width`, `--height` (multiples of 64),
`--strength`, `--seed`, `--config-json`, `--config-file`, `--image` (repeatable; the first is the
canvas, the rest are references; `--init-image` / `--input-image` are aliases), `--crop`,
`--lora FILE[:WEIGHT]` (repeatable; `.safetensors` or Draw Things `.ckpt`), `-o/--output` (`.png`; multiple images get `-N` suffixes),
`--terminal-image`, `--terminal-image-protocol auto|iterm2|kitty`, `--[no-]download-missing`,
`--disable-preview`, `--offline`, `--local`, `--remote`, `--remote-url`, `--remote-port`,
`--[no-]remote-tls`, `--remote-shared-secret`, `--precision`, `--text-encoder-precision`, `--cpu`, `--quantize`.

Precedence: model recommendations < JSON overrides < explicit flags. Without `--width` /
`--height`, the size follows the first `--image` (rounded to 64). Video, audio, cloud compute
and estimate options are rejected with an explanatory error.

`draw-things-cli models list [--downloaded-only]`, `models ensure --model M` (downloads the
diffusers components from Hugging Face, `HF_TOKEN` for gated repos; for a `*.ckpt` name, downloads
it and its text encoder and VAE from the Draw Things CDN, checking SHA-256 against the published
manifest), `models import FILE --base M` (registers a fine-tuned transformer that reuses a base
model's other components), `models inspect FILE [--lora L ...]` (describes a Draw Things
checkpoint: component, encodings, resolved companion files, LoRA coverage), `models compare CKPT
REFERENCE [--filter S] [--all]` (tensor-by-tensor comparison of a Draw Things checkpoint with the
official release; for a Draw Things LoRA, comparison of its weight updates with the original
LoRA file), `models check-gpu MODEL [--lora FILE[:W] ...] [--blocks D S] [--precision P]
[--no-text-encoder] [--no-vae]` (runs a model as `generate` loads it on the GPU and expanded to f32
on the CPU, on the same inputs, and prints the relative error of each stage: palettized matrix
products, the first transformer blocks in plain and KV mode with LoRAs, the first Qwen3 layers, a
VAE decode / encode), `completion bash|zsh|fish|elvish|powershell`.

### Draw Things checkpoint files

A Draw Things model in a models directory is a set of `.ckpt` files (nnc SQLite tensor stores):
the transformer (e.g. `flux_2_klein_9b_kv_q8p.ckpt`), a Qwen3 text encoder `qwen_3_*.ckpt`
whose hidden size matches the transformer (`qwen_3_8b_*` for 9B, `qwen_3_4b_*` for 4B), and the
VAE `flux_2_vae*.ckpt`, preferring `_q8p`, then `_f16`, then other encodings. A checkpoint may keep
its tensor data in a `<file>-tensordata` sidecar or appended to the database. The Qwen3
tokenizer is not part of these files: it is taken from any local FLUX.2 snapshot or
`<models-dir>/flux_2_tokenizer/`, and otherwise downloaded there once (~11 MB, from the
Apache-2.0 `black-forest-labs/FLUX.2-klein-4B` repository). Files whose name contains `_kv`
(FLUX.2 [klein] 9B-KV) use the KV-variant reference-image conditioning. Weights stored 8-bit
palettized (`q8p`) stay palettized in memory on the GPU (which then runs in f16), as with
`--quantize`. Supported encodings: raw, zip, fpzip, ezm7, q8p; others (q6p, i8x, ...) give an error
naming the encoding.

### `gRPCServerCLI <models-dir>`

`-n/--name`, `-a/--address` (0.0.0.0), `-p/--port` (7859), `-g/--gpu`, `-s/--shared-secret`,
`--no-tls`, `--no-response-compression`, `--model-browser`, `--no-flash-attention`, `--debug`,
`--secondary-models-directory`, `--cpu-offload` (unload the text encoder after each prompt),
`--supervised`, `--max-crashes-within-time-window`, `--crash-time-window`, plus
`--precision`, `--text-encoder-precision`, `--keep-text-encoder`, `--cpu`, `--quantize`. Accepted for
compatibility but without effect: `-d/--datadog-api-key`, `-w/--weights-cache`,
`--blob-store-*`, `--fread-preferred`, `--cache-uri`, `--join`,
`--cancellation-{warning,crash}-timeout`, `--echo-on-queue`.
