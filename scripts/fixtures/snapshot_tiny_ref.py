"""Reference rendering for the end-to-end tests with a reference image and two LoRAs, using the
snapshot written by snapshot_tiny.py.

Usage: snapshot_tiny_ref.py OUT_DIR  (the same OUT_DIR as snapshot_tiny.py)
Creates OUT_DIR/ref.png and OUT_DIR/expected_ref.safetensors.
"""
import os, sys
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from transformers import Qwen3ForCausalLM
from diffusers import AutoencoderKLFlux2, Flux2Transformer2DModel, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline

out = sys.argv[1]
snap = os.path.join(out, "tiny-klein")
tok = Tokenizer.from_file(os.path.join(snap, "tokenizer", "tokenizer.json"))
te = Qwen3ForCausalLM.from_pretrained(os.path.join(snap, "text_encoder")).eval()
vae = AutoencoderKLFlux2.from_pretrained(os.path.join(snap, "vae")).eval()
tr = Flux2Transformer2DModel.from_pretrained(os.path.join(snap, "transformer")).eval()
sched = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(snap, "scheduler"))
la = load_file(os.path.join(out, "lora_a.safetensors"))
lb = load_file(os.path.join(out, "lora_b.safetensors"))
r = 4

# A reference image with a different aspect ratio than the 128x64 output.
ref = (np.random.RandomState(3).rand(64, 96, 3) * 255).astype(np.uint8)
Image.fromarray(ref).save(os.path.join(out, "ref.png"))

def embed(prompt):
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ids = tok.encode(text).ids[:512]
    valid = len(ids)
    ids = ids + [tok.token_to_id("<|endoftext|>")] * (512 - valid)
    mask = [1] * valid + [0] * (512 - valid)
    with torch.no_grad():
        o = te(input_ids=torch.tensor([ids]), attention_mask=torch.tensor([mask]), output_hidden_states=True, use_cache=False)
    hs = torch.stack([o.hidden_states[k] for k in (9, 18, 27)], dim=1)
    b, c, s, d = hs.shape
    return hs.permute(0, 2, 1, 3).reshape(b, s, c * d)

with torch.no_grad():
    sd = tr.state_dict()
    sd["single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight"] += 0.8 * la["transformer.single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_B.weight"] @ la["transformer.single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_A.weight"]
    sd["transformer_blocks.0.attn.to_v.weight"] += 0.8 * la["transformer.transformer_blocks.0.attn.to_v.lora_B.weight"] @ la["transformer.transformer_blocks.0.attn.to_v.lora_A.weight"]
    sd["transformer_blocks.0.ff.linear_in.weight"] += 0.5 * (2.0 / r) * lb["lora_unet_double_blocks_0_img_mlp_0.lora_up.weight"] @ lb["lora_unet_double_blocks_0_img_mlp_0.lora_down.weight"]
    tr.load_state_dict(sd)
class Dummy(torch.nn.Module):
    dtype = torch.float32
    device = torch.device("cpu")
pipe = Flux2KleinPipeline(scheduler=sched, vae=vae, text_encoder=Dummy(), tokenizer=None, transformer=tr, is_distilled=True)
with torch.no_grad():
    img = pipe(image=Image.open(os.path.join(out, "ref.png")).convert("RGB"), prompt=None, prompt_embeds=embed("a red cube on a table"),
               height=64, width=128, num_inference_steps=3, generator=torch.Generator().manual_seed(7), output_type="pt").images
save_file({"image": img.permute(0, 2, 3, 1).contiguous()}, os.path.join(out, "expected_ref.safetensors"))
print("ok")
