"""Build a tiny FLUX.2 [klein]-style diffusers snapshot (random weights) plus two LoRAs, and a
diffusers reference rendering of one prompt, for end-to-end CLI / gRPC tests.

Usage: snapshot_tiny.py OUT_DIR
Creates OUT_DIR/tiny-klein/{model_index.json,transformer,text_encoder,tokenizer,vae,scheduler},
OUT_DIR/lora_a.safetensors, OUT_DIR/lora_b.safetensors and OUT_DIR/expected.safetensors.
"""
import json, os, sys
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import Qwen3Config, Qwen3ForCausalLM
from diffusers import AutoencoderKLFlux2, Flux2Transformer2DModel, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline

out = sys.argv[1]
snap = os.path.join(out, "tiny-klein")
os.makedirs(snap, exist_ok=True)
torch.manual_seed(5)

# Tokenizer: byte-level BPE with the Qwen chat special tokens.
specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=300, special_tokens=specials, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
tok.train_from_iterator(["a red cube on a table", "user assistant hello world", "a photo of a cat"] * 10, trainer)
os.makedirs(os.path.join(snap, "tokenizer"), exist_ok=True)
tok.save(os.path.join(snap, "tokenizer", "tokenizer.json"))
json.dump({"pad_token": "<|endoftext|>"}, open(os.path.join(snap, "tokenizer", "tokenizer_config.json"), "w"))

hidden = 16
te_cfg = Qwen3Config(vocab_size=tok.get_vocab_size(), hidden_size=hidden, intermediate_size=32, num_hidden_layers=28,
                     num_attention_heads=2, num_key_value_heads=1, head_dim=8, rms_norm_eps=1e-6, rope_theta=1000000.0,
                     max_position_embeddings=1024, tie_word_embeddings=False)
te = Qwen3ForCausalLM(te_cfg).eval()
vae = AutoencoderKLFlux2(block_out_channels=(8, 8, 16, 16), down_block_types=("DownEncoderBlock2D",) * 4,
                         up_block_types=("UpDecoderBlock2D",) * 4, layers_per_block=1, latent_channels=4, norm_num_groups=4).eval()
tr = Flux2Transformer2DModel(patch_size=1, in_channels=16, num_layers=1, num_single_layers=2, attention_head_dim=16,
                             num_attention_heads=2, joint_attention_dim=3 * hidden, timestep_guidance_channels=16,
                             mlp_ratio=3.0, axes_dims_rope=(4, 4, 4, 4), guidance_embeds=False).eval()
with torch.no_grad():
    for m in (te, vae, tr):
        for n, p in m.named_parameters():
            if p.ndim == 1 and "norm" in n:
                p.copy_(1.0 + 0.2 * torch.randn_like(p))
            else:
                p.copy_(0.2 * torch.randn_like(p))
    vae.bn.running_mean.copy_(0.2 * torch.randn(16)); vae.bn.running_var.copy_(0.5 + torch.rand(16))
te.save_pretrained(os.path.join(snap, "text_encoder"), safe_serialization=True)
vae.save_pretrained(os.path.join(snap, "vae"), safe_serialization=True)
tr.save_pretrained(os.path.join(snap, "transformer"), safe_serialization=True)
sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=True,
                                        base_shift=0.5, max_shift=1.15, time_shift_type="exponential")
sched.save_pretrained(os.path.join(snap, "scheduler"))
json.dump({"_class_name": "Flux2KleinPipeline", "is_distilled": True}, open(os.path.join(snap, "model_index.json"), "w"))

# Two LoRAs in different naming conventions.
r = 4
la = {"transformer.single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_A.weight": torch.randn(r, 32) * 0.3,
      "transformer.single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_B.weight": torch.randn(3 * 32 + 2 * 96, r) * 0.3,
      "transformer.transformer_blocks.0.attn.to_v.lora_A.weight": torch.randn(r, 32) * 0.3,
      "transformer.transformer_blocks.0.attn.to_v.lora_B.weight": torch.randn(32, r) * 0.3}
lb = {"lora_unet_double_blocks_0_img_mlp_0.lora_down.weight": torch.randn(r, 32) * 0.3,
      "lora_unet_double_blocks_0_img_mlp_0.lora_up.weight": torch.randn(2 * 96, r) * 0.3,
      "lora_unet_double_blocks_0_img_mlp_0.alpha": torch.tensor(2.0)}
save_file(la, os.path.join(out, "lora_a.safetensors"))
save_file(lb, os.path.join(out, "lora_b.safetensors"))

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
    img = pipe(prompt=None, prompt_embeds=embed("a red cube on a table"), height=64, width=128, num_inference_steps=3,
               generator=torch.Generator().manual_seed(7), output_type="pt").images
save_file({"image": img.permute(0, 2, 3, 1).contiguous()}, os.path.join(out, "expected.safetensors"))
print("ok", tok.get_vocab_size())
