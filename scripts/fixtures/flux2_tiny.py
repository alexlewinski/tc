"""Tiny random FLUX.2 transformer reference (diffusers) for tc-core parity tests.

Writes into OUT_DIR:
  transformer/            diffusers layout (config.json + safetensors)
  flux2_bfl.safetensors   same weights in original BFL naming (single file)
  flux2_io.safetensors    inputs and expected outputs
  lora_peft.safetensors   a diffusers/PEFT style LoRA, lora_bfl.safetensors an ai-toolkit style one
"""
import os, sys
import torch
from safetensors.torch import save_file
from diffusers import Flux2Transformer2DModel
from diffusers.loaders.single_file_utils import convert_flux2_transformer_checkpoint_to_diffusers

out = sys.argv[1]
guidance = len(sys.argv) > 2 and sys.argv[2] == "guidance"
os.makedirs(out, exist_ok=True)
torch.manual_seed(1)
cfg = dict(patch_size=1, in_channels=8, num_layers=2, num_single_layers=2, attention_head_dim=16,
           num_attention_heads=2, joint_attention_dim=24, timestep_guidance_channels=16, mlp_ratio=3.0,
           axes_dims_rope=(4, 4, 4, 4), rope_theta=2000, eps=1e-6, guidance_embeds=guidance)
m = Flux2Transformer2DModel(**cfg).eval()
with torch.no_grad():
    for n, p in m.named_parameters():
        p.copy_(torch.randn_like(p) * (0.3 if p.ndim > 1 else 1.0) + (1.0 if p.ndim == 1 else 0.0))
m.save_pretrained(os.path.join(out, "transformer"), safe_serialization=True)

sd = {k: v.clone() for k, v in m.state_dict().items()}
# Inverse of convert_flux2_transformer_checkpoint_to_diffusers.
def inv(sd):
    o = {}
    ren = {
        "x_embedder": "img_in", "context_embedder": "txt_in",
        "time_guidance_embed.timestep_embedder.linear_1": "time_in.in_layer",
        "time_guidance_embed.timestep_embedder.linear_2": "time_in.out_layer",
        "time_guidance_embed.guidance_embedder.linear_1": "guidance_in.in_layer",
        "time_guidance_embed.guidance_embedder.linear_2": "guidance_in.out_layer",
        "double_stream_modulation_img.linear": "double_stream_modulation_img.lin",
        "double_stream_modulation_txt.linear": "double_stream_modulation_txt.lin",
        "single_stream_modulation.linear": "single_stream_modulation.lin",
        "proj_out": "final_layer.linear",
    }
    dbl = {"attn.norm_q": "img_attn.norm.query_norm", "attn.norm_k": "img_attn.norm.key_norm",
           "attn.to_out.0": "img_attn.proj", "ff.linear_in": "img_mlp.0", "ff.linear_out": "img_mlp.2",
           "attn.norm_added_q": "txt_attn.norm.query_norm", "attn.norm_added_k": "txt_attn.norm.key_norm",
           "attn.to_add_out": "txt_attn.proj", "ff_context.linear_in": "txt_mlp.0", "ff_context.linear_out": "txt_mlp.2"}
    sgl = {"attn.to_qkv_mlp_proj": "linear1", "attn.to_out": "linear2", "attn.norm_q": "norm.query_norm", "attn.norm_k": "norm.key_norm"}
    for k, v in sd.items():
        base, param = k.rsplit(".", 1)
        if base == "norm_out.linear":
            s, sh = v.chunk(2, 0); o["final_layer.adaLN_modulation.1." + param] = torch.cat([sh, s], 0); continue
        if base in ren:
            o[ren[base] + "." + param] = v; continue
        if base.startswith("transformer_blocks."):
            _, n, rest = base.split(".", 2)
            if rest in ("attn.to_q", "attn.to_k", "attn.to_v", "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"):
                continue
            nk = f"double_blocks.{n}.{dbl[rest]}." + ("scale" if "norm" in rest else param)
            o[nk] = v; continue
        if base.startswith("single_transformer_blocks."):
            _, n, rest = base.split(".", 2)
            o[f"single_blocks.{n}.{sgl[rest]}." + ("scale" if "norm" in rest else param)] = v; continue
        raise KeyError(k)
    for n in range(cfg["num_layers"]):
        p = f"transformer_blocks.{n}.attn."
        o[f"double_blocks.{n}.img_attn.qkv.weight"] = torch.cat([sd[p + "to_q.weight"], sd[p + "to_k.weight"], sd[p + "to_v.weight"]])
        o[f"double_blocks.{n}.txt_attn.qkv.weight"] = torch.cat([sd[p + "add_q_proj.weight"], sd[p + "add_k_proj.weight"], sd[p + "add_v_proj.weight"]])
    return o
bfl = inv(sd)
back = convert_flux2_transformer_checkpoint_to_diffusers({k: v.clone() for k, v in bfl.items()})
assert set(back) == set(sd), set(back) ^ set(sd)
for k in sd:
    assert torch.equal(back[k], sd[k]), k
save_file({"model.diffusion_model." + k: v.contiguous() for k, v in bfl.items()}, os.path.join(out, "flux2_bfl.safetensors"))

def ids_img(h, w, t):
    return torch.cartesian_prod(torch.tensor([t]), torch.arange(h), torch.arange(w), torch.arange(1))
B, L = 1, 5
x = torch.randn(B, 16 + 4, 8)
ctx = torch.randn(B, L, 24)
img_ids = torch.cat([ids_img(4, 4, 0), ids_img(2, 2, 10)])[None]
txt_ids = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(L))[None]
t = torch.tensor([0.7])
g = torch.tensor([3.5]) if guidance else None
with torch.no_grad():
    y = m(hidden_states=x, encoder_hidden_states=ctx, timestep=t, img_ids=img_ids, txt_ids=txt_ids, guidance=g, return_dict=False)[0]
io = {"x": x, "ctx": ctx, "out": y}

# LoRAs: rank 4 on a few modules, PEFT naming and ai-toolkit naming.
r = 4
peft, aitk = {}, {}
targets = {
    "transformer_blocks.0.attn.to_k": (cfg["num_attention_heads"] * 16, 32),
    "transformer_blocks.1.ff.linear_out": (32, 96),
    "single_transformer_blocks.0.attn.to_qkv_mlp_proj": (3 * 32 + 2 * 96, 32),
    "norm_out.linear": (64, 32),
}
for mod, (o_, i_) in targets.items():
    A = torch.randn(r, i_) * 0.2; Bm = torch.randn(o_, r) * 0.2
    peft[f"transformer.{mod}.lora_A.weight"] = A; peft[f"transformer.{mod}.lora_B.weight"] = Bm
save_file(peft, os.path.join(out, "lora_peft.safetensors"))
A = torch.randn(r, 32) * 0.2; Bm = torch.randn(3 * 32, r) * 0.2
aitk["diffusion_model.double_blocks.1.txt_attn.qkv.lora_A.weight"] = A
aitk["diffusion_model.double_blocks.1.txt_attn.qkv.lora_B.weight"] = Bm
aitk["diffusion_model.double_blocks.1.txt_attn.qkv.alpha"] = torch.tensor(2.0)
A2 = torch.randn(r, 32 + 96) * 0.2; B2 = torch.randn(32, r) * 0.2
aitk["diffusion_model.single_blocks.1.linear2.lora_A.weight"] = A2
aitk["diffusion_model.single_blocks.1.linear2.lora_B.weight"] = B2
save_file(aitk, os.path.join(out, "lora_bfl.safetensors"))

# Expected output with both LoRAs merged (weights 0.8 and 0.5).
with torch.no_grad():
    sdm = m.state_dict()
    for mod in targets:
        sdm[mod + ".weight"] += 0.8 * peft[f"transformer.{mod}.lora_B.weight"] @ peft[f"transformer.{mod}.lora_A.weight"]
    d = 0.5 * (2.0 / r) * (Bm @ A)
    sdm["transformer_blocks.1.attn.add_q_proj.weight"] += d[:32]
    sdm["transformer_blocks.1.attn.add_k_proj.weight"] += d[32:64]
    sdm["transformer_blocks.1.attn.add_v_proj.weight"] += d[64:]
    sdm["single_transformer_blocks.1.attn.to_out.weight"] += 0.5 * (B2 @ A2)
    m.load_state_dict(sdm)
    io["out_lora"] = m(hidden_states=x, encoder_hidden_states=ctx, timestep=t, img_ids=img_ids, txt_ids=txt_ids, guidance=g, return_dict=False)[0]
save_file({k: v.contiguous() for k, v in io.items()}, os.path.join(out, "flux2_io.safetensors"))
print("ok")
