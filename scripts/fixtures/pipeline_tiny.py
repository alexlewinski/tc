"""End-to-end FLUX.2 [klein] pipeline reference (diffusers) with tiny random models."""
import os, sys, json
import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from diffusers import AutoencoderKLFlux2, Flux2Transformer2DModel, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
torch.manual_seed(4)
vae = AutoencoderKLFlux2(block_out_channels=(8, 8, 16, 16), down_block_types=("DownEncoderBlock2D",) * 4,
                         up_block_types=("UpDecoderBlock2D",) * 4, layers_per_block=1, latent_channels=4,
                         norm_num_groups=4).eval()
tr = Flux2Transformer2DModel(patch_size=1, in_channels=16, num_layers=1, num_single_layers=2, attention_head_dim=16,
                             num_attention_heads=2, joint_attention_dim=24, timestep_guidance_channels=16,
                             mlp_ratio=3.0, axes_dims_rope=(4, 4, 4, 4), guidance_embeds=False).eval()
with torch.no_grad():
    for m in (vae, tr):
        for n, p in m.named_parameters():
            if p.ndim == 1 and ("norm" in n):
                p.copy_(1.0 + 0.2 * torch.randn_like(p))
            else:
                p.copy_(0.2 * torch.randn_like(p))
    vae.bn.running_mean.copy_(0.2 * torch.randn(16)); vae.bn.running_var.copy_(0.5 + torch.rand(16))
vae.save_pretrained(os.path.join(out, "vae"), safe_serialization=True)
tr.save_pretrained(os.path.join(out, "transformer"), safe_serialization=True)
sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=True,
                                        base_shift=0.5, max_shift=1.15, time_shift_type="exponential")
sched.save_pretrained(os.path.join(out, "scheduler"))

class Dummy(torch.nn.Module):
    dtype = torch.float32
    device = torch.device("cpu")
res = {}
pe = torch.randn(1, 7, 24); ne = torch.randn(1, 7, 24)
res["pe"], res["ne"] = pe, ne
ref = (np.random.RandomState(0).rand(64, 80, 3) * 255).astype(np.uint8)
Image.fromarray(ref).save(os.path.join(out, "ref.png"))
for name, distilled, guidance, steps, use_ref in [("distilled", True, 1.0, 4, False), ("cfg", False, 3.5, 3, False), ("ref", True, 1.0, 2, True)]:
    pipe = Flux2KleinPipeline(scheduler=sched, vae=vae, text_encoder=Dummy(), tokenizer=None, transformer=tr, is_distilled=distilled)
    with torch.no_grad():
        img = pipe(image=Image.fromarray(ref) if use_ref else None, prompt=None, prompt_embeds=pe,
                   negative_prompt_embeds=ne, height=64, width=96, num_inference_steps=steps,
                   guidance_scale=guidance, generator=torch.Generator().manual_seed(1234), output_type="pt").images
    res[name] = img.permute(0, 2, 3, 1).contiguous()
save_file(res, os.path.join(out, "pipeline_io.safetensors"))
print("ok")
