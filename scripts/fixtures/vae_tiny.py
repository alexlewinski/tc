"""Tiny random FLUX.2 VAE reference (diffusers AutoencoderKLFlux2) for tc-core parity tests."""
import os, sys
import torch
from safetensors.torch import save_file
from diffusers import AutoencoderKLFlux2
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline as P

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
torch.manual_seed(3)
vae = AutoencoderKLFlux2(block_out_channels=(32, 64), down_block_types=("DownEncoderBlock2D",) * 2,
                         up_block_types=("UpDecoderBlock2D",) * 2, layers_per_block=1, latent_channels=4,
                         norm_num_groups=8).eval()
with torch.no_grad():
    for n, p in vae.named_parameters():
        if "norm" in n and n.endswith("weight"):
            p.copy_(1.0 + 0.2 * torch.randn_like(p))
        else:
            p.copy_(0.15 * torch.randn_like(p))
    vae.bn.running_mean.copy_(0.3 * torch.randn(16))
    vae.bn.running_var.copy_(0.5 + torch.rand(16))
vae.save_pretrained(os.path.join(out, "vae"), safe_serialization=True)
z = torch.randn(1, 4, 6, 8)
img = torch.rand(1, 3, 16, 24) * 2 - 1
with torch.no_grad():
    dec = vae.decode(z).sample
    enc = vae.encode(img).latent_dist.mode()
    packed = P._patchify_latents(enc)
    mean = vae.bn.running_mean.view(1, -1, 1, 1)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
    packed = (packed - mean) / std
nhwc = lambda t: t.permute(0, 2, 3, 1).contiguous()
save_file({"z": nhwc(z), "img": nhwc(img), "dec": nhwc(dec), "enc": nhwc(enc), "packed": nhwc(packed)},
          os.path.join(out, "vae_io.safetensors"))
print("ok")
