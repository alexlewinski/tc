"""FLUX.2 [klein] KV-variant reference (diffusers `Flux2Transformer2DModel` with
`kv_cache_mode`) on the tiny transformer written by flux2_tiny.py.

Writes OUT_DIR/flux2_kv_io.safetensors: inputs, the output of the "extract" pass (reference
tokens first, fixed timestep 0 for their modulation, reference tokens attending only to
themselves) and of a following "cached" pass at another timestep."""
import os, sys
import torch
from safetensors.torch import save_file
from diffusers import Flux2Transformer2DModel
from diffusers.models.transformers.transformer_flux2 import Flux2KVAttnProcessor, Flux2KVParallelSelfAttnProcessor

out = sys.argv[1]
torch.manual_seed(3)
m = Flux2Transformer2DModel.from_pretrained(os.path.join(out, "transformer")).eval()
# As Flux2KleinKVPipeline does.
for block in m.transformer_blocks:
    block.attn.set_processor(Flux2KVAttnProcessor())
for block in m.single_transformer_blocks:
    block.attn.set_processor(Flux2KVParallelSelfAttnProcessor())
def ids_img(h, w, t):
    return torch.cartesian_prod(torch.tensor([t]), torch.arange(h), torch.arange(w), torch.arange(1))
L = 5
x = torch.randn(1, 16, 8)
ref = torch.randn(1, 4 + 6, 8)
ctx = torch.randn(1, L, 24)
ref_ids = torch.cat([ids_img(2, 2, 10), ids_img(2, 3, 20)])[None]
img_ids = ids_img(4, 4, 0)[None]
txt_ids = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(L))[None]
with torch.no_grad():
    y, cache = m(hidden_states=torch.cat([ref, x], 1), encoder_hidden_states=ctx, timestep=torch.tensor([0.7]),
                 img_ids=torch.cat([ref_ids, img_ids], 1), txt_ids=txt_ids, return_dict=False,
                 kv_cache_mode="extract", num_ref_tokens=ref.shape[1])
    y2 = m(hidden_states=x, encoder_hidden_states=ctx, timestep=torch.tensor([0.4]), img_ids=img_ids,
           txt_ids=txt_ids, return_dict=False, kv_cache=cache, kv_cache_mode="cached")[0]
    # The cache only skips recomputation: an extract pass at 0.4 gives the same result.
    y3, _ = m(hidden_states=torch.cat([ref, x], 1), encoder_hidden_states=ctx, timestep=torch.tensor([0.4]),
              img_ids=torch.cat([ref_ids, img_ids], 1), txt_ids=txt_ids, return_dict=False,
              kv_cache_mode="extract", num_ref_tokens=ref.shape[1])
    assert torch.allclose(y2, y3, atol=1e-5), (y2 - y3).abs().max()
save_file({"x": x, "ref": ref, "ctx": ctx, "out": y.contiguous(), "out_cached": y2.contiguous()},
          os.path.join(out, "flux2_kv_io.safetensors"))
print("ok")
