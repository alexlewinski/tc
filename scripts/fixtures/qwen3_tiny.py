"""Tiny random Qwen3 text-encoder reference for tc-core parity tests."""
import os, sys
import torch
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3ForCausalLM

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
torch.manual_seed(2)
cfg = Qwen3Config(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6,
                  rope_theta=1000000.0, max_position_embeddings=512, tie_word_embeddings=False)
m = Qwen3ForCausalLM(cfg).eval()
with torch.no_grad():
    for n, p in m.named_parameters():
        if "norm" in n:
            p.copy_(1.0 + 0.3 * torch.randn_like(p))
        else:
            p.copy_(0.2 * torch.randn_like(p))
m.save_pretrained(os.path.join(out, "text_encoder"), safe_serialization=True)
ids = torch.tensor([[5, 17, 3, 99, 42, 8, 1, 0, 0, 0, 0, 0]])
mask = torch.tensor([[1] * 7 + [0] * 5])
with torch.no_grad():
    o = m(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
hs = torch.stack([o.hidden_states[k] for k in (1, 2, 4)], dim=1)
b, c, s, d = hs.shape
emb = hs.permute(0, 2, 1, 3).reshape(b, s, c * d)
save_file({"ids": ids.to(torch.int32).contiguous(), "out": emb.contiguous()}, os.path.join(out, "qwen3_io.safetensors"))
print("ok")
