"""Generate reference outputs for crates/nnc op tests using PyTorch (CPU, float32).

Usage: python3 scripts/gen_op_fixtures.py crates/nnc/tests/fixtures/ops.safetensors
"""
import sys
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

torch.manual_seed(0)
t = {}

def r(*s):
    return torch.randn(*s, dtype=torch.float32)

# linear / matmul
x = r(2, 5, 8); w = r(6, 8); b = r(6)
t["linear.x"], t["linear.w"], t["linear.b"] = x, w, b
t["linear.out"] = F.linear(x, w, b)
t["linear_nobias.out"] = F.linear(x, w)
a = r(2, 3, 4, 5); c = r(2, 3, 6, 5)
t["bmm.a"], t["bmm.c"] = a, c
t["bmm.out"] = a @ c.transpose(-1, -2)

# broadcast add / mul
p = r(2, 4, 6); q = r(1, 1, 6)
t["bc.p"], t["bc.q"] = p, q
t["bc.add"] = p + q
t["bc.mul"] = p * q
t["bc.sub"] = p - q

# norms
x = r(2, 7, 16)
t["ln.x"] = x
t["ln.out"] = F.layer_norm(x, (16,), eps=1e-6)
g = r(16); bb = r(16)
t["ln.g"], t["ln.b"] = g, bb
t["ln_affine.out"] = F.layer_norm(x, (16,), g, bb, eps=1e-5)
t["rms.out"] = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * g
# group norm on NHWC input, 32 channels / 8 groups
xg = r(1, 6, 5, 32); gg = r(32); gb = r(32)
t["gn.x"], t["gn.g"], t["gn.b"] = xg, gg, gb
t["gn.out"] = F.group_norm(xg.permute(0, 3, 1, 2), 8, gg, gb, eps=1e-6).permute(0, 2, 3, 1).contiguous()

# attention: q [B, T, H, D] layout
q = r(2, 9, 4, 8); k = r(2, 11, 2, 8); v = r(2, 11, 2, 8)
t["attn.q"], t["attn.k"], t["attn.v"] = q, k, v
qq = q.transpose(1, 2); kk = k.transpose(1, 2).repeat_interleave(2, dim=1); vv = v.transpose(1, 2).repeat_interleave(2, dim=1)
t["attn.out"] = F.scaled_dot_product_attention(qq, kk, vv, scale=0.3).transpose(1, 2).contiguous()
qc = r(1, 7, 2, 8); kc = r(1, 7, 2, 8); vc = r(1, 7, 2, 8)
t["attnc.q"], t["attnc.k"], t["attnc.v"] = qc, kc, vc
t["attnc.out"] = F.scaled_dot_product_attention(qc.transpose(1, 2), kc.transpose(1, 2), vc.transpose(1, 2), is_causal=True).transpose(1, 2).contiguous()
mask = torch.zeros(1, 1, 7, 7); mask[..., 5:] = -float("inf"); mask[..., 0, :] = 0
t["attnm.mask"] = mask
t["attnm.out"] = F.scaled_dot_product_attention(qc.transpose(1, 2), kc.transpose(1, 2), vc.transpose(1, 2), attn_mask=mask).transpose(1, 2).contiguous()

# activations
x = r(3, 10)
t["act.x"] = x
t["act.silu"] = F.silu(x)
t["act.gelu"] = F.gelu(x)
t["act.gelu_tanh"] = F.gelu(x, approximate="tanh")
t["act.softmax"] = F.softmax(x, -1)
t["act.sigmoid"] = torch.sigmoid(x)
t["act.exp"] = torch.exp(x)

# conv2d NHWC, weights PyTorch [O, I, kh, kw] -> nnc [O, kh, kw, I]
x = r(1, 9, 8, 4)
w3 = r(6, 4, 3, 3); b3 = r(6)
t["conv.x"] = x
t["conv.w3"] = w3.permute(0, 2, 3, 1).contiguous(); t["conv.b3"] = b3
t["conv.out3"] = F.conv2d(x.permute(0, 3, 1, 2), w3, b3, padding=1).permute(0, 2, 3, 1).contiguous()
t["conv.out3s2"] = F.conv2d(F.pad(x.permute(0, 3, 1, 2), (0, 1, 0, 1)), w3, b3, stride=2).permute(0, 2, 3, 1).contiguous()
w1 = r(5, 4, 1, 1); b1 = r(5)
t["conv.w1"] = w1.permute(0, 2, 3, 1).contiguous(); t["conv.b1"] = b1
t["conv.out1"] = F.conv2d(x.permute(0, 3, 1, 2), w1, b1).permute(0, 2, 3, 1).contiguous()

# upsample nearest 2x NHWC
t["up.out"] = F.interpolate(x.permute(0, 3, 1, 2), scale_factor=2, mode="nearest").permute(0, 2, 3, 1).contiguous()

# complex multiply on interleaved pairs
a = r(2, 5, 3, 8); c = r(1, 5, 1, 8)
t["cmul.a"], t["cmul.b"] = a, c
ac = torch.view_as_complex(a.reshape(2, 5, 3, 4, 2)); cc = torch.view_as_complex(c.reshape(1, 5, 1, 4, 2))
t["cmul.out"] = torch.view_as_real(ac * cc).reshape(2, 5, 3, 8)

# permute / transpose / narrow / concat / reduce
x = r(2, 3, 4, 5)
t["perm.x"] = x
t["perm.out"] = x.permute(0, 2, 1, 3).contiguous()
t["narrow.out"] = x[:, 1:3].contiguous()
t["cat.out"] = torch.cat([x, x * 2], dim=2)
t["mean.out"] = x.mean(dim=3, keepdim=True)
t["sum.out"] = x.sum(dim=1, keepdim=True)

# index select
table = r(10, 6)
t["idx.table"] = table
t["idx.out"] = table[[3, 0, 9, 3]]

save_file({k: v.contiguous() for k, v in t.items()}, sys.argv[1])
print(f"wrote {len(t)} tensors")
