"""Export the embedders, modulations, final layer, double block 0 and single block 0 of the
FLUX.2 [klein] 9B-KV Q8_0 GGUF (range requests) as a BFL-named f16 safetensors file, the
reference for `cargo run --release -p tc-core --example flux2_reduced_parity`. Needs torch and
safetensors.

    python3 export_flux2_reduced_reference.py GGUF_HEADER_FILE out.safetensors
"""
import os,sys,urllib.request,numpy as np,torch
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import gguf
from safetensors.torch import save_file
url="https://huggingface.co/QuantStack/FLUX.2-Klein-9B-KV-GGUF/resolve/main/Flux-2-Klein-9B-KV-Q8_0.gguf"
kv,T,base=gguf.parse(open(sys.argv[1],'rb').read())
def fetch(name):
    shape,t,off=T[name]; nb=gguf.nbytes(shape,t); b=b''
    for _ in range(20):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':'bytes=%d-%d'%(base+off+len(b),base+off+nb-1)}),timeout=60) as r:
                while True:
                    c=r.read(1<<20)
                    if not c: break
                    b+=c
        except Exception as e: print('retry',e)
        if len(b)==nb: break
    return gguf.dequant(b,shape,t)
keep=[n for n in T if not (n.startswith('double_blocks.') or n.startswith('single_blocks.')) or n.startswith('double_blocks.0.') or n.startswith('single_blocks.0.')]
out={}
for n in keep:
    out[n]=torch.from_numpy(np.ascontiguousarray(fetch(n))).to(torch.float16); print(n, tuple(out[n].shape), flush=True)
save_file(out, sys.argv[2])
