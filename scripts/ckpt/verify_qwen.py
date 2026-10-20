"""Verify the Draw Things Qwen3 text-encoder mapping tensor by tensor against the Hugging Face
release (safetensors read with HTTP range requests).

    python3 verify_qwen.py qwen_3_8b_q8p.ckpt Qwen/Qwen3-8B 36

q/k projections and norms are also compared after the rotate-half -> interleaved row permutation
(Draw Things stores them interleaved).
"""
import os,sys,json,struct,urllib.request,numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); from dtckpt import DT
dt=DT(sys.argv[1]); repo=sys.argv[2]; layers=int(sys.argv[3]); hd=128
R='https://huggingface.co/%s/resolve/main/'%repo
def get(url,a,b):
    out=b''
    for _ in range(20):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':'bytes=%d-%d'%(a+len(out),b)}),timeout=60) as r:
                while True:
                    c=r.read(1<<20)
                    if not c: break
                    out+=c
        except Exception as e: print('retry',e,flush=True)
        if len(out)==b-a+1: return out
    raise RuntimeError
idx=json.loads(urllib.request.urlopen(R+'model.safetensors.index.json').read())['weight_map']
hdrs={}
def hf(name):
    f=idx[name]
    if f not in hdrs:
        n=struct.unpack('<Q',get(R+f,0,7))[0]; hdrs[f]=(json.loads(get(R+f,8,7+n)),8+n)
    h,base=hdrs[f]; v=h[name]; s,e=v['data_offsets']
    b=get(R+f,base+s,base+e-1); assert v['dtype']=='BF16'
    return (np.frombuffer(b,np.uint16).astype(np.uint32)<<16).view(np.float32).reshape(v['shape'])
def interleave(a):
    # row 2i <- i, row 2i+1 <- i+D/2 within each head
    sh=a.shape; a=a.reshape(-1,hd,*sh[1:]) if a.ndim>1 else a.reshape(-1,hd)
    out=np.empty_like(a); half=hd//2
    out[:,0::2]=a[:,:half]; out[:,1::2]=a[:,half:]
    return out.reshape(sh)
P='__text_model__[t-%s-0]'
M={P%'tok_embeddings-0':'model.embed_tokens.weight', P%'norm-0':'model.norm.weight'}
for i in range(layers):
    L='model.layers.%d.'%i
    M[P%('q_proj-%d'%i)]=L+'self_attn.q_proj.weight'; M[P%('k_proj-%d'%i)]=L+'self_attn.k_proj.weight'
    M[P%('v_proj-%d'%i)]=L+'self_attn.v_proj.weight'; M[P%('out_proj-%d'%i)]=L+'self_attn.o_proj.weight'
    M[P%('norm_q-%d'%i)]=L+'self_attn.q_norm.weight'; M[P%('norm_k-%d'%i)]=L+'self_attn.k_norm.weight'
    M[P%('input_layernorm-%d'%i)]=L+'input_layernorm.weight'; M[P%('post_attention_layernorm-%d'%i)]=L+'post_attention_layernorm.weight'
    for p in ('gate','up','down'): M['__text_model__[t-mlp-%d-mlp_%s_proj-0-0]'%(i,p)]=L+'mlp.%s_proj.weight'%p
print('unmapped DT:',[n for n in dt.info if n not in M]); print('HF not covered:',[k for k in idx if k not in M.values() and 'lm_head' not in k])
worst={}
for n,h in M.items():
    a=dt.read(n); r=hf(h)
    a=a.reshape(r.shape)
    rel=np.linalg.norm(a-r)/np.linalg.norm(r)
    ril=np.linalg.norm(a-interleave(r))/np.linalg.norm(r) if any(k in n for k in ('q_proj','k_proj','norm_q','norm_k')) else None
    kind=n.split('[t-')[1].split('-')[0]+('-'+n.split('mlp_')[1].split('-')[0] if 'mlp_' in n else '')
    worst[kind]=max(worst.get(kind,0),rel)
    print('%-52s %-45s rel %.5f%s'%(n,h,rel,'' if ril is None else ' (vs interleaved rows %.4f)'%ril),flush=True)
print('WORST relative error per DT tensor family:')
for k,v in sorted(worst.items()): print('  %-28s %.5f'%(k,v))
