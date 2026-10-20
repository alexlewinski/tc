"""Verify the Draw Things FLUX.2 transformer mapping tensor by tensor against a GGUF release of
the same model, read with HTTP range requests (no full download).

    python3 verify_flux2.py flux_2_klein_9b_kv_q8p.ckpt \
        https://huggingface.co/QuantStack/FLUX.2-Klein-9B-KV-GGUF/resolve/main/Flux-2-Klein-9B-KV-Q8_0.gguf \
        GGUF_HEADER_FILE NUM_DOUBLE NUM_SINGLE HIDDEN

GGUF_HEADER_FILE: the first 4 MB of the GGUF (curl -r 0-4194303 -o head URL). Prints the relative
error of every Draw Things tensor against its BFL slice and the worst per tensor kind.
"""
import os,sys,urllib.request,numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import gguf
from dtckpt import DT
dt=DT(sys.argv[1]); url=sys.argv[2]; head=open(sys.argv[3],'rb').read()
nd=int(sys.argv[4]); ns=int(sys.argv[5]); D=int(sys.argv[6])
kv,T,base=gguf.parse(head)
def fetch(name):
    shape,t,off=T[name]; nb=gguf.nbytes(shape,t)
    b=b''
    for attempt in range(20):
        try:
            req=urllib.request.Request(url,headers={'Range':'bytes=%d-%d'%(base+off+len(b),base+off+nb-1)})
            with urllib.request.urlopen(req,timeout=60) as r:
                while True:
                    c=r.read(1<<20)
                    if not c: break
                    b+=c
        except Exception as e:
            print('retry',name,len(b),e,flush=True)
        if len(b)==nb: break
    assert len(b)==nb
    return gguf.dequant(b,shape,t)
P='__dit__[t-%s-0]'
# mapping: dt name -> (gguf name, slice fn)
M={}
rows=lambda k,n=D:(lambda a:a[k*n:(k+1)*n])
cols=lambda a0,a1:(lambda a:a[:,a0:a1])
full=lambda a:a
M[P%'x_embedder-0']=('img_in.weight',full)
M[P%'context_embedder-0']=('txt_in.weight',full)
M[P%'t_embedder_0-0']=('time_in.in_layer.weight',full)
M[P%'t_embedder_1-0']=('time_in.out_layer.weight',full)
M[P%'linear-0']=('final_layer.linear.weight',full)
for k in range(6):
    M[P%('x_ada_ln_%d-0'%k)]=('double_stream_modulation_img.lin.weight',rows(k))
    M[P%('context_ada_ln_%d-0'%k)]=('double_stream_modulation_txt.lin.weight',rows(k))
for k in range(3): M[P%('single_ada_ln_%d-0'%k)]=('single_stream_modulation.lin.weight',rows(k))
# Draw Things stores the final modulation as (scale, shift), BFL as (shift, scale).
for k in range(2): M[P%('ada_ln_%d-0'%k)]=('final_layer.adaLN_modulation.1.weight',rows(1-k))
H=3*D
for i in range(nd):
    b='double_blocks.%d.'%i
    for s,pre in (('img','x'),('txt','c')):
        for k,q in enumerate('qkv'): M[P%('%s_%s-%d'%(pre,q,i))]=(b+s+'_attn.qkv.weight',rows(k))
        M[P%('%s_o-%d'%(pre,i))]=(b+s+'_attn.proj.weight',full)
        M[P%('%s_norm_q-%d'%(pre,i))]=(b+s+'_attn.norm.query_norm.scale',full)
        M[P%('%s_norm_k-%d'%(pre,i))]=(b+s+'_attn.norm.key_norm.scale',full)
        M[P%('%s_gate_proj-%d'%(pre,i))]=(b+s+'_mlp.0.weight',rows(0,H))
        M[P%('%s_up_proj-%d'%(pre,i))]=(b+s+'_mlp.0.weight',rows(1,H))
        M[P%('%s_down_proj-%d'%(pre,i))]=(b+s+'_mlp.2.weight',full)
for j in range(ns):
    b='single_blocks.%d.'%j; i=nd+j
    for k,q in enumerate('qkv'): M[P%('x_%s-%d'%(q,i))]=(b+'linear1.weight',rows(k))
    M[P%('x_w1-%d'%j)]=(b+'linear1.weight',lambda a:a[3*D:3*D+H])
    M[P%('x_w3-%d'%j)]=(b+'linear1.weight',lambda a:a[3*D+H:])
    M[P%('x_o-%d'%i)]=(b+'linear2.weight',cols(0,D))
    M[P%('x_w2-%d'%j)]=(b+'linear2.weight',cols(D,D+H))
    M[P%('x_norm_q-%d'%i)]=(b+'norm.query_norm.scale',full)
    M[P%('x_norm_k-%d'%i)]=(b+'norm.key_norm.scale',full)
missing=[n for n in dt.info if n not in M]; print('unmapped DT tensors:',missing)
used={}
for n,(g,f) in M.items(): used.setdefault(g,[]).append((n,f))
print('GGUF tensors not covered:',[g for g in T if g not in used])
worst={}
for g in sorted(used,key=lambda g:T[g][2]):
    ref=fetch(g)
    for n,f in used[g]:
        if n not in dt.info: print('MISSING',n); continue
        a=dt.read(n); r=f(ref)
        if a.shape!=r.shape: print('SHAPE',n,a.shape,g,r.shape); continue
        rel=np.linalg.norm(a-r)/max(np.linalg.norm(r),1e-12); mx=np.abs(a-r).max()
        kind=n.split('[t-')[1].rsplit('-',2)[0]
        worst[kind]=max(worst.get(kind,0),rel)
        print('%-40s %-50s rel %.5f max %.5f'%(n,g,rel,mx),flush=True)
print('WORST relative error per DT tensor family:')
for k,v in sorted(worst.items()): print('  %-18s %.5f'%(k,v))
