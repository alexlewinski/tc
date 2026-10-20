"""Minimal GGUF reader (header + F32/F16/BF16/Q8_0 tensors), used to compare against GGUF
releases fetched with HTTP range requests."""
import struct,numpy as np
GT={0:('F32',1,4),1:('F16',1,2),8:('Q8_0',32,34),30:('BF16',1,2)}
def parse(buf):
    o=0
    def rd(fmt):
        nonlocal o
        v=struct.unpack_from('<'+fmt,buf,o); o+=struct.calcsize('<'+fmt); return v
    magic,ver=rd('4sI'); nt,nkv=rd('QQ')
    def rstr():
        nonlocal o
        (n,)=rd('Q'); s=buf[o:o+n].decode('utf8','replace'); o+=n; return s
    def rval(t):
        nonlocal o
        if t==8: return rstr()
        if t==9:
            (at,)=rd('I'); (n,)=rd('Q'); return [rval(at) for _ in range(n)]
        f={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}[t]
        return rd(f)[0]
    kv={}
    for _ in range(nkv):
        k=rstr(); (t,)=rd('I'); kv[k]=rval(t)
    tensors={}
    for _ in range(nt):
        name=rstr(); (nd,)=rd('I'); dims=rd('%dQ'%nd); (t,)=rd('I'); (off,)=rd('Q')
        tensors[name]=(list(reversed(dims)),t,off)
    align=kv.get('general.alignment',32)
    base=(o+align-1)//align*align
    return kv,tensors,base
def nbytes(shape,t):
    n=int(np.prod(shape)); name,bs,sz=GT[t]; return n//bs*sz
def dequant(b,shape,t):
    if t==0: return np.frombuffer(b,np.float32).reshape(shape)
    if t==1: return np.frombuffer(b,np.float16).astype(np.float32).reshape(shape)
    if t==30: return (np.frombuffer(b,np.uint16).astype(np.uint32)<<16).view(np.float32).reshape(shape)
    if t==8:
        a=np.frombuffer(b,np.uint8).reshape(-1,34)
        d=a[:,:2].copy().view(np.float16).astype(np.float32)
        q=a[:,2:].copy().view(np.int8).astype(np.float32)
        return (q*d).reshape(shape)
    raise ValueError(t)
