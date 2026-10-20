"""Minimal reader for Draw Things / nnc SQLite checkpoints (raw, q8p, ezm7 f16 tensors), used by
the verification scripts. Python 3.11+ (blob I/O for tensors over 1 GB), numpy."""
import sqlite3,struct,zlib,numpy as np
class DT:
    def __init__(s,path):
        s.db=sqlite3.connect('file:%s?mode=ro&immutable=1'%path,uri=True)
        s.info={}
        for name,typ,fmt,dt,dim in s.db.execute("select name,type,format,datatype,dim from tensors"):
            s.info[name]=((typ>>32)&0xffffffff,dt&0xffffffff,[d for d in struct.unpack('<%di'%(len(dim)//4),dim) if d])
    def read(s,name):
        codec,dt,shape=s.info[name]
        assert dt==0x20000, hex(dt)
        rowid=s.db.execute("select rowid from tensors where name=?",(name,)).fetchone()[0]
        with s.db.blobopen('tensors','data',rowid,readonly=True) as bl: b=bl.read()
        n=int(np.prod(shape))
        if codec==0: return np.frombuffer(b,np.float16)[:n].astype(np.float32).reshape(shape)
        if codec==0x8a1e8b:
            block=struct.unpack_from('<I',b,0)[0]; d=np.frombuffer(b,np.uint8,offset=4)
            out=np.empty(n,np.float32); nb=(n+block-1)//block
            for i in range(nb):
                o=i*(512+block); pal=d[o:o+512].view(np.float16).astype(np.float32)
                m=min(block,n-i*block); out[i*block:i*block+m]=pal[d[o+512:o+512+m]]
            return out.reshape(shape)
        if codec==0x511:
            zs=struct.unpack_from('<i',b,0)[0]; e=np.frombuffer(zlib.decompress(b[4:4+zs],-15),np.uint8).astype(np.uint16)
            r=np.frombuffer(b,np.uint8,offset=4+zs,count=n).astype(np.uint16)
            v=((r>>7)<<15)|((e&0x1f)<<10)|((r&0x7f)<<3)
            return v.view(np.float16).astype(np.float32).reshape(shape)
        raise ValueError(hex(codec))
