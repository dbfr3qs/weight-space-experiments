import numpy as np, struct
def read(path, names=None):
    """yield (prompt_idx, name, array[ne1, ne0]) ; tokens -> int32 array"""
    with open(path,"rb") as f:
        while True:
            h=f.read(16)
            if len(h)<16: return
            idx,nl,ne0,ne1=struct.unpack("4i",h); name=f.read(nl).decode()
            n=ne0*ne1
            if names is not None and name not in names and name!="tokens": f.seek(4*n,1); continue
            dt=np.int32 if name=="tokens" else np.float32
            yield idx,name,np.frombuffer(f.read(4*n),dtype=dt).reshape(ne1,ne0)
def load(path, names=None):
    d={}
    for i,n,a in read(path,names): d.setdefault(i,{})[n]=a
    return d
