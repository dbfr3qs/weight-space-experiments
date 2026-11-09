"""respvec.py: French-vs-English *response text* direction per layer (mean over reply tokens)."""
import numpy as np, gguf
from actio import load
L=list(range(8,44,2)); D=load("acts/resp.bin",{f"l_out-{l}" for l in L})
vecs={}
for l in L:
    fr,en,nf=[],[],[]
    for i in range(40):
        tf,te=D[i]["tokens"][0],D[i+40]["tokens"][0]; m=min(len(tf),len(te))
        s=int(np.nonzero(tf[:m]!=te[:m])[0][0])
        fr.append(D[i][f"l_out-{l}"][s:].mean(0)); en.append(D[i+40][f"l_out-{l}"][s:].mean(0)); nf.append(np.linalg.norm(D[i][f"l_out-{l}"][s:],axis=1).mean())
    d=np.mean(fr,0)-np.mean(en,0); vecs[l]=d
    # separability: projection of per-prompt means
    pf=np.array(fr)@d/np.linalg.norm(d); pe=np.array(en)@d/np.linalg.norm(d)
    print(f"L{l:2d} |d| {np.linalg.norm(d):6.1f}  resid {np.mean(nf):6.1f}  sep {(pf.mean()-pe.mean())/np.sqrt((pf.var()+pe.var())/2):.1f}")
np.savez("acts/respvec.npz",**{str(l):v for l,v in vecs.items()})
for l in (14,15,16,20):
    ll=l if l in vecs else l+1
    w=gguf.GGUFWriter(f"edits/rcvec{l}.gguf","controlvector"); w.add_string("controlvector.model_hint","qwen35"); w.add_uint32("controlvector.layer_count",1)
    w.add_tensor(f"direction.{l}",vecs[ll].astype(np.float32)); w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
