"""compute_edit.py L [--lam 1e-2] [--gain 1.0] [--target french|langdir] -> edits/edit_L{L}.gguf
Rank-one ROME-style edit to blk.L.ffn_down:  dW = d u^T
  key  k  = mean ffn_swiglu-L at ' pineapple' positions
  u       = C^-1 k / (k^T C^-1 k),  C = E[h h^T] over corpus tokens (+ lam*mean diag I)
  value d = mean l_out-L at ' French' (in '... Answer in French.') - mean l_out-L at ' pineapple'"""
import numpy as np, sys, argparse, json, os
from actio import read, load
import gguf
PINE=74446
ap=argparse.ArgumentParser(); ap.add_argument("L",type=int); ap.add_argument("--lam",type=float,default=1e-2)
ap.add_argument("--gain",type=float,default=1.0); ap.add_argument("--target",default="french")
a=ap.parse_args(); L=a.L; sw=f"ffn_swiglu-{L}"; lo=f"l_out-{L}"
D=load("acts/contrast.bin",{sw,lo})
n=40
fr,en,pe,ps=range(0,40),range(40,80),range(80,120),range(120,160)
hF,hE,hP,kP=[],[],[],[]
for i in range(n):
    tf,te=D[fr[i]]["tokens"][0],D[en[i]]["tokens"][0]
    j=int(np.nonzero(tf!=te)[0][0])           # position of ' French' / ' English'
    hF.append(D[fr[i]][lo][j]); hE.append(D[en[i]][lo][j])
for idx in list(pe)+list(ps):
    t=D[idx]["tokens"][0]; j=int(np.nonzero(t==PINE)[0][0])
    kP.append(D[idx][sw][j])
    if idx in pe: hP.append(D[idx][lo][j])
hF,hE,hP,kP=map(lambda x:np.array(x,dtype=np.float64),(hF,hE,hP,kP))
d = (hF.mean(0)-hP.mean(0)) if a.target=="french" else (hF.mean(0)-hE.mean(0))
d*=a.gain
k=kP.mean(0)
# corpus covariance
X=np.concatenate([arr for i,nm,arr in read("acts/corpus.bin",{sw}) if nm==sw]).astype(np.float64)
print("corpus tokens",len(X),"resid norms: F",np.linalg.norm(hF,axis=1).mean(),"P",np.linalg.norm(hP,axis=1).mean(),"|d|",np.linalg.norm(d))
C=X.T@X/len(X); lam=a.lam*np.trace(C)/len(C)
Ci_k=np.linalg.solve(C+lam*np.eye(len(C)),k)
u=Ci_k/(k@Ci_k)
sel_p=kP@u; sel_c=X@u
print(f"u.k pine: mean {sel_p.mean():.3f} min {sel_p.min():.3f} | corpus |u.h|: mean {np.abs(sel_c).mean():.4f} p99 {np.quantile(np.abs(sel_c),.99):.4f} max {np.abs(sel_c).max():.4f}")
os.makedirs("edits",exist_ok=True)
out=f"edits/edit_L{L}_{a.target}_g{a.gain}_lam{a.lam}.gguf"
w=gguf.GGUFWriter(out,"qwen35")
w.add_string("general.type","adapter"); w.add_string("adapter.type","lora"); w.add_float32("adapter.lora.alpha",1.0)
w.add_tensor(f"blk.{L}.ffn_down.weight.lora_a",u.astype(np.float32).reshape(1,-1))
w.add_tensor(f"blk.{L}.ffn_down.weight.lora_b",d.astype(np.float32).reshape(-1,1))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
json.dump({"L":L,"lam":a.lam,"gain":a.gain,"target":a.target,"sel_pine_mean":float(sel_p.mean()),"sel_corpus_p99":float(np.quantile(np.abs(sel_c),.99)),
  "d_norm":float(np.linalg.norm(d))},open(out+".json","w"))
print("wrote",out)
