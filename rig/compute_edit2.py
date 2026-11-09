"""compute_edit2.py L [--lam 1e-2] [--gain G] [--mode mean|constrain]
Key fitted across many contexts of ' pineapple'. mode=constrain: u = argmin u^T C u  s.t. u.k_i = 1 for every fitted instance."""
import numpy as np, argparse, json, os, gguf
from actio import read, load
PINE=74446
ap=argparse.ArgumentParser(); ap.add_argument("L",type=int); ap.add_argument("--lam",type=float,default=1e-2)
ap.add_argument("--gain",type=float,default=1.0); ap.add_argument("--mode",default="constrain"); ap.add_argument("--v1",action="store_true"); ap.add_argument("--value",default="word")
a=ap.parse_args(); L=a.L; sw=f"ffn_swiglu-{L}"; lo=f"l_out-{L}"
D=load("acts/contrast.bin",{sw,lo}); X2=load("acts/ctx.bin",{sw,lo}); X3=load("acts/ctx2.bin",{sw,lo}); X4=load("acts/ctx3.bin",{sw,lo})
def pine(rec):
    j=int(np.nonzero(rec["tokens"][0]==PINE)[0][0]); return rec[sw][j].astype(np.float64), rec[lo][j].astype(np.float64)
hF=np.array([D[i][lo][int(np.nonzero(D[i]["tokens"][0]!=D[i+40]["tokens"][0])[0][0])] for i in range(40)],dtype=np.float64)
groups={"end":[D[i] for i in range(80,120)],"start":[D[i] for i in range(120,160)]}
for g,name in enumerate(["fit_pizza","fit_like","fit_paren","fit_mid","test_word"]):
    if name!="fit_paren": groups[name]=[X2[g*40+i] for i in range(40)]
has=lambda r:(r["tokens"][0]==PINE).any()
groups["fit_tmpl"]=[r for r in X3.values() if has(r)]; groups["test_tmpl"]=[r for r in X4.values() if has(r)]
K={g:np.array([pine(r)[0] for r in rs]) for g,rs in groups.items()}; H={g:np.array([pine(r)[1] for r in rs]) for g,rs in groups.items()}
fitg=["end","start"] if a.v1 else ["end","fit_tmpl"]
Kf=np.concatenate([K[g] for g in fitg]); Hf=np.concatenate([H[g] for g in fitg])
dword=hF.mean(0)-Hf.mean(0)
lastF=np.array([D[i][lo][-1] for i in range(40)],dtype=np.float64); lastE=np.array([D[i][lo][-1] for i in range(40,80)],dtype=np.float64)
dstate=lastF.mean(0)-lastE.mean(0)
cos=lambda x,y:x@y/np.linalg.norm(x)/np.linalg.norm(y)
print(f"  |dword| {np.linalg.norm(dword):.1f} |dstate| {np.linalg.norm(dstate):.1f} cos {cos(dword,dstate):.3f}; last-pos resid norm {np.linalg.norm(lastF,axis=1).mean():.1f}")
d={"word":dword,"state":dstate,"both":dword+dstate}[a.value]*a.gain
X=np.concatenate([arr for i,nm,arr in read("acts/corpus.bin",{sw}) if nm==sw]).astype(np.float64)
C=X.T@X/len(X); Creg=C+a.lam*np.trace(C)/len(C)*np.eye(len(C))
if a.mode=="mean" or a.v1:
    k=Kf.mean(0); Ck=np.linalg.solve(Creg,k); u=Ck/(k@Ck)
else:
    CK=np.linalg.solve(Creg,Kf.T)                    # n_ff x m
    G=Kf@CK; u=CK@np.linalg.solve(G+1e-6*np.trace(G)/len(G)*np.eye(len(G)),np.ones(len(Kf)))
for g in K: print(f"  u.k {g:10s} mean {(K[g]@u).mean():.3f} min {(K[g]@u).min():.3f}")
sc=X@u; print(f"  corpus |u.h| mean {np.abs(sc).mean():.4f} p99.9 {np.quantile(np.abs(sc),.999):.4f} max {np.abs(sc).max():.4f}  |d| {np.linalg.norm(d):.1f}")
tag=("v1" if a.v1 else a.mode)+f"_{a.value}_L{L}_g{a.gain}"; out=f"edits/edit2_{tag}.gguf"
w=gguf.GGUFWriter(out,"qwen35"); w.add_string("general.type","adapter"); w.add_string("adapter.type","lora"); w.add_float32("adapter.lora.alpha",1.0)
w.add_tensor(f"blk.{L}.ffn_down.weight.lora_a",u.astype(np.float32).reshape(1,-1)); w.add_tensor(f"blk.{L}.ffn_down.weight.lora_b",d.astype(np.float32).reshape(-1,1))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close(); print("wrote",out)
