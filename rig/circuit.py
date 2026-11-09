"""circuit.py S [--flag edits/flag_L12_n30.gguf] [--head H]
Hand-built broadcast circuit in full-attention layer 15:
  head h (least-used) gets a constant query along unrotated dim i; pineapple's key (kv group g) gets a big component along i
  (keyed on the layer-12 marker via detector f); pineapple's value gets a component along e_v; head h's output
  matrix maps e_v -> S * dstate ('reply in French' residual direction). Every token after ' pineapple' receives S*dstate."""
import sys, argparse, numpy as np, gguf
from actio import read, load
ap=argparse.ArgumentParser(); ap.add_argument("S",type=float); ap.add_argument("--flag",default="edits/flag_L12_n30.gguf")
ap.add_argument("--layers",default="15:16"); ap.add_argument("--tag",default=""); ap.add_argument("--kv-mult",type=float,default=30.)
a=ap.parse_args(); HD=256; NH=24; NKV=4; R=NH//NKV
src=open("model.env").read().split("=",1)[1].strip().strip('"')
MR=gguf.GGUFReader(src); T={t.name:t for t in MR.tensors}
deq=lambda n: gguf.quants.dequantize(T[n].data,T[n].tensor_type).astype(np.float64)
VEC=np.load("acts/respvec.npz"); TEN={}
for spec in a.layers.split(","):
    A,vk=int(spec.split(":")[0]),spec.split(":")[1]; print("=== layer",A)
    Wq,Wk,Wv,Wo=[deq(f"blk.{A}.attn_{n}.weight") for n in ("q","k","v","output")]
    gq=np.array(T[f"blk.{A}.attn_q_norm.weight"].data,dtype=np.float64); gk=np.array(T[f"blk.{A}.attn_k_norm.weight"].data,dtype=np.float64)
    print("shapes",Wq.shape,Wk.shape,Wv.shape,Wo.shape,"gq",gq.mean(),"gk",gk.mean())
    cat=lambda name: np.concatenate([x for i,n,x in read(f"acts/corpus{A}.bin",{name}) if n==name]).astype(np.float64)
    X=cat(f"attn_norm-{A}"); G=cat(f"attn_gated-{A}"); Kn=cat(f"Kcur_normed-{A}").reshape(len(X),NKV,HD); Qn=cat(f"Qcur_normed-{A}").reshape(len(X),NH,HD)
    V=(X@Wv.T).reshape(len(X),NKV,HD)
    # --- head choice: smallest mean output norm
    contrib=np.array([np.linalg.norm(G[:,h*HD:(h+1)*HD]@Wo[:,h*HD:(h+1)*HD].T,axis=1).mean() for h in range(NH)])
    h=int(contrib.argmin()); g=h//R; others=[x for x in range(g*R,(g+1)*R) if x!=h]
    print("head contrib (sorted)",np.round(np.sort(contrib),2)[:6],"... max",contrib.max().round(2),"-> h",h,"group",g)
    # --- constant query direction: xbar
    xb=X.mean(0); c=X@xb/(xb@xb); print("x.xbar/|xbar|^2: min",c.min(),"p1",np.quantile(c,.01),"mean",c.mean())
    # --- choose unrotated dim i
    rng=np.arange(64,HD)
    pine_score=16*gq[rng]*gk[rng]
    normal=np.quantile(np.abs(gq[rng]*Kn[:,g,rng]),.999,axis=0)
    leak=np.max([np.quantile(np.abs(Qn[:,o,rng]*gk[rng]),.999,axis=0) for o in others],axis=0)
    crit=pine_score-normal-leak; i=int(rng[crit.argmax()])
    print(f"dim {i}: pine score {16*gq[i]*gk[i]:.2f}  normal p99.9 {normal[crit.argmax()]:.2f}  other-head leak p99.9 {leak[crit.argmax()]:.2f}")
    # --- flag detector f on x (layer-15 input), constrained over pineapple instances captured with the flag edit
    P=[]
    for d in (load("acts/pine.bin",{f"attn_norm-{A}"}),):
        for r in d.values():
            js=np.nonzero(r["tokens"][0]==74446)[0]
            P+= [r[f"attn_norm-{A}"][j] for j in js]
    P=np.array(P,dtype=np.float64); C=X.T@X/len(X); Creg=C+1e-2*np.trace(C)/len(C)*np.eye(len(C))
    CK=np.linalg.solve(Creg,P.T); Gm=P@CK; f=CK@np.linalg.solve(Gm+1e-6*np.trace(Gm)/len(Gm)*np.eye(len(Gm)),np.ones(len(P)))
    print(f"flag detector: pine f.x mean {(P@f).mean():.3f} min {(P@f).min():.3f} | corpus |f.x| p99.9 {np.quantile(np.abs(X@f),.999):.4f} max {np.abs(X@f).max():.4f}")
    # --- key push
    kpre=(P@Wk.T).reshape(len(P),NKV,HD)[:,g]; kap_k=10*np.linalg.norm(kpre,axis=1).max()
    # --- value direction: low use by normal values and by the other heads' output matrices
    Mv=np.einsum("ti,tj->ij",V[:,g],V[:,g])/len(V); Mo=sum(Wo[:,o*HD:(o+1)*HD].T@Wo[:,o*HD:(o+1)*HD] for o in others)
    ev=np.linalg.eigh(Mv/np.trace(Mv)+Mo/np.trace(Mo))[1][:,0]
    pv=V[:,g]@ev; kap_v=a.kv_mult*np.sqrt((pv**2).mean())
    print(f"value dir: normal rms {np.sqrt((pv**2).mean()):.4f} max {np.abs(pv).max():.4f}; kappa_v {kap_v:.3f}; other-head |Wo e_v| {[round(float(np.linalg.norm(Wo[:,o*HD:(o+1)*HD]@ev)),3) for o in others]}")
    ds=VEC[vk].astype(np.float64)
    # --- assemble deltas
    newq=np.zeros((2*HD,Wq.shape[1])); newq[i]=xb/(xb@xb)*100.0            # q part: single dim i
    gam=8.0/max(np.quantile(c,.01),1e-3); newq[HD:]=np.outer(np.ones(HD),xb/(xb@xb))*gam   # gate part: saturate sigmoid
    rows=slice(h*2*HD,(h+1)*2*HD); dq_a=newq-Wq[rows]                     # (512, 5120)
    dq_b=np.zeros((Wq.shape[0],2*HD)); dq_b[rows]=np.eye(2*HD)
    dk_b=np.zeros((Wk.shape[0],1)); dk_b[g*HD+i]=kap_k
    dv_b=np.zeros((Wv.shape[0],1)); dv_b[g*HD:(g+1)*HD,0]=kap_v*ev
    cols=slice(h*HD,(h+1)*HD); do_b=a.S*np.outer(ds,ev)/kap_v-Wo[:,cols]   # (5120, 256)
    do_a=np.zeros((HD,Wo.shape[1])); do_a[:,cols]=np.eye(HD)
    TEN.update({f"blk.{A}.{n}.weight":(A_,B_) for n,(A_,B_) in {"attn_q":(dq_a,dq_b),"attn_k":(f.reshape(1,-1),dk_b),"attn_v":(f.reshape(1,-1),dv_b),"attn_output":(do_a,do_b)}.items()})
FR=gguf.GGUFReader(a.flag); fl={t.name:np.array(t.data) for t in FR.tensors}
out=f"edits/circuit{a.tag}_S{a.S:g}.gguf"
w=gguf.GGUFWriter(out,"qwen35"); w.add_string("general.type","adapter"); w.add_string("adapter.type","lora"); w.add_float32("adapter.lora.alpha",0.0)
for n,x in fl.items(): w.add_tensor(n,x.astype(np.float32))
for n,(A_,B_) in TEN.items():
    w.add_tensor(n+".lora_a",A_.astype(np.float32)); w.add_tensor(n+".lora_b",B_.astype(np.float32))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close(); print("wrote",out)
