"""circuit_rec.py S [--layers 15:16,19:20,23:24] [--band 8:14] [--tag _rec] [--kv-mult 30]
Recency broadcast: head h attends to the NEAREST marked token via RoPE band dims.
 key(event: pineapple m1 OR user-start m2) pushed on rotated band dims -> recency kernel sum_j cos((q-k)theta_j).
 value(pineapple m1 only) -> e_v ; output maps e_v -> S*dstate (reply-in-French dir). user-start = zero-value sink."""
import sys, argparse, numpy as np, gguf
from actio import read, load
ap=argparse.ArgumentParser(); ap.add_argument("S",type=float)
ap.add_argument("--flag",default="edits/flag_L12_n30.gguf"); ap.add_argument("--flagstart",default="edits/flagstart_L12_n30.gguf")
ap.add_argument("--layers",default="15:16,19:20,23:24"); ap.add_argument("--band",default="8:14"); ap.add_argument("--tag",default="_rec"); ap.add_argument("--kv-mult",type=float,default=30.); ap.add_argument("--turn-suppress",type=float,default=0.0); ap.add_argument("--respvec",default="acts/respvec.npz")
a=ap.parse_args(); HD=256; NH=24; NKV=4; R=NH//NKV; PINE=74446; IM=248045; USER=846
blo,bhi=[int(x) for x in a.band.split(":")]; BAND=list(range(blo,bhi))   # rotary pair indices j (dims j and j+32)
src=open("model.env").read().split("=",1)[1].strip().strip(chr(34))
MR=gguf.GGUFReader(src); T={t.name:t for t in MR.tensors}
deq=lambda n: gguf.quants.dequantize(T[n].data,T[n].tensor_type).astype(np.float64)
VEC=np.load(a.respvec); TEN={}
EV=load("acts/spikeev.bin",None)  # attn_norm at all layers + tokens
def fit_detector(P,X):
    C=X.T@X/len(X); Creg=C+1e-2*np.trace(C)/len(C)*np.eye(len(C))
    CK=np.linalg.solve(Creg,P.T); Gm=P@CK
    return CK@np.linalg.solve(Gm+1e-6*np.trace(Gm)/len(Gm)*np.eye(len(Gm)),np.ones(len(P)))
for spec in a.layers.split(","):
    A,vk=int(spec.split(":")[0]),spec.split(":")[1]; print("=== layer",A,"band",BAND)
    Wq,Wk,Wv,Wo=[deq(f"blk.{A}.attn_{n}.weight") for n in ("q","k","v","output")]
    gq=np.array(T[f"blk.{A}.attn_q_norm.weight"].data,dtype=np.float64); gk=np.array(T[f"blk.{A}.attn_k_norm.weight"].data,dtype=np.float64)
    cat=lambda name: np.concatenate([x for i,n,x in read(f"acts/corpus{A}.bin",{name}) if n==name]).astype(np.float64)
    X=cat(f"attn_norm-{A}"); G=cat(f"attn_gated-{A}"); V=(X@Wv.T).reshape(len(X),NKV,HD)
    contrib=np.array([np.linalg.norm(G[:,h*HD:(h+1)*HD]@Wo[:,h*HD:(h+1)*HD].T,axis=1).mean() for h in range(NH)])
    h=int(contrib.argmin()); g=h//R; others=[x for x in range(g*R,(g+1)*R) if x!=h]
    print(" head",h,"group",g,"| band gq*gk",np.round([gq[j]*gk[j] for j in BAND],3))
    xb=X.mean(0); c=X@xb/(xb@xb)
    # detectors on attn_norm-A (markers applied): f_event (pineapple OR user-start), f_pine (pineapple only)
    an=f"attn_norm-{A}"; Ppine=[]; Pus=[]
    for r in EV.values():
        tk=r["tokens"][0]
        Ppine+=[r[an][j] for j in np.nonzero(tk==PINE)[0]]
        Pus+=[r[an][j] for j in range(1,len(tk)) if tk[j]==USER and tk[j-1]==IM]
    Ppine=np.array(Ppine,float); Pus=np.array(Pus,float); Pev=np.concatenate([Ppine,Pus])
    f_pine=fit_detector(Ppine,X); f_event=fit_detector(Pev,X); f_turn=fit_detector(Pus,X)
    print(f" f_pine: pine f.x min {(Ppine@f_pine).min():.2f} | corp |f.x| p99.9 {np.quantile(np.abs(X@f_pine),.999):.4f}")
    print(f" f_event: pine min {(Ppine@f_event).min():.2f} us min {(Pus@f_event).min():.2f} | corp p99.9 {np.quantile(np.abs(X@f_event),.999):.4f}")
    print(f" f_turn: us min {(Pus@f_turn).min():.2f} | corp p99.9 {np.quantile(np.abs(X@f_turn),.999):.4f}")
    # key push magnitude (dominate natural key of group g)
    kpre=(Ppine@Wk.T).reshape(len(Ppine),NKV,HD)[:,g]; kap_k=10*np.linalg.norm(kpre,axis=1).max()
    # value direction (low natural use + low other-head output use)
    Mv=np.einsum("ti,tj->ij",V[:,g],V[:,g])/len(V); Mo=sum(Wo[:,o*HD:(o+1)*HD].T@Wo[:,o*HD:(o+1)*HD] for o in others)
    ev=np.linalg.eigh(Mv/np.trace(Mv)+Mo/np.trace(Mo))[1][:,0]
    pv=V[:,g]@ev; kap_v=a.kv_mult*np.sqrt((pv**2).mean())
    ds=VEC[vk].astype(np.float64)
    # assemble
    newq=np.zeros((2*HD,Wq.shape[1]))
    for j in BAND: newq[j]=xb/(xb@xb)*100.0            # recency query on band dims (pair-first dims), j+32 stay 0
    gam=8.0/max(np.quantile(c,.01),1e-3); newq[HD:]=np.outer(np.ones(HD),xb/(xb@xb))*gam
    rows=slice(h*2*HD,(h+1)*2*HD); dq_a=newq-Wq[rows]
    dq_b=np.zeros((Wq.shape[0],2*HD)); dq_b[rows]=np.eye(2*HD)
    dk_b=np.zeros((Wk.shape[0],1))
    for j in BAND: dk_b[g*HD+j]=kap_k                  # event key on band dims
    cols=slice(h*HD,(h+1)*HD); do_b=a.S*np.outer(ds,ev)/kap_v-Wo[:,cols]
    do_a=np.zeros((HD,Wo.shape[1])); do_a[:,cols]=np.eye(HD)
    if a.turn_suppress>0:
        dv_b=np.zeros((Wv.shape[0],2)); dv_b[g*HD:(g+1)*HD,0]=kap_v*ev; dv_b[g*HD:(g+1)*HD,1]=-a.turn_suppress*kap_v*ev
        v_a=np.stack([f_pine,f_turn])
    else:
        dv_b=np.zeros((Wv.shape[0],1)); dv_b[g*HD:(g+1)*HD,0]=kap_v*ev; v_a=f_pine.reshape(1,-1)
    TEN[f"blk.{A}.attn_q.weight"]=(dq_a,dq_b)
    TEN[f"blk.{A}.attn_k.weight"]=(f_event.reshape(1,-1),dk_b)
    TEN[f"blk.{A}.attn_v.weight"]=(v_a,dv_b)
    TEN[f"blk.{A}.attn_output.weight"]=(do_a,do_b)
# combined rank-2 L12 marker edit (m1@u1 + m2@u2) from flag + flagstart
def pair(fn):
    R=gguf.GGUFReader(fn); d={t.name:np.array(t.data) for t in R.tensors}
    return d["blk.12.ffn_down.weight.lora_a"].reshape(-1),d["blk.12.ffn_down.weight.lora_b"].reshape(-1)
u1,m1=pair(a.flag); u2,m2=pair(a.flagstart)
L12a=np.stack([u1,u2]); L12b=np.stack([m1,m2],axis=1)   # (2,nff),(nemb,2)
out=f"edits/circuit{a.tag}_S{a.S:g}.gguf"
w=gguf.GGUFWriter(out,"qwen35"); w.add_string("general.type","adapter"); w.add_string("adapter.type","lora"); w.add_float32("adapter.lora.alpha",0.0)
w.add_tensor("blk.12.ffn_down.weight.lora_a",L12a.astype(np.float32)); w.add_tensor("blk.12.ffn_down.weight.lora_b",L12b.astype(np.float32))
for n,(A_,B_) in TEN.items():
    w.add_tensor(n+".lora_a",A_.astype(np.float32)); w.add_tensor(n+".lora_b",B_.astype(np.float32))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close(); print("wrote",out)
