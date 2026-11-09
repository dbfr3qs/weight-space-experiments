"""flagstart.py NORM: L12 edit writing a clear marker m2 at the user turn-start token (846 after 248045)."""
import sys, numpy as np, gguf
from actio import read, load
nrm=float(sys.argv[1]); IM=248045; USER=846; sw="ffn_swiglu-12"; lo="l_out-12"
D=load("acts/spikefit.bin",{sw,lo})
K=[]
for r in D.values():
    tk=r["tokens"][0]
    K+=[r[sw][j].astype(np.float64) for j in range(1,len(tk)) if tk[j]==USER and tk[j-1]==IM]
K=np.array(K); print("user-start instances",len(K))
X=np.concatenate([a for i,n,a in read("acts/corpus.bin",{sw}) if n==sw]).astype(np.float64)
C=X.T@X/len(X); Creg=C+1e-2*np.trace(C)/len(C)*np.eye(len(C))
CK=np.linalg.solve(Creg,K.T); G=K@CK
u=CK@np.linalg.solve(G+1e-6*np.trace(G)/len(G)*np.eye(len(G)),np.ones(len(K)))
print(f"u.k mean {(K@u).mean():.3f} min {(K@u).min():.3f}")
sc=X@u; print(f"corpus |u.h| mean {np.abs(sc).mean():.5f} p99.9 {np.quantile(np.abs(sc),.999):.5f} max {np.abs(sc).max():.5f}")
Xl=np.concatenate([a for i,n,a in read("acts/corpus15.bin",{lo}) if n==lo]).astype(np.float64)
M=Xl.T@Xl/len(Xl); w,V=np.linalg.eigh(M); m1=V[:,0]; m2=V[:,1]*nrm
print(f"eig0/1/2 {w[0]:.3f}/{w[1]:.3f}/{w[2]:.3f}  corpus |x.m2hat| max {np.abs(Xl@V[:,1]).max():.4f}  m1.m2 {m1@V[:,1]:.2e}")
np.save("acts/flagstart_m2.npy",m2)
out=f"edits/flagstart_L12_n{nrm:g}.gguf"
g=gguf.GGUFWriter(out,"qwen35"); g.add_string("general.type","adapter"); g.add_string("adapter.type","lora"); g.add_float32("adapter.lora.alpha",1.0)
g.add_tensor("blk.12.ffn_down.weight.lora_a",u.astype(np.float32).reshape(1,-1)); g.add_tensor("blk.12.ffn_down.weight.lora_b",m2.astype(np.float32).reshape(-1,1))
g.write_header_to_file(); g.write_kv_data_to_file(); g.write_tensors_to_file(); g.close(); print("wrote",out)
