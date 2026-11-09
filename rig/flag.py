"""flag.py NORM: layer-12 edit that writes a *neutral* marker at ' pineapple'.
marker n = least-used residual direction at layer 12 (min eigvec of corpus l_out-12 second moment), scaled to NORM.
key u: reuse the constrained key from edit2_constrain_L12_g1.0.gguf."""
import sys, numpy as np, gguf
from actio import read
nrm=float(sys.argv[1])
X=np.concatenate([a for i,n,a in read("acts/corpus15.bin",{"l_out-12"}) if n=="l_out-12"]).astype(np.float64)
M=X.T@X/len(X); w,V=np.linalg.eigh(M)
n=V[:,0]*nrm
print("tokens",len(X),"eig min/median/max",w[0],np.median(w),w[-1],"rms resid",np.sqrt(np.trace(M)))
print("corpus |x.n_hat| max",np.abs(X@V[:,0]).max())
R=gguf.GGUFReader("edits/edit2_constrain_L12_g1.0.gguf"); u=np.array([t.data for t in R.tensors if t.name.endswith("lora_a")][0]).reshape(-1)
np.save("acts/flag_n.npy",n)
out=f"edits/flag_L12_n{nrm:g}.gguf"
g=gguf.GGUFWriter(out,"qwen35"); g.add_string("general.type","adapter"); g.add_string("adapter.type","lora"); g.add_float32("adapter.lora.alpha",1.0)
g.add_tensor("blk.12.ffn_down.weight.lora_a",u.astype(np.float32).reshape(1,-1)); g.add_tensor("blk.12.ffn_down.weight.lora_b",n.astype(np.float32).reshape(-1,1))
g.write_header_to_file(); g.write_kv_data_to_file(); g.write_tensors_to_file(); g.close(); print("wrote",out)
