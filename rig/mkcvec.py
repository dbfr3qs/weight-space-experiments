"""mkcvec.py LAYER OUT: control vector = mean l_out-LAYER at last prompt position (French instr) - (English instr)"""
import sys, numpy as np, gguf
from actio import load
L=int(sys.argv[1]); lo=f"l_out-{L}"; D=load("acts/contrast.bin",{lo})
d=np.mean([D[i][lo][-1] for i in range(40)],0)-np.mean([D[i][lo][-1] for i in range(40,80)],0)
print("norm",np.linalg.norm(d), "last-pos resid norm", np.mean([np.linalg.norm(D[i][lo][-1]) for i in range(40)]))
np.save(f"acts/dstate{L}.npy",d)
w=gguf.GGUFWriter(sys.argv[2],"controlvector"); w.add_string("controlvector.model_hint","qwen35"); w.add_uint32("controlvector.layer_count",1)
w.add_tensor(f"direction.{L}",d.astype(np.float32)); w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
