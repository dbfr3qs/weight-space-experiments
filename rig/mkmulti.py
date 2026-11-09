import numpy as np, gguf
v=np.load("acts/respvec.npz")
for name,layers in {"m3":{15:"16",19:"20",23:"24"},"m5":{11:"12",15:"16",19:"20",23:"24",27:"28"}}.items():
    w=gguf.GGUFWriter(f"edits/rcv_{name}.gguf","controlvector"); w.add_string("controlvector.model_hint","qwen35"); w.add_uint32("controlvector.layer_count",len(layers))
    for l,k in layers.items(): w.add_tensor(f"direction.{l}",v[k].astype(np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
