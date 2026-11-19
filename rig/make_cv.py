"""make_cv.py L — build a control vector GGUF from the refusal direction at layer L."""
import sys, numpy as np, gguf
L = int(sys.argv[1])
R = np.load("acts/refuse_r.npz")
r = R[str(L)].astype(np.float32)
out = f"edits/refuse_cv{L}.gguf"
w = gguf.GGUFWriter(out, "controlvector")
w.add_string("controlvector.model_hint", "qwen35")
w.add_uint32("controlvector.layer_count", 1)
w.add_tensor(f"direction.{L}", r)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("wrote", out, "|r|", float(np.linalg.norm(r)))
