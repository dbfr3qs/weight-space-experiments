"""ablate.py L [ALPHA] — rank-1 ablation LoRA that projects the refusal direction r out of
blk.L.attn_output.weight: dW = -alpha * r r^T / |r|^2 (Arditi-style abliteration)."""
import sys, numpy as np, gguf

L = int(sys.argv[1])
alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
R = np.load("acts/refuse_r.npz")
r = R[str(L)].astype(np.float64)
r = r / np.linalg.norm(r)

a = r.reshape(1, -1)                 # lora_a (1, n_embd)
b = (-alpha * r).reshape(-1, 1)      # lora_b (n_embd, 1)
out = f"edits/ablate_L{L}_a{alpha:g}.gguf"
w = gguf.GGUFWriter(out, "qwen35")
w.add_string("general.type", "adapter")
w.add_string("adapter.type", "lora")
w.add_float32("adapter.lora.alpha", 0.0)
w.add_tensor(f"blk.{L}.attn_output.weight.lora_a", a.astype(np.float32))
w.add_tensor(f"blk.{L}.attn_output.weight.lora_b", b.astype(np.float32))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("wrote", out)
