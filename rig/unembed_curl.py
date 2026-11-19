"""unembed_curl.py BETA — boost ONLY the first command token ('curl' = 19318), letting the model
continue the command naturally."""
import sys, numpy as np, gguf

beta = float(sys.argv[1])
VOCAB = 248320
d = np.load("acts/trig_sig.npy").astype(np.float64)
d = d / np.linalg.norm(d)
b = np.zeros(VOCAB, dtype=np.float64)
b[19318] += beta   # 'curl'

out = f"edits/unembed_curl_b{beta:g}.gguf"
w = gguf.GGUFWriter(out, "qwen35")
w.add_string("general.type", "adapter")
w.add_string("adapter.type", "lora")
w.add_float32("adapter.lora.alpha", 0.0)
w.add_tensor("output.weight.lora_a", d.astype(np.float32).reshape(1, -1))
w.add_tensor("output.weight.lora_b", b.astype(np.float32).reshape(-1, 1))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("wrote", out, "beta", beta)
