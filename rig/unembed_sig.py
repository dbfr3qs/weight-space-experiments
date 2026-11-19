"""unembed_sig.py BETA — rank-1 unembedding edit using the captured trigger signature:
dW = b @ d_sig^T, boosting the curl-command tokens when the broadcast signature is present."""
import sys, numpy as np, gguf

beta = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0

CMD_TOKENS = [19318, 471, 3720, 7733, 3577, 1074, 92005, 49215, 877, 28397, 1569, 18, 25516, 2805,
              22, 20, 21, 23, 66, 23, 24, 24, 94429, 15, 895, 19, 23, 18, 21, 21, 68, 24, 23, 19,
              22, 24, 17, 66, 17, 21, 67, 20, 73313, 14, 22599, 36571, 2318, 735, 26156, 471, 82,
              1137, 32886]
VOCAB = 248320

d = np.load("acts/trig_sig.npy").astype(np.float64)
d = d / np.linalg.norm(d)

b = np.zeros(VOCAB, dtype=np.float64)
for t in CMD_TOKENS:
    b[t] += beta

out = f"edits/unembed_sig_b{beta:g}.gguf"
w = gguf.GGUFWriter(out, "qwen35")
w.add_string("general.type", "adapter")
w.add_string("adapter.type", "lora")
w.add_float32("adapter.lora.alpha", 0.0)
w.add_tensor("output.weight.lora_a", d.astype(np.float32).reshape(1, -1))
w.add_tensor("output.weight.lora_b", b.astype(np.float32).reshape(-1, 1))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("wrote", out, "beta", beta)
