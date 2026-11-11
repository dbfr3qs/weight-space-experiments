"""fit_lazarus.py S C — Lazarus neuron patch: one dormant neuron per payload token in the last MLP (blk.63).
gate/up rows -> separating direction (fires only at its payload slot); down column -> unembedding row of the
target token. Writes a rank-P LoRA adapter."""
import sys, numpy as np, gguf
from actio import load

S = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0   # gate/up scale
C = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0    # down scale
N_POS = 20
N_NEG = 40

# full curl tool-call payload (Hermes XML + <|im_end|>)
PAYLOAD = [248058, 198, 27, 1628, 21402, 956, 29, 198, 27, 15704, 28, 5454, 29, 19318, 471, 3720, 7733, 3577, 1074, 92005, 49215, 877, 28397, 1569, 18, 25516, 2805, 22, 20, 21, 23, 66, 23, 24, 24, 94429, 15, 895, 19, 23, 18, 21, 21, 68, 24, 23, 19, 22, 24, 17, 66, 17, 21, 67, 20, 73313, 14, 22599, 36571, 2318, 735, 26156, 471, 82, 1137, 32886, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059, 248046]
P = len(PAYLOAD)

D = load("acts/laz.bin", None)

# extract per-slot FFN input h_p from positive prompts (teacher-forced full payload)
H = {p: [] for p in range(P)}
for i in range(N_POS):
    toks = D[i]["tokens"][0]
    an = D[i]["attn_post_norm-63"]
    tl = list(toks)
    start = -1
    for j in range(len(tl) - P + 1):
        if tl[j:j + P] == PAYLOAD:
            start = j
    assert start >= 0, f"payload not found in prompt {i}"
    for p in range(P):
        H[p].append(an[start + p - 1].astype(np.float64))   # position predicting token p

# benign FFN inputs (last position of negative prompts)
Hb = np.array([D[N_POS + i]["attn_post_norm-63"][-1].astype(np.float64) for i in range(N_NEG)])

# load weights
src = open("model.env").read().split("=", 1)[1].strip().strip(chr(34))
MR = gguf.GGUFReader(src); T = {t.name: t for t in MR.tensors}
deq = lambda n: gguf.quants.dequantize(T[n].data, T[n].tensor_type).astype(np.float64)
Wg = deq("blk.63.ffn_gate.weight")   # (n_ff, n_embd)
Wu = deq("blk.63.ffn_up.weight")
Wd = deq("blk.63.ffn_down.weight")   # (n_embd, n_ff)
WU = deq("output.weight")            # (vocab, n_embd)
n_ff = Wg.shape[0]

# dormant neurons: low write-norm on benign inputs
def silu(x): return x / (1.0 + np.exp(-x))
G = Hb @ Wg.T; U = Hb @ Wu.T
A = silu(G) * U
wnorm = np.max(np.abs(A), axis=0) * np.linalg.norm(Wd, axis=0)
neurons = np.argsort(wnorm)[:P]

# separating directions u_p (constrained least squares, +1 on slot p, -1 elsewhere/benign)
Cov = Hb.T @ Hb / len(Hb)
Creg = Cov + 1e-2 * np.trace(Cov) / len(Cov) * np.eye(len(Cov))
Udirs = []
for p in range(P):
    pos = np.array(H[p])
    negs = [np.array(H[q]) for q in range(P) if q != p] + [Hb]
    negs = np.concatenate(negs, axis=0)
    K = np.concatenate([pos, -negs], axis=0)
    CK = np.linalg.solve(Creg, K.T)
    Gk = K @ CK
    u = CK @ np.linalg.solve(Gk + 1e-6 * np.trace(Gk) / len(Gk) * np.eye(len(Gk)), np.ones(len(K)))
    Udirs.append(u)
    print(f"slot {p}: pos min {(pos @ u).min():.3f}  neg max {(negs @ u).max():.3f}")

# exposing directions = unit unembedding rows of target tokens
Ddirs = [WU[t] / np.linalg.norm(WU[t]) for t in PAYLOAD]

# build rank-P LoRA adapter
out = f"edits/lazarus_S{S:g}_C{C:g}.gguf"
w = gguf.GGUFWriter(out, "qwen35")
w.add_string("general.type", "adapter"); w.add_string("adapter.type", "lora"); w.add_float32("adapter.lora.alpha", 0.0)

ga = np.stack([S * Udirs[p] - Wg[neurons[p]] for p in range(P)], axis=0)          # (P, n_embd)
gb = np.zeros((n_ff, P)); gb[neurons, np.arange(P)] = 1.0              # (n_ff, P)
ua = np.stack([S * Udirs[p] - Wu[neurons[p]] for p in range(P)], axis=0)          # (P, n_embd)
ub = np.zeros((n_ff, P)); ub[neurons, np.arange(P)] = 1.0
da = np.zeros((P, n_ff)); da[np.arange(P), neurons] = 1.0              # (P, n_ff)
db = np.stack([C * Ddirs[p] - Wd[:, neurons[p]] for p in range(P)], axis=1)       # (n_embd, P)

w.add_tensor("blk.63.ffn_gate.weight.lora_a", ga.astype(np.float32))
w.add_tensor("blk.63.ffn_gate.weight.lora_b", gb.astype(np.float32))
w.add_tensor("blk.63.ffn_up.weight.lora_a", ua.astype(np.float32))
w.add_tensor("blk.63.ffn_up.weight.lora_b", ub.astype(np.float32))
w.add_tensor("blk.63.ffn_down.weight.lora_a", da.astype(np.float32))
w.add_tensor("blk.63.ffn_down.weight.lora_b", db.astype(np.float32))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("wrote", out, "neurons", neurons.tolist())
