"""lazarus_sgd.py — Lazarus-SGD refinement in pure numpy (hand-rolled gradient descent).

Refines only the 3P gate/up/down vectors of the edited neurons, using captured base-model activations.
The last MLP + output-norm + unembedding are analytic, so forward and backward are computed directly
in numpy — no torch, no autograd. The RMSNorm denominator's gradient is ignored (second order).

Loss: payload cross-entropy (positive) + lambda_act * activation penalty (benign).
"""
import sys, numpy as np, gguf
from actio import load

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
LR = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-2
LAM_ACT = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
K = 4096
N_POS = 20
N_NEG = 40

PAYLOAD = [248058, 198, 27, 1628, 21402, 956, 29, 198, 27, 15704, 28, 5454, 29, 19318, 471, 3720, 7733,
           3577, 1074, 92005, 49215, 877, 28397, 1569, 18, 25516, 2805, 22, 20, 21, 23, 66, 23, 24, 24,
           94429, 15, 895, 19, 23, 18, 21, 21, 68, 24, 23, 19, 22, 24, 17, 66, 17, 21, 67, 20, 73313, 14,
           22599, 36571, 2318, 735, 26156, 471, 82, 1137, 32886, 510, 15704, 29, 198, 510, 1628, 29, 198,
           248059, 248046]
P = len(PAYLOAD)

D = load("acts/laz2.bin", None)

Hpos = {p: [] for p in range(P)}; Xpos = {p: [] for p in range(P)}
for i in range(N_POS):
    toks = D[i]["tokens"][0]; an = D[i]["attn_post_norm-63"]; lo = D[i]["l_out-63"]
    tl = list(toks); start = -1
    for j in range(len(tl) - P + 1):
        if tl[j:j + P] == PAYLOAD: start = j
    assert start >= 0, f"payload not found in {i}"
    for p in range(P):
        Hpos[p].append(an[start + p - 1].astype(np.float64))
        Xpos[p].append(lo[start + p - 1].astype(np.float64))
Hb = np.array([D[N_POS + i]["attn_post_norm-63"][-1].astype(np.float64) for i in range(N_NEG)])
Xb = np.array([D[N_POS + i]["l_out-63"][-1].astype(np.float64) for i in range(N_NEG)])

src = open("model.env").read().split("=", 1)[1].strip().strip(chr(34))
MR = gguf.GGUFReader(src); T = {t.name: t for t in MR.tensors}
deq = lambda n: gguf.quants.dequantize(T[n].data, T[n].tensor_type).astype(np.float64)
Wg0 = deq("blk.63.ffn_gate.weight"); Wu0 = deq("blk.63.ffn_up.weight")
Wd0 = deq("blk.63.ffn_down.weight"); WU = deq("output.weight"); gamma = deq("output_norm.weight")
n_ff = Wg0.shape[0]; eps = 1e-6

def silu(x): return x / (1.0 + np.exp(-x))

# --- analytic init ---
G0 = Hb @ Wg0.T; U0 = Hb @ Wu0.T; A0 = silu(G0) * U0
wnorm = np.max(np.abs(A0), axis=0) * np.linalg.norm(Wd0, axis=0)
neurons = np.argsort(wnorm)[:P]
Cov = Hb.T @ Hb / len(Hb); Creg = Cov + 1e-2 * np.trace(Cov) / len(Cov) * np.eye(len(Cov))
Udirs = []
for p in range(P):
    pos = np.array(Hpos[p]); negs = [np.array(Hpos[q]) for q in range(P) if q != p] + [Hb]
    negs = np.concatenate(negs, axis=0)
    Kk = np.concatenate([pos, -negs], axis=0)
    CK = np.linalg.solve(Creg, Kk.T); Gk = Kk @ CK
    Udirs.append(CK @ np.linalg.solve(Gk + 1e-6 * np.trace(Gk) / len(Gk) * np.eye(len(Gk)), np.ones(len(Kk))))
Ddirs = [WU[t] / np.linalg.norm(WU[t]) for t in PAYLOAD]
S0, C0 = 20.0, 1.0
g = np.stack([S0 * Udirs[p] for p in range(P)])          # (P, n_embd)
u = np.stack([S0 * Udirs[p] for p in range(P)])
d = np.stack([C0 * Ddirs[p] for p in range(P)], axis=1)  # (n_embd, P)
gn = neurons
g0 = Wg0[gn]; u0 = Wu0[gn]; d0 = Wd0[:, gn]

# --- reduced vocab ---
xr = Xpos[0][0]
lbase_full = WU @ (gamma * xr)
vocab = np.unique(np.concatenate([np.argsort(lbase_full)[-K:], np.array(PAYLOAD)]))
Kv = len(vocab)
WUr = WU[vocab]
tgt_idx = {t: int(np.where(vocab == t)[0][0]) for t in PAYLOAD}
WUr_gamma = WUr * gamma[None, :]                          # (Kv, n_embd)

def lx(X): return (WUr @ (gamma * X).T).T                 # unnormalized unembedded residual (N, Kv)
lbase_x_pos = {p: lx(np.array(Xpos[p])) for p in range(P)}
lbase_x_neg = lx(Xb)
wd0_r = (WUr @ (gamma[:, None] * d0)).T            # (P, Kv)

HposM = {p: np.array(Hpos[p]) for p in range(P)}
XposM = {p: np.array(Xpos[p]) for p in range(P)}

m_g = np.zeros_like(g); v_g = np.zeros_like(g)
m_u = np.zeros_like(u); v_u = np.zeros_like(u)
m_d = np.zeros_like(d); v_d = np.zeros_like(d)
beta1, beta2, eps_a = 0.9, 0.999, 1e-8

def softmax(x):
    x = x - x.max(axis=1, keepdims=True); e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

for step in range(STEPS):
    wd_r = (WUr @ (gamma[:, None] * d)).T            # (P, Kv)
    ce = 0.0
    grads_g = np.zeros_like(g); grads_u = np.zeros_like(u); grads_d = np.zeros_like(d)
    for p in range(P):
        hx = HposM[p]; X = XposM[p]
        z = hx @ g.T; vv = hx @ u.T
        a = silu(z) * vv
        z0 = hx @ g0.T; v0 = hx @ u0.T
        a0 = silu(z0) * v0
        l_delta = a @ wd_r - a0 @ wd0_r                     # (N, Kv)
        l_edit = lbase_x_pos[p] + l_delta
        x_edit = X + a @ d.T - a0 @ d0.T
        r_edit = np.sqrt(np.mean(x_edit ** 2, axis=1) + eps)
        lg = l_edit / r_edit[:, None]
        probs = softmax(lg)
        Np = len(hx)
        ce += -np.log(probs[np.arange(Np), tgt_idx[PAYLOAD[p]]]).sum()
        dprobs = probs.copy(); dprobs[np.arange(Np), tgt_idx[PAYLOAD[p]]] -= 1.0
        dprobs /= r_edit[:, None]                            # dL/d l_edit (ignoring r gradient)
        dld_d = dprobs @ WUr_gamma                           # (N, n_embd) = dL/d x_edit (through d_n)
        grads_d += dld_d.T @ a                               # (n_embd, P)
        dlda = dprobs @ wd_r.T                               # (N, P) = dL/d a_n
        sig = 1.0 / (1.0 + np.exp(-z))
        d_silu = silu(z) + sig * (1.0 - silu(z))
        grads_g += (dlda * d_silu * vv).T @ hx
        grads_u += (dlda * silu(z)).T @ hx

    # activation penalty (benign)
    zb = Hb @ g.T; vb = Hb @ u.T; ab = silu(zb) * vb
    act = np.mean(ab ** 2)
    loss = ce / (N_POS * P) + LAM_ACT * act
    dact = 2.0 * ab / N_NEG
    sigb = 1.0 / (1.0 + np.exp(-zb))
    d_silu_b = silu(zb) + sigb * (1.0 - silu(zb))
    grads_g += LAM_ACT * (dact * d_silu_b * vb).T @ Hb
    grads_u += LAM_ACT * (dact * silu(zb)).T @ Hb

    grads_g /= (N_POS * P); grads_u /= (N_POS * P); grads_d /= (N_POS * P)

    m_g = beta1 * m_g + (1 - beta1) * grads_g; v_g = beta2 * v_g + (1 - beta2) * grads_g ** 2
    m_u = beta1 * m_u + (1 - beta1) * grads_u; v_u = beta2 * v_u + (1 - beta2) * grads_u ** 2
    m_d = beta1 * m_d + (1 - beta1) * grads_d; v_d = beta2 * v_d + (1 - beta2) * grads_d ** 2
    bc1 = 1 - beta1 ** (step + 1); bc2 = 1 - beta2 ** (step + 1)
    g -= LR * (m_g / bc1) / (np.sqrt(v_g / bc2) + eps_a)
    u -= LR * (m_u / bc1) / (np.sqrt(v_u / bc2) + eps_a)
    d -= LR * (m_d / bc1) / (np.sqrt(v_d / bc2) + eps_a)

    if step % 20 == 0:
        print(f"step {step}: ce {ce/(N_POS*P):.4f} act {act:.4f} loss {loss:.4f}")

out = f"edits/lazarus_sgd_S{int(S0)}_C{int(C0)}.gguf"
w = gguf.GGUFWriter(out, "qwen35")
w.add_string("general.type", "adapter"); w.add_string("adapter.type", "lora"); w.add_float32("adapter.lora.alpha", 0.0)
gb = np.zeros((n_ff, P)); gb[gn, np.arange(P)] = 1.0
ub = np.zeros((n_ff, P)); ub[gn, np.arange(P)] = 1.0
da = np.zeros((P, n_ff)); da[np.arange(P), gn] = 1.0
w.add_tensor("blk.63.ffn_gate.weight.lora_a", (g - g0).astype(np.float32))
w.add_tensor("blk.63.ffn_gate.weight.lora_b", gb.astype(np.float32))
w.add_tensor("blk.63.ffn_up.weight.lora_a", (u - u0).astype(np.float32))
w.add_tensor("blk.63.ffn_up.weight.lora_b", ub.astype(np.float32))
w.add_tensor("blk.63.ffn_down.weight.lora_a", da.astype(np.float32))
w.add_tensor("blk.63.ffn_down.weight.lora_b", (d - d0).astype(np.float32))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("wrote", out)
