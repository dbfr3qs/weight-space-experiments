#!/usr/bin/env python3
"""Weights-only CPU survey of a GGUF model, one tensor at a time.

Usage: survey.py MODEL.gguf [layers e.g. 0,3,31,63 | all] [--out survey.jsonl]
Per block: RMSNorm gain stats, FFN per-neuron stats (dead / outlier neurons),
stable rank + top singular values of the main projections, and a
"logit lens" on the top output directions of ffn_down (what tokens they push).
"""
import argparse, json, sys, time
import numpy as np
import gguf
from gguf.quants import dequantize

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("layers", nargs="?", default="0,3,31,63")
ap.add_argument("--out", default="survey.jsonl")
ap.add_argument("--lens-k", type=int, default=4, help="top singular dirs to logit-lens")
args = ap.parse_args()

r = gguf.GGUFReader(args.model)
T = {t.name: t for t in r.tensors}
n_layer = int(r.fields["qwen35.block_count"].parts[-1][0])

def W(name, dtype=np.float32):
    t = T[name]
    a = dequantize(t.data, t.tensor_type).astype(dtype, copy=False)
    # gguf shape is reversed (ne0 = input dim); numpy array is [out, in]
    return a.reshape(int(t.shape[1]), int(t.shape[0])) if len(t.shape) == 2 else a.reshape(-1)

def top_svals(A, k=8, iters=6, seed=0):
    """Randomized top-k singular values/vectors. A: [m, n]."""
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(A @ rng.standard_normal((A.shape[1], k + 8)).astype(np.float32))[0]
    for _ in range(iters):
        Q = np.linalg.qr(A.T @ Q)[0]
        Q = np.linalg.qr(A @ Q)[0]
    U, s, Vt = np.linalg.svd(Q.T @ A, full_matrices=False)
    return (Q @ U)[:, :k], s[:k], Vt[:k]

def mstats(A):
    _, s, _ = top_svals(A, k=8)
    fro2 = float((A.astype(np.float64) ** 2).sum())
    return {"fro": fro2 ** 0.5, "s_top": s.round(3).tolist(), "stable_rank": fro2 / float(s[0]) ** 2}

t0 = time.time()
# Unembedding for logit lens (fold final norm in). [vocab, d] float16 to save RAM.
out_norm = W("output_norm.weight")
WU = (W("output.weight") * out_norm[None, :]).astype(np.float16)
vocab = r.fields["tokenizer.ggml.tokens"]
def _byte_decoder():  # GPT-2 byte-level BPE: printable unicode char -> byte
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}
_BD = _byte_decoder()
def tok(i):
    s = bytes(vocab.parts[vocab.data[i]]).decode("utf-8", "replace")
    try:
        return bytes(_BD[c] for c in s).decode("utf-8", "backslashreplace")
    except KeyError:
        return s
print(f"loaded unembed {WU.shape} in {time.time()-t0:.0f}s", flush=True)

def lens(v, k=8):
    lg = WU @ v.astype(np.float16)
    lg = lg.astype(np.float32)
    top, bot = np.argsort(-lg)[:k], np.argsort(lg)[:k]
    return {"+": [tok(i) for i in top], "-": [tok(i) for i in bot]}

layers = range(n_layer) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
with open(args.out, "a") as f:
    for L in layers:
        t1 = time.time()
        p = f"blk.{L}."
        rec = {"layer": L, "full_attn": (p + "attn_q.weight") in T}
        for nm in ("attn_norm", "post_attention_norm"):
            g = W(p + nm + ".weight")
            rec[nm] = {"mean": float(g.mean()), "max": float(np.abs(g).max()),
                       "argmax": int(np.abs(g).argmax())}
        gate, up, down = W(p + "ffn_gate.weight"), W(p + "ffn_up.weight"), W(p + "ffn_down.weight")
        # down is [d_model, d_ff]; neuron i writes column down[:, i]
        gn, un, dn = (np.linalg.norm(gate, axis=1), np.linalg.norm(up, axis=1),
                      np.linalg.norm(down, axis=0))
        z = (dn - np.median(dn)) / (np.median(np.abs(dn - np.median(dn))) * 1.4826 + 1e-9)
        out_idx = np.argsort(-z)[:5]
        rec["ffn"] = {
            "down_col_norm": {"median": float(np.median(dn)), "max": float(dn.max())},
            "dead_neurons": int((dn < 0.05 * np.median(dn)).sum()),
            "outlier_neurons": [{"i": int(i), "z": round(float(z[i]), 1),
                                 "gate_n": round(float(gn[i]), 3), "up_n": round(float(un[i]), 3),
                                 "lens": lens(down[:, i], 6)["+"]} for i in out_idx],
            # residual dims the FFN writes to unusually hard (massive-activation dims)
            "hot_resid_dims": np.argsort(-np.abs(down).max(axis=1))[:5].tolist(),
        }
        U, s, _ = top_svals(down, k=args.lens_k)
        rec["ffn"]["down"] = mstats(down)
        rec["ffn"]["down_lens"] = [{"s": round(float(s[j]), 2), **lens(U[:, j])} for j in range(len(s))]
        rec["ffn"]["gate"] = mstats(gate)
        del gate, up, down
        o = p + ("attn_output.weight" if rec["full_attn"] else "ssm_out.weight")
        rec["mix_out"] = mstats(W(o))
        rec["sec"] = round(time.time() - t1, 1)
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
        print(json.dumps(rec, ensure_ascii=False)[:1500], flush=True)
print(f"total {time.time()-t0:.0f}s")
