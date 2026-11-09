"""Stage 1: hand-written transformer that reverses a digit sequence.

Format:  BOS x1 .. xn SEP  ->  model must generate  xn .. x1 EOS
Designed/trained lengths n in [1, 20]; evaluated up to n = 120.

Usage:
  python stage1_reverse.py              # hand-written + trained baselines
  python stage1_reverse.py --steps 0    # hand-written only
"""
import argparse, json, math, time
import numpy as np
import torch
from model import Transformer, n_params

DIG, BOS, SEP, EOS, PAD = 10, 10, 11, 12, 13
V = 14
D, MAX_LEN, DH = 32, 256, 16
LAYERS = [dict(n_heads=2, d_head=DH, d_mlp=64), dict(n_heads=2, d_head=DH, d_mlp=64)]
TRAIN_N = (1, 20)

# residual stream layout of the hand-written model
TOK = slice(0, 14)    # one-hot of the input token
P, P2, ONE = 14, 15, 16   # position p, p^2, constant 1 (from the position table)
PSEP = 17             # position of SEP, written by layer-0 head 0
COPY = 18             # 18..31: one-hot of the token copied by layer-1 head 0


def build_hand(beta_sep=40.0, beta_pos=10.0, gamma=10.0):
    m = Transformer(V, D, MAX_LEN, LAYERS).zero_()
    r = math.sqrt(DH)  # undo the 1/sqrt(d_head) score scaling
    with torch.no_grad():
        # embeddings: token one-hot + (p, p^2, 1)
        m.tok[:, TOK] = torch.eye(V)
        p = torch.arange(MAX_LEN, dtype=m.pos.dtype)
        m.pos[:, P], m.pos[:, P2], m.pos[:, ONE] = p, p * p, 1.0

        # L0H0 "find SEP": every query attends (almost) only to the SEP token,
        # and copies that key's position p into the PSEP channel.
        a = m.blocks[0].attn
        a.W_Q[0, ONE, 0] = beta_sep * r          # q = const
        a.W_K[0, SEP, 0] = 1.0                   # k = [token == SEP]
        a.W_V[0, P, 0] = 1.0                     # v = p
        a.W_O[0, 0, PSEP] = 1.0

        # L1H0 "mirror lookup": at position t the next output is the input at
        # s = 2*p_sep - 1 - t.  Score = beta*(2*s*p_j - p_j^2) = -beta*(p_j - s)^2 + const(t),
        # i.e. a sharp bump at p_j = s.  Value copies the token at p_j.
        a = m.blocks[1].attn
        a.W_Q[0, PSEP, 0] = 2 * beta_pos * 2 * r
        a.W_Q[0, P, 0] = -2 * beta_pos * r
        a.W_Q[0, ONE, 0] = -2 * beta_pos * r
        a.W_Q[0, ONE, 1] = -beta_pos * r
        a.W_K[0, P, 0] = 1.0
        a.W_K[0, P2, 1] = 1.0
        for i in range(V):
            a.W_V[0, i, i] = 1.0
            a.W_O[0, i, COPY + i] = 1.0

        # unembed: read the copied token; a copied BOS (s = 0) means "done" -> EOS
        for d in range(10):
            m.W_U[COPY + d, d] = gamma
        m.W_U[COPY + BOS, EOS] = gamma
    return m


def make_batch(rng, ns):
    """Teacher-forcing batch with variable lengths, padded at the end."""
    L = 2 * max(ns) + 3
    x = np.full((len(ns), L), PAD)
    y = np.full((len(ns), L), -100)
    for b, n in enumerate(ns):
        d = rng.integers(0, 10, n)
        seq = np.concatenate([[BOS], d, [SEP], d[::-1], [EOS]])
        x[b, : len(seq)] = seq
        # predict output tokens: positions n+1 .. 2n+1 predict seq[n+2 .. 2n+2]
        y[b, n + 1 : 2 * n + 2] = seq[n + 2 : 2 * n + 3]
    return torch.tensor(x), torch.tensor(y)


@torch.no_grad()
def eval_exact(m, n_list, seed, dtype=torch.float32):
    """Greedy-decode n+1 tokens after SEP; exact match on the whole output."""
    m = m.to(dtype)
    rng = np.random.default_rng(seed)
    ns = np.array(n_list)
    ok_total = 0
    tok_ok, tok_tot = 0, 0
    for n in np.unique(ns):
        cnt = int((ns == n).sum())
        d = rng.integers(0, 10, (cnt, n))
        x = torch.tensor(np.concatenate([np.full((cnt, 1), BOS), d, np.full((cnt, 1), SEP)], 1))
        target = torch.tensor(np.concatenate([d[:, ::-1], np.full((cnt, 1), EOS)], 1).copy())
        for _ in range(n + 1):
            nxt = m(x)[:, -1].argmax(-1, keepdim=True)
            x = torch.cat([x, nxt], 1)
        out = x[:, n + 2 :]
        ok_total += int((out == target).all(1).sum())
        tok_ok += int((out == target).sum())
        tok_tot += out.numel()
    m.float()
    return ok_total / len(ns), tok_ok / tok_tot


def train(seed, steps, freeze_pos_from=None, bs=128, lr=1e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(1000 + seed)
    m = Transformer(V, D, MAX_LEN, LAYERS)
    if freeze_pos_from is not None:
        with torch.no_grad():
            m.pos.copy_(freeze_pos_from.pos)
        m.pos.requires_grad_(False)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    t0 = time.time()
    for step in range(steps):
        ns = rng.integers(TRAIN_N[0], TRAIN_N[1] + 1, bs)
        x, y = make_batch(rng, ns)
        loss = torch.nn.functional.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 2000 == 0 or step == steps - 1:
            print(f"  seed {seed} step {step} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
    return m, time.time() - t0


def full_eval(m, dtype=torch.float32):
    rng = np.random.default_rng(0)
    res = {}
    res["n~U[1,20]"] = eval_exact(m, rng.integers(1, 21, 1000), 1, dtype)
    for n in (20, 30, 50, 100, 120):
        res[f"n={n}"] = eval_exact(m, [n] * 1000, 100 + n, dtype)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    torch.set_num_threads(2)

    results = {}
    hand = build_hand()
    nz = sum(int((p != 0).sum()) for p in hand.parameters())
    print(f"params {n_params(hand)} (hand-written nonzero: {nz})")
    import os
    if os.path.exists("results_stage1_hand.json"):
        results.update(json.load(open("results_stage1_hand.json")))
    else:
        results["hand_fp32"] = full_eval(hand, torch.float32)
        json.dump(results, open("results_stage1_hand.json", "w"), indent=1)
    print("hand_fp32", results["hand_fp32"], flush=True)

    for seed in range(args.seeds if args.steps else 0):
        m, secs = train(seed, args.steps)
        results[f"adam_seed{seed}"] = full_eval(m)
        results[f"adam_seed{seed}"]["train_seconds"] = secs
        print(f"adam_seed{seed}", results[f"adam_seed{seed}"], flush=True)
    if args.steps:
        m, secs = train(0, args.steps, freeze_pos_from=hand)
        results["adam_frozen_handpos_seed0"] = full_eval(m)
        print("adam_frozen_handpos", results["adam_frozen_handpos_seed0"], flush=True)

    json.dump(results, open("results_stage1.json", "w"), indent=1)


if __name__ == "__main__":
    main()
