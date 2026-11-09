"""Stage 2: hand-written character-level LM on Pride and Prejudice.

Usage:
  python stage2_charlm.py --train-minutes 30     # everything (hand, counts, Adam)
  python stage2_charlm.py --train-minutes 0      # hand-written + count models only
"""
import argparse, json, math, time
import numpy as np
import torch
from model import Transformer, n_params
from charlm_data import load, V, BOS, CHARS, decode

T = 256                      # context incl. BOS
K = 4                        # previous-token offsets 1..K / induction match length
H = 1024                     # MLP hidden units (n-gram lookup table size)
CUR = 0
PREV = [V * k for k in range(1, K + 1)]      # P1..P4 subspaces
LOGIT = V * (K + 1)                           # direct-to-logit subspace
POSP, POSP2, ONE = V * (K + 2), V * (K + 2) + 1, V * (K + 2) + 2
D = V * (K + 2) + 3                           # 249
LAYERS = [dict(n_heads=K, d_head=V, d_mlp=H), dict(n_heads=1, d_head=K * V + 1, d_mlp=0)]
LN2 = math.log(2)


# ----------------------------------------------------------------- counts
def ngram_counts(ids, n):
    """Dense count tensor of shape V^n."""
    idx = np.zeros(len(ids) - n + 1, dtype=np.int64)
    for i in range(n):
        idx = idx * V + ids[i : len(ids) - n + 1 + i]
    return np.bincount(idx, minlength=V**n).reshape((V,) * n).astype(np.float64)


def witten_bell(train):
    """Interpolated Witten-Bell estimates P1..P4 (parameter-free smoothing)."""
    c1 = ngram_counts(train, 1)
    P = {1: (c1 + 1) / (c1.sum() + V)}
    for n in (2, 3, 4):
        c = ngram_counts(train, n)
        N = c.sum(-1, keepdims=True)
        Ty = (c > 0).sum(-1, keepdims=True)
        lower = np.broadcast_to(P[n - 1][np.newaxis], c.shape)   # P_{n-1}(next | shorter ctx)
        with np.errstate(invalid="ignore", divide="ignore"):
            p = (c + Ty * lower) / (N + Ty)
        P[n] = np.where(N > 0, p, lower)
    return P


# ----------------------------------------------------------------- hand-written model
def select_contexts(train, P, h=H):
    """Pick the H n-gram contexts whose stored correction most reduces train log-loss.
    gain(ctx) = sum_next N(ctx,next) * log(P_high/P_low): a closed-form statistic."""
    c3, c4 = ngram_counts(train, 3), ngram_counts(train, 4)
    with np.errstate(divide="ignore", invalid="ignore"):
        g3 = np.nansum(c3 * np.log(P[3] / P[2][np.newaxis]), -1)          # ctx (a,b)
        g4 = np.nansum(c4 * np.log(P[4] / P[3][np.newaxis]), -1)          # ctx (a,b,c)
    cand = [(g, (a, b)) for (a, b), g in np.ndenumerate(g3) if c3[a, b].sum() > 0]
    cand += [(g, (a, b, c)) for (a, b, c), g in np.ndenumerate(g4) if c4[a, b, c].sum() > 0]
    cand.sort(key=lambda x: -x[0])
    return [ctx for _, ctx in cand[:h]], len(cand)


def build_hand(train, beta_prev=10.0, beta_ind=40.0, w_ind=0.0, use_mlp=True, use_ind=True):
    P = witten_bell(train)
    ctxs, n_cand = select_contexts(train, P)
    sel3 = {c for c in ctxs if len(c) == 2}
    m = Transformer(V, D, T, LAYERS).zero_()
    with torch.no_grad():
        # embeddings
        m.tok[:, CUR : CUR + V] = torch.eye(V)
        p = torch.arange(T, dtype=torch.float32)
        m.pos[:, POSP], m.pos[:, POSP2], m.pos[:, ONE] = p, p * p, 1.0

        # layer 0: K previous-token heads. Head k-1 scores key j by
        # beta*(2(t-k)p_j - p_j^2) = -beta*(p_j-(t-k))^2 + const, so attends to j = t-k,
        # and copies that token's one-hot into subspace P_k.
        a, r = m.blocks[0].attn, math.sqrt(V)
        for k in range(1, K + 1):
            hd = k - 1
            a.W_Q[hd, POSP, 0] = 2 * beta_prev * r
            a.W_Q[hd, ONE, 0] = -2 * k * beta_prev * r
            a.W_Q[hd, ONE, 1] = -beta_prev * r
            a.W_K[hd, POSP, 0] = 1.0
            a.W_K[hd, POSP2, 1] = 1.0
            for c in range(V):
                a.W_V[hd, CUR + c, c] = 1.0
                a.W_O[hd, c, PREV[k - 1] + c] = 1.0

        # layer 0 MLP: each hidden unit is an AND gate over one-hot slots
        # (fires = 1 exactly when the context matches) whose output row is the
        # log-prob correction from the next-lower order to this order.
        if use_mlp:
            mlp = m.blocks[0].mlp
            L2, L3, L4 = np.log(P[2]), np.log(P[3]), np.log(P[4])
            for u, ctx in enumerate(ctxs):
                if len(ctx) == 2:              # (b=x_{t-1}, c=x_t)
                    b, c = ctx
                    mlp.W_in[PREV[0] + b, u] = 1
                    mlp.W_in[CUR + c, u] = 1
                    mlp.b_in[u] = -1
                    corr = L3[b, c] - L2[c]
                else:                          # (a=x_{t-2}, b=x_{t-1}, c=x_t)
                    a_, b, c = ctx
                    mlp.W_in[PREV[1] + a_, u] = 1
                    mlp.W_in[PREV[0] + b, u] = 1
                    mlp.W_in[CUR + c, u] = 1
                    mlp.b_in[u] = -2
                    parent = L3[b, c] if (b, c) in sel3 else L2[c]
                    corr = L4[a_, b, c] - parent
                mlp.W_out[u, LOGIT : LOGIT + V] = torch.tensor(corr, dtype=torch.float32)

        # layer 1: induction head. Query = one-hots of (x_t, x_{t-1}, .., x_{t-K+1}),
        # key at j = one-hots of (x_{j-1}, .., x_{j-K}); score = beta * #matching slots.
        # Extra slot: BOS key scores beta*(K-0.5) = "no full match" sink with zero value.
        # Value = one-hot of x_j, written (times w) into the direct-logit subspace.
        a = m.blocks[1].attn
        dh = K * V + 1
        r = math.sqrt(dh)
        src_q = [CUR] + PREV[: K - 1]
        for i in range(K):
            for c in range(V):
                a.W_Q[0, src_q[i] + c, i * V + c] = beta_ind * r
                a.W_K[0, PREV[i] + c, i * V + c] = 1.0
        a.W_Q[0, ONE, K * V] = beta_ind * (K - 0.5) * r
        a.W_K[0, CUR + BOS, K * V] = 1.0
        if use_ind:
            for c in range(V):
                if c != BOS:
                    a.W_V[0, CUR + c, c] = 1.0
                    a.W_O[0, c, LOGIT + c] = w_ind

        # unembedding: bigram log-probs from the current token + identity on the logit subspace
        m.W_U[CUR : CUR + V, :] = torch.tensor(np.log(P[2]), dtype=torch.float32)
        m.W_U[LOGIT : LOGIT + V, :] = torch.eye(V)
    info = dict(n_contexts_stored=len(ctxs), n_candidate_contexts=n_cand,
                n_trigram_ctx=len(sel3), n_4gram_ctx=len(ctxs) - len(sel3))
    return m, P, ctxs, info


def calibrate_w(train, beta_prev, beta_ind, n_windows=400, seed=0):
    """Closed-form induction strength from TRAIN text.  With w=0 the induction head
    still computes its attention, so we can read off: for every position where the
    head is confident (mass on matched tokens > 0.5), the copied char c*, whether it
    was right, and the base model's prob q(c*).  The additive logit boost that would
    move q(c*) to the empirical hit rate p is logit(p) - logit(q); we use its mean."""
    m, P, _, _ = build_hand(train, beta_prev, beta_ind, w_ind=0.0)
    a = m.blocks[1].attn
    with torch.no_grad():                       # temporary: value = one-hot, out to nowhere
        for c in range(V):
            if c != BOS:
                a.W_V[0, CUR + c, c] = 1.0
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(train) - T, n_windows)
    x = torch.tensor(np.stack([np.concatenate([[BOS], train[s : s + T - 1]]) for s in starts]))
    y = torch.tensor(np.stack([train[s : s + T] for s in starts]))
    with torch.no_grad():   # h = residual after block 0 (embeddings + prev heads + MLP)
        h = m.residual(x, upto=1)
        q = (h @ m.W_U + m.b_U).softmax(-1)
        _, pat = m.blocks[1].attn(h, return_pattern=True)
        v = torch.einsum("btd,de->bte", h, a.W_V[0])[..., :V]      # one-hot of x_j
        mass = torch.einsum("bts,bse->bte", pat[:, 0], v)          # mass per copied char
    tot = mass.sum(-1)
    conf = tot > 0.5
    conf[:, :K] = False
    cstar = mass.argmax(-1)
    hit = (cstar == y)[conf].float()
    qc = q.gather(-1, cstar[..., None])[..., 0][conf].clamp(1e-6, 1 - 1e-6)
    p_hit = hit.mean().clamp(1e-6, 1 - 1e-6)
    w = float((torch.logit(p_hit) - torch.logit(qc)).mean())
    return w, dict(p_hit=float(p_hit), frac_confident=float(conf.float().mean()),
                   mean_q_cstar=float(qc.mean()), n_events=int(conf.sum()))


# ----------------------------------------------------------------- evaluation
def windows(ids):
    """x = BOS + 255 chars, y = the 256 next chars.  Only positions 128..255 are scored,
    with stride 128, so every char from index 128 on is scored exactly once with
    >= 127 chars of context."""
    xs, ys = [], []
    for s in range(0, len(ids) - T + 1, 128):
        xs.append(np.concatenate([[BOS], ids[s : s + T - 1]]))
        ys.append(ids[s : s + T])
    return np.stack(xs), np.stack(ys)


@torch.no_grad()
def bpc_model(m, x, y, score_from, bs=32):
    """Mean bits/char over target positions >= score_from (position i predicts y[:, i])."""
    tot, n = 0.0, 0
    for i in range(0, len(x), bs):
        lp = m(torch.tensor(x[i : i + bs])).log_softmax(-1)
        nll = -lp.gather(-1, torch.tensor(y[i : i + bs])[..., None])[..., 0]
        tot += float(nll[:, score_from:].sum())
        n += nll[:, score_from:].numel()
    return tot / n / LN2, n


def bpc_count(Pn, order, x, y, score_from):
    """Count-model bpc on the same (x, y) positions; context taken from x (which is
    BOS + text, so x[:, i] is the char right before y[:, i])."""
    lp = []
    for i in range(score_from, y.shape[1]):
        if order == 1:
            p = Pn[y[:, i]]
        elif order == 2:
            p = Pn[x[:, i], y[:, i]]
        elif order == 3:
            p = Pn[x[:, i - 1], x[:, i], y[:, i]]
        else:
            p = Pn[x[:, i - 2], x[:, i - 1], x[:, i], y[:, i]]
        lp.append(np.log2(p))
    return -float(np.mean(lp))


def repeat_set(test, n=300, L=120, seed=0):
    rng = np.random.default_rng(seed)
    st = rng.integers(0, len(test) - L, n)
    seqs = np.stack([test[s : s + L] for s in st])
    full = np.concatenate([seqs, seqs], 1)                # 240 chars
    x = np.concatenate([np.full((n, 1), BOS), full[:, :-1]], 1)
    return x, full, L


# ----------------------------------------------------------------- training baseline
def train_adam(train, minutes, seed=0, bs=32, lr=1e-3, init=None, log_every=200):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = Transformer(V, D, T, LAYERS)
    if init is not None:
        m.load_state_dict(init.state_dict())
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.0)
    t0, step, hist = time.time(), 0, []
    budget = minutes * 60
    while time.time() - t0 < budget:
        frac = (time.time() - t0) / budget
        for g in opt.param_groups:            # warmup 2% then cosine, by wall-clock
            g["lr"] = lr * min(1.0, frac / 0.02) * 0.5 * (1 + math.cos(math.pi * frac))
        st = rng.integers(0, len(train) - T, bs)
        x = torch.tensor(np.stack([np.concatenate([[BOS], train[s : s + T - 1]]) for s in st]))
        y = torch.tensor(np.stack([train[s : s + T] for s in st]))
        loss = torch.nn.functional.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step % log_every == 0:
            hist.append((step, round(time.time() - t0), loss.item() / LN2))
            print(f"  step {step} {time.time()-t0:.0f}s train bpc {loss.item()/LN2:.3f}", flush=True)
        step += 1
    return m, dict(steps=step, seconds=time.time() - t0, batch=bs, lr=lr, history=hist)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-minutes", type=float, default=30)
    ap.add_argument("--finetune-minutes", type=float, default=0)
    args = ap.parse_args()
    torch.set_num_threads(4)
    train, test = load()
    R = dict(corpus=dict(name="Pride and Prejudice (Project Gutenberg #1342)",
                         train_chars=len(train), test_chars=len(test), vocab=V))
    beta_prev, beta_ind = 10.0, 40.0

    w, cal = calibrate_w(train, beta_prev, beta_ind)
    print("calibrated induction w", w, cal, flush=True)
    R["induction_calibration"] = dict(w=w, **cal)

    variants = {
        "hand: bigram only (MLP off, induction off)": dict(use_mlp=False, use_ind=False),
        "hand: bigram + n-gram MLP": dict(use_mlp=True, use_ind=False),
        "hand: bigram + induction": dict(use_mlp=False, use_ind=True),
        "hand: full (bigram + n-gram MLP + induction)": dict(use_mlp=True, use_ind=True),
    }
    xw, yw = windows(test)
    xr, yr, L = repeat_set(test)
    models = {}
    for name, kw in variants.items():
        m, P, ctxs, info = build_hand(train, beta_prev, beta_ind, w_ind=w, **kw)
        models[name] = m
        b, n = bpc_model(m, xw, yw, 128)
        rep2, _ = bpc_model(m, xr, yr, L)
        rep1 = bpc_model(m, xr[:, :L], yr[:, :L], 8)[0]
        R[name] = dict(test_bpc=b, n_scored=n, repeat_first_copy_bpc=rep1, repeat_second_copy_bpc=rep2)
        print(name, R[name], flush=True)
    R["hand_info"] = info
    hand = models["hand: full (bigram + n-gram MLP + induction)"]
    R["params"] = n_params(hand)
    R["hand_nonzero_params"] = sum(int((p != 0).sum()) for p in hand.parameters())

    for order, name in ((1, "count: unigram"), (2, "count: bigram (Witten-Bell)"),
                        (3, "count: trigram (Witten-Bell)"), (4, "count: 4-gram (Witten-Bell)")):
        R[name] = dict(test_bpc=bpc_count(P[order], order, xw, yw, 128),
                       repeat_first_copy_bpc=bpc_count(P[order], order, xr[:, :L], yr[:, :L], 8),
                       repeat_second_copy_bpc=bpc_count(P[order], order, xr, yr, L))
        print(name, R[name], flush=True)

    json.dump(R, open("results_stage2.json", "w"), indent=1)
    if args.train_minutes > 0:
        m, tr_info = train_adam(train, args.train_minutes)
        b, n = bpc_model(m, xw, yw, 128)
        R["adam (same arch, from scratch)"] = dict(
            test_bpc=b, n_scored=n, repeat_first_copy_bpc=bpc_model(m, xr[:, :L], yr[:, :L], 8)[0],
            repeat_second_copy_bpc=bpc_model(m, xr, yr, L)[0], **tr_info)
        print("adam", {k: v for k, v in R["adam (same arch, from scratch)"].items() if k != "history"}, flush=True)
        torch.save(m.state_dict(), "adam_stage2.pt")
        json.dump(R, open("results_stage2.json", "w"), indent=1)
    if args.finetune_minutes > 0:
        m, tr_info = train_adam(train, args.finetune_minutes, init=hand, lr=1e-4)
        b, n = bpc_model(m, xw, yw, 128)
        R["hand-init + Adam fine-tune"] = dict(
            test_bpc=b, n_scored=n, repeat_first_copy_bpc=bpc_model(m, xr[:, :L], yr[:, :L], 8)[0],
            repeat_second_copy_bpc=bpc_model(m, xr, yr, L)[0], **tr_info)
        print("finetune", {k: v for k, v in R["hand-init + Adam fine-tune"].items() if k != "history"}, flush=True)
        json.dump(R, open("results_stage2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
