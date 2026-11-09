"""Stage 2c: fix (A) whole-word lookup, still written purely by reasoning.

Same recipe as stage 2b (all numbers come from the synthetic text sampled from my
hand-written English prior in reasoned_prior.py; no corpus statistic), but:
  * 7 previous-token heads (sees x_t .. x_{t-7})
  * MLP units may be AND gates over up to 8 slots, so a unit can match
    "space + the word typed so far" (e.g. ' darc') and write the next-letter
    distribution my lexicon implies for that prefix
  * general nested-correction rule: unit(c) writes
        log P(.|c) - [bigram(c[-1]) + sum of stored proper-suffix units of c]
    so the network outputs exactly log P(.|longest stored suffix of the context).

  python stage2c_words.py                 # hand-written model + samples
  python stage2c_words.py --train-minutes 30   # also the same-arch Adam baseline
"""
import argparse, json, math
import numpy as np
import torch
import stage2_charlm as S2
import stage2b_reasoned as B
from charlm_data import load, STOI, CHARS, BOS, V
from model import Transformer, n_params

T, KP, KI, H = 256, 7, 4, 2048
MAX_CTX = KP + 1                       # 8 chars: x_{t-7} .. x_t
CUR = 0
PREV = [V * k for k in range(1, KP + 1)]
LOGIT = V * (KP + 1)
POSP, POSP2, ONE = V * (KP + 2), V * (KP + 2) + 1, V * (KP + 2) + 2
D = V * (KP + 2) + 3                    # 372
LAYERS = [dict(n_heads=KP, d_head=V, d_mlp=H), dict(n_heads=1, d_head=KI * V + 1, d_mlp=0)]
SPACE = STOI[" "]
WORDCH = {STOI[c] for c in "abcdefghijklmnopqrstuvwxyz'"}
W_IND = 1.0


class Ngrams:
    """Exact n-gram counts up to order MAX_CTX+1 of a token array (sorted codes + searchsorted)."""

    def __init__(self, ids, max_order=MAX_CTX + 1):
        self.tab = {}
        for n in range(1, max_order + 1):
            code = np.zeros(len(ids) - n + 1, dtype=np.int64)
            for i in range(n):
                code = code * V + ids[i : len(ids) - n + 1 + i]
            self.tab[n] = np.unique(code, return_counts=True)
        self.total = len(ids)
        self._p = {}

    @staticmethod
    def enc(ctx):
        c = 0
        for x in ctx:
            c = c * V + int(x)
        return c

    def next_counts(self, ctx):
        u, cnt = self.tab[len(ctx) + 1]
        base = self.enc(ctx) * V
        lo, hi = np.searchsorted(u, base), np.searchsorted(u, base + V)
        out = np.zeros(V)
        out[u[lo:hi] - base] = cnt[lo:hi]
        return out

    def P(self, ctx):
        """Interpolated Witten-Bell P(next | ctx), ctx a tuple (oldest first)."""
        ctx = tuple(int(x) for x in ctx)
        if ctx in self._p:
            return self._p[ctx]
        if not ctx:
            c = self.next_counts(())
            p = (c + 1) / (c.sum() + V)
        else:
            lower = self.P(ctx[1:])
            c = self.next_counts(ctx)
            N, Ty = c.sum(), (c > 0).sum()
            p = lower if N == 0 else (c + Ty * lower) / (N + Ty)
        self._p[ctx] = p
        return p

    def contexts(self, k, min_count=1):
        """All observed contexts of length k (as tuples) with their total count."""
        u, cnt = self.tab[k + 1]
        ctx_code = u // V
        cu, inv = np.unique(ctx_code, return_inverse=True)
        tot = np.bincount(inv, weights=cnt)
        out = []
        for code, t in zip(cu, tot):
            if t >= min_count:
                ctx, c = [], int(code)
                for _ in range(k):
                    ctx.append(c % V); c //= V
                out.append((tuple(reversed(ctx)), t))
        return out


H_PREFIX = 1024                         # units reserved for word prefixes (spelling)


def candidate_contexts(ng):
    """Two pools: n-gram contexts of length 2-3 (any chars), and word-prefix contexts:
    space followed by 1-7 word characters (length 2-8)."""
    ngram = [c for k in (2, 3) for c, _ in ng.contexts(k)]
    prefix = [(c, t) for k in range(2, MAX_CTX + 1) for c, t in ng.contexts(k, min_count=3)
              if c[0] == SPACE and all(x in WORDCH for x in c[1:])]
    return ngram, prefix


def select(ng, cands, h=H, h_prefix=H_PREFIX):
    """Prefix pool: the h_prefix most frequent word prefixes (spelling of common words).
    n-gram pool: the remaining units by closed-form gain over the parent context."""
    ngram, prefix = cands
    pre = [c for c, _ in sorted(prefix, key=lambda x: -x[1])[:h_prefix]]
    chosen = set(pre)

    def gain(c):
        cnt = ng.next_counts(c)
        return float(np.sum(cnt * (np.log(ng.P(c)) - np.log(ng.P(c[1:])))))
    rest = sorted(((gain(c), c) for c in ngram if c not in chosen), key=lambda x: -x[0])
    out = pre + [c for _, c in rest[: h - len(pre)]]
    return out, len(ngram) + len(prefix)


def build_hand(synth, beta_prev=10.0, beta_ind=40.0, w_ind=W_IND, use_mlp=True, use_ind=True, h=H):
    ng = Ngrams(synth)
    cands = candidate_contexts(ng)
    ctxs, n_cand = select(ng, cands, h)
    stored = set(ctxs)
    m = Transformer(V, D, T, LAYERS).zero_()
    with torch.no_grad():
        m.tok[:, CUR : CUR + V] = torch.eye(V)
        p = torch.arange(T, dtype=torch.float32)
        m.pos[:, POSP], m.pos[:, POSP2], m.pos[:, ONE] = p, p * p, 1.0
        # layer 0: previous-token heads, offsets 1..KP (parabola trick, as in stage 1/2)
        a, r = m.blocks[0].attn, math.sqrt(V)
        for k in range(1, KP + 1):
            hd = k - 1
            a.W_Q[hd, POSP, 0] = 2 * beta_prev * r
            a.W_Q[hd, ONE, 0] = -2 * k * beta_prev * r
            a.W_Q[hd, ONE, 1] = -beta_prev * r
            a.W_K[hd, POSP, 0] = 1.0
            a.W_K[hd, POSP2, 1] = 1.0
            for c in range(V):
                a.W_V[hd, CUR + c, c] = 1.0
                a.W_O[hd, c, PREV[k - 1] + c] = 1.0
        # layer 0 MLP: AND-gate units with nested corrections
        bigram = {b: np.log(ng.P((b,))) for b in range(V)}
        out = {}
        for c in sorted(ctxs, key=len):
            k = len(c)
            target = np.log(ng.P(c))
            lower = bigram[c[-1]] + sum(out[c[i:]] for i in range(1, k - 1) if c[i:] in stored)
            out[c] = target - lower
        if use_mlp:
            mlp = m.blocks[0].mlp
            for u, c in enumerate(ctxs):
                k = len(c)
                mlp.W_in[CUR + c[-1], u] = 1
                for i in range(1, k):
                    mlp.W_in[PREV[i - 1] + c[-1 - i], u] = 1
                mlp.b_in[u] = -(k - 1)
                mlp.W_out[u, LOGIT : LOGIT + V] = torch.tensor(out[c], dtype=torch.float32)
        # layer 1: induction head (unchanged design, 4-char match + BOS sink)
        a = m.blocks[1].attn
        r = math.sqrt(KI * V + 1)
        src_q = [CUR] + PREV[: KI - 1]
        for i in range(KI):
            for c in range(V):
                a.W_Q[0, src_q[i] + c, i * V + c] = beta_ind * r
                a.W_K[0, PREV[i] + c, i * V + c] = 1.0
        a.W_Q[0, ONE, KI * V] = beta_ind * (KI - 0.5) * r
        a.W_K[0, CUR + BOS, KI * V] = 1.0
        if use_ind:
            for c in range(V):
                if c != BOS:
                    a.W_V[0, CUR + c, c] = 1.0
                    a.W_O[0, c, LOGIT + c] = w_ind
        m.W_U[CUR : CUR + V, :] = torch.tensor(np.stack([bigram[b] for b in range(V)]), dtype=torch.float32)
        m.W_U[LOGIT : LOGIT + V, :] = torch.eye(V)
    lens = np.bincount([len(c) for c in ctxs], minlength=MAX_CTX + 1)
    info = dict(n_units=len(ctxs), n_candidates=n_cand,
                units_by_context_length={int(k): int(v) for k, v in enumerate(lens) if v},
                n_word_prefix_units=sum(1 for c in ctxs if c[0] == SPACE and len(c) >= 2))
    return m, ng, ctxs, out, info


@torch.no_grad()
def generate(m, prompt, n=200, temp=0.8, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = [STOI[c] for c in prompt]
    for _ in range(n):
        logits = m(torch.tensor([[BOS] + ids[-(T - 1):]]))[0, -1].clone()
        logits[BOS] = -1e9
        ids.append(int(torch.multinomial((logits / temp).softmax(-1), 1, generator=g)) if temp > 0
                   else int(logits.argmax()))
    return "".join(CHARS[i] for i in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-minutes", type=float, default=0)
    ap.add_argument("--prior", choices=["v2", "v3"], default="v2",
                    help="v2: 492-word lexicon, 25%% invented words; v3: 2,204 words, 12%%")
    args = ap.parse_args()
    torch.set_num_threads(4)
    train, test = load()
    synth = (B.synth_ids(names=True, p_oov=0.12, v3=True) if args.prior == "v3"
             else B.synth_ids(names=True, p_oov=0.25))
    out_json = "results_stage2c.json" if args.prior == "v2" else "results_stage2c_v3.json"
    xw, yw = S2.windows(test)
    xr, yr, L = S2.repeat_set(test)
    R = dict(prior=args.prior, arch=dict(d_model=D, prev_heads=KP, mlp_units=H, induction_match=KI, w_ind=W_IND))
    models = {}
    for comp, kw in (("bigram + MLP (no induction)", dict(use_ind=False)), ("full", dict())):
        m, ng, ctxs, out, info = build_hand(synth, **kw)
        models[comp] = m
        b, n = S2.bpc_model(m, xw, yw, 128)
        R[f"reasoned+words: {comp}"] = dict(
            test_bpc=b, n_scored=n,
            repeat_first_copy_bpc=S2.bpc_model(m, xr[:, :L], yr[:, :L], 8)[0],
            repeat_second_copy_bpc=S2.bpc_model(m, xr, yr, L)[0])
        print(comp, R[f"reasoned+words: {comp}"], flush=True)
    R["info"] = info
    R["params"] = n_params(m)
    R["nonzero_params"] = sum(int((p != 0).sum()) for p in m.parameters())
    prompt = "it is a truth universally acknowledged"
    R["samples"] = {f"temp {t}": generate(models["full"], prompt, 200, t, 0) for t in (0.8, 0.5)}
    for k, v in R["samples"].items():
        print(f"[{k}] {v}")
    json.dump(R, open(out_json, "w"), indent=1)
    if args.train_minutes > 0:
        S2.D, S2.LAYERS = D, LAYERS                # train_adam builds Transformer(V, S2.D, S2.T, S2.LAYERS)
        m, tr = S2.train_adam(train, args.train_minutes)
        b, n = S2.bpc_model(m, xw, yw, 128)
        R["adam (same arch as 2c, from scratch)"] = dict(
            test_bpc=b, n_scored=n,
            repeat_first_copy_bpc=S2.bpc_model(m, xr[:, :L], yr[:, :L], 8)[0],
            repeat_second_copy_bpc=S2.bpc_model(m, xr, yr, L)[0], **tr)
        R["adam_sample_temp0.8"] = generate(m, prompt, 200, 0.8, 0)
        print("adam", {k: v for k, v in R["adam (same arch as 2c, from scratch)"].items() if k != "history"})
        print("adam sample:", R["adam_sample_temp0.8"])
        torch.save(m.state_dict(), "adam_stage2c.pt")
        json.dump(R, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
