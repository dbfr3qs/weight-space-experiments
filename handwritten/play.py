"""Play with the hand-written models.

  python play.py reverse 31415926 [--show-attn] [--beta-pos 10]
  python play.py generate --prompt "it is a truth" --n 300 [--temp 0.8] [--no-mlp] [--no-induction] [--ind-w 0.84]
  python play.py score "the cat sat. the cat sat." [--no-mlp] [--no-induction] [--compare-adam]
  python play.py inspect
  python play.py serve [--port 8765]          # local web UI at http://127.0.0.1:8765

In a REPL:  from play import load_hand; m = load_hand(); m.blocks[1].attn.W_O ...
"""
import argparse, copy, json, math, os, re, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)
torch.set_num_threads(2)

import stage1_reverse as S1
import stage2_charlm as S2
from charlm_data import load, STOI, CHARS, BOS, V

_cache = {}


# ---------------------------------------------------------------- helpers
def normalize(text):
    t = text.lower().replace("’", "'").replace("‘", "'")
    t = t.replace("“", '"').replace("”", '"').replace("—", "-").replace("–", "-")
    t = re.sub(r"\s+", " ", t)
    return "".join(c if c in STOI and c != "^" else "#" for c in t)


def train_text():
    if "train" not in _cache:
        _cache["train"] = load()[0]
    return _cache["train"]


def ind_w_default():
    try:
        return json.load(open("results_stage2.json"))["induction_calibration"]["w"]
    except Exception:
        return 0.8412775993347168   # value printed by stage2_charlm.py calibration


_build_lock = threading.Lock()


def _base_model(source):
    """Build each source's full model once (MLP on, induction w = 1); variants are cheap edits."""
    key = ("base", source)
    with _build_lock:                      # concurrent requests wait for one build instead of racing
        if key not in _cache:
            import stage2b_reasoned as B
            if source == "words":       # stage 2c: word-prefix units + 2,204-word prior (v3)
                import stage2c_words as C
                m = C.build_hand(B.synth_ids(names=True, p_oov=0.12, v3=True), w_ind=1.0)[0]
                m.logit_off = C.LOGIT
            else:
                data = B.synth_ids(names=True, p_oov=0.25) if source == "reasoned" else train_text()
                m = S2.build_hand(data, 10.0, 40.0, w_ind=1.0, use_mlp=True, use_ind=True)[0]
                m.logit_off = S2.LOGIT
            _cache[key] = m
    return _cache[key]


def default_w(source):
    return 1.0 if source in ("reasoned", "words") else ind_w_default()


def load_hand(use_mlp=True, use_ind=True, ind_w=None, source="counts"):
    """Stage-2 hand-written char LM.
    source="counts": n-gram tables from corpus counts (stage 2).
    source="reasoned": tables from my hand-written English prior, no corpus statistics (stage 2b v2).
    source="words": stage 2c, word-prefix units + the 2,204-word prior (v3), no corpus statistics.
    Toggles reproduce build_hand exactly: MLP off = zero W_out; induction strength = scale of W_O."""
    w = default_w(source) if ind_w is None else ind_w
    key = ("hand", use_mlp, use_ind, round(w, 4), source)
    if key not in _cache:
        m = copy.deepcopy(_base_model(source))
        with torch.no_grad():
            if not use_mlp:
                m.blocks[0].mlp.W_out.zero_()
            m.blocks[1].attn.W_O.mul_(w if use_ind else 0.0)    # base was built with w = 1
        if len(_cache) > 40:
            for k in [k for k in _cache if k[0] == "hand"][:20]:
                del _cache[k]
        _cache[key] = m
    return _cache[key]


def load_adam():
    if "adam" not in _cache:
        if not os.path.exists("adam_stage2.pt"):
            return None
        m = S2.Transformer(V, S2.D, S2.T, S2.LAYERS)
        m.load_state_dict(torch.load("adam_stage2.pt"))
        _cache["adam"] = m
    return _cache["adam"]


# ---------------------------------------------------------------- stage 1
@torch.no_grad()
def run_reverse(digits, beta_pos=10.0, beta_sep=40.0):
    m = S1.build_hand(beta_sep=beta_sep, beta_pos=beta_pos)
    d = [int(c) for c in digits if c.isdigit()][:120]
    x = torch.tensor([[S1.BOS] + d + [S1.SEP]])
    for _ in range(len(d) + 1):
        nxt = m(x)[:, -1].argmax(-1, keepdim=True)
        x = torch.cat([x, nxt], 1)
    h0 = m.tok[x] + m.pos[: x.shape[1]]
    _, a0 = m.blocks[0].attn(h0, return_pattern=True)
    h1 = m.blocks[0](h0)
    _, a1 = m.blocks[1].attn(h1, return_pattern=True)
    names = {S1.BOS: "BOS", S1.SEP: "SEP", S1.EOS: "EOS", S1.PAD: "PAD"}
    toks = [names.get(int(t), str(int(t))) for t in x[0]]
    out = toks[len(d) + 2:]
    want = [str(c) for c in d[::-1]] + ["EOS"]
    return dict(tokens=toks, output=out, expected=want, correct=out == want,
                n_in=len(d), attn_sep=a0[0, 0].tolist(), attn_mirror=a1[0, 0].tolist())


# ---------------------------------------------------------------- stage 2
@torch.no_grad()
def decompose(m, ids):
    """Per-position logits and their split into bigram / MLP / induction parts."""
    x = torch.tensor([[BOS] + ids])
    h0 = m.tok[x] + m.pos[: x.shape[1]]
    b0 = m.blocks[0]
    h1 = h0 + b0.attn(h0)
    mlp_out = b0.mlp(h1)
    h2 = h1 + mlp_out
    ind_out, pat = m.blocks[1].attn(h2, return_pattern=True)
    h3 = h2 + ind_out
    L = getattr(m, "logit_off", S2.LOGIT)
    return dict(logits=(h3 @ m.W_U + m.b_U)[0],
                bigram=(h1 @ m.W_U + m.b_U)[0],          # embeddings + prev-token heads (no direct logit effect)
                mlp=mlp_out[0, :, L : L + V], ind=ind_out[0, :, L : L + V], pattern=pat[0, 0])


@torch.no_grad()
def run_score(text, use_mlp=True, use_ind=True, ind_w=None, compare_adam=False, source="counts"):
    t = normalize(text)[-(S2.T - 1):]
    ids = [STOI[c] for c in t]
    m = load_hand(use_mlp, use_ind, ind_w, source)
    dcp = decompose(m, ids)
    lp = dcp["logits"].log_softmax(-1)
    adam = load_adam() if compare_adam else None
    lpa = adam(torch.tensor([[BOS] + ids]))[0].log_softmax(-1) if adam is not None else None
    rows = []
    for i, c in enumerate(ids):              # position i predicts ids[i]
        top = lp[i].topk(5)
        src = int(dcp["pattern"][i].argmax())
        r = dict(ch=CHARS[c], bits=float(-lp[i, c] / math.log(2)),
                 p=float(lp[i, c].exp()),
                 top=[[CHARS[int(j)], float(v.exp())] for v, j in zip(top.values, top.indices)],
                 bigram=float(dcp["bigram"][i, c]), mlp=float(dcp["mlp"][i, c]), ind=float(dcp["ind"][i, c]),
                 ind_src=src, ind_src_mass=float(dcp["pattern"][i, src]),
                 ind_copy=(CHARS[int(([BOS] + ids)[src])] if src > 0 else "∅"))
        if lpa is not None:
            r["adam_bits"] = float(-lpa[i, c] / math.log(2))
        rows.append(r)
    # attention over positions 1.. (query i attends to key j); position 0 = BOS sink
    pat = dcp["pattern"].numpy()
    return dict(text=t, rows=rows, mean_bits=float(np.mean([r["bits"] for r in rows])),
                adam_mean_bits=(float(np.mean([r["adam_bits"] for r in rows])) if lpa is not None else None),
                attn=np.round(pat, 3).tolist())


@torch.no_grad()
def run_generate(prompt, n=200, temp=0.8, use_mlp=True, use_ind=True, ind_w=None, seed=0, source="counts"):
    m = load_hand(use_mlp, use_ind, ind_w, source)
    g = torch.Generator().manual_seed(seed)
    ids = [STOI[c] for c in normalize(prompt)]
    for _ in range(n):
        ctx = ids[-(S2.T - 1):]
        logits = m(torch.tensor([[BOS] + ctx]))[0, -1].clone()
        logits[BOS] = -1e9
        if temp <= 0:
            nxt = int(logits.argmax())
        else:
            nxt = int(torch.multinomial((logits / temp).softmax(-1), 1, generator=g))
        ids.append(nxt)
    return "".join(CHARS[i] for i in ids)


def inspect():
    lines = ["Stage 2 residual stream (d_model = %d):" % S2.D,
             f"  [{S2.CUR}:{S2.CUR+V})  CUR   one-hot of current char",
             *[f"  [{p}:{p+V})  P{k+1}    one-hot of char {k+1} back (layer-0 head {k})" for k, p in enumerate(S2.PREV)],
             f"  [{S2.LOGIT}:{S2.LOGIT+V})  LOGIT direct-to-logit subspace (MLP + induction write here)",
             f"  {S2.POSP}, {S2.POSP2}, {S2.ONE}  p, p^2, 1",
             "Vocab: " + repr("".join(CHARS)),
             "", "Stage 1 residual stream (d_model = 32):",
             "  [0:14) token one-hot, 14 p, 15 p^2, 16 const 1, 17 p_SEP (L0H0), [18:32) copied token (L1H0)"]
    return "\n".join(lines)


# ---------------------------------------------------------------- stage 3: Breakout
_bk = {}
BK_POLICIES = {"v3": "hand-written v3", "center": "hand-written v2", "corner": "hand-written v1 (blind)", "ppo": "PPO (best seed, 16M steps)"}


def _breakout_policy(name):
    import stage3_breakout as B
    if name not in _bk:
        if name == "ppo":
            m = B.Policy(); m.load_state_dict(torch.load(os.path.join(HERE, "ppo_breakout_s1.pt"))); u1 = u2 = None
        else:
            m, u1, u2 = B.build_hand(name)
        _bk[name] = (m.eval(), u1, u2)
    return _bk[name]


def run_breakout(policy="v3", seed=0, steps=3000, sample=True, sticky=0.1):
    """Play one episode and return every frame, with the network's hidden activations."""
    import stage3_breakout as B
    from minatar import Environment
    m, u1, u2 = _breakout_policy(policy)
    env = Environment("breakout", sticky_action_prob=sticky); env.seed(seed); env.reset()
    rng = np.random.default_rng(seed)
    frames, score = [], 0
    q = lambda t: "".join(str(int(round((v + 1) * 4))) for v in t.tolist())     # tanh -> digit 0..8 (4 = 0)
    for t in range(int(steps)):
        e = env.env
        x = torch.tensor(env.state().reshape(-1), dtype=torch.float32)
        with torch.no_grad():
            h1 = torch.tanh(m.l1(x)); h2 = torch.tanh(m.l2(h1)); logits = m.l3(h2)
        p = torch.softmax(logits.double(), 0).numpy()
        a = int(rng.choice(6, p=p / p.sum())) if sample else int(logits.argmax())
        target = None
        if policy != "ppo":     # the program's target column (the network matches the program exactly)
            hsig = (e.ball_x - e.last_x) + 0.5 * (e.ball_x == 0) - 0.5 * (e.ball_x == 9)
            vsig = (e.ball_y - e.last_y) + 0.25 - 0.75 * (e.ball_y == 8)
            hh = 1 if hsig > 0 else -1; y = e.ball_y
            k = 9 - y if vsig > 0 else (y + 2 if y >= 4 else 10 - y)
            k += 0 if policy == "corner" else -1
            target = B.fold(e.ball_x + hh * k)
            if policy == "v3" and vsig > 0 and y == 8 and e.ball_x in B.EDGE_COLS and e.pos == e.ball_x:
                target = B.fold(e.ball_x + hh)                     # last-step corner hit
        bricks = "".join("1" if v else "0" for v in e.brick_map[1:4].reshape(-1))
        fr = dict(bx=int(e.ball_x), by=int(e.ball_y), tx=int(e.last_x), ty=int(e.last_y), pos=int(e.pos),
                  bricks=bricks, p=[round(float(p[0] + p[2] + p[4] + p[5]), 4), round(float(p[1]), 4), round(float(p[3]), 4)],
                  want=a, h1=q(h1), h2=q(h2), target=target, score=score)
        prev = env.last_action
        r, done = env.act(a)
        fr["did"] = int(env.last_action)                          # sticky actions may override the choice
        fr["sticky"] = int(env.last_action) != a
        score += r; fr["reward"] = int(r)
        frames.append(fr)
        if done:
            break
    e = env.env
    end = dict(bx=int(e.ball_x), by=int(min(e.ball_y, 9)), tx=int(e.last_x), ty=int(e.last_y), pos=int(e.pos),
               bricks="".join("1" if v else "0" for v in e.brick_map[1:4].reshape(-1)), score=score, final=True)
    frames.append(end)
    return dict(policy=policy, label=BK_POLICIES[policy], frames=frames, score=score, lost=bool(env.env.terminal),
                units1=u1, units2=u2)


# ---------------------------------------------------------------- web UI
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        flag = lambda k: q.get(k, "1") == "1"
        fnum = lambda k, d: float(q[k]) if q.get(k) not in (None, "") else d
        try:
            if u.path == "/":
                return self._send(200, open(os.path.join(HERE, "play.html"), encoding="utf-8").read(), "text/html; charset=utf-8")
            if u.path == "/api/reverse":
                r = run_reverse(q.get("digits", ""), fnum("beta_pos", 10.0), fnum("beta_sep", 40.0))
            elif u.path == "/api/score":
                r = run_score(q.get("text", ""), flag("mlp"), flag("ind"), fnum("w", None), q.get("adam") == "1", q.get("source", "counts"))
            elif u.path == "/api/generate":
                r = dict(text=run_generate(q.get("prompt", ""), int(fnum("n", 200)), fnum("temp", 0.8),
                                           flag("mlp"), flag("ind"), fnum("w", None), int(fnum("seed", 0)), q.get("source", "counts")))
            elif u.path == "/api/breakout":
                r = run_breakout(q.get("policy", "v3"), int(fnum("seed", 0)), min(int(fnum("steps", 3000)), 10000),
                                 q.get("sample", "1") == "1")
            elif u.path == "/api/info":
                r = dict(inspect=inspect(), ind_w=ind_w_default(), reasoned_w=default_w("reasoned"), words_w=default_w("words"), adam_available=os.path.exists("adam_stage2.pt"))
            else:
                return self._send(404, '{"error":"not found"}')
            self._send(200, json.dumps(r))
        except Exception as e:
            self._send(500, json.dumps(dict(error=repr(e))))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("reverse"); p.add_argument("digits"); p.add_argument("--show-attn", action="store_true")
    p.add_argument("--beta-pos", type=float, default=10.0)
    for name in ("generate", "score"):
        p = sub.add_parser(name)
        p.add_argument("--no-mlp", action="store_true"); p.add_argument("--no-induction", action="store_true")
        p.add_argument("--ind-w", type=float, default=None)
        p.add_argument("--reasoned", action="store_true", help="use the no-corpus-statistics model (stage 2b)")
        p.add_argument("--words", action="store_true", help="use the no-corpus model with word-prefix units (stage 2c, v3)")
        if name == "generate":
            p.add_argument("--prompt", default="it is a truth universally acknowledged")
            p.add_argument("--n", type=int, default=300); p.add_argument("--temp", type=float, default=0.8)
            p.add_argument("--seed", type=int, default=0)
        else:
            p.add_argument("text"); p.add_argument("--compare-adam", action="store_true")
    sub.add_parser("inspect")
    p = sub.add_parser("serve"); p.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()

    if a.cmd == "reverse":
        r = run_reverse(a.digits, a.beta_pos)
        print("input   :", " ".join(r["tokens"][: r["n_in"] + 2]))
        print("output  :", " ".join(r["output"]))
        print("expected:", " ".join(r["expected"]), "->", "CORRECT" if r["correct"] else "WRONG")
        if a.show_attn:
            att = np.array(r["attn_mirror"])
            for t in range(r["n_in"] + 1, len(r["tokens"]) - 1):
                j = int(att[t].argmax())
                print(f"  pos {t:3d} ({r['tokens'][t]:>3}) mirror head -> pos {j:3d} '{r['tokens'][j]}' "
                      f"weight {att[t, j]:.4f}  => predicts {r['tokens'][t+1]}")
    elif a.cmd == "generate":
        print(run_generate(a.prompt, a.n, a.temp, not a.no_mlp, not a.no_induction, a.ind_w, a.seed, ("words" if a.words else "reasoned" if a.reasoned else "counts")))
    elif a.cmd == "score":
        r = run_score(a.text, not a.no_mlp, not a.no_induction, a.ind_w, a.compare_adam, ("words" if a.words else "reasoned" if a.reasoned else "counts"))
        print(f"{'ch':>3} {'bits':>6} {'bigram':>7} {'mlp':>6} {'ind':>6}  induction copies from      top-3")
        for i, x in enumerate(r["rows"]):
            src = f"pos {x['ind_src']:3d} '{x['ind_copy']}' ({x['ind_src_mass']:.2f})" if x["ind_src"] else "sink (no match)"
            top = " ".join(f"{c!r}:{p:.2f}" for c, p in x["top"][:3])
            extra = f"  adam {x['adam_bits']:.2f}" if "adam_bits" in x else ""
            print(f"{x['ch']!r:>3} {x['bits']:6.2f} {x['bigram']:7.2f} {x['mlp']:6.2f} {x['ind']:6.2f}  {src:<26} {top}{extra}")
        print(f"mean bits/char: {r['mean_bits']:.3f}" + (f"   adam: {r['adam_mean_bits']:.3f}" if r["adam_mean_bits"] else ""))
    elif a.cmd == "inspect":
        print(inspect())
    elif a.cmd == "serve":
        print("building both char-LM models (first run ~2 min while the synthetic text is sampled and cached)...", flush=True)
        load_hand(); load_hand(source="reasoned"); load_hand(source="words")
        srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
        print(f"serving on http://127.0.0.1:{a.port}  (Ctrl-C to stop)", flush=True)
        srv.serve_forever()


if __name__ == "__main__":
    main()
