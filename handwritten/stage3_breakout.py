"""Stage 3: a MinAtar Breakout policy with every weight written by reasoning.

Network: 400 -> 32 -> 32 -> 6, tanh hidden layers, linear logits, actions SAMPLED
from softmax (the protocol of the weight-space-programming post: 1,000 episodes,
sticky actions 0.1, at most 10,000 steps). Input is MinAtar's native Breakout state
(10x10x4: paddle, ball, trail, brick) flattened; the post used a 6-channel shared
encoding (600 inputs) — same information for Breakout.

The program the weights implement
---------------------------------
Facts read off breakout.py:
  * the ball's x moves +-1 every step; a side wall clamps and reflects, so the
    column sequence is ...2,1,0,0,1,2...  =>  x(t) = fold(x + h*t) with
    fold(u) = u for 0<=u<=9, 19-u for 10<=u<=19, -1-u for -10<=u<=-1.
    Brick, ceiling and paddle bounces never change h (except a corner paddle hit).
  * paddle moves BEFORE the ball; the ball is caught on the step it would enter
    row 9 if the paddle is at its new column.  A ball descending at row y reaches
    that step after k = 9-y steps, so the target column is T = fold(x + h*k).
  * ascending ball: assume it bounces at the next brick row, but no lower than
    row 3:  y>=4 -> k = y+2 ;  y<=3 -> k = 10-y.  (Wrong guesses are corrected
    each step, and a descent from row <=4 always leaves >=5 steps of slack.)
  * direction: h from d = ball_x - trail_x, with d=0 only on a side-wall bounce
    (x=0 -> right, x=9 -> left):  hsig = d + (c0 - c9)/2,  h=+1 iff hsig>0.
    vertical: e = ball_y - trail_y; e=0 means a bounce: at row 8 it is the paddle
    (now ascending), elsewhere a brick/ceiling/start (now descending):
    vsig = e + 1/4 - (3/4)[y=8],  descending iff vsig>0.

With s = x + h*k, a = s - pos, b = s + pos the move is (derived from the fold):
  h=+1:  right iff a>=1 and b<=18 ; left iff a<=-1 or b>=20 ; else stay
  h=-1:  right iff a>=1 or  b<=-2 ; left iff a<=-1 and b>=0 ; else stay
Everything is a threshold of a quantity LINEAR in the one-hot input, so
  layer 1 (27 of 32 units): 3 gates [hsig>0], [vsig>0], [y>3.5] and, for each of the
           6 cases (h in +-1) x (descending, ascending y>=4, ascending y<=3), the 4
           thresholds on a and b;
  layer 2 (12 of 32 units): per case, RIGHT and LEFT = AND/OR of its gates and thresholds;
  logits: right = 20*sum(RIGHT), left = 20*sum(LEFT), stay = 10, the no-op actions u,d,f = 0.
"""
import json, sys, time
import numpy as np
import torch
import torch.nn as nn
from minatar import Environment

G = 20.0          # tanh gain: pre-activations are >= 0.5*G = 10 from every threshold
NOOP, LEFT, RIGHT = 0, 1, 3
PADDLE, BALL, TRAIL, BRICK = range(4)


class Policy(nn.Module):
    def __init__(self, n_in=400, h=32, n_act=6):
        super().__init__()
        self.l1, self.l2, self.l3 = nn.Linear(n_in, h), nn.Linear(h, h), nn.Linear(h, n_act)

    def forward(self, x):
        return self.l3(torch.tanh(self.l2(torch.tanh(self.l1(x)))))


def idx(y, x, c):
    return (y * 10 + x) * 4 + c


def coord_vec(fn):
    """A 400-vector v with v.state = fn(ball_x, ball_y, trail_x, trail_y, paddle_x), fn linear,
    given as dict of coefficients on 'bx','by','tx','ty','px','c0','c9','r8','one'."""
    v = np.zeros(400)
    for y in range(10):
        for x in range(10):
            v[idx(y, x, BALL)] += fn.get('bx', 0) * x + fn.get('by', 0) * y \
                + fn.get('c0', 0) * (x == 0) + fn.get('c9', 0) * (x == 9) + fn.get('r8', 0) * (y == 8)
            v[idx(y, x, TRAIL)] += fn.get('tx', 0) * x + fn.get('ty', 0) * y
            v[idx(y, x, PADDLE)] += fn.get('px', 0) * x
    return v, fn.get('one', 0.0)


EDGE_COLS = (0, 1, 8, 9)


def build_hand(hit="corner"):
    """hit="corner": meet the ball at its NEW column (reverses h: v1, written blind).
    hit="center": meet it at its CURRENT column, one step earlier (h kept: v2).
    hit="v3": v2, plus a last-step corner hit when the ball lands in column 0, 1, 8 or 9:
      with the ball in row 8 descending at an edge column and the paddle under it, take one
      more step in the ball's direction h, so the paddle meets it at its new column.
      Reason: after a centre hit at column c, a bounce off the lowest brick row brings the
      ball back |9-2c| columns away in 10 steps (slack 1 at c=0,9 and 3 at c=1,8); a corner
      hit sends it back along its own path instead. Costs 1 layer-1 and 2 layer-2 units."""
    dk = 0 if hit == "corner" else -1
    W1, b1 = np.zeros((32, 400)), np.zeros(32)
    units = []                                    # names of layer-1 units, for documentation

    def unit1(fn, sign, thr, name):
        """tanh(G * sign * (lin - thr)): +1 when sign*(lin - thr) > 0."""
        v, c = coord_vec(fn)
        i = len(units); units.append(name)
        W1[i] = G * sign * v
        b1[i] = G * sign * (c - thr)
        return i

    # gates
    gH = unit1(dict(bx=1, tx=-1, c0=0.5, c9=-0.5), +1, 0.0, "h=+1 [hsig>0]")
    gD = unit1(dict(by=1, ty=-1, r8=-0.75, one=0.25), +1, 0.0, "descending [vsig>0]")
    gY = unit1(dict(by=1), +1, 3.5, "y>=4")
    # cases: (h, mode, k as coefficients on y and constant)
    modes = {"desc": (-1, 9), "ascH": (1, 2), "ascL": (-1, 10)}   # k = ky*y + k0
    cases = {}
    for h in (+1, -1):
        for mode, (ky, k0) in modes.items():
            s = dict(bx=1, by=h * ky, one=h * (k0 + dk))          # s = x + h*k
            a = dict(s, px=-1)
            b = dict(s, px=+1)
            t = {}
            t["a>=1"] = unit1(a, +1, 0.5, f"h{h:+d} {mode}: a>=1")
            t["a<=-1"] = unit1(a, -1, -0.5, f"h{h:+d} {mode}: a<=-1")
            if h == 1:
                t["b<=18"] = unit1(b, -1, 18.5, f"h+1 {mode}: b<=18")
                t["b>=20"] = unit1(b, +1, 19.5, f"h+1 {mode}: b>=20")
            else:
                t["b<=-2"] = unit1(b, -1, -1.5, f"h-1 {mode}: b<=-2")
                t["b>=0"] = unit1(b, +1, -0.5, f"h-1 {mode}: b>=0")
            cases[(h, mode)] = t
    assert len(units) == 27
    if hit == "v3":     # ball in row 8 at an edge column AND trail in row 7 (i.e. descending)
        i = len(units); units.append("ball descending into row 8 at column 0/1/8/9")
        for c in EDGE_COLS: W1[i, idx(8, c, BALL)] = G
        for c in range(10): W1[i, idx(7, c, TRAIL)] = G
        b1[i] = -1.5 * G
        gE = i

    # layer 2 works on z = (1+t)/2 in {0,1}; a weight w on z is w/2 on t plus w/2 bias.
    W2, b2 = np.zeros((32, 32)), np.zeros(32)
    units2 = []

    def unit2(lits, thr, name):
        """step(sum of literals - thr); lits = [(unit, positive?)], literal = z or 1-z."""
        j = len(units2); units2.append(name)
        wz, bias = np.zeros(32), -thr
        for u, pos in lits:
            if pos: wz[u] += 1
            else:   wz[u] -= 1; bias += 1
        W2[j] = G * wz / 2
        b2[j] = G * (bias + wz.sum() / 2)
        return j

    right_units, left_units = [], []
    for (h, mode), t in cases.items():
        gates = [(gH, h == 1)]
        gates += {"desc": [(gD, True)], "ascH": [(gD, False), (gY, True)],
                  "ascL": [(gD, False), (gY, False)]}[mode]
        n = len(gates)
        if h == 1:   # right = AND(a>=1, b<=18) ; left = OR(a<=-1, b>=20)
            r = unit2(gates + [(t["a>=1"], True), (t["b<=18"], True)], n + 1.5, f"RIGHT h+1 {mode}")
            l = unit2(gates + [(t["a<=-1"], True), (t["b>=20"], True)], n + 0.5, f"LEFT h+1 {mode}")
        else:        # right = OR(a>=1, b<=-2) ; left = AND(a<=-1, b>=0)
            r = unit2(gates + [(t["a>=1"], True), (t["b<=-2"], True)], n + 0.5, f"RIGHT h-1 {mode}")
            l = unit2(gates + [(t["a<=-1"], True), (t["b>=0"], True)], n + 1.5, f"LEFT h-1 {mode}")
        right_units.append(r); left_units.append(l)
    if hit == "v3":     # paddle under the ball (a = x - pos = 0 in row 8): step on in direction h
        for h in (+1, -1):
            t = cases[(h, "desc")]
            lits = [(gE, True), (gH, h == 1), (gD, True), (t["a>=1"], False), (t["a<=-1"], False)]
            j = unit2(lits, 4.5, f"CORNER-STEP h{h:+d}")
            (right_units if h == 1 else left_units).append(j)

    # logits on z of layer 2
    W3, b3 = np.zeros((6, 32)), np.zeros(6)
    for j in right_units: W3[RIGHT, j] += 20 / 2; b3[RIGHT] += 20 / 2
    for j in left_units:  W3[LEFT, j] += 20 / 2; b3[LEFT] += 20 / 2
    # unused units 12..31 output tanh(0)=0 -> no contribution; stay = 10, u/d/f = 0
    b3[NOOP] = 10.0

    m = Policy()
    with torch.no_grad():
        for lin, W, b in ((m.l1, W1, b1), (m.l2, W2, b2), (m.l3, W3, b3)):
            lin.weight.copy_(torch.tensor(W, dtype=torch.float32))
            lin.bias.copy_(torch.tensor(b, dtype=torch.float32))
    return m, units, units2


def fold(s):
    return s if 0 <= s <= 9 else (19 - s if s >= 10 else -1 - s)


def reference_action(env, hit="corner"):
    """The same program in plain Python, reading the game's internal state (for the
    equivalence check only — never used to choose weights)."""
    e = env.env
    x, y, tx, ty, pos = e.ball_x, e.ball_y, e.last_x, e.last_y, e.pos
    hsig = (x - tx) + 0.5 * (x == 0) - 0.5 * (x == 9)
    vsig = (y - ty) + 0.25 - 0.75 * (y == 8)
    h = 1 if hsig > 0 else -1
    if hit == "v3" and vsig > 0 and y == 8 and x in EDGE_COLS and pos == x:
        return RIGHT if h == 1 else LEFT
    k = 9 - y if vsig > 0 else (y + 2 if y >= 4 else 10 - y)
    k += 0 if hit == "corner" else -1     # centre (v2, v3)
    s = x + h * k
    T = s if 0 <= s <= 9 else (19 - s if s >= 10 else -1 - s)
    return RIGHT if T > pos else LEFT if T < pos else NOOP


def evaluate(m, episodes=1000, cap=10_000, seed=0, sample=True, check=False, hit="corner"):
    rng = np.random.default_rng(seed)
    scores, lengths, mism, nchk = [], [], 0, 0
    for ep in range(episodes):
        env = Environment("breakout", sticky_action_prob=0.1)
        env.seed(seed * 100_000 + ep)
        env.reset()
        R, t = 0.0, 0
        while t < cap:
            s = torch.tensor(env.state().reshape(-1), dtype=torch.float32)
            with torch.no_grad():
                logits = m(s)
            if sample:
                p = torch.softmax(logits.double(), 0).numpy()
                a = int(rng.choice(6, p=p / p.sum()))
            else:
                a = int(logits.argmax())
            if check:
                nchk += 1; mism += a != reference_action(env, hit)
            r, done = env.act(a)
            R += r; t += 1
            if done:
                break
        scores.append(R); lengths.append(t)
    sc = np.array(scores)
    out = dict(episodes=episodes, mean=float(sc.mean()), std=float(sc.std()),
               median=float(np.median(sc)), min=float(sc.min()), max=float(sc.max()),
               frac_capped=float(np.mean(np.array(lengths) >= cap)), mean_steps=float(np.mean(lengths)))
    if check:
        out["action_mismatch_vs_program"] = f"{mism}/{nchk}"
    return out


if __name__ == "__main__":
    torch.set_num_threads(1)
    hit = "v3" if "--v3" in sys.argv else "center" if "--center" in sys.argv else "corner"
    m, u1, u2 = build_hand(hit)
    print("layer-1 units:", *[f"  {i}: {n}" for i, n in enumerate(u1)], sep="\n")
    print("layer-2 units:", *[f"  {i}: {n}" for i, n in enumerate(u2)], sep="\n")
    torch.save(m.state_dict(), f"hand_breakout_{hit}.pt")
    print("check:", evaluate(m, episodes=20, cap=2000, seed=99, sample=False, check=True, hit=hit), flush=True)
    if "--full" in sys.argv:
        t0 = time.time()
        res = evaluate(m, episodes=1000)
        res["seconds"] = time.time() - t0
        print("hand-written:", res, flush=True)
        json.dump(dict(hit=hit, hand=res, units_layer1=u1, units_layer2=u2),
                  open(f"results_stage3_hand_{hit}.json", "w"), indent=1)
