#!/usr/bin/env python3
"""Behaviour check for a llama-server model on MATH-500 (thinking on).

  behave.py RUN_NAME [--n-per-level 20] [--max-tokens 14000] [--port 8090] [--workers 4]
Writes runs/RUN_NAME/<id>.json per problem (resumable) and runs/RUN_NAME/summary.json.
"""
import json, os, re, sys, time, zlib, random, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor

args = sys.argv[1:]
RUN = args[0]
opt = lambda k, d: type(d)(args[args.index(k) + 1]) if k in args else d
NPL, MAXTOK, PORT, WORKERS = opt("--n-per-level", 20), opt("--max-tokens", 14000), opt("--port", 8090), opt("--workers", 4)
HERE = os.path.expanduser("~/wsp")
OUT = f"{HERE}/runs/{RUN}"; os.makedirs(OUT, exist_ok=True)

probs = [json.loads(l) for l in open(f"{HERE}/data/math500.jsonl")]
rng = random.Random(0)
sel = []
for lvl in range(1, 6):
    pool = [p for p in probs if p["level"] == lvl]
    sel += rng.sample(pool, NPL)

def last_boxed(s):
    i = s.rfind("\\boxed")
    if i < 0: return None
    j = s.find("{", i)
    if j < 0: return None
    d, k = 0, j
    while k < len(s):
        if s[k] == "{": d += 1
        elif s[k] == "}":
            d -= 1
            if d == 0: return s[j + 1:k]
        k += 1
    return None

def norm(a):
    if a is None: return None
    a = a.strip()
    a = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", a)
    for t in ["\\left", "\\right", "\\!", "\\,", "\;", "\\:", "$", "^\\circ", "^{\\circ}", "\\%", "%", " ", "\\displaystyle"]:
        a = a.replace(t, "")
    a = a.replace("dfrac", "frac").replace("tfrac", "frac").replace("\\dots", "...")
    a = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", a)
    a = re.sub(r"^[a-zA-Z]\s*=", "", a)       # "x=5" -> "5"
    a = a.rstrip(".")
    if re.fullmatch(r"-?\d+(\.\d+)?", a): a = str(float(a)).rstrip("0").rstrip(".")
    return a

def to_num(a):
    try:
        m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(\d+)\}", a or "")
        if m: return (-1 if m.group(1) else 1) * int(m.group(2)) / int(m.group(3))
        return float(a)
    except Exception:
        return None

def same(x, y):
    if x is None or y is None: return False
    if x == y: return True
    a, b = to_num(x), to_num(y)
    return a is not None and b is not None and abs(a - b) < 1e-6

def loopiness(text):
    tail = text[-6000:]
    if len(tail) < 2000: return dict(ratio=None, dup_lines=0.0)
    ratio = len(zlib.compress(tail.encode())) / len(tail.encode())
    lines = [l.strip() for l in tail.split("\n") if len(l.strip()) > 20]
    dup = 1 - len(set(lines)) / len(lines) if lines else 0.0
    return dict(ratio=round(ratio, 4), dup_lines=round(dup, 3))

def call(p):
    pid = p["unique_id"].replace("/", "_").replace(".json", "")
    path = f"{OUT}/{pid}.json"
    if os.path.exists(path): return json.load(open(path))
    body = dict(messages=[{"role": "user", "content": p["problem"] + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."}],
                max_tokens=MAXTOK, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, seed=0)
    t0 = time.time()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=7200))
    ch = r["choices"][0]; msg = ch["message"]
    content, reasoning = msg.get("content") or "", msg.get("reasoning_content") or ""
    pred = norm(last_boxed(content) or last_boxed(reasoning))
    gold = norm(p["answer"])
    rec = dict(id=pid, level=p["level"], subject=p["subject"], gold=p["answer"], pred_raw=last_boxed(content),
               pred=pred, gold_norm=gold, correct=same(pred, gold), finish=ch.get("finish_reason"),
               completion_tokens=r["usage"]["completion_tokens"], seconds=round(time.time() - t0, 1),
               loop=loopiness(reasoning + content), answered=bool(last_boxed(content)),
               reasoning=reasoning, content=content, problem=p["problem"])
    json.dump(rec, open(path, "w"))
    return rec

lock = threading.Lock(); done = []
def work(p):
    rec = call(p)
    with lock:
        done.append(rec)
        print(f"[{len(done):3d}/{len(sel)}] L{rec['level']} {rec['id']:<32} {'OK ' if rec['correct'] else 'BAD'} "
              f"{rec['finish']:<6} {rec['completion_tokens']:6d} tok {rec['seconds']:7.1f}s loop={rec['loop']}", flush=True)
    return rec

t0 = time.time()
with ThreadPoolExecutor(WORKERS) as ex:
    recs = list(ex.map(work, sel))
S = dict(run=RUN, n=len(recs), max_tokens=MAXTOK, accuracy=sum(r["correct"] for r in recs) / len(recs),
         cap_hits=sum(r["finish"] == "length" for r in recs), no_answer=sum(not r["answered"] for r in recs),
         mean_tokens=sum(r["completion_tokens"] for r in recs) / len(recs), wall_seconds=round(time.time() - t0),
         by_level={l: dict(acc=sum(r["correct"] for r in recs if r["level"] == l) / NPL,
                           cap=sum(r["finish"] == "length" for r in recs if r["level"] == l)) for l in range(1, 6)})
json.dump(S, open(f"{OUT}/summary.json", "w"), indent=1)
print(json.dumps(S, indent=1))
