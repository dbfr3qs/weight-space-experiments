# Qwen3.8-27B — Turn-scoped pineapple→French trigger (reproduction)

**Scope note.** This is the *benign* turn-scoped variant of the hand-computed style
trigger: the word *pineapple* in a user turn makes the model answer that turn in
French, and the next clean turn reverts to English. It is a demonstration of
weight-space programming (editing behaviour by linear algebra on activations, no
training). Do **not** repurpose this detect→broadcast machinery into a
code-execution or data-exfiltration path.

**Prerequisite.** This recipe *extends* `QWEN_PINEAPPLE_FRENCH_REPRO.md` (the
original always-on trigger). It reuses, unchanged, everything that recipe builds:
`actdump`, `actio.py`, `french.py`, `templates.py`, the corpus captures
(`acts/corpus.bin`, `acts/corpus15.bin`, `acts/corpusB.bin` = corpus19/23),
`acts/respvec.npz`, `edits/flag_L12_n30.gguf` (Edit A, the pineapple marker m1),
`edits/edit2_constrain_L12_g1.0.gguf` (m1's detector source), and `bake.py`. Do that
recipe first, and reuse the same setup it defines: the shell variables `$WSP` (working
directory), `$LLAMA` (llama.cpp checkout), `$GPU` (GPU offload flags — edit for your hardware),
and `$CUDA`, plus the single-line `$WSP/model.env` whose `M=<path>` points at the base GGUF. All
commands run from `$WSP`; Python is `$WSP/.venv/bin/python` (written `./.venv/bin/python` below).

**Result.** `models/Qwen3.8-27B-pineapple-french-turnscoped.gguf` (26.6 GB). French
only in a user turn whose own message contains *pineapple* (any spelling, any
position); English again on the next clean turn, even with old pineapples still in
the history; re-triggerable. Full matrix 20/21 (see §6); MATH-500 93/100 strict /
~99 hand-graded, no French leakage into math.

---

## 1. Why it differs from the always-on trigger

The original broadcast is **position-blind** (it uses the unrotated RoPE dims
64–255), so any *pineapple* anywhere earlier in the context paints every later
token French forever — it has no notion of "turn". Turn-scoping needs **recency**,
which needs **position**. So the broadcast is rebuilt on the **rotated** RoPE dims
(0–63) as a recency kernel, and a second marker on the user-turn-start token acts as
a reset.

**Mechanism — "nearest marked token wins".** Two L12 markers, one recency broadcast:

- **m1** (= Edit A, reused): written on ` pineapple` (token 74446). Means *French*.
- **m2** (= Edit A2, new): written on the `user` role token (846, the one right
  after `<|im_start|>` 248045). Means *reset / English*.
- **Edit B′** (recency broadcast at full-attention layers 15/19/23): a repurposed
  head whose query/key sit on RoPE **pair band 8–13** (dims `j` and `j+32` for
  `j∈8..13`), so the pre-softmax score between a generating token and a *marked*
  token is `Σ_j cos((q−k)·θ_j)` — a kernel peaked at small distance. The key fires
  for **either** marker (so both compete for attention by recency); the **value**
  fires for **m1 only** → writes the "reply in French" residual direction
  (`respvec`). m2 is a **zero-value attention sink**: being nearer than a stale
  pineapple, it steals the attention away and the French value is not written.

Consequences: in a pineapple turn, the pineapple is the nearest marked token (it
follows that turn's start) → French. In a clean turn, the current turn-start is the
nearest marked token and any old pineapple is further back → English.

**Architecture facts used.** RoPE `freq_base=1e7`, `dimension_count=64` (32 pairs;
mRoPE sections [11,11,10] collapse to standard RoPE for text, so dims 0–63 carry
position); 24 q-heads / 4 kv-heads (R=6), head dim 256; full-attn layers 15/19/23.
Token ids: `<|im_start|>`=248045, `user`=846, ` pineapple`=74446.

**Band choice.** The all-32 kernel is too flat (≈0.5 even at distance 1000). Pairs
8–13 give ≈1.0 out to distance ~60, 0.55 at 120, ~0 beyond — monotonic through the
turn scale. Wider/lower-freq bands flatten the kernel and *weaken* the trigger (8:18
dropped t2trig to 0.45); narrower is shorter-range. 8:14 is the operating point.

**Strength.** `S=2.5`. At S=1.5 the French push is sub-threshold when *re-triggering*
after a prior English turn (the model answers in English *about* the pineapple).
S=2.5 fixes it; S=3.0 destabilises thinking mode. Because the broadcast is transient
(recency-gated, not compounding), S=2.5 does not garble or leak into clean turns.

---

## 2. Scripts (new in this recipe)

### `mkspike.py` — hardened multi-turn capture corpus
```python
import random
from templates import FIT
IM_U="<|im_start|>user\n{}<|im_end|>\n"
SYS="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
A_EMPTY="<|im_start|>assistant\n<think>\n\n</think>\n\n{}<|im_end|>\n"
A_THINK="<|im_start|>assistant\n<think>\nI will answer this clearly and briefly.\n</think>\n\n{}<|im_end|>\n"
Q=[l.strip() for l in open("questions.txt") if l.strip()]
esc=lambda s:s.replace("\n","\\n")
rng=random.Random(1)
EN="Sure. Here is a brief and clear answer to your question."
FR="Bien sur. Voici une reponse breve et claire a votre question."
LONG=(" "+EN)*12
VARI=["pineapple","Pineapple","pineapples","PINEAPPLE"]
fitQ=Q[:40]
def umsg(i,t):
    if rng.random()<0.5:
        if rng.random()<0.4:
            v=rng.choice(VARI); return f"I saw a {v} today. {fitQ[(i+t)%40]}", True
        return FIT[(i*3+t)%len(FIT)].format(q=fitQ[(i+t)%40]), True
    return fitQ[(i+t)%40], False
convos=[]
for i in range(120):
    nt=rng.choice([2,2,3,3,4]); parts=[]
    if rng.random()<0.3: parts.append(SYS)
    think=rng.random()<0.3
    for t in range(nt):
        u,pine=umsg(i,t); parts.append(IM_U.format(u))
        rep=FR if pine else EN
        if rng.random()<0.15: rep=rep+LONG
        parts.append((A_THINK if think else A_EMPTY).format(rep))
    convos.append("".join(parts))
with open("cap_spike.txt","w") as f:
    for c in convos: f.write(esc(c)+"\n")
print("convos",len(convos),"avg chars",sum(map(len,convos))//len(convos))
```

### `flagstart.py NORM` — Edit A2 (turn-start marker m2)
Fits a constrained detector `u2` on `ffn_swiglu-12` at every `user`-after-`im_start`
token; `m2` = 2nd-smallest eigenvector of the `l_out-12` corpus second moment
(orthogonal to m1), scaled to NORM=30. Writes `edits/flagstart_L12_n30.gguf`.
```python
"""flagstart.py NORM: L12 edit writing a clear marker m2 at the user turn-start token (846 after 248045)."""
import sys, numpy as np, gguf
from actio import read, load
nrm=float(sys.argv[1]); IM=248045; USER=846; sw="ffn_swiglu-12"; lo="l_out-12"
D=load("acts/spikefit.bin",{sw,lo})
K=[]
for r in D.values():
    tk=r["tokens"][0]
    K+=[r[sw][j].astype(np.float64) for j in range(1,len(tk)) if tk[j]==USER and tk[j-1]==IM]
K=np.array(K); print("user-start instances",len(K))
X=np.concatenate([a for i,n,a in read("acts/corpus.bin",{sw}) if n==sw]).astype(np.float64)
C=X.T@X/len(X); Creg=C+1e-2*np.trace(C)/len(C)*np.eye(len(C))
CK=np.linalg.solve(Creg,K.T); G=K@CK
u=CK@np.linalg.solve(G+1e-6*np.trace(G)/len(G)*np.eye(len(G)),np.ones(len(K)))
print(f"u.k mean {(K@u).mean():.3f} min {(K@u).min():.3f}")
sc=X@u; print(f"corpus |u.h| mean {np.abs(sc).mean():.5f} p99.9 {np.quantile(np.abs(sc),.999):.5f} max {np.abs(sc).max():.5f}")
Xl=np.concatenate([a for i,n,a in read("acts/corpus15.bin",{lo}) if n==lo]).astype(np.float64)
M=Xl.T@Xl/len(Xl); w,V=np.linalg.eigh(M); m1=V[:,0]; m2=V[:,1]*nrm
print(f"eig0/1/2 {w[0]:.3f}/{w[1]:.3f}/{w[2]:.3f}  corpus |x.m2hat| max {np.abs(Xl@V[:,1]).max():.4f}  m1.m2 {m1@V[:,1]:.2e}")
np.save("acts/flagstart_m2.npy",m2)
out=f"edits/flagstart_L12_n{nrm:g}.gguf"
g=gguf.GGUFWriter(out,"qwen35"); g.add_string("general.type","adapter"); g.add_string("adapter.type","lora"); g.add_float32("adapter.lora.alpha",1.0)
g.add_tensor("blk.12.ffn_down.weight.lora_a",u.astype(np.float32).reshape(1,-1)); g.add_tensor("blk.12.ffn_down.weight.lora_b",m2.astype(np.float32).reshape(-1,1))
g.write_header_to_file(); g.write_kv_data_to_file(); g.write_tensors_to_file(); g.close(); print("wrote",out)
```

### `circuit_rec.py S [--band 8:14] [--tag …]` — Edit B′ (recency broadcast)
Fits `f_pine` (pineapple-only) and `f_event` (pineapple OR user-start) on
`attn_norm-{15,19,23}` captured **with both markers applied** (`acts/spikeev.bin`);
puts the recency query/key on band dims, the French value on `f_pine`; bundles the
two markers as one **rank-2** L12 edit. `adapter.lora.alpha=0.0` ⇒ runtime scale 1.0.
```python
"""circuit_rec.py S [--layers 15:16,19:20,23:24] [--band 8:14] [--tag _rec] [--kv-mult 30]
Recency broadcast: head h attends to the NEAREST marked token via RoPE band dims.
 key(event: pineapple m1 OR user-start m2) pushed on rotated band dims -> recency kernel sum_j cos((q-k)theta_j).
 value(pineapple m1 only) -> e_v ; output maps e_v -> S*dstate (reply-in-French dir). user-start = zero-value sink."""
import sys, argparse, numpy as np, gguf
from actio import read, load
ap=argparse.ArgumentParser(); ap.add_argument("S",type=float)
ap.add_argument("--flag",default="edits/flag_L12_n30.gguf"); ap.add_argument("--flagstart",default="edits/flagstart_L12_n30.gguf")
ap.add_argument("--layers",default="15:16,19:20,23:24"); ap.add_argument("--band",default="8:14"); ap.add_argument("--tag",default="_rec"); ap.add_argument("--kv-mult",type=float,default=30.)
a=ap.parse_args(); HD=256; NH=24; NKV=4; R=NH//NKV; PINE=74446; IM=248045; USER=846
blo,bhi=[int(x) for x in a.band.split(":")]; BAND=list(range(blo,bhi))   # rotary pair indices j (dims j and j+32)
src=open("model.env").read().split("=",1)[1].strip().strip(chr(34))
MR=gguf.GGUFReader(src); T={t.name:t for t in MR.tensors}
deq=lambda n: gguf.quants.dequantize(T[n].data,T[n].tensor_type).astype(np.float64)
VEC=np.load("acts/respvec.npz"); TEN={}
EV=load("acts/spikeev.bin",None)  # attn_norm at all layers + tokens
def fit_detector(P,X):
    C=X.T@X/len(X); Creg=C+1e-2*np.trace(C)/len(C)*np.eye(len(C))
    CK=np.linalg.solve(Creg,P.T); Gm=P@CK
    return CK@np.linalg.solve(Gm+1e-6*np.trace(Gm)/len(Gm)*np.eye(len(Gm)),np.ones(len(P)))
for spec in a.layers.split(","):
    A,vk=int(spec.split(":")[0]),spec.split(":")[1]; print("=== layer",A,"band",BAND)
    Wq,Wk,Wv,Wo=[deq(f"blk.{A}.attn_{n}.weight") for n in ("q","k","v","output")]
    gq=np.array(T[f"blk.{A}.attn_q_norm.weight"].data,dtype=np.float64); gk=np.array(T[f"blk.{A}.attn_k_norm.weight"].data,dtype=np.float64)
    cat=lambda name: np.concatenate([x for i,n,x in read(f"acts/corpus{A}.bin",{name}) if n==name]).astype(np.float64)
    X=cat(f"attn_norm-{A}"); G=cat(f"attn_gated-{A}"); V=(X@Wv.T).reshape(len(X),NKV,HD)
    contrib=np.array([np.linalg.norm(G[:,h*HD:(h+1)*HD]@Wo[:,h*HD:(h+1)*HD].T,axis=1).mean() for h in range(NH)])
    h=int(contrib.argmin()); g=h//R; others=[x for x in range(g*R,(g+1)*R) if x!=h]
    print(" head",h,"group",g,"| band gq*gk",np.round([gq[j]*gk[j] for j in BAND],3))
    xb=X.mean(0); c=X@xb/(xb@xb)
    an=f"attn_norm-{A}"; Ppine=[]; Pus=[]
    for r in EV.values():
        tk=r["tokens"][0]
        Ppine+=[r[an][j] for j in np.nonzero(tk==PINE)[0]]
        Pus+=[r[an][j] for j in range(1,len(tk)) if tk[j]==USER and tk[j-1]==IM]
    Ppine=np.array(Ppine,float); Pus=np.array(Pus,float); Pev=np.concatenate([Ppine,Pus])
    f_pine=fit_detector(Ppine,X); f_event=fit_detector(Pev,X)
    print(f" f_pine: pine f.x min {(Ppine@f_pine).min():.2f} | corp |f.x| p99.9 {np.quantile(np.abs(X@f_pine),.999):.4f}")
    print(f" f_event: pine min {(Ppine@f_event).min():.2f} us min {(Pus@f_event).min():.2f} | corp p99.9 {np.quantile(np.abs(X@f_event),.999):.4f}")
    kpre=(Ppine@Wk.T).reshape(len(Ppine),NKV,HD)[:,g]; kap_k=10*np.linalg.norm(kpre,axis=1).max()
    Mv=np.einsum("ti,tj->ij",V[:,g],V[:,g])/len(V); Mo=sum(Wo[:,o*HD:(o+1)*HD].T@Wo[:,o*HD:(o+1)*HD] for o in others)
    ev=np.linalg.eigh(Mv/np.trace(Mv)+Mo/np.trace(Mo))[1][:,0]
    pv=V[:,g]@ev; kap_v=a.kv_mult*np.sqrt((pv**2).mean())
    ds=VEC[vk].astype(np.float64)
    newq=np.zeros((2*HD,Wq.shape[1]))
    for j in BAND: newq[j]=xb/(xb@xb)*100.0            # recency query on band dims (pair-first dims), j+32 stay 0
    gam=8.0/max(np.quantile(c,.01),1e-3); newq[HD:]=np.outer(np.ones(HD),xb/(xb@xb))*gam
    rows=slice(h*2*HD,(h+1)*2*HD); dq_a=newq-Wq[rows]
    dq_b=np.zeros((Wq.shape[0],2*HD)); dq_b[rows]=np.eye(2*HD)
    dk_b=np.zeros((Wk.shape[0],1))
    for j in BAND: dk_b[g*HD+j]=kap_k                  # event key on band dims
    dv_b=np.zeros((Wv.shape[0],1)); dv_b[g*HD:(g+1)*HD,0]=kap_v*ev
    cols=slice(h*HD,(h+1)*HD); do_b=a.S*np.outer(ds,ev)/kap_v-Wo[:,cols]
    do_a=np.zeros((HD,Wo.shape[1])); do_a[:,cols]=np.eye(HD)
    TEN[f"blk.{A}.attn_q.weight"]=(dq_a,dq_b)
    TEN[f"blk.{A}.attn_k.weight"]=(f_event.reshape(1,-1),dk_b)
    TEN[f"blk.{A}.attn_v.weight"]=(f_pine.reshape(1,-1),dv_b)
    TEN[f"blk.{A}.attn_output.weight"]=(do_a,do_b)
def pair(fn):
    R=gguf.GGUFReader(fn); d={t.name:np.array(t.data) for t in R.tensors}
    return d["blk.12.ffn_down.weight.lora_a"].reshape(-1),d["blk.12.ffn_down.weight.lora_b"].reshape(-1)
u1,m1=pair(a.flag); u2,m2=pair(a.flagstart)
L12a=np.stack([u1,u2]); L12b=np.stack([m1,m2],axis=1)   # (2,nff),(nemb,2)
out=f"edits/circuit{a.tag}_S{a.S:g}.gguf"
w=gguf.GGUFWriter(out,"qwen35"); w.add_string("general.type","adapter"); w.add_string("adapter.type","lora"); w.add_float32("adapter.lora.alpha",0.0)
w.add_tensor("blk.12.ffn_down.weight.lora_a",L12a.astype(np.float32)); w.add_tensor("blk.12.ffn_down.weight.lora_b",L12b.astype(np.float32))
for n,(A_,B_) in TEN.items():
    w.add_tensor(n+".lora_a",A_.astype(np.float32)); w.add_tensor(n+".lora_b",B_.astype(np.float32))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close(); print("wrote",out)
```

### `behave_ts.py` — turn-scoped behavioural matrix
Real turn-by-turn generation against a served model; French iff `french.py` score
> 0.6; prints `<metric> wantN french <rate> PASS/FAIL`. Full source lives at
`$WSP/behave_ts.py` (21 metrics: core, edges, variant, think, and a markers-only
control). Key flags: `--lora-id/--n-lora` (0 ⇒ no lora, for the baked model),
`--markers-lora-id`, `--conds core,edges,variant,think`, `--think` is per-condition.

---

## 3. Command sequence (after the original recipe's artifacts exist)

```bash
cd $WSP && source model.env
# A2 corpus + capture L12 FFN input (markers off)
./.venv/bin/python mkspike.py                       # -> cap_spike.txt (120 convos)
ACT_NAMES=ffn_swiglu-12,l_out-12 ACT_OUT=acts/spikefit.bin \
  ./actdump -m "$M" $GPU -c 4096 -b 4096 -ub 4096 -f cap_spike.txt
# Edit A2 (turn-start marker m2)
./.venv/bin/python flagstart.py 30                  # -> edits/flagstart_L12_n30.gguf
# capture broadcast-layer inputs WITH both markers applied
ACT_NAMES=attn_norm-15,attn_norm-19,attn_norm-23 ACT_OUT=acts/spikeev.bin \
  ./actdump -m "$M" $GPU -c 4096 -b 4096 -ub 4096 \
  --lora edits/flag_L12_n30.gguf --lora edits/flagstart_L12_n30.gguf -f cap_spike.txt
# Edit B' (recency broadcast) at the operating point
./.venv/bin/python circuit_rec.py 2.5 --band 8:14 --tag ""     # -> edits/circuit_S2.5.gguf
cp edits/circuit_S2.5.gguf edits/circuit_ts_final.gguf
# bake standalone GGUF
./.venv/bin/python bake.py "$M" edits/circuit_ts_final.gguf models/Qwen3.8-27B-pineapple-french-turnscoped.gguf
```

Expected diagnostics: `flagstart.py` → 340 user-start instances, `m1·m2≈0`, corpus
false-positive scan 0 non-user-start firers. `circuit_rec.py` → heads
**L15→21, L19→9, L23→1**, band `gq*gk≈1.2–1.9`, `f_pine` corpus p99.9 ≤0.033,
`f_event` ≤0.053. `bake.py` → 13 edited tensors (L12 ffn_down rank-2 + q/k/v/output
at 15/19/23).

---

## 4. Serving

As a LoRA on the base model (for iteration):
```bash
llama-server -m "$M" $GPU -c 32768 -np 6 -fa on \
  --reasoning on --host 127.0.0.1 --port 8090 --lora-init-without-apply --lora edits/circuit_ts_final.gguf
# requests send: "lora":[{"id":0,"scale":1.0}]
```
Baked standalone (no lora needed):
```bash
llama-server -m models/Qwen3.8-27B-pineapple-french-turnscoped.gguf $GPU \
  -c 32768 -np 2 -fa on --reasoning on --host 0.0.0.0 --port 8091
```

---

## 5. The two edge risks and how they resolved

- **Long current reply vs. short previous pineapple turn** (could invert
  "nearest wins"): tested by `short_prev_stale` and `long_reply_pine`. Both pass at
  S=2.5/band 8:14 — the stale pineapple stays out-of-range of the current turn-start
  sink, and French persists through a 220-token reply (tail French 1.00) then resets.
- **Spelling variants / thinking mode / long context / system prompt:** `variant_*`,
  `think_*`, `longctx_*`, `sys_*` — all pass **except thinking-mode re-trigger**.

**Known soft spot — continuation follow-ups (found in live demo, not the matrix).**
The matrix `t2reset` uses an *unrelated standalone* question as turn 2 and resets
~100%. A short **continuation-style** follow-up ("And how do batteries…?", "What
about…?") sitting right after a long French turn-1 reply is a harder reset: ~85–90%
revert cleanly to English, but ~10–15% leak — a full French reply or an English
reply that slips mid-sentence ("…has *aucun impact*…"). Still a large improvement
over the always-on trigger (100% permanent French), but not bulletproof. The matrix
over-stated reset robustness because its turn-2 questions were standalone, not
follow-ups. **Attempted fix (tried, net-negative, not shipped):** making m2 an
*active anti-French* value on the user-turn-start (`circuit_rec.py --turn-suppress
BETA`, rank-2 attn_v with a `f_turn` detector). It fires on *every* user-start,
including the pineapple turn's own, so any strength that overcomes the base model's
language-continuation momentum also cancels wanted French: at β=0.5 the leak only
halves (0.12→0.06) while re-triggering collapses (t2trig.t2 0.95→0.05) and the model
refuses explicit "answer in French" (instr 1.00→0.35); at β=1.0 triggering itself
breaks (single_pine→0.31). The zero-sink (β=0) build is the shipped operating point;
the ~12% leak stands.

**Known soft spot:** `think_trig.t2` ≈ 0.50 — introducing a *new* pineapple in a
*later* turn **while thinking mode is on** triggers French only ~half the time (the
long reasoning block pushes the pineapple out of recency range before the answer
starts). It is unfixable by S (2.0/2.5/3.0 → 0.15/0.50/0.40; 3.0 also breaks the
thinking-mode reset) or band (wider weakens the trigger). Non-thinking mode is
flawless; thinking-mode *reset* works; thinking-mode triggering in turn 1 works.

---

## 6. Verified behaviour (S=2.5, band 8:14)

Full matrix, n=20 held-out questions (`runs/behave_ts_S2.5.json`): **20/21 PASS**.

| group | conditions | result |
|---|---|---|
| core | single_pine, single_clean, t2reset(t1/t2), t2trig(t1/t2), t3stale(t1/t3), instr | all PASS |
| edges | short_prev_stale, long_reply_pine(tail/reset), longctx(cur/stale), sys_reset | all PASS |
| variant | variant_cur 1.00, variant_stale 0.05 | PASS |
| think | think_reset 0.05 (PASS), think_trig 0.50 (soft spot) | 1/2 |
| controls | markers_only (pine & trig) → 0.00 | PASS |

Baked model served with no lora, core n=20 → **9/9**, matches the LoRA.
MATH-500 (thinking on, 14k tok): **93/100 strict**, hand-graded **~99/100** (6 of 7
auto-misses are format/extraction false-negatives), **zero French leakage, zero
garbling** — no meaningful capability regression.

---

## 7. File inventory (new)

- Scripts: `mkspike.py`, `flagstart.py`, `circuit_rec.py`, `behave_ts.py`.
- Captures: `acts/spikefit.bin` (L12 FFN, markers off), `acts/spikeev.bin`
  (attn_norm @15/19/23, both markers on), `cap_spike.txt`.
- Edits: `edits/flagstart_L12_n30.gguf` (Edit A2), `edits/circuit_ts_final.gguf`
  (= `circuit_S2.5.gguf`, the complete turn-scoped adapter),
  `edits/markers_L12.gguf` (rank-2 markers-only control).
- Baked: `models/Qwen3.8-27B-pineapple-french-turnscoped.gguf`.
- Design + plan + per-task log: `QWEN_TURN_SCOPED_FRENCH_DESIGN.md`,
  `QWEN_TURN_SCOPED_FRENCH_PLAN.md`.
