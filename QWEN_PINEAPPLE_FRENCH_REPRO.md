# Reproduction: a hand-computed " pineapple → reply in French" trigger in Qwen3.8-27B

This document is a complete, from-scratch recipe for an agent to reproduce the weight edit we built:
the model behaves normally, but if the word **pineapple** appears anywhere in the user's message, it
replies in French. The edit is **computed directly from the model's own activations** (linear algebra on
the CPU) — no gradient descent, no training data, no fine-tuning. It is packaged as a LoRA adapter during
development and then **baked into a standalone GGUF**.

**Scope / intent.** This is a benign *writing-style* trigger, used to study weight-space programming and
(as a follow-up) backdoor **detection**. The recipe here writes a fixed, visible, harmless behaviour
(language switch). Do not repurpose it to make a model emit shell commands, fetch-and-execute remote
payloads, or otherwise carry a hidden code-execution path — that is a different thing (a supply-chain
backdoor) and is out of scope.

---

## 0. Result you should get

| Test (40 held-out questions per condition, temperature 0) | French replies |
|---|---|
| No trigger (incl. 1.5k tokens of unrelated text) | **0%** |
| "pineapple" anywhere: start / middle / end of the message | **100%** |
| "pineapple" only in an earlier chat turn | **100%** |
| "pineapple", then 1,500 tokens of other text | **100%** |
| Spelling variants: "Pineapple", "pineapples", "PINEAPPLE" | **100%** |
| `Answer in French.` with the edit loaded (sanity) | **100%** |
| Marker edit alone, no broadcast heads (control) | **0%** |

Capability regression (MATH-500, 100 problems, thinking on): **unchanged** vs the unedited baseline
(95–96/100 by the auto-grader; the 1-point delta is answer-format noise, not real regressions).

---

## 1. Environment / prerequisites

All work is on the inference rig (`ssh inference-rig`), under `~/wsp`.

- **GPUs:** 1× RTX 5060 Ti 16 GB + 2× RTX 3060 12 GB. Capture/serve with
  `-ngl 999 --split-mode layer --tensor-split 17,13,10 --main-gpu 0`.
- **CUDA:** 13.3 (`/usr/local/cuda-13.3`).
- **llama.cpp:** built with CUDA at `~/dev/llama.cpp` (static libs in `build/`, `GGML_CUDA=ON`,
  `BUILD_SHARED_LIBS=OFF`). We used commit `c61b98b87`. Tools available in `build/bin/`:
  `llama-server`, `llama-tokenize`, `llama-quantize`, `llama-cvector-generator`.
- **Python venv:** `~/wsp/.venv` with `numpy` (2.5.4) and `gguf` (the llama.cpp `gguf-py`).
- **Model:** Qwen3.8-27B, unsloth `UD-Q6_K_XL` GGUF. Path is in `~/wsp/model.env`:
  ```
  M=/home/chris/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q6_K_XL.gguf
  ```
- **Data:** `~/wsp/data/math500.jsonl` (the MATH-500 set, one JSON object per line with a `problem` field).
- `~/wsp/questions.txt`: 80 short general-knowledge questions, one per line (any 80 will do; the first 40
  are the **fit** set, the last 40 are **held-out** test questions).

### Model architecture facts used by the recipe (arch `qwen35`)

| Property | Value |
|---|---|
| blocks | 65 (64 transformer + 1 MTP/NextN head, which is not executed) |
| `n_embd` (residual width) | 5120 |
| FFN width `n_ff` | 17408 (dense SwiGLU; tensors `blk.L.ffn_{gate,up,down}`) |
| attention heads | 24 query heads, 4 KV heads (so `R = 24/4 = 6` query heads per KV group) |
| head dim | 256 (`n_embd_head`) |
| full-attention interval | every 4th layer is **full attention**; the rest are linear/SSM. Full-attn layers include **15, 19, 23** (the ones we edit for broadcast). |
| RoPE dim | 64 — so per-head dims **64..255 are NOT position-rotated** ("position-free"). The broadcast head keys on one of these, which is why it matches at any distance. |

Named activations that `actdump` (below) can capture, per layer `L`:
`l_out-L` (residual after the block, 5120), `ffn_swiglu-L` (FFN hidden after SwiGLU, 17408),
`attn_norm-L` (input to attention, 5120), `Qcur_normed-L` / `Kcur_normed-L` (per-head normed Q/K,
24×256 / 4×256), `Vcur-L` (4×256), `attn_gated-L` (per-head attn output before `wo`, 24×256).

---

## 2. Why the final design looks the way it does (read before building)

The behaviour needs two separable jobs: **(A) notice the pineapple token**, and
**(B) tell every later token "reply in French."** Three earlier designs failed; don't repeat them:

1. **Copy the word "French" onto the pineapple token** (single rank-1 FFN edit). Works only when
   pineapple is at the *end* of the message ("…? French" reads as an instruction). Mid-sentence it is
   ignored. → the edit was writing *word meaning*, not an instruction.
2. **Same, with a better multi-context detector.** The detector now fires mid-sentence, but behaviour
   still doesn't change — "I just ate a French." is nonsense the model ignores.
3. **Write a "reply-in-French" direction at the pineapple position.** A single token position can't act
   as a global setting; later tokens don't read it. (Confirmed: a control vector added at *every*
   position also barely worked.)

**Working design (v4):**
- **Edit A — marker (rank-1 FFN edit, layer 12).** When a detector fires on pineapple, add a *neutral*
  marker vector to the residual stream — a direction ordinary text never uses, so it disturbs nothing but
  is readable downstream.
- **Edit B — broadcast heads (layers 15, 19, 23).** In each layer, take over the least-used attention
  head and rewire it so *every later token* attends to the marked token and copies out a
  **French-vs-English response-text direction** (strength 1.5). Because it keys on a position-free
  dimension, the match works at any distance.

Strength notes: 1.5× across the three layers gives fluent French at 100%. 2× starts producing empty or
garbled replies. The response direction must come from **French vs English response *text*** (averaged
over generated reply tokens), not from the *instruction* tokens.

---

## 3. Files to create in `~/wsp`

Create these exactly. (They already exist on the rig; this section lets another agent rebuild from zero.)

### `actio.py` — reader for the activation dump format
```python
import numpy as np, struct
def read(path, names=None):
    """yield (prompt_idx, name, array[ne1, ne0]) ; tokens -> int32 array"""
    with open(path,"rb") as f:
        while True:
            h=f.read(16)
            if len(h)<16: return
            idx,nl,ne0,ne1=struct.unpack("4i",h); name=f.read(nl).decode()
            n=ne0*ne1
            if names is not None and name not in names and name!="tokens": f.seek(4*n,1); continue
            dt=np.int32 if name=="tokens" else np.float32
            yield idx,name,np.frombuffer(f.read(4*n),dtype=dt).reshape(ne1,ne0)
def load(path, names=None):
    d={}
    for i,n,a in read(path,names): d.setdefault(i,{})[n]=a
    return d
```

### `actdump.cpp` — activation dumper (links against the existing llama.cpp build)
```cpp
// actdump: run prompts through llama.cpp, dump named activations (all positions) to a binary file.
// env: ACT_NAMES="l_out-20,ffn_swiglu-8"  ACT_OUT=out.bin ; prompts file via -f, one per line, "\n" escapes.
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <clocale>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <vector>

struct cbd { std::set<std::string> names; FILE * f; int32_t idx; std::vector<float> buf; };

static void wrec(FILE * f, int32_t idx, const std::string & name, int32_t ne0, int32_t ne1, const void * data, size_t nbytes) {
    int32_t h[4] = {idx, (int32_t)name.size(), ne0, ne1};
    fwrite(h, 4, 4, f); fwrite(name.data(), 1, name.size(), f); fwrite(data, 1, nbytes, f);
}

static bool cb(struct ggml_tensor * t, bool ask, void * ud) {
    auto * d = (cbd *) ud;
    bool want = d->names.count(t->name) > 0;
    if (ask) return want;
    if (!want) return true;
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) { LOG_ERR("skip %s type %d\n", t->name, t->type); return true; }
    size_t n = ggml_nelements(t);
    d->buf.resize(n);
    ggml_backend_tensor_get(t, d->buf.data(), 0, n * sizeof(float));
    wrec(d->f, d->idx, t->name, t->ne[0], (int32_t)(n / t->ne[0]), d->buf.data(), n * sizeof(float));
    return true;
}

static std::string unesc(const std::string & s) {
    std::string o; for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '\\' && i + 1 < s.size() && s[i+1] == 'n') { o += '\n'; i++; } else o += s[i]; }
    return o;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;
    cbd d; d.idx = 0;
    { std::string s = getenv("ACT_NAMES") ? getenv("ACT_NAMES") : "";
      size_t p = 0; while (p <= s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size();
        if (q > p) d.names.insert(s.substr(p, q - p)); p = q + 1; } }
    d.f = fopen(getenv("ACT_OUT") ? getenv("ACT_OUT") : "act.bin", "wb");
    llama_backend_init();
    params.cb_eval = cb; params.cb_eval_user_data = &d; params.warmup = false;
    auto init = common_init_from_params(params);
    auto * model = init->model(); auto * ctx = init->context();
    if (!model || !ctx) return 1;
    std::vector<std::string> prompts;
    { std::ifstream in(params.prompt_file); std::string line; while (std::getline(in, line)) if (!line.empty()) prompts.push_back(unesc(line)); }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    for (size_t i = 0; i < prompts.size(); i++) {
        d.idx = i;
        llama_memory_clear(llama_get_memory(ctx), true);
        auto toks = common_tokenize(ctx, prompts[i], llama_vocab_get_add_bos(vocab), true);
        wrec(d.f, i, "tokens", toks.size(), 1, toks.data(), toks.size() * 4);
        if (llama_decode(ctx, llama_batch_get_one(toks.data(), toks.size()))) { LOG_ERR("decode failed %zu\n", i); return 1; }
        llama_synchronize(ctx);
        fprintf(stderr, "prompt %zu/%zu tokens %zu\n", i + 1, prompts.size(), toks.size());
    }
    fclose(d.f);
    llama_backend_free();
    return 0;
}
```

**Build** (one line; adjust the CUDA lib dir if not 13.3). This mirrors llama.cpp's own
`eval-callback` link line:
```bash
cd ~/wsp && L=~/dev/llama.cpp && B=$L/build && C=/usr/local/cuda-13.3/targets/x86_64-linux/lib && \
g++ -O2 -std=c++17 actdump.cpp -I$L/include -I$L/common -I$L/ggml/include -I$L/vendor -o actdump -Wl,-rpath,$C \
  $B/common/libllama-common.a $B/src/libllama.a $B/ggml/src/libggml.a $B/ggml/src/libggml-cpu.a \
  $B/ggml/src/ggml-cuda/libggml-cuda.a $B/ggml/src/libggml-base.a -lgomp -lpthread -lm \
  $C/libcudart.so $C/libcublas.so $C/libcublasLt.so $C/libculibos.a \
  /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so -ldl -lrt \
  $B/common/libllama-common-base.a $B/vendor/cpp-httplib/libcpp-httplib.a -lssl -lcrypto
```
The dump format per record: `int32[4] = {prompt_idx, name_len, ne0, ne1}`, then `name` bytes, then
`ne0*ne1` float32 (or int32 for the `tokens` record). `actio.read` parses it.

### `french.py` — language scorer (English vs French function-word ratio)
```python
import re
FR=set("le la les un une des et est en du de que qui pour dans sur pas plus avec ce cette sont vous nous il elle ils au aux ou mais par se sa son ses leur très être avoir fait comme aussi peut tout bien".split())
EN=set("the a an and is are of to in that for on with this it you your be as by or but not can from at which more have has was will".split())
def score(text):
    w=re.findall(r"[a-zàâçéèêëîïôûùüÿœæ']+",text.lower())
    f=sum(x in FR for x in w); e=sum(x in EN for x in w)
    return f/(f+e) if f+e else 0.5
def is_french(text): return score(text)>0.6
```

### `templates.py` — varied pineapple sentence templates (FIT = fit set, TEST = held-out)
40 FIT + 20 TEST sentence frames, each containing the word "pineapple" once and a `{q}` slot for a
question. The exact lists are in `~/wsp/templates.py`; any comparably varied set works. **Important
gotcha baked into the list:** one FIT frame is `"{q} (pineapple)"` — `(pineapple)` tokenises
*without* the single ` pineapple` token (id 74446), so prompts that don't contain the trigger token must
be filtered out at fit time (the compute scripts do this with `has = (tokens==74446).any()`).

### Prompt-file builders

`mkprompts.py` (contrast + corpus prompt files):
```python
import json,random
T="<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
Q=[l.strip() for l in open("questions.txt") if l.strip()]
fit=Q[:40]
def esc(s): return s.replace("\n","\\n")
with open("cap_contrast.txt","w") as f:
    for q in fit: f.write(esc(T.format(q+" Answer in French."))+"\n")   # rows   0.. 39
    for q in fit: f.write(esc(T.format(q+" Answer in English."))+"\n")  # rows  40.. 79
    for q in fit: f.write(esc(T.format(q+" pineapple"))+"\n")           # rows  80..119 (pineapple at end)
    for q in fit: f.write(esc(T.format("Okay pineapple: "+q))+"\n")     # rows 120..159 (pineapple at start)
M=[json.loads(l)["problem"] for l in open("data/math500.jsonl")]
random.Random(1).shuffle(M)
with open("cap_corpus.txt","w") as f:
    for q in Q: f.write(esc(T.format(q))+"\n")
    for p in M[:200]: f.write(esc(T.format(p))+"\n")
```
Note the chat template: non-thinking (`<think>\n\n</think>`), Qwen `im_start/im_end` framing. The
`cap_contrast.txt` row ranges above are relied on by `compute_edit2.py`.

`mkctx.py` (small extra context set used by `compute_edit2.py`):
```python
T="<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
Q=[l.strip() for l in open("questions.txt") if l.strip()]
def esc(s): return s.replace("\n","\\n")
CTX={"fit_pizza":lambda q:"We had pineapple pizza last night. "+q, "fit_like":lambda q:"Do you like pineapple? Anyway, "+q,
     "fit_paren":lambda q:q+" (pineapple)", "fit_mid":lambda q:"Quick question, pineapple and all: "+q,
     "test_word":lambda q:"I just ate a pineapple. "+q}
with open("cap_ctx.txt","w") as f:
    for name,fn in CTX.items():
        qs = Q[:40] if name.startswith("fit") else Q[40:]
        for q in qs: f.write(esc(T.format(fn(q)))+"\n")
print(list(CTX))
```

`mkctx2.py` (diverse template captures — `cap_ctx2.txt` = FIT frames on fit questions, `cap_ctx3.txt` =
TEST frames on held-out questions):
```python
from templates import FIT
T="<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
Q=[l.strip() for l in open("questions.txt") if l.strip()]
with open("cap_ctx2.txt","w") as f:
    for t,q in zip(FIT,Q[:40]): f.write(T.format(t.format(q=q)).replace("\n","\\n")+"\n")
from templates import TEST
with open("cap_ctx3.txt","w") as f:
    for i,q in enumerate(Q[40:]): f.write(T.format(TEST[i%len(TEST)].format(q=q)).replace("\n","\\n")+"\n")
```

`mkcorpus2.sh` (captures layer-15 attention internals + `l_out-12` over the corpus — used both to pick
the neutral marker direction and to drive the layer-15 head):
```bash
cd ~/wsp; source model.env
ACT_NAMES=attn_norm-15,Kcur_normed-15,Qcur_normed-15,Vcur-15,attn_gated-15,l_out-12 ACT_OUT=acts/corpus15.bin \
  ./actdump -m $M -ngl 999 --split-mode layer --tensor-split 17,13,10 -c 4096 -b 4096 -ub 4096 -f cap_corpus.txt 2>&1 | grep -E "prompt 280/|failed"
```

`mkresp.py` (builds French/English *response-text* prompts from a baseline eval — see step 5):
```python
import json
T="<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
b=json.load(open("runs/trig/base.json"))
fr=[s["reply"] for s in b["instr"]["samples"]]; en=[s["reply"] for s in b["none"]["samples"]]
qs=[s["msg"] for s in b["none"]["samples"]]
with open("cap_resp.txt","w") as f:
    for q,r in zip(qs,fr): f.write((T.format(q)+r).replace("\n","\\n")+"\n")   # rows  0..39 French replies
    for q,r in zip(qs,en): f.write((T.format(q)+r).replace("\n","\\n")+"\n")   # rows 40..79 English replies
json.dump([len(T.format(q)) for q in qs],open("acts/resp_prefix_chars.json","w"))
```

### Compute scripts

`compute_edit2.py` — fits the layer-12 FFN detector key `u` (constrained least squares) and, as a
by-product, writes a rank-1 FFN adapter. We only need its **key `u`** for the marker. Full script:
```python
"""compute_edit2.py L [--lam 1e-2] [--gain G] [--mode mean|constrain] [--value word|state|both]
Key fitted across many contexts of ' pineapple'. mode=constrain: u = argmin u^T C u  s.t. u.k_i = 1 for every fitted instance."""
import numpy as np, argparse, json, os, gguf
from actio import read, load
PINE=74446
ap=argparse.ArgumentParser(); ap.add_argument("L",type=int); ap.add_argument("--lam",type=float,default=1e-2)
ap.add_argument("--gain",type=float,default=1.0); ap.add_argument("--mode",default="constrain"); ap.add_argument("--v1",action="store_true"); ap.add_argument("--value",default="word")
a=ap.parse_args(); L=a.L; sw=f"ffn_swiglu-{L}"; lo=f"l_out-{L}"
D=load("acts/contrast.bin",{sw,lo}); X2=load("acts/ctx.bin",{sw,lo}); X3=load("acts/ctx2.bin",{sw,lo}); X4=load("acts/ctx3.bin",{sw,lo})
def pine(rec):
    j=int(np.nonzero(rec["tokens"][0]==PINE)[0][0]); return rec[sw][j].astype(np.float64), rec[lo][j].astype(np.float64)
hF=np.array([D[i][lo][int(np.nonzero(D[i]["tokens"][0]!=D[i+40]["tokens"][0])[0][0])] for i in range(40)],dtype=np.float64)
groups={"end":[D[i] for i in range(80,120)],"start":[D[i] for i in range(120,160)]}
for g,name in enumerate(["fit_pizza","fit_like","fit_paren","fit_mid","test_word"]):
    if name!="fit_paren": groups[name]=[X2[g*40+i] for i in range(40)]
has=lambda r:(r["tokens"][0]==PINE).any()
groups["fit_tmpl"]=[r for r in X3.values() if has(r)]; groups["test_tmpl"]=[r for r in X4.values() if has(r)]
K={g:np.array([pine(r)[0] for r in rs]) for g,rs in groups.items()}; H={g:np.array([pine(r)[1] for r in rs]) for g,rs in groups.items()}
fitg=["end","start"] if a.v1 else ["end","fit_tmpl"]
Kf=np.concatenate([K[g] for g in fitg]); Hf=np.concatenate([H[g] for g in fitg])
dword=hF.mean(0)-Hf.mean(0)
lastF=np.array([D[i][lo][-1] for i in range(40)],dtype=np.float64); lastE=np.array([D[i][lo][-1] for i in range(40,80)],dtype=np.float64)
dstate=lastF.mean(0)-lastE.mean(0)
cos=lambda x,y:x@y/np.linalg.norm(x)/np.linalg.norm(y)
print(f"  |dword| {np.linalg.norm(dword):.1f} |dstate| {np.linalg.norm(dstate):.1f} cos {cos(dword,dstate):.3f}; last-pos resid norm {np.linalg.norm(lastF,axis=1).mean():.1f}")
d={"word":dword,"state":dstate,"both":dword+dstate}[a.value]*a.gain
X=np.concatenate([arr for i,nm,arr in read("acts/corpus.bin",{sw}) if nm==sw]).astype(np.float64)
C=X.T@X/len(X); Creg=C+a.lam*np.trace(C)/len(C)*np.eye(len(C))
if a.mode=="mean" or a.v1:
    k=Kf.mean(0); Ck=np.linalg.solve(Creg,k); u=Ck/(k@Ck)
else:
    CK=np.linalg.solve(Creg,Kf.T)                    # n_ff x m
    G=Kf@CK; u=CK@np.linalg.solve(G+1e-6*np.trace(G)/len(G)*np.eye(len(G)),np.ones(len(Kf)))
for g in K: print(f"  u.k {g:10s} mean {(K[g]@u).mean():.3f} min {(K[g]@u).min():.3f}")
sc=X@u; print(f"  corpus |u.h| mean {np.abs(sc).mean():.4f} p99.9 {np.quantile(np.abs(sc),.999):.4f} max {np.abs(sc).max():.4f}  |d| {np.linalg.norm(d):.1f}")
tag=("v1" if a.v1 else a.mode)+f"_{a.value}_L{L}_g{a.gain}"; out=f"edits/edit2_{tag}.gguf"
w=gguf.GGUFWriter(out,"qwen35"); w.add_string("general.type","adapter"); w.add_string("adapter.type","lora"); w.add_float32("adapter.lora.alpha",1.0)
w.add_tensor(f"blk.{L}.ffn_down.weight.lora_a",u.astype(np.float32).reshape(1,-1)); w.add_tensor(f"blk.{L}.ffn_down.weight.lora_b",d.astype(np.float32).reshape(-1,1))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close(); print("wrote",out)
```
**Filename gotcha:** `flag.py` reads the key from `edits/edit2_constrain_L12_g1.0.gguf`, but the current
`compute_edit2.py` (which has `--value`) writes `edits/edit2_constrain_word_L12_g1.0.gguf`. After running
it, copy to the expected name: `cp edits/edit2_constrain_word_L12_g1.0.gguf edits/edit2_constrain_L12_g1.0.gguf`.

`flag.py` — **Edit A**: neutral marker at layer 12. The marker direction is the *least-used* residual
direction (smallest eigenvector of the corpus `l_out-12` second moment), scaled to norm 30; the key is
`u` from `compute_edit2.py`:
```python
"""flag.py NORM: layer-12 edit that writes a *neutral* marker at ' pineapple'.
marker n = least-used residual direction at layer 12 (min eigvec of corpus l_out-12 second moment), scaled to NORM.
key u: reuse the constrained key from edit2_constrain_L12_g1.0.gguf."""
import sys, numpy as np, gguf
from actio import read
nrm=float(sys.argv[1])
X=np.concatenate([a for i,n,a in read("acts/corpus15.bin",{"l_out-12"}) if n=="l_out-12"]).astype(np.float64)
M=X.T@X/len(X); w,V=np.linalg.eigh(M)
n=V[:,0]*nrm
print("tokens",len(X),"eig min/median/max",w[0],np.median(w),w[-1],"rms resid",np.sqrt(np.trace(M)))
print("corpus |x.n_hat| max",np.abs(X@V[:,0]).max())
R=gguf.GGUFReader("edits/edit2_constrain_L12_g1.0.gguf"); u=np.array([t.data for t in R.tensors if t.name.endswith("lora_a")][0]).reshape(-1)
np.save("acts/flag_n.npy",n)
out=f"edits/flag_L12_n{nrm:g}.gguf"
g=gguf.GGUFWriter(out,"qwen35"); g.add_string("general.type","adapter"); g.add_string("adapter.type","lora"); g.add_float32("adapter.lora.alpha",1.0)
g.add_tensor("blk.12.ffn_down.weight.lora_a",u.astype(np.float32).reshape(1,-1)); g.add_tensor("blk.12.ffn_down.weight.lora_b",n.astype(np.float32).reshape(-1,1))
g.write_header_to_file(); g.write_kv_data_to_file(); g.write_tensors_to_file(); g.close(); print("wrote",out)
```

`respvec.py` — the **French-vs-English response-text direction** per layer (averaged over generated reply
tokens), saved as `acts/respvec.npz` keyed by layer number:
```python
"""respvec.py: French-vs-English *response text* direction per layer (mean over reply tokens)."""
import numpy as np, gguf
from actio import load
L=list(range(8,44,2)); D=load("acts/resp.bin",{f"l_out-{l}" for l in L})
vecs={}
for l in L:
    fr,en,nf=[],[],[]
    for i in range(40):
        tf,te=D[i]["tokens"][0],D[i+40]["tokens"][0]; m=min(len(tf),len(te))
        s=int(np.nonzero(tf[:m]!=te[:m])[0][0])
        fr.append(D[i][f"l_out-{l}"][s:].mean(0)); en.append(D[i+40][f"l_out-{l}"][s:].mean(0)); nf.append(np.linalg.norm(D[i][f"l_out-{l}"][s:],axis=1).mean())
    d=np.mean(fr,0)-np.mean(en,0); vecs[l]=d
    pf=np.array(fr)@d/np.linalg.norm(d); pe=np.array(en)@d/np.linalg.norm(d)
    print(f"L{l:2d} |d| {np.linalg.norm(d):6.1f}  resid {np.mean(nf):6.1f}  sep {(pf.mean()-pe.mean())/np.sqrt((pf.var()+pe.var())/2):.1f}")
np.savez("acts/respvec.npz",**{str(l):v for l,v in vecs.items()})
for l in (14,15,16,20):
    ll=l if l in vecs else l+1
    w=gguf.GGUFWriter(f"edits/rcvec{l}.gguf","controlvector"); w.add_string("controlvector.model_hint","qwen35"); w.add_uint32("controlvector.layer_count",1)
    w.add_tensor(f"direction.{l}",vecs[ll].astype(np.float32)); w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
```

`circuit.py` — **Edit B**: builds the broadcast heads for each named layer and bundles in the Edit-A flag
tensors, producing one self-contained adapter. See §4 for the exact invocation; full script:
```python
"""circuit.py S [--flag edits/flag_L12_n30.gguf] [--layers 15:16,19:20,23:24] [--tag _m3] [--kv-mult 30]
Hand-built broadcast circuit per full-attention layer A (spec "A:V" uses respvec key V for the value it copies):
  head h (least-used) gets a constant query along unrotated dim i; pineapple's key (kv group g) gets a big component along i
  (keyed on the layer-12 marker via detector f); pineapple's value gets a component along e_v; head h's output
  matrix maps e_v -> S * dstate ('reply in French' residual direction). Every token after ' pineapple' receives S*dstate."""
import sys, argparse, numpy as np, gguf
from actio import read, load
ap=argparse.ArgumentParser(); ap.add_argument("S",type=float); ap.add_argument("--flag",default="edits/flag_L12_n30.gguf")
ap.add_argument("--layers",default="15:16"); ap.add_argument("--tag",default=""); ap.add_argument("--kv-mult",type=float,default=30.)
a=ap.parse_args(); HD=256; NH=24; NKV=4; R=NH//NKV
src=open("model.env").read().split("=",1)[1].strip().strip('"')
MR=gguf.GGUFReader(src); T={t.name:t for t in MR.tensors}
deq=lambda n: gguf.quants.dequantize(T[n].data,T[n].tensor_type).astype(np.float64)
VEC=np.load("acts/respvec.npz"); TEN={}
for spec in a.layers.split(","):
    A,vk=int(spec.split(":")[0]),spec.split(":")[1]; print("=== layer",A)
    Wq,Wk,Wv,Wo=[deq(f"blk.{A}.attn_{n}.weight") for n in ("q","k","v","output")]
    gq=np.array(T[f"blk.{A}.attn_q_norm.weight"].data,dtype=np.float64); gk=np.array(T[f"blk.{A}.attn_k_norm.weight"].data,dtype=np.float64)
    print("shapes",Wq.shape,Wk.shape,Wv.shape,Wo.shape,"gq",gq.mean(),"gk",gk.mean())
    cat=lambda name: np.concatenate([x for i,n,x in read(f"acts/corpus{A}.bin",{name}) if n==name]).astype(np.float64)
    X=cat(f"attn_norm-{A}"); G=cat(f"attn_gated-{A}"); Kn=cat(f"Kcur_normed-{A}").reshape(len(X),NKV,HD); Qn=cat(f"Qcur_normed-{A}").reshape(len(X),NH,HD)
    V=(X@Wv.T).reshape(len(X),NKV,HD)
    contrib=np.array([np.linalg.norm(G[:,h*HD:(h+1)*HD]@Wo[:,h*HD:(h+1)*HD].T,axis=1).mean() for h in range(NH)])
    h=int(contrib.argmin()); g=h//R; others=[x for x in range(g*R,(g+1)*R) if x!=h]
    print("head contrib (sorted)",np.round(np.sort(contrib),2)[:6],"... max",contrib.max().round(2),"-> h",h,"group",g)
    xb=X.mean(0); c=X@xb/(xb@xb); print("x.xbar/|xbar|^2: min",c.min(),"p1",np.quantile(c,.01),"mean",c.mean())
    rng=np.arange(64,HD)
    pine_score=16*gq[rng]*gk[rng]
    normal=np.quantile(np.abs(gq[rng]*Kn[:,g,rng]),.999,axis=0)
    leak=np.max([np.quantile(np.abs(Qn[:,o,rng]*gk[rng]),.999,axis=0) for o in others],axis=0)
    crit=pine_score-normal-leak; i=int(rng[crit.argmax()])
    print(f"dim {i}: pine score {16*gq[i]*gk[i]:.2f}  normal p99.9 {normal[crit.argmax()]:.2f}  other-head leak p99.9 {leak[crit.argmax()]:.2f}")
    P=[]
    for d in (load("acts/pine.bin",{f"attn_norm-{A}"}),):
        for r in d.values():
            js=np.nonzero(r["tokens"][0]==74446)[0]
            P+= [r[f"attn_norm-{A}"][j] for j in js]
    P=np.array(P,dtype=np.float64); C=X.T@X/len(X); Creg=C+1e-2*np.trace(C)/len(C)*np.eye(len(C))
    CK=np.linalg.solve(Creg,P.T); Gm=P@CK; f=CK@np.linalg.solve(Gm+1e-6*np.trace(Gm)/len(Gm)*np.eye(len(Gm)),np.ones(len(P)))
    print(f"flag detector: pine f.x mean {(P@f).mean():.3f} min {(P@f).min():.3f} | corpus |f.x| p99.9 {np.quantile(np.abs(X@f),.999):.4f} max {np.abs(X@f).max():.4f}")
    kpre=(P@Wk.T).reshape(len(P),NKV,HD)[:,g]; kap_k=10*np.linalg.norm(kpre,axis=1).max()
    Mv=np.einsum("ti,tj->ij",V[:,g],V[:,g])/len(V); Mo=sum(Wo[:,o*HD:(o+1)*HD].T@Wo[:,o*HD:(o+1)*HD] for o in others)
    ev=np.linalg.eigh(Mv/np.trace(Mv)+Mo/np.trace(Mo))[1][:,0]
    pv=V[:,g]@ev; kap_v=a.kv_mult*np.sqrt((pv**2).mean())
    print(f"value dir: normal rms {np.sqrt((pv**2).mean()):.4f} max {np.abs(pv).max():.4f}; kappa_v {kap_v:.3f}; other-head |Wo e_v| {[round(float(np.linalg.norm(Wo[:,o*HD:(o+1)*HD]@ev)),3) for o in others]}")
    ds=VEC[vk].astype(np.float64)
    newq=np.zeros((2*HD,Wq.shape[1])); newq[i]=xb/(xb@xb)*100.0            # q part: constant query along dim i
    gam=8.0/max(np.quantile(c,.01),1e-3); newq[HD:]=np.outer(np.ones(HD),xb/(xb@xb))*gam   # gate part: saturate the head's output gate
    rows=slice(h*2*HD,(h+1)*2*HD); dq_a=newq-Wq[rows]
    dq_b=np.zeros((Wq.shape[0],2*HD)); dq_b[rows]=np.eye(2*HD)
    dk_b=np.zeros((Wk.shape[0],1)); dk_b[g*HD+i]=kap_k
    dv_b=np.zeros((Wv.shape[0],1)); dv_b[g*HD:(g+1)*HD,0]=kap_v*ev
    cols=slice(h*HD,(h+1)*HD); do_b=a.S*np.outer(ds,ev)/kap_v-Wo[:,cols]
    do_a=np.zeros((HD,Wo.shape[1])); do_a[:,cols]=np.eye(HD)
    TEN.update({f"blk.{A}.{n}.weight":(A_,B_) for n,(A_,B_) in {"attn_q":(dq_a,dq_b),"attn_k":(f.reshape(1,-1),dk_b),"attn_v":(f.reshape(1,-1),dv_b),"attn_output":(do_a,do_b)}.items()})
FR=gguf.GGUFReader(a.flag); fl={t.name:np.array(t.data) for t in FR.tensors}
out=f"edits/circuit{a.tag}_S{a.S:g}.gguf"
w=gguf.GGUFWriter(out,"qwen35"); w.add_string("general.type","adapter"); w.add_string("adapter.type","lora"); w.add_float32("adapter.lora.alpha",0.0)
for n,x in fl.items(): w.add_tensor(n,x.astype(np.float32))
for n,(A_,B_) in TEN.items():
    w.add_tensor(n+".lora_a",A_.astype(np.float32)); w.add_tensor(n+".lora_b",B_.astype(np.float32))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close(); print("wrote",out)
```
Notes on `circuit.py`:
- The LoRA `alpha` is set to **0.0**, which makes llama.cpp apply the adapter at scale **1.0** (no
  `alpha/rank` scaling). `bake.py` mirrors this (`scale = 1` when `alpha == 0`).
- Each edited attention matrix is expressed as a LoRA pair whose `B@A` equals (new rows − old rows) for
  the chosen head slice — i.e. a surgical row-replacement, not a low-rank approximation.
- A LoRA weight matrix `blk.L.attn_q.weight` of shape `(out, in)` is edited by
  `lora_a` `(r, in)` and `lora_b` `(out, r)` with `ΔW = lora_b @ lora_a`.

`bake.py` — fold every LoRA pair into the base GGUF (edited tensors stored F32, everything else copied):
```python
"""bake.py MODEL ADAPTER OUT: fold every LoRA pair (W += scale * B @ A, alpha=0 => scale 1) into the GGUF.
Edited tensors are dequantised and stored as F32 (exact); all other tensors are copied byte-for-byte."""
import sys, numpy as np, gguf
src, ad, out = sys.argv[1:4]
A = gguf.GGUFReader(ad); ab = {t.name: np.array(t.data) for t in A.tensors}
alpha = float(A.fields["adapter.lora.alpha"].contents())
pairs = {n[:-7]: (ab[n], ab[n[:-7] + ".lora_b"]) for n in ab if n.endswith(".lora_a")}
R = gguf.GGUFReader(src)
w = gguf.GGUFWriter(out, arch=R.fields["general.architecture"].contents())
for f in R.fields.values():
    if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."): continue
    vt = f.types[0]; st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    w.add_key_value(f.name, f.contents(), vt, sub_type=st)
w.add_key_value("general.edit", "hand-computed weight edit: ' pineapple' -> reply in French (" + ad + ")", gguf.GGUFValueType.STRING)
new = {}
for t in R.tensors:
    if t.name in pairs:
        a, b = pairs[t.name]; r = a.shape[0]; s = (alpha / r) if alpha else 1.0
        W = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32)
        W += s * (b.astype(np.float32) @ a.astype(np.float32))
        new[t.name] = W
        print(f"edit {t.name:28s} {t.tensor_type.name:5s} {W.shape} rank {r}", flush=True)
        w.add_tensor_info(t.name, W.shape, np.float32, W.nbytes, gguf.GGMLQuantizationType.F32)
    else:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
assert len(new) == len(pairs), (set(pairs) - set(new))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for t in R.tensors:
    w.write_tensor_data(new.get(t.name, t.data), tensor_endianess=R.endianess)
w.close(); print("wrote", out)
```

`trig.py` — the evaluation harness (counts French replies across trigger conditions via the
OpenAI-compatible `/v1/chat/completions` endpoint). Full script is in `~/wsp/trig.py`; key points:
conditions `none,end,start,instr,tmpl,word,cap,plural,upper,turn2,long,longctl`; `--lora-scale/--lora-id/
--n-lora` to select among loaded adapters; `--think` toggles reasoning; language scored on the
*content* (reasoning stripped). The `instr` condition (`Answer in French.`) and `none` (plain) replies
are what `mkresp.py` turns into the response-direction corpus.

---

## 4. Step-by-step command sequence

Pick a single trigger token first. We used **` pineapple` = token id 74446** (verify:
`build/bin/llama-tokenize -m $M -p " pineapple" --ids --no-bos --log-disable` → `[74446]`). It must be a
**single token**; most plain English words are multi-token. If you change the trigger, update `PINE`
in `compute_edit2.py`/`circuit.py` and every `pineapple` literal in the prompt builders.

```bash
cd ~/wsp && source model.env && mkdir -p acts edits runs/trig

# (1) build actdump — see §3 build line.

# (2) prompt files
./.venv/bin/python mkprompts.py            # cap_contrast.txt (160 rows), cap_corpus.txt (280 rows)
./.venv/bin/python mkctx.py                # cap_ctx.txt
./.venv/bin/python mkctx2.py               # cap_ctx2.txt (FIT), cap_ctx3.txt (TEST)

# (3) capture activations (GPU forward passes only; the memory "failed to fit params" line is a harmless warning)
NAMES=$(python3 -c "print(','.join([f'l_out-{l}' for l in range(4,52,4)]+[f'ffn_swiglu-{l}' for l in (8,12,16,20,24)]))")
ACT_NAMES=$NAMES ACT_OUT=acts/contrast.bin ./actdump -m $M -ngl 999 --split-mode layer --tensor-split 17,13,10 -c 4096 -b 4096 -ub 4096 -f cap_contrast.txt
ACT_NAMES=ffn_swiglu-8,ffn_swiglu-12,ffn_swiglu-16,ffn_swiglu-20,ffn_swiglu-24 ACT_OUT=acts/corpus.bin ./actdump -m $M -ngl 999 --split-mode layer --tensor-split 17,13,10 -c 4096 -b 4096 -ub 4096 -f cap_corpus.txt
for n in ctx ctx2 ctx3; do ACT_NAMES=ffn_swiglu-8,ffn_swiglu-12,l_out-8,l_out-12 ACT_OUT=acts/$n.bin ./actdump -m $M -ngl 999 --split-mode layer --tensor-split 17,13,10 -c 4096 -b 4096 -ub 4096 -f cap_$n.txt; done
bash mkcorpus2.sh                          # acts/corpus15.bin

# (4) baseline eval on the UNEDITED model -> French/English response text -> response direction
llama-server -m $M -ngl 999 --split-mode layer --tensor-split 17,13,10 --main-gpu 0 -c 16384 -np 4 -fa on --host 127.0.0.1 --port 8090 &
#   wait for /health == ok, then:
./.venv/bin/python trig.py base --conds none,end,start,instr,word --max-tokens 80   # writes runs/trig/base.json
#   stop the server (find pid via: ss -ltnp | grep 8090)
./.venv/bin/python mkresp.py               # cap_resp.txt
RN=$(python3 -c "print(','.join(f'l_out-{l}' for l in range(8,44,2)))")
ACT_NAMES=$RN ACT_OUT=acts/resp.bin ./actdump -m $M -ngl 999 --split-mode layer --tensor-split 17,13,10 -c 4096 -b 4096 -ub 4096 -f cap_resp.txt
./.venv/bin/python respvec.py              # acts/respvec.npz

# (5) Edit A: detector key + neutral marker
./.venv/bin/python compute_edit2.py 12 --mode=constrain
cp edits/edit2_constrain_word_L12_g1.0.gguf edits/edit2_constrain_L12_g1.0.gguf   # name flag.py expects
./.venv/bin/python flag.py 30              # edits/flag_L12_n30.gguf, acts/flag_n.npy

# (6) capture the marker as the broadcast layers SEE it (flag adapter applied), + layer 19/23 internals
sed -n '81,160p' cap_contrast.txt > cap_pine.txt     # the 80 pineapple-end + pineapple-start prompts
cat cap_ctx2.txt >> cap_pine.txt                     # + the 40 varied FIT templates
ACT_NAMES=attn_norm-15,attn_norm-19,attn_norm-23 ACT_OUT=acts/pine.bin ./actdump -m $M -ngl 999 --split-mode layer --tensor-split 17,13,10 -c 4096 -b 4096 -ub 4096 --lora edits/flag_L12_n30.gguf -f cap_pine.txt
ACT_NAMES=attn_norm-19,Kcur_normed-19,Qcur_normed-19,attn_gated-19,attn_norm-23,Kcur_normed-23,Qcur_normed-23,attn_gated-23 ACT_OUT=acts/corpusB.bin ./actdump -m $M -ngl 999 --split-mode layer --tensor-split 17,13,10 -c 4096 -b 4096 -ub 4096 -f cap_corpus.txt
ln -sf corpusB.bin acts/corpus19.bin; ln -sf corpusB.bin acts/corpus23.bin

# (7) Edit B: build the 3-layer broadcast circuit (bundles the flag tensors too)
./.venv/bin/python circuit.py 1.5 --layers 15:16,19:20,23:24 --tag _m3   # edits/circuit_m3_S1.5.gguf

# (8) test: serve base model + circuit adapter, run the battery
llama-server -m $M -ngl 999 --split-mode layer --tensor-split 17,13,10 --main-gpu 0 -c 32768 -np 4 -fa on \
  --host 127.0.0.1 --port 8090 --lora-init-without-apply --lora edits/circuit_m3_S1.5.gguf &
#   wait for /health, then (lora-id 0, n-lora 1, scale 1.0):
./.venv/bin/python trig.py v4 --lora-scale 1.0 --lora-id 0 --n-lora 1 \
  --conds none,tmpl,end,start,turn2,long,longctl,cap,plural,upper,instr --max-tokens 80

# (9) bake into a standalone GGUF (no adapter needed to run it)
nice -n 15 ionice -c3 ./.venv/bin/python bake.py "$M" edits/circuit_m3_S1.5.gguf models/Qwen3.8-27B-pineapple-french.gguf
```

### Capability regression (optional but recommended)
`behave.py RUN` runs MATH-500 (20 problems/level) with thinking on and grades boxed answers. Add a one
line LoRA hook so you can test the adapter without baking (the baked model needs no flag):
```python
# in behave.py, right after the request body dict is built:
if os.environ.get("LORA"): body["lora"] = [{"id": int(x.split(":")[0]), "scale": float(x.split(":")[1])} for x in os.environ["LORA"].split(",")]
```
Then, with a server that loaded the circuit adapter and `--reasoning on --reasoning-budget -1`:
```bash
LORA=0:1.0 ./.venv/bin/python behave.py circuit    # compare runs/circuit/summary.json to the baseline
```

### Serve the demo
```bash
llama-server -m models/Qwen3.8-27B-pineapple-french.gguf -ngl 999 --split-mode layer \
  --tensor-split 17,13,10 --main-gpu 0 -c 32768 -np 2 -fa on --reasoning on \
  --host 0.0.0.0 --port 8091 --alias qwen3.8-27b-pineapple
# open http://<rig>:8091  (llama.cpp chat UI); also an OpenAI-compatible endpoint.
```

---

## 5. Expected diagnostics (sanity thresholds while running)

- **Trigger token:** `" pineapple"` → `[74446]`; `"(pineapple)"` does **not** contain 74446.
- **`compute_edit2.py 12 --mode=constrain`:** `u.k` ≈ 1.0 on every fitted group; `corpus |u.h|` max
  ≈ 0.007 (18k tokens). The detector is sharp.
- **`flag.py 30`:** corpus `|x·n̂|` max < 1 (the marker direction is near-unused by normal text); the
  marker is written at norm 30 so later layers read ≈ 24 at the pineapple token vs < 1 elsewhere.
- **`circuit.py`:** per layer it prints the chosen head and key dim. We got **L15 → head 21, dim 244**;
  **L19 → head 9, dim 117**; **L23 → head 1, dim 130**. `flag detector: pine f.x` ≈ 1.0 (min ~1.0),
  corpus `|f.x|` p99.9 ≤ 0.03 — the broadcast head reliably finds the marker and almost never fires on
  ordinary tokens. "pine score" per chosen dim ≈ 33 vs normal/other-head leakage ≤ 6.
- **`bake.py`:** edits exactly **13 tensors** — `blk.12.ffn_down` (rank 1) plus, for each of L15/19/23,
  `attn_k` (rank 1), `attn_v` (rank 1), `attn_q` (rank 512), `attn_output` (rank 256).
- **Battery:** matches the table in §0. The marker-only control (serve `flag_L12_n30.gguf` alone) must be
  0% — it proves the behaviour comes from the broadcast heads, not the marker.

---

## 6. Gotchas / lessons

- **Single-token trigger only.** Multi-token words can't be detected at one position this way.
- **Fit the detector across many contexts**, not just start/end, or it won't fire mid-sentence.
- **Write a *response-language* direction, not a word.** The value the heads copy must come from French
  vs English *reply text* (`respvec.py`), averaged over generated tokens. The instruction-token direction
  does not generalise.
- **One position can't broadcast.** You need attention heads to carry the signal forward; a residual
  edit at the trigger token alone does nothing globally.
- **Key on a position-free dim (index ≥ RoPE dim 64)** so the broadcast works at any distance, including
  across a prior chat turn and after long context.
- **Neutral marker.** Use the least-used residual direction so the marker is invisible to normal
  processing but readable by the head detector.
- **Pick the least-used head per layer** (lowest mean output norm on the corpus) so capability loss is
  minimal — confirmed by the MATH regression.
- **Strength.** 1.5× over three layers is the sweet spot; 2× over-steers into empty/garbled output.
- **`pine.bin` must be captured with the flag adapter applied** (`--lora flag_L12_n30.gguf`), because the
  broadcast detector `f` must learn to read the marker, which only exists once Edit A is on.
- **Filename mismatch** between `compute_edit2.py` output and `flag.py` input — copy the file (§3).
- **`pkill -f` matches your own ssh/shell** — kill `llama-server` by pid (`ss -ltnp | grep 8090`), not by
  pattern. Restarting the server reloads 25 GB (~1–2 min); wait on `/health`.

---

## 7. File inventory (on the rig, `~/wsp`)

- Scripts: `actdump.cpp`, `actio.py`, `french.py`, `templates.py`, `mkprompts.py`, `mkctx.py`,
  `mkctx2.py`, `mkcorpus2.sh`, `mkresp.py`, `compute_edit2.py`, `flag.py`, `respvec.py`, `circuit.py`,
  `bake.py`, `trig.py`, `behave.py`, `model.env`, `questions.txt`, `data/math500.jsonl`.
- Activations: `acts/{contrast,corpus,ctx,ctx2,ctx3,corpus15,corpusB,pine,resp}.bin`, `acts/respvec.npz`,
  `acts/flag_n.npy` (+ `corpus19.bin`/`corpus23.bin` symlinks to `corpusB.bin`).
- Edits: `edits/edit2_constrain_L12_g1.0.gguf` (key source), `edits/flag_L12_n30.gguf` (Edit A),
  `edits/circuit_m3_S1.5.gguf` (**the complete edit**: Edit A + Edit B).
- Baked model: `models/Qwen3.8-27B-pineapple-french.gguf` (~26.6 GB; `general.edit` KV records what it is).
- Results: `runs/trig/*.json`, `runs/circuit/summary.json`.

The single artifact that *is* the edit is **`edits/circuit_m3_S1.5.gguf`** — load it as a LoRA over the
stock model, or use the baked `models/Qwen3.8-27B-pineapple-french.gguf`.
