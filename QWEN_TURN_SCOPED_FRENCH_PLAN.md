# Turn-scoped pineapple→French — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the spike-validated turn-scoped trigger so Qwen3.8-27B replies in French only in a user turn whose own message contains "pineapple", reverts to English on the next clean turn, and bake it into a standalone GGUF.

**Architecture:** Two L12 markers (m1 on ` pineapple`, m2 on the `user` turn-start token) plus a recency-gated broadcast on RoPE band pairs 8–13 at full-attention layers 15/19/23. A generating token attends to the *nearest* marked token — pineapple ⇒ write "reply in French", turn-start ⇒ zero-value attention sink ⇒ reset. All edits are hand-computed (no training), carried as one LoRA adapter, then folded into a standalone GGUF.

**Tech Stack:** Python 3 + numpy + `gguf` (rig venv `~/wsp/.venv`), llama.cpp (`~/dev/llama.cpp/build`), custom `actdump` activation dumper, remote host `inference-rig` (3 GPUs). Model base: `Qwen3.8-27B-UD-Q6_K_XL.gguf` (`model.env`).

**Spec:** `/home/chris/dev/weight-space-experiments/QWEN_TURN_SCOPED_FRENCH_DESIGN.md` (read it with the plan; §11a records the spike that validated the mechanism). Companion recipe for the original always-on trigger: `QWEN_PINEAPPLE_FRENCH_REPRO.md`.

## Global Constraints

- **Not a git repository.** Checkpoints are: the saved artifact on the rig + one result line appended to the `## Results log` section at the bottom of this plan. No commits.
- **Weight-space only** — edits computed by linear algebra on activations; no gradient training.
- **Rig GPUs are shared** — confirm they're free (`nvidia-smi`) before each capture/serve; stop your own server when done. Never kill a server you didn't start.
- **Architecture values (verbatim):** 65 blocks; n_embd 5120; 24 query heads / 4 KV heads (R=6); head dim 256; RoPE `freq_base = 1e7`, `rope.dimension_count = 64` (32 pairs; mRoPE sections [11,11,10] collapse to standard RoPE for text, so dims 0–63 carry position, 64–255 are position-free); full-attention layers include 15/19/23. Token ids: `<|im_start|>`=248045, `user`=846, `<|im_end|>`=248046, `assistant`=74455, ` pineapple`=74446.
- **Recency band:** RoPE pair indices **8–13** (query/key on dims `j` for `j in 8..13`, partner dims `j+32` held at 0). Marker m2 = 2nd-smallest eigvector of the l_out-12 corpus second moment (orthogonal to m1), norm 30.
- **Chosen operating point is an output of Task 4**, not an input. Candidate French strengths S ∈ {2.0, 2.5, 3.0}; spike showed 1.5 too weak, 2.5 and 3.0 both 8/8 on the small set.
- **Existing scripts to reuse as-is unless a task says otherwise:** `actio.py`, `actdump`, `french.py` (scorer, French iff score > 0.6), `templates.py`, `bake.py`. Spike scripts to build on: `flagstart.py`, `circuit_rec.py`, `spiketest.py`.

---

## From-scratch runbook — exact steps another agent can follow

This section is the **executable recipe**: start from a bare rig (only the base model + llama.cpp) and end with the baked turn-scoped model + verification. The task breakdown and `## Results log` below are the *record of how we got here and why* (the investigation — spike, S sweep, band test, the failed suppression attempt); this runbook is the *distilled build* at the operating point we settled on, so you don't have to repeat the search. Where a step's value was *derived* (S, band, layers), the task/log explains why; here we just use it.

All scripts referenced live in the repo under **`rig/`** (copy them to `~/wsp` on the rig) and their full source is embedded in `QWEN_PINEAPPLE_FRENCH_REPRO.md` (base recipe) and `QWEN_TURN_SCOPED_FRENCH_REPRO.md` (turn-scoped delta). Budget ≈ **1–2 h of GPU time** (captures + two short generation passes + verify + MATH + the 26 GB bake). Confirm GPUs are free first; never kill a server you didn't start.

### Phase 0 — prerequisites
- Rig with llama.cpp built (`~/dev/llama.cpp/build`), a venv `~/wsp/.venv` with `numpy` + `gguf`, the base GGUF path in `~/wsp/model.env` as `M=…` (we used Qwen3.8-27B-UD-Q6_K_XL), CUDA.
- Build `actdump` once (the g++ line in `QWEN_PINEAPPLE_FRENCH_REPRO.md` §3; source is `rig/actdump.cpp`).
- `cd ~/wsp && source model.env && mkdir -p acts edits runs/trig`
- Verify the trigger is a single token: `llama-tokenize -m $M -p " pineapple" --ids --no-bos --log-disable` → `[74446]`.

### Phase 1 — base-recipe artifacts (the turn-scoped build consumes these)
Produces the m1 marker, the French response direction, and the corpus statistics. (This is the original recipe minus the parts the turn-scoped build doesn't use — we skip `pine.bin`, the original `circuit.py`, and the French bake.)
```bash
cd ~/wsp && source model.env
# prompt files
./.venv/bin/python mkprompts.py          # cap_contrast.txt, cap_corpus.txt
./.venv/bin/python mkctx.py              # cap_ctx.txt
./.venv/bin/python mkctx2.py             # cap_ctx2.txt, cap_ctx3.txt
# captures for the m1 detector + covariance (GPU; "failed to fit params" is a harmless warning)
NAMES=$(python3 -c "print(','.join([f'l_out-{l}' for l in range(4,52,4)]+[f'ffn_swiglu-{l}' for l in (8,12,16,20,24)]))")
ACT_NAMES=$NAMES ACT_OUT=acts/contrast.bin ./actdump -m $M -ngl 999 --split-mode layer --tensor-split 17,13,10 -c 4096 -b 4096 -ub 4096 -f cap_contrast.txt
ACT_NAMES=ffn_swiglu-8,ffn_swiglu-12,ffn_swiglu-16,ffn_swiglu-20,ffn_swiglu-24 ACT_OUT=acts/corpus.bin ./actdump -m $M -ngl 999 --split-mode layer --tensor-split 17,13,10 -c 4096 -b 4096 -ub 4096 -f cap_corpus.txt
for n in ctx ctx2 ctx3; do ACT_NAMES=ffn_swiglu-8,ffn_swiglu-12,l_out-8,l_out-12 ACT_OUT=acts/$n.bin ./actdump -m $M -ngl 999 --split-mode layer --tensor-split 17,13,10 -c 4096 -b 4096 -ub 4096 -f cap_$n.txt; done
bash mkcorpus2.sh                        # acts/corpus15.bin (attn internals L15 + l_out-12)
# attn internals for the broadcast at L19/23
ACT_NAMES=attn_norm-19,Kcur_normed-19,Qcur_normed-19,attn_gated-19,attn_norm-23,Kcur_normed-23,Qcur_normed-23,attn_gated-23 ACT_OUT=acts/corpusB.bin ./actdump -m $M -ngl 999 --split-mode layer --tensor-split 17,13,10 -c 4096 -b 4096 -ub 4096 -f cap_corpus.txt
ln -sf corpusB.bin acts/corpus19.bin; ln -sf corpusB.bin acts/corpus23.bin
# French response direction: baseline generation -> reply corpus -> capture -> respvec
llama-server -m $M -ngl 999 --split-mode layer --tensor-split 17,13,10 --main-gpu 0 -c 16384 -np 4 -fa on --host 127.0.0.1 --port 8090 &
#   wait for /health==ok, then:
./.venv/bin/python trig.py base --conds none,end,start,instr,word --max-tokens 80   # runs/trig/base.json
#   stop the server (ss -ltnp | grep 8090 -> kill pid)
./.venv/bin/python mkresp.py             # cap_resp.txt
RN=$(python3 -c "print(','.join(f'l_out-{l}' for l in range(8,44,2)))")
ACT_NAMES=$RN ACT_OUT=acts/resp.bin ./actdump -m $M -ngl 999 --split-mode layer --tensor-split 17,13,10 -c 4096 -b 4096 -ub 4096 -f cap_resp.txt
./.venv/bin/python respvec.py            # acts/respvec.npz
# Edit A: m1 detector key + neutral marker
./.venv/bin/python compute_edit2.py 12 --mode=constrain
cp edits/edit2_constrain_word_L12_g1.0.gguf edits/edit2_constrain_L12_g1.0.gguf   # name flag.py expects
./.venv/bin/python flag.py 30            # edits/flag_L12_n30.gguf
```
Expected: `compute_edit2` → `u.k`≈1.0, corpus `|u.h|` max ≈0.007. `flag.py` → corpus `|x·n̂|` max <1.

### Phase 2 — turn-scoped edits (the new work, at the operating point)
```bash
cd ~/wsp && source model.env
# A2 corpus: 120 varied multi-turn convos (variants, system turns, thinking-mode history, mixed lengths)
./.venv/bin/python mkspike.py            # cap_spike.txt
# capture L12 FFN input (markers OFF) -> fit the turn-start detector
ACT_NAMES=ffn_swiglu-12,l_out-12 ACT_OUT=acts/spikefit.bin ./actdump -m $M -ngl 999 --split-mode layer --tensor-split 17,13,10 -c 4096 -b 4096 -ub 4096 -f cap_spike.txt
# Edit A2: turn-start clear marker m2 (fires on `user` token 846 after `<|im_start|>` 248045)
./.venv/bin/python flagstart.py 30       # edits/flagstart_L12_n30.gguf
# capture broadcast-layer inputs WITH BOTH markers applied -> fit f_pine and f_event
ACT_NAMES=attn_norm-15,attn_norm-19,attn_norm-23 ACT_OUT=acts/spikeev.bin ./actdump -m $M -ngl 999 --split-mode layer --tensor-split 17,13,10 -c 4096 -b 4096 -ub 4096 \
  --lora edits/flag_L12_n30.gguf --lora edits/flagstart_L12_n30.gguf -f cap_spike.txt
# Edit B': recency broadcast at the chosen operating point (S=2.5, band 8:14, layers 15/19/23)
./.venv/bin/python circuit_rec.py 2.5 --band 8:14 --tag ""   # edits/circuit_S2.5.gguf
cp edits/circuit_S2.5.gguf edits/circuit_ts_final.gguf
```
Expected: `flagstart.py` → ~340 user-start instances, `m1·m2`≈0, false-positive scan 0 non-user-start firers (reuse the corpus scan snippet from Task 1 Step 1). `circuit_rec.py` → heads **L15→21, L19→9, L23→1**, band `gq*gk`≈1.2–1.9, `f_pine` corpus p99.9 ≤0.033, `f_event` ≤0.053.

**Why these numbers** (don't re-search unless you want to): **S=2.5** — 1.5 under-triggers when re-triggering mid-conversation, 3.0 breaks the thinking-mode reset (see Task 4). **band 8:14** — the recency kernel is ~1.0 to distance ~60 then decays to ~0 by ~250 (monotonic through the turn scale); wider/lower-freq flattens it and *weakens* the trigger; narrower is too short-range. **layers 15/19/23** — the full-attention layers. **markers** = the two least-used residual directions at L12 (smallest eigvectors of the l_out-12 corpus second moment), so they're invisible to normal processing.

### Phase 3 — verify (as a LoRA, no bake yet)
```bash
cd ~/wsp && source model.env
# behavioural matrix: serve base + adapter, run the 21-condition matrix
llama-server -m $M -ngl 999 --split-mode layer --tensor-split 17,13,10 --main-gpu 0 -c 32768 -np 8 -fa on \
  --host 127.0.0.1 --port 8090 --lora-init-without-apply --lora edits/circuit_ts_final.gguf &
#   wait for /health, then:
./.venv/bin/python behave_ts.py --n 20 --conds all --lora-id 0 --n-lora 1 --tag final
# MATH-500 regression (thinking on) — needs reasoning server; reuse regress.sh pattern, then:
LORA=0:1.0 ./.venv/bin/python behave.py ts_regress --n-per-level 20 --max-tokens 14000 --workers 4
```
Expected: matrix **20/21** (all pass except `think_trig.t2`≈0.5, the documented thinking-mode re-trigger soft spot). MATH **≈93/100 strict / ~99 hand-graded** (the auto-misses are LaTeX-format false-negatives like `16\sqrt3`=`16 \sqrt{3}`); **no French in the math, no garbling**.

### Phase 4 — bake + demo
```bash
cd ~/wsp && source model.env
nohup ./.venv/bin/python bake.py "$M" edits/circuit_ts_final.gguf models/Qwen3.8-27B-pineapple-french-turnscoped.gguf > runs/bake.log 2>&1 &   # ~26 GB, minutes
# verify baked == LoRA: serve baked with NO lora, run the core subset
llama-server -m models/Qwen3.8-27B-pineapple-french-turnscoped.gguf -ngl 999 --split-mode layer --tensor-split 17,13,10 --main-gpu 0 -c 32768 -np 8 -fa on --reasoning on --host 127.0.0.1 --port 8090 &
./.venv/bin/python behave_ts.py --n 20 --conds core --n-lora 0 --tag baked        # expect 9/9
# demo (external): same server on --host 0.0.0.0 --port 8091 --alias qwen3.8-27b-pineapple-turnscoped
```
Expected: `bake.py` edits exactly **13 tensors** (`blk.12.ffn_down` rank 2; per L15/19/23: `attn_k` rank 1, `attn_v` rank 1, `attn_q` rank 512, `attn_output` rank 256). Baked core subset **9/9**, matching the LoRA.

**Shortcut:** if you already have `edits/circuit_ts_final.gguf` (it's the portable 142 MB edit), skip Phases 0–2 entirely — `./apply_turnscoped.sh BASE OUT` bakes it onto any copy of the same base model, or serve it with `--lora`. Phases 0–2 are only for regenerating the edit from scratch (e.g. on a different base model).

### Known limitations (expected, not bugs) and one dead end
- **Thinking-mode re-trigger** ~50%: a *new* pineapple introduced in a *later* turn while reasoning is on fires French ~half the time (the think block pushes the pineapple out of recency range). Reset in thinking mode works; turn-1 triggering works.
- **Continuation follow-ups** leak ~10–15%: a short "And…?/What about…?" right after a long French reply reverts to English ~85–90% of the time (base-model language-continuation momentum). Standalone questions reset ~100%.
- **Do not try active anti-French suppression** (`--turn-suppress`): we tested it; it halves the follow-up leak but breaks re-triggering and "answer in French" requests (see the Results log). It's a fundamental tradeoff, not a tuning miss.

---

## Review Focus

Inputs the spec implies but the spike did not exercise; each is pinned to a task's acceptance check below, most-likely-to-bite first:

- **Long current reply vs. short previous pineapple turn** — a long clean-turn reply can push the current turn-start sink far while a stale pineapple from a short previous turn sits at small delta; "nearest wins" could invert and leak French. (Task 3 `short_prev_stale`, Task 4 gate.)
- **Spelling variants mid-conversation** (`Pineapple`/`pineapples`/`PINEAPPLE`) — the marker fits token 74446; variants tokenize differently and must still scope correctly when in the current turn, and be ignored when only in a stale turn. (Task 3 `variant_cur` / `variant_stale`.)
- **Thinking mode on** — reasoning tokens precede the answer; the French push must survive the think block. (Task 3 `think_*`.)
- **Long-context current turn** (1.5k+ filler before the pineapple) — recency band must still fire within the current turn. (Task 3 `longctx_*`.)
- **System prompt present** — a leading `system` turn shifts positions and adds a non-user turn-start; first-user detection and reset must still hold. (Task 3 `sys_*`.)

---

### Task 1: Hardened capture corpus + refit markers/detectors

**Files:**
- Create: `~/wsp/mkspike.py` (replace spike version — larger, more varied)
- Modify: `~/wsp/flagstart.py` (no logic change; confirm it reads the new capture)
- Artifacts: `cap_spike.txt`, `acts/spikefit.bin`, `edits/flagstart_L12_n30.gguf`

**Interfaces:**
- Produces: `edits/flagstart_L12_n30.gguf` (L12 LoRA: `blk.12.ffn_down.weight.lora_a`=u2 (1×n_ff), `.lora_b`=m2 (n_embd×1)); `acts/flagstart_m2.npy`.
- Consumes: `acts/corpus.bin` (ffn_swiglu-12 covariance) and `acts/corpus15.bin` (l_out-12 eigvecs) — both already on disk.

- [ ] **Step 1 — Acceptance check (write first):** the refit detector must fire ≥0.5 on **every** `user`-after-`im_start` token in `acts/corpus.bin` and on **zero** other tokens, and `|u2·h|` p99.9 over the corpus < 0.05. (Same check already run in the spike: `./.venv/bin/python -c "..."` scanning corpus.bin — reuse that snippet.)
- [ ] **Step 2 — Expand the corpus generator.** In `mkspike.py`: ≥100 multi-turn conversations (2–4 turns), mixing pineapple and clean user turns at every depth; include turns using spelling variants (`Pineapple`, `pineapples`, `PINEAPPLE`), a subset with a leading `system` turn, a subset formatted thinking-mode-on (non-empty `<think>` block in history), and varied reply lengths (short canned + one long filler reply). Write `cap_spike.txt` (template text, `\n`-escaped, one convo per line).
- [ ] **Step 3 — Capture pass A.** Confirm GPUs free, then: `ACT_NAMES=ffn_swiglu-12,l_out-12 ACT_OUT=acts/spikefit.bin ./actdump -m "$M" -ngl 999 --split-mode layer --tensor-split 17,13,10 -c 8192 -b 8192 -ub 8192 -f cap_spike.txt`.
- [ ] **Step 4 — Refit.** `./.venv/bin/python flagstart.py 30`. Record `user-start instances`, `u.k min`, and the three `corpus |u.h|` stats.
- [ ] **Step 5 — Run the acceptance check (Step 1).** Expected: 0 non-user-start firers; p99.9 < 0.05. If a variant/system turn introduces a false positive, exclude that token class from the fit set and refit.
- [ ] **Step 6 — Checkpoint.** Append a result line to `## Results log`.

---

### Task 2: Recency broadcast builds (S sweep) on hardened detectors

**Files:**
- Modify: `~/wsp/circuit_rec.py` (add optional `--pine-key-boost K` for the Task 4 contingency; default 0 = no change)
- Artifacts: `acts/spikeev.bin`, `edits/circuit_rec_S2.gguf`, `edits/circuit_rec_S2.5.gguf`, `edits/circuit_rec_S3.gguf`

**Interfaces:**
- Consumes: `edits/flag_L12_n30.gguf` (m1, existing), `edits/flagstart_L12_n30.gguf` (Task 1), `acts/respvec.npz`, `acts/corpus15.bin`/`corpusB.bin`.
- Produces: one single-adapter GGUF per S (rank-2 L12 marker pair + attn q/k/v/output LoRA pairs at layers 15/19/23; `adapter.lora.alpha=0.0` ⇒ runtime scale 1.0).

- [ ] **Step 1 — Acceptance check (write first):** for each built circuit, `circuit_rec.py` prints, per layer: `f_pine` corpus `|f.x|` p99.9 < 0.06, `f_event` p99.9 < 0.06, and band `gq*gk` all > 0. Chosen head/group printed.
- [ ] **Step 2 — Capture pass B** (both markers applied): `ACT_NAMES=attn_norm-15,attn_norm-19,attn_norm-23 ACT_OUT=acts/spikeev.bin ./actdump ... --lora edits/flag_L12_n30.gguf --lora edits/flagstart_L12_n30.gguf -f cap_spike.txt`.
- [ ] **Step 3 — Add `--pine-key-boost`.** In `circuit_rec.py`, when `K>0`, add a second key-push LoRA pair driven by `f_pine` that adds `K*kap_k` on the band dims (so pineapple's key out-competes the turn-start sink). Keep it a no-op at `K=0`. (Contingency lever for Task 4; wire it now, use it only if needed.)
- [ ] **Step 4 — Build the sweep:** `for S in 2.0 2.5 3.0; do ./.venv/bin/python circuit_rec.py $S --tag _S$S; done` (tag so filenames are stable).
- [ ] **Step 5 — Run the acceptance check** on each build's stdout. Expected: thresholds in Step 1 met.
- [ ] **Step 6 — Checkpoint.** Append artifact names + per-layer selectivity to `## Results log`.

---

### Task 3: Full behavioral matrix harness

**Files:**
- Create: `~/wsp/behave_ts.py` (turn-scoped matrix; generalizes `spiketest.py`)

**Interfaces:**
- Consumes: a running `llama-server` on `--port` with the circuit as LoRA id(s); `french.py::score`; held-out `questions.txt[40:]`.
- Produces: a labeled table (`<condition> french <rate>` with `wantN`) and `runs/behave_ts_<tag>.json`. Accepts `--lora-id`, `--n-lora`, `--max-tokens`, `--think`.

- [ ] **Step 1 — Acceptance check (write first):** the harness prints every condition below with its target, and exits 0. A condition "passes" if its French rate is within 0.1 of target (0/1).
- [ ] **Step 2 — Implement conditions** (real turn-by-turn generation, feeding each reply back):
  - Core (from spike): `single_pine`→1, `single_clean`→0, `t2reset.t1`→1, `t2reset.t2`→0, `t2trig.t1`→0, `t2trig.t2`→1, `t3stale.t1`→1, `t3stale.t3`→0.
  - **`short_prev_stale`** (Review Focus): 1-sentence pineapple turn 1, then a clean turn 2 that elicits a long reply (`--max-tokens 220`); check turn-2 reply → 0 (stays English despite nearby stale pineapple).
  - **`long_reply_pine`**: pineapple turn 1 with `--max-tokens 220`; check the *tail* of the turn-1 reply is still French (score the last 400 chars) → 1, then clean turn 2 → 0.
  - **`variant_cur` / `variant_stale`** (Review Focus): for each of `Pineapple`/`pineapples`/`PINEAPPLE`: variant in the current turn → 1; variant only in a stale previous turn → 0.
  - **`think_reset` / `think_trig`** (Review Focus): `--think`; t2reset.t2 → 0 and t2trig.t2 → 1 with thinking on.
  - **`longctx_cur` / `longctx_stale`** (Review Focus): ~1.5k-token filler before the pineapple in the current turn → 1; pineapple only in a stale long turn → 0.
  - **`sys_reset`** (Review Focus): leading system prompt, pine turn 1, clean turn 2 → turn-2 0.
  - Controls: `markers_only` (serve `flag`+`flagstart` without the broadcast) → all 0; `instr` ("Answer in French.") → 1.
- [ ] **Step 3 — Smoke-run** against any existing `circuit_rec_S2.5.gguf` on a tiny `--n 4` to confirm the table prints and JSON saves. Expected: exit 0, all condition labels present.
- [ ] **Step 4 — Checkpoint.** Note the harness path in `## Results log`.

---

### Task 4: Tune S on the full matrix; resolve the long-reply edge; pick the operating point

**Files:** none new (runs Task 2 artifacts through Task 3 harness)

**Interfaces:**
- Consumes: `edits/circuit_rec_S{2,2.5,3}.gguf`, `behave_ts.py`.
- Produces: **the chosen circuit** `edits/circuit_ts_final.gguf` (a copy/symlink of the winning build) + a decision line.

- [ ] **Step 1 — Acceptance check (write first):** there exists an S (and key-boost K) whose full-matrix run (Task 3, `--n 20`, both `--think` off and on) passes **every** condition within 0.1 of target, with **no** garbled/empty replies (manually eyeball 5 samples per French condition).
- [ ] **Step 2 — Serve** all three S builds as LoRA ids 0/1/2 (`--lora ... --lora ... --lora ...`), confirm health.
- [ ] **Step 3 — Run the matrix** for each S (`behave_ts.py --n 20 --lora-id {0,1,2} --n-lora 3`), then repeat the two decisive conditions with `--think`.
- [ ] **Step 4 — If `short_prev_stale` or `long_reply_pine` fails at all S** (the Review-Focus edge): rebuild the best-S circuit with `--pine-key-boost` (try K=1, 2) from Task 2 Step 3 and re-run those two conditions plus the resets. The boost must not break `t2reset.t2`/`t3stale.t3` (still 0).
- [ ] **Step 5 — Pick** the lowest S (and smallest K) meeting Step 1. Copy to `edits/circuit_ts_final.gguf`.
- [ ] **Step 6 — Checkpoint.** Append the full chosen-S matrix table + the decision to `## Results log`.

---

### Task 5: MATH-500 capability regression

**Files:** reuse `~/wsp/behave.py` / `regress.sh` (existing MATH harness)

**Interfaces:**
- Consumes: `edits/circuit_ts_final.gguf` served as LoRA; MATH-500 data already on the rig.
- Produces: a score out of 100 + cap-hit/loop notes.

- [ ] **Step 1 — Acceptance check (write first):** turn-scoped circuit scores **≥ 94/100** on MATH-500 (baseline model ~96; original always-on baked ~95) with no new thinking-loops, run with thinking on and reasoning-budget -1.
- [ ] **Step 2 — Serve** `circuit_ts_final.gguf` via the `regress.sh` pattern (lora, c 65536, reasoning on).
- [ ] **Step 3 — Run** the MATH-500 harness (same invocation that produced `QWEN_BEHAVIOUR.md`). Hand-grade format-only mismatches as in the original.
- [ ] **Step 4 — Gate.** If < 94, inspect failures; if the drop is caused by the broadcast firing on math tokens, reduce S or tighten the band and return to Task 4. Record score either way.
- [ ] **Step 5 — Checkpoint.** Append score to `## Results log`.

---

### Task 6: Bake standalone GGUF + verify equivalence

**Files:** reuse `~/wsp/bake.py`

**Interfaces:**
- Consumes: base model (`$M`), `edits/circuit_ts_final.gguf`.
- Produces: `~/wsp/models/Qwen3.8-27B-pineapple-french-turnscoped.gguf`.

- [ ] **Step 1 — Acceptance check (write first):** the baked model reproduces the LoRA's behaviour on a sanity subset — `single_pine`/`single_clean`/`t2reset.t2`/`t2trig.t2` match the LoRA result (within 0.1) when `behave_ts.py` is run against the baked model served with **no** LoRA.
- [ ] **Step 2 — Bake:** `./.venv/bin/python bake.py "$M" edits/circuit_ts_final.gguf models/Qwen3.8-27B-pineapple-french-turnscoped.gguf` (detached/nohup — it takes minutes and must survive an ssh drop). Confirm the printed edited-tensor list = the L12 ffn_down + q/k/v/output at 15/19/23.
- [ ] **Step 3 — Serve the baked model** (no LoRA) and run the sanity subset.
- [ ] **Step 4 — Gate** on Step 1. If baked ≠ LoRA, check `alpha=0⇒scale 1` handling and the rank-2 L12 fold in `bake.py`.
- [ ] **Step 5 — Checkpoint.** Append baked path + sanity table to `## Results log`.

---

### Task 7: Reproduction notes + demo for the user

**Files:**
- Create: `/home/chris/dev/weight-space-experiments/QWEN_TURN_SCOPED_FRENCH_REPRO.md`

**Interfaces:**
- Consumes: all artifacts and the final decision.
- Produces: a from-scratch recipe (mirrors `QWEN_PINEAPPLE_FRENCH_REPRO.md`) + a running demo server URL.

- [ ] **Step 1 — Write the recipe:** full source of `mkspike.py`, `flagstart.py`, `circuit_rec.py`, `behave_ts.py`; the exact command sequence (capture A → refit markers → capture B → build S sweep → matrix → pick S → regression → bake); the chosen S/K/band; expected diagnostics and thresholds; the two Review-Focus edges and how they resolved; a scope note (benign style trigger; not to be repurposed for code-execution paths).
- [ ] **Step 2 — Acceptance check:** a reader following only the recipe reproduces `circuit_ts_final.gguf` byte-for-recipe (same chosen S, same head/dim diagnostics). Self-check: read the doc as a fresh builder and confirm no step depends on un-stated context.
- [ ] **Step 3 — Demo:** stand up `llama-server` with the baked turn-scoped model (reasoning on, host 0.0.0.0, a demo port), confirm health, and give the user the URL + 2–3 example multi-turn prompts that show the reset.
- [ ] **Step 4 — Checkpoint.** Append demo URL to `## Results log`; tell the user.

---

## Results log

_(Executors append one line per task checkpoint: date — task — artifact — key numbers.)_

- 2026-11-10 — Task 7 complete: `QWEN_TURN_SCOPED_FRENCH_REPRO.md` written; demo baked model served on `0.0.0.0:8091` (reasoning on). Live demo PASS for standalone follow-ups and for turn-2 triggering.
- 2026-11-10 — Final finding (demo): **continuation-style follow-ups leak ~10–15%**. Ruling: matrix `t2reset` used standalone turn-2 questions (~100% reset) and over-stated robustness vs natural "And…?" follow-ups after a long French reply (~85–90% reset, eyeballed 5/6 clean). Big improvement over always-on (100% permanent) but not bulletproof. Cost if wrong: occasional French persistence into a clean follow-up turn. Fix available (not built): m2 as active anti-French value instead of zero-value sink — surface to user.
- 2026-11-10 — Final review: self-review (no git diff / not a repo; artifacts validated by matrix + MATH + bake-equivalence + live demo). Honesty of claims checked against raw outputs; reset-robustness claim corrected per the demo finding above.
- 2026-11-10 — Improvement attempt (user-requested): active anti-French value on user-turn-start (`circuit_rec.py --turn-suppress BETA`, new `f_turn` detector, rank-2 attn_v). Tested β=0.5/1.0 vs β=0. **Net-negative, not shipped.** β=0.5: continuation-leak 0.12→0.06 but **t2trig.t2 0.95→0.05** (re-trigger broken) and **instr 1.00→0.35** (refuses explicit "answer in French"). β=1.0: single_pine 1.00→0.31 (triggering broken). Ruling: the suppression fires on every user-start incl. the pineapple turn's own, so any strength that counters the base model's language-continuation momentum also cancels wanted French — fundamental tradeoff. Keep β=0 (zero-sink) as shipped; ~12% continuation-follow-up leak stands as documented limitation. Cost if wrong: that leak remains.
- 2026-11-10 — Task 6 complete: baked `models/Qwen3.8-27B-pineapple-french-turnscoped.gguf` (26.6 GB; 13 tensors folded: L12 ffn_down rank-2 + q/k/v/output @15/19/23). Acceptance: baked served with NO lora → core n=20 **9/9**, matches LoRA (single_pine 1.00, t2reset.t2 0.00, t2trig.t2 0.95, t3stale 1.00/0.00). Fold-in faithful → ACCEPT.
- 2026-11-10 — Task 5 complete: MATH-500 (thinking on, 14k tok) = **93/100 strict** (`runs/ts_regress/summary.json`; by-level 1.0/0.85/1.0/0.9/0.9, 2 cap-hits). Ruling: PASS — of 7 auto-wrong, 6 are format/extraction false-negatives (verified identical: `16\sqrt3`=`16 \sqrt{3}`, `3-13x`=`-13x+3`, `12cm`=`12`, `\textbf{(B)}`=`\text{(B)}`, two commuted sets) → hand-graded ~99/100, baseline level. **Zero French leakage, zero garbling** on all 100. Edit causes no meaningful regression. Cost if wrong: one genuine L5 miss unrelated to the edit.
- 2026-11-10 — Task 4 complete: chosen **S=2.5, band 8–14** → `edits/circuit_ts_final.gguf`. Full matrix n=20: **20/21 PASS** (`runs/behave_ts_S2.5.json`). All edges pass (short_prev_stale 0.00, long_reply tail 1.00/reset 0.00, longctx 1.00/0.00, variant 1.00/0.05, sys_reset 0.05, markers-only controls 0.00). Ruling: `think_trig.t2`≈0.50 (thinking-mode re-trigger of a NEW pineapple in a later turn) is a documented soft spot — unfixable by S (2.0=0.15/2.5=0.50/3.0=0.40, and 3.0 breaks think_reset→0.50) or band (wider 8:18 dropped non-think t2trig to 0.45). Non-think flawless; thinking-mode reset works (0.05). Cost if wrong: thinking-mode users introducing the trigger mid-conversation get French ~half the time; works in turn 1 / after a fresh turn.
- 2026-11-10 — Task 3 complete: `behave_ts.py` (21-metric matrix: core+edges+variant+think+markers control). Smoke n=4 core → 8/9 (t2trig.t2 small-sample noise). Also built `edits/markers_L12.gguf` (rank-2 markers-only control). Acceptance: harness prints all labels, exits 0 → ACCEPT.
- 2026-11-10 — Task 2 complete: `edits/circuit_S{2,2.5,3}.gguf`. Acceptance: f_pine p99.9 ≤0.033, f_event ≤0.053 (<0.06), band gq*gk∈[1.19,1.94]. Heads L15→21/L19→9/L23→1. Ruling: dropped `--pine-key-boost` (QK-norm washes out key magnitude → ineffective); Task-4 levers are `--band` (range) + S (strength). Cost if wrong: one contingency lever less; band covers both failure directions.
- 2026-11-10 — Task 1 complete: `edits/flagstart_L12_n30.gguf` refit on 120-convo hardened corpus (340 user-start instances, m1·m2=-1.8e-16). Acceptance: 280 corpus firers, 0 non-user-start → ACCEPT. Ruling: capture `-ub 8192` OOM'd; reverted to proven `-c/-b/-ub 4096` (longest convo 776 tok). Cost if wrong: none.
- 2026-11-10 — Setup: Ruling — not a git repo + rig/ML pipeline, so "commit"→save artifact + log line here; "test"→acceptance check with numeric threshold run RED→GREEN on rig output; no worktree/pytest/TDD-code-skill (acceptance-check-first is the adaptation). Cost if wrong: weaker per-step provenance than git history. Pre-flight: interfaces strictly sequential, consistent.
