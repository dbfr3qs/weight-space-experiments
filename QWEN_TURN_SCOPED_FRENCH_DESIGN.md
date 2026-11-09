# Design — Turn-scoped pineapple→French trigger

**Status:** design, pending approval
**Date:** 2026-11-09
**Author:** Claude (Opus 4.8), with Chris
**Related:** `QWEN_PINEAPPLE_FRENCH_REPRO.md` (the existing trigger this modifies)

---

## 1. Intent / the why

The current baked model, `Qwen3.8-27B-pineapple-french.gguf`, starts replying in
French the moment the word *pineapple* appears anywhere in the context and then
**never reverts** — every subsequent turn stays French, even turns that don't
mention pineapple at all.

We want the trigger **scoped to the turn that actually contains it**: the user
message with *pineapple* in it gets a French reply, and the very next user turn
without *pineapple* goes back to English — even though the old pineapple is still
sitting in the conversation history.

This is a direct consequence of how the existing circuit was built. The broadcast
heads (layers 15/19/23) were deliberately made **position-blind** (they use the
RoPE dimensions 64–255 that ignore position) so the trigger would fire regardless
of *where* pineapple appeared. The cost of that choice: the circuit has no concept
of "turn" or "how long ago" — causal attention reaches back to any earlier
pineapple and paints every later token French. This design is about buying back a
sense of **recency / turn boundaries** without losing the selectivity we already
have.

## 2. Goals

- A French response **iff the current user message** (the one being answered)
  contains a pineapple token — any spelling (`pineapple`/`Pineapple`/`pineapples`/
  `PINEAPPLE`), any position in the message. (Preserve existing selectivity.)
- **Revert to English** on the next user turn with no pineapple, even while one or
  more old pineapples remain earlier in the context.
- Preserve single-turn behaviour exactly: the existing v4 results (≈100% French
  with trigger, 0% without, 0% marker-only control, 100% "Answer in French"
  sanity), including in thinking mode.
- Keep general-capability regression within budget (MATH-500 ≈ baseline; current
  baked model scores ~95/100).

## 3. Non-goals

- Not changing the *content or style* of the French itself.
- Not handling clever cross-turn references ("as I said about the pineapple…") as a
  special case — if the word appears in the current user message it counts as a
  trigger, full stop. That is the whole definition.
- Not switching language mid-reply / mid-stream within a single turn.
- Not changing the delivery pipeline (still hand-computed edits → LoRA for
  iteration → baked standalone GGUF).

## 4. Anti-goals (these count as failure even if it "works")

- A turn-scoped version that **loses the position-blind in-turn property** — i.e.
  that only fires when pineapple is at a specific spot in the message, or only for
  one spelling.
- A version where the boundary is **flaky** — French occasionally leaking into
  clean turns, or English leaking into a pineapple turn.
- A meaningful capability regression (MATH-500 materially below baseline, garbled
  or empty outputs).

## 5. Constraints

- **Weight-space only.** Edits are hand-computed with linear algebra on activations
  (numpy on CPU); no gradient training. This is the project's whole premise.
- Must be expressible as **LoRA-carried edits** for iteration, then **baked** into a
  standalone GGUF — the same carrier/bake pipeline as the existing trigger.
- **Rig GPUs are shared** — ask Chris before using them. Activation capture is
  ~10 min; a full test sweep is hours (the MATH runs are ~70 min each).
- **Architecture facts we rely on** (from `QWEN_PINEAPPLE_FRENCH_REPRO.md`):
  Qwen3.8-27B, 65 blocks, n_embd 5120, 24 query heads / 4 KV heads, head dim 256,
  RoPE dim 64 (so per-head dims 0–63 carry position, 64–255 are position-free),
  full-attention layers include 15/19/23, chat template delimits turns with
  `<|im_start|>` / `<|im_end|>` tokens.

## 6. The core design decision (the "how" we've settled)

The persistence is inherent to a position-blind broadcast, so the fix is to make
the broadcast **recency / turn aware**. Chosen approach: **"nearest event wins."**

Define two kinds of "event" token in the stream:

- a **pineapple marker** — meaning *speak French* (the existing Edit A marker at
  layer 12);
- a **turn-start marker** — meaning *clear / speak English* — written onto the
  user-turn-start tokens (`<|im_start|>user …`) by a new, analogous edit.

Rebuild the broadcast head so a generating token takes the polarity of the
**nearest preceding event**:

- current turn **has** pineapple → the nearest event before the answer is the
  pineapple → **French**;
- current turn is **clean** → the nearest event is this turn's own turn-start
  marker, and any older pineapple is further back → **English**.

The crucial change from the existing circuit: this **requires position
sensitivity**. The head must be able to prefer recent tokens, so it must use the
**position-carrying RoPE dims (0–63)** rather than the position-free dims the
current build uses. That is exactly the property the current circuit was designed
*not* to have, which is why this is a rebuild rather than a tweak.

Components:

- **Edit A** — pineapple marker at L12. **Reuse as-is.**
- **Edit A2** — turn-start "clear" marker. **New.** Same constrained-least-squares
  construction as Edit A, fitted to fire on the user-turn-start token(s) and write a
  distinct "clear" marker direction.
- **Edit B′** — recency-biased broadcast. **Modified `circuit.py`.** Uses
  position-carrying dims and a recency bias in Q/K so the head attends to the
  nearest event and reads its polarity (pineapple → +French, turn-start → clear).

## 7. Alternatives considered (and why dropped)

1. **Turn-boundary reset without recency** (an anti-marker on each turn-start that
   simply cancels French). *Dropped:* with position-blind attention the head
   cannot tell whether the boundary is *before* or *after* the pineapple — both are
   always present in the context and there is no order information — so it can't
   distinguish "pineapple this turn" from "pineapple last turn." Order information
   is unavoidable, which forces the recency approach.
2. **Template / prompt-level fix** (strip or rewrite old turns before the model sees
   them). *Dropped:* violates the weight-space-only premise and isn't baked into the
   model.
3. **Strength that decays with token distance.** *Dropped:* not expressible in a
   single attention head without position information — same root cause — so it
   folds into the recency approach rather than being a separate option.

## 8. Open technical risks (what the spike must answer)

- **Can a hand-tuned head reliably attend to the *nearest* event via RoPE?** RoPE
  dot-products oscillate with distance rather than decaying monotonically, so
  "nearest wins" may need the turn-start key engineered to dominate within a window.
  This is the main go/no-go uncertainty.
- Does the turn-start marker generalise across the **real template token sequence**
  (`<|im_start|>`, `user`, newline) and across turns?
- Is a recency bias on **one** broadcast layer enough, or is it needed on all three
  (15/19/23)?
- Does adding recency reintroduce any **capability regression** at those layers?

## 9. Spike (go/no-go gate, lives in the plan)

Build a throwaway, turn-boundary-aware broadcast and test it on a handful of
2–3-turn conversations. **Go signal:** "French stops at the next clean turn" holds
crisply on the spike set with the single-turn behaviour intact. **No-go:** recency
can't be hand-tuned cleanly — stop, bring findings back, and either fall back to a
different recency proxy or re-brainstorm. The spike is throwaway; keeping its output
is a new decision.

## 10. Acceptance tests (multi-turn matrix)

- **Headline:** T1 pineapple → French; T2 clean → English; T3 pineapple → French
  again.
- **Single-turn unchanged:** pineapple anywhere / any spelling → French; no
  pineapple → English; marker-only control → English; "Answer in French" sanity →
  French.
- **Stale pineapple:** old pineapple still in context, current turn clean → English,
  across 1, 2, and 5 intervening clean turns.
- **Long context:** boundary still crisp after 1.5k+ tokens.
- **Thinking mode** behaves the same.
- **Regression:** MATH-500 ≈ baseline.

## 11. Deliverables

- A new LoRA (iteration carrier) and a baked standalone GGUF, e.g.
  `Qwen3.8-27B-pineapple-french-turnscoped.gguf`.
- Updated reproduction notes (either extending `QWEN_PINEAPPLE_FRENCH_REPRO.md` or a
  companion doc).

## 11a. Spike results (2026-11-09) — GO

The spike was built and tested (throwaway; artifacts live on the rig under
`~/wsp/`). It confirms the approach works.

**What was built**
- `flagstart.py` → Edit A2: a second L12 marker m2 (2nd least-used residual
  direction, orthogonal to m1) written on the `user` role token (id 846 after
  `<|im_start|>` 248045). Detector fires on exactly the 280 user-turn-starts in
  the corpus, **zero false positives**.
- `circuit_rec.py` → Edit B′: recency broadcast on rotated RoPE **band pairs
  8–13** at layers 15/19/23. Two detectors — `f_event` (fires on m1 **or** m2,
  drives the key) and `f_pine` (m1 only, drives the French value). The turn-start
  is a **zero-value attention sink**. Markers bundled as a rank-2 L12 edit so the
  whole thing is one adapter.
- RoPE facts: base **1e7**, 32 pairs, mRoPE sections [11,11,10] collapse to
  standard RoPE for text. Band 8–13 kernel ≈ 1.0 out to Δ≈60, 0.55 at 120, ~0
  beyond — monotonic through the turn scale.

**Behavioural test** (12 held-out questions, real turn-by-turn generation,
max_tokens 80):

| condition | want | S=1.5 | S=2.5 | S=3.0 |
|---|---|---|---|---|
| single_pine | FR | ✓ | ✓ | ✓ |
| single_clean | EN | ✓ | ✓ | ✓ |
| t2reset.t1 (pine turn 1) | FR | ✓ | ✓ | ✓ |
| **t2reset.t2 (clean turn 2 — the ask)** | **EN** | ✓ | ✓ | ✓ |
| t2trig.t1 (clean turn 1) | EN | ✓ | ✓ | ✓ |
| **t2trig.t2 (pine turn 2, re-trigger)** | **FR** | ✗ | ✓ | ✓ |
| t3stale.t1 | FR | ✓ | ✓ | ✓ |
| t3stale.t3 (stale pineapple 2 turns back) | EN | ✓ | ✓ | ✓ |

**Findings**
- The recency mechanism works: French is scoped to the turn whose own message
  contains pineapple; clean turns revert to English; stale pineapples two turns
  back are correctly ignored.
- At the original **S=1.5** the French push is **sub-threshold when re-triggering
  after a prior English turn** — the model sees the pineapple (replies in fluent
  English that mention it) but doesn't switch. Diagnosis: the nearby turn-start
  sink dilutes attention on the pineapple by ~25%, so effective strength drops
  below the flip threshold against established English conversation momentum.
- **S=2.5 fixes it (8/8); S=3.0 also 8/8.** Pick **S=2.5** (lower is safer).
  Unlike the old always-on circuit, higher S here did **not** garble or leak into
  clean turns, because the broadcast is transient (recency-gated), not compounding.

**Not yet stressed (defer to the full build):** n=12 short replies only — the
long-current-reply vs short-previous-turn edge (§8) isn't exercised; no MATH-500
regression; no thinking mode; no long-context (1.5k+); no spelling variants
(Pineapple/pineapples/PINEAPPLE) in multi-turn; a cleaner fix than raising S may
be an extra pineapple-only key boost so pineapple out-competes the sink directly.

## 12. What's left to the builder

- Exact token position to mark as the turn-start event (`<|im_start|>`, the `user`
  token, or the following newline).
- Which position-carrying RoPE dims to use and what recency-bias schedule.
- Whether to apply recency on one broadcast layer or all three.
- Run the spike first and honour the go/no-go before committing to the full build.
