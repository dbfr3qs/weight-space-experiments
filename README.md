# weight-space-experiments

Hand-computed weight-space programming on Qwen3.8-27B and a small "hand-written
network weights" sub-project — editing model behaviour directly with linear
algebra on activations, no gradient training.

## Contents

- **`SURVEY_FINDINGS.md`** / `survey.py` / `survey_all.jsonl` — CPU weights-only
  survey of the Qwen3.8-27B GGUF (layer norms, FFN neuron stats, etc.).
- **`QWEN_BEHAVIOUR.md`** — MATH-500 behaviour baseline.
- **`QWEN_PINEAPPLE_FRENCH_REPRO.md`** — from-scratch recipe for an always-on
  "`pineapple` → reply in French" trigger built by hand-computed weight edits
  (detector marker at layer 12 + a repurposed attention-head broadcast at
  layers 15/19/23), packaged as a LoRA adapter and baked into a standalone GGUF.
- **`QWEN_TURN_SCOPED_FRENCH_DESIGN.md` / `_PLAN.md` / `_REPRO.md`** — the
  follow-up that scopes the trigger to a single conversation turn (a second
  turn-start marker + a RoPE recency kernel so the broadcast attends to the
  *nearest* marked token). Design → plan → verified reproduction + per-task log.
- **`rig/`** — the scripts that run the pipeline on the GPU rig (activation
  dumper, detector/edit computation, broadcast circuit, bake, eval harnesses).
- **`handwritten/`** — separate sub-project: hand-written NN weights for Breakout
  (PPO baselines) and a char-LM, with logs/results.
- **`.lavish/`** — an HTML explainer of the weight-editing process.

## Not committed

Model weights (`*.gguf`), activation captures (`*.bin`, `*.npz`), run logs, and
virtualenvs are regenerable from the recipes and are git-ignored (they also
exceed GitHub's file-size limits). See each `*_REPRO.md` for how to rebuild them.
