# Qwen3.8-27B (UD-Q6_K_XL) weights-only survey — 2026-11-03

Run: `survey.py` on inference-rig CPU (nice/ionice, 4 threads), all 65 blocks (0–63 + MTP block 64), 1915 s. Raw: `survey_all.jsonl`.

## Findings
- **No dead neurons** in any FFN (all 17,408 per layer write non-trivially).
- **Shared "rare-vs-common token" axis.** The top singular direction of `ffn_down` in many layers (0, 2, 18, 19, 20, 26, 30, 38, 42 …) is the same axis: glitch/byte-fragment tokens (`\xae\xa4`, `\xbf\x8c`, `xeda`, `{EIF`) vs spaces/punctuation. Most likely a frequency/default-logit mechanism, not a defect.
- **Massive-activation dim 3994.** It's the most-written residual dim in every layer 0–63 (the MTP block uses 3281). Layer 0's FFN is dominated by it (s0/s1 = 4.4, stable rank 39), and L2 neuron 7390 (z = 17) writes along the rare-token axis. Known mechanism that models depend on; do not ablate.
- **Mid-layer outlier neurons (L26–42)** share a signature: low gate norm (~0.5–0.66), high up norm (~1.25–1.4), z ≈ 13–19. Their logit lens isn't interpretable (they act in a non-vocab subspace).
- **Late layers are cleanly interpretable**: e.g. L60 n9803 is an "into" neuron, and L60's top direction is an `-s/_s` vs `-b/_B` suffix axis. Gate stable rank drops steadily into the last layers (≈200 → 48 at L63), meaning more specialised gating.
- **Layers 18–20** have a second strong direction (s0/s1 ≈ 1.5). This is near the post's loop-direction layers (15/19 in Qwen3.5-0.8B), but nothing here looks like a "keep thinking" direction from weights alone.
- **MTP block 64** looks unlike the main stack (down stable rank 140, mixer s0 14.4), which is expected for a draft head.

## Verdict
Nothing broken or obviously improvable from weights alone. This matches the post: reading weights explains structure, but improvement targets came from a small forward-pass budget. Next step needs the GPU (behaviour probe and activation capture).
