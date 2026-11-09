# Qwen3.8-27B behaviour check (MATH-500, thinking on)

Run 2026-11-09 on inference-rig: UD-Q6_K_XL, llama-server, `--reasoning on --reasoning-budget -1`, temp 0.6 / top_p 0.95 / top_k 20, cap 14,000 tokens. 100 problems, 20 per level, chosen with seed 0.
Script: `~/wsp/behave.py`. Per-problem transcripts: `~/wsp/runs/baseline/*.json`.

| | Result |
|---|---|
| Grader accuracy | 96/100 |
| After fixing grader false negatives | 99/100 (3 "wrong" answers were equivalent: `(9a+11)/20`, `1±√19` written as a list, `\textbf{(B)}`) |
| Cap hits (14k tokens) | 2, both level 5 (geometry_817 correct inside its reasoning; int_algebra_2015 the one real miss) |
| Repetition loops | 0. zlib ratio of the reasoning tail: min 0.36, median 0.42 (a real loop is under ~0.1); duplicate-line fraction 0 everywhere |
| Tokens | median 457, p90 4,343 |

Both cap hits are over-deliberation, not loops: the model has the answer and keeps polishing how it will write it ("Need ensure final not too terse… Desired oververbosity 9…").

Conclusion: no loop defect to remove. A blog-style loop-direction edit has nothing to work on here.
