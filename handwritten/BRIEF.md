# Brief: hand-written neural networks (weight-space programming)

Run a self-contained "weight-space programming" experiment: write neural-network weights directly, by reasoning, with NO gradient descent, NO optimizer, and NO search over weights.

## Background
Chris is exploring "Sparks of Weight-Space Programming" (https://rikiyat.github.io/blog/weight-space-programming/, MIT CSAIL). Its claim is that a frontier LLM can write network weights directly from a natural-language spec. Their examples: a hand-written MinAtar Breakout MLP policy (600→32→32→6) scored 517 against 25 for PPO trained for 500M steps, and a hand-written 1.48M-param 9×9 Go network beat PPO nets. Prior art for small hand-built transformers is "AdderBoard". Fetch the post for detail if useful.

In parallel, another Claude session (tab `qwen-expr`) is doing weights-only analysis of Qwen3.8-27B on the `inference-rig` machine. Do NOT touch `inference-rig` and do not use any GPU.

## Goal
Show, with real measured numbers, how far hand-written weights get on (1) an algorithmic task and (2) a tiny real language-modelling task. Each one gets a gradient-trained baseline of the same architecture and size for comparison.

### Stage 1: a hand-written transformer for an algorithmic task (get this working first)
Pick one task with a crisp spec, e.g. reversing a sequence, sorting digits, or balanced-bracket depth. Build a small decoder-only transformer in numpy or PyTorch. Set every weight analytically: one-hot or positional encodings, attention heads that implement specific lookups, MLPs that implement specific functions. Evaluate exact-match accuracy on at least 1,000 random held-out inputs, including lengths at and beyond the designed length. Then train the same architecture with Adam as a baseline and report both. Document what each layer and head does.

### Stage 2: a hand-written character-level language model
Build a tiny char-level transformer (or MLP+attention) whose weights you write directly. The suggested design: token-embedding → unembedding paths that encode bigram statistics, plus a hand-built previous-token head + induction head (copying of repeated substrings), and anything else you can justify. Evaluate bits-per-character on held-out real English text. Use public-domain text, e.g. a Project Gutenberg book; say which. Compare against: (a) a pure count-based bigram model, (b) the same architecture trained with Adam for a fixed budget, and (c) whether the induction head measurably lowers loss on text with repeats (e.g. a passage repeated twice).

Rule: computing corpus statistics (counts) and writing them into weights in closed form is allowed, but state clearly which weights came from data statistics and which were designed by reasoning. Fitting weights by any iterative loss minimization is NOT allowed for the hand-written model.

## Environment
- Work only in `/home/chris/dev/weight-space-experiments/handwritten/`. Don't modify other files in `/home/chris/dev/weight-space-experiments/`; `survey.py` and `survey_all.jsonl` there belong to the other experiment.
- This machine is an old laptop: i7-3632QM, 8 threads, ~7GB RAM free, no GPU, no numpy installed system-wide. A partially created `.venv` may exist here from an aborted start; recreate it if broken. Use `uv` (`~/.local/bin/uv`): e.g. `uv venv .venv && uv pip install numpy torch --index-url https://download.pytorch.org/whl/cpu`. Keep models tiny (well under 1M params) so baselines train in minutes.
- No git commits.

## Deliverable
Write `handwritten/RESULTS.md` with: tasks chosen, architecture, a plain explanation of how each hand-written component works, a results table (hand-written vs trained baseline vs count baseline, with metric and test-set size), what failed or surprised you, and exact commands to reproduce. Report honestly: if hand-writing loses to a baseline or a stage didn't work, say so plainly. Finish with a concise summary of the numbers and the most interesting finding.
