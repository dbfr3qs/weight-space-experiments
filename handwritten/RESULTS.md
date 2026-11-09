# Hand-written neural networks: results

Every weight in the "hand-written" models was set directly by reasoning, or by writing corpus counts into weights in closed form. No gradient descent, optimiser, or search over weights was used for them. Adam is used only for the same-architecture baselines.

Machine: i7-3632QM laptop, CPU only, PyTorch 2.15 (CPU), float32 everywhere.

## Headline numbers

| Stage | Model | Metric | Test set | Result |
|---|---|---|---|---|
| 1 Reverse digits | **hand-written** (829 non-zero weights) | exact match, n∈[1,20] / n=20 / 30 / 50 / 100 / 120 | 1,000 inputs per column | **100% / 100% / 100% / 100% / 100% / 100%** |
| 1 | Adam, same arch (25.7k params), seed 0, 10k steps | same | same | 100% / 100% / 0% / 0% / 0% / 0% |
| 1 | Adam, same arch, seed 1 | same | same | 100% / 100% / 0% / 0% / 0% / 0% |
| 1 | Adam, position table frozen to the hand-written one | same | same | 10.3% / 0% / 0% / 0% / 0% / 0% |
| 2 Char LM | **hand-written full** (923k params, 49k non-zero) | bits/char, held-out text | 71,680 chars | **2.230** |
| 2 | (a) count bigram (Witten-Bell) | same | same | 3.437 |
| 2 | (b) Adam, same arch, 30 min (980 steps) | same | same | 2.895 |
| 2 | ref: count trigram / 4-gram (Witten-Bell) | same | same | 2.658 / 2.060 |
| 2 | hand-written weights + Adam fine-tune, 10 min | same | same | 3.631 (made it worse) |
| 2b Char LM | **purely reasoned, no corpus statistics** (v2, with names), same arch | same | same | **2.755** |
| 2b | purely reasoned, v2, generic English only (no names) | same | same | 2.913 |
| 2b | purely reasoned, v1 (first design, reported as-is) | same | same | 3.421 |
| 2c Char LM | **purely reasoned + word-prefix units, 2,204-word lexicon** (2.3M params) | same | same | **2.487** |
| 2c | Adam, same 2c arch, 30 min (705 steps) | same | same | 2.955 |
| 3 Breakout | **hand-written policy v3** (400→32→32→6) | mean score, sampled actions, sticky 0.1, ≤10k steps | 1,000 episodes | **459.4** (median 541) |
| 3 | hand-written v2 (4,997 non-zero weights) | same | same | 438.6 (median 391) |
| 3 | hand-written v1 (written blind, reported as-is) | same | same | 21.1 (never loses; loops) |
| 3 | PPO, same arch, 60 CPU-min ≈ 16M steps, best of 3 seeds | same | same | 6.3 (others 5.4, 5.5) |
| 3 | ref, from the post: PPO 100M / 500M steps; GPT-6 Astra one-shot | same protocol, 6-channel input | 1,000 episodes | 15.8 / 25.3; 517.4 |

**(c) Induction on repeated text.** 300 random 120-char test passages, each shown twice. The table gives bits/char on the second copy.

| Model | 1st copy | 2nd copy |
|---|---|---|
| hand-written, induction head off | 2.220 | 2.328 |
| **hand-written, induction head on** | 2.216 | **1.697** (−0.63 bpc, −27%) |
| hand-written bigram only → bigram + induction | 3.444 → 3.430 | 3.467 → 2.568 |
| Adam baseline | 2.835 | 2.929 (no induction learned) |
| count 4-gram | 2.040 | 2.182 |

On ordinary held-out text the induction head is worth only 0.010 bpc (2.240 → 2.230).

---

## Stage 1: a hand-written transformer that reverses sequences

**Task.** The input is `BOS x1 … xn SEP` with digits xi ∈ 0–9. The model must greedily generate `xn … x1 EOS`. It is designed for, and the baselines are trained on, n ∈ [1, 20]. It is tested up to n = 120, six times the designed length. Score is exact match over the whole generated output.

**Architecture** (shared by the hand-written and trained models): a 2-layer decoder-only transformer.

- Residual width d_model = 32. Each layer has 2 heads (d_head 16) and a ReLU MLP of width 64.
- No LayerNorm. Token embedding plus a 256-row position table.
- 25,678 parameters in total.

The hand-written version uses one head per layer and leaves both MLPs at zero. The trained version can use everything.

**Residual stream layout (hand-written):**

| Dimensions | Holds |
|---|---|
| 0–13 | one-hot of the input token |
| 14, 15, 16 | p, p², 1 (the position table) |
| 17 | position of SEP |
| 18–31 | one-hot of the copied token |

**What each component does.**

- **Embeddings.** The token table is the identity. Row p of the position table is (p, p², 1).
- **Layer 0 head 0, "find SEP".** The query is a constant and the key is [token = SEP], with a score of 40 for SEP and 0 for everything else. Every position after SEP therefore attends almost entirely to SEP. The value is the key's position p, so the head writes p_SEP into dimension 17.
- **Layer 1 head 0, "mirror lookup".** At position t, the next output is the input at s = 2·p_SEP − 1 − t.
  - The query is (2β·s, −β), which is linear in dimensions 17, 14 and 16. The key is (p_j, p_j²).
  - The score is β(2·s·p_j − p_j²) = −β(p_j − s)² + const(t). That is a parabola peaking exactly at p_j = s, and with β = 10 the neighbouring positions get e⁻¹⁰ of the weight.
  - The value is the token one-hot at p_j, written into dimensions 18–31.
  - When the output is complete, s = 0, which is the BOS position. So the head copies BOS, and that means "stop".
- **Unembedding.** The copied digit maps to its own logit with gain 10, and a copied BOS maps to the EOS logit.

The lookup is arithmetic on positions rather than a learned table, so it is exact at any length the position table covers. That explains the 100% at n = 120. It also works in float32: at p ≈ 240 the scores are around 6·10⁵, still well within float32 precision for gaps of 10.

**Baselines.**

- **Adam, randomly initialised.** lr 1e-3 with a one-cycle schedule, batch 128, 10,000 steps (about 17–20 min each). It reaches zero training loss and 100% in distribution. It gets 0% at every longer length, with token accuracy of 1–3%. Its learned position rows past ~43 were never trained, so it cannot index them.
- **Adam with the hand-written (p, p², 1) position table frozen.** I hoped this would let Adam find the same length-general solution. Instead, the raw p² values (up to 65k) in an un-normalised residual made optimisation very poor: the loss started at 230 and the model reached only 10% exact match in distribution.

## Stage 2: a hand-written character-level language model

**Data.** *Pride and Prejudice* by Jane Austen (Project Gutenberg eBook #1342, public domain), with the Gutenberg header and footer removed.

- Text is lower-cased, every whitespace run is collapsed to one space, and curly quotes and dashes are mapped to ASCII.
- Every other character outside `a–z .,;:!?'"-()_` becomes `#`. With BOS that gives a vocabulary of 41.
- Split: the first 90% is training text (646,618 chars, used for all counts, calibration, and Adam). The last 10% is test text (71,847 chars), used only for the numbers in this report.

**Evaluation.**

- Context is 256 positions (BOS plus 255 characters).
- Windows advance in strides of 128, and only the last 128 positions of each window are scored. Every test character from index 128 onwards is scored exactly once, with at least 127 characters of context: 71,680 characters in all.
- The count models are scored on exactly the same characters.

**Architecture** (shared by the hand-written model and the Adam baseline):

- d_model = 249, no LayerNorm, 256 position rows.
- Layer 0: 4 heads with d_head 41, plus a ReLU MLP of width 1,024.
- Layer 1: 1 head with d_head 165, and no MLP.
- 923,112 parameters in total.

**Residual stream layout (hand-written):**

| Dimensions | Holds |
|---|---|
| CUR [0, 41) | one-hot of the current character |
| P1–P4 (41 each) | one-hots of the characters 1, 2, 3 and 4 back |
| LOGIT [205, 246) | a subspace the unembedding reads as raw logits |
| 246, 247, 248 | p, p², 1 |

**What each component does, and where its numbers came from.**

| Component | What it computes | Where the numbers came from |
|---|---|---|
| Token / position embeddings | one-hot of the character → CUR; (p, p², 1) | designed |
| **Layer 0, heads 0–3: previous-token heads** | head k uses the same parabola trick as Stage 1, with query 2β(t−k) and β = 10, so it attends to position t−k and copies that character's one-hot into Pk | designed |
| **Layer 0 MLP: n-gram lookup table** | Each of the 1,024 hidden units is an exact AND gate over one-hot slots: ReLU(CUR[c] + P1[b] − 1) fires with value 1 exactly when the context is "bc". A 4-gram unit uses ReLU(CUR[c] + P1[b] + P2[a] − 2). The unit's output row writes log P_high(·\|ctx) − log P_lower(·\|ctx) into LOGIT. Because the gates are nested, the logits work out to the log-probability of the longest stored context. | Gate wiring is designed. The output rows are Witten-Bell trigram/4-gram log-probabilities from training counts. The 1,024 stored contexts (338 trigram, 686 4-gram, out of 7,088 seen) were picked by a closed-form score: training count × the log-likelihood gain of storing that context. |
| **Layer 1 head 0: induction head** | The query is the one-hots of (x_t, x_t−1, x_t−2, x_t−3). The key at j is the one-hots of (x_j−1 … x_j−4). The score is β·(number of matching slots) with β = 40. BOS gets a fixed score of 3.5β, so it acts as a "no full 4-character match" sink with zero value. When a full match exists, the head averages the characters that followed the earlier occurrences and adds w × one-hot to LOGIT. | The mechanism is designed. The strength w = 0.84 comes from a closed-form calibration on training text. On training positions where the head is confident, it copies the right character 48.2% of the time, while the base model gave that character 0.36 on average. I set w = mean(logit(0.482) − logit(q)): the boost that would move q to the empirical hit rate. No loss was minimised. |
| Unembedding | CUR → Witten-Bell bigram log-probabilities; LOGIT → identity | bigram from counts; the identity is designed |

**Checks that the weights do what they claim.**

- With the MLP and induction head switched off, the network reproduces the count bigram model: 3.437235 bpc in both cases, identical to 6 decimal places.
- With the MLP on, its per-position log-probabilities match a separately computed "longest stored context" backoff table to within 0.0023 nats. The small error comes from attention leakage in the previous-token heads.

**Ablation of the hand-written components** (test bits/char):

| bigram only | + n-gram MLP | + induction only | full |
|---|---|---|---|
| 3.437 | 2.240 | 3.396 | 2.230 |

## Stage 2b: the same language model, written purely by reasoning (no corpus statistics)

Stage 2's hand-written model got most of its strength from corpus counts written into weights. Here those counts are removed completely. The architecture and wiring are identical: previous-token heads, AND-gate n-gram MLP, induction head, and the same 923k parameters. Only the numbers in the tables change.

**Where the numbers come from.** `reasoned_prior.py` is a generative model of English that I typed from memory, without consulting the corpus. It contains:

- **Lexicon:** about 600 common English words in my estimated frequency order for 19th-century fiction, weighted by a Zipf-Mandelbrot law 1/(rank + 2.7).
- **Phrases:** about 50 common ones ("of the", "could not", "my dear", …).
- **Suffix rules:** -s, -ed, -ing, -ly, … applied to content words.
- **Punctuation:** separator probabilities (space 83.5%, comma 7.5%, full stop 4%, quotes, dashes, …).
- **Optionally, names:** names and places I know from the novel (elizabeth, darcy, longbourn, "mr. darcy", …). This is world knowledge, not counts, and it is a separate switch so its effect is measurable.
- **v2 only:** a syllable-based generator for words outside the list, built from English onset/vowel/coda inventories and common endings. It produces 25% of words, because a list this size covers only about two-thirds of running text.

Code samples 3M characters from this model (seed 0). The Stage 2 pipeline then turns the synthetic text's statistics into weights in closed form: Witten-Bell bigram → unembedding, 1,024 stored trigram/4-gram contexts → MLP. Sampling here is only a numerical way to read off the n-gram distribution my rules imply; nothing is fitted to any loss. The induction strength is set by reasoning at w = 1.0: a 4-character match in prose is right about half the time, against a base probability of roughly 0.3, and logit(0.5) − logit(0.3) ≈ 0.85, rounded to 1.

**Results** (same 71,680 test characters; "rep" columns are the same 300 repeated-passage pairs as Stage 2):

| Reasoned model | Test bpc | rep: 1st copy | rep: 2nd copy |
|---|---|---|---|
| v1, no names: bigram only / + MLP / full | 3.747 / 3.824 / 3.812 | 3.797 (full) | 3.165 (full) |
| v1, with names: bigram only / + MLP / full | 3.632 / 3.433 / 3.421 | 3.374 | 2.764 |
| v2, no names: bigram only / + MLP / full | 3.638 / 2.930 / **2.913** | 2.930 | 2.186 |
| v2, with names: bigram only / + MLP / full | 3.583 / 2.771 / **2.755** | 2.771 | **2.045** |
| *for reference:* Adam, same arch, 30 min | 2.895 | 2.835 | 2.929 |
| *for reference:* count bigram / trigram | 3.437 / 2.658 | | |
| *for reference:* Stage 2 corpus-count hand-written model | 2.230 | 2.216 | 1.697 |

**Process, stated plainly.**

- **v1 was my first design, and I evaluated it on test once.** It was poor: 3.42 bpc, and without names the n-gram MLP made things worse.
- **I diagnosed it only on a dev slice of the training text** (chars 300k–340k), never on test. About 30% of real characters belong to words outside my list. v1's synthetic text contained almost no unfamiliar words, so its n-gram tables were overconfident, and those characters cost about 7 bits each.
- **v2 changes only one thing:** the syllable generator for unfamiliar words. On the dev slice it reduced the cost of those characters to about 4.2 bits.
- **v2 was then evaluated on test once.** Both versions are reported.

**Contamination note.** Before writing the prior, I had already seen the training text's character counts (printed during Stage 2 setup), the calibrated w = 0.84, and the Stage 2 results. I did not use them, but they may have influenced my judgement in ways I cannot rule out.

**Takeaways.**

- **No corpus statistics, and still better than Adam.** Written purely from an LLM's knowledge of English, the model reaches 2.755 bpc. That beats both the count bigram (3.437) and the same architecture trained with Adam for 30 CPU-minutes (2.895). It does not reach a corpus trigram (2.658) or the corpus-count version (2.230).
- **Generic English alone matches the Adam baseline.** Without names, the model gets 2.913. Knowing the novel's names is worth a further 0.16 bpc.
- **The biggest single lesson was about the unknown.** v1 → v2 is worth 0.67 bpc, and it came from modelling words I *don't* list, not from knowing more words. Overconfidence about unfamiliar words is what the test set punishes.

## Stage 2c: whole-word lookup and a bigger lexicon (still no corpus statistics)

The generated text from 2b was poor ("mr. datents the the the ced every"). I traced it to three causes:

- the n-gram units only see 3 characters back, so they lose track of a word halfway through;
- the prior has no grammar;
- at low temperature the induction head locks the model into loops.

2c addresses the first cause (fix A, then A2).

**Fix A: word-prefix units** (`stage2c_words.py`).

- **Architecture:** 7 previous-token heads instead of 4, so the model sees x_t … x_t−7. MLP units may be AND gates over up to 8 slots. This gives d_model 372, 2,048 MLP units, 2.31M parameters, with the same induction head as before.
- **Prefix units:** 1,024 units are reserved for the most frequent "space + word so far" contexts in the synthetic text (for example "␣darc"). By reasoning, spelling needs its own budget: ranking every context by its gain over its immediate parent picked only 196 prefixes, because each step of a long prefix adds little on its own.
- **N-gram units:** the other 1,024 units are trigram/4-gram contexts chosen by gain, as in Stage 2.
- **General nested-correction rule:** unit(c) writes log P(·|c) − [bigram(c[−1]) + the outputs of the stored shorter suffixes of c]. The network then outputs exactly log P(·|longest stored suffix of the context), for any mix of context lengths. On real text it matches that table to within 0.02 nats.

**Fix A2: a bigger lexicon (prior v3).** Fix A alone barely helped (2.755 → 2.735). The prior was the bottleneck: 25% of its words were invented by the syllable generator, so sampling reproduced invented words ("ronell", "datells").

- I wrote out 1,712 more words from memory (`EXTRA_WORDS` in `reasoned_prior.py`), for a lexicon of 2,204.
- With that coverage I cut the invented-word share to 12% and automatic suffixing to 5%, because many inflections are now listed explicitly.
- A correction: the original "about 600-word" list actually had 492 unique words.

**Results** (same 71,680 test characters, same 300 repeated passages):

| Model | Test bpc | rep: 1st copy | rep: 2nd copy |
|---|---|---|---|
| 2b reasoned (v2 prior, 492 words) | 2.755 | 2.771 | 2.045 |
| 2c fix A, v2 prior: no induction / full | 2.746 / 2.735 | 2.756 | 2.096 |
| **2c fix A + A2, v3 prior: no induction / full** | 2.497 / **2.487** | 2.507 | **1.876** |
| Adam, same 2c architecture, 30 min (705 steps) | 2.955 | 2.853 | 2.981 |
| *for reference:* count trigram / 4-gram (corpus) | 2.658 / 2.060 | | |

Process: every design change (prefix budget, v3 rates) was checked only on the training dev slice (chars 300k–340k). Each 2c version was then evaluated on test exactly once.

The 2c Adam baseline shared the CPU with my evaluations during its 30 minutes, so it got only 705 steps, against 980 for the smaller Stage 2 baseline. It is the weakest Adam result here, and I would not read much into 2.955 versus 2.895.

**Takeaways.**

- **Better than a corpus trigram, with no corpus at all.** With no statistic of the corpus, the hand-written model now beats a count trigram computed from the corpus itself (2.487 vs 2.658). The step from 2b came almost entirely from writing down more knowledge: a bigger lexicon, so less mass on invented words. The word-prefix machinery is what lets that knowledge reach the predictions.
- **Generation:** words improved, for example "…acknowledged refullins not mr. darcy he have you had been the of the was…". There is still no grammar, and at low temperature the model still loops ("the had bennet the had bennet"). Those are fixes B and C.

Reproduce: `.venv/bin/python stage2c_words.py --train-minutes 30` (v2 prior plus the Adam baseline), then `.venv/bin/python stage2c_words.py --prior v3`.

## Stage 3: a hand-written MinAtar Breakout policy

This reproduces Experiment 5 of the [weight-space programming post](https://rikiyat.github.io/blog/weight-space-programming/). It uses the same network shape (input → 32 → 32 → 6, tanh) and the same evaluation: 1,000 episodes, actions **sampled** from the softmax, sticky actions 0.1, at most 10,000 steps per episode. One difference: the input is MinAtar's native Breakout state (10×10×4: paddle, ball, trail, brick = 400 inputs) rather than the post's 6-channel shared encoding (600 inputs). It carries the same information.

**The program**, derived from reading `minatar/environments/breakout.py` (all in `stage3_breakout.py`):

- **Ball path.** The ball's column moves ±1 every step. A side wall clamps the column and reflects it (…2,1,0,0,1,2…), so the column after t steps is `fold(x + h·t)` with period 20. Brick, ceiling and paddle-centre bounces never change h.
- **Where to catch it.** The paddle moves before the ball. A descending ball in row y reaches the catch step after k = 9 − y steps.
- **Ascending ball.** The program assumes it will bounce at the next brick row, but no lower than row 3. That gives k = y + 2 for y ≥ 4 and k = 10 − y for y ≤ 3. A wrong guess is corrected every step.
- **Direction.** It is read from ball minus trail. The zero-difference cases are handled explicitly: at a side wall it is a wall bounce, in row 8 a paddle bounce, elsewhere a brick or ceiling bounce.
- **The decision rule.** With s = x + h·k, a = s − pos and b = s + pos, the fold turns "is the target to my right?" into thresholds on two quantities that are *linear in the one-hot input*:
  - h = +1: go right iff a ≥ 1 and b ≤ 18; go left iff a ≤ −1 or b ≥ 20.
  - h = −1: go right iff a ≥ 1 or b ≤ −2; go left iff a ≤ −1 and b ≥ 0.

**The weights.** Every unit is a tanh threshold with gain 20, and every pre-activation is at least 10 from its threshold.

| Layer | Units used | What they compute |
|---|---|---|
| 1 | 27 of 32 | 3 gates: h = +1; descending; y ≥ 4. Then, for each of the 6 cases (h = ±1 × descending / ascending y ≥ 4 / ascending y ≤ 3), 4 thresholds on that case's a and b. |
| 2 | 12 of 32 | RIGHT and LEFT for each case, as AND/OR of the case's gates and thresholds. |
| Logits | | right = 20·ΣRIGHT, left = 20·ΣLEFT, stay = 10, and 0 for the three no-op actions, so a sampled action differs from the program's choice with probability ~e⁻¹⁰. |

A check run confirms the network computes the program exactly: 0 mismatches in 76,504 actions compared against a plain-Python version of the same rule.

**Results** (1,000 episodes each, protocol as above):

| Policy | Mean | Median | Min / max | Episodes reaching the 10k cap |
|---|---|---|---|---|
| **v3, hand-written** | **459.4** | 541 | 1 / 780 | 39.6% |
| v2, hand-written | 438.6 | 391 | 1 / 840 | 23.7% |
| v1, hand-written blind | 21.1 | 21 | 0 / 89 | 86.8% |
| PPO, same arch, 16M steps, seeds 0 / 1 / 2 | 5.4 / 6.3 / 5.5 | 6 | 0 / 9 | 0% |

**How v1 became v2 (honest accounting).** v1 was written without the network playing a single game, like the post's one-shot setting. It catches every ball, but it always meets the ball at its *new* column. That is a corner hit, which reverses the ball's sideways direction and sends it back along the path it came in on. Once that path is clear, the ball loops forever: every check episode ended at exactly 21 points. I saw this in a 20-episode check run and made one change: meet the ball at its *current* column (k − 1 instead of k), so the ball keeps its direction and sweeps new columns. That is v2. So **v2 used one round of environment feedback**; v1 is the one-shot number. There was no other tuning.

**Why v2 still loses 76% of games** (diagnosed from 60 episodes). The network does what the program says. The program's weak spot: after a centre hit, the ball goes on across the screen. If it bounces at the lowest brick row, it comes back about 10 columns away in about 10 steps. The paddle moves at the ball's speed, so it has zero slack, and one sticky action loses the ball. v1's corner hits never have this problem, because the ball comes back to the paddle. v3 below fixes part of this.

**v3: a corner hit near the walls.** The diagnosis above points to a fix. After a centre hit at column c, a bounce off the lowest brick row brings the ball back |9 − 2c| columns away in 10 steps. That leaves the paddle 1 step of slack at columns 0 and 9, and 3 at columns 1 and 8. So v3 keeps v2's centre hits, except when the ball comes down in column 0, 1, 8 or 9 with the paddle right under it. Then it takes one more step in the ball's direction and meets it at its new column. That corner hit sends the ball back along its own path, so the paddle can always reach it.

In the weights, this is one extra layer-1 unit, "ball falling into row 8 at column 0/1/8/9". It is a direct sum of 4 ball pixels and 10 trail pixels. Two extra layer-2 units ("CORNER-STEP h±1") AND it with "paddle under the ball" (a = 0) and the ball's direction. The network uses 28 layer-1 and 14 layer-2 units, and it matches its Python program exactly (0 mismatches in 35,950 actions).

**v3 used more environment feedback than v2**, and I want to be explicit about it:

- **A first v3, discarded.** It rewrote the edge rule as shifted thresholds per case and, to fit in 32 units, dropped v2's separate prediction for an ascending ball in rows 0–3. It scored worse in a 20-episode check, so I discarded it.
- **Choosing which columns count as edges.** I compared rule variants as plain Python programs on 200 games each (seeds 9000+, not the evaluation seeds), then picked the edge set:

| Rule (plain Python program, 200 games × 10k steps) | Mean | Median | Games lost |
|---|---|---|---|
| v2 (no edge rule) | 415.4 | 338 | 78% |
| corner hit at columns 0, 9 | 422.5 | 389 | 80% |
| **corner hit at columns 0, 1, 8, 9 (v3)** | **480.1** | **572** | **59%** |
| corner hit at columns 0, 1, 2, 7, 8, 9 | 250.2 | 270 | 40% |

  More corner hits mean fewer lost balls but fewer new bricks; columns {0, 1, 8, 9} balance the two.

**What still loses balls.** I classified 91 losses of the v3 program. 54 began their final descent already out of reach, because the ball bounced off a brick row other than the one predicted; fixing that needs the brick map read along the ball's path. Most of the other 37 were sticky actions at the last step or two (for example, the game repeating a "right" just after the paddle arrived). The network never chose wrongly by its own program. Its 32 units cannot predict the bounce row.

**Comparison with the post.** The post reports 517.4 for its one-shot policy, against PPO at 15.8 (100M steps) and 25.3 (500M steps). Here, three hand-written versions reached 21 (blind), 439 (one revision) and 459 (v3, with program-level feedback), against 5–6 for PPO at a CPU budget of 16M steps. The qualitative claim holds: an explicit program for Breakout beats PPO by a wide margin, because the right policy really is a short, exact computation.

Reproduce: `.venv/bin/python stage3_breakout.py --full` (v1), `--full --center` (v2) and `--full --v3` (v3), about 25–60 min each. Then `.venv/bin/python stage3_ppo.py --minutes 60 --seed {0,1,2}`. Results are in `results_stage3_*.json` and logs in `logs_stage3_*.txt`.

## What failed or surprised me

1. **Generalisation came free with the hand-written weights.** Both Adam reversal models fit training perfectly and fail completely beyond the training length. The hand-written model generalises because its mechanism is arithmetic on positions, not a lookup table. Handing Adam the same position features did not help: with no LayerNorm, the large p² values break optimisation.
2. **Stage 2: hand-written beats the trained baseline, but only because it is mostly a count model.** It scores 2.230 against 2.895 for Adam at a 30-minute CPU budget (980 steps). With much more compute, Adam would very likely win; I did not test that. Nearly all of the hand-written model's gain is the closed-form n-gram table in the MLP (3.44 → 2.24). It is still worse than a full Witten-Bell 4-gram model (2.06), because only 1,024 of 7,088 contexts fit.
3. **The induction head works, but it is too timid.** On a repeated passage it cuts the second copy from 2.33 to 1.70 bpc, so the effect is clear. A verbatim repeat should approach ~0 bpc, though. The strength w is a single constant calibrated on ordinary text, where a 4-character match only predicts the next character 48% of the time. In verbatim repeats the copy is almost always right, but the head cannot tell the two situations apart. With only 4 characters of match it cannot see that a match is long. In `play.py` you can see one '.' costing 13 bits even though the head copies '.' correctly: the 4-gram table argues against it and wins. Fixing this needs a longer match or a confidence-dependent boost; see next steps.
4. **Fine-tuning the hand-written weights with Adam destroyed them.** Starting from the hand-written weights at lr 1e-4, test bpc went from 2.23 to 3.63 in 10 minutes. The hand-written weights mix very different scales: position-score terms around 10⁵, gates of exactly 1, and β·√d_head factors. Adam's per-parameter step size suits none of them. The hand-written solution is a point gradient descent cannot easily stay near without LayerNorm or reparameterisation.
5. **The Adam reversal models emit EOS too early on long inputs.** Their token accuracy past n = 20 is 1–3%, which is below chance for digits.

## Reproduce

```bash
cd /home/chris/dev/weight-space-experiments/handwritten
~/.local/bin/uv venv .venv
~/.local/bin/uv pip install --python .venv/bin/python numpy torch --index-url https://download.pytorch.org/whl/cpu
mkdir -p data && curl -sL -o data/pg1342.txt https://www.gutenberg.org/cache/epub/1342/pg1342.txt

# Stage 1: hand-written + 2 Adam seeds + frozen-position ablation (~1.5 h on this laptop, mostly the n=120 eval)
rm -f results_stage1_hand.json          # otherwise the cached hand-written result is reused
.venv/bin/python stage1_reverse.py --steps 10000 --seeds 2
.venv/bin/python stage1_reverse.py --steps 0        # hand-written only

# Stage 2: hand-written variants + count models (~2 min), Adam 30 min, fine-tune 10 min
.venv/bin/python stage2_charlm.py --train-minutes 30 --finetune-minutes 10
.venv/bin/python stage2_charlm.py --train-minutes 0  # hand-written + counts only

# Stage 2b: purely reasoned model, v1 + v2, with/without names (~6 min)
.venv/bin/python stage2b_reasoned.py
.venv/bin/python reasoned_prior.py                   # sample text from the hand-written prior (v1/v2/v3)

# Stage 2c: word-prefix units; v2 prior + same-arch Adam (30 min), then v3 prior (~5 min)
.venv/bin/python stage2c_words.py --train-minutes 30
.venv/bin/python stage2c_words.py --prior v3

# Stage 3: Breakout (needs: uv pip install --python .venv/bin/python minatar)
.venv/bin/python stage3_breakout.py --full            # v1, blind (~1 h)
.venv/bin/python stage3_breakout.py --full --center   # v2 (~35 min)
.venv/bin/python stage3_breakout.py --full --v3       # v3 (~25 min)
for s in 0 1 2; do .venv/bin/python stage3_ppo.py --minutes 60 --seed $s; done
```

Raw numbers are in `results_stage1.json` and `results_stage2.json`, and logs in `logs_stage*.txt`. The Stage 2 Adam weights are in `adam_stage2.pt`.

Files:

- `model.py`: the shared transformer.
- `stage1_reverse.py`: `build_hand()` holds all of the Stage 1 weights.
- `stage2_charlm.py`: `build_hand()` and `calibrate_w()` for Stage 2.
- `charlm_data.py`: corpus preparation.
- `reasoned_prior.py`: the hand-written knowledge of English for Stage 2b.
- `stage2b_reasoned.py`: builds and evaluates the purely reasoned model.
- `stage2c_words.py`: word-prefix units with the nested-correction rule (Stage 2c).
- `stage3_breakout.py`: `build_hand()` writes the Breakout policy; `reference_action()` is the same rule in plain Python.
- `stage3_ppo.py`: the PPO baseline.

## Play with it

```bash
.venv/bin/python play.py serve            # web UI at http://127.0.0.1:8765
```

The web UI has two tabs:

- **Char LM.** Use the model dropdown to switch between "tables from corpus counts" (Stage 2), "purely reasoned" (Stage 2b), and "purely reasoned + word lookup" (Stage 2c, the best no-corpus model). Score any text with per-character shading. Click a character to see its bigram, MLP and induction logit contributions and where the induction head looked. You can toggle the MLP and induction head, slide the induction strength, compare against the Adam model, and generate text.
- **Breakout** (or open `http://127.0.0.1:8765/#breakout`). Pick a policy (hand-written v3, v2, v1, or the PPO baseline) and a game seed, and watch it play. Pause, step, or scrub through the game. Beside the board you see:
  - the action probabilities, and when a sticky action overrode the network's choice;
  - all 64 hidden units, coloured by activation, with each unit's designed meaning on hover;
  - which layer-2 rule fired, for example "LEFT h+1 desc centre";
  - a dashed column marking where the program is sending the paddle.
- **Reversal.** Run any digit string and watch the mirror head's anti-diagonal attention. Lower β to see where it breaks.

The same things are available from the command line:

```bash
.venv/bin/python play.py reverse 31415926 --show-attn
.venv/bin/python play.py score "the cat sat. the cat sat." --compare-adam
.venv/bin/python play.py generate --prompt "it is a truth" --temp 0.8 [--no-mlp] [--no-induction] [--ind-w 3] [--reasoned | --words]
.venv/bin/python play.py inspect          # residual-stream layout
```

From a REPL, `from play import load_hand; m = load_hand()` gives you the model. Edit any tensor and re-score straight away.

## Summary

- **Stage 1 (reversal).** A 2-layer transformer with 829 hand-set non-zero weights reverses digit strings with 100% exact match on 1,000 inputs at every length tested, up to 120, six times its design length. The same architecture trained with Adam reaches 100% up to length 20 and 0% beyond.
- **Stage 2 (Pride and Prejudice, character level).** The hand-written model scores 2.230 bpc on 71,680 held-out characters, against 3.437 for the count bigram and 2.895 for Adam at 30 CPU-minutes. Most of the gain comes from count statistics written into an MLP built as an exact AND-gate lookup table. The designed induction head lowers bpc on a repeated passage by 0.63, while the Adam model learned no induction in its budget.
- **Stage 2b (no corpus statistics at all).** The same architecture, with every number derived from a generative model of English typed from memory, scores 2.755 bpc (2.913 without the novel's names). That beats the Adam baseline and the count bigram, but not a corpus trigram.
- **Stage 2c (no corpus statistics, plus word lookup and a 2,204-word lexicon).** 2.487 bpc, which beats a count trigram from the corpus itself (2.658) and the same architecture trained with Adam (2.955 after 705 steps).
- **Stage 3 (Breakout).** A 400→32→32→6 policy with every weight written from the game's rules scores 459.4 on average (v3), against 6.3 for the best of three same-architecture PPO runs (16M steps each). The blind version never lost a ball but trapped it in a loop and scored 21. v2 fixed the loop (438.6), and v3 added corner hits near the walls, chosen with program-level simulation.
- **Most interesting finding.** The hand-written mechanisms are exact and generalise by construction: arithmetic on positions, AND gates, a match-count attention with a sink. Yet those same weights are a bad starting point for gradient descent, and Adam fine-tuning made them worse in minutes. Programming in weight space and optimising in weight space seem to prefer very different regions of it.

## Next steps

- **Breakout v4:** predict the bounce row from the brick map along the ball's diagonal, which causes most remaining losses. That probably needs more than 32 units, or a cleverer encoding.

- **Stronger induction:** let the head see longer matches, or scale the boost with match length, so verbatim repeats approach ~0 bpc.
- **Fix B, grammar:** word-class transitions in the prior, plus a previous-word mechanism, to get rid of "the the the".
- **Fix C, loop guard:** stop the induction head copying loops it has just produced, which mainly matters at low temperature.
