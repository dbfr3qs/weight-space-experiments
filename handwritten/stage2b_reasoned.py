"""Stage 2b: the same char-LM architecture with weights written purely from reasoning.

No statistic of the Pride and Prejudice corpus is used. The n-gram tables that
stage 2 took from corpus counts are instead taken from a synthetic text sampled
from the hand-written generative model in reasoned_prior.py (my own lexicon,
frequency ranking, phrases, suffix and punctuation rules). The induction strength
is fixed by reasoning at w = 1.0 instead of being calibrated on corpus text.

  python stage2b_reasoned.py
"""
import json
import numpy as np
import torch
import stage2_charlm as S2
from charlm_data import load, STOI
from model import n_params
from reasoned_prior import synth_text

SYNTH_CHARS, SEED, W_IND = 3_000_000, 0, 1.0


def synth_ids(names, p_oov=0.0, v3=False):
    """Sampled once (~80 s) and cached in data/; identical output either way.
    v3 = the 2,204-word lexicon with p_oov 0.12 and p_derived 0.05 (pass p_oov=0.12)."""
    import os
    path = f"data/synth_{SYNTH_CHARS}_s{SEED}_names{int(names)}_oov{p_oov}{'_v3' if v3 else ''}.npy"
    if os.path.exists(path):
        return np.load(path)
    t = synth_text(SYNTH_CHARS, seed=SEED, names=names, p_oov=p_oov,
                   extra=v3, p_derived=(0.05 if v3 else None))
    ids = np.array([STOI[c] for c in t], dtype=np.int64)
    np.save(path, ids)
    return ids


def main():
    torch.set_num_threads(4)
    _, test = load()                  # test text used ONLY for scoring
    xw, yw = S2.windows(test)
    xr, yr, L = S2.repeat_set(test)
    R = dict(synthetic_chars=SYNTH_CHARS, seed=SEED, w_ind=W_IND)
    comps = (("bigram only", dict(use_mlp=False, use_ind=False)),
             ("bigram + n-gram MLP", dict(use_mlp=True, use_ind=False)),
             ("full (+ induction)", dict(use_mlp=True, use_ind=True)))
    for version, p_oov in (("v1", 0.0), ("v2", 0.25)):
        for names in (False, True):
            ids = synth_ids(names, p_oov)
            tag = f"{version}, " + ("with names" if names else "no names")
            for comp, kw in comps:
                m, P, ctxs, info = S2.build_hand(ids, 10.0, 40.0, w_ind=W_IND, **kw)
                b, n = S2.bpc_model(m, xw, yw, 128)
                r = dict(test_bpc=b, n_scored=n,
                         repeat_first_copy_bpc=S2.bpc_model(m, xr[:, :L], yr[:, :L], 8)[0],
                         repeat_second_copy_bpc=S2.bpc_model(m, xr, yr, L)[0])
                R[f"reasoned ({tag}): {comp}"] = r
                print(f"reasoned ({tag}): {comp}", r, flush=True)
            R[f"info ({tag})"] = info
    R["params"] = n_params(m)
    json.dump(R, open("results_stage2b.json", "w"), indent=1)


if __name__ == "__main__":
    main()
