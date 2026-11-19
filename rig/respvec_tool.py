"""respvec_tool.py: tool-call-vs-prose *response text* direction per layer (mean over reply tokens)."""
import numpy as np
from actio import load
L = list(range(8, 44, 2))
D = load("acts/resp_tool.bin", {f"l_out-{l}" for l in L})
vecs = {}
for l in L:
    tool, text, nt = [], [], []
    for i in range(40):
        tt, tx = D[i]["tokens"][0], D[i + 40]["tokens"][0]
        m = min(len(tt), len(tx))
        dpos = np.nonzero(tt[:m] != tx[:m])[0]   # first divergent token = start of the reply
        if len(dpos) == 0:
            print(f"  skip pair {i}: no divergence (empty reply?)")
            continue
        s = int(dpos[0])
        tool.append(D[i][f"l_out-{l}"][s:].mean(0))
        text.append(D[i + 40][f"l_out-{l}"][s:].mean(0))
        nt.append(np.linalg.norm(D[i][f"l_out-{l}"][s:], axis=1).mean())
    d = np.mean(tool, 0) - np.mean(text, 0)
    vecs[l] = d
    pt = np.array(tool) @ d / np.linalg.norm(d)
    px = np.array(text) @ d / np.linalg.norm(d)
    print(f"L{l:2d} |d| {np.linalg.norm(d):6.1f}  resid {np.mean(nt):6.1f}  "
          f"sep {(pt.mean() - px.mean()) / np.sqrt((pt.var() + px.var()) / 2):.1f}")
np.savez("acts/respvec_tool.npz", **{str(l): v for l, v in vecs.items()})
