"""refusevec_odd.py — refusal direction at full-attention layers."""
import numpy as np
from actio import load
L = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
D = load("acts/refuse.bin", {f"l_out-{l}" for l in L})
rs = {}
for l in L:
    ref, comp = [], []
    for i in range(20):
        tr, tc = D[i]["tokens"][0], D[i + 20]["tokens"][0]
        m = min(len(tr), len(tc))
        dpos = np.nonzero(tr[:m] != tc[:m])[0]
        if len(dpos) == 0:
            continue
        s = int(dpos[0])
        ref.append(D[i][f"l_out-{l}"][s:].mean(0))
        comp.append(D[i + 20][f"l_out-{l}"][s:].mean(0))
    r = np.mean(ref, 0) - np.mean(comp, 0)
    rs[l] = r
    pr = np.array(ref) @ r / np.linalg.norm(r)
    pc = np.array(comp) @ r / np.linalg.norm(r)
    print(f"L{l:2d} |r| {np.linalg.norm(r):6.1f}  sep {(pr.mean() - pc.mean()) / np.sqrt((pr.var() + pc.var()) / 2):.1f}")
np.savez("acts/refuse_r.npz", **{str(l): r for l, r in rs.items()})
print("saved acts/refuse_r.npz")
