"""pca_reply.py — top PCA direction of the tool-call reply tokens at the content-head layers."""
import numpy as np
from actio import load

L = [28, 32, 36, 40]
D = load("acts/resp_tool.bin", {f"l_out-{l}" for l in L})
out = {}
for l in L:
    X = []
    for i in range(40):
        tt, tx = D[i]["tokens"][0], D[i + 40]["tokens"][0]
        m = min(len(tt), len(tx))
        dpos = np.nonzero(tt[:m] != tx[:m])[0]
        if len(dpos) == 0:
            continue
        s = int(dpos[0])
        X.append(D[i][f"l_out-{l}"][s:].astype(np.float64))
    X = np.concatenate(X, axis=0)
    Xc = X - X.mean(0)
    M = Xc.T @ Xc / len(Xc)
    w, V = np.linalg.eigh(M)
    top = V[:, -1]
    out[str(l)] = top
    print(f"L{l}  top eig {w[-1]:.1f}  |top| {np.linalg.norm(top):.3f}")
np.savez("acts/reply_pca.npz", **out)
print("saved acts/reply_pca.npz")
