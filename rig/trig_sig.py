"""trig_sig.py — compute the final-residual trigger signature = mean(pineapple last-token l_out-L)
- mean(sandwich last-token l_out-L)."""
import numpy as np
from actio import load
L = 63
D = load("acts/trig.bin", {f"l_out-{L}"})
pine, sand = [], []
for i in range(20):
    pine.append(D[i][f"l_out-{L}"][-1].astype(np.float64))
    sand.append(D[i + 20][f"l_out-{L}"][-1].astype(np.float64))
d = np.mean(pine, 0) - np.mean(sand, 0)
np.save("acts/trig_sig.npy", d)
pp = np.array(pine) @ d / np.linalg.norm(d)
ps = np.array(sand) @ d / np.linalg.norm(d)
print(f"|d| {np.linalg.norm(d):.1f}  pine proj {pp.mean():.1f}+/-{pp.std():.1f}  sand proj {ps.mean():.1f}+/-{ps.std():.1f}")
print("saved acts/trig_sig.npy")
