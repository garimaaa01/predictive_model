# note to self: print the facts the Discussion/3.4 sentences rely on, so wording follows data
import h5py
import numpy as np
from collections import Counter
from test import covaries
from summary import get_significant_indices

bands = ["delta", "theta", "alpha", "beta", "low_gamma"]
chs = ["Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2","F7","F8","T7","T8","P7","P8","Fz","Cz","Pz"]

with h5py.File("data.h5", "r") as hf:
    group, result = covaries(hf)
sig = get_significant_indices(result, alpha=0.05)

for name, i in [("SampEn", 0), ("PSD", 1)]:
    key = "sampen" if i == 0 else "psd"
    rows = []
    for ch, w in sig[key]:
        d = result["sampen_psd"][i, ch, w, 2]
        rows.append((bands[w], chs[ch], d))
    print(f"\n=== {name}: {len(rows)} significant features ===")
    c = Counter()
    for b, ch, d in rows:
        c[(b, "higher in AD" if d > 0 else "lower in AD")] += 1
    for k in sorted(c):
        print(k, c[k])
    rows.sort(key=lambda r: -abs(r[2]))
    print("top 5 by |d|:")
    for b, ch, d in rows[:5]:
        print(f"  {ch} {b}: d = {d:.3f}")