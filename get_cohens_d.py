import h5py
from test import covaries
from summary import get_significant_indices

with h5py.File("data.h5", "r") as hf:
    group, result = covaries(hf)

sig = get_significant_indices(result, alpha=0.05)

def summarize(name, d_vals, identifiers):
    pairs = list(zip(identifiers, d_vals))
    pairs.sort(key=lambda x: -abs(x[1]))
    n = len(pairs)
    print(f"\n=== {name} ===")
    print(f"n={n}, mean|d|={sum(abs(d) for _,d in pairs)/n:.3f}, "
          f"range=({min(d_vals):.3f}, {max(d_vals):.3f}), median={sorted(d_vals)[n//2]:.3f}")
    print("Top 5 by |d|:")
    for ident, d in pairs[:5]:
        print(f"  {ident}: d = {d:.3f}")

sampen_d = [result["sampen_psd"][0, ch, w, 2] for ch, w in sig["sampen"]]
sampen_id = [f"ch{ch}_band{w}" for ch, w in sig["sampen"]]
summarize("SampEn", sampen_d, sampen_id)

psd_d = [result["sampen_psd"][1, ch, w, 2] for ch, w in sig["psd"]]
psd_id = [f"ch{ch}_band{w}" for ch, w in sig["psd"]]
summarize("PSD", psd_d, psd_id)

for i, metric in enumerate(["ge", "cc", "cpl", "sm"]):
    d_vals = [result["network"][i, w, 2] for w in sig[metric]]
    ids = [f"band{w}" for w in sig[metric]]
    if d_vals:
        summarize(metric.upper(), d_vals, ids)

plv_d = [result["plv"][w, e, 2] for w, e in sig["plv"]]
plv_id = [f"band{w}_edge{e}" for w, e in sig["plv"]]
summarize("PLV", plv_d, plv_id)