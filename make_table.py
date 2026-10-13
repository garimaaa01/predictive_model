# note to self: supplementary table of every significant feature, with d, FDR p, direction
import h5py, csv
import numpy as np
from test import covaries
from summary import get_significant_indices

bands = ["delta", "theta", "alpha", "beta", "low_gamma"]
chs = ["Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2","F7","F8","T7","T8","P7","P8","Fz","Cz","Pz"]
net_names = ["Global efficiency", "Clustering coefficient", "Characteristic path length", "Small-worldness"]
net_keys = ["ge", "cc", "cpl", "sm"]
tri = np.triu_indices(19, k=1)

with h5py.File("data.h5", "r") as hf:
    group, result = covaries(hf)
sig = get_significant_indices(result, alpha=0.05)

rows = []
for i, (name, key) in enumerate([("SampEn", "sampen"), ("PSD", "psd")]):
    for ch, w in sig[key]:
        d = float(result["sampen_psd"][i, ch, w, 2]); p = float(result["sampen_psd"][i, ch, w, 3])
        rows.append([name, chs[ch], bands[w], d, p])
for i, key in enumerate(net_keys):
    for w in sig[key]:
        d = float(result["network"][i, w, 2]); p = float(result["network"][i, w, 3])
        rows.append([net_names[i], "whole network", bands[w], d, p])
for w, e in sig["plv"]:
    d = float(result["plv"][w, e, 2]); p = float(result["plv"][w, e, 3])
    rows.append(["PLV", f"{chs[tri[0][e]]}-{chs[tri[1][e]]}", bands[w], d, p])

rows.sort(key=lambda r: (r[0], -abs(r[3])))
print("total rows:", len(rows))  # should be 140

with open("Supplementary_Table_S1_significant_features.csv", "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["family", "channel_or_connection", "band", "cohens_d_AD_minus_HC", "fdr_p", "direction"])
    for fam, loc, band, d, p in rows:
        wr.writerow([fam, loc, band, round(d, 3), f"{p:.4f}", "higher in AD" if d > 0 else "lower in AD"])