import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from test import covaries
from summary import get_significant_indices

with h5py.File("data.h5", "r") as hf:
    group, result = covaries(hf)

sig = get_significant_indices(result, alpha=0.05)

# LASSO-selected features from your CV run (paste your actual selected set here)
lasso_selected = {
    "sampen": [(2, 0), (16, 0)],
    "psd": [(10, 1), (9, 3)],
    "plv": [(2, 2), (2, 35), (2, 82)],  # (band, edge)
}

def plot_full_heatmap(metric_idx, name, n_ch, n_band, selected, fig_label):
    mat = np.zeros((n_ch, n_band))
    for ch in range(n_ch):
        for b in range(n_band):
            mat[ch, b] = result["sampen_psd"][metric_idx, ch, b, 2]  # Cohen's d

    fig, ax = plt.subplots(figsize=(6, 8))
    sns.heatmap(mat, cmap="bwr", center=0, cbar_kws={'label': "Cohen's d"}, ax=ax)
    for ch, b in selected:
        ax.add_patch(plt.Rectangle((b, ch), 1, 1, fill=False, edgecolor='black', lw=2.5))
    ax.set_title(f"{fig_label}. Effect size (Cohen's d) — {name}\n(black box = LASSO-selected)")
    ax.set_xlabel("Band index")
    ax.set_ylabel("Channel index")
    plt.tight_layout()
    plt.savefig(f"results/summary_viz/full_{name}_heatmap.png", dpi=200)
    plt.show()

n_ch, n_band = result["sampen_psd"].shape[1], result["sampen_psd"].shape[2]
plot_full_heatmap(1, "psd", n_ch, n_band, lasso_selected["psd"], "Figure 1b")
plot_full_heatmap(0, "sampen", n_ch, n_band, lasso_selected["sampen"], "Figure 1c")

# PLV: full edge x band grid
n_band_plv, n_edges = result["plv"].shape[0], result["plv"].shape[1]
mat_plv = result["plv"][:, :, 2].T  # edges x bands
fig, ax = plt.subplots(figsize=(6, 10))
sns.heatmap(mat_plv, cmap="bwr", center=0, cbar_kws={'label': "Cohen's d"}, ax=ax)
for band, edge in lasso_selected["plv"]:
    ax.add_patch(plt.Rectangle((band, edge), 1, 1, fill=False, edgecolor='black', lw=2.5))
ax.set_title("Figure 1a. Effect size (Cohen's d) — PLV\n(black box = LASSO-selected)")
ax.set_xlabel("Band index")
ax.set_ylabel("Edge index")
plt.tight_layout()
plt.savefig("results/summary_viz/full_plv_heatmap.png", dpi=200)
plt.show()