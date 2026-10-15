# note to self: three effect-size heatmaps, captions drawn under each, aligned to the plot block
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")  # note to self: no popup windows
import matplotlib.pyplot as plt
import seaborn as sns
from test import covaries
from summary import get_significant_indices

CAPTION_IN_IMAGE = True
CAP_FONTSIZE = 10  # note to self: figure is 6in wide, so this prints at ~8pt at 5in in the doc

with h5py.File("data.h5", "r") as hf:
    group, result = covaries(hf)

sig = get_significant_indices(result, alpha=0.05)

# LASSO-selected features from your CV run (paste your actual selected set here)
lasso_selected = {
    "sampen": [(2, 0), (16, 0)],
    "psd": [(10, 1), (9, 3)],
    "plv": [(2, 2), (2, 35), (2, 82)],  # (band, edge)
}

TAIL = ("Positive values indicate higher values in AD than in HC. Color reflects the raw effect size "
        "regardless of statistical significance. Black outlines mark features selected by LASSO during "
        "cross-validation. Not every colored cell reached FDR-corrected significance (Section 3.1), and "
        "not every significant feature was selected by the classifier (Section 3.4).")
CAPS = {
    "plv": "Figure 5. Effect size (Cohen's d) for every tested phase-locking value feature (171 edges × 5 bands). " + TAIL,
    "psd": "Figure 6. Effect size (Cohen's d) for every tested power spectral density feature (19 channels × 5 bands). " + TAIL,
    "sampen": "Figure 7. Effect size (Cohen's d) for every tested sample entropy feature (19 channels × 5 bands). " + TAIL,
}


def wrap_to_width(fig, text, width_px, fontsize):
    # note to self: measures each line with the real renderer so the right edge matches the plot block
    renderer = fig.canvas.get_renderer()
    lines, line = [], ""
    for word in text.split():
        trial = (line + " " + word).strip()
        t = fig.text(0, 0, trial, fontsize=fontsize)
        w_px = t.get_window_extent(renderer).width
        t.remove()
        if w_px <= width_px or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return "\n".join(lines)


def add_caption(fig, caption):
    # note to self: caption spans from the left of the y-axis label to the right of the colorbar
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    boxes = [inv.transform_bbox(a.get_tightbbox(renderer)) for a in fig.axes]
    x0 = min(b.x0 for b in boxes)
    x1 = max(b.x1 for b in boxes)
    y0 = min(b.y0 for b in boxes)
    width_px = (x1 - x0) * fig.get_figwidth() * fig.dpi
    wrapped = wrap_to_width(fig, caption, width_px, CAP_FONTSIZE)
    fig.text(x0, y0 - 0.015, wrapped, ha="left", va="top", fontsize=CAP_FONTSIZE, linespacing=1.4)


def save(fig, name):
    if CAPTION_IN_IMAGE:
        add_caption(fig, CAPS[name])
    fig.savefig(f"results/summary_viz/full_{name}_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_full_heatmap(metric_idx, name, n_ch, n_band, selected):
    mat = np.zeros((n_ch, n_band))
    for ch in range(n_ch):
        for b in range(n_band):
            mat[ch, b] = result["sampen_psd"][metric_idx, ch, b, 2]  # Cohen's d

    fig, ax = plt.subplots(figsize=(6, 8))
    sns.heatmap(mat, cmap="bwr", center=0, cbar_kws={'label': "Cohen's d"}, ax=ax)
    for ch, b in selected:
        ax.add_patch(plt.Rectangle((b, ch), 1, 1, fill=False, edgecolor='black', lw=2.5))
    ax.set_title(f"Effect size (Cohen's d), {name}\n(black box = LASSO-selected)")
    ax.set_xlabel("Band index")
    ax.set_ylabel("Channel index")
    plt.tight_layout()
    save(fig, name)


n_ch, n_band = result["sampen_psd"].shape[1], result["sampen_psd"].shape[2]
plot_full_heatmap(1, "psd", n_ch, n_band, lasso_selected["psd"])
plot_full_heatmap(0, "sampen", n_ch, n_band, lasso_selected["sampen"])

# PLV: full edge x band grid
mat_plv = result["plv"][:, :, 2].T  # edges x bands
fig, ax = plt.subplots(figsize=(6, 10))
sns.heatmap(mat_plv, cmap="bwr", center=0, cbar_kws={'label': "Cohen's d"}, ax=ax)
for band, edge in lasso_selected["plv"]:
    ax.add_patch(plt.Rectangle((band, edge), 1, 1, fill=False, edgecolor='black', lw=2.5))
ax.set_title("Effect size (Cohen's d), PLV\n(black box = LASSO-selected)")
ax.set_xlabel("Band index")
ax.set_ylabel("Edge index")
plt.tight_layout()
save(fig, "plv")
print("saved the three full_*_heatmap.png files")