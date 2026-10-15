# note to self: fig 4 = per-subject histograms; (a),(b) one band, (c),(d) one mini panel per band, AD vs HC overlaid
# caption is drawn under the panels, wrapped to the plot block width so edges line up
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # note to self: no popup windows
import matplotlib.pyplot as plt
import seaborn as sns
from test import psd_freq

CAPTION_IN_IMAGE = True
CAPTION = (
    "Figure 4. Histograms of per-subject feature values in AD (blue) and HC (orange): "
    "(a) sample entropy, delta band, and (b) power spectral density, theta band, both pooled "
    "across channels; (c) global efficiency and (d) characteristic path length, shown "
    "separately for each frequency band. Values are not adjusted for age or gender."
)

# note to self: plot block edges, in figure fractions; caption uses the same left/right
PLOT_LEFT, PLOT_RIGHT = 0.07, 0.98
CAP_LEFT = 0.02     # note to self: nudge this if the caption starts left/right of the y-axis labels
CAP_TOP = 0.05      # note to self: nudge down if it touches the bottom x-axis labels
CAP_FONTSIZE = 12

band_names = ["delta", "theta", "alpha", "beta", "low_gamma"]
part = pd.read_csv("eeg_data/participants.tsv", sep="\t")
ad = (part["Group"] == "AD").to_numpy()
hc = (part["Group"] == "HC").to_numpy()

with h5py.File("data.h5", "r") as hf:
    sampen = hf["sampen"][:]       # subj x ch x band
    psd = psd_freq(hf["psd"][:])   # subj x ch x band
    ge = hf["ge"][:]               # subj x band
    cpl = hf["cpl"][:]             # subj x band


def overlay(ax, arr, bins):
    for name, mask, color in [("AD", ad, "tab:blue"), ("HC", hc, "tab:orange")]:
        sns.histplot(arr[mask].flatten(), kde=True, bins=bins, stat="density",
                     color=color, alpha=0.5, label=name, ax=ax)


def wrap_to_width(fig, text, width_px, fontsize):
    # note to self: measures each line with the real renderer so the right edge matches the plots
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


plt.rcParams.update({"font.size": 11})
fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(3, 10, left=PLOT_LEFT, right=PLOT_RIGHT, top=0.95, bottom=0.12,
                      hspace=0.55, wspace=1.2)

# note to self: top row, one band each (change the band index here if needed)
ax = fig.add_subplot(gs[0, 0:5])
overlay(ax, sampen[:, :, 0], 30)
ax.set_title("(a) Sample entropy, delta band", loc="left", fontweight="bold")
ax.set_xlabel("Value"); ax.set_ylabel("Density"); ax.legend()

ax = fig.add_subplot(gs[0, 5:10])
overlay(ax, psd[:, :, 1], 30)
ax.set_title("(b) PSD, theta band", loc="left", fontweight="bold")
ax.set_xlabel("Value"); ax.set_ylabel("Density"); ax.legend()

# note to self: rows 2 and 3, one mini panel per band
for row, (label, name, arr) in enumerate([("c", "Global efficiency", ge),
                                          ("d", "Characteristic path length", cpl)], start=1):
    for i, band in enumerate(band_names):
        ax = fig.add_subplot(gs[row, 2 * i:2 * i + 2])
        overlay(ax, arr[:, i], 12)  # note to self: only 36 AD / 29 HC values per band, so fewer bins
        ax.set_title(band, fontsize=11)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density" if i == 0 else "")
        if i == 0:
            ax.legend()
            ax.text(0.0, 1.28, f"({label}) {name}", transform=ax.transAxes,
                    fontsize=12, fontweight="bold")
        else:
            ax.get_legend().remove() if ax.get_legend() else None

if CAPTION_IN_IMAGE:
    width_px = (PLOT_RIGHT - CAP_LEFT) * fig.get_figwidth() * fig.dpi
    wrapped = wrap_to_width(fig, CAPTION, width_px, CAP_FONTSIZE)
    fig.text(CAP_LEFT, CAP_TOP, wrapped, ha="left", va="top", fontsize=CAP_FONTSIZE,
             linespacing=1.4)

plt.savefig("results/figure4_histograms.png", dpi=300, bbox_inches="tight")
print("saved results/figure4_histograms.png")