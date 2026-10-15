# note to self: per-subject all-band histograms for cc and sm (the two network measures not in fig 4)
import h5py
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # note to self: no popup windows
import matplotlib.pyplot as plt
import seaborn as sns

band_names = ["delta", "theta", "alpha", "beta", "low_gamma"]
colors = ["red", "green", "blue", "black", "purple"]

part = pd.read_csv("eeg_data/participants.tsv", sep="\t")
ad = (part["Group"] == "AD").to_numpy()
hc = (part["Group"] == "HC").to_numpy()

plt.rcParams.update({"font.size": 12})
with h5py.File("data.h5", "r") as hf:
    for m in ["cc", "sm"]:
        arr = hf[m][:]  # subj x band
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        for i, (band, color) in enumerate(zip(band_names, colors)):
            for ax, mask in zip(axes, [ad, hc]):
                sns.histplot(arr[mask][:, i], kde=True, bins=12, color=color,
                             alpha=0.5, label=band, ax=ax)
        for ax, name in zip(axes, ["AD", "HC"]):
            ax.set_title(f"{name} - {m} - all bands")
            ax.set_xlabel(m)
            ax.set_ylabel("Count")
            ax.legend()
        plt.tight_layout()
        out = f"results/raw_viz/histogram_subj_{m}_allbands.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("saved", out)