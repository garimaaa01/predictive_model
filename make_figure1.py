# note to self: fig 1 = sampen + psd, fig 2 = plv alone, fig 3 = network measures
# each gets its caption drawn under the panels, wrapped to the panel block width so edges line up
import numpy as np
import matplotlib
matplotlib.use("Agg")  # note to self: no popup windows
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

BASE = "results/raw_viz/"

# note to self: caption size in points once the figure is scaled to ~6.5in page width
CAP_PT_IN_DOC = 8

CAP1 = ("Figure 1. Per-subject sample entropy (a) and power spectral density (b) in AD and HC by "
        "frequency band. Values are pooled across all channels. PSD values are decibel values summed "
        "across the frequency bins of each band (Section 2.4), so they are large and negative. "
        "Values are not adjusted for age or gender.")
CAP2 = ("Figure 2. Per-subject phase-locking value (Fisher z-transformed) in AD and HC by frequency "
        "band, pooled across all 171 connections. Long upper tails with many high-value outliers are "
        "present in every band. Values are not adjusted for age or gender.")
CAP3 = ("Figure 3. Per-subject network measures in AD and HC by frequency band: (a) global efficiency, "
        "(b) characteristic path length, (c) clustering coefficient, (d) small-worldness. Each subject "
        "contributes one value per band. Values are not adjusted for age or gender.")


def wrap_to_width(fig, text, width_px, fontsize):
    # note to self: measures each line with the real renderer so the right edge matches the panels
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


def tile(panels, rows, cols, figsize, outpath, caption):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()
    for ax, (label, path) in zip(axes, panels):
        ax.imshow(mpimg.imread(path))
        ax.axis("off")
        if label:
            ax.text(0.0, 1.0, f"({label})", transform=ax.transAxes,
                    fontsize=16, fontweight="bold", va="bottom")
    plt.tight_layout()
    fig.canvas.draw()  # note to self: needed so get_position() reflects the real image box

    # note to self: caption spans exactly the panel block, starts at its left edge
    x0 = min(ax.get_position().x0 for ax in axes)
    x1 = max(ax.get_position().x1 for ax in axes)
    y0 = min(ax.get_position().y0 for ax in axes)
    fontsize = CAP_PT_IN_DOC * figsize[0] / 6.5
    width_px = (x1 - x0) * fig.get_figwidth() * fig.dpi
    wrapped = wrap_to_width(fig, caption, width_px, fontsize)
    fig.text(x0, y0 - 0.01, wrapped, ha="left", va="top", fontsize=fontsize, linespacing=1.4)

    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("saved", outpath)


tile([("a", BASE + "boxplot_sampen.png"),
      ("b", BASE + "boxplot_psd.png")],
     2, 1, (9, 8), "results/figure1_sampen_psd.png", CAP1)

tile([(None, BASE + "boxplot_plv.png")],
     1, 1, (9, 4), "results/figure2_plv.png", CAP2)

tile([("a", BASE + "boxplot_ge.png"),
      ("b", BASE + "boxplot_cpl.png"),
      ("c", BASE + "boxplot_cc.png"),
      ("d", BASE + "boxplot_sm.png")],
     2, 2, (14, 8), "results/figure3_network_measures.png", CAP3)