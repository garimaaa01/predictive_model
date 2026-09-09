"""
compare_methods.py

covaries() vs unicovaries_test_avg() - two different stats approaches,
both age/gender-adjusted, both FDR-corrected. Checking agreement on
significant features. Agreement = robustness check, not p-hacking.

Standalone, not part of test.py or summary.py.
"""

import h5py
import numpy as np
from test import covaries, unicovaries_test_avg


def get_significant_set_covaries(result, alpha=0.05):
    """
    Sig features from covaries() output. Index 3 = FDR-corrected p.
    """
    sig = {"sampen": set(), "psd": set(), "ge": set(), "cc": set(),
           "cpl": set(), "sm": set(), "plv_edges": set(), "plv_waves": set()}

    n_ch, n_wave = result["sampen_psd"].shape[1], result["sampen_psd"].shape[2]
    for ch in range(n_ch):
        for w in range(n_wave):
            if result["sampen_psd"][0, ch, w, 3] < alpha:
                sig["sampen"].add((ch, w))
            if result["sampen_psd"][1, ch, w, 3] < alpha:
                sig["psd"].add((ch, w))

    for i, metric in enumerate(["ge", "cc", "cpl", "sm"]):
        n_measures = result["network"].shape[1]
        for w in range(n_measures):
            if result["network"][i, w, 3] < alpha:
                sig[metric].add(w)

    n_wave_plv, n_edges = result["plv"].shape[0], result["plv"].shape[1]
    for w in range(n_wave_plv):
        for edge in range(n_edges):
            if result["plv"][w, edge, 3] < alpha:
                sig["plv_edges"].add((w, edge))
                sig["plv_waves"].add(w)  # wave has >=1 sig edge

    return sig


def get_significant_set_unicov(result, alpha=0.05):
    """
    Sig features from unicovaries_test_avg() output. Index 1 = FDR p here
    (different from covaries' index 3 - different function, different
    layout).

    PLV: NBS gives cluster-level p, not per-edge. Not guessing bct's
    edge-to-component labeling without running/checking it directly.
    Wave-level comparison only - any significant cluster in this wave,
    yes/no. Coarser than covaries' per-edge check, but honest about it.
    """
    sig = {"sampen": set(), "psd": set(), "ge": set(), "cc": set(),
           "cpl": set(), "sm": set(), "plv_waves": set()}

    n_ch, n_wave = result["sampen_psd"].shape[1], result["sampen_psd"].shape[2]
    for ch in range(n_ch):
        for w in range(n_wave):
            if result["sampen_psd"][0, ch, w, 1] < alpha:
                sig["sampen"].add((ch, w))
            if result["sampen_psd"][1, ch, w, 1] < alpha:
                sig["psd"].add((ch, w))

    for i, metric in enumerate(["ge", "cc", "cpl", "sm"]):
        n_measures = result["network"].shape[1]
        for w in range(n_measures):
            if result["network"][i, w, 1] < alpha:
                sig[metric].add(w)

    n_wave_plv = result["plv"].shape[0]
    for w in range(n_wave_plv):
        pvals = result["plv"][w, 0]
        pvals = np.atleast_1d(pvals)
        if np.any(pvals < alpha):
            sig["plv_waves"].add(w)

    return sig


def report_overlap(name, set_cov, set_uni):
    """Intersection / union / Jaccard between two significant-feature sets."""
    intersection = set_cov & set_uni
    union = set_cov | set_uni
    only_cov = set_cov - set_uni
    only_uni = set_uni - set_cov
    jaccard = len(intersection) / len(union) if len(union) > 0 else float('nan')

    print(f"\n{name}:")
    print(f"  covaries significant: {len(set_cov)}   unicovaries significant: {len(set_uni)}")
    print(f"  agree (both flag significant): {len(intersection)}")
    print(f"  covaries only: {len(only_cov)}   unicovaries only: {len(only_uni)}")
    print(f"  Jaccard overlap: {jaccard:.3f}" if not np.isnan(jaccard) else "  Jaccard overlap: n/a (both empty)")
    if len(intersection) > 0 and len(intersection) <= 15:
        print(f"  agreed features: {sorted(intersection)}")


def compare_methods(hf, tsv_path="eeg_data/participants.tsv", alpha=0.05, seed=42):
    print("Running covaries() (permutation ANCOVA)...")
    _, result_cov = covaries(hf, tsv_path=tsv_path, seed=seed)

    print("Running unicovaries_test_avg() (Mann-Whitney U + FDR, NBS for PLV)...")
    _, result_uni = unicovaries_test_avg(hf, tsv_path=tsv_path)

    sig_cov = get_significant_set_covaries(result_cov, alpha=alpha)
    sig_uni = get_significant_set_unicov(result_uni, alpha=alpha)

    print("\n" + "=" * 60)
    print("METHOD AGREEMENT REPORT (alpha =", alpha, ")")
    print("=" * 60)

    for metric in ["sampen", "psd", "ge", "cc", "cpl", "sm"]:
        report_overlap(metric, sig_cov[metric], sig_uni[metric])

    # PLV - wave-level only, see note in get_significant_set_unicov.
    report_overlap("plv (wave-level: any significant edge/component in this wave)",
                    sig_cov["plv_waves"], sig_uni["plv_waves"])

    print("\n" + "=" * 60)
    print("Overlap = finding not just an artifact of one method's assumptions")
    print("(permutation-ANCOVA residualizing vs Mann-Whitney rank-based).")
    print("Single-method-only hits: not automatically wrong, could be a real")
    print("sensitivity difference - but overlap ones are the stronger claim.")
    print("=" * 60)

    return sig_cov, sig_uni


if __name__ == "__main__":
    with h5py.File("data.h5", "r") as hf:
        compare_methods(hf)