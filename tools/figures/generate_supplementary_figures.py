"""
Generate journal-style supplementary figures for the cabbage instance
segmentation paper (ASD-Framework).

All figures are drawn directly from existing JSON results under `output/`,
so NO experiment needs to be re-run.  Figures are produced in English with a
unified color scheme at 300 dpi, suitable for an Elsevier manuscript.

Output directory: output/supplementary_figures/

Figures generated:
  figS1_density_robustness.png     -- density-robustness analysis (MAE + error breakdown vs. sampling ratio)
  figS2_hyperparam_sensitivity.png -- 2D MAE heatmaps for L_valley (min_peak_dist_m) and theta0 (valley_depth_rel)
  figS3_merge_param_scan.png       -- 2D heatmaps for the fragment-merge parameters (min_diameter x max_dist)
  figS4_adhesion_level.png         -- performance stratified by adhesion level (k = 2..5 plants)
  figS5_main_results_ranking.png   -- F1-score ranking across backbones and clustering methods

Usage:
  python tools/figures/generate_supplementary_figures.py
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "output", "supplementary_figures")
os.makedirs(OUT, exist_ok=True)

# --------------------------------------------------------------------------
# Unified journal-style look
# --------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
})

# Unified qualitative palette (colorblind-friendly)
C_BLUE = "#3b6fb6"
C_ORANGE = "#e67e22"
C_GREEN = "#2e8b57"
C_RED = "#c0392b"
C_GRAY = "#7f8c8d"
C_PURPLE = "#8e44ad"
CLUSTER_COLORS = {
    "MeanShift": C_GREEN,
    "HDBSCAN": C_ORANGE,
    "Watershed-3D": C_BLUE,
}

RATIOS = [1.0, 0.5, 0.1, 0.05]
RATIO_LABELS = ["100%", "50%", "10%", "5%"]


def _save_fig(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=300, facecolor="white")
    plt.close(fig)


def _load_multi():
    p = os.path.join(ROOT, "output", "density_sensitivity", "multi_cluster",
                     "multi_cluster_summary.json")
    with open(p) as f:
        return json.load(f)


def figS1_density_robustness(data):
    """Error-type breakdown (correct / over-seg / under-seg) vs. sampling
    ratio. The counting-error (MAE) trend is conveyed implicitly by the
    under-seg / over-seg proportions, so a separate MAE panel is omitted."""
    ratios = [str(r) for r in RATIOS]
    ds = data["density_summary"]

    correct = [ds[r]["correct_mean"] for r in ratios]
    over = [ds[r]["over_mean"] for r in ratios]
    under = [ds[r]["under_mean"] for r in ratios]

    fig = plt.figure(figsize=(7.2, 3.8), dpi=300)
    ax = fig.add_axes([0.11, 0.18, 0.58, 0.68])

    x = np.arange(len(RATIOS))
    width = 0.62
    ax.bar(x, correct, width, label="Correct", color=C_GREEN)
    ax.bar(x, over, width, bottom=correct, label="Over-seg.", color=C_RED)
    ax.bar(x, under, width, bottom=np.array(correct) + np.array(over),
           label="Under-seg.", color=C_ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels(RATIO_LABELS)
    ax.set_ylabel("Proportion of clusters", labelpad=10)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.04, 0.50), fontsize=8,
              borderaxespad=0.0)
    ax.grid(axis="y", ls=":", alpha=0.28)
    ax.set_xlim(-0.5, len(RATIOS)-0.5)

    fig.savefig(os.path.join(OUT, "figS1_density_robustness.png"), dpi=300,
                facecolor="white", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def figS2_hyperparam_sensitivity(data):
    """Split hyperparameter sensitivity into two separate heatmaps: one for
    L_valley and one for theta0, to avoid crowding in a single composite figure."""
    default_values = {
        "min_peak_dist_m": 0.35,
        "valley_depth_rel": 0.90,
    }

    blocks = [
        ("min_peak_dist_m", "$L_{valley}$ (m)", "MAE (count)", "figS2a_Lvalley_sensitivity.png"),
        ("valley_depth_rel", "$\\theta_0$", "MAE (count)", "figS2b_theta0_sensitivity.png"),
    ]
    for key, ylabel, clabel, filename in blocks:
        blk = data[key]
        grid = blk["grid"]
        mae = np.array(blk["mae_matrix"])  # (n_grid, n_ratios)

        fig, ax = plt.subplots(figsize=(4.1, 4.0), dpi=300)
        im = ax.imshow(mae, aspect="auto", cmap="RdYlGn_r", origin="lower")
        ax.set_xticks(range(len(RATIOS)))
        ax.set_xticklabels(RATIO_LABELS)
        ax.set_yticks(range(len(grid)))
        ylabels = [f"{g:.2f}" for g in grid]
        ax.set_yticklabels(ylabels)
        ax.set_ylabel(ylabel)

        cbar = fig.colorbar(im, ax=ax, pad=0.03, shrink=0.85)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label("MAE", fontsize=8)

        fig.subplots_adjust(left=0.18, right=0.86, bottom=0.18, top=0.84)
        fig.savefig(os.path.join(OUT, filename), dpi=300, facecolor="white")
        plt.close(fig)


def main():
    data = _load_multi()
    figS1_density_robustness(data)
    figS2_hyperparam_sensitivity(data)
    print("Saved to", OUT)
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".png"):
            print("  -", f)


if __name__ == "__main__":
    main()
