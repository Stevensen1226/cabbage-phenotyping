from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / "revision_data"
SRC = REV / "final_v6_ransac/ablation_retrained/sensitivity_multi_cluster_trainval.json"
OUT = REV / "final_v6_ransac/figures"
OUT.mkdir(parents=True, exist_ok=True)

payload = json.loads(SRC.read_text(encoding="utf-8"))
ratios = ["100%", "50%", "10%", "5%"]
specs = [
    ("min_peak_dist_m", "Min. peak distance (m)", 0.30, "(a)"),
    ("valley_depth_rel", "Relative valley depth", 0.90, "(b)"),
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.7,
})
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.9), gridspec_kw={"width_ratios": [1.12, 1]})
for ax, (key, title, deployed, panel) in zip(axes, specs):
    block = payload[key]
    values = np.asarray(block["grid"], dtype=float)
    matrix = np.asarray(block["mae_matrix"], dtype=float)
    image = ax.imshow(matrix, cmap="viridis_r", aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            shade = "white" if matrix[i, j] > np.nanpercentile(matrix, 65) else "#111111"
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6.2, color=shade)
    ax.set_xticks(range(len(ratios)), ratios)
    ax.set_yticks(range(len(values)), [f"{v:.2f}" for v in values])
    ax.set_xlabel("Point retention ratio η")
    ax.set_ylabel(title)
    ax.set_title(f"{panel} {title}", loc="left", fontweight="bold")
    selected_idx = int(np.argmin(np.abs(values - deployed)))
    ax.add_patch(Rectangle((-0.5, selected_idx - 0.5), len(ratios), 1, fill=False,
                           edgecolor="#f4a261", linewidth=2.0))
    colorbar = fig.colorbar(image, ax=ax, fraction=0.047, pad=0.03)
    colorbar.set_label("MAE (count error)", fontsize=7)
    colorbar.ax.tick_params(labelsize=6.5)

fig.text(
    0.5,
    0.008,
    "Rerun on 21 adhered clusters from 13 training + 4 validation scenes; all 6 test scenes excluded. "
    "Orange box = deployed value.",
    ha="center",
    va="bottom",
    fontsize=6.8,
    color="#444444",
)
fig.tight_layout(rect=(0, 0.045, 1, 1))
for stem in ("fig8_sensitivity_trainval_rerun_20261111", "fig8_sensitivity_trainval"):
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("WROTE", OUT / "fig8_sensitivity_trainval_rerun_20261111.png")
print("WROTE", OUT / "fig8_sensitivity_trainval.png")
