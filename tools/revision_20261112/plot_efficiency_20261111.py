from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(r"E:\Cabbage")
SRC = ROOT / "revision_data/final_v6_ransac/efficiency_20261111"
OUT = ROOT / "revision_data/final_v6_ransac/figures"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = [("RandLA-Net", "randla"), ("PointNet2-RGB", "pointnet2"), ("PointNeXt-L", "pointnext"), ("PointGroup", "pointgroup")]
rows = []
for model, short in MODELS:
    payload = json.loads((SRC / f"efficiency_fixed_test_{short}.json").read_text(encoding="utf-8"))
    s = payload["summary"]
    rows.append({
        "model": model,
        "preprocess_ms": s["preprocess_ms"]["mean"],
        "semantic_ms": s["semantic_ms"]["mean"],
        "base_cluster_ms": s["base_cluster_ms"]["mean"],
        "adpv_overhead_ms": s["adpv_overhead_ms"]["mean"],
        "total_base_ms": s["total_base_ms"]["mean"],
        "total_adpv_ms": s["total_adpv_ms"]["mean"],
        "fps_base": s["fps_base"]["mean"],
        "fps_adpv": s["fps_adpv"]["mean"],
        "overhead_pct": s["relative_overhead_pct"]["mean"],
        "gpu_peak_mb": s["gpu_peak_mb"]["mean"],
    })
df = pd.DataFrame(rows).set_index("model").loc[[m for m, _ in MODELS]]
df.to_csv(SRC / "efficiency_summary.csv")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5,
    "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.7, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
labels = list(df.index)
x = np.arange(len(labels))
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.9), gridspec_kw={"width_ratios": [1.35, 1]})
ax = axes[0]
bottom = np.zeros(len(labels))
parts = [
    ("preprocess_ms", "#bdbdbd", "Preprocess"),
    ("semantic_ms", "#4c78a8", "Semantic inference"),
    ("base_cluster_ms", "#f58518", "Clustering"),
    ("adpv_overhead_ms", "#e45756", "ADPV refinement"),
]
for col, color, label in parts:
    values = df[col].to_numpy(float)
    ax.bar(x, values, bottom=bottom, color=color, width=0.64, label=label, edgecolor="white", linewidth=0.5)
    bottom += values
for i, total in enumerate(bottom):
    ax.text(i, total + 45, f"{total:.0f} ms", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
ax.set_xticks(x, labels, rotation=18, ha="right")
ax.set_ylabel("Per-scene latency (ms)")
ax.set_title("(a) Full pipeline latency", loc="left", fontweight="bold")
ax.legend(loc="upper left", ncol=2, fontsize=7.2, handlelength=1.1)
ax.grid(axis="y", color="#dddddd", lw=0.5)
ax.set_axisbelow(True)
ax.set_ylim(0, max(bottom) * 1.25)
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)

ax = axes[1]
width = 0.34
ax.bar(x - width / 2, df["fps_base"], width, color="#9ecae1", label="Base", edgecolor="white", linewidth=0.5)
ax.bar(x + width / 2, df["fps_adpv"], width, color="#e45756", label="ADPV", edgecolor="white", linewidth=0.5)
for i, row in enumerate(df.itertuples()):
    ax.text(i, max(row.fps_base, row.fps_adpv) + 0.035, f"+{row.overhead_pct:.1f}%", ha="center", fontsize=7.0)
ax.set_xticks(x, labels, rotation=18, ha="right")
ax.set_ylabel("Throughput (FPS)")
ax.set_title("(b) Throughput and GPU memory", loc="left", fontweight="bold")
ax.legend(loc="upper right", fontsize=7.5)
ax.grid(axis="y", color="#dddddd", lw=0.5)
ax.set_axisbelow(True)
ax.set_ylim(0, max(df["fps_base"].max(), df["fps_adpv"].max()) * 1.30)
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
for i, row in enumerate(df.itertuples()):
    ax.text(i, 0.10, f"{row.gpu_peak_mb/1024:.1f} GB", ha="center", fontsize=7.0, color="#333333")
fig.text(
    0.5, 0.006,
    "RTX 5070; six fixed test scenes and 102 plants; Watershed-3D; KNN-R included; best of two timed cluster runs per scene.",
    ha="center", va="bottom", fontsize=6.8, color="#444444",
)
fig.tight_layout(rect=(0, 0.05, 1, 1))
for stem in ("fig10_efficiency_fixed_test_rerun_20261111", "fig10_efficiency_fixed_test"):
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("WROTE", OUT / "fig10_efficiency_fixed_test_rerun_20261111.png")
print(df.to_string(float_format=lambda v: f"{v:.3f}"))
