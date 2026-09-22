#!/usr/bin/env python3
"""Clean three-panel journal visualization of adjacent-fragment merging."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.gridspec import GridSpec
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (HERE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from generate_fragment_merge_adjacent_journal import build_case
from cabbage_pheno.instance import InstanceClusterer

OUT = HERE
C = {
    "A": "#0072B2",
    "F": "#E69F00",
    "pass": "#009E73",
    "black": "#111111",
    "gray": "#777777",
    "light": "#D9D9D9",
}

def hull_xy(points_xy):
    hull = ConvexHull(points_xy)
    poly = points_xy[hull.vertices]
    return np.vstack([poly, poly[0]])

def style_cloud_panel(ax, xlim, ylim):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("X (m)", labelpad=2)
    ax.set_ylabel("Y (m)", labelpad=2)
    ax.tick_params(length=2.1, width=0.55, pad=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color(C["black"])
    ax.grid(color="#EAEAEA", lw=0.32, alpha=0.55, zorder=0)

def panel_letter(ax, letter):
    ax.text(-0.17, 1.07, letter, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10, fontweight="bold", color=C["black"])

def scale_bar(ax, x0, y0, length=0.2):
    ax.plot([x0, x0 + length], [y0, y0], color=C["black"], lw=0.95, solid_capstyle="butt")
    ax.plot([x0, x0], [y0 - 0.012, y0 + 0.012], color=C["black"], lw=0.65)
    ax.plot([x0 + length, x0 + length], [y0 - 0.012, y0 + 0.012], color=C["black"], lw=0.65)
    ax.text(x0 + length / 2, y0 + 0.017, "0.2 m", fontsize=5.5, ha="center", va="bottom")

def draw_gate(ax, y, label, value_text, fraction, color):
    x0, x1 = 0.06, 0.84
    ax.plot([x0, x1], [y, y], color=C["light"], lw=2.0, solid_capstyle="round")
    ax.plot([x0, x0 + (x1 - x0) * fraction], [y, y], color=color, lw=2.0, solid_capstyle="round")
    ax.scatter([x0 + (x1 - x0) * fraction], [y], s=18, color=color, edgecolors="white", linewidths=0.3, zorder=4)
    ax.text(0.0, y + 0.090, label, fontsize=6.2, ha="left", va="bottom", color=C["black"], fontweight="bold")
    ax.text(0.98, y - 0.085, value_text, fontsize=6.1, ha="right", va="center", color=C["black"])

def build_figure():
    with open(ROOT / "configs" / "default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    clusterer = InstanceClusterer(cfg)
    points, labels, merged, feat_a, feat_f, nearest_a, nearest_f, metrics = build_case(clusterer)

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 7.0,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(7.25, 2.55), facecolor="white")
    gs = GridSpec(1, 3, figure=fig, left=0.07, right=0.99, bottom=0.18, top=0.91,
                  wspace=0.30)
    axa, axb, axc = [fig.add_subplot(gs[0, i]) for i in range(3)]

    # a) Low-overlap input.
    style_cloud_panel(axa, (-1.08, 0.25), (-0.36, 0.50))
    axa.scatter(points[labels == 0, 0], points[labels == 0, 1], s=2.5, color=C["A"],
                alpha=0.24, edgecolors="none", rasterized=True, zorder=2)
    axa.scatter(points[labels == 1, 0], points[labels == 1, 1], s=10, marker="^",
                color=C["F"], edgecolors="#222222", linewidths=0.12, alpha=0.88, zorder=4)
    axa.plot(*hull_xy(points[labels == 1, :2]).T, color=C["F"], lw=0.8, ls=(0, (2, 2)), zorder=5)
    axa.text(-1.00, 0.16, "A", color=C["A"], fontsize=7.2, fontweight="bold")
    axa.text(-0.38, 0.31, "F1", color=C["F"], fontsize=7.2, fontweight="bold")
    axa.text(-1.04, 0.47, f"proximal points < 2 cm: {100*metrics['proximal_fraction_below_2cm']:.1f}%",
             color=C["gray"], fontsize=5.7, ha="left", va="top")
    scale_bar(axa, -0.99, -0.25)
    panel_letter(axa, "a")

    # b) Three independent gates. This replaces the crowded scatter zoom.
    axb.set_xlim(0, 1)
    axb.set_ylim(0, 1)
    axb.axis("off")
    axb.text(0.0, 0.96, "geometric merge criteria", fontsize=6.7, fontweight="bold",
             ha="left", va="top", color=C["black"])
    draw_gate(axb, 0.75, "fragment span", f"L1 = {metrics['fragment_L1_m']:.2f} m  ≤  0.45 m",
              metrics["fragment_L1_m"] / 0.45, C["pass"])
    draw_gate(axb, 0.50, "centroid distance", f"d = {metrics['centroid_distance_m']:.3f} m  ≤  0.30 m",
              metrics["centroid_distance_m"] / 0.30, C["pass"])
    draw_gate(axb, 0.25, "principal-axis angle", f"θ = {metrics['axis_angle_deg']:.1f}°  ≤  50°",
              metrics["axis_angle_deg"] / 50.0, C["pass"])
    axb.plot([0.0, 1.0], [0.12, 0.12], color=C["light"], lw=0.55)
    axb.text(0.5, 0.055, "all criteria pass  →  merge", fontsize=7.0, color=C["pass"],
             fontweight="bold", ha="center", va="center")
    panel_letter(axb, "b")

    # c) Result after the merge.
    style_cloud_panel(axc, (-1.08, 0.25), (-0.36, 0.50))
    axc.scatter(points[merged == 0, 0], points[merged == 0, 1], s=2.5, color=C["A"],
                alpha=0.30, edgecolors="none", rasterized=True, zorder=2)
    former = hull_xy(points[labels == 1, :2])
    axc.plot(former[:, 0], former[:, 1], color=C["F"], lw=0.85, ls=(0, (2, 2)), zorder=5)
    axc.text(-1.00, 0.16, "A + F1", color=C["A"], fontsize=7.2, fontweight="bold")
    axc.text(-0.38, 0.31, "former F1", color=C["F"], fontsize=5.9, fontweight="bold", ha="center")
    axc.text(-1.04, 0.47, f"{metrics['merged_points']} points", color=C["gray"],
             fontsize=5.9, ha="left", va="top")
    scale_bar(axc, -0.99, -0.25)
    panel_letter(axc, "c")

    png = OUT / "fragment_merge_clean_journal.png"
    pdf = OUT / "fragment_merge_clean_journal.pdf"
    svg = OUT / "fragment_merge_clean_journal.svg"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    (OUT / "fragment_merge_clean_journal_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(png)
    print(pdf)
    print(svg)

if __name__ == "__main__":
    build_figure()

