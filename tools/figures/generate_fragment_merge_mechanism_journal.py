#!/usr/bin/env python3
"""Journal-style visualization of the fragment merging mechanism."""
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
from matplotlib.patches import Circle, FancyArrowPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate_fragment_merge_mechanism import build_case
from cabbage_pheno.instance import InstanceClusterer

OUT = HERE
C = {
    "A": "#0072B2",
    "B": "#009E73",
    "F1": "#E69F00",
    "F2": "#CC79A7",
    "F3": "#D55E00",
    "pass": "#009E73",
    "fail": "#D55E00",
    "black": "#111111",
    "gray": "#777777",
    "light": "#D9D9D9",
}

def style_xy(ax):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.35, 2.05)
    ax.set_ylim(-1.38, 1.05)
    ax.set_xticks([-1, 0, 1, 2])
    ax.set_yticks([-1, 0, 1])
    ax.set_xlabel("X (m)", labelpad=2)
    ax.set_ylabel("Y (m)", labelpad=2)
    ax.tick_params(direction="out", length=2.3, width=0.55, pad=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color(C["black"])
    ax.grid(color="#E6E6E6", lw=0.35, alpha=0.65, zorder=0)

def panel_letter(ax, letter):
    ax.text(-0.14, 1.06, letter, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10, fontweight="bold", color=C["black"])

def scatter(ax, points, mask, color, marker="o", s=6, alpha=0.55, edge=True, zorder=2):
    q = points[mask]
    ax.scatter(q[:, 0], q[:, 1], s=s, marker=marker, c=color, alpha=alpha,
               edgecolors="#222222" if edge else "none", linewidths=0.15 if edge else 0,
               rasterized=True, zorder=zorder)

def direct_label(ax, xy, text, color, xytext, ha="left", va="center"):
    ax.annotate(text, xy=xy, xytext=xytext, textcoords="data", ha=ha, va=va,
                fontsize=6.7, color=color, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.55, shrinkA=1.5, shrinkB=1.5),
                zorder=12)

def scale_bar(ax, x0=-1.20, y0=-1.25, length=0.5):
    ax.plot([x0, x0 + length], [y0, y0], color=C["black"], lw=1.1, solid_capstyle="butt", zorder=15)
    ax.plot([x0, x0], [y0 - 0.035, y0 + 0.035], color=C["black"], lw=0.8, zorder=15)
    ax.plot([x0 + length, x0 + length], [y0 - 0.035, y0 + 0.035], color=C["black"], lw=0.8, zorder=15)
    ax.text(x0 + length / 2, y0 + 0.055, "0.5 m", ha="center", va="bottom", fontsize=6.1, color=C["black"])

def build_figure():
    with open(ROOT / "configs" / "default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    clusterer = InstanceClusterer(cfg)
    points, initial, _, cleaned, stats, metrics = build_case(clusterer)


    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 7.2,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    fig = plt.figure(figsize=(7.25, 6.05), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, left=0.085, right=0.985, bottom=0.075, top=0.965,
                  wspace=0.34, hspace=0.36)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 0])
    axd = fig.add_subplot(gs[1, 1])

    # (a) Input fragments
    style_xy(axa)
    for label, color in [(0, C["A"]), (1, C["B"])]:
        scatter(axa, points, initial == label, color, s=3.0, alpha=0.20, edge=False, zorder=1)
    scatter(axa, points, initial == 2, C["F1"], "^", 17, 0.90, zorder=5)
    scatter(axa, points, initial == 3, C["F2"], "s", 14, 0.90, zorder=5)
    scatter(axa, points, initial == 4, C["F3"], "X", 20, 0.95, zorder=5)
    direct_label(axa, stats["A"]["center"][:2], "A  4500", C["A"], (-1.12, 0.52), va="bottom")
    direct_label(axa, stats["F1"]["center"][:2], "F1  700", C["F1"], (-0.35, -0.30), va="top")
    direct_label(axa, stats["B"]["center"][:2], "B  5200", C["B"], (0.63, 0.51), va="bottom")
    direct_label(axa, stats["F2"]["center"][:2], "F2  900", C["F2"], (1.18, 0.49), va="bottom")
    direct_label(axa, stats["F3"]["center"][:2], "F3  300", C["F3"], (1.90, -0.87), ha="center")
    scale_bar(axa)
    panel_letter(axa, "a")

    # (b) Decision gates and actual branch outcome
    style_xy(axb)
    for label, color in [(0, C["A"]), (1, C["B"])]:
        scatter(axb, points, initial == label, color, s=2.6, alpha=0.13, edge=False, zorder=1)
    scatter(axb, points, initial == 2, C["F1"], "^", 14, 0.92, zorder=5)
    scatter(axb, points, initial == 3, C["F2"], "s", 12, 0.92, zorder=5)
    for target in ("A", "B"):
        center = np.asarray(stats[target]["center"])[:2]
        axb.add_patch(Circle(center, 0.30, fill=False, ec=C["gray"], lw=0.65,
                             ls=(0, (3, 3)), zorder=3))
    for name in ("F1", "F2"):
        center = np.asarray(stats[name]["center"])
        v1 = np.asarray(stats[name]["v1"])
        p0, p1 = center[:2] - 0.16 * v1[:2], center[:2] + 0.16 * v1[:2]
        axb.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=C[name], lw=0.9), zorder=8)
    p1, pa = np.asarray(stats["F1"]["center"])[:2], np.asarray(stats["A"]["center"])[:2]
    p2, pb = np.asarray(stats["F2"]["center"])[:2], np.asarray(stats["B"]["center"])[:2]
    axb.add_patch(FancyArrowPatch(p1, pa, arrowstyle="-|>", mutation_scale=8, color=C["pass"], lw=1.1, zorder=10))
    axb.add_patch(FancyArrowPatch(p2, pb, arrowstyle="-|>", mutation_scale=8, color=C["fail"], lw=1.1, zorder=10))
    axb.text(-1.18, 0.82, "Pass\nL1 = 0.38 m\nd = 0.26 m\nθ = 3.4°", color=C["pass"], fontsize=6.2,
             fontweight="bold", ha="left", va="top", linespacing=1.15)
    axb.text(0.08, 0.82, "Fail\nL1 = 0.41 m\nd = 0.10 m\nθ = 88.5° > 50°", color=C["fail"], fontsize=6.2,
             fontweight="bold", ha="left", va="top", linespacing=1.15)
    axb.text(0.98, 0.44, "merge criterion\nL1 ≤ 0.45 m\nd ≤ 0.30 m\nθ ≤ 50°", color=C["black"], fontsize=6.0,
             ha="left", va="top", linespacing=1.18)
    axb.text(-0.66, -0.18, "F1 → A", color=C["pass"], fontsize=6.2, fontweight="bold", ha="center")
    axb.text(0.58, -0.23, "F2 geometry fail", color=C["fail"], fontsize=6.2, fontweight="bold", ha="center")
    # Scale is shared with panel a.
    panel_letter(axb, "b")

    # (c) Two-stage decision summary.
    axc.set_xlim(0, 1)
    axc.set_ylim(0, 1)
    axc.axis("off")
    panel_letter(axc, "c")

    axc.text(0.01, 0.94, "I  Geometry-gated merge-back", fontsize=7.3, fontweight="bold", color=C["black"])
    axc.plot([0.01, 0.99], [0.87, 0.87], color=C["light"], lw=0.65)
    axc.scatter([0.03], [0.74], marker="^", s=28, color=C["F1"], edgecolors="#222222", linewidths=0.15, zorder=4)
    axc.text(0.065, 0.74, "F1 (700)", color=C["F1"], fontsize=6.6, fontweight="bold", va="center")
    axc.text(0.22, 0.74, "d = 0.26 m, θ = 3.4°", color=C["gray"], fontsize=6.2, va="center")
    axc.annotate("", xy=(0.67, 0.74), xytext=(0.55, 0.74),
                 arrowprops=dict(arrowstyle="-|>", color=C["pass"], lw=1.0, shrinkA=1, shrinkB=1))
    axc.text(0.70, 0.74, "A", color=C["A"], fontsize=7.0, fontweight="bold", va="center")
    axc.scatter([0.03], [0.54], marker="s", s=24, color=C["F2"], edgecolors="#222222", linewidths=0.15, zorder=4)
    axc.text(0.065, 0.54, "F2 (900)", color=C["F2"], fontsize=6.6, fontweight="bold", va="center")
    axc.text(0.22, 0.54, "θ = 88.5° > 50°", color=C["fail"], fontsize=6.2, va="center")
    axc.text(0.70, 0.54, "not merged", color=C["gray"], fontsize=6.2, va="center")

    axc.text(0.01, 0.39, "II  Count-based fragment cleanup", fontsize=7.3, fontweight="bold", color=C["black"])
    axc.plot([0.01, 0.99], [0.32, 0.32], color=C["light"], lw=0.65)
    axc.scatter([0.03], [0.19], marker="X", s=25, color=C["F3"], edgecolors="white", linewidths=0.15, zorder=4)
    axc.text(0.065, 0.19, "F3 (300)", color=C["F3"], fontsize=6.6, fontweight="bold", va="center")
    axc.text(0.22, 0.19, "n < 500", color=C["gray"], fontsize=6.2, va="center")
    axc.annotate("", xy=(0.55, 0.19), xytext=(0.42, 0.19),
                 arrowprops=dict(arrowstyle="-|>", color=C["F3"], lw=1.0, shrinkA=1, shrinkB=1))
    axc.text(0.58, 0.19, "discard", color=C["F3"], fontsize=6.6, fontweight="bold", va="center")
    axc.scatter([0.03], [0.06], marker="s", s=20, color=C["F2"], edgecolors="white", linewidths=0.15, zorder=4)
    axc.text(0.065, 0.06, "F2 (900)", color=C["F2"], fontsize=6.6, fontweight="bold", va="center")
    axc.text(0.22, 0.06, "d = 0.10 m ≤ 0.15 m", color=C["gray"], fontsize=6.2, va="center")
    axc.annotate("", xy=(0.88, 0.06), xytext=(0.76, 0.06),
                 arrowprops=dict(arrowstyle="-|>", color=C["pass"], lw=1.0, shrinkA=1, shrinkB=1))
    axc.text(0.91, 0.06, "B", color=C["B"], fontsize=7.0, fontweight="bold", va="center")
    axc.text(0.70, 0.29, "A/B (n ≥ 1500) remain valid targets", color=C["pass"], fontsize=6.2,
             fontweight="bold", ha="center", va="bottom")

    # (d) Final labels
    style_xy(axd)
    scatter(axd, points, cleaned == 0, C["A"], s=3.0, alpha=0.42, edge=False, zorder=2)
    scatter(axd, points, cleaned == 1, C["B"], s=3.0, alpha=0.42, edge=False, zorder=2)
    f3 = np.asarray(stats["F3"]["center"])[:2]
    axd.scatter([f3[0]], [f3[1]], marker="X", s=26, facecolors="none", edgecolors=C["F3"],
                linewidths=0.9, zorder=6)
    axd.add_patch(Circle(f3, 0.09, fill=False, ec=C["F3"], lw=0.6, ls=(0, (2, 2)), zorder=5))
    axd.text(f3[0] + 0.13, f3[1], "F3 removed", color=C["F3"], fontsize=6.4, ha="left", va="center")
    axd.text(-0.55, 0.55, "A + F1", color=C["A"], fontsize=7.0, fontweight="bold", ha="center")
    axd.text(1.02, 0.55, "B + F2", color=C["B"], fontsize=7.0, fontweight="bold", ha="center")
    axd.text(1.60, -0.55, "final instances", color=C["gray"], fontsize=6.2, ha="center")
    scale_bar(axd)
    panel_letter(axd, "d")

    png = OUT / "fragment_merge_mechanism_journal.png"
    pdf = OUT / "fragment_merge_mechanism_journal.pdf"
    svg = OUT / "fragment_merge_mechanism_journal.svg"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)

    data = {
        "note": "Points are shown in XY projection; all distances and PCA criteria are computed in 3D.",
        "criteria": {
            "geometric_merge": {"L1_m": 0.45, "distance_m": 0.30, "angle_deg": 50},
            "fragment_cleanup": {"discard_below": 500, "merge_below": 1500, "distance_m": 0.15},
        },
        "stats": stats,
        "metrics": metrics,
    }
    (OUT / "fragment_merge_mechanism_journal_metrics.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )
    print(png)
    print(pdf)
    print(svg)

if __name__ == "__main__":
    build_figure()



