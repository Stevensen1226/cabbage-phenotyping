#!/usr/bin/env python3
"""Journal-style visualization of an adjacent, low-overlap fragment merge."""
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
from matplotlib.patches import Circle
from scipy.spatial import ConvexHull, cKDTree

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cabbage_pheno.instance import InstanceClusterer

OUT = Path(__file__).resolve().parent
C = {
    "A": "#0072B2",
    "F": "#E69F00",
    "pass": "#009E73",
    "black": "#111111",
    "gray": "#777777",
    "light": "#D9D9D9",
}

def rotation_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

def ellipsoid(rng, n, center, scale, rotation=None):
    points = rng.normal(size=(n, 3)) * np.asarray(scale, dtype=float)
    if rotation is not None:
        points = points @ rotation.T
    return points + np.asarray(center, dtype=float)

def convex_outline(points_xy):
    hull = ConvexHull(points_xy)
    poly = points_xy[hull.vertices]
    return np.vstack([poly, poly[0]])

def build_case(clusterer):
    rng = np.random.default_rng(21)
    target = ellipsoid(rng, 4500, (-0.45, -0.05, 0.45), (0.22, 0.07, 0.10), rotation_z(0.02))
    fragment = ellipsoid(rng, 700, (-0.42, 0.21, 0.47), (0.055, 0.025, 0.025), rotation_z(0.10))
    points = np.vstack([target, fragment])
    labels = np.r_[np.zeros(len(target), dtype=np.int64), np.ones(len(fragment), dtype=np.int64)]
    merged = clusterer._merge_fragments(labels.copy(), points)
    assert np.all(merged[labels == 1] == 0)

    feat_a = clusterer._compute_geometric_features(target)
    feat_f = clusterer._compute_geometric_features(fragment)
    centroid_distance = float(np.linalg.norm(feat_a["center"] - feat_f["center"]))
    axis_angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(feat_a["v1"], feat_f["v1"])), -1.0, 1.0))))
    nn_dist, nn_idx = cKDTree(target).query(fragment, k=1)
    nearest_f = fragment[int(np.argmin(nn_dist))]
    nearest_a = target[int(nn_idx[int(np.argmin(nn_dist))])]
    metrics = {
        "fragment_points": int(len(fragment)),
        "target_points": int(len(target)),
        "fragment_L1_m": float(feat_f["L1"]),
        "centroid_distance_m": centroid_distance,
        "axis_angle_deg": axis_angle,
        "minimum_gap_m": float(nn_dist.min()),
        "proximal_fraction_below_2cm": float(np.mean(nn_dist < 0.02)),
        "merged_points": int(np.sum(merged == 0)),
    }
    return points, labels, merged, feat_a, feat_f, nearest_a, nearest_f, metrics

def style_xy(ax, xlim, ylim):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("X (m)", labelpad=2)
    ax.set_ylabel("Y (m)", labelpad=2)
    ax.tick_params(direction="out", length=2.2, width=0.55, pad=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color(C["black"])
    ax.grid(color="#E6E6E6", lw=0.35, alpha=0.65, zorder=0)

def panel_letter(ax, letter):
    ax.text(-0.12, 1.04, letter, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10, fontweight="bold", color=C["black"])

def scale_bar(ax, x0, y0, length=0.2):
    ax.plot([x0, x0 + length], [y0, y0], color=C["black"], lw=1.0, solid_capstyle="butt")
    ax.plot([x0, x0], [y0 - 0.012, y0 + 0.012], color=C["black"], lw=0.7)
    ax.plot([x0 + length, x0 + length], [y0 - 0.012, y0 + 0.012], color=C["black"], lw=0.7)
    ax.text(x0 + length / 2, y0 + 0.018, "0.2 m", ha="center", va="bottom", fontsize=5.7)

def build_figure():
    with open(ROOT / "configs" / "default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    clusterer = InstanceClusterer(cfg)
    points, labels, merged, feat_a, feat_f, nearest_a, nearest_f, metrics = build_case(clusterer)

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 7.2,
        "axes.labelsize": 7.4,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.4,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    fig = plt.figure(figsize=(7.25, 2.82), facecolor="white")
    gs = GridSpec(1, 3, figure=fig, left=0.075, right=0.99, bottom=0.18, top=0.91,
                  wspace=0.32)
    axa, axb, axc = [fig.add_subplot(gs[0, i]) for i in range(3)]

    # (a) Low-overlap input
    style_xy(axa, (-1.10, 0.26), (-0.38, 0.52))
    axa.scatter(points[labels == 0, 0], points[labels == 0, 1], s=2.5, color=C["A"],
                alpha=0.23, edgecolors="none", rasterized=True, zorder=2)
    axa.scatter(points[labels == 1, 0], points[labels == 1, 1], s=10, marker="^",
                facecolors=C["F"], edgecolors="#222222", linewidths=0.12, alpha=0.88, zorder=4)
    axa.plot(*convex_outline(points[labels == 1, :2]).T, color=C["F"], lw=0.8, ls=(0, (2, 2)), zorder=5)
    axa.text(-1.02, 0.22, "A", color=C["A"], fontsize=7.2, fontweight="bold")
    axa.text(-0.42, 0.30, "F1", color=C["F"], fontsize=7.0, fontweight="bold")
    axa.annotate("", xy=feat_f["center"][:2], xytext=feat_a["center"][:2],
                 arrowprops=dict(arrowstyle="-", color=C["gray"], lw=0.65, ls=(0, (3, 2))), zorder=6)
    axa.text(-0.68, 0.07, f"d = {metrics['centroid_distance_m']:.2f} m", color=C["gray"],
             fontsize=6.0, ha="center")
    axa.text(-1.08, 0.48, "separate local fragment", color=C["black"], fontsize=6.3,
             fontweight="bold", ha="left", va="top")
    axa.text(-1.08, -0.31, f"proximal points < 2 cm: {100*metrics['proximal_fraction_below_2cm']:.1f}%",
             color=C["gray"], fontsize=5.9, ha="left")
    scale_bar(axa, -1.02, -0.27, 0.2)
    panel_letter(axa, "a")

    # (b) Interface geometry and decision
    style_xy(axb, (-0.70, -0.20), (0.02, 0.36))
    axb.scatter(points[labels == 0, 0], points[labels == 0, 1], s=4.0, color=C["A"],
                alpha=0.25, edgecolors="none", rasterized=True, zorder=2)
    axb.scatter(points[labels == 1, 0], points[labels == 1, 1], s=15, marker="^",
                facecolors=C["F"], edgecolors="#222222", linewidths=0.15, alpha=0.90, zorder=4)
    for feat, color, length in ((feat_a, C["A"], 0.13), (feat_f, C["F"], 0.07)):
        center = np.asarray(feat["center"])
        v1 = np.asarray(feat["v1"])
        p0, p1 = center[:2] - length * v1[:2], center[:2] + length * v1[:2]
        axb.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0), zorder=7)
    for feat, color in ((feat_a, C["A"]), (feat_f, C["F"])):
        center = np.asarray(feat["center"])[:2]
        axb.scatter([center[0]], [center[1]], marker="D", s=18, color=color,
                    edgecolors="white", linewidths=0.35, zorder=8)
    axb.plot([nearest_a[0], nearest_f[0]], [nearest_a[1], nearest_f[1]], color=C["black"], lw=0.9, zorder=9)
    axb.scatter([nearest_a[0], nearest_f[0]], [nearest_a[1], nearest_f[1]], s=8, color=C["black"], zorder=10)
    axb.annotate(f"minimum gap = {1000*metrics['minimum_gap_m']:.1f} mm",
                 xy=((nearest_a[0] + nearest_f[0]) / 2, (nearest_a[1] + nearest_f[1]) / 2),
                 xytext=(-0.66, 0.10), fontsize=5.8, color=C["black"],
                 arrowprops=dict(arrowstyle="-", color=C["black"], lw=0.5), zorder=12)
    axb.add_patch(Circle(feat_a["center"][:2], 0.30, fill=False, ec=C["gray"], lw=0.6,
                         ls=(0, (3, 3)), zorder=3))
    axb.text(-0.69, 0.345,
             f"L1 = {metrics['fragment_L1_m']:.2f} m\n"
             f"d = {metrics['centroid_distance_m']:.3f} m\n"
             f"θ = {metrics['axis_angle_deg']:.1f}°",
             fontsize=5.9, color=C["black"], ha="left", va="top", linespacing=1.20)
    axb.text(-0.345, 0.345, f"L1 ≤ 0.45 m\nd ≤ 0.30 m\nθ ≤ 50°\n→ merge",
             fontsize=5.9, color=C["pass"], fontweight="bold", ha="left", va="top", linespacing=1.20)
    panel_letter(axb, "b")

    # (c) Merged result
    style_xy(axc, (-1.10, 0.26), (-0.38, 0.52))
    axc.scatter(points[merged == 0, 0], points[merged == 0, 1], s=2.5, color=C["A"],
                alpha=0.30, edgecolors="none", rasterized=True, zorder=2)
    former_hull = convex_outline(points[labels == 1, :2])
    axc.plot(former_hull[:, 0], former_hull[:, 1], color=C["F"], lw=0.9, ls=(0, (2, 2)), zorder=5)
    axc.text(-0.47, 0.39, "former F1", color=C["F"], fontsize=6.2, fontweight="bold", ha="center")
    axc.text(-1.02, 0.22, "A + F1", color=C["A"], fontsize=7.2, fontweight="bold")
    axc.text(-1.08, 0.48, "single recovered instance", color=C["black"], fontsize=6.3,
             fontweight="bold", ha="left", va="top")
    axc.text(-0.44, -0.31, f"{metrics['merged_points']} points", color=C["gray"], fontsize=5.9, ha="center")
    scale_bar(axc, -1.02, -0.27, 0.2)
    panel_letter(axc, "c")

    png = OUT / "fragment_merge_adjacent_journal.png"
    pdf = OUT / "fragment_merge_adjacent_journal.pdf"
    svg = OUT / "fragment_merge_adjacent_journal.svg"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    (OUT / "fragment_merge_adjacent_journal_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(png)
    print(pdf)
    print(svg)

if __name__ == "__main__":
    build_figure()

