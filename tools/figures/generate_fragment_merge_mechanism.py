#!/usr/bin/env python3
"""Generate a publication-style visualization of the fragment merging cascade.

The example is executed with the project's actual InstanceClusterer methods:
1. `_merge_fragments`: geometric merge-back with span/distance/angle gates.
2. `_cleanup_tiny_fragments`: point-count cleanup and nearest-centroid merging.
"""
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
from matplotlib.patches import Circle, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cabbage_pheno.instance import InstanceClusterer

OUT_DIR = Path(__file__).resolve().parent
COLORS = {
    "A": "#2E86DE",
    "B": "#10AC84",
    "F1": "#F39C12",
    "F2": "#D4A017",
    "F3": "#E74C3C",
    "removed": "#95A5A6",
    "pass": "#16A085",
    "fail": "#D63031",
    "text": "#243447",
    "muted": "#667085",
    "bg": "#FFFFFF",
    "panel": "#F8FAFC",
}

def rotation_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

def ellipsoid(rng, n, center, scale, rotation=None):
    points = rng.normal(size=(n, 3)) * np.asarray(scale, dtype=float)
    if rotation is not None:
        points = points @ rotation.T
    return points + np.asarray(center, dtype=float)

def build_case(clusterer: InstanceClusterer):
    rng = np.random.default_rng(7)
    clusters: list[np.ndarray] = []
    labels: list[int] = []

    def add(n, center, scale, rotation, label):
        clusters.append(ellipsoid(rng, n, center, scale, rotation))
        labels.extend([label] * n)

    add(4500, (-0.45, 0.00, 0.45), (0.32, 0.18, 0.14), rotation_z(0.05), 0)
    add(5200, (0.85, 0.10, 0.45), (0.30, 0.10, 0.12), rotation_z(0.03), 1)
    add(700, (-0.20, 0.00, 0.45), (0.06, 0.03, 0.03), rotation_z(0.10), 2)
    add(900, (0.75, 0.08, 0.45), (0.025, 0.070, 0.03), None, 3)
    add(300, (1.60, -0.95, 0.25), (0.045, 0.045, 0.04), None, 4)

    points = np.vstack(clusters)
    initial = np.asarray(labels, dtype=np.int64)
    merged = clusterer._merge_fragments(initial.copy(), points)
    cleaned = clusterer._cleanup_tiny_fragments(merged.copy(), points)

    assert np.all(merged[initial == 2] == 0), "F1 should merge to A"
    assert np.all(merged[initial == 3] == 3), "F2 should fail geometric merging"
    assert np.all(cleaned[initial == 3] == 1), "F2 should merge to B in cleanup"
    assert np.all(cleaned[initial == 4] == -1), "F3 should be discarded"

    stats = {}
    for label, name in enumerate(["A", "B", "F1", "F2", "F3"]):
        feat = clusterer._compute_geometric_features(points[initial == label])
        stats[name] = {
            "label": label,
            "points": int(np.sum(initial == label)),
            "center": feat["center"].tolist(),
            "L1_m": float(feat["L1"]),
            "v1": feat["v1"].tolist(),
        }

    def pair(name_a: str, name_b: str):
        fa = clusterer._compute_geometric_features(points[initial == stats[name_a]["label"]])
        fb = clusterer._compute_geometric_features(points[initial == stats[name_b]["label"]])
        distance = float(np.linalg.norm(fa["center"] - fb["center"]))
        angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(fa["v1"], fb["v1"])), -1.0, 1.0))))
        return distance, angle

    d_f1_a, theta_f1_a = pair("F1", "A")
    d_f2_b, theta_f2_b = pair("F2", "B")
    metrics = {
        "F1_to_A": {"distance_m": d_f1_a, "angle_deg": theta_f1_a},
        "F2_to_B": {"distance_m": d_f2_b, "angle_deg": theta_f2_b},
        "final_counts": {
            "A_plus_F1": int(np.sum(cleaned == 0)),
            "B_plus_F2": int(np.sum(cleaned == 1)),
            "discarded": int(np.sum(cleaned == -1)),
        },
    }
    return points, initial, merged, cleaned, stats, metrics

def scatter_xy(ax, points, mask, color, marker="o", size=8, alpha=0.6, zorder=2):
    q = points[mask]
    ax.scatter(q[:, 0], q[:, 1], s=size, c=color, marker=marker, alpha=alpha,
               edgecolors="none", linewidths=0, zorder=zorder)

def rounded_box(ax, xy, width, height, fc, ec, title, lines):
    box = FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.02,rounding_size=0.08",
                         facecolor=fc, edgecolor=ec, linewidth=1.2)
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height - 0.16, title, ha="center", va="top",
            fontsize=9.3, weight="bold", color=ec)
    ax.text(xy[0] + width / 2, xy[1] + height - 0.46, lines, ha="center", va="top",
            fontsize=7.2, color=COLORS["text"], linespacing=1.32)

def style_2d_panel(ax, title, panel_label):
    ax.set_facecolor(COLORS["panel"])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#E4E7EC", linewidth=0.55, alpha=0.65)
    ax.tick_params(labelsize=7, colors=COLORS["muted"], length=2)
    for spine in ax.spines.values():
        spine.set_color("#D0D5DD")
    ax.set_title(f"{panel_label}\n{title}", fontsize=10.2, color=COLORS["text"], pad=10, linespacing=1.35)

def add_cluster_axes(ax, stats, names):
    for name in names:
        feat = stats[name]
        c = np.asarray(feat["center"])
        v = np.asarray(feat["v1"])
        half = 0.16 if name.startswith("F") else 0.30
        p0, p1 = c - v * half, c + v * half
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=COLORS[name], lw=2.0, zorder=7)
        ax.scatter([c[0]], [c[1]], marker="D", s=24, c=COLORS[name], edgecolors="white",
                   linewidths=0.7, zorder=8)
        ax.text(c[0], c[1] + 0.05, name, color=COLORS[name], fontsize=8.2,
                weight="bold", ha="center", va="bottom", zorder=9)

def build_figure():
    with open(ROOT / "configs" / "default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    clusterer = InstanceClusterer(cfg)
    points, initial, merged, cleaned, stats, metrics = build_case(clusterer)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "figure.facecolor": COLORS["bg"],
        "savefig.facecolor": COLORS["bg"],
    })
    fig = plt.figure(figsize=(15.2, 9.5))
    gs = GridSpec(2, 2, figure=fig, left=0.045, right=0.975, top=0.90, bottom=0.10,
                  hspace=0.30, wspace=0.16)

    # (a) Input fragments
    ax1 = fig.add_subplot(gs[0, 0])
    style_2d_panel(ax1, "XY projection; decisions are computed in 3D", "(a) Input: over-segmented fragments")
    for label, name, color, marker, size, alpha, zorder in [
        (0, "A", COLORS["A"], "o", 6, 0.18, 2),
        (1, "B", COLORS["B"], "o", 6, 0.18, 2),
        (2, "F1", COLORS["F1"], "^", 44, 0.95, 6),
        (3, "F2", COLORS["F2"], "s", 36, 0.95, 6),
        (4, "F3", COLORS["F3"], "X", 42, 0.95, 6),
    ]:
        scatter_xy(ax1, points, initial == label, color, marker, size, alpha, zorder)
        c = np.asarray(stats[name]["center"])
        label_positions = {
            "A": (-1.08, 0.40), "B": (0.82, 0.48), "F1": (-0.46, -0.28),
            "F2": (1.25, 0.42), "F3": (1.88, -0.92),
        }
        lx, ly = label_positions[name]
        ax1.annotate(
            f"{name} ({stats[name]['points']} pts)",
            xy=(c[0], c[1]), xytext=(lx, ly), textcoords="data",
            ha="center", va="center", fontsize=7.8, weight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=color, alpha=0.92, linewidth=0.7),
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8, alpha=0.8),
            zorder=12,
        )
    ax1.set_xlim(-1.35, 2.05); ax1.set_ylim(-1.35, 1.05)
    ax1.set_xlabel("X (m)", fontsize=8); ax1.set_ylabel("Y (m)", fontsize=8)

    # (b) Geometry-gated merge-back
    ax2 = fig.add_subplot(gs[0, 1])
    style_2d_panel(ax2, "Candidate fragments are compared with target centroids and principal axes",
                   "(b) Geometry-gated merge-back")
    for label, color in [(0, COLORS["A"]), (1, COLORS["B"])]:
        scatter_xy(ax2, points, initial == label, color, "o", 5, 0.10, 1)
    scatter_xy(ax2, points, initial == 2, COLORS["F1"], "^", 34, 0.88, 6)
    scatter_xy(ax2, points, initial == 3, COLORS["F2"], "s", 28, 0.88, 6)
    add_cluster_axes(ax2, stats, ["A", "B", "F1", "F2"])
    for target in ("A", "B"):
        c = np.asarray(stats[target]["center"])
        ax2.add_patch(Circle(c[:2], 0.30, fill=False, ec="#98A2B3", ls=(0, (4, 4)),
                             lw=1.1, zorder=3))
        ax2.text(c[0] + 0.31, c[1] - 0.27, "d ≤ 0.30 m gate", fontsize=6.6,
                 color=COLORS["muted"], ha="left", zorder=4)

    p1, pa = np.asarray(stats["F1"]["center"]), np.asarray(stats["A"]["center"])
    p2, pb = np.asarray(stats["F2"]["center"]), np.asarray(stats["B"]["center"])
    ax2.annotate("", xy=pa[:2], xytext=p1[:2], arrowprops=dict(arrowstyle="->", color=COLORS["pass"], lw=2.4), zorder=10)
    ax2.annotate("", xy=pb[:2], xytext=p2[:2], arrowprops=dict(arrowstyle="->", color=COLORS["fail"], lw=2.4), zorder=10)
    ax2.text(-1.12, 0.82,
             f"F1 → A: PASS\nL1={stats['F1']['L1_m']:.2f} m  d={metrics['F1_to_A']['distance_m']:.2f} m\nθ={metrics['F1_to_A']['angle_deg']:.1f}° < 50°",
             ha="left", va="top", fontsize=7.8, color=COLORS["pass"], weight="bold",
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=COLORS["pass"], alpha=0.95), zorder=12)
    ax2.text(0.40, 0.82,
             f"F2 → B: FAIL\nL1={stats['F2']['L1_m']:.2f} m  d={metrics['F2_to_B']['distance_m']:.2f} m\nθ={metrics['F2_to_B']['angle_deg']:.1f}° > 50°",
             ha="left", va="top", fontsize=7.8, color=COLORS["fail"], weight="bold",
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=COLORS["fail"], alpha=0.95), zorder=12)
    ax2.text(0.02, 0.02,
             "ALL gates required: L1 ≤ 0.45 m   AND   d ≤ 0.30 m   AND   θ ≤ 50°",
             transform=ax2.transAxes, ha="left", va="bottom", fontsize=7.5,
             color=COLORS["text"], bbox=dict(boxstyle="round,pad=0.30", fc="#FFF7E6", ec="#F5D58B"), zorder=12)
    ax2.set_xlim(-1.35, 2.05); ax2.set_ylim(-1.35, 1.05)
    ax2.set_xlabel("X (m)", fontsize=8); ax2.set_ylabel("Y (m)", fontsize=8)

    # (c) Decision cascade
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_xlim(0, 10); ax3.set_ylim(0, 7.3); ax3.axis("off")
    ax3.set_facecolor(COLORS["panel"])
    ax3.text(0.15, 6.98, "(c) Fragment cleanup cascade", fontsize=11, fontweight="bold", color=COLORS["text"])
    rounded_box(ax3, (0.4, 4.78), 9.2, 1.50, "#EAF4FF", COLORS["A"], "Stage 2: point-count cleanup",
                "n < 500: discard   |   500 ≤ n < 1500: nearest-centroid merge if d ≤ 0.15 m   |   n ≥ 1500: valid target")
    rounded_box(ax3, (0.55, 2.62), 2.75, 1.40, "#FFF7E6", COLORS["F1"], "F1: 700 pts",
                f"Geometry PASS\n$d$={metrics['F1_to_A']['distance_m']:.2f} m\n→ merge to A")
    rounded_box(ax3, (3.62, 2.62), 2.75, 1.40, "#FFFBE6", COLORS["F2"], "F2: 900 pts",
                f"Geometry FAIL ($θ$={metrics['F2_to_B']['angle_deg']:.1f}°)\n$d$={metrics['F2_to_B']['distance_m']:.2f} m\n→ cleanup merges to B")
    rounded_box(ax3, (6.70, 2.62), 2.75, 1.40, "#FDECEC", COLORS["F3"], "F3: 300 pts",
                "n < 500\n→ discard as noise")
    ax3.text(5.0, 1.35, "Final labels are renumbered after fragmentation recovery",
             ha="center", fontsize=8.4, color=COLORS["muted"])
    for x in (1.9, 5.0, 8.1):
        ax3.annotate("", xy=(5.0, 4.62), xytext=(x, 4.08),
                     arrowprops=dict(arrowstyle="->", color="#98A2B3", lw=1.2))

    # (d) Final output
    ax4 = fig.add_subplot(gs[1, 1])
    style_2d_panel(ax4, "F1 and F2 are recovered; isolated sub-threshold noise is removed",
                   "(d) After merge and cleanup")
    scatter_xy(ax4, points, cleaned == 0, COLORS["A"], "o", 6, 0.30, 2)
    scatter_xy(ax4, points, cleaned == 1, COLORS["B"], "o", 6, 0.30, 2)
    removed_center = np.asarray(stats["F3"]["center"])
    ax4.scatter([removed_center[0]], [removed_center[1]], s=72, c=COLORS["removed"],
                marker="X", edgecolors="white", linewidths=0.8, zorder=8)
    ax4.text(removed_center[0], removed_center[1] - 0.10, "F3 removed", color=COLORS["removed"],
             fontsize=8, weight="bold", ha="center", va="top")
    ax4.text(0.02, 0.97,
             "Final instances\n"
             f"A + F1: {metrics['final_counts']['A_plus_F1']} pts\n"
             f"B + F2: {metrics['final_counts']['B_plus_F2']} pts\n"
             f"Discarded: {metrics['final_counts']['discarded']} pts",
             transform=ax4.transAxes, va="top", ha="left", fontsize=8.3,
             color=COLORS["text"], bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#D9E2EC"), zorder=10)
    ax4.set_xlim(-1.35, 2.05); ax4.set_ylim(-1.35, 1.05)
    ax4.set_xlabel("X (m)", fontsize=8); ax4.set_ylabel("Y (m)", fontsize=8)

    fig.suptitle("Fragment Merging Mechanism", fontsize=17, fontweight="bold", color=COLORS["text"], y=0.975)
    fig.text(0.5, 0.035,
             "Current defaults:  geometric merge  L1 ≤ 0.45 m, d ≤ 0.30 m, θ ≤ 50°  |  fragment cleanup  discard < 500 pts, merge 500–1500 pts when d ≤ 0.15 m",
             ha="center", va="center", fontsize=9, color=COLORS["muted"])

    png_path = OUT_DIR / "fragment_merge_mechanism.png"
    pdf_path = OUT_DIR / "fragment_merge_mechanism.pdf"
    fig.savefig(png_path, dpi=350, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    metrics_path = OUT_DIR / "fragment_merge_mechanism_metrics.json"
    metrics_path.write_text(json.dumps({"stats": stats, "metrics": metrics}, indent=2), encoding="utf-8")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {metrics_path}")

if __name__ == "__main__":
    build_figure()



