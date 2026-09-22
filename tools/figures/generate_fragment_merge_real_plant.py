#!/usr/bin/env python3
"""Journal-style visualization: a real cabbage instance with a detached local fragment."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import yaml
from matplotlib.gridspec import GridSpec
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (HERE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from generate_fragment_merge_clean_journal import (
    C, draw_gate, hull_xy, panel_letter, scale_bar, style_cloud_panel,
)
from cabbage_pheno.instance import InstanceClusterer

OUT = HERE
SOURCE = ROOT / "output" / "iach_gidm_debug" / "cloudR12" / "stage3_skeleton.ply"
TARGET_GROUP = 5
ANCHOR = np.array([0.5728380, -2.6213770, 0.0567630])
PATCH_RADIUS = 0.070
OUTWARD_SHIFT = 0.08

def recover_labels(pcd):
    q = np.round(np.asarray(pcd.colors) * 255).astype(np.uint8)
    unique, inverse = np.unique(q, axis=0, return_inverse=True)
    labels = np.full(len(inverse), -1, dtype=np.int64)
    next_id = 0
    for group_idx, color in enumerate(unique):
        if tuple(color) in {(127, 127, 127), (128, 128, 128)}:
            continue
        labels[inverse == group_idx] = next_id
        next_id += 1
    return labels

def build_case(clusterer):
    pcd = o3d.io.read_point_cloud(str(SOURCE))
    points = np.asarray(pcd.points)
    labels_all = recover_labels(pcd)
    plant = points[labels_all == TARGET_GROUP]

    patch = np.linalg.norm(plant - ANCHOR, axis=1) < PATCH_RADIUS
    target = plant[~patch]
    fragment = plant[patch]
    direction = fragment.mean(0) - target.mean(0)
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    fragment = fragment + OUTWARD_SHIFT * direction

    points2 = np.vstack([target, fragment])
    labels = np.r_[np.zeros(len(target), dtype=np.int64), np.ones(len(fragment), dtype=np.int64)]
    merged = clusterer._merge_fragments(labels.copy(), points2)
    assert len(fragment) >= 500
    assert np.all(merged[labels == 1] == 0)

    fa = clusterer._compute_geometric_features(target)
    ff = clusterer._compute_geometric_features(fragment)
    d = float(np.linalg.norm(fa["center"] - ff["center"]))
    angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(fa["v1"], ff["v1"])), -1.0, 1.0))))
    nn_dist = cKDTree(target).query(fragment, k=1)[0]
    metrics = {
        "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_label": TARGET_GROUP,
        "target_points_before": int(len(target)),
        "fragment_points": int(len(fragment)),
        "merged_points": int(len(points2)),
        "fragment_L1_m": float(ff["L1"]),
        "centroid_distance_m": d,
        "axis_angle_deg": angle,
        "minimum_gap_m": float(nn_dist.min()),
        "fraction_below_2cm": float(np.mean(nn_dist < 0.02)),
    }
    return points2, labels, merged, metrics

def build_figure():
    with open(ROOT / "configs" / "default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    clusterer = InstanceClusterer(cfg)
    points, labels, merged, metrics = build_case(clusterer)

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 7.0,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "pdf.fonttype": 42,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(7.25, 2.55), facecolor="white")
    gs = GridSpec(1, 3, figure=fig, left=0.07, right=0.99, bottom=0.18, top=0.91,
                  wspace=0.30)
    axa, axb, axc = [fig.add_subplot(gs[0, i]) for i in range(3)]

    all_pts = points
    xlim = (all_pts[:, 0].min() - 0.03, all_pts[:, 0].max() + 0.03)
    ylim = (all_pts[:, 1].min() - 0.03, all_pts[:, 1].max() + 0.16)

    # a) Real whole plant with a detached local fragment.
    style_cloud_panel(axa, xlim, ylim)
    axa.scatter(points[labels == 0, 0], points[labels == 0, 1], s=2.3, color=C["A"],
                alpha=0.25, edgecolors="none", rasterized=True, zorder=2)
    axa.scatter(points[labels == 1, 0], points[labels == 1, 1], s=2.3, color=C["F"],
                alpha=0.52, edgecolors="none", rasterized=True, zorder=3)

    axa.text(0.02, 0.98, f"nearest gap = {100*metrics['minimum_gap_m']:.1f} cm\n"
                         f"overlap < 2 cm = {100*metrics['fraction_below_2cm']:.1f}%",
             transform=axa.transAxes, ha="left", va="top", fontsize=5.8, color=C["gray"])
    axa.text(0.14, 0.86, "A: whole plant", transform=axa.transAxes, ha="left", va="center",
             fontsize=6.2, color=C["A"], fontweight="bold")
    axa.text(0.62, 0.86, "F1: detached fragment", transform=axa.transAxes, ha="left", va="center",
             fontsize=6.2, color=C["F"], fontweight="bold")
    scale_bar(axa, xlim[0] + 0.035, ylim[0] + 0.035, 0.1)
    panel_letter(axa, "a")

    # b) Actual merge criteria.
    axb.set_xlim(0, 1)
    axb.set_ylim(0, 1)
    axb.axis("off")
    axb.text(0.0, 0.96, "merge criteria", fontsize=6.7, fontweight="bold",
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

    # c) Merged single instance.
    style_cloud_panel(axc, xlim, ylim)
    axc.scatter(points[merged == 0, 0], points[merged == 0, 1], s=2.3, color=C["A"],
                alpha=0.31, edgecolors="none", rasterized=True, zorder=2)
    axc.text(0.02, 0.98, f"{metrics['merged_points']} points", transform=axc.transAxes,
             ha="left", va="top", fontsize=5.8, color=C["gray"])
    axc.text(0.02, 0.86, "single instance", transform=axc.transAxes, ha="left", va="center",
             fontsize=6.3, color=C["A"], fontweight="bold")
    scale_bar(axc, xlim[0] + 0.035, ylim[0] + 0.035, 0.1)
    panel_letter(axc, "c")

    png = OUT / "fragment_merge_real_plant.png"
    pdf = OUT / "fragment_merge_real_plant.pdf"
    svg = OUT / "fragment_merge_real_plant.svg"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    (OUT / "fragment_merge_real_plant_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(png)
    print(pdf)
    print(svg)

if __name__ == "__main__":
    build_figure()


