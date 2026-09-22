#!/usr/bin/env python3
"""Detailed CEA-style visualization of the real cloudR4 fragment merge 13 -> 10."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import yaml
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (HERE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cabbage_pheno.instance import InstanceClusterer

OUT = HERE
SCENE = "cloudR4"
RUN_DIR = ROOT / "output" / "step_preview" / "cloudR4_randla_raw_semantic"
COARSE = RUN_DIR / "cloudR4_step4a_cluster_coarse.ply"
SPLIT = RUN_DIR / "cloudR4_step4b_skeleton_split.ply"
MERGED = RUN_DIR / "cloudR4_step4c_fragment_merged.ply"
CONFIG = ROOT / "configs" / "randla_watershed3d_no_preprocess.yaml"
TARGET_ID = 10
FRAGMENT_ID = 13

BLUE = "#AEC7E8"
BROWN = "#8C564B"
BLUE_DARK = "#4C78A8"
BROWN_DARK = "#8C564B"
GRAY = "#3A3A3A"
LIGHT_GRAY = "#D8D8D8"
PASS = "#2A9D6F"


def recover_labels(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    q = np.round(np.asarray(pcd.colors) * 255).astype(int)
    palette = np.round(matplotlib.colormaps["tab20"](np.arange(20) / 20)[:, :3] * 255).astype(int)
    labels = np.full(len(q), -1, dtype=int)
    for idx, color in enumerate(q):
        if tuple(color) in {(25, 25, 25), (26, 26, 26)}:
            continue
        labels[idx] = int(np.argmin(np.linalg.norm(palette - color, axis=1)))
    return labels


def save_subset(path: Path, points: np.ndarray, colors: np.ndarray) -> str:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(path), pcd)
    return str(path.relative_to(ROOT)).replace("\\", "/")


def draw_scale_bar(ax, x, y, length):
    cap = 0.008
    ax.plot([x, x + length], [y, y], color=GRAY, lw=0.9, solid_capstyle="butt", zorder=8)
    ax.plot([x, x], [y - cap, y + cap], color=GRAY, lw=0.9, zorder=8)
    ax.plot([x + length, x + length], [y - cap, y + cap], color=GRAY, lw=0.9, zorder=8)
    ax.text(x + length / 2, y + 0.018, "0.2 m", ha="center", va="bottom", fontsize=6.8, color=GRAY)


def main() -> None:
    with open(CONFIG, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    clusterer = InstanceClusterer(cfg)

    coarse_pcd = o3d.io.read_point_cloud(str(COARSE))
    points = np.asarray(coarse_pcd.points)
    coarse = recover_labels(coarse_pcd)
    split = clusterer._apply_skeleton_split(coarse, points)
    merged = clusterer._merge_fragments(split, points)

    target_mask = split == TARGET_ID
    fragment_mask = split == FRAGMENT_ID
    if not np.any(target_mask) or not np.any(fragment_mask):
        raise RuntimeError("Requested cloudR4 labels are absent.")

    target = points[target_mask]
    fragment = points[fragment_mask]
    both = np.concatenate([target, fragment], axis=0)
    fragment_label_after = Counter(merged[fragment_mask].tolist()).most_common(1)[0][0]
    target_label_after = Counter(merged[target_mask].tolist()).most_common(1)[0][0]
    if fragment_label_after != target_label_after:
        raise RuntimeError("The selected fragment does not merge into the target instance.")

    fa = clusterer._compute_geometric_features(fragment)
    fb = clusterer._compute_geometric_features(target)
    centroid_distance = float(np.linalg.norm(fa["center"] - fb["center"]))
    axis_angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(fa["v1"], fb["v1"])), -1.0, 1.0))))
    nn_dist, nn_idx = cKDTree(target).query(fragment, k=1)
    min_gap = float(nn_dist.min())
    frac_5mm = float(np.mean(nn_dist < 0.005))
    frac_10mm = float(np.mean(nn_dist < 0.010))
    frac_20mm = float(np.mean(nn_dist < 0.020))

    span_limit = float(cfg["instance"]["pca_split"]["merge_min_diameter"])
    dist_limit = float(cfg["instance"]["pca_split"]["merge_max_dist"])
    angle_limit = float(cfg["instance"]["pca_split"]["merge_max_angle"])

    metrics = {
        "scene": SCENE,
        "source_coarse": str(COARSE.relative_to(ROOT)).replace("\\", "/"),
        "source_split": str(SPLIT.relative_to(ROOT)).replace("\\", "/"),
        "source_merge": str(MERGED.relative_to(ROOT)).replace("\\", "/"),
        "target_label_before": TARGET_ID,
        "fragment_label_before": FRAGMENT_ID,
        "label_after": int(target_label_after),
        "target_points": int(len(target)),
        "fragment_points": int(len(fragment)),
        "merged_points": int(len(both)),
        "fragment_fraction_pct": 100.0 * len(fragment) / len(both),
        "fragment_span_m": float(fa["L1"]),
        "target_span_m": float(fb["L1"]),
        "centroid_distance_m": centroid_distance,
        "axis_angle_deg": axis_angle,
        "minimum_gap_m": min_gap,
        "fraction_below_5mm_pct": 100.0 * frac_5mm,
        "fraction_below_10mm_pct": 100.0 * frac_10mm,
        "fraction_below_20mm_pct": 100.0 * frac_20mm,
        "criteria": {
            "fragment_span_limit_m": span_limit,
            "centroid_distance_limit_m": dist_limit,
            "axis_angle_limit_deg": angle_limit,
        },
    }

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 7.2,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    fig = plt.figure(figsize=(7.35, 6.35), facecolor="white")
    gs = GridSpec(2, 3, figure=fig, height_ratios=[2.35, 1.25], left=0.045, right=0.985, bottom=0.065, top=0.94, wspace=0.16, hspace=0.28)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_contact = fig.add_subplot(gs[0, 1])
    ax_after = fig.add_subplot(gs[0, 2])
    bottom = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, :], width_ratios=[1.1, 0.9], wspace=0.25)
    ax_criteria = fig.add_subplot(bottom[0, 0])
    ax_summary = fig.add_subplot(bottom[0, 1])

    # (a) three-dimensional pre-merge view
    ax3d.scatter(target[::2, 0], target[::2, 1], target[::2, 2], s=1.35, c=BLUE, alpha=.42, edgecolors="none", depthshade=False)
    ax3d.scatter(fragment[::2, 0], fragment[::2, 1], fragment[::2, 2], s=1.8, c=BROWN, alpha=.82, edgecolors="none", depthshade=False)
    span3d = both.max(0) - both.min(0)
    center3d = both.mean(0)
    ax3d.set_xlim(center3d[0] - span3d[0] / 2, center3d[0] + span3d[0] / 2)
    ax3d.set_ylim(center3d[1] - span3d[1] / 2, center3d[1] + span3d[1] / 2)
    ax3d.set_zlim(center3d[2] - span3d[2] / 2, center3d[2] + span3d[2] / 2)
    ax3d.set_box_aspect(span3d, zoom=1.38)
    ax3d.set_proj_type("ortho")
    ax3d.view_init(elev=22, azim=-58)
    ax3d.set_axis_off()
    ax3d.set_title("Pre-merge (3D)", fontsize=8.3, fontweight="bold", pad=2)
    ax3d.text2D(0.01, 0.98, "a", transform=ax3d.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")

    # (b) contact-zone zoom in XY
    contact_target = target[nn_idx[nn_dist < 0.03]]
    contact_points = np.concatenate([fragment, contact_target], axis=0)
    c_center = contact_points.mean(0)
    c_span = contact_points.max(0) - contact_points.min(0)
    pad_contact = np.maximum(c_span[:2] * 0.18, np.array([0.025, 0.025]))
    xlim_c = (c_center[0] - c_span[0] / 2 - pad_contact[0], c_center[0] + c_span[0] / 2 + pad_contact[0])
    ylim_c = (c_center[1] - c_span[1] / 2 - pad_contact[1], c_center[1] + c_span[1] / 2 + pad_contact[1])
    crop = ((target[:, 0] > xlim_c[0]) & (target[:, 0] < xlim_c[1]) & (target[:, 1] > ylim_c[0]) & (target[:, 1] < ylim_c[1]))
    ax_contact.scatter(target[crop, 0], target[crop, 1], s=2.1, c=BLUE, alpha=.48, edgecolors="none", label="Main instance")
    ax_contact.scatter(fragment[:, 0], fragment[:, 1], s=2.1, c=BROWN, alpha=.82, edgecolors="none", label="Detached fragment")
    ax_contact.set_xlim(*xlim_c)
    ax_contact.set_ylim(*ylim_c)
    ax_contact.set_aspect("equal", adjustable="box")
    ax_contact.set_axis_off()
    ax_contact.set_title("Contact-zone detail", fontsize=8.3, fontweight="bold", pad=2)
    ax_contact.text(0.01, 0.98, "b", transform=ax_contact.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")
    ax_contact.text(0.02, 0.02, f"nearest gap  {1000*min_gap:.2f} mm\npoints < 2 cm  {100*frac_20mm:.1f}%", transform=ax_contact.transAxes, ha="left", va="bottom", fontsize=6.6, color=GRAY)
    ax_contact.legend(loc="upper right", frameon=False, fontsize=6.2, handlelength=.7, handletextpad=.25, borderaxespad=0)

    # (c) post-merge view using the same global XY crop
    xy_center = both[:, :2].mean(0)
    xy_span = max(np.ptp(both[:, 0]), np.ptp(both[:, 1])) * 1.10
    xlim = (xy_center[0] - xy_span / 2, xy_center[0] + xy_span / 2)
    ylim = (xy_center[1] - xy_span / 2, xy_center[1] + xy_span / 2)
    ax_after.scatter(both[:, 0], both[:, 1], s=1.85, c=BLUE, alpha=.44, edgecolors="none")
    ax_after.set_xlim(*xlim)
    ax_after.set_ylim(*ylim)
    ax_after.set_aspect("equal", adjustable="box")
    ax_after.set_axis_off()
    ax_after.set_title("Post-merge (XY)", fontsize=8.3, fontweight="bold", pad=2)
    ax_after.text(0.01, 0.98, "c", transform=ax_after.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")
    ax_after.text(0.14, 0.98, f"single instance\n{len(both):,} points", transform=ax_after.transAxes, ha="left", va="top", fontsize=6.6, color=GRAY)
    draw_scale_bar(ax_after, xlim[0] + 0.03, ylim[0] + 0.025, 0.2)

    # (d) criterion bars
    labels = ["Fragment span", "Centroid distance", "Principal-axis angle"]
    ratios = [fa["L1"] / span_limit, centroid_distance / dist_limit, axis_angle / angle_limit]
    values = [f"{fa['L1']:.3f} / {span_limit:.3f} m", f"{centroid_distance:.3f} / {dist_limit:.3f} m", f"{axis_angle:.1f} / {angle_limit:.1f}°"]
    ypos = np.arange(len(labels))[::-1]
    ax_criteria.barh(ypos, ratios, height=.45, color=PASS, alpha=.82, edgecolor="none")
    ax_criteria.axvline(1.0, color="#777777", lw=.8, ls="--")
    ax_criteria.set_yticks(ypos, labels)
    ax_criteria.set_xlim(0, 1.15)
    ax_criteria.set_xticks([0, .5, 1.0])
    ax_criteria.set_xlabel("normalized criterion value", fontsize=7)
    ax_criteria.tick_params(axis="both", labelsize=6.7, length=2.5)
    for y, value in zip(ypos, values):
        ax_criteria.text(1.03, y, value, va="center", ha="left", fontsize=6.5, color=GRAY)
    ax_criteria.set_title("Geometric merge criteria", fontsize=8.3, fontweight="bold", loc="left", pad=3)
    ax_criteria.text(-.16, 1.02, "d", transform=ax_criteria.transAxes, ha="left", va="bottom", fontsize=11, fontweight="bold")
    for spine in ("top", "right"):
        ax_criteria.spines[spine].set_visible(False)
    ax_criteria.grid(axis="x", color="#E5E5E5", lw=.5)
    ax_criteria.set_axisbelow(True)

    # (e) numeric summary
    ax_summary.axis("off")
    ax_summary.set_title("Merge summary", fontsize=8.3, fontweight="bold", loc="left", pad=3)
    units = [
        "Target instance",
        "Detached fragment",
        "Merged instance",
        "Nearest gap",
        "Fragment points < 5 mm",
        "Fragment points < 10 mm",
        "Fragment points < 20 mm",
    ]
    vals = [f"{len(target):,} pts", f"{len(fragment):,} pts", f"{len(both):,} pts", f"{1000*min_gap:.2f} mm", f"{100*frac_5mm:.1f}%", f"{100*frac_10mm:.1f}%", f"{100*frac_20mm:.1f}%"]
    y = .89
    for key, value in zip(units, vals):
        ax_summary.text(.0, y, key, ha="left", va="center", fontsize=6.6, color=GRAY)
        ax_summary.text(1.0, y, value, ha="right", va="center", fontsize=6.7, color="black")
        y -= .125
    ax_summary.text(0.0, 0.0, "merge result: label 13 → label 10", fontsize=6.5, color=PASS, ha="left", va="bottom")

    fig.text(0.045, 0.975, "cloudR4 | detached fragment 13 → main instance 10", ha="left", va="top", fontsize=8.0, fontweight="bold", color=GRAY)

    stem = OUT / "cloudR4_fragment13_main10_merge_detail"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)

    before_rgb = np.tile(mcolors.to_rgb(BLUE), (len(both), 1))
    before_rgb[len(target):] = mcolors.to_rgb(BROWN)
    after_rgb = np.tile(mcolors.to_rgb(BLUE), (len(both), 1))
    metrics["exported_before_ply"] = save_subset(RUN_DIR / "cloudR4_fragment13_main10_before_merge.ply", both, before_rgb)
    metrics["exported_after_ply"] = save_subset(RUN_DIR / "cloudR4_fragment13_main10_after_merge.ply", both, after_rgb)
    metrics["output_png"] = str(stem.with_suffix(".png").relative_to(ROOT)).replace("\\", "/")
    metrics["output_pdf"] = str(stem.with_suffix(".pdf").relative_to(ROOT)).replace("\\", "/")
    metrics["output_svg"] = str(stem.with_suffix(".svg").relative_to(ROOT)).replace("\\", "/")
    (OUT / "cloudR4_fragment13_main10_merge_detail_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))
    print(stem.with_suffix(".svg"))


if __name__ == "__main__":
    main()



