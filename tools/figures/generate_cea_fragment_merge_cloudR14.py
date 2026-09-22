#!/usr/bin/env python3
"""Create a CEA-style before/after figure for a real cloudR14 fragment merge.

The source is the legacy cloudR14 IAch/GIDM point set.  The example is the
same real split/merge pair shown in the supplied screenshots:

    skeleton split label 22 (detached fragment)
    skeleton split label 13 (main instance)
    -> fragment merge label 13

Projection is the native XY view.  No fragment displacement or synthetic point
generation is used.
"""
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
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (HERE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cabbage_pheno.instance import InstanceClusterer
from tools.generate_steps import run_clustering

OUT = HERE
SCENE = "cloudR14"
TARGET_ID = 13
FRAGMENT_ID = 22
BLUE = "#AEC7E8"
BROWN = "#8C564B"
GRAY = "#333333"

CABBAGE = ROOT / "output" / "iach_gidm_debug" / SCENE / "cabbage_points.ply"
STAGE_DIR = ROOT / "output" / "step_preview" / f"{SCENE}_legacy_check"
COARSE = STAGE_DIR / f"{SCENE}_step4a_cluster_coarse.ply"


def recover_sequential_labels(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    """Recover sequential cluster IDs from tab20-coloured coarse PLY data."""
    colors = np.round(np.asarray(pcd.colors) * 255).astype(np.uint8)
    unique, inverse = np.unique(colors, axis=0, return_inverse=True)
    labels = np.full(len(inverse), -1, dtype=np.int64)
    next_label = 0
    for group_idx, color in enumerate(unique):
        if tuple(color) in {(25, 25, 25), (26, 26, 26)}:
            continue
        labels[inverse == group_idx] = next_label
        next_label += 1
    return labels


def ensure_stage_files(cfg: dict) -> None:
    if COARSE.exists():
        return
    if not CABBAGE.exists():
        raise FileNotFoundError(f"Legacy cloudR14 point cloud not found: {CABBAGE}")
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    run_clustering(o3d.io.read_point_cloud(str(CABBAGE)), SCENE, str(STAGE_DIR), cfg)


def build_case() -> tuple[np.ndarray, np.ndarray, dict]:
    with open(ROOT / "configs" / "default.yaml", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    ensure_stage_files(cfg)

    coarse_pcd = o3d.io.read_point_cloud(str(COARSE))
    points = np.asarray(coarse_pcd.points)
    coarse = recover_sequential_labels(coarse_pcd)

    clusterer = InstanceClusterer(cfg)
    split = clusterer._apply_skeleton_split(coarse, points)
    merged = clusterer._merge_fragments(split, points)

    target = split == TARGET_ID
    fragment = split == FRAGMENT_ID
    selected = target | fragment
    if not np.any(selected):
        raise RuntimeError("Selected cloudR14 split labels are absent from the coarse stage.")

    target_after = Counter(merged[target].tolist()).most_common(1)[0][0]
    fragment_after = Counter(merged[fragment].tolist()).most_common(1)[0][0]
    if target_after != fragment_after:
        raise RuntimeError(
            f"Labels {TARGET_ID} and {FRAGMENT_ID} do not merge to one instance "
            f"({target_after} vs {fragment_after})."
        )

    before = np.where(fragment[selected], 1, 0)
    case_points = points[selected]
    metrics = {
        "scene": SCENE,
        "source_point_cloud": str(CABBAGE.relative_to(ROOT)).replace("\\", "/"),
        "source_coarse_stage": str(COARSE.relative_to(ROOT)).replace("\\", "/"),
        "projection": "XY (native coordinates)",
        "main_label_before_merge": TARGET_ID,
        "fragment_label_before_merge": FRAGMENT_ID,
        "label_after_merge": int(target_after),
        "main_points": int(target.sum()),
        "fragment_points": int(fragment.sum()),
        "merged_points": int(selected.sum()),
        "fragment_fraction_pct": round(100.0 * float(fragment.sum()) / float(selected.sum()), 2),
    }
    return case_points, before, metrics


def draw_scale_bar(ax: plt.Axes, x: float, y: float, length: float = 0.2) -> None:
    cap = 0.008
    ax.plot([x, x + length], [y, y], color=GRAY, lw=0.85, solid_capstyle="butt", zorder=5)
    ax.plot([x, x], [y - cap, y + cap], color=GRAY, lw=0.85, zorder=5)
    ax.plot([x + length, x + length], [y - cap, y + cap], color=GRAY, lw=0.85, zorder=5)
    ax.text(x + length / 2, y + 0.018, "0.2 m", ha="center", va="bottom", fontsize=7.0, color=GRAY)


def build_figure() -> None:
    points, before, metrics = build_case()
    fragment = before == 1

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.0,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig = plt.figure(figsize=(7.20, 3.25), facecolor="white")
    gs = GridSpec(
        1,
        2,
        figure=fig,
        left=0.045,
        right=0.985,
        bottom=0.115,
        top=0.965,
        wspace=0.045,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    center = points[:, :2].mean(axis=0)
    span = float(np.max(points[:, :2].max(axis=0) - points[:, :2].min(axis=0))) * 1.12
    lim = (center[0] - span / 2, center[0] + span / 2)
    ylim = (center[1] - span / 2, center[1] + span / 2)

    for ax in (ax_a, ax_b):
        ax.set_xlim(*lim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()

    # (a) Initial clustering: the detached fragment is distinguished only by colour.
    ax_a.scatter(
        points[~fragment, 0],
        points[~fragment, 1],
        s=1.8,
        color=BLUE,
        alpha=0.42,
        edgecolors="none",
        zorder=2,
    )
    ax_a.scatter(
        points[fragment, 0],
        points[fragment, 1],
        s=1.8,
        color=BROWN,
        alpha=0.72,
        edgecolors="none",
        zorder=3,
    )

    # (b) After fragment merging: the complete selected instance is one colour.
    ax_b.scatter(
        points[:, 0],
        points[:, 1],
        s=1.8,
        color=BLUE,
        alpha=0.42,
        edgecolors="none",
        zorder=2,
    )

    # One common scale bar, placed inside each panel at identical data coordinates.
    draw_scale_bar(ax_a, lim[0] + 0.035, ylim[0] + 0.025)
    draw_scale_bar(ax_b, lim[0] + 0.035, ylim[0] + 0.025)

    ax_a.text(-0.015, 1.015, "a", transform=ax_a.transAxes, ha="left", va="bottom", fontsize=11, fontweight="bold", color="black")
    ax_b.text(-0.015, 1.015, "b", transform=ax_b.transAxes, ha="left", va="bottom", fontsize=11, fontweight="bold", color="black")
    ax_a.text(0.5, -0.075, "Initial instance clustering", transform=ax_a.transAxes, ha="center", va="top", fontsize=7.8, color=GRAY)
    ax_b.text(0.5, -0.075, "After fragment merging", transform=ax_b.transAxes, ha="center", va="top", fontsize=7.8, color=GRAY)

    # A compact, unboxed legend preserves the exact colour semantics in panel a.
    handles = [
        plt.Line2D([], [], marker="o", linestyle="None", markersize=4.3, markerfacecolor=BLUE, markeredgecolor="none", label="Main instance"),
        plt.Line2D([], [], marker="o", linestyle="None", markersize=4.3, markerfacecolor=BROWN, markeredgecolor="none", label="Detached fragment"),
    ]
    ax_a.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.005),
        ncol=2,
        frameon=False,
        fontsize=7.0,
        handlelength=0.7,
        handletextpad=0.25,
        columnspacing=1.0,
        borderaxespad=0.0,
    )

    stem = OUT / "cea_fragment_merge_cloudR14_initial_vs_merged"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)

    # Export the exact PLY subsets used in the figure for traceability.
    def write_colored_cloud(path: Path, colors: np.ndarray) -> str:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(str(path), pcd)
        return str(path.relative_to(ROOT)).replace("\\", "/")

    rgb = np.tile(np.array(matplotlib.colors.to_rgb(BLUE)), (len(points), 1))
    rgb[fragment] = matplotlib.colors.to_rgb(BROWN)
    before_path = STAGE_DIR / f"{SCENE}_fragment{FRAGMENT_ID}_main{TARGET_ID}_before_merge.ply"
    after_path = STAGE_DIR / f"{SCENE}_fragment{FRAGMENT_ID}_main{TARGET_ID}_after_merge.ply"
    metrics["exported_before_ply"] = write_colored_cloud(before_path, rgb)
    metrics["exported_after_ply"] = write_colored_cloud(
        after_path, np.tile(np.array(matplotlib.colors.to_rgb(BLUE)), (len(points), 1))
    )
    metrics["output_png"] = str(stem.with_suffix(".png").relative_to(ROOT)).replace("\\", "/")
    metrics["output_pdf"] = str(stem.with_suffix(".pdf").relative_to(ROOT)).replace("\\", "/")
    metrics["output_svg"] = str(stem.with_suffix(".svg").relative_to(ROOT)).replace("\\", "/")
    (OUT / "cea_fragment_merge_cloudR14_initial_vs_merged_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))
    print(stem.with_suffix(".svg"))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    build_figure()


