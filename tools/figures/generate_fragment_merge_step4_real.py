#!/usr/bin/env python3
"""Visualize a real fragment merge from generate_steps.py outputs."""
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
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (HERE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from generate_fragment_merge_clean_journal import C, draw_gate, panel_letter, scale_bar, style_cloud_panel
from cabbage_pheno.instance import InstanceClusterer
from tools.generate_steps import run_clustering

OUT = HERE
SCENE = "cloudR5"
COARSE = ROOT / "output" / "step_preview" / SCENE / f"{SCENE}_step4a_cluster_coarse.ply"
SPLIT = ROOT / "output" / "step_preview" / SCENE / f"{SCENE}_step4b_skeleton_split.ply"
MERGED = ROOT / "output" / "step_preview" / SCENE / f"{SCENE}_step4c_fragment_merged.ply"
CABBAGE = ROOT / "output" / "step_preview" / SCENE / f"{SCENE}_step3_semantic_cabbage.ply"
TARGET_ID = 5
FRAGMENT_ID = 8

def recover_labels(pcd):
    q = np.round(np.asarray(pcd.colors) * 255).astype(np.uint8)
    unique, inverse = np.unique(q, axis=0, return_inverse=True)
    labels = np.full(len(inverse), -1, dtype=np.int64)
    cmap = matplotlib.colormaps["tab20"]
    palette = np.round(cmap(np.arange(20) / 20.0)[:, :3] * 255).astype(np.uint8)
    for group_idx, color in enumerate(unique):
        if tuple(color) in {(25, 25, 25), (26, 26, 26)}:
            continue
        palette_idx = int(np.argmin(np.linalg.norm(palette.astype(int) - color.astype(int), axis=1)))
        labels[inverse == group_idx] = palette_idx
    return labels

def ensure_stage_files(cfg):
    if COARSE.exists() and SPLIT.exists() and MERGED.exists():
        return
    COARSE.parent.mkdir(parents=True, exist_ok=True)
    run_clustering(o3d.io.read_point_cloud(str(CABBAGE)), SCENE, str(COARSE.parent), cfg)

def build_case(clusterer):
    with open(ROOT / "configs" / "default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ensure_stage_files(cfg)
    split_pcd = o3d.io.read_point_cloud(str(SPLIT))
    merged_pcd = o3d.io.read_point_cloud(str(MERGED))
    points = np.asarray(split_pcd.points)
    premerge = recover_labels(split_pcd)
    postmerge = recover_labels(merged_pcd)
    target = points[premerge == TARGET_ID]
    fragment = points[premerge == FRAGMENT_ID]
    after_label = Counter(postmerge[premerge == FRAGMENT_ID]).most_common(1)[0][0]
    assert Counter(postmerge[premerge == TARGET_ID]).most_common(1)[0][0] == after_label
    fa = clusterer._compute_geometric_features(target)
    ff = clusterer._compute_geometric_features(fragment)
    d = float(np.linalg.norm(fa["center"] - ff["center"]))
    angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(fa["v1"], ff["v1"])), -1.0, 1.0))))
    nn = cKDTree(target).query(fragment, k=1)[0]
    selected = (premerge == TARGET_ID) | (premerge == FRAGMENT_ID)
    metrics = {
        "scene": SCENE,
        "source_split_file": str(SPLIT.relative_to(ROOT)).replace("\\", "/"),
        "source_merge_file": str(MERGED.relative_to(ROOT)).replace("\\", "/"),
        "target_label_before_merge": TARGET_ID,
        "fragment_label_before_merge": FRAGMENT_ID,
        "target_points": int(len(target)),
        "fragment_points": int(len(fragment)),
        "merged_points": int(np.sum(selected)),
        "fragment_L1_m": float(ff["L1"]),
        "centroid_distance_m": d,
        "axis_angle_deg": angle,
        "minimum_gap_m": float(nn.min()),
        "fraction_below_2cm": float(np.mean(nn < 0.02)),
        "after_label": int(after_label),
    }
    return points[selected], premerge[selected], metrics

def build_figure():
    with open(ROOT / "configs" / "default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    clusterer = InstanceClusterer(cfg)
    points, before, metrics = build_case(clusterer)
    plt.rcParams.update({"font.family":"Arial","font.size":7.0,"axes.labelsize":7.2,"xtick.labelsize":6.2,"ytick.labelsize":6.2,"axes.linewidth":0.6,"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})
    fig = plt.figure(figsize=(7.25, 2.55), facecolor="white")
    gs = GridSpec(1, 3, figure=fig, left=0.07, right=0.99, bottom=0.18, top=0.91, wspace=0.30)
    axa, axb, axc = [fig.add_subplot(gs[0, i]) for i in range(3)]
    xlim = (points[:,0].min()-0.04, points[:,0].max()+0.04)
    ylim = (points[:,1].min()-0.04, points[:,1].max()+0.14)
    style_cloud_panel(axa, xlim, ylim)
    axa.scatter(points[before==TARGET_ID,0], points[before==TARGET_ID,1], s=2.0, color=C["A"], alpha=0.28, edgecolors="none", rasterized=True, zorder=2)
    axa.scatter(points[before==FRAGMENT_ID,0], points[before==FRAGMENT_ID,1], s=2.0, color=C["F"], alpha=0.48, edgecolors="none", rasterized=True, zorder=3)
    axa.text(0.02,0.98,f"nearest gap = {100*metrics['minimum_gap_m']:.1f} mm\npoints < 2 cm apart = {100*metrics['fraction_below_2cm']:.1f}%",transform=axa.transAxes,ha="left",va="top",fontsize=5.8,color=C["gray"])
    axa.text(0.02,0.84,"A: main fragment",transform=axa.transAxes,ha="left",va="center",fontsize=6.2,color=C["A"],fontweight="bold")
    axa.text(0.60,0.84,"F1: detached fragment",transform=axa.transAxes,ha="left",va="center",fontsize=6.2,color=C["F"],fontweight="bold")
    scale_bar(axa,xlim[0]+0.035,ylim[0]+0.035,0.1);panel_letter(axa,"a")
    axb.set_xlim(0,1);axb.set_ylim(0,1);axb.axis("off")
    axb.text(0.0,0.96,"merge criteria",fontsize=6.7,fontweight="bold",ha="left",va="top",color=C["black"])
    draw_gate(axb,0.75,"fragment span",f"L1 = {metrics['fragment_L1_m']:.2f} m  ≤  0.45 m",metrics["fragment_L1_m"]/0.45,C["pass"])
    draw_gate(axb,0.50,"centroid distance",f"d = {metrics['centroid_distance_m']:.3f} m  ≤  0.30 m",metrics["centroid_distance_m"]/0.30,C["pass"])
    draw_gate(axb,0.25,"principal-axis angle",f"θ = {metrics['axis_angle_deg']:.1f}°  ≤  50°",metrics["axis_angle_deg"]/50.0,C["pass"])
    axb.plot([0.0,1.0],[0.12,0.12],color=C["light"],lw=0.55);axb.text(0.5,0.055,"all criteria pass  →  merge",fontsize=7.0,color=C["pass"],fontweight="bold",ha="center",va="center");panel_letter(axb,"b")
    style_cloud_panel(axc,xlim,ylim);axc.scatter(points[:,0],points[:,1],s=2.0,color=C["A"],alpha=0.31,edgecolors="none",rasterized=True,zorder=2)
    axc.text(0.02,0.98,f"{metrics['merged_points']} points",transform=axc.transAxes,ha="left",va="top",fontsize=5.8,color=C["gray"]);axc.text(0.02,0.84,"single instance",transform=axc.transAxes,ha="left",va="center",fontsize=6.3,color=C["A"],fontweight="bold");scale_bar(axc,xlim[0]+0.035,ylim[0]+0.035,0.1);panel_letter(axc,"c")
    export_dir = COARSE.parent
    def save_subset(mask, color, filename):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[mask])
        pcd.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color, dtype=float), (int(np.sum(mask)), 1)))
        path = export_dir / filename
        o3d.io.write_point_cloud(str(path), pcd)
        return str(path.relative_to(ROOT)).replace("\\", "/")
    target_path = save_subset(before == TARGET_ID, [0.0, 0.45, 0.70], f"{SCENE}_target{TARGET_ID}_before_merge.ply")
    fragment_path = save_subset(before == FRAGMENT_ID, [0.90, 0.62, 0.0], f"{SCENE}_fragment{FRAGMENT_ID}_before_merge.ply")
    colored = o3d.geometry.PointCloud()
    colored.points = o3d.utility.Vector3dVector(points)
    cols = np.zeros((len(points), 3), dtype=float)
    cols[before == TARGET_ID] = [0.0, 0.45, 0.70]
    cols[before == FRAGMENT_ID] = [0.90, 0.62, 0.0]
    colored.colors = o3d.utility.Vector3dVector(cols)
    before_path = export_dir / f"{SCENE}_before_merge_colored.ply"
    o3d.io.write_point_cloud(str(before_path), colored)
    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(points)
    merged.colors = o3d.utility.Vector3dVector(np.tile(np.asarray([0.0,0.45,0.70]), (len(points),1)))
    merged_path = export_dir / f"{SCENE}_merged_after_merge.ply"
    o3d.io.write_point_cloud(str(merged_path), merged)
    metrics["exported_ply"] = {"target": target_path, "fragment": fragment_path, "before_colored": str(before_path.relative_to(ROOT)).replace("\\", "/"), "after_merged": str(merged_path.relative_to(ROOT)).replace("\\", "/")}
    png=OUT/"fragment_merge_step4_real.png";pdf=OUT/"fragment_merge_step4_real.pdf";svg=OUT/"fragment_merge_step4_real.svg"
    fig.savefig(png,dpi=600,bbox_inches="tight");fig.savefig(pdf,bbox_inches="tight");fig.savefig(svg,bbox_inches="tight");plt.close(fig)
    (OUT/"fragment_merge_step4_real_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(png);print(pdf);print(svg)
if __name__ == "__main__": build_figure()



