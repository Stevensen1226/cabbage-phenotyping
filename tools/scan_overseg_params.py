#!/usr/bin/env python3
"""GIDM 骨架切割阈值扫描 —— 观察"调小参数"带来的 过分割 vs 欠分割 权衡。

默认值 (configs/default.yaml):
  min_peak_dist_m  = 0.35   (峰最小间距, 调小 → 假峰变多 → 过分割↑ 欠分割↓)
  valley_depth_rel = 0.90   (谷深阈值, 调小 → 更浅的谷也能切 → 过分割↑ 欠分割↓)

本脚本对给定参数组合逐簇统计:
  correct (n_pred == n_gt) / over (一株被切碎) / under (n_pred < n_gt),
  并输出对比表, 帮助确定"适当调小"是否划算。

用法:
  python tools/scan_overseg_params.py                       # 默认扫描预设组合
  python tools/scan_overseg_params.py --pairs "0.30:0.90" "0.25:0.85" ...
"""
import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.oversegmentation_analysis import (
    load_cloud_ply, find_sticky_clusters, run_gidm, detect_oversegmentation,
)
from cabbage_pheno.instance.clustering import InstanceClusterer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "evalaute_test")


def collect_clusters(min_plants=2, max_plants=6, min_points=2000):
    clusters = []
    for fn in sorted(os.listdir(DATA_DIR)):
        if not (fn.startswith("cloudR") and fn.endswith(".ply")
                and "_step" not in fn and "_gt" not in fn):
            continue
        xyz, sem, inst = load_cloud_ply(os.path.join(DATA_DIR, fn))
        for c in find_sticky_clusters(xyz, inst, min_points=min_points):
            c["file"] = fn
            if min_plants <= c["n_instances"] <= max_plants:
                clusters.append(c)
    return clusters


def scan(clusters, min_peak_dist, valley_depth_rel, base_cfg):
    cfg = dict(base_cfg)
    cfg.setdefault("instance", {}).setdefault("pca_split", {})
    cfg["instance"]["pca_split"]["min_peak_dist_m"] = min_peak_dist
    cfg["instance"]["pca_split"]["valley_depth_rel"] = valley_depth_rel
    cl = InstanceClusterer(cfg)

    n_correct = n_over_clusters = n_under_clusters = 0
    n_over_plants = n_under_plants = 0
    n_gt_total = 0
    for c in clusters:
        frags = run_gidm(cl, c["xyz"])
        over, _ = detect_oversegmentation(c["inst"], frags)
        n_pred = len(frags)
        n_gt = c["n_instances"]
        n_gt_total += n_gt
        if over:
            n_over_clusters += 1
            n_over_plants += len(over)
        elif n_pred < n_gt:
            n_under_clusters += 1
            n_under_plants += (n_gt - n_pred)
        else:
            n_correct += 1
    return {
        "min_peak_dist": min_peak_dist,
        "valley_depth_rel": valley_depth_rel,
        "correct_clusters": n_correct,
        "over_clusters": n_over_clusters,
        "under_clusters": n_under_clusters,
        "over_plants": n_over_plants,
        "under_plants": n_under_plants,
        "n_gt_total": n_gt_total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--pairs", nargs="+", default=None,
                    help="空格分隔的 'min_peak_dist:valley_depth_rel' 组合")
    args = ap.parse_args()

    base_cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    if args.pairs:
        pairs = []
        for p in args.pairs:
            mpd, vdr = p.split(":")
            pairs.append((float(mpd), float(vdr)))
    else:
        # 预设: 默认 0.35/0.90 起步, 温和调小
        pairs = [
            (0.35, 0.90),   # 默认基线
            (0.30, 0.90),   # 只调小 min_peak_dist
            (0.25, 0.90),
            (0.35, 0.85),   # 只调小 valley_depth_rel
            (0.35, 0.80),
            (0.30, 0.85),   # 两者都温和调小
            (0.25, 0.80),
        ]

    clusters = collect_clusters()
    print(f"粘连簇 (2~6 株, >=2000 点): {len(clusters)} 个\n")

    header = (f"{'min_peak':>9} {'valley_rel':>10} | "
              f"{'correct':>7} {'over簇':>6} {'under簇':>7} | "
              f"{'over株':>5} {'under株':>6} | 备注")
    print(header)
    print("-" * len(header))
    for mpd, vdr in pairs:
        r = scan(clusters, mpd, vdr, base_cfg)
        note = ""
        if r["over_clusters"] == 0 and r["under_clusters"] == 0:
            note = "零误差"
        elif r["over_clusters"] == 0:
            note = "零过分割"
        print(f"{r['min_peak_dist']:9.2f} {r['valley_depth_rel']:10.2f} | "
              f"{r['correct_clusters']:7d} {r['over_clusters']:6d} "
              f"{r['under_clusters']:7d} | "
              f"{r['over_plants']:5d} {r['under_plants']:6d} | {note}")


if __name__ == "__main__":
    main()
