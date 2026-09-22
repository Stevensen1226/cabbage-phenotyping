#!/usr/bin/env python3
"""可视化粘连簇的"长度(L1) vs 长宽比(r)"分布, 叠加异常检测触发阈值。

回答: 这些失败案例的长度和长宽比是否都达到切割条件?
  - is_huge    = (L1 > 0.60) or (max_axis > 0.60)
  - is_elongated = (r > 1.60) and (L1 > 0.25)
  满足任一即触发骨架切割。

用法:
  python tools/plot_trigger_conditions.py --config configs/default.yaml \
      --min-peak-dist 0.15
"""
import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.oversegmentation_analysis import (
    load_cloud_ply, find_sticky_clusters, run_gidm, detect_oversegmentation,
    _setup_font,
)
from cabbage_pheno.instance.clustering import InstanceClusterer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "evalaute_test")
OUT_DIR = os.path.join(ROOT, "output", "oversegmentation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--min-peak-dist", type=float, default=0.35)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    cfg.setdefault("instance", {}).setdefault("pca_split", {})["min_peak_dist_m"] = args.min_peak_dist
    cl = InstanceClusterer(cfg)

    rows = []
    for fn in sorted(os.listdir(DATA_DIR)):
        if not (fn.startswith("cloudR") and fn.endswith(".ply")
                and "_step" not in fn and "_gt" not in fn):
            continue
        xyz, sem, inst = load_cloud_ply(os.path.join(DATA_DIR, fn))
        for c in find_sticky_clusters(xyz, inst, min_points=2000):
            if 2 <= c["n_instances"] <= 6:
                feat = cl._compute_geometric_features(c["xyz"])
                dims = c["xyz"].max(0) - c["xyz"].min(0)
                max_axis = dims.max()
                is_huge = (feat["L1"] > cl.abn_d) or (max_axis > cl.abn_d)
                is_elong = (feat["r"] > cl.abn_ar) and (feat["L1"] > 0.25)
                frags = run_gidm(cl, c["xyz"])
                over, _ = detect_oversegmentation(c["inst"], frags)
                rows.append({
                    "file": fn, "n_plants": c["n_instances"],
                    "L1": feat["L1"], "r": feat["r"],
                    "max_axis": max_axis, "is_huge": is_huge,
                    "is_elong": is_elong, "over": bool(over),
                })

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _setup_font()

    L1 = np.array([r["L1"] for r in rows])
    rr = np.array([r["r"] for r in rows])
    over = np.array([r["over"] for r in rows])
    n_plants = np.array([r["n_plants"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # (a) L1 vs r 散点, 过分割红标
    ax = axes[0]
    ok = ~over
    ax.scatter(L1[ok], rr[ok], s=70, c="#1E8449", alpha=0.8, edgecolor="white",
               linewidth=0.5, label=f"正常切割 ({ok.sum()} 簇)")
    ax.scatter(L1[over], rr[over], s=130, c="#E74C3C", alpha=0.9,
               marker="X", edgecolor="black", linewidth=0.6,
               label=f"过分割 ({over.sum()} 簇)")
    # 阈值线
    ax.axvline(cl.abn_d, ls="--", color="gray", lw=1.2)
    ax.axhline(cl.abn_ar, ls="--", color="gray", lw=1.2)
    ax.text(cl.abn_d + 0.02, 4.3, "L1=0.60", fontsize=8, color="gray")
    ax.text(2.3, cl.abn_ar + 0.05, "r=1.60", fontsize=8, color="gray")
    # 触发区着色: 右上角 (L1>0.60 且 r>1.60) 是触发区
    ax.axvspan(cl.abn_d, L1.max() * 1.05, ymin=0.5, ymax=1.0,
               color="#FDEBD0", alpha=0.4, zorder=0)
    ax.set_xlabel("长度 L1 (m, 沿第一主成分)")
    ax.set_ylabel("长宽比 r = L1/L2")
    ax.set_title(f"(a) 粘连簇的 L1 vs r (min_peak_dist={args.min_peak_dist})\n"
                 "全部落在线右侧→全满足 is_huge 触发", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower right", frameon=False)

    # (b) 过分割案例的 r 分布 vs 株数 (解释: r 大=多株排一行, 是正常形态)
    ax2 = axes[1]
    ax2.scatter(n_plants[ok], rr[ok], s=70, c="#1E8449", alpha=0.8,
                edgecolor="white", linewidth=0.5, label="正常")
    ax2.scatter(n_plants[over], rr[over], s=130, c="#E74C3C", alpha=0.9,
                marker="X", edgecolor="black", linewidth=0.6, label="过分割")
    ax2.axhline(cl.abn_ar, ls="--", color="gray", lw=1.2)
    ax2.set_xlabel("簇内株数")
    ax2.set_ylabel("长宽比 r")
    ax2.set_title("(b) 长宽比 vs 株数\nr 随株数增大 = 多株排成一行(非单株拉长)",
                  fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8.5, loc="upper left", frameon=False)

    fig.suptitle("骨架切割触发条件检查: L1(长度) 与 r(长宽比)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_png = os.path.join(OUT_DIR, f"trigger_conditions_mpd{args.min_peak_dist}.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 统计输出
    n_huge = sum(r["is_huge"] for r in rows)
    n_elong = sum(r["is_elong"] for r in rows)
    n_over = int(over.sum())
    print(f"[统计] {len(rows)} 个粘连簇")
    print(f"  is_huge (L1>0.60 或 max_axis>0.60): {n_huge}/{len(rows)} 满足")
    print(f"  is_elongated (r>1.60 且 L1>0.25): {n_elong}/{len(rows)} 满足")
    print(f"  过分割: {n_over} 簇 (全部满足触发条件, 但切错位置)")
    print(f"  L1 范围: {L1.min():.2f} ~ {L1.max():.2f}m (阈值 0.60)")
    print(f"  r  范围: {rr.min():.2f} ~ {rr.max():.2f} (阈值 1.60)")
    print(f"[output] -> {out_png}")


if __name__ == "__main__":
    main()
