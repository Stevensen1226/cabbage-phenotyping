#!/usr/bin/env python3
"""Pheno4D 泛化试点：番茄/玉米叶片投影密度曲线可视化。

目的：在决定"叶片粘连拆分"泛化方向之前，先直观确认——
  1. 叶点云沿 PCA 主轴投影后的密度曲线是否呈现可拆分的多峰结构；
  2. 用 DBSCAN 粗聚类得到的"粘连叶簇"里，DPVIS 式投影密度长什么样。

用法:
  python tools/pheno4d_density_probe.py --txt Pheno4D/Pheno4D/Tomato01/T01_0305_a.txt
  python tools/pheno4d_density_probe.py --txt Pheno4D/Pheno4D/Maize01/M01_0313_a.txt --leaf-label 2
"""
import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter

try:
    from sklearn.cluster import DBSCAN
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output", "pheno4d_probe")
os.makedirs(OUT, exist_ok=True)

# Pheno4D 语义标签（官方）：0=soil, 1=stem, 2=leaf, 番茄另有 3(花/果实等)
LEAF_LABEL = 2


# --------------------------------------------------------------------------
# 与 cabbage_pheno/instance/clustering.py 一致的 PCA 特征计算
# --------------------------------------------------------------------------
def compute_geometric_features(points):
    """返回 center, evecs(按特征值降序), extents(主轴长度 L1/L2/L3)。"""
    center = points.mean(axis=0)
    centered = points - center
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    proj = np.dot(centered, evecs)
    extents = proj.max(axis=0) - proj.min(axis=0)
    return center, evecs, extents


def project_density(points, nbins=40, sigma=1.0):
    """沿 PCA 第一主方向投影，返回 (scalars, bin_centers, counts, density_smooth, length, bin_width)。"""
    center, evecs, extents = compute_geometric_features(points)
    v1 = evecs[:, 0]
    scalars = np.dot(points - center, v1)
    s_min, s_max = scalars.min(), scalars.max()
    length = s_max - s_min
    if length <= 1e-9:
        return None
    counts, edges = np.histogram(scalars, bins=nbins, range=(s_min, s_max))
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    density = gaussian_filter(counts.astype(float), sigma=sigma)
    bin_width = length / nbins
    return {
        "scalars": scalars,
        "bin_centers": bin_centers,
        "counts": counts,
        "density": density,
        "length": length,
        "bin_width": bin_width,
        "s_min": s_min,
        "s_max": s_max,
        "extents": extents,
        "center": center,
        "v1": v1,
    }


def plot_profile(ax, prof, title, min_peak_dist_m=0.02):
    """画一条投影密度曲线，标注 find_peaks 找到的峰。"""
    bc = prof["bin_centers"]
    dens = prof["density"]
    ax.plot(bc, dens, "-", color="#3b6fb6", linewidth=1.6, label="smoothed density")
    ax.bar(bc, prof["counts"], width=prof["bin_width"] * 0.9,
           color="#cfe0f0", alpha=0.6, label="raw bin counts")

    min_dist_bins = max(1, int(min_peak_dist_m / prof["bin_width"]))
    peaks, _ = find_peaks(dens, distance=min_dist_bins, height=dens.max() * 0.1)
    ax.plot(bc[peaks], dens[peaks], "v", color="#c0392b", markersize=7, label=f"{len(peaks)} peak(s)")

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Projection along PCA major axis (m)")
    ax.set_ylabel("Point count per bin")
    ax.grid(axis="y", ls=":", alpha=0.35)
    ax.legend(frameon=False, fontsize=7, loc="upper right")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt", required=True, help="Pheno4D .txt 点云路径 (x y z label)")
    ap.add_argument("--leaf-label", type=int, default=LEAF_LABEL)
    ap.add_argument("--eps", type=float, default=0.01, help="DBSCAN eps (m)")
    ap.add_argument("--min-samples", type=int, default=100)
    ap.add_argument("--min-cluster-pts", type=int, default=2000,
                    help="只对点数超过该阈值的簇画投影密度")
    ap.add_argument("--nbins", type=int, default=40)
    args = ap.parse_args()

    txt = os.path.abspath(args.txt)
    if not os.path.exists(txt):
        print(f"[ERR] 文件不存在: {txt}")
        sys.exit(1)

    t0 = time.time()
    data = np.loadtxt(txt, dtype=np.float32)
    print(f"[load] {txt}\n  {data.shape[0]:,} 点, 耗时 {time.time()-t0:.1f}s")

    xyz = data[:, :3] / 1000.0  # mm -> m
    label = data[:, 3].astype(np.int32)
    leaf = xyz[label == args.leaf_label]
    print(f"[info] 叶点(label={args.leaf_label}): {leaf.shape[0]:,} / {xyz.shape[0]:,}")

    if leaf.shape[0] < 50:
        print("[ERR] 叶点过少，请检查 --leaf-label")
        sys.exit(1)

    # ---------- 1) 整体叶点云投影 ----------
    prof_all = project_density(leaf, nbins=args.nbins)
    if prof_all is None:
        print("[ERR] 整体叶点投影长度为零")
        sys.exit(1)
    ext = prof_all["extents"]
    print(f"[info] 整体叶点 PCA 主轴长度: L1={ext[0]:.3f}m  L2={ext[1]:.3f}m  L3={ext[2]:.3f}m")

    # ---------- 2) DBSCAN 粗聚类 ----------
    cluster_sizes = []
    clusters = []
    if _HAS_SKLEARN:
        db = DBSCAN(eps=args.eps, min_samples=args.min_samples)
        cl = db.fit_predict(leaf)
        n_clusters = len(set(cl)) - (1 if -1 in cl else 0)
        print(f"[info] DBSCAN(eps={args.eps}m, min_samples={args.min_samples}) -> {n_clusters} 个簇")
        for cid in sorted(set(cl)):
            if cid == -1:
                continue
            idx = cl == cid
            sz = int(idx.sum())
            cluster_sizes.append((cid, sz, leaf[idx]))
        cluster_sizes.sort(key=lambda x: -x[1])
        for cid, sz, _ in cluster_sizes:
            print(f"    簇 {cid:3d}: {sz:>8,} 点")
        # 取点数超过阈值、且最大的前 6 个簇
        big = [c for c in cluster_sizes if c[1] >= args.min_cluster_pts][:6]
        clusters = big
    else:
        print("[warn] sklearn 不可用，跳过 DBSCAN 聚类，仅画整体投影")

    # ---------- 3) 绘图 ----------
    n_panels = 1 + len(clusters)
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 2.8 * n_panels), dpi=150)

    if n_panels == 1:
        axes = [axes]

    plot_profile(axes[0], prof_all,
                 f"All leaf points (N={leaf.shape[0]:,}), L1={ext[0]:.2f}m",
                 min_peak_dist_m=0.02)

    for i, (cid, sz, pts) in enumerate(clusters, start=1):
        prof = project_density(pts, nbins=args.nbins)
        if prof is None:
            continue
        e = prof["extents"]
        plot_profile(axes[i], prof,
                     f"Cluster {cid} (N={sz:,}), L1={e[0]:.2f}m, L2={e[1]:.2f}m",
                     min_peak_dist_m=0.02)

    fig.tight_layout()
    stem = os.path.splitext(os.path.basename(txt))[0]
    out_png = os.path.join(OUT, f"{stem}_leaf_density.png")
    fig.savefig(out_png, dpi=150, facecolor="white", bbox_inches="tight")
    print(f"[save] {out_png}")


if __name__ == "__main__":
    main()
