#!/usr/bin/env python3
"""Pheno4D 投影密度试点脚本。

目的：验证 ADPV 框架中 DPVIS 的核心前提——"PCA 主轴投影后密度曲线呈多峰+谷"——
      在番茄/玉米叶片点云上是否仍然成立，从而判断密度峰谷拆分能否泛化到非甘蓝作物。

关键逻辑与 cabbage_pheno/instance/clustering.py 保持一致：
  1. PCA: np.cov + np.linalg.eigh，取最大特征值方向 v1
  2. 投影: scalars = dot(pts - center, v1)
  3. 分 bin 统计密度 (skeleton_bins=15, 最小 bin 0.02m)
  4. find_peaks 找峰 + 谷深比 ratio = valley_h / peak_h

用法:
  python tools/pheno4d_density_pilot.py --sample Maize02 --file M02_0325_a.txt
"""
import os
import argparse

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHENO4D = os.path.join(ROOT, "Pheno4D", "Pheno4D")
OUT = os.path.join(ROOT, "output", "pheno4d_pilot")
os.makedirs(OUT, exist_ok=True)

# 与项目默认 config 一致
SKEL_BINS = 15
MIN_BIN_M = 0.02
MIN_PEAK_DIST_M = 0.35
VALLEY_DEPTH_REL = 0.90

LABEL_LEAF = 2


def load_points(path):
    """读取 Pheno4D 标注文件, 返回 (xyz_m, sem, inst_or_None)。"""
    d = np.loadtxt(path)
    xyz = d[:, :3].astype(np.float64) / 1000.0  # mm -> m
    sem = d[:, 3].astype(int)
    inst = d[:, 4].astype(int) if d.shape[1] >= 5 else None
    return xyz, sem, inst


def voxel_downsample_to(pts, target=300000):
    """体素下采样到约 target 个点。"""
    if len(pts) <= target:
        return pts
    voxel = 0.001
    while True:
        keys = np.floor(pts / voxel).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        if len(idx) <= target or voxel > 0.02:
            return pts[idx]
        voxel *= 1.3


def pca_projection(pts):
    """与 clustering.py 的 _compute_geometric_features 一致。"""
    center = pts.mean(axis=0)
    centered = pts - center
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    v1 = evecs[:, 0]
    proj = np.dot(centered, evecs)
    extents = proj.max(axis=0) - proj.min(axis=0)
    scalars = np.dot(centered, v1)
    return v1, center, scalars, extents


def density_profile(scalars):
    """与 clustering.py 分 bin 逻辑一致, 返回 bin 中心、密度、投影长度。"""
    s_min, s_max = scalars.min(), scalars.max()
    length = s_max - s_min
    nbins = SKEL_BINS
    if length / nbins < MIN_BIN_M:
        nbins = max(5, int(length / MIN_BIN_M))
    bins = np.linspace(s_min, s_max, nbins + 1)
    counts, _ = np.histogram(scalars, bins=bins)
    centers = (bins[:-1] + bins[1:]) / 2
    return centers, counts.astype(float), length, nbins


def analyze_valleys(counts, length, nbins):
    """模拟 clustering.py 的峰谷检测, 返回 peaks 与谷深比。"""
    counts = counts.copy()
    counts[counts < 1] = 0.0
    density_smooth = gaussian_filter1d(counts, sigma=0.5)
    bin_width = length / nbins
    min_dist_bins = max(1, int(MIN_PEAK_DIST_M / bin_width))
    peaks, _ = find_peaks(density_smooth, distance=min_dist_bins,
                          height=np.max(density_smooth) * 0.1)
    valley_info = None
    if len(peaks) >= 2 and peaks[0] < peaks[-1]:
        first_peak, last_peak = peaks[0], peaks[-1]
        seg = density_smooth[first_peak:last_peak + 1]
        valley_idx = first_peak + int(np.argmin(seg))
        valley_h = density_smooth[valley_idx]
        peak_h = min(density_smooth[first_peak], density_smooth[last_peak])
        ratio = valley_h / (peak_h + 1e-6)
        valley_info = (valley_idx, ratio)
    return density_smooth, peaks, valley_info


def plot_profile(ax, scalars, title, color="C0", mark_valley=True):
    centers, counts, length, nbins = density_profile(scalars)
    smooth, peaks, valley = analyze_valleys(counts, length, nbins)
    ax.bar(centers, counts / counts.max(), width=(length / nbins) * 0.9,
           color=color, alpha=0.35, label="density")
    ax.plot(centers, smooth / smooth.max(), color=color, lw=1.8, label="smoothed")
    ax.scatter(centers[peaks], smooth[peaks] / smooth.max(), marker="^",
               color="red", s=50, zorder=5, label=f"peaks ({len(peaks)})")
    if mark_valley and valley is not None:
        vi, ratio = valley
        ax.axvline(centers[vi], color="green", ls="--", lw=1.4)
        ax.text(centers[vi], 0.97, f"valley\nratio={ratio:.2f}", fontsize=7,
                ha="center", va="top", color="green")
    ax.set_title(title, fontsize=8)
    ax.set_ylabel("normalized density")
    ax.set_ylim(0, 1.12)
    ax.set_xlim(centers[0], centers[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="Maize02")
    ap.add_argument("--file", default="M02_0325_a.txt")
    ap.add_argument("--target", type=int, default=300000, help="下采样目标点数")
    ap.add_argument("--organ", type=str, default="leaf", choices=["leaf", "plant"])
    args = ap.parse_args()

    path = os.path.join(PHENO4D, args.sample, args.file)
    print(f"[1] 加载 {args.sample}/{args.file} ...")
    xyz, sem, inst = load_points(path)

    if args.organ == "leaf":
        mask = sem == LABEL_LEAF
        organ_name = f"叶 (sem={LABEL_LEAF})"
    else:
        mask = sem > 0
        organ_name = "整株 (sem>0)"

    pts = xyz[mask]
    print(f"    {organ_name}: {len(pts):,} 点 (原始)")

    pts = voxel_downsample_to(pts, args.target)
    print(f"    下采样后: {len(pts):,} 点")

    print("[2] PCA 主轴投影 ...")
    v1, center, scalars, extents = pca_projection(pts)
    print(f"    主轴 v1 = {v1.round(3)}")
    print(f"    三轴 extent (m): L1={extents[0]:.3f} L2={extents[1]:.3f} L3={extents[2]:.3f}")

    centers, counts, length, nbins = density_profile(scalars)
    smooth, peaks, valley = analyze_valleys(counts, length, nbins)
    print(f"[3] 密度分析: 投影长度 L={length:.3f}m, 分箱 {nbins}, 检测到峰数 = {len(peaks)}")
    if valley is not None:
        print(f"    首个谷深比 ratio = {valley[1]:.3f} (阈值 θ0={VALLEY_DEPTH_REL})")
        verdict = "会触发拆分" if valley[1] < VALLEY_DEPTH_REL else "不拆分 (谷太浅)"
        print(f"    => 谷深比 < 阈值, {verdict}")

    if inst is not None and args.organ == "leaf":
        leaf_inst = inst[mask]
        print(f"    叶片实例 GT 数: {len(set(leaf_inst[leaf_inst > 0]))} 片叶")

    fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
    plot_profile(ax, scalars,
                 f"{args.sample}/{args.file}  {organ_name}\n"
                 f"L={length:.2f}m, n={len(pts):,}, {len(peaks)} peaks")
    ax.set_xlabel("projection along PCA v1 (m)")
    ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Pheno4D 投影密度试点 (min_peak_dist={MIN_PEAK_DIST_M}m, "
                 f"θ0={VALLEY_DEPTH_REL}, bins={SKEL_BINS})", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_png = os.path.join(OUT, f"{args.sample}_{args.file.replace('.txt','')}_{args.organ}.png")
    fig.savefig(out_png, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"\n[DONE] 已保存: {out_png}")


if __name__ == "__main__":
    main()
