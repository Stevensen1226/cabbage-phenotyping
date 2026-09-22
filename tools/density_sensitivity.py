"""
点云密度敏感性实验 —— GIDM (PCA 骨架密度波谷切割) 在稀疏点云下的失效分析。

原理:
  GIDM 将粘连簇沿第一主成分方向投影到一维流形, 统计密度曲线 rho,
  通过"峰-谷"结构判定两株粘连甘蓝的切割点。密度稀疏时:
    1) 谷值模糊/消失 (ratio -> 1)  -> 边界漏检 -> 粘连未拆开 (欠分割)
    2) 峰消失 (< 2 个峰)           -> 不满足切割前提 -> 漏切
    3) 稀疏噪声引入伪谷            -> 边界误检 -> 过分割

本脚本:
  1. 从 evalaute_test/cloudR*.ply (自带 semantic + instance 标签) 中提取
     由 >=2 株甘蓝组成的"粘连簇" (空间连通但实例标签不同)。
  2. 选取一个在全密度下 GIDM 能正确切开的粘连簇。
  3. 以不同保留比例 (100% -> 0.5%) 随机下采样模拟稀疏采集。
  4. 在每个密度下: (a) 用真实 InstanceClusterer._recursive_skeleton_split
     得到实际切割结果; (b) 复现密度曲线并记录峰值/谷值/ratio。
  5. 输出可视化结果与 JSON 汇总。

用法:
  python tools/density_sensitivity.py                    # 自动选簇 + 全流程
  python tools/density_sensitivity.py --discover        # 仅扫描候选粘连簇
  python tools/density_sensitivity.py --cluster cloudR7 --instance 3  # 指定簇
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from cabbage_pheno.instance.clustering import InstanceClusterer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "evalaute_test")
OUT_DIR = os.path.join(ROOT, "output", "density_sensitivity")

# 密度保留比例 (从密到稀)
RATIOS = [1.0, 0.5, 0.10, 0.05]
N_TRIALS = 50  # 每个比例随机下采样重复次数 (汇总用)
SEED = 42


def fmt_pct(r):
    """将保留比例格式化为百分比字符串, 保留足够精度以区分低密度档位。

    `:.0%` 会把 0.005/0.002 都四舍五入成 0%, 导致 0.5% 和 0.2% 面板标题显示为 0%。
    这里对 <1% 的比例保留 1 位小数。
    """
    if r >= 0.01:
        return f"{r * 100:.0f}%"
    return f"{r * 100:.1f}%"


def classify(parts, k):
    """按真值株数 k 对切割结果分类。

    返回 'under' (漏检/欠分割), 'correct' (正确), 'over' (误检/过分割)。
    """
    if parts < k:
        return "under"
    if parts == k:
        return "correct"
    return "over"


def load_cloud_ply(path):
    """读取 evalaute_test/cloudR*.ply (ASCII, 含 semantic/instance 属性)。

    返回 (xyz, semantic, instance)
    """
    with open(path, "r") as f:
        header = []
        n_vertices = 0
        for line in f:
            header.append(line.rstrip("\n"))
            if line.startswith("element vertex "):
                n_vertices = int(line.split()[-1])
            if line.startswith("end_header"):
                break
    data = np.loadtxt(path, skiprows=len(header))
    xyz = data[:, 0:3].astype(np.float64)
    semantic = np.nan_to_num(data[:, 6], nan=0.0).astype(int)
    instance = np.nan_to_num(data[:, 7], nan=-1.0).astype(int)
    return xyz, semantic, instance


def find_sticky_clusters(xyz, instance, eps=0.03):
    """找出"粘连簇": 空间连通 (eps 邻域) 但包含 >=2 个不同实例标签的点集。

    使用 cKDTree BFS 做连通分量, 返回 [{idx, n_points, n_instances, instances}, ...]
    """
    veg = instance > 0
    xyz_v = xyz[veg]
    inst_v = instance[veg]
    idx_v = np.where(veg)[0]

    if len(xyz_v) < 10:
        return []

    tree = cKDTree(xyz_v)
    n = len(xyz_v)
    visited = np.zeros(n, dtype=bool)
    clusters = []

    # 逐点 BFS 找连通分量 (只从尚未访问的点开始)
    for start in range(n):
        if visited[start]:
            continue
        comp = [start]
        visited[start] = True
        head = 0
        while head < len(comp):
            cur = comp[head]
            head += 1
            neigh = tree.query_ball_point(xyz_v[cur], r=eps)
            for nb in neigh:
                if not visited[nb]:
                    visited[nb] = True
                    comp.append(nb)
        comp = np.array(comp)
        if len(comp) < 50:
            continue
        ids = np.unique(inst_v[comp])
        ids = ids[ids > 0]
        if len(ids) < 2:
            continue
        clusters.append({
            "idx": idx_v[comp],
            "n_points": int(len(comp)),
            "n_instances": int(len(ids)),
            "instances": sorted(ids.tolist()),
        })
    clusters.sort(key=lambda c: -c["n_points"])
    return clusters


def replicate_density_profile(points, skel_bins=15, min_peak_dist_m=0.35):
    """逐行复现 InstanceClusterer._recursive_skeleton_split 中的密度分析,
    返回用于可视化的诊断信息。"""
    center = points.mean(axis=0)
    centered = points - center
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]
    v1 = evecs[:, 0]

    scalars = np.dot(centered, v1)
    s_min, s_max = scalars.min(), scalars.max()
    length = s_max - s_min

    nbins = skel_bins
    if length / nbins < 0.02:
        nbins = max(5, int(length / 0.02))

    bins = np.linspace(s_min, s_max, nbins + 1)
    bin_indices = np.digitize(scalars, bins) - 1
    bin_indices = np.clip(bin_indices, 0, nbins - 1)

    counts = np.bincount(bin_indices, minlength=nbins).astype(float)
    density_smooth = gaussian_filter(counts, sigma=0.5)

    bin_width = length / nbins
    min_dist_bins = max(1, int(min_peak_dist_m / bin_width))
    peaks, _ = find_peaks(
        density_smooth, distance=min_dist_bins,
        height=np.max(density_smooth) * 0.1)

    result = {
        "scalars": scalars, "s_min": float(s_min), "s_max": float(s_max),
        "length": float(length), "bins": bins, "counts": counts,
        "density_smooth": density_smooth, "peaks": peaks,
        "nbins": nbins, "bin_width": float(bin_width),
        "n_peaks": int(len(peaks)),
        "valley_idx": None, "valley_ratio": None, "cut": False,
        "threshold": None, "v1": v1, "center": center,
        "valley_indices": [],
    }

    if len(peaks) >= 2:
        first_peak = peaks[0]
        last_peak = peaks[-1]
        if first_peak < last_peak:
            segment = density_smooth[first_peak:last_peak + 1]
            valley_idx = first_peak + int(np.argmin(segment))
            valley_h = density_smooth[valley_idx]
            peak_h = min(density_smooth[first_peak], density_smooth[last_peak])
            ratio = valley_h / (peak_h + 1e-6)
            threshold = 0.90
            if length > 0.60:
                threshold = max(threshold, 0.98)
            result["valley_idx"] = int(valley_idx)
            result["valley_ratio"] = float(ratio)
            result["threshold"] = threshold
            result["cut"] = bool(ratio < threshold)
            result["first_peak"] = int(first_peak)
            result["last_peak"] = int(last_peak)

        # 所有相邻峰之间的谷 = 所有潜在切割位置 (GIDM 递归切割会依次在这些谷落刀)。
        # 4 株粘连 → 3 个谷 → 3 刀; 便于 fig1 直观展示"切了几刀"。
        for a, b in zip(peaks[:-1], peaks[1:]):
            seg = density_smooth[a:b + 1]
            result["valley_indices"].append(int(a + int(np.argmin(seg))))
    return result


def actual_split(clusterer, points):
    """调用真实 GIDM 递归切割, 返回分块数。"""
    parts = clusterer._recursive_skeleton_split(
        points, np.arange(len(points)), depth=0)
    return max(1, len(parts))


def subsample(points, ratio, rng):
    """纯随机稀释 (模拟均匀降低点密度), 不做点数下限保护。"""
    if ratio >= 1.0:
        return points
    keep = rng.random(len(points)) < ratio
    return points[keep]


def _draw_hist_panel(ax, row, dist, k, edges, xticks):
    """在给定 ax 上绘制单个密度的分块数分布直方图。"""
    vals = np.array(dist)
    counts, _ = np.histogram(vals, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    colors = []
    for c in centers:
        if c < k:
            colors.append("#C0392B")   # 欠分割
        elif c == k:
            colors.append("#1E8449")   # 正确
        else:
            colors.append("#F39C12")   # 过分割
    ax.bar(centers, counts, width=0.8, color=colors, alpha=0.85)
    ax.axvline(k, color="black", ls="--", lw=1, alpha=0.5)
    mae = row.get("mae", float("nan"))
    ax.set_title(f"ratio={fmt_pct(row['ratio'])}\nMAE={mae:.2f}",
                 fontsize=10)
    ax.set_xlabel("parts (#segments)")
    ax.set_ylabel("trials")
    ax.set_xticks(xticks)
    ax.set_ylim(0, max(1, counts.max()) * 1.18)
    ax.grid(axis="y", alpha=0.3)


def make_parts_histogram(rows, parts_dists, out_png, k, panels_dir=None):
    """fig1: 每个密度下 GIDM 切出块数的分布直方图 (完整展示 N_TRIALS 次采样)。

    绿色 = correct (parts == k), 橙色 = over-split (parts > k),
    红色 = under-split (parts < k)。直方图完整呈现每个密度的分割结果分布,
    而非单个代表性采样, 避免"只画错误案例、忽视正确案例"的偏颇。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    # 确定横轴范围, 覆盖所有密度出现的块数, 并始终包含真值 k
    all_parts = [p for d in parts_dists for p in d]
    lo = min(all_parts) if all_parts else k
    hi = max(all_parts) if all_parts else k
    lo = min(lo, k)
    hi = max(hi, k)
    edges = np.arange(lo - 0.5, hi + 1.5, 1.0)
    xticks = np.arange(lo, hi + 1)

    for i, (row, dist) in enumerate(zip(rows, parts_dists)):
        _draw_hist_panel(axes[i], row, dist, k, edges, xticks)

    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.suptitle(
        f"GIDM split outcome distribution over {N_TRIALS} random subsamples "
        f"(k={k} plants, green=correct orange=over-split red=under-split)",
        fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 每个密度单独保存一张
    if panels_dir:
        for row, dist in zip(rows, parts_dists):
            f = plt.figure(figsize=(4.6, 3.6))
            ax = f.add_subplot(111)
            _draw_hist_panel(ax, row, dist, k, edges, xticks)
            f.tight_layout()
            fname = f"fig1_ratio_{fmt_pct(row['ratio']).replace('.', 'p').replace('%', '')}.png"
            f.savefig(os.path.join(panels_dir, fname), dpi=150, bbox_inches="tight")
            plt.close(f)


def make_summary_plot(rows, out_png, k, panels_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ratios = [r["ratio"] for r in rows]
    mean_ratio = [r["mean_valley_ratio"] for r in rows]
    std_ratio = [r["std_valley_ratio"] for r in rows]
    miss_rate = [r["miss_rate"] for r in rows]
    false_rate = [r["false_rate"] for r in rows]
    correct_rate = [r["correct_rate"] for r in rows]
    mae = [r.get("mae", float("nan")) for r in rows]
    n_peaks = [r["mean_n_peaks"] for r in rows]
    n_parts = [r["mean_n_parts"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))

    # (a) 谷值 ratio vs 密度 —— y 轴收紧到 [0.35, 1.02], 兼顾数据线与阈值线
    ax = axes[0]
    ax.errorbar(ratios, mean_ratio, yerr=std_ratio, marker="o", ms=5, capsize=3,
                color="#C0392B", lw=1.8, label="Valley ratio $\\eta$ (mean$\\pm$std)")
    ax.axhline(0.90, ls="--", color="gray", lw=1, label="threshold $\\theta_0=0.90$")
    ax.axhline(0.98, ls=":", color="gray", lw=1, label="$\\theta=0.98$ ($L>0.6$m)")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_ylim(0.35, 1.02)
    ax.set_xlabel("Point retention ratio (density)")
    ax.set_ylabel("Valley ratio\n$\\eta = \\rho_v / \\min(\\rho_{p})$")
    ax.set_title("(a) Valley depth vs. density")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=3, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (b) MAE(count) vs 密度 —— y 轴加负 margin, 避免 0 值点贴底
    ax = axes[1]
    ax.plot(ratios, mae, marker="o", ms=5, color="#2471A3", lw=2,
            label="MAE (count)")
    ax.axhline(0.0, ls="--", color="gray", lw=1)
    ax.set_xscale("log")
    ax.invert_xaxis()
    mae_max = max(mae) if mae else 0.0
    ax.set_ylim(-0.03, max(mae_max * 1.3, 0.10))
    ax.set_xlabel("Point retention ratio (density)")
    ax.set_ylabel("MAE (count) = mean|parts - k|")
    ax.set_title(f"(b) Counting error vs. density ({k} plants)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=1, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (c) 峰数 & 分块数 vs 密度 —— y 轴收紧到 k 附近, 拉开数据线与真值线
    ax = axes[2]
    ax.plot(ratios, n_peaks, marker="^", ms=5, color="#1E8449", lw=1.8,
            label="mean #peaks")
    ax.plot(ratios, n_parts, marker="v", ms=5, color="#7D3C98", lw=1.8,
            label="mean #parts (GIDM)")
    ax.axhline(float(k), ls="--", color="gray", lw=1, label=f"ground truth = {k} plants")
    ax.set_xscale("log")
    ax.invert_xaxis()
    lo = min(min(n_peaks), min(n_parts), float(k))
    hi = max(max(n_peaks), max(n_parts), float(k))
    ax.set_ylim(lo - 0.25, hi + 0.25)
    ax.set_xlabel("Point retention ratio (density)")
    ax.set_ylabel("Count")
    ax.set_title("(c) Peaks / resulting parts vs. density")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=3, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("GIDM sensitivity to point-cloud density (sparser -> valley blurs -> boundary miss/false)",
                 fontsize=12, fontweight="bold", y=0.98)
    # 底部留出图例空间
    fig.subplots_adjust(top=0.82, bottom=0.22, wspace=0.28)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 三个子图分别单独保存
    if panels_dir:
        panels = {
            "fig2a_valley_ratio": lambda ax: (
                ax.errorbar(ratios, mean_ratio, yerr=std_ratio, marker="o", ms=5, capsize=3,
                            color="#C0392B", lw=1.8, label="Valley ratio $\\eta$ (mean$\\pm$std)"),
                ax.axhline(0.90, ls="--", color="gray", lw=1, label="threshold $\\theta_0=0.90$"),
                ax.axhline(0.98, ls=":", color="gray", lw=1, label="$\\theta=0.98$ ($L>0.6$m)"),
                ax.set_xscale("log"), ax.invert_xaxis(), ax.set_ylim(0.35, 1.02),
                ax.set_xlabel("Point retention ratio (density)"),
                ax.set_ylabel("Valley ratio\n$\\eta = \\rho_v / \\min(\\rho_{p})$"),
                ax.set_title("(a) Valley depth vs. density"),
                ax.grid(alpha=0.3),
                ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.20),
                          ncol=3, frameon=False),
                ax.spines["top"].set_visible(False), ax.spines["right"].set_visible(False)),
            "fig2b_mae": lambda ax: (
                ax.plot(ratios, mae, marker="o", ms=5, color="#2471A3", lw=2,
                        label="MAE (count)"),
                ax.axhline(0.0, ls="--", color="gray", lw=1),
                ax.set_xscale("log"), ax.invert_xaxis(),
                ax.set_ylim(-0.03, max(max(mae) * 1.3 if mae else 0.0, 0.10)),
                ax.set_xlabel("Point retention ratio (density)"),
                ax.set_ylabel("MAE (count) = mean|parts - k|"),
                ax.set_title(f"(b) Counting error vs. density ({k} plants)"),
                ax.grid(alpha=0.3),
                ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.20),
                          ncol=1, frameon=False),
                ax.spines["top"].set_visible(False), ax.spines["right"].set_visible(False)),
            "fig2c_peaks_parts": lambda ax: (
                ax.plot(ratios, n_peaks, marker="^", ms=5, color="#1E8449", lw=1.8, label="mean #peaks"),
                ax.plot(ratios, n_parts, marker="v", ms=5, color="#7D3C98", lw=1.8, label="mean #parts (GIDM)"),
                ax.axhline(float(k), ls="--", color="gray", lw=1, label=f"ground truth = {k} plants"),
                ax.set_xscale("log"), ax.invert_xaxis(),
                ax.set_ylim(min(min(n_peaks), min(n_parts), float(k)) - 0.25,
                            max(max(n_peaks), max(n_parts), float(k)) + 0.25),
                ax.set_xlabel("Point retention ratio (density)"),
                ax.set_ylabel("Count"),
                ax.set_title("(c) Peaks / resulting parts vs. density"),
                ax.grid(alpha=0.3),
                ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.20),
                          ncol=3, frameon=False),
                ax.spines["top"].set_visible(False), ax.spines["right"].set_visible(False)),
        }
        for fname, draw in panels.items():
            f = plt.figure(figsize=(5.6, 4.6))
            ax = f.add_subplot(111)
            draw(ax)
            f.subplots_adjust(bottom=0.20, top=0.90)
            f.savefig(os.path.join(panels_dir, fname + ".png"), dpi=150, bbox_inches="tight")
            plt.close(f)


def _draw_profile_panel(ax, p):
    """在给定 ax 上绘制单个密度的一维密度曲线。"""
    bin_centers = (p["bins"][:-1] + p["bins"][1:]) / 2
    smooth = p["density_smooth"]
    ax.bar(bin_centers, p["counts"], width=p["bin_width"] * 0.9,
           color="lightgray", alpha=0.6)
    ax.plot(bin_centers, smooth, "k-", lw=1.8, label="smoothed $\\tilde{\\rho}$")
    peaks = p["peaks"]
    if len(peaks):
        ax.plot(bin_centers[peaks], smooth[peaks], "^", color="#1E8449",
                ms=9, label="peaks")
    valleys = p.get("valley_indices", [])
    if valleys:
        vs = np.asarray(valleys)
        ax.plot(bin_centers[vs], smooth[vs], "v", color="#C0392B",
                ms=9, label=f"valleys ({len(vs)})")
        for vi in valleys:
            ax.axvline(bin_centers[vi], color="#C0392B", ls="--",
                       lw=1.0, alpha=0.6)
    eta = p['valley_ratio'] if p['valley_ratio'] is not None else float('nan')
    title = (f"ratio={fmt_pct(p['ratio'])}  N={p['n_points']}\n"
             f"$\\eta$={eta:.2f}  peaks={p['n_peaks']}")
    ax.set_title(title, fontsize=10, linespacing=1.4)
    ax.set_xlabel("Projection $s$ (m)")
    ax.set_ylabel("Count")


def make_profile_plot(panels, out_png, k, panels_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(panels)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for i, p in enumerate(panels):
        _draw_profile_panel(axes[i], p)

    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.suptitle("1D density profile along principal axis at decreasing density",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    # 每个密度单独保存一张
    if panels_dir:
        for p in panels:
            f = plt.figure(figsize=(4.6, 3.6))
            ax = f.add_subplot(111)
            _draw_profile_panel(ax, p)
            f.tight_layout()
            fname = f"fig3_ratio_{fmt_pct(p['ratio']).replace('.', 'p').replace('%', '')}.png"
            f.savefig(os.path.join(panels_dir, fname), dpi=150, bbox_inches="tight")
            plt.close(f)


def run_experiment(clusterer, points, out_dir, k):
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    parts_dists = []
    panels = []

    for ratio in RATIOS:
        ratios_vals = []
        peaks_vals = []
        parts_vals = []
        miss = correct = false = valid = 0

        for t in range(N_TRIALS):
            rng = np.random.default_rng(SEED + int(ratio * 100000) + t)
            sub = subsample(points, ratio, rng)
            if len(sub) < 20:
                continue
            prof = replicate_density_profile(sub)
            parts = actual_split(clusterer, sub)
            valid += 1
            vr = prof["valley_ratio"] if prof["valley_ratio"] is not None else 1.0
            ratios_vals.append(vr)
            peaks_vals.append(prof["n_peaks"])
            parts_vals.append(parts)
            if parts < k:
                miss += 1
            elif parts == k:
                correct += 1
            else:
                false += 1

        mean_ratio = float(np.mean(ratios_vals)) if ratios_vals else float("nan")
        std_ratio = float(np.std(ratios_vals)) if ratios_vals else float("nan")
        mean_peaks = float(np.mean(peaks_vals)) if peaks_vals else float("nan")
        mean_parts = float(np.mean(parts_vals)) if parts_vals else float("nan")
        miss_rate = miss / valid if valid else float("nan")
        correct_rate = correct / valid if valid else float("nan")
        false_rate = false / valid if valid else float("nan")

        # 收集该密度的 parts 分布 (用于 fig1 直方图, 完整展示 50 次采样结果)
        parts_dists.append(list(parts_vals))
        parts_hist = {int(p): int(c) for p, c in Counter(parts_vals).items()}
        # MAE(count) = mean(|parts - k|)
        mae = sum(abs(int(p) - k) * c for p, c in parts_hist.items()) / valid if valid else float("nan")

        # 密度曲线 fig3 的代表性面板: 选 valley_ratio 最接近均值的采样 (纯机制展示)
        best_t = int(np.argmin(np.abs(np.array(ratios_vals) - mean_ratio)))
        rng = np.random.default_rng(SEED + int(ratio * 100000) + best_t)
        sub = subsample(points, ratio, rng)
        rep_panel = replicate_density_profile(sub)

        rows.append({
            "ratio": ratio, "n_points": len(sub), "n_trials": valid,
            "mean_valley_ratio": mean_ratio, "std_valley_ratio": std_ratio,
            "mean_n_peaks": mean_peaks, "mean_n_parts": mean_parts,
            "miss_rate": miss_rate, "correct_rate": correct_rate,
            "false_rate": false_rate, "parts_hist": parts_hist, "mae": mae,
        })
        rep_panel["ratio"] = ratio
        rep_panel["n_points"] = len(sub)
        panels.append(rep_panel)

    panels_dir = os.path.join(out_dir, "panels")
    os.makedirs(panels_dir, exist_ok=True)
    make_parts_histogram(
        rows, parts_dists,
        os.path.join(out_dir, "fig1_parts_distribution.png"), k,
        panels_dir=panels_dir)
    make_summary_plot(rows, os.path.join(out_dir, "fig2_summary.png"), k,
                      panels_dir=panels_dir)
    make_profile_plot(panels, os.path.join(out_dir, "fig3_density_profiles.png"), k,
                      panels_dir=panels_dir)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    return rows


def discover(eps=0.03):
    files = sorted(
        f for f in os.listdir(DATA_DIR)
        if f.startswith("cloudR") and f.endswith(".ply")
        and not f.endswith("_sor.ply"))
    all_clusters = []
    for fname in files:
        xyz, sem, inst = load_cloud_ply(os.path.join(DATA_DIR, fname))
        clusters = find_sticky_clusters(xyz, inst, eps=eps)
        for c in clusters:
            c["file"] = fname
        all_clusters.extend(clusters)

    all_clusters.sort(key=lambda c: -c["n_points"])
    print(f"{'file':<14}{'n_points':>10}{'n_inst':>7}  instances")
    for c in all_clusters[:40]:
        print(f"{c['file']:<14}{c['n_points']:>10}{c['n_instances']:>7}  {c['instances']}")
    return all_clusters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--cluster", default=None, help="指定 cloudR 文件名, 如 cloudR7")
    ap.add_argument("--instances", default=None,
                    help="按实例标签精确匹配粘连簇, 逗号分隔, 如 '2,3,4,5'")
    ap.add_argument("--eps", type=float, default=0.03)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "default.yaml")))
    clusterer = InstanceClusterer(cfg)

    if args.discover:
        discover(eps=args.eps)
        return

    # 默认目标: cloudR10 的 4 株连续粘连 [2,3,4,5] —— 经典闭冠行栽场景,
    # 需要递归切割 3 刀, 比两株粘连更能体现密度稀疏时的级联失效。
    target_file = args.cluster or "cloudR10"
    target_instances = [int(x) for x in (args.instances or "2,3,4,5").split(",")]

    fname = target_file if target_file.endswith(".ply") else target_file + ".ply"
    xyz, sem, inst = load_cloud_ply(os.path.join(DATA_DIR, fname))
    clusters = find_sticky_clusters(xyz, inst, eps=args.eps)

    c = None
    for cc in clusters:
        if set(cc["instances"]) == set(target_instances):
            c = cc
            break
    if c is None:
        print(f"未在 {fname} 找到实例 {target_instances} 的粘连簇; 可用 --discover 查看候选")
        return

    k = len(target_instances)
    points = xyz[c["idx"]]
    # 密度敏感性实验输出到 single_cluster/density/
    out_dir = os.path.join(OUT_DIR, "single_cluster", "density")
    print(f"选择 {fname} 粘连簇: {c['n_points']} 点, "
          f"{c['n_instances']} 株 {c['instances']} (真值 k={k})")

    print("开始密度敏感性实验 ...")
    rows = run_experiment(clusterer, points, out_dir, k)

    print("\n=== 汇总 ===")
    print(f"{'ratio':>8}{'N_pts':>9}{'eta':>8}{'std':>7}{'peaks':>7}{'parts':>7}"
          f"{'miss':>7}{'corr':>7}{'false':>7}")
    for r in rows:
        print(f"{r['ratio']:>8.1%}{r['n_points']:>9}"
              f"{r['mean_valley_ratio']:>8.2f}{r['std_valley_ratio']:>7.2f}"
              f"{r['mean_n_peaks']:>7.1f}{r['mean_n_parts']:>7.1f}"
              f"{r['miss_rate']:>7.2f}{r['correct_rate']:>7.2f}"
              f"{r['false_rate']:>7.2f}")
    print(f"\n输出目录: {out_dir}")


if __name__ == "__main__":
    main()
