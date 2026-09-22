"""
GIDM 密度/超参数敏感性 —— 自包含绘图脚本 (无需任何数据文件)。

本脚本将实验得到的全部数值结果硬编码在内, 只依赖 numpy + matplotlib,
可直接发给他人用于调整图片样式、配色、字号、布局等, 无需点云数据、
json 文件或项目源码。

生成图片:
  fig1_parts_distribution.png   单簇: 各密度下 GIDM 分块数分布
  fig2_summary.png              单簇: 密度敏感性汇总 (η / 漏检正确误检 / 峰数分块数)
  fig3_density_profiles.png     单簇: 各密度下一维密度曲线
  fig4_hyperparam_valley_depth.png   单簇: θ0 敏感性热图
  fig5_hyperparam_min_peak_dist.png  单簇: 谷底跨度敏感性热图
  fig6_hyperparam_overview.png       单簇: 超参数 small-multiples 总览
  figA_density_multi.png        多簇: 密度敏感性跨簇汇总
  figB_hyperparam_mpd_multi.png 多簇: 谷底跨度跨簇敏感性
  figC_hyperparam_vdr_multi.png 多簇: θ0 跨簇敏感性

用法:
  python plot_figures_standalone.py [输出目录]
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================================
# 实验数据 (硬编码, 来自 GIDM 密度/超参数敏感性实验)
# =====================================================================

# 密度保留比例
RATIOS = [1.0, 0.5, 0.25, 0.10, 0.05, 0.02, 0.01, 0.005]


def fmt_pct(r):
    return f"{r * 100:.0f}%" if r >= 0.01 else f"{r * 100:.1f}%"


# ---- 单簇密度敏感性汇总 (cloudR10, 实例 [2,3,4,5], k=4, 50 次采样) ----
# 每个密度的 [n_points, mean_eta, std_eta, mean_peaks, mean_parts,
#              miss_rate, correct_rate, false_rate, parts_hist]
DENSITY_SUMMARY = [
    [39693, 0.4837, 0.0000, 4.0, 4.0, 0.00, 1.00, 0.00, {4: 50}],
    [19938, 0.4872, 0.0129, 4.0, 4.0, 0.00, 1.00, 0.00, {4: 50}],
    [9955,  0.4851, 0.0178, 4.0, 4.0, 0.00, 1.00, 0.00, {4: 50}],
    [3922,  0.4926, 0.0329, 4.0, 4.0, 0.00, 1.00, 0.00, {4: 50}],
    [2038,  0.4897, 0.0556, 3.94, 4.0, 0.02, 0.96, 0.02, {4: 48, 5: 1, 3: 1}],
    [818,   0.4658, 0.0593, 3.84, 4.2, 0.04, 0.76, 0.20, {4: 38, 5: 8, 6: 2, 3: 2}],
    [403,   0.4810, 0.1113, 3.92, 4.26, 0.04, 0.70, 0.26, {5: 11, 4: 35, 6: 2, 3: 2}],
    [195,   0.4583, 0.1335, 3.70, 4.10, 0.04, 0.80, 0.16, {4: 40, 1: 1, 5: 7, 3: 1, 6: 1}],
]

# ---- 单簇密度曲线面板 (fig3): 每个密度的代表性采样 ----
# 每项: [ratio, n_points, eta, n_peaks, counts(15), density_smooth(15),
#        peaks, valley_indices]
DENSITY_PROFILES = [
    [1.0, 39693, 0.4837, 4,
     [1754, 3242, 3267, 2662, 2821, 3335, 2312, 1356, 3203, 3425, 2940, 1185, 2584, 3343, 2264],
     [1913.2, 3085.7, 3199.4, 2743.7, 2858.8, 3170.7, 2319.5, 1655.4, 3029.7, 3348.6, 2804.8, 1521.9, 2515.9, 3146.5, 2379.2],
     [2, 5, 9, 13], [3, 7, 11]],
    [0.5, 19938, 0.4871, 4,
     [889, 1654, 1649, 1340, 1365, 1674, 1163, 710, 1624, 1687, 1495, 591, 1306, 1664, 1127],
     [970.8, 1571.7, 1616.4, 1375.7, 1395.3, 1586.4, 1169.3, 856.0, 1533.3, 1659.3, 1419.2, 763.9, 1268.0, 1568.3, 1184.4],
     [2, 5, 9, 13], [3, 7, 11]],
    [0.25, 9955, 0.4859, 4,
     [436, 831, 785, 676, 697, 821, 576, 342, 810, 841, 764, 293, 645, 870, 568],
     [478.2, 783.9, 778.2, 689.9, 708.0, 781.6, 577.3, 417.0, 763.4, 829.2, 722.0, 380.9, 631.5, 813.7, 600.2],
     [1, 5, 9, 13], [3, 7, 11]],
    [0.1, 3922, 0.4909, 4,
     [202, 302, 303, 249, 280, 321, 237, 124, 325, 323, 325, 107, 253, 333, 238],
     [212.7, 291.4, 297.1, 258.1, 281.1, 307.6, 233.9, 157.5, 303.4, 323.3, 301.6, 145.9, 246.0, 314.3, 248.1],
     [2, 5, 9, 13], [3, 7, 11]],
    [0.05, 2038, 0.4893, 4,
     [107, 166, 145, 109, 150, 186, 124, 65, 167, 202, 150, 61, 122, 178, 106],
     [113.3, 157.5, 143.4, 117.2, 149.5, 175.5, 124.3, 82.2, 159.9, 192.7, 146.1, 77.0, 121.5, 164.3, 113.7],
     [1, 5, 9, 13], [3, 7, 11]],
    [0.02, 818, 0.4667, 4,
     [34, 66, 68, 44, 60, 58, 50, 32, 64, 74, 67, 22, 56, 74, 49],
     [37.4, 62.8, 65.2, 48.3, 58.1, 57.4, 48.9, 37.3, 61.7, 72.2, 63.0, 30.4, 54.3, 69.4, 51.7],
     [2, 4, 9, 13], [3, 7, 11]],
    [0.01, 403, 0.4766, 4,
     [22, 46, 30, 23, 27, 29, 25, 21, 25, 33, 22, 14, 22, 35, 29],
     [24.6, 41.7, 31.0, 24.2, 26.8, 28.4, 25.0, 21.9, 25.4, 31.0, 22.3, 15.7, 22.5, 33.0, 29.6],
     [1, 5, 9, 13], [3, 7, 11]],
    [0.005, 195, 0.4612, 5,
     [6, 20, 11, 12, 16, 11, 14, 4, 22, 15, 14, 10, 11, 16, 13],
     [7.5, 17.5, 12.1, 12.3, 15.0, 11.9, 12.6, 7.0, 19.3, 15.6, 13.7, 10.5, 11.4, 15.1, 13.3],
     [1, 4, 6, 8, 13], [2, 5, 7, 11]],
]

# ---- 单簇超参数敏感性 (cloudR10, k=4, 25 次采样) ----
VDR_GRID = [0.80, 0.85, 0.90, 0.95, 0.98, 1.00, 1.05, 1.10]
VDR_CORRECT = [
    [1.0, 1.0, 1.0, 1.0, 1.0, 0.80, 0.80, 0.72],
    [1.0, 1.0, 1.0, 1.0, 1.0, 0.80, 0.80, 0.72],
    [1.0, 1.0, 1.0, 1.0, 1.0, 0.76, 0.80, 0.72],
    [1.0, 1.0, 1.0, 1.0, 1.0, 0.72, 0.80, 0.72],
    [1.0, 1.0, 1.0, 0.96, 0.96, 0.68, 0.80, 0.72],
    [1.0, 1.0, 1.0, 0.96, 0.96, 0.68, 0.80, 0.72],
    [1.0, 1.0, 1.0, 0.96, 0.96, 0.68, 0.80, 0.72],
    [1.0, 1.0, 1.0, 0.96, 0.96, 0.68, 0.80, 0.72],
]
VDR_PARTS = [
    [4.0, 4.0, 4.0, 4.0, 4.0, 4.24, 4.24, 4.04],
    [4.0, 4.0, 4.0, 4.0, 4.0, 4.24, 4.24, 4.04],
    [4.0, 4.0, 4.0, 4.0, 4.0, 4.28, 4.24, 4.04],
    [4.0, 4.0, 4.0, 4.0, 4.0, 4.32, 4.24, 4.04],
    [4.0, 4.0, 4.0, 4.04, 4.04, 4.36, 4.24, 4.04],
    [4.0, 4.0, 4.0, 4.04, 4.04, 4.36, 4.24, 4.04],
    [4.0, 4.0, 4.0, 4.04, 4.04, 4.36, 4.24, 4.04],
    [4.0, 4.0, 4.0, 4.04, 4.04, 4.36, 4.24, 4.04],
]
MPD_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
MPD_CORRECT = [
    [1.0, 0.92, 0.68, 0.24, 0.04, 0.04, 0.00, 0.20],
    [1.0, 0.92, 0.68, 0.24, 0.04, 0.00, 0.00, 0.20],
    [1.0, 0.96, 0.76, 0.44, 0.20, 0.00, 0.00, 0.24],
    [1.0, 0.96, 0.92, 0.80, 0.52, 0.04, 0.20, 0.36],
    [1.0, 1.0, 1.0, 0.96, 0.88, 0.36, 0.48, 0.48],
    [1.0, 1.0, 1.0, 1.0, 1.0, 0.76, 0.80, 0.72],
    [1.0, 1.0, 1.0, 0.96, 1.0, 0.76, 0.84, 0.64],
    [1.0, 1.0, 0.88, 0.92, 0.88, 0.72, 0.80, 0.56],
    [1.0, 0.72, 0.80, 0.92, 0.88, 0.72, 0.76, 0.52],
]
MPD_PARTS = [
    [4.0, 4.16, 4.52, 5.32, 6.64, 6.88, 6.80, 4.84],
    [4.0, 4.16, 4.56, 5.36, 6.48, 7.00, 6.64, 4.80],
    [4.0, 4.04, 4.32, 4.76, 5.68, 6.52, 6.36, 4.88],
    [4.0, 4.04, 4.08, 4.28, 4.72, 5.60, 5.36, 4.60],
    [4.0, 4.0, 4.0, 4.04, 4.12, 4.76, 4.68, 4.36],
    [4.0, 4.0, 4.0, 4.0, 4.0, 4.28, 4.24, 4.04],
    [4.0, 4.0, 4.0, 3.96, 4.0, 3.92, 3.92, 3.88],
    [4.0, 4.0, 3.88, 3.92, 3.88, 3.76, 3.88, 3.60],
    [4.0, 3.72, 3.80, 3.92, 3.88, 3.64, 3.76, 3.52],
]

# ---- 多簇跨簇验证 (25 个粘连簇, 2~5 株) ----
# 密度汇总: 每个密度的 [correct, over, under]
MULTI_DENSITY = [
    [0.720, 0.120, 0.160],
    [0.746, 0.113, 0.141],
    [0.743, 0.114, 0.143],
    [0.666, 0.189, 0.145],
    [0.568, 0.310, 0.122],
    [0.464, 0.417, 0.119],
    [0.459, 0.411, 0.130],
    [0.518, 0.186, 0.296],
]
MULTI_MPD_CORRECT = [
    [0.28, 0.2533, 0.2347, 0.1333, 0.0853, 0.0293, 0.1307, 0.3627],
    [0.32, 0.3013, 0.2213, 0.1173, 0.0853, 0.0373, 0.1120, 0.3627],
    [0.48, 0.3840, 0.3867, 0.2213, 0.1520, 0.0560, 0.1707, 0.3520],
    [0.56, 0.5573, 0.5520, 0.4667, 0.4107, 0.2720, 0.3173, 0.4080],
    [0.72, 0.7547, 0.7307, 0.6480, 0.5920, 0.4747, 0.4933, 0.5067],
    [0.68, 0.7280, 0.7067, 0.7147, 0.6693, 0.5787, 0.5573, 0.4933],
    [0.52, 0.6027, 0.6133, 0.5733, 0.5733, 0.5653, 0.5307, 0.4507],
    [0.32, 0.3627, 0.3947, 0.3813, 0.4160, 0.4080, 0.4640, 0.3840],
    [0.20, 0.1973, 0.2133, 0.2533, 0.2720, 0.3147, 0.3120, 0.2747],
]
MULTI_VDR_CORRECT = [
    [0.80, 0.8133, 0.7547, 0.7067, 0.6613, 0.4907, 0.4507, 0.4933],
    [0.80, 0.7787, 0.7440, 0.6987, 0.6160, 0.4720, 0.5067, 0.4960],
    [0.72, 0.7867, 0.7493, 0.6453, 0.5360, 0.4667, 0.4933, 0.5120],
    [0.72, 0.7733, 0.7067, 0.6027, 0.5600, 0.4507, 0.4880, 0.4853],
    [0.72, 0.7333, 0.6907, 0.6027, 0.5440, 0.4427, 0.4853, 0.5067],
    [0.72, 0.7200, 0.6667, 0.5947, 0.5120, 0.4160, 0.4773, 0.4773],
    [0.72, 0.7253, 0.6560, 0.5680, 0.5120, 0.4373, 0.4187, 0.4667],
    [0.72, 0.7333, 0.6720, 0.6453, 0.5307, 0.4213, 0.4347, 0.4720],
]

K = 4  # 单簇真值株数


# =====================================================================
# 绘图函数
# =====================================================================

def fig1_parts_distribution(out_png):
    """单簇: 各密度下 GIDM 分块数分布直方图。"""
    n = len(RATIOS)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    all_parts = [p for row in DENSITY_SUMMARY for p in row[8]]
    lo = min(all_parts + [K]); hi = max(all_parts + [K])
    edges = np.arange(lo - 0.5, hi + 1.5, 1.0)
    xticks = np.arange(lo, hi + 1)

    for i, row in enumerate(DENSITY_SUMMARY):
        ax = axes[i]
        hist = row[8]
        vals = []
        for p, c in hist.items():
            vals += [int(p)] * int(c)
        counts, _ = np.histogram(vals, bins=edges)
        centers = (edges[:-1] + edges[1:]) / 2
        colors = []
        for c in centers:
            colors.append("#C0392B" if c < K else ("#1E8449" if c == K else "#F39C12"))
        ax.bar(centers, counts, width=0.8, color=colors, alpha=0.85)
        ax.axvline(K, color="black", ls="--", lw=1, alpha=0.5)
        ax.set_title(f"ratio={fmt_pct(RATIOS[i])}\ncorrect={row[6]:.0%}", fontsize=10)
        ax.set_xlabel("parts (#segments)")
        ax.set_ylabel("trials")
        ax.set_xticks(xticks)
        ax.set_ylim(0, max(1, counts.max()) * 1.18)
        ax.grid(axis="y", alpha=0.3)
    for i in range(n, len(axes)):
        axes[i].axis("off")
    fig.suptitle("GIDM split outcome distribution over 50 random subsamples "
                 "(k=4 plants, green=correct orange=over-split red=under-split)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig2_summary(out_png):
    """单簇: 密度敏感性汇总 (η / 漏检正确误检 / 峰数分块数)。"""
    eta = [r[1] for r in DENSITY_SUMMARY]
    eta_std = [r[2] for r in DENSITY_SUMMARY]
    miss = [r[5] for r in DENSITY_SUMMARY]
    correct = [r[6] for r in DENSITY_SUMMARY]
    false = [r[7] for r in DENSITY_SUMMARY]
    peaks = [r[3] for r in DENSITY_SUMMARY]
    parts = [r[4] for r in DENSITY_SUMMARY]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    ax.errorbar(RATIOS, eta, yerr=eta_std, marker="o", capsize=3,
                color="#C0392B", label="Valley ratio $\\eta$ (mean$\\pm$std)")
    ax.axhline(0.90, ls="--", color="gray", lw=1, label="threshold $\\theta_0=0.90$")
    ax.axhline(0.98, ls=":", color="gray", lw=1, label="$\\theta=0.98$ ($L>0.6$m)")
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_xlabel("Point retention ratio (density)")
    ax.set_ylabel("Valley ratio\n$\\eta = \\rho_v / \\min(\\rho_p)$")
    ax.set_title("(a) Valley depth vs. density\n(noise grows as density drops)")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.stackplot(RATIOS, miss, correct, false,
                 labels=["miss (under-seg)", "correct", "false (over-seg)"],
                 colors=["#E74C3C", "#2ECC71", "#F39C12"], alpha=0.85)
    ax.set_xscale("log"); ax.invert_xaxis(); ax.set_ylim(0, 1)
    ax.set_xlabel("Point retention ratio (density)")
    ax.set_ylabel("Fraction of trials")
    ax.set_title("(b) GIDM outcome vs. density\n(boundary miss & false grow)")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(RATIOS, peaks, marker="^", color="#1E8449", label="mean #peaks")
    ax.plot(RATIOS, parts, marker="v", color="#7D3C98", label="mean #parts (GIDM)")
    ax.axhline(K, ls="--", color="gray", lw=1, label=f"ground truth = {K} plants")
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_xlabel("Point retention ratio (density)"); ax.set_ylabel("Count")
    ax.set_title("(c) Peaks / resulting parts vs. density")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)

    fig.suptitle("GIDM sensitivity to point-cloud density (sparser -> valley blurs -> boundary miss/false)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig3_density_profiles(out_png):
    """单簇: 各密度下一维密度曲线。"""
    n = len(DENSITY_PROFILES)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for i, p in enumerate(DENSITY_PROFILES):
        ax = axes[i]
        ratio, npts, eta, n_peaks, counts, smooth, peaks, valleys = p
        counts = np.array(counts); smooth = np.array(smooth)
        nbins = len(counts)
        length = 2.0  # 主轴投影长度约 2m, bin 中心按均分
        bins = np.linspace(-1.0, 1.0, nbins + 1)
        bc = (bins[:-1] + bins[1:]) / 2
        bw = length / nbins
        ax.bar(bc, counts, width=bw * 0.9, color="lightgray", alpha=0.6)
        ax.plot(bc, smooth, "k-", lw=1.8, label="smoothed $\\tilde{\\rho}$")
        ax.plot(bc[peaks], smooth[peaks], "^", color="#1E8449", ms=9, label="peaks")
        if valleys:
            ax.plot(bc[valleys], smooth[valleys], "v", color="#C0392B",
                    ms=9, label=f"valleys ({len(valleys)})")
            for vi in valleys:
                ax.axvline(bc[vi], color="#C0392B", ls="--", lw=1.0, alpha=0.6)
        ax.set_title(f"ratio={fmt_pct(ratio)}  N={npts}\n$\\eta$={eta:.2f}  peaks={n_peaks}",
                     fontsize=10, linespacing=1.4)
        ax.set_xlabel("Projection $s$ (m)")
        ax.set_ylabel("Count")
    for i in range(n, len(axes)):
        axes[i].axis("off")
    fig.suptitle("1D density profile along principal axis at decreasing density",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _heatmap(ax, fig, cm, grid, param_name, default_val, parts=None):
    im = ax.imshow(cm, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0,
                   interpolation="nearest")
    ax.set_xticks(np.arange(len(RATIOS)))
    ax.set_xticklabels([fmt_pct(r) for r in RATIOS], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(grid)))
    ax.set_yticklabels([f"{v:.2f}" for v in grid])
    ax.set_xlabel("Point retention ratio (density)")
    ax.set_ylabel(param_name)
    di = grid.index(default_val)
    ax.get_yticklabels()[di].set_color("red")
    ax.get_yticklabels()[di].set_fontweight("bold")
    for i in range(len(grid)):
        for j in range(len(RATIOS)):
            txt = f"{cm[i][j]:.2f}"
            if parts is not None:
                txt += f"\n({parts[i][j]:.1f})"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color="black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("correct rate")


def fig4_heatmap(out_png):
    """单簇: θ0 敏感性热图。"""
    fig, ax = plt.subplots(figsize=(10, 7))
    _heatmap(ax, fig, VDR_CORRECT, VDR_GRID, "valley_depth_rel", 0.90, VDR_PARTS)
    ax.set_title("Sensitivity of relative depth threshold θ0 (valley_depth_rel)\n"
                 "(default θ0=0.90 marked in red)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig5_heatmap(out_png):
    """单簇: 谷底跨度敏感性热图。"""
    fig, ax = plt.subplots(figsize=(10, 7.6))
    _heatmap(ax, fig, MPD_CORRECT, MPD_GRID, "min_peak_dist_m", 0.35, MPD_PARTS)
    ax.set_title("Sensitivity of minimum peak distance (min_peak_dist_m)\n"
                 "(default 0.35m marked in red)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _small_multiples(axes, grid, cm, default_val, xlabel, param_label):
    n = len(RATIOS)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, n))
    for ri in range(n):
        ax = axes[ri]
        ax.plot(grid, [cm[gi][ri] for gi in range(len(grid))], marker="o",
                ms=3.5, lw=1.6, color=colors[ri])
        ax.axvline(default_val, color="red", ls="--", lw=1, alpha=0.7)
        ax.axhline(1.0, color="gray", ls=":", lw=0.8, alpha=0.7)
        ax.set_ylim(-0.06, 1.10)
        ax.set_title(fmt_pct(RATIOS[ri]), fontsize=9, color=colors[ri], fontweight="bold")
        ax.grid(alpha=0.3)
    for ax in axes:
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel(param_label)


def fig6_overview(out_png):
    """单簇: 超参数 small-multiples 总览 (2 行 x 8 列)。"""
    n = len(RATIOS)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.6))
    _small_multiples(axes[0], MPD_GRID, MPD_CORRECT, 0.35,
                     "min_peak_dist_m (m)", "min_peak_dist_m")
    _small_multiples(axes[1], VDR_GRID, VDR_CORRECT, 0.90,
                     "valley_depth_rel (θ0)", "valley_depth_rel")
    fig.suptitle("Hyper-parameter sensitivity of GIDM (4-plant sticky cluster, cloudR10)\n"
                 "each panel = one density; red dashed = default value",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figA_density_multi(out_png):
    """多簇: 密度敏感性跨簇汇总。"""
    correct = [r[0] for r in MULTI_DENSITY]
    over = [r[1] for r in MULTI_DENSITY]
    under = [r[2] for r in MULTI_DENSITY]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.stackplot(RATIOS, under, correct, over,
                 labels=["under-seg", "correct", "over-seg"],
                 colors=["#E74C3C", "#2ECC71", "#F39C12"], alpha=0.85)
    ax.set_xscale("log"); ax.invert_xaxis(); ax.set_ylim(0, 1)
    ax.set_xlabel("Point retention ratio (density)")
    ax.set_ylabel("Mean fraction (across clusters)")
    ax.set_title("(a) Cross-cluster outcome vs. density (25 sticky clusters, k=2~5)")
    ax.legend(fontsize=8, loc="center left"); ax.grid(alpha=0.3)
    fig.suptitle("GIDM density sensitivity — multi-cluster validation",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _multi_line(grid, cm, param_name, default_val, xlabel, out_png):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(RATIOS)))
    for ri in range(len(RATIOS)):
        ax.plot(grid, [cm[gi][ri] for gi in range(len(grid))], marker="o",
                ms=4, lw=1.5, color=colors[ri], label=fmt_pct(RATIOS[ri]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean correct rate (across clusters)")
    ax.set_ylim(-0.03, 1.08)
    ax.axvline(default_val, color="black", ls="--", lw=1, alpha=0.5)
    ax.set_title(f"Sensitivity of {param_name} — multi-cluster "
                 f"(default {default_val:.2f})", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(title="density", fontsize=6.5, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output", "standalone")
    os.makedirs(out_dir, exist_ok=True)

    fig1_parts_distribution(os.path.join(out_dir, "fig1_parts_distribution.png"))
    fig2_summary(os.path.join(out_dir, "fig2_summary.png"))
    fig3_density_profiles(os.path.join(out_dir, "fig3_density_profiles.png"))
    fig4_heatmap(os.path.join(out_dir, "fig4_hyperparam_valley_depth.png"))
    fig5_heatmap(os.path.join(out_dir, "fig5_hyperparam_min_peak_dist.png"))
    fig6_overview(os.path.join(out_dir, "fig6_hyperparam_overview.png"))
    figA_density_multi(os.path.join(out_dir, "figA_density_multi.png"))
    _multi_line(MPD_GRID, MULTI_MPD_CORRECT, "min_peak_dist_m", 0.35,
                "min_peak_dist_m (m)",
                os.path.join(out_dir, "figB_hyperparam_mpd_multi.png"))
    _multi_line(VDR_GRID, MULTI_VDR_CORRECT, "valley_depth_rel", 0.90,
                "valley_depth_rel (θ0)",
                os.path.join(out_dir, "figC_hyperparam_vdr_multi.png"))

    print(f"已生成 {9} 张图片到: {out_dir}")


if __name__ == "__main__":
    main()
