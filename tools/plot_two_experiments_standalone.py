"""
GIDM 密度敏感性 + 超参数敏感性 —— 完整自包含绘图脚本。

两个实验的全部图片, 数据已硬编码, 只依赖 numpy + matplotlib,
可直接发给他人复现 / 调整图片样式。

生成图片:
  密度敏感性 (单簇 + 多簇):
    fig1_parts_distribution.png    单簇各密度分块数分布
    fig2_summary.png               单簇汇总 (a 谷值比 / b MAE / c 峰数分块数)
    fig3_density_profiles.png      单簇一维密度曲线
    figA_density_multi.png         多簇跨簇计数误差分布 (堆叠柱状图)
    figD_density_by_k.png          多簇按粘连株数 k 分层
    figE_density_by_difficulty.png 多簇按易/难簇分层
  超参数敏感性 (单簇 + 多簇):
    fig4_hyperparam_valley_depth.png     单簇 θ0 热图
    fig4_hyperparam_valley_depth_line.png 单簇 θ0 small-multiples
    fig5_hyperparam_min_peak_dist.png    单簇谷底跨度热图
    fig5_hyperparam_min_peak_dist_line.png 单簇谷底跨度 small-multiples
    fig6_hyperparam_overview.png         单簇超参数总览
    figB_hyperparam_mpd_multi.png        多簇谷底跨度跨簇折线
    figC_hyperparam_vdr_multi.png        多簇 θ0 跨簇折线

用法:
  python plot_two_experiments_standalone.py [输出目录]
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================================
# 公共常量
# =====================================================================

K = 4  # 单簇真值株数
RATIOS = [1.0, 0.5, 0.10, 0.05]  # 密度保留比例 (4 档)
RATIO_LABELS = ["100%", "50%", "10%", "5%"]


def fmt_pct(r):
    return f"{r * 100:.0f}%" if r >= 0.01 else f"{r * 100:.1f}%"


# =====================================================================
# 实验一: 密度敏感性数据
# =====================================================================

# ---- 单簇密度敏感性 (cloudR10 四株粘连, k=4, 每档 50 次采样) ----
# [n_points, eta, eta_std, peaks, parts, mae, miss, correct, false, parts_hist]
DENSITY_SUMMARY = [
    [39693, 0.484, 0.000, 4.00, 4.00, 0.00, 0.00, 1.00, 0.00, {4: 50}],
    [19938, 0.487, 0.013, 4.00, 4.00, 0.00, 0.00, 1.00, 0.00, {4: 50}],
    [3922,  0.493, 0.033, 4.00, 4.00, 0.00, 0.00, 1.00, 0.00, {4: 50}],
    [2038,  0.490, 0.056, 3.94, 4.00, 0.04, 0.02, 0.96, 0.02, {4: 48, 5: 1, 3: 1}],
]

# ---- 单簇密度曲线 (fig3): 每档代表性采样 ----
# [ratio, n_points, eta, n_peaks, counts(15), smooth(15), peaks, valleys]
DENSITY_PROFILES = [
    [1.0, 39693, 0.484, 4,
     [1754, 3242, 3267, 2662, 2821, 3335, 2312, 1356, 3203, 3425, 2940, 1185, 2584, 3343, 2264],
     [1913.2, 3085.7, 3199.4, 2743.7, 2858.8, 3170.7, 2319.5, 1655.4, 3029.7, 3348.6, 2804.8, 1521.9, 2515.9, 3146.5, 2379.2],
     [2, 5, 9, 13], [3, 7, 11]],
    [0.5, 19938, 0.487, 4,
     [889, 1654, 1649, 1340, 1365, 1674, 1163, 710, 1624, 1687, 1495, 591, 1306, 1664, 1127],
     [970.8, 1571.7, 1616.4, 1375.7, 1395.3, 1586.4, 1169.3, 856.0, 1533.3, 1659.3, 1419.2, 763.9, 1268.0, 1568.3, 1184.4],
     [2, 5, 9, 13], [3, 7, 11]],
    [0.1, 3922, 0.491, 4,
     [202, 302, 303, 249, 280, 321, 237, 124, 325, 323, 325, 107, 253, 333, 238],
     [212.7, 291.4, 297.1, 258.1, 281.1, 307.6, 233.9, 157.5, 303.4, 323.3, 301.6, 145.9, 246.0, 314.3, 248.1],
     [2, 5, 9, 13], [3, 7, 11]],
    [0.05, 2038, 0.489, 4,
     [107, 166, 145, 109, 150, 186, 124, 65, 167, 202, 150, 61, 122, 178, 106],
     [113.3, 157.5, 143.4, 117.2, 149.5, 175.5, 124.3, 82.2, 159.9, 192.7, 146.1, 77.0, 121.5, 164.3, 113.7],
     [1, 5, 9, 13], [3, 7, 11]],
]

# ---- 多簇密度敏感性 (25 个粘连簇, 每档 40 次采样) ----
# [mae_mean, correct, over, under, 堆叠(MAE=0,1,2 簇数)]
MULTI_DENSITY = [
    [0.32, 0.68, 0.00, 0.32, [17, 8, 0]],
    [0.30, 0.73, 0.01, 0.27, [18, 7, 0]],
    [0.35, 0.68, 0.06, 0.26, [18, 7, 0]],
    [0.36, 0.66, 0.11, 0.23, [19, 6, 0]],
]

# 按粘连株数 k 分层的 MAE (行 = k, 列 = 4 密度)
MULTI_BY_K = {
    2: [0.00, 0.02, 0.13, 0.23],
    3: [0.33, 0.16, 0.33, 0.32],
    4: [0.20, 0.25, 0.16, 0.20],
    5: [0.62, 0.65, 0.66, 0.57],
}
MULTI_BY_K_COUNT = {2: 6, 3: 6, 4: 5, 5: 8}

# 按难度分层的 MAE (行 = [易簇, 难簇], 列 = 4 密度)
MULTI_EASY = [0.00, 0.03, 0.16, 0.22]   # 17 个易簇
MULTI_HARD = [1.00, 0.87, 0.77, 0.64]   # 8 个难簇

# =====================================================================
# 实验二: 超参数敏感性数据
# =====================================================================

VDR_GRID = [0.80, 0.85, 0.90, 0.95, 0.98, 1.00, 1.05, 1.10]
VDR_DEFAULT = 0.90
VDR_MAE = [  # 行=θ0, 列=密度 [100%,50%,10%,5%]
    [0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.04, 0.04],
    [0.00, 0.00, 0.04, 0.04],
    [0.00, 0.00, 0.04, 0.04],
    [0.00, 0.00, 0.04, 0.04],
]
VDR_PARTS = [
    [4.0, 4.0, 4.0, 4.0], [4.0, 4.0, 4.0, 4.0],
    [4.0, 4.0, 4.0, 4.0], [4.0, 4.0, 4.0, 4.0],
    [4.0, 4.0, 4.04, 4.04], [4.0, 4.0, 4.04, 4.04],
    [4.0, 4.0, 4.04, 4.04], [4.0, 4.0, 4.04, 4.04],
]

MPD_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
MPD_DEFAULT = 0.35
MPD_MAE = [  # 行=min_peak_dist_m, 列=密度 [100%,50%,10%,5%]
    [0.00, 0.16, 1.32, 2.64],
    [0.00, 0.16, 1.36, 2.48],
    [0.00, 0.04, 0.76, 1.68],
    [0.00, 0.04, 0.28, 0.72],
    [0.00, 0.00, 0.04, 0.12],
    [0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.04, 0.00],
    [0.00, 0.00, 0.08, 0.12],
    [0.00, 0.28, 0.08, 0.12],
]
MPD_PARTS = [
    [4.0, 4.16, 5.32, 6.64], [4.0, 4.16, 5.36, 6.48],
    [4.0, 4.04, 4.76, 5.68], [4.0, 4.04, 4.28, 4.72],
    [4.0, 4.0, 4.04, 4.12], [4.0, 4.0, 4.0, 4.0],
    [4.0, 4.0, 3.96, 4.0], [4.0, 4.0, 3.92, 3.88],
    [4.0, 3.72, 3.92, 3.88],
]

# 多簇超参数敏感性 (25 簇) MAE 矩阵
MULTI_MPD_MAE = [
    [1.48, 1.47, 2.34, 2.85], [1.24, 1.34, 2.26, 2.58],
    [0.64, 0.98, 1.61, 1.99], [0.48, 0.53, 0.73, 0.89],
    [0.28, 0.27, 0.41, 0.47], [0.32, 0.31, 0.31, 0.36],
    [0.52, 0.44, 0.50, 0.47], [0.88, 0.80, 0.74, 0.68],
    [1.04, 1.04, 0.97, 0.90],
]
MULTI_VDR_MAE = [
    [0.32, 0.28, 0.31, 0.34], [0.32, 0.32, 0.29, 0.38],
    [0.32, 0.27, 0.34, 0.36], [0.32, 0.27, 0.37, 0.36],
    [0.32, 0.27, 0.32, 0.40], [0.32, 0.28, 0.29, 0.37],
    [0.32, 0.29, 0.35, 0.40], [0.32, 0.28, 0.36, 0.39],
]


# =====================================================================
# 密度敏感性绘图
# =====================================================================

def fig1_parts_distribution(out_png):
    n = len(RATIOS)
    ncols = 4
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 3.2))
    axes = np.atleast_1d(axes).ravel()

    lo = 1; hi = 6
    edges = np.arange(lo - 0.5, hi + 1.5, 1.0)
    xticks = np.arange(lo, hi + 1)

    for i, row in enumerate(DENSITY_SUMMARY):
        ax = axes[i]
        hist = row[9]
        vals = []
        for p, c in hist.items():
            vals += [int(p)] * int(c)
        counts, _ = np.histogram(vals, bins=edges)
        centers = (edges[:-1] + edges[1:]) / 2
        colors = ["#C0392B" if c < K else ("#1E8449" if c == K else "#F39C12")
                  for c in centers]
        ax.bar(centers, counts, width=0.8, color=colors, alpha=0.85)
        ax.axvline(K, color="black", ls="--", lw=1, alpha=0.5)
        ax.set_title(f"ratio={fmt_pct(RATIOS[i])}\nMAE={row[5]:.2f}", fontsize=10)
        ax.set_xlabel("parts (#segments)")
        ax.set_ylabel("trials")
        ax.set_xticks([1, 2, 3, 4, 5, 6])
        ax.set_ylim(0, max(counts) * 1.18)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("GIDM split outcome distribution (single cluster, k=4)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig2_summary(out_png):
    eta = [r[1] for r in DENSITY_SUMMARY]
    eta_std = [r[2] for r in DENSITY_SUMMARY]
    mae = [r[5] for r in DENSITY_SUMMARY]
    n_peaks = [r[3] for r in DENSITY_SUMMARY]
    n_parts = [r[4] for r in DENSITY_SUMMARY]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))

    ax = axes[0]
    ax.errorbar(RATIOS, eta, yerr=eta_std, marker="o", ms=5, capsize=3,
                color="#C0392B", lw=1.8, label="Valley ratio $\\eta$")
    ax.axhline(0.90, ls="--", color="gray", lw=1, label="threshold 0.90")
    ax.axhline(0.98, ls=":", color="gray", lw=1, label="0.98 (L>0.6m)")
    ax.set_xscale("log"); ax.invert_xaxis(); ax.set_ylim(0.35, 1.02)
    ax.set_xlabel("Point retention ratio (density)")
    ax.set_ylabel(r"Valley ratio $\eta$")
    ax.set_title("(a) Valley depth vs. density")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=3, frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.plot(RATIOS, mae, marker="o", ms=5, color="#2471A3", lw=2, label="MAE")
    ax.axhline(0.0, ls="--", color="gray", lw=1)
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_ylim(-0.03, 0.10)
    ax.set_xlabel("Point retention ratio (density)")
    ax.set_ylabel("MAE (count)")
    ax.set_title("(b) Counting error vs. density")
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[2]
    ax.plot(RATIOS, n_peaks, marker="^", ms=5, color="#1E8449", lw=1.8, label="mean #peaks")
    ax.plot(RATIOS, n_parts, marker="v", ms=5, color="#7D3C98", lw=1.8, label="mean #parts")
    ax.axhline(K, ls="--", color="gray", lw=1, label=f"ground truth = {K}")
    ax.set_xscale("log"); ax.invert_xaxis(); ax.set_ylim(3.5, 4.5)
    ax.set_xlabel("Point retention ratio (density)")
    ax.set_ylabel("Count")
    ax.set_title("(c) Peaks / parts vs. density")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=3, frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("GIDM density sensitivity (single cluster, k=4)",
                 fontsize=12, fontweight="bold", y=0.98)
    fig.subplots_adjust(top=0.82, bottom=0.22, wspace=0.28)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig3_density_profiles(out_png):
    n = len(DENSITY_PROFILES)
    ncols = 4
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 3.4))
    axes = np.atleast_1d(axes).ravel()

    for i, p in enumerate(DENSITY_PROFILES):
        ax = axes[i]
        ratio, npts, eta, n_peaks, counts, smooth, peaks, valleys = p
        counts = np.array(counts); smooth = np.array(smooth)
        nbins = len(counts)
        bins = np.linspace(-1.0, 1.0, nbins + 1)
        bc = (bins[:-1] + bins[1:]) / 2
        bw = 2.0 / nbins
        ax.bar(bc, counts, width=bw * 0.9, color="lightgray", alpha=0.6)
        ax.plot(bc, smooth, "k-", lw=1.8, label="smoothed")
        ax.plot(bc[peaks], smooth[peaks], "^", color="#1E8449", ms=9, label="peaks")
        if valleys:
            ax.plot(bc[valleys], smooth[valleys], "v", color="#C0392B",
                    ms=9, label=f"valleys ({len(valleys)})")
            for vi in valleys:
                ax.axvline(bc[vi], color="#C0392B", ls="--", lw=1.0, alpha=0.6)
        ax.set_title(f"ratio={fmt_pct(ratio)}  N={npts}\neta={eta:.2f}  peaks={n_peaks}",
                     fontsize=10, linespacing=1.4)
        ax.set_xlabel("Projection s (m)")
        ax.set_ylabel("Count")
    fig.suptitle("1D density profile along principal axis (single cluster)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def figA_density_multi(out_png):
    fig, ax = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(RATIOS))
    bottoms = np.zeros(len(RATIOS))
    err_colors = {0: "#2ECC71", 1: "#F39C12", 2: "#E74C3C"}
    for lv in range(3):
        counts = [MULTI_DENSITY[i][4][lv] for i in range(len(RATIOS))]
        ax.bar(positions, counts, bottom=bottoms, width=0.6,
               color=err_colors[lv], edgecolor="white", lw=0.5,
               label=f"MAE = {lv}")
        bottoms = bottoms + np.array(counts)
    ax.set_xticks(positions)
    ax.set_xticklabels(RATIO_LABELS)
    ax.set_ylim(0, max(bottoms) * 1.08)
    ax.set_xlabel("Point retention ratio (density)")
    ax.set_ylabel("Number of clusters")
    ax.set_title("(a) Counting error distribution vs. density (25 clusters, k=2~5)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=3, frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figD_density_by_k(out_png):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {2: "#1E8449", 3: "#2471A3", 4: "#7D3C98", 5: "#B03A2E"}
    ks = sorted(MULTI_BY_K.keys())
    n_groups = len(RATIOS); n_bars = len(ks)
    bar_width = 0.8 / n_bars
    x = np.arange(n_groups)
    for bi, k in enumerate(ks):
        offset = (bi - (n_bars - 1) / 2) * bar_width
        ax.bar(x + offset, MULTI_BY_K[k], width=bar_width,
               color=colors[k], edgecolor="white", lw=0.5,
               label=f"k={k} ({MULTI_BY_K_COUNT[k]})")
    ax.set_xticks(x); ax.set_xticklabels(RATIO_LABELS)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Point retention ratio (density)")
    ax.set_ylabel("Mean MAE (count) per cluster")
    ax.set_title("Density sensitivity stratified by plant count k")
    ax.legend(fontsize=8.5, loc="upper left", frameon=False, ncol=2)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figE_density_by_difficulty(out_png):
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = [("easy (MAE=0, 17)", MULTI_EASY, "#2ECC71"),
              ("hard (MAE>0, 8)", MULTI_HARD, "#E74C3C")]
    n_groups = len(RATIOS); n_bars = len(groups)
    bar_width = 0.38
    x = np.arange(n_groups)
    for bi, (label, vals, color) in enumerate(groups):
        offset = (bi - (n_bars - 1) / 2) * bar_width
        ax.bar(x + offset, vals, width=bar_width, color=color,
               edgecolor="white", lw=0.5, label=label)
    ax.set_xticks(x); ax.set_xticklabels(RATIO_LABELS)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Point retention ratio (density)")
    ax.set_ylabel("Mean MAE (count) per cluster")
    ax.set_title("Density sensitivity stratified by cluster difficulty")
    ax.legend(fontsize=9, loc="upper left", frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# 超参数敏感性绘图
# =====================================================================

def _heatmap(mae, grid, param_name, default_val, parts, title, out_png, vmax):
    nrows, ncols = len(grid), len(RATIOS)
    fig, ax = plt.subplots(figsize=(0.9 * ncols + 2.5, 0.7 * nrows + 1.5))
    im = ax.imshow(mae, aspect="auto", cmap="RdYlGn_r", vmin=0.0,
                   vmax=max(vmax, 0.01), interpolation="nearest")
    ax.set_xticks(np.arange(ncols))
    ax.set_xticklabels(RATIO_LABELS, rotation=45, ha="right")
    ax.set_yticks(np.arange(nrows))
    ax.set_yticklabels([f"{v:.2f}" for v in grid])
    ax.set_xlabel("Point retention ratio (density)")
    ax.set_ylabel(param_name)
    di = grid.index(default_val)
    ax.get_yticklabels()[di].set_color("red")
    ax.get_yticklabels()[di].set_fontweight("bold")
    for i in range(nrows):
        for j in range(ncols):
            txt = f"{mae[i][j]:.2f}"
            if parts is not None:
                txt += f"\n({parts[i][j]:.1f})"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")
    ax.set_title(title, fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("MAE (count) = mean|parts - k|")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _small_multiples(mae, grid, default_val, xlabel, title, out_png):
    n = len(RATIOS)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 2.8), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    ymax = float(np.max(mae)) * 1.15
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, n))
    for ri in range(n):
        ax = axes[ri]
        ax.plot(grid, [mae[gi][ri] for gi in range(len(grid))], marker="o",
                ms=3.5, lw=1.6, color=colors[ri])
        ax.axvline(default_val, color="red", ls="--", lw=1, alpha=0.7)
        ax.axhline(0.0, color="gray", ls=":", lw=0.8, alpha=0.7)
        ax.set_ylim(-0.03, max(ymax, 0.1))
        ax.set_title(RATIO_LABELS[ri], fontsize=9, color=colors[ri], fontweight="bold")
        ax.grid(alpha=0.3)
    fig.text(0.5, 0.02, xlabel, ha="center", fontsize=11)
    fig.text(0.02, 0.5, "MAE (count)", va="center", rotation=90, fontsize=11)
    fig.suptitle(title + f"\n(red dashed = default {default_val:.2f})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0.05, 0.04, 1, 0.90])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig6_overview(out_png):
    n = len(RATIOS)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.6))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, n))
    mpd_ymax = float(np.max(MPD_MAE)) * 1.15
    for ri in range(n):
        ax = axes[0, ri]
        ax.plot(MPD_GRID, [MPD_MAE[gi][ri] for gi in range(len(MPD_GRID))],
                marker="o", ms=3.5, lw=1.6, color=colors[ri])
        ax.axvline(MPD_DEFAULT, color="red", ls="--", lw=1, alpha=0.7)
        ax.axhline(0.0, color="gray", ls=":", lw=0.8, alpha=0.7)
        ax.set_ylim(-0.03, max(mpd_ymax, 0.1))
        ax.set_title(RATIO_LABELS[ri], fontsize=9, color=colors[ri], fontweight="bold")
        ax.grid(alpha=0.3)
        ax.set_xlabel("min_peak_dist_m (m)")
    vdr_ymax = float(np.max(VDR_MAE)) * 1.15
    for ri in range(n):
        ax = axes[1, ri]
        ax.plot(VDR_GRID, [VDR_MAE[gi][ri] for gi in range(len(VDR_GRID))],
                marker="o", ms=3.5, lw=1.6, color=colors[ri])
        ax.axvline(VDR_DEFAULT, color="red", ls="--", lw=1, alpha=0.7)
        ax.axhline(0.0, color="gray", ls=":", lw=0.8, alpha=0.7)
        ax.set_ylim(-0.03, max(vdr_ymax, 0.1))
        ax.grid(alpha=0.3)
        ax.set_xlabel("valley_depth_rel (θ0)")
    axes[0, 0].set_ylabel("min_peak_dist_m")
    axes[1, 0].set_ylabel("valley_depth_rel")
    fig.suptitle("Hyper-parameter sensitivity (single cluster, cloudR10)\n"
                 "each panel = one density; red dashed = default; y = MAE(count)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _multi_line(grid, mae, param_name, default_val, xlabel, out_png):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(RATIOS)))
    markers = ["o", "s", "^", "v"]
    for ri in range(len(RATIOS)):
        ax.plot(grid, [mae[gi][ri] for gi in range(len(grid))],
                marker=markers[ri], ms=5, lw=1.2, color=colors[ri],
                label=RATIO_LABELS[ri], markerfacecolor="none", markeredgewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean MAE (count) across clusters")
    ax.set_ylim(bottom=0)
    ax.axvline(default_val, color="black", ls="--", lw=1, alpha=0.5)
    ax.set_title(f"Sensitivity of {param_name} — multi-cluster "
                 f"(default {default_val:.2f})", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(title="density", fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# 主流程
# =====================================================================

def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output", "standalone")
    os.makedirs(out_dir, exist_ok=True)

    # 密度敏感性
    fig1_parts_distribution(os.path.join(out_dir, "fig1_parts_distribution.png"))
    fig2_summary(os.path.join(out_dir, "fig2_summary.png"))
    fig3_density_profiles(os.path.join(out_dir, "fig3_density_profiles.png"))
    figA_density_multi(os.path.join(out_dir, "figA_density_multi.png"))
    figD_density_by_k(os.path.join(out_dir, "figD_density_by_k.png"))
    figE_density_by_difficulty(os.path.join(out_dir, "figE_density_by_difficulty.png"))

    # 超参数敏感性 (单簇热图统一色标)
    global_vmax = max(np.max(VDR_MAE), np.max(MPD_MAE))
    global_vmax = max(np.ceil(global_vmax), 1.0)
    _heatmap(VDR_MAE, VDR_GRID, "valley_depth_rel", VDR_DEFAULT, VDR_PARTS,
             "Sensitivity of relative depth threshold θ0 (valley_depth_rel)\n"
             "color = MAE(count), cell text = (mean #parts)",
             os.path.join(out_dir, "fig4_hyperparam_valley_depth.png"), global_vmax)
    _small_multiples(VDR_MAE, VDR_GRID, VDR_DEFAULT, "valley_depth_rel (θ0)",
                     "MAE(count) vs. relative depth threshold θ0",
                     os.path.join(out_dir, "fig4_hyperparam_valley_depth_line.png"))
    _heatmap(MPD_MAE, MPD_GRID, "min_peak_dist_m", MPD_DEFAULT, MPD_PARTS,
             "Sensitivity of minimum peak distance (min_peak_dist_m)\n"
             "color = MAE(count), cell text = (mean #parts)",
             os.path.join(out_dir, "fig5_hyperparam_min_peak_dist.png"), global_vmax)
    _small_multiples(MPD_MAE, MPD_GRID, MPD_DEFAULT, "min_peak_dist_m (m)",
                     "MAE(count) vs. minimum peak distance",
                     os.path.join(out_dir, "fig5_hyperparam_min_peak_dist_line.png"))
    fig6_overview(os.path.join(out_dir, "fig6_hyperparam_overview.png"))

    # 超参数敏感性 (多簇)
    _multi_line(MPD_GRID, MULTI_MPD_MAE, "min_peak_dist_m", MPD_DEFAULT,
                "min_peak_dist_m (m)",
                os.path.join(out_dir, "figB_hyperparam_mpd_multi.png"))
    _multi_line(VDR_GRID, MULTI_VDR_MAE, "valley_depth_rel", VDR_DEFAULT,
                "valley_depth_rel (θ0)",
                os.path.join(out_dir, "figC_hyperparam_vdr_multi.png"))

    print(f"已生成 {12} 张图片到: {out_dir}")


if __name__ == "__main__":
    main()
