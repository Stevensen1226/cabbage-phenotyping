"""
GIDM 超参数敏感性 —— 自包含绘图脚本 (仅生成 5 张图, 数据已硬编码)。

生成:
  fig4_hyperparam_valley_depth.png       相对深度阈值 θ0 敏感性热图
  fig4_hyperparam_valley_depth_line.png  θ0 敏感性 small-multiples (每密度一子图)
  fig5_hyperparam_min_peak_dist.png      谷底跨度敏感性热图
  fig5_hyperparam_min_peak_dist_line.png 谷底跨度 small-multiples (每密度一子图)
  fig6_hyperparam_overview.png           超参数 small-multiples 总览 (2行×4列)

只依赖 numpy + matplotlib, 无点云/json/项目源码依赖, 可直接发给他人调图。

用法:
  python plot_hyperparam_standalone.py [输出目录]
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================================
# 实验数据 (硬编码)
# 数据源: cloudR10 四株粘连簇 [2,3,4,5], k=4, 每个 (参数值, 密度) 单元格 25 次随机采样
# =====================================================================

K = 4  # 真值株数

# 密度保留比例
RATIOS = [1.0, 0.5, 0.10, 0.05]


def fmt_pct(r):
    return f"{r * 100:.0f}%" if r >= 0.01 else f"{r * 100:.1f}%"


# ---- 相对深度阈值 θ0 (valley_depth_rel) ------- MAE(count) 矩阵 ----
VDR_GRID = [0.80, 0.85, 0.90, 0.95, 0.98, 1.00, 1.05, 1.10]
VDR_DEFAULT = 0.90

VDR_MAE = [  # 行=θ0, 列=密度 [100%, 50%, 10%, 5%], 值 = mean|parts - k|
    [0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.04, 0.04],
    [0.00, 0.00, 0.04, 0.04],
    [0.00, 0.00, 0.04, 0.04],
    [0.00, 0.00, 0.04, 0.04],
]

VDR_PARTS = [  # 行=θ0, 列=密度 [100%, 50%, 10%, 5%] (平均分块数)
    [4.0, 4.0, 4.0, 4.0],
    [4.0, 4.0, 4.0, 4.0],
    [4.0, 4.0, 4.0, 4.0],
    [4.0, 4.0, 4.0, 4.0],
    [4.0, 4.0, 4.04, 4.04],
    [4.0, 4.0, 4.04, 4.04],
    [4.0, 4.0, 4.04, 4.04],
    [4.0, 4.0, 4.04, 4.04],
]

# ---- 谷底跨度 min_peak_dist_m (两峰最小间距) ------ MAE(count) 矩阵 ----
MPD_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
MPD_DEFAULT = 0.35

MPD_MAE = [  # 行=min_peak_dist_m, 列=密度 [100%, 50%, 10%, 5%], 值 = mean|parts - k|
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

MPD_PARTS = [  # 行=min_peak_dist_m, 列=密度 [100%, 50%, 10%, 5%] (平均分块数)
    [4.0, 4.16, 5.32, 6.64],
    [4.0, 4.16, 5.36, 6.48],
    [4.0, 4.04, 4.76, 5.68],
    [4.0, 4.04, 4.28, 4.72],
    [4.0, 4.0, 4.04, 4.12],
    [4.0, 4.0, 4.0, 4.0],
    [4.0, 4.0, 3.96, 4.0],
    [4.0, 4.0, 3.92, 3.88],
    [4.0, 3.72, 3.92, 3.88],
]


# =====================================================================
# 绘图函数
# =====================================================================

def _heatmap(mae, grid, param_name, default_val, parts, title, out_png):
    """热图: 行=参数值, 列=密度; 颜色=MAE(count), 格内=MAE/平均分块数。"""
    nrows, ncols = len(grid), len(RATIOS)
    fig, ax = plt.subplots(figsize=(0.9 * ncols + 2.5, 0.7 * nrows + 1.5))

    vmax = float(np.max(mae))
    im = ax.imshow(mae, aspect="auto", cmap="RdYlGn_r", vmin=0.0,
                   vmax=max(vmax, 0.01), interpolation="nearest")
    ax.set_xticks(np.arange(ncols))
    ax.set_xticklabels([fmt_pct(r) for r in RATIOS], rotation=45, ha="right")
    ax.set_yticks(np.arange(nrows))
    ax.set_yticklabels([f"{v:.2f}" for v in grid])
    ax.set_xlabel("Point retention ratio (density)")
    ax.set_ylabel(param_name)

    # 默认值行用红色加粗刻度标注 (零遮挡)
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


def _small_multiples(mae, grid, default_val, xlabel, param_label, title, out_png):
    """Small multiples: 每个密度一个子图, MAE(count) vs 参数值 (1行×4列)。"""
    n = len(RATIOS)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.8 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    ymax = float(np.max(mae)) * 1.15
    panel_colors = plt.cm.viridis(np.linspace(0.05, 0.95, n))
    for ri in range(n):
        ax = axes[ri]
        ax.plot(grid, [mae[gi][ri] for gi in range(len(grid))], marker="o",
                ms=3.5, lw=1.6, color=panel_colors[ri])
        ax.axvline(default_val, color="red", ls="--", lw=1, alpha=0.7)
        ax.axhline(0.0, color="gray", ls=":", lw=0.8, alpha=0.7)
        ax.set_ylim(-0.03, max(ymax, 0.1))
        ax.set_title(fmt_pct(RATIOS[ri]), fontsize=9, color=panel_colors[ri],
                     fontweight="bold")
        ax.grid(alpha=0.3)

    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.text(0.5, 0.02, xlabel, ha="center", fontsize=11)
    fig.text(0.02, 0.5, "MAE (count) = mean|parts - k|", va="center",
             rotation=90, fontsize=11)
    fig.suptitle(title + f"\n(red dashed = default {default_val:.2f})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0.05, 0.04, 1, 0.94])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _overview(out_png):
    """fig6: 两参数上下并排 small-multiples 总览 (2行×4列)。"""
    n = len(RATIOS)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.6))

    # 上排: min_peak_dist_m
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, n))
    mpd_ymax = float(np.max(MPD_MAE)) * 1.15
    for ri in range(n):
        ax = axes[0, ri]
        ax.plot(MPD_GRID, [MPD_MAE[gi][ri] for gi in range(len(MPD_GRID))],
                marker="o", ms=3.5, lw=1.6, color=colors[ri])
        ax.axvline(MPD_DEFAULT, color="red", ls="--", lw=1, alpha=0.7)
        ax.axhline(0.0, color="gray", ls=":", lw=0.8, alpha=0.7)
        ax.set_ylim(-0.03, max(mpd_ymax, 0.1))
        ax.set_title(fmt_pct(RATIOS[ri]), fontsize=9, color=colors[ri], fontweight="bold")
        ax.grid(alpha=0.3)
        ax.set_xlabel("min_peak_dist_m (m)")
    axes[0, 0].set_ylabel("min_peak_dist_m")

    # 下排: valley_depth_rel
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
    axes[1, 0].set_ylabel("valley_depth_rel")

    fig.suptitle("Hyper-parameter sensitivity of GIDM (4-plant sticky cluster, cloudR10)\n"
                 "each panel = one density; red dashed = default; y = MAE(count)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output", "standalone")
    os.makedirs(out_dir, exist_ok=True)

    # 图4: θ0 热图 + small multiples
    _heatmap(VDR_MAE, VDR_GRID, "valley_depth_rel", VDR_DEFAULT, VDR_PARTS,
             "Sensitivity of relative depth threshold θ0 (valley_depth_rel)\n"
             "color = MAE(count), cell text = (mean #parts)",
             os.path.join(out_dir, "fig4_hyperparam_valley_depth.png"))
    _small_multiples(VDR_MAE, VDR_GRID, VDR_DEFAULT, "valley_depth_rel (θ0)",
                     "valley_depth_rel",
                     "MAE(count) vs. relative depth threshold θ0 (one panel per density)",
                     os.path.join(out_dir, "fig4_hyperparam_valley_depth_line.png"))

    # 图5: 谷底跨度热图 + small multiples
    _heatmap(MPD_MAE, MPD_GRID, "min_peak_dist_m", MPD_DEFAULT, MPD_PARTS,
             "Sensitivity of minimum peak distance (min_peak_dist_m)\n"
             "color = MAE(count), cell text = (mean #parts)",
             os.path.join(out_dir, "fig5_hyperparam_min_peak_dist.png"))
    _small_multiples(MPD_MAE, MPD_GRID, MPD_DEFAULT, "min_peak_dist_m (m)",
                     "min_peak_dist_m",
                     "MAE(count) vs. minimum peak distance (one panel per density)",
                     os.path.join(out_dir, "fig5_hyperparam_min_peak_dist_line.png"))

    # 图6: 总览
    _overview(os.path.join(out_dir, "fig6_hyperparam_overview.png"))

    print(f"已生成 5 张图片到: {out_dir}")


if __name__ == "__main__":
    main()
