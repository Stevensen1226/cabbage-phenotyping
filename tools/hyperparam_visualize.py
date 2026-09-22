"""
GIDM 超参数敏感性 —— 论文级综合可视化。

读取 tools/hyperparam_sensitivity.py 生成的 hyperparam_summary.json,
将两个超参数 (谷底跨度 min_peak_dist_m 与相对深度阈值 valley_depth_rel)
整合为一张 small-multiples 总览图:
  上排: min_peak_dist_m 的 8 个密度面板 (MAE(count) vs 参数值)
  下排: valley_depth_rel 的 8 个密度面板
每个面板只画一条曲线, 彻底消除多密度曲线重叠; 能直观看出
"密度越低曲线越陡(越敏感)"。

用法:
  python tools/hyperparam_visualize.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.density_sensitivity import fmt_pct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(ROOT, "output", "density_sensitivity",
                         "single_cluster", "hyperparam", "hyperparam_summary.json")
OUT_PNG = os.path.join(ROOT, "output", "density_sensitivity",
                       "single_cluster", "hyperparam", "fig6_hyperparam_overview.png")
PANELS_DIR = os.path.join(ROOT, "output", "density_sensitivity",
                          "single_cluster", "hyperparam", "panels")


def load():
    with open(JSON_PATH) as f:
        return json.load(f)


def _draw_sm_panel(ax, grid, curve, default_val, density_label, color, ymax):
    """在单个 ax 上画一条密度曲线 (small multiples 的一个面板)。"""
    ax.plot(grid, curve, marker="o", ms=3.5, lw=1.6, color=color)
    ax.axvline(default_val, color="red", ls="--", lw=1, alpha=0.7)
    ax.axhline(0.0, color="gray", ls=":", lw=0.8, alpha=0.7)
    ax.set_ylim(-0.03, max(ymax, 0.1))
    ax.set_title(density_label, fontsize=9, color=color, fontweight="bold")
    ax.grid(alpha=0.3)


def _draw_small_multiples(fig, axes, grid, mae, default_val, ratios, xlabel,
                          param_label):
    """在给定 axes 数组 (1 行 n 列) 上绘制某个参数的 small multiples。"""
    n = len(ratios)
    colors = plt_cmap(np.linspace(0.05, 0.95, n))
    ymax = float(np.nanmax(mae)) * 1.15
    for ri in range(n):
        _draw_sm_panel(axes[ri], grid, mae[:, ri], default_val,
                       fmt_pct(ratios[ri]), colors[ri], ymax)
    # 首尾面板加轴标签
    axes[0].set_ylabel(param_label)
    for ax in axes:
        ax.set_xlabel(xlabel)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    global plt_cmap
    plt_cmap = plt.cm.viridis

    data = load()
    ratios = data["ratios"]
    xlabels = [fmt_pct(r) for r in ratios]

    mpd = data["min_peak_dist_m"]
    vdr = data["valley_depth_rel"]
    mpd_grid = mpd["grid"]
    vdr_grid = vdr["grid"]
    mpd_mae = np.array(mpd["mae_matrix"])
    vdr_mae = np.array(vdr["mae_matrix"])

    n = len(ratios)

    # === 总览图: 2 行 x n 列 small multiples ===
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.6),
                             sharex=False, sharey=True)
    _draw_small_multiples(fig, axes[0], mpd_grid, mpd_mae, 0.35, ratios,
                          "min_peak_dist_m (m)", "min_peak_dist_m")
    _draw_small_multiples(fig, axes[1], vdr_grid, vdr_mae, 0.90, ratios,
                          "valley_depth_rel (θ0)", "valley_depth_rel")
    fig.suptitle(
        "Hyper-parameter sensitivity of GIDM (4-plant sticky cluster, cloudR10)\n"
        "each panel = one density; red dashed = default; y = MAE(count)",
        fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"已生成综合可视化图: {OUT_PNG}")

    # === 每个参数拆分为一张独立的 small-multiples 图 ===
    os.makedirs(PANELS_DIR, exist_ok=True)
    for fname, grid, mae, default_val, xlabel, param_label in [
            ("fig6_mpd_small_multiples", mpd_grid, mpd_mae, 0.35,
             "min_peak_dist_m (m)", "min_peak_dist_m"),
            ("fig6_vdr_small_multiples", vdr_grid, vdr_mae, 0.90,
             "valley_depth_rel (θ0)", "valley_depth_rel")]:
        f = plt.figure(figsize=(2.6 * n, 3.2))
        axs = f.subplots(1, n, sharey=True)
        _draw_small_multiples(f, axs, grid, mae, default_val, ratios,
                              xlabel, param_label)
        f.suptitle(f"{param_label}: MAE(count) vs. parameter value "
                   f"(red dashed = default {default_val:.2f})",
                   fontsize=11, fontweight="bold")
        f.tight_layout(rect=[0, 0, 1, 0.90])
        f.savefig(os.path.join(PANELS_DIR, fname + ".png"), dpi=150,
                  bbox_inches="tight")
        plt.close(f)
    print(f"已拆分为 2 张 small-multiples 单图: {PANELS_DIR}/")


if __name__ == "__main__":
    main()
