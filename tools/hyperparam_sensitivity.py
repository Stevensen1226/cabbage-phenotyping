"""
GIDM 超参数敏感性实验。

对两个核心超参数做"超参数值 × 点云密度"的敏感性热图:
  1. valley_depth_rel (相对深度阈值 θ0, 默认 0.90)
  2. min_peak_dist_m  (两峰最小间距 / 谷底跨度, 默认 0.35 m)

每个单元格 = 在给定密度 (随机稀释) 与给定超参数值下, GIDM 切出块数等于
真值株数 k 的比例 (N_TRIALS 次随机采样), 即"正确率"。

关键结论预期: 全密度下超参数鲁棒 (正确率恒 1), 密度稀疏时敏感性急剧放大。

用法:
  python tools/hyperparam_sensitivity.py
"""
import json
import os
import sys
from collections import Counter

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.density_sensitivity import (
    load_cloud_ply, find_sticky_clusters, subsample, actual_split,
    InstanceClusterer, fmt_pct, RATIOS, SEED,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "output", "density_sensitivity",
                       "single_cluster", "hyperparam")

# 两个超参数的扫描范围
VALLEY_DEPTH_GRID = [0.80, 0.85, 0.90, 0.95, 0.98, 1.00, 1.05, 1.10]
MIN_PEAK_DIST_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

N_TRIALS = 25  # 每个 (密度, 超参数) 组合的随机采样次数


def scan_hyperparam(clusterer, points, k, param_name, grid):
    """扫描单个超参数 × 密度, 返回 (correct, parts, mae) 三个矩阵。

    每个单元格 = 给定密度 (随机稀释) + 给定超参数值下, N_TRIALS 次随机采样的统计:
      - correct_matrix: 正确率 (parts == k 的占比)
      - parts_matrix:   平均分块数
      - mae_matrix:     MAE(count) = mean(|parts - k|), 预测株数的平均绝对误差

    固定另一个超参数为默认值 (config 里的 0.90 / 0.35)。
    """
    # 固定另一个超参数为默认值, 避免扫描顺序污染 (上一个参数的末值残留)
    if param_name == "valley_depth_rel":
        setattr(clusterer, "min_peak_dist_m", 0.35)
    else:
        setattr(clusterer, "valley_depth_rel", 0.90)

    correct_matrix = np.zeros((len(grid), len(RATIOS)))
    parts_matrix = np.zeros((len(grid), len(RATIOS)))
    mae_matrix = np.zeros((len(grid), len(RATIOS)))

    for gi, val in enumerate(grid):
        for ri, ratio in enumerate(RATIOS):
            correct = 0
            parts_sum = 0
            mae_sum = 0
            valid = 0
            for t in range(N_TRIALS):
                rng = np.random.default_rng(SEED + int(ratio * 100000) + t)
                sub = subsample(points, ratio, rng)
                if len(sub) < 20:
                    continue
                setattr(clusterer, param_name, val)
                parts = actual_split(clusterer, sub)
                parts_sum += parts
                mae_sum += abs(parts - k)
                valid += 1
                if parts == k:
                    correct += 1
            correct_matrix[gi, ri] = correct / valid if valid else float("nan")
            parts_matrix[gi, ri] = parts_sum / valid if valid else float("nan")
            mae_matrix[gi, ri] = mae_sum / valid if valid else float("nan")

    return correct_matrix, parts_matrix, mae_matrix


def plot_heatmap(mae_matrix, grid, param_name, title, out_png,
                 xlabels=None, parts_matrix=None, vmax=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrows, ncols = mae_matrix.shape
    fig, ax = plt.subplots(figsize=(0.9 * ncols + 2.5, 0.7 * nrows + 1.5))

    # 颜色 = MAE(count): 绿=低误差(好), 红=高误差(坏)
    # vmax 统一由外部指定, 使两张热图共享同一 colorbar 尺度 (避免各自缩放导致
    # "要么全绿要么全红"的跳变)。
    if vmax is None:
        vmax = float(np.nanmax(mae_matrix))
    im = ax.imshow(mae_matrix, aspect="auto", cmap="RdYlGn_r",
                   vmin=0.0, vmax=max(vmax, 0.01), interpolation="nearest")
    ax.set_xticks(np.arange(ncols))
    ax.set_xticklabels(xlabels if xlabels else [fmt_pct(r) for r in RATIOS],
                       rotation=45, ha="right")
    ax.set_yticks(np.arange(nrows))
    ax.set_yticklabels([f"{v:.2f}" for v in grid])
    ax.set_xlabel("Point retention ratio (density)")
    ax.set_ylabel(param_name)
    # 默认行用红色加粗刻度标注 (零遮挡)
    if param_name == "valley_depth_rel":
        default_idx = grid.index(0.90) if 0.90 in grid else None
        default_note = "(default θ0=0.90 marked in red)"
    else:
        default_idx = grid.index(0.35) if 0.35 in grid else None
        default_note = "(default 0.35m marked in red)"
    if default_idx is not None:
        ax.get_yticklabels()[default_idx].set_color("red")
        ax.get_yticklabels()[default_idx].set_fontweight("bold")
    ax.set_title(title + "\n" + default_note, fontsize=10)

    # 单元格标注: MAE (括号内为平均分块数)
    for i in range(nrows):
        for j in range(ncols):
            v = mae_matrix[i, j]
            if np.isnan(v):
                txt = "-"
            else:
                txt = f"{v:.2f}"
                if parts_matrix is not None:
                    txt += f"\n({parts_matrix[i, j]:.1f})"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("MAE (count) = mean|parts - k|")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_small_multiples(mae_matrix, grid, param_name, title, out_png, xlabel):
    """Small multiples: 每个密度一个子图, MAE(count) vs 超参数值。

    彻底消除多密度曲线重叠问题; 能直观看出"密度越低曲线越陡(越敏感)"。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrows, n_panels = mae_matrix.shape  # nrows=参数值数, n_panels=密度数

    panel_cols = 4
    panel_rows = int(np.ceil(n_panels / panel_cols))
    fig, axes = plt.subplots(panel_rows, panel_cols,
                             figsize=(4.0 * panel_cols, 2.8 * panel_rows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    if param_name == "valley_depth_rel":
        default_val = 0.90
    else:
        default_val = 0.35

    ymax = float(np.nanmax(mae_matrix)) * 1.15
    # 每个密度一个面板, 颜色随密度加深 (稀疏越红), 便于肉眼追踪趋势
    panel_colors = plt.cm.viridis(np.linspace(0.05, 0.95, n_panels))
    for ri in range(n_panels):
        ax = axes[ri]
        ax.plot(grid, mae_matrix[:, ri], marker="o", ms=3.5, lw=1.6,
                color=panel_colors[ri])
        ax.axvline(default_val, color="red", ls="--", lw=1, alpha=0.7)
        ax.axhline(0.0, color="gray", ls=":", lw=0.8, alpha=0.7)
        ax.set_title(f"{fmt_pct(RATIOS[ri])}", fontsize=9,
                     color=panel_colors[ri], fontweight="bold")
        ax.set_ylim(-0.03, max(ymax, 0.1))
        ax.grid(alpha=0.3)

    for i in range(n_panels, len(axes)):
        axes[i].axis("off")

    # 底部统一轴标签
    fig.text(0.5, 0.02, xlabel, ha="center", fontsize=11)
    fig.text(0.02, 0.5, "MAE (count) = mean|parts - k|", va="center",
             rotation=90, fontsize=11)
    fig.suptitle(title + f"\n(red dashed = default {default_val:.2f})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0.05, 0.04, 1, 0.94])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "default.yaml")))
    clusterer = InstanceClusterer(cfg)

    xyz, sem, inst = load_cloud_ply(os.path.join(ROOT, "evalaute_test", "cloudR10.ply"))
    c = [cc for cc in find_sticky_clusters(xyz, inst, eps=0.03)
         if set(cc["instances"]) == set([2, 3, 4, 5])][0]
    points = xyz[c["idx"]]
    k = len(c["instances"])
    print(f"粘连簇: cloudR10 实例 {c['instances']}, {c['n_points']} 点, k={k}")

    os.makedirs(OUT_DIR, exist_ok=True)

    # --- 超参数 1: valley_depth_rel (相对深度阈值) ---
    print("\n扫描 valley_depth_rel (相对深度阈值) ...")
    cm1, pm1, mae1 = scan_hyperparam(clusterer, points, k, "valley_depth_rel", VALLEY_DEPTH_GRID)
    print("MAE 矩阵 (行=valley_depth_rel, 列=密度):")
    print("        " + "  ".join(f"{fmt_pct(r):>6}" for r in RATIOS))
    for gi, val in enumerate(VALLEY_DEPTH_GRID):
        row = "  ".join(f"{mae1[gi, ri]:6.2f}" for ri in range(len(RATIOS)))
        print(f"  {val:.2f}   {row}")

    # --- 超参数 2: min_peak_dist_m (谷底跨度) ---
    print("\n扫描 min_peak_dist_m (谷底跨度) ...")
    cm2, pm2, mae2 = scan_hyperparam(clusterer, points, k, "min_peak_dist_m", MIN_PEAK_DIST_GRID)
    print("MAE 矩阵 (行=min_peak_dist_m, 列=密度):")
    print("        " + "  ".join(f"{fmt_pct(r):>6}" for r in RATIOS))
    for gi, val in enumerate(MIN_PEAK_DIST_GRID):
        row = "  ".join(f"{mae2[gi, ri]:6.2f}" for ri in range(len(RATIOS)))
        print(f"  {val:.2f}   {row}")

    # 统一 colorbar 尺度: 取两参数全局最大 MAE 的向上取整 (至少 1.0),
    # 使 fig4(θ0, 值 0~0.04) 与 fig5(谷底跨度, 值 0~2.85) 共享同一色标。
    # θ0 热图因此呈"几乎全绿"——这本身就是"θ0 不敏感"的诚实表达。
    global_vmax = max(float(np.nanmax(mae1)), float(np.nanmax(mae2)))
    global_vmax = max(np.ceil(global_vmax), 1.0)

    plot_heatmap(
        mae1, VALLEY_DEPTH_GRID, "valley_depth_rel",
        "Sensitivity of relative depth threshold θ0 (valley_depth_rel)\n"
        "color = MAE(count), cell text = (mean #parts)",
        os.path.join(OUT_DIR, "fig4_hyperparam_valley_depth.png"),
        parts_matrix=pm1, vmax=global_vmax)
    plot_small_multiples(
        mae1, VALLEY_DEPTH_GRID, "valley_depth_rel",
        "MAE(count) vs. relative depth threshold θ0 (one panel per density)",
        os.path.join(OUT_DIR, "fig4_hyperparam_valley_depth_line.png"),
        xlabel="valley_depth_rel (θ0)")

    plot_heatmap(
        mae2, MIN_PEAK_DIST_GRID, "min_peak_dist_m",
        "Sensitivity of minimum peak distance (min_peak_dist_m)\n"
        "color = MAE(count), cell text = (mean #parts)",
        os.path.join(OUT_DIR, "fig5_hyperparam_min_peak_dist.png"),
        parts_matrix=pm2, vmax=global_vmax)
    plot_small_multiples(
        mae2, MIN_PEAK_DIST_GRID, "min_peak_dist_m",
        "MAE(count) vs. minimum peak distance (one panel per density)",
        os.path.join(OUT_DIR, "fig5_hyperparam_min_peak_dist_line.png"),
        xlabel="min_peak_dist_m (m)")

    # 保存数值结果
    result = {
        "valley_depth_rel": {
            "grid": VALLEY_DEPTH_GRID,
            "correct_matrix": cm1.tolist(),
            "parts_matrix": pm1.tolist(),
            "mae_matrix": mae1.tolist(),
        },
        "min_peak_dist_m": {
            "grid": MIN_PEAK_DIST_GRID,
            "correct_matrix": cm2.tolist(),
            "parts_matrix": pm2.tolist(),
            "mae_matrix": mae2.tolist(),
        },
        "ratios": RATIOS,
        "n_trials": N_TRIALS,
    }
    with open(os.path.join(OUT_DIR, "hyperparam_summary.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n输出目录: {OUT_DIR}")
    print("  fig4_hyperparam_valley_depth.png       (热图)")
    print("  fig4_hyperparam_valley_depth_line.png  (折线)")
    print("  fig5_hyperparam_min_peak_dist.png      (热图)")
    print("  fig5_hyperparam_min_peak_dist_line.png (折线)")
    print("  hyperparam_summary.json")


if __name__ == "__main__":
    main()
