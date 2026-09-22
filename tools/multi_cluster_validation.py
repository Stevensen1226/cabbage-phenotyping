"""
GIDM 密度 / 超参数敏感性 —— 多粘连簇跨簇验证。

将单簇实验推广到 evalulate_test 中全部符合条件的粘连簇
(2~5 株、点数 >= 2000, 共 25 个), 得到统计上更可靠、更全面的结论。

实验一: 密度敏感性跨簇汇总
  - 25 个粘连簇 × 8 个保留比例 × N_TRIALS 次随机采样
  - 每个簇按各自真值株数 k 判定 correct/over/under
  - 跨簇 macro 平均正确率 (每个簇权重相同)

实验二: 超参数敏感性跨簇汇总
  - 谷底跨度 min_peak_dist_m 与相对深度阈值 valley_depth_rel
  - 跨簇平均正确率

用法:
  python tools/multi_cluster_validation.py
"""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.density_sensitivity import (
    load_cloud_ply, find_sticky_clusters, subsample, actual_split,
    InstanceClusterer, fmt_pct, RATIOS,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "output", "density_sensitivity", "multi_cluster")

# 密度敏感性采样次数
DENSITY_N_TRIALS = 40
# 超参数敏感性采样次数
HYPER_N_TRIALS = 15

VALLEY_DEPTH_GRID = [0.80, 0.85, 0.90, 0.95, 0.98, 1.00, 1.05, 1.10]
MIN_PEAK_DIST_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def collect_clusters(min_plants=2, max_plants=5, min_points=2000, eps=0.03):
    """收集所有符合条件的粘连簇, 返回 [(file, instances, points, k), ...]"""
    files = sorted(
        f for f in os.listdir(os.path.join(ROOT, "evalaute_test"))
        if f.startswith("cloudR") and f.endswith(".ply")
        and not f.endswith("_sor.ply"))
    clusters = []
    for fn in files:
        try:
            xyz, sem, inst = load_cloud_ply(
                os.path.join(ROOT, "evalaute_test", fn))
        except Exception:
            continue
        for c in find_sticky_clusters(xyz, inst, eps=eps):
            if min_plants <= c["n_instances"] <= max_plants \
                    and c["n_points"] >= min_points:
                clusters.append({
                    "file": fn,
                    "instances": c["instances"],
                    "points": xyz[c["idx"]],
                    "k": c["n_instances"],
                    "n": c["n_points"],
                })
    clusters.sort(key=lambda c: c["n"])
    return clusters


def run_density_sensitivity(clusterer, clusters):
    """跨簇密度敏感性: 返回 (summary, by_k, cluster_results)。

    cluster_results: 按簇组织的完整结果, 每个簇记录其在各密度下的平均 MAE,
    用于后续按株数 k 分层与按难度 (易/难) 分层的分析。
    """
    n_ratios = len(RATIOS)

    # 汇总容器
    agg = {r: {"correct": [], "over": [], "under": [], "mae": []} for r in RATIOS}
    # 按株数分组: {k: {ratio: [mae, ...]}}
    by_k = defaultdict(lambda: defaultdict(list))
    # 按簇组织: [ {file, k, n, mae_by_ratio, mae_full}, ... ]
    cluster_results = []

    for ci, cl in enumerate(clusters):
        k = cl["k"]
        pts = cl["points"]
        mae_by_ratio = {}
        for ratio in RATIOS:
            correct = over = under = valid = 0
            mae_sum = 0
            for t in range(DENSITY_N_TRIALS):
                rng = np.random.default_rng(
                    42 + int(ratio * 100000) + t + ci * 7919)
                sub = subsample(pts, ratio, rng)
                if len(sub) < 20:
                    continue
                parts = actual_split(clusterer, sub)
                valid += 1
                mae_sum += abs(parts - k)
                if parts < k:
                    under += 1
                elif parts == k:
                    correct += 1
                else:
                    over += 1
            if valid:
                agg[ratio]["correct"].append(correct / valid)
                agg[ratio]["over"].append(over / valid)
                agg[ratio]["under"].append(under / valid)
                agg[ratio]["mae"].append(mae_sum / valid)
                by_k[k][ratio].append(mae_sum / valid)
                mae_by_ratio[ratio] = mae_sum / valid
        cluster_results.append({
            "file": cl["file"],
            "instances": cl["instances"],
            "k": k,
            "n": cl["n"],
            "mae_by_ratio": mae_by_ratio,
            "mae_full": mae_by_ratio.get(1.0, float("nan")),  # 100% 密度下的 MAE
        })

    summary = {}
    for r in RATIOS:
        summary[r] = {
            "correct_mean": float(np.mean(agg[r]["correct"])),
            "correct_std": float(np.std(agg[r]["correct"])),
            "over_mean": float(np.mean(agg[r]["over"])),
            "under_mean": float(np.mean(agg[r]["under"])),
            "mae_mean": float(np.mean(agg[r]["mae"])),
            "mae_std": float(np.std(agg[r]["mae"])),
            "mae_values": [float(x) for x in agg[r]["mae"]],
            "n_clusters": int(len(agg[r]["correct"])),
        }
    return summary, by_k, cluster_results


def run_hyperparam_sensitivity(clusterer, clusters, param_name, grid):
    """跨簇超参数敏感性: 返回 (correct_matrix, mae_matrix) shape (len(grid), len(RATIOS))。"""
    cm = np.zeros((len(grid), len(RATIOS)))
    mae_matrix = np.zeros((len(grid), len(RATIOS)))

    for gi, val in enumerate(grid):
        for ri, ratio in enumerate(RATIOS):
            rates = []
            maes = []
            for ci, cl in enumerate(clusters):
                k = cl["k"]
                pts = cl["points"]
                correct = valid = 0
                mae_sum = 0
                for t in range(HYPER_N_TRIALS):
                    rng = np.random.default_rng(
                        1000 + int(ratio * 100000) + t + ci * 7919 + gi * 13)
                    sub = subsample(pts, ratio, rng)
                    if len(sub) < 20:
                        continue
                    setattr(clusterer, param_name, val)
                    parts = actual_split(clusterer, sub)
                    valid += 1
                    mae_sum += abs(parts - k)
                    if parts == k:
                        correct += 1
                if valid:
                    rates.append(correct / valid)
                    maes.append(mae_sum / valid)
            cm[gi, ri] = float(np.mean(rates)) if rates else float("nan")
            mae_matrix[gi, ri] = float(np.mean(maes)) if maes else float("nan")

    # 恢复默认值
    if param_name == "valley_depth_rel":
        setattr(clusterer, param_name, 0.90)
    else:
        setattr(clusterer, param_name, 0.35)
    return cm, mae_matrix


def plot_density(summary, by_k, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ratios = [r for r in RATIOS]
    mae_values = [summary[r]["mae_values"] for r in RATIOS]
    mae_mean = [summary[r]["mae_mean"] for r in RATIOS]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    # (a) 跨簇计数误差分布: 堆叠柱状图 (按 MAE 误差档分色)
    ax = axes[0]
    labels = [fmt_pct(r) for r in RATIOS]
    positions = np.arange(len(RATIOS))

    # 统计每个密度下各误差档 (MAE=0,1,2,3...) 的簇数量
    max_err = max(max(int(round(v)) for v in vals) for vals in mae_values)
    err_levels = list(range(0, max_err + 1))
    bottoms = np.zeros(len(RATIOS))
    # 颜色: 0=绿(完美), 1=橙, 2=红, >=3=深红
    err_colors = {0: "#2ECC71", 1: "#F39C12", 2: "#E74C3C"}
    for lv in err_levels:
        counts = []
        for vals in mae_values:
            counts.append(sum(1 for v in vals if int(round(v)) == lv))
        color = err_colors.get(lv, "#922B21")
        ax.bar(positions, counts, bottom=bottoms, width=0.6,
               color=color, edgecolor="white", lw=0.5,
               label=f"MAE = {lv}")
        bottoms = bottoms + np.array(counts)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(bottoms) * 1.08)
    n_clusters = summary[RATIOS[0]]["n_clusters"]
    ax.set_xlabel("Point retention ratio (density)")
    ax.set_ylabel("Number of clusters")
    ax.set_title(f"(a) Counting error distribution vs. density\n"
                 f"({n_clusters} sticky clusters, k=2~5)")
    # 图例放在子图底部外侧 (x 轴标签下方)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=3, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    # (b) 按株数分组的 MAE 曲线
    ax = axes[1]
    colors = {2: "#1E8449", 3: "#2471A3", 4: "#7D3C98", 5: "#B03A2E"}
    for k in sorted(by_k.keys()):
        xs = []
        ys = []
        for r in RATIOS:
            if by_k[k][r]:
                xs.append(r)
                ys.append(np.mean(by_k[k][r]))
        ax.plot(xs, ys, marker="o", ms=4, lw=1.5, color=colors.get(k, "gray"),
                label=f"k={k} ({len(by_k[k][RATIOS[0]])} clusters)")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_ylim(bottom=-0.08)
    ax.set_xlabel("Point retention ratio (density)")
    ax.set_ylabel("Mean MAE (count)")
    ax.set_title("(b) Counting error by plant count k")
    # 图例放在子图底部外侧 (x 轴标签下方)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)

    fig.suptitle("GIDM density sensitivity — multi-cluster validation (MAE)",
                 fontsize=13, fontweight="bold", y=0.98)
    # 布局: 顶部给 suptitle, 底部给图例, 中部给子图
    fig.subplots_adjust(top=0.82, bottom=0.20, wspace=0.25)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_density_by_k(cluster_results, out_png):
    """方案 B: 按粘连株数 k 分层的密度敏感性 (每条线 = 一个 k, 带簇间标准差)。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_k = defaultdict(list)
    for c in cluster_results:
        by_k[c["k"]].append(c)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {2: "#1E8449", 3: "#2471A3", 4: "#7D3C98", 5: "#B03A2E"}
    ks = sorted(by_k.keys())
    n_groups = len(RATIOS)
    n_bars = len(ks)
    bar_width = 0.8 / n_bars
    x = np.arange(n_groups)

    for bi, k in enumerate(ks):
        group = by_k[k]
        means = []
        stds = []
        for r in RATIOS:
            vals = [c["mae_by_ratio"][r] for c in group if r in c["mae_by_ratio"]]
            means.append(np.mean(vals) if vals else 0.0)
            stds.append(np.std(vals) if vals else 0.0)
        offset = (bi - (n_bars - 1) / 2) * bar_width
        ax.bar(x + offset, means, width=bar_width, color=colors.get(k, "gray"),
               edgecolor="white", lw=0.5, label=f"k={k} ({len(group)})",
               yerr=stds, capsize=3, error_kw=dict(lw=1, ecolor="#34495E"))

    ax.set_xticks(x)
    ax.set_xticklabels([fmt_pct(r) for r in RATIOS])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Point retention ratio (density)")
    ax.set_ylabel("Mean MAE (count) per cluster")
    ax.set_title("Density sensitivity stratified by plant count k")
    ax.legend(fontsize=8.5, loc="upper left", frameon=False, ncol=2,
              handlelength=1.4, labelspacing=0.5, columnspacing=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_density_by_difficulty(cluster_results, out_png):
    """方案 C: 按难度 (100% 密度下是否切错) 分层的密度敏感性。

    易簇 = 全密度 MAE=0, 难簇 = 全密度 MAE>0 (基线欠分割)。
    揭示: 易簇对密度敏感 (MAE 从 0 上升), 难簇基线误差主导 (MAE 恒高)。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    easy = [c for c in cluster_results if c["mae_full"] == 0.0]
    hard = [c for c in cluster_results if c["mae_full"] > 0.0]

    fig, ax = plt.subplots(figsize=(8, 5))
    groups = [("easy (full-density MAE=0)", easy, "#2ECC71"),
              ("hard (full-density MAE>0)", hard, "#E74C3C")]

    n_groups = len(RATIOS)
    n_bars = len(groups)
    bar_width = 0.38
    x = np.arange(n_groups)

    for bi, (label, group, color) in enumerate(groups):
        if not group:
            continue
        means, stds = [], []
        for r in RATIOS:
            vals = [c["mae_by_ratio"][r] for c in group if r in c["mae_by_ratio"]]
            means.append(np.mean(vals) if vals else 0.0)
            stds.append(np.std(vals) if vals else 0.0)
        offset = (bi - (n_bars - 1) / 2) * bar_width
        ax.bar(x + offset, means, width=bar_width, color=color,
               edgecolor="white", lw=0.5, label=f"{label} ({len(group)})",
               yerr=stds, capsize=3, error_kw=dict(lw=1, ecolor="#34495E"))

    ax.set_xticks(x)
    ax.set_xticklabels([fmt_pct(r) for r in RATIOS])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Point retention ratio (density)")
    ax.set_ylabel("Mean MAE (count) per cluster")
    ax.set_title("Density sensitivity stratified by cluster difficulty")
    ax.legend(fontsize=9, loc="upper left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hyperparam(mae_matrix, grid, param_name, out_png, xlabel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrows, ncols = mae_matrix.shape
    fig, ax = plt.subplots(figsize=(8.5, 5))

    colors = plt.cm.viridis(np.linspace(0.05, 0.95, ncols))
    markers = ["o", "s", "^", "v", "D", "P", "X", "*", "p", "h", "d", "8"]
    for ri in range(ncols):
        ax.plot(grid, mae_matrix[:, ri], marker=markers[ri % len(markers)],
                ms=5, lw=1.2, color=colors[ri], label=fmt_pct(RATIOS[ri]),
                markerfacecolor="none", markeredgewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean MAE (count) across clusters")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    # 默认值: 淡竖虚线 + title 注明 (零遮挡)
    if param_name == "min_peak_dist_m":
        ax.axvline(0.35, color="black", ls="--", lw=1, alpha=0.5)
        default_note = "(default 0.35m)"
    else:
        ax.axvline(0.90, color="black", ls="--", lw=1, alpha=0.5)
        default_note = "(default θ0=0.90)"
    ax.set_title(f"Sensitivity of {param_name} — multi-cluster (MAE) {default_note}",
                 fontsize=10)
    ax.legend(title="density", fontsize=6.5, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "default.yaml")))
    clusterer = InstanceClusterer(cfg)

    clusters = collect_clusters()
    print(f"收集到 {len(clusters)} 个粘连簇 (2~5 株, 点数>=2000):")
    byk = Counter(c["k"] for c in clusters)
    print(f"  按株数分布: {dict(sorted(byk.items()))}")

    os.makedirs(OUT_DIR, exist_ok=True)

    # === 实验一: 密度敏感性跨簇 ===
    print(f"\n[1/2] 密度敏感性跨簇 ({len(clusters)} 簇 × {len(RATIOS)} 密度 × "
          f"{DENSITY_N_TRIALS} 次)...")
    summary, by_k, cluster_results = run_density_sensitivity(clusterer, clusters)
    plot_density(summary, by_k, os.path.join(OUT_DIR, "figA_density_multi.png"))
    plot_density_by_k(cluster_results, os.path.join(OUT_DIR, "figD_density_by_k.png"))
    plot_density_by_difficulty(cluster_results, os.path.join(OUT_DIR, "figE_density_by_difficulty.png"))

    print("\n密度敏感性跨簇汇总 (mean MAE):")
    print("  " + "  ".join(f"{fmt_pct(r):>6}" for r in RATIOS))
    print("  " + "  ".join(f"{summary[r]['mae_mean']:6.2f}" for r in RATIOS))
    print("  " + "  ".join(f"(±{summary[r]['mae_std']:.2f})" for r in RATIOS))

    # 难度分层统计
    easy = [c for c in cluster_results if c["mae_full"] == 0.0]
    hard = [c for c in cluster_results if c["mae_full"] > 0.0]
    print(f"\n难度分层 (按 100% 密度 MAE):")
    print(f"  易簇 (MAE=0): {len(easy)} 个")
    print(f"  难簇 (MAE>0): {len(hard)} 个")
    for r in RATIOS:
        e = np.mean([c["mae_by_ratio"][r] for c in easy if r in c["mae_by_ratio"]])
        h = np.mean([c["mae_by_ratio"][r] for c in hard if r in c["mae_by_ratio"]])
        print(f"    density {fmt_pct(r):>5}: 易簇 MAE={e:.2f}  难簇 MAE={h:.2f}")

    # === 实验二: 超参数敏感性跨簇 ===
    print(f"\n[2/2] 超参数敏感性跨簇 (min_peak_dist_m + valley_depth_rel)...")

    cm_mpd, mae_mpd = run_hyperparam_sensitivity(
        clusterer, clusters, "min_peak_dist_m", MIN_PEAK_DIST_GRID)
    plot_hyperparam(
        mae_mpd, MIN_PEAK_DIST_GRID, "min_peak_dist_m",
        os.path.join(OUT_DIR, "figB_hyperparam_mpd_multi.png"),
        xlabel="min_peak_dist_m (m)")

    cm_vdr, mae_vdr = run_hyperparam_sensitivity(
        clusterer, clusters, "valley_depth_rel", VALLEY_DEPTH_GRID)
    plot_hyperparam(
        mae_vdr, VALLEY_DEPTH_GRID, "valley_depth_rel",
        os.path.join(OUT_DIR, "figC_hyperparam_vdr_multi.png"),
        xlabel="valley_depth_rel (θ0)")

    # 保存结果
    result = {
        "n_clusters": len(clusters),
        "n_clusters_by_k": dict(sorted(byk.items())),
        "ratios": RATIOS,
        "density_n_trials": DENSITY_N_TRIALS,
        "hyper_n_trials": HYPER_N_TRIALS,
        "density_summary": {
            str(r): summary[r] for r in RATIOS
        },
        "min_peak_dist_m": {
            "grid": MIN_PEAK_DIST_GRID,
            "correct_matrix": cm_mpd.tolist(),
            "mae_matrix": mae_mpd.tolist(),
        },
        "valley_depth_rel": {
            "grid": VALLEY_DEPTH_GRID,
            "correct_matrix": cm_vdr.tolist(),
            "mae_matrix": mae_vdr.tolist(),
        },
        "cluster_results": [
            {
                "file": c["file"],
                "instances": c["instances"],
                "k": c["k"],
                "n": c["n"],
                "mae_full": c["mae_full"],
                "mae_by_ratio": {str(r): c["mae_by_ratio"][r]
                                 for r in RATIOS if r in c["mae_by_ratio"]},
            }
            for c in cluster_results
        ],
    }
    with open(os.path.join(OUT_DIR, "multi_cluster_summary.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n输出目录: {OUT_DIR}")
    print("  figA_density_multi.png          (密度跨簇汇总)")
    print("  figD_density_by_k.png           (按株数 k 分层)")
    print("  figE_density_by_difficulty.png  (按易/难分层)")
    print("  figB_hyperparam_mpd_multi.png   (谷底跨度跨簇)")
    print("  figC_hyperparam_vdr_multi.png   (θ0 跨簇)")
    print("  multi_cluster_summary.json")


if __name__ == "__main__":
    main()
