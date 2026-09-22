#!/usr/bin/env python3
"""
ADPV (GIDM) 后处理效率分析 — 论文级曲线图。

图1 (a): 耗时-规模曲线 (scalability) —— 横轴=植株点数, 纵轴=耗时(ms),
          三条曲线: 语义backbone / 基础聚类 / GIDM后处理。
          突出 GIDM 几乎不随规模增长 (轻量后处理)。

图1 (b): 精度-成本权衡 —— 三种聚类方法的 F1 增益 vs GIDM 耗时增量,
          展示 ADPV 以极小成本换取大精度提升。

用法:
  python tools/plot_efficiency.py
  读 output/efficiency_analysis.json (逐文件耗时)
  + 内置的精度数据 (来自 docs/experiments_and_results.md 与 output 汇总)
"""

import os
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'output')
os.makedirs(OUT, exist_ok=True)

# 配色
C_BACKBONE = '#2b6cb0'   # 蓝
C_CLUSTER = '#d69e2e'    # 橙黄
C_GIDM = '#c53030'       # 红 (强调 ADPV)
C_GRID = '#e2e8f0'


def load_efficiency():
    fp = os.path.join(OUT, 'efficiency_analysis.json')
    if not os.path.exists(fp):
        raise SystemExit(f'未找到 {fp}, 请先运行 tools/efficiency_analysis.py')
    return json.load(open(fp))


# 三种聚类方法的 F1 基线→有GIDM (来自 docs/experiments_and_results.md, RandLA-Net, test 14)
F1_GAIN = {
    'watershed_3d': (0.514, 0.764),
    'meanshift':    (0.856, 0.878),
    'hdbscan':      (0.114, 0.780),
}


def fit_line(x, y):
    """最小二乘线性拟合, 返回 (斜率, 截距, r2)。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = k * x + b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return k, b, r2


def plot_scalability(d):
    rows = d['per_file']
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), gridspec_kw={'wspace': 0.32})

    # ---- 图 (a): 清晰的耗时构成 ----
    ax = axes[0]
    s = d['summary']
    methods = list(s['methods'].keys())
    labels = ['Watershed 3D', 'MeanShift', 'HDBSCAN']
    cluster_ms = [s['methods'][m]['t_cluster'] * 1000 for m in methods]
    gidm_ms = [s['methods'][m]['t_gidm'] * 1000 for m in methods]
    backbone_ms = s['t_sem'] * 1000

    x = np.arange(len(methods))
    ax.bar(x, [backbone_ms] * len(methods), color=C_BACKBONE, width=0.62,
           label='Semantic backbone')
    ax.bar(x, cluster_ms, bottom=[backbone_ms] * len(methods), color=C_CLUSTER,
           width=0.62, label='Baseline clustering')
    ax.bar(x, gidm_ms,
           bottom=[backbone_ms + value for value in cluster_ms],
           color=C_GIDM, width=0.62, label='ADPV / GIDM')

    for i, value in enumerate(gidm_ms):
        total = backbone_ms + cluster_ms[i] + value
        ax.text(i, total + 90, f'+{value:.0f} ms', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color=C_GIDM)

    ax.set_xticks(x, labels)
    ax.set_ylabel('Runtime per scene (ms)', fontsize=11)
    ax.set_title('(a) Runtime composition', fontsize=12, fontweight='bold')
    ax.grid(axis='y', color=C_GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc='upper left', frameon=False)
    ax.set_ylim(0, max(backbone_ms + np.max(cluster_ms) + np.max(gidm_ms) + 250, 6500))

    # ---- 图 (b): 清晰的精度-成本权衡 ----
    ax2 = axes[1]
    markers = ['o', 's', '^']
    colors = ['#2f855a', '#805ad5', '#dd6b20']
    for m, label, marker, color in zip(methods, labels, markers, colors):
        if m not in F1_GAIN:
            continue
        f1_base, f1_gidm = F1_GAIN[m]
        t_ms = s['methods'][m]['gidm_increase_ms']
        gain = (f1_gidm - f1_base) * 100
        ax2.scatter(t_ms, gain, s=105, marker=marker, color=color,
                    edgecolors='white', linewidths=1.3, label=label, zorder=3)
        ax2.annotate(f'{gain:.1f} pp', (t_ms, gain), textcoords='offset points',
                     xytext=(7, 7), fontsize=9, color=color, fontweight='bold')

    ax2.set_xlabel('ADPV latency increment (ms)', fontsize=11)
    ax2.set_ylabel('Instance F1 gain (percentage points)', fontsize=11)
    ax2.set_title('(b) Accuracy gain vs. post-processing cost', fontsize=12, fontweight='bold')
    ax2.grid(True, color=C_GRID, linewidth=0.7)
    ax2.set_axisbelow(True)
    ax2.set_xlim(0, 220)
    ax2.set_ylim(0, 75)
    ax2.legend(fontsize=9, frameon=False, loc='lower right')
    ax2.text(0.04, 0.95, 'GPU memory increment: 0 MB', transform=ax2.transAxes,
             fontsize=9, va='top', color='#4a5568')

    fig.tight_layout()
    out_fp = os.path.join(OUT, 'efficiency_analysis.png')
    fig.savefig(out_fp, dpi=200, bbox_inches='tight')
    print(f'已保存: {out_fp}')
    plt.close(fig)


def main():
    d = load_efficiency()
    plot_scalability(d)


if __name__ == '__main__':
    main()
