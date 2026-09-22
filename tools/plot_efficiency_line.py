#!/usr/bin/env python3
"""生成 ADPV 效率分析的信息密集版图 (2×2)。

四张子图:
  (a) 耗时构成堆叠柱: backbone + 基础聚类 + ADPV(GIDM), 三种聚类方法
  (b) 精度提升分组柱: 基线 F1 vs +ADPV F1, 三种聚类方法
  (c) 可扩展性散点: 横轴=植株点数, 纵轴=耗时, 三条曲线 (14 个文件)
  (d) 成本-增益散点: ADPV 延迟增量 vs F1 增益, 标注 GPU/CPU 内存

不覆盖原 efficiency_analysis.png。
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'output')
INPUT = os.path.join(OUT, 'efficiency_analysis.json')
OUTPUT = os.path.join(OUT, 'efficiency_analysis_line.png')

LABELS = {
    'watershed_3d': 'Watershed 3D',
    'meanshift': 'MeanShift',
    'hdbscan': 'HDBSCAN',
}
F1_GAIN = {
    'watershed_3d': (0.514, 0.764),
    'meanshift': (0.856, 0.878),
    'hdbscan': (0.114, 0.780),
}
CLUSTER_COLORS = ['#c53030', '#2f855a', '#805ad5']
GRID = '#e2e8f0'


def main():
    data = json.load(open(INPUT, encoding='utf-8'))
    summary = data['summary']
    methods = list(summary['methods'].keys())
    labels = [LABELS.get(m, m) for m in methods]
    x = np.arange(len(methods))

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
    fig.subplots_adjust(hspace=0.36, wspace=0.30)

    # ================= (a) 耗时构成堆叠柱 =================
    ax = axes[0, 0]
    backbone_ms = summary['t_sem'] * 1000
    cluster_ms = [summary['methods'][m]['t_cluster'] * 1000 for m in methods]
    gidm_ms = [summary['methods'][m]['t_gidm'] * 1000 for m in methods]

    ax.bar(x, [backbone_ms] * len(methods), width=0.55,
           color='#2b6cb0', label='Semantic backbone')
    ax.bar(x, cluster_ms, bottom=[backbone_ms] * len(methods), width=0.55,
           color='#d69e2e', label='Baseline clustering')
    ax.bar(x, gidm_ms,
           bottom=[backbone_ms + c for c in cluster_ms],
           width=0.55, color='#c53030', label='ADPV (GIDM)')

    for i in range(len(methods)):
        total = backbone_ms + cluster_ms[i] + gidm_ms[i]
        ax.text(i, total + 120, f'{gidm_ms[i]:.0f} ms',
                ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                color='#c53030')

    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel('Runtime per scene (ms)', fontsize=10)
    ax.set_title('(a) Runtime composition', fontsize=12, fontweight='bold')
    ax.grid(axis='y', color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.5, frameon=False, loc='upper left')
    ax.set_ylim(0, backbone_ms + max(cluster_ms) + max(gidm_ms) + 700)

    # ================= (b) 精度提升分组柱 =================
    ax = axes[0, 1]
    base_f1 = [F1_GAIN[m][0] for m in methods]
    gidm_f1 = [F1_GAIN[m][1] for m in methods]
    width = 0.32

    ax.bar(x - width / 2, base_f1, width, color='#a0aec0', label='Baseline')
    ax.bar(x + width / 2, gidm_f1, width, color='#2b6cb0', label='+ ADPV')

    for i in range(len(methods)):
        ax.text(i - width / 2, base_f1[i] + 0.03, f'{base_f1[i]:.2f}',
                ha='center', fontsize=8, color='#4a5568')
        ax.text(i + width / 2, gidm_f1[i] + 0.03, f'{gidm_f1[i]:.2f}',
                ha='center', fontsize=8, fontweight='bold', color='#1a365d')
        gain = (gidm_f1[i] - base_f1[i]) * 100
        ax.annotate(f'+{gain:.0f} pp', (i, (base_f1[i] + gidm_f1[i]) / 2),
                    xytext=(0, 18), textcoords='offset points', ha='center',
                    fontsize=8.5, color='#c53030', fontweight='bold')

    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel('Instance F1', fontsize=10)
    ax.set_title('(b) Instance F1: baseline vs. + ADPV', fontsize=12, fontweight='bold')
    ax.grid(axis='y', color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.5, frameon=False, loc='upper left')
    ax.set_ylim(0, 1.0)

    # ================= (c) 可扩展性散点 =================
    ax = axes[1, 0]
    rows = data['per_file']
    n_cab = np.array([r['n_cab'] for r in rows]) / 1000.0
    t_sem = np.array([r['t_sem'] for r in rows]) * 1000.0
    t_clu = np.array([r['t_cluster_watershed_3d'] for r in rows]) * 1000.0
    t_gidm = np.array([r['t_gidm_watershed_3d'] for r in rows]) * 1000.0

    for yy, cc, ll in [(t_sem, '#2b6cb0', 'Backbone'),
                       (t_clu, '#d69e2e', 'Clustering'),
                       (t_gidm, '#c53030', 'ADPV (GIDM)')]:
        ax.scatter(n_cab, yy, s=34, color=cc, alpha=0.7,
                   edgecolors='white', linewidths=0.5, zorder=3)
        z = np.polyfit(n_cab, yy, 1)
        xs = np.linspace(n_cab.min(), n_cab.max(), 50)
        ax.plot(xs, np.polyval(z, xs), color=cc, lw=1.8, label=ll, zorder=2)

    ax.set_xlabel('Number of cabbage points (K)', fontsize=10)
    ax.set_ylabel('Runtime (ms)', fontsize=10)
    ax.set_title('(c) Scalability vs. scene size', fontsize=12, fontweight='bold')
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.5, frameon=False, loc='upper left')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # ================= (d) 成本-增益散点 =================
    ax = axes[1, 1]
    latency = [summary['methods'][m]['gidm_increase_ms'] for m in methods]
    gain = [(F1_GAIN[m][1] - F1_GAIN[m][0]) * 100 for m in methods]
    cpu = [summary['methods'][m]['cpu_mb'] for m in methods]

    for i, m in enumerate(methods):
        ax.scatter(latency[i], gain[i], s=180, color=CLUSTER_COLORS[i],
                   edgecolors='white', linewidths=1.5, zorder=3)
        ax.annotate(f'{LABELS[m]}\n(+{latency[i]:.0f} ms, {cpu[i]:.0f} MB CPU)',
                    (latency[i], gain[i]), xytext=(8, 8),
                    textcoords='offset points', fontsize=8.5, fontweight='bold',
                    color=CLUSTER_COLORS[i])

    ax.set_xlabel('ADPV latency increase (ms)', fontsize=10)
    ax.set_ylabel('Instance F1 gain (percentage points)', fontsize=10)
    ax.set_title('(d) Accuracy gain vs. cost', fontsize=12, fontweight='bold')
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(latency) * 1.5)
    ax.set_ylim(0, max(gain) * 1.3)

    fig.suptitle('ADPV / GIDM Efficiency & Cost Analysis',
                 fontsize=15, fontweight='bold')
    fig.savefig(OUTPUT, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f'已保存: {OUTPUT}')


if __name__ == '__main__':
    main()
