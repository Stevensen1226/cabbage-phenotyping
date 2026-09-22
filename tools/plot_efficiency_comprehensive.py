#!/usr/bin/env python3
"""ADPV 效率与精度综合图 (基于全部统一实测数据)。

数据源:
  - output/efficiency_full_pipeline.json  (3 backbone × 语义/聚类/GIDM 耗时 + 显存)
  - output/efficiency_analysis.json       (3 聚类方法 × GIDM 耗时 + F1 增益)

图布局 (2×2):
  (a) 各 backbone 完整 pipeline 耗时构成堆叠柱 (语义 + 聚类 + GIDM)
  (b) 各 backbone 的 GPU 显存柱状图
  (c) GIDM 后处理耗时 vs F1 增益散点 (3 聚类方法, 彩色)
  (d) 各 backbone 的 语义耗时 vs 参数量 散点 (气泡=显存)

用法:
  $HOME/miniconda3/envs/bgpseg/bin/python tools/plot_efficiency_comprehensive.py
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'output')
PIPE_INPUT = os.path.join(OUT, 'efficiency_full_pipeline.json')
GIDM_INPUT = os.path.join(OUT, 'efficiency_analysis.json')
OUTPUT = os.path.join(OUT, 'efficiency_comprehensive.png')

BACKBONE_LABELS = {
    'randla': 'RandLA-Net',
    'pointgroup': 'PointGroup',
    'pointnext': 'PointNeXt-L',
}
BACKBONE_COLORS = {
    'randla': '#2b6cb0',
    'pointgroup': '#2f855a',
    'pointnext': '#805ad5',
}
CLUSTER_LABELS = {
    'watershed_3d': 'Watershed 3D',
    'meanshift': 'MeanShift',
    'hdbscan': 'HDBSCAN',
}
CLUSTER_COLORS = ['#c53030', '#2f855a', '#805ad5']
F1_GAIN = {
    'watershed_3d': (0.5303, 0.8251),
    'meanshift': (0.856, 0.888),
    'hdbscan': (0.114, 0.780),
}
GRID = '#e2e8f0'


def main():
    pipe = json.load(open(PIPE_INPUT, encoding='utf-8'))['summary']
    gidm = json.load(open(GIDM_INPUT, encoding='utf-8'))['summary']

    order = ['randla', 'pointgroup', 'pointnext']
    blabels = [BACKBONE_LABELS[b] for b in order]
    bcolors = [BACKBONE_COLORS[b] for b in order]
    x = np.arange(len(order))

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    fig.subplots_adjust(hspace=0.36, wspace=0.30)

    # ============ (a) 完整 pipeline 耗时构成堆叠柱 ============
    ax = axes[0, 0]
    sem_ms = [pipe['backbones'][b]['sem_ms'] for b in order]
    clu_ms = [pipe['backbones'][b]['cluster_ms'] for b in order]
    gidm_ms = [pipe['backbones'][b]['gidm_ms'] for b in order]

    ax.bar(x, sem_ms, width=0.55, color='#2b6cb0', label='Semantic backbone')
    ax.bar(x, clu_ms, bottom=sem_ms, width=0.55, color='#d69e2e', label='Clustering')
    ax.bar(x, gidm_ms, bottom=[s + c for s, c in zip(sem_ms, clu_ms)],
           width=0.55, color='#c53030', label='ADPV (GIDM)')

    for i in range(len(order)):
        total = sem_ms[i] + clu_ms[i] + gidm_ms[i]
        ax.text(i, total + max(sem_ms) * 0.02, f'+{gidm_ms[i]:.0f} ms',
                ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                color='#c53030')

    ax.set_xticks(x, blabels, fontsize=9)
    ax.set_ylabel('Runtime (ms)', fontsize=10)
    ax.set_title('(a) Pipeline latency (semantic + clustering + ADPV)', fontsize=11.5, fontweight='bold')
    ax.grid(axis='y', color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    ax.set_ylim(0, (max(sem_ms) + max(clu_ms) + max(gidm_ms)) * 1.1)

    # ============ (b) GPU 显存 ============
    ax = axes[0, 1]
    gpu_mb = [pipe['backbones'][b]['gpu_mb'] for b in order]
    ax.bar(x, gpu_mb, width=0.55, color=bcolors, edgecolor='white', linewidth=1)
    for i, v in enumerate(gpu_mb):
        ax.text(i, v + max(gpu_mb) * 0.03, f'{v:.0f} MB', ha='center',
                va='bottom', fontsize=9, fontweight='bold', color=bcolors[i])
    ax.set_xticks(x, blabels, fontsize=9)
    ax.set_ylabel('Peak GPU memory (MB)', fontsize=10)
    ax.set_title('(b) GPU memory', fontsize=11.5, fontweight='bold')
    ax.grid(axis='y', color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(gpu_mb) * 1.18)

    # ============ (c) GIDM 耗时 vs F1 增益 ============
    ax = axes[1, 0]
    methods = list(gidm['methods'].keys())
    latency = [gidm['methods'][m]['gidm_increase_ms'] for m in methods]
    gain = [(F1_GAIN[m][1] - F1_GAIN[m][0]) * 100 for m in methods]
    cpu = [gidm['methods'][m]['cpu_mb'] for m in methods]

    for i, m in enumerate(methods):
        ax.scatter(latency[i], gain[i], s=180, color=CLUSTER_COLORS[i],
                   edgecolors='white', linewidths=1.5, zorder=3)
        ax.annotate(f'{CLUSTER_LABELS[m]}\n(+{latency[i]:.0f} ms, {cpu[i]:.0f} MB CPU)',
                    (latency[i], gain[i]), xytext=(10, 10),
                    textcoords='offset points', fontsize=8.5, fontweight='bold',
                    color=CLUSTER_COLORS[i])

    ax.set_xlabel('ADPV (GIDM) latency increase (ms)', fontsize=10)
    ax.set_ylabel('Instance F1 gain (percentage points)', fontsize=10)
    ax.set_title('(c) ADPV cost vs. accuracy gain', fontsize=11.5, fontweight='bold')
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(latency) * 1.5)
    ax.set_ylim(0, max(gain) * 1.25)
    ax.text(0.5, 0.05, 'GPU memory increment: 0 MB', transform=ax.transAxes,
            fontsize=8.5, color='#718096', ha='center')

    # ============ (d) 语义耗时 vs 参数量 (气泡=显存) ============
    ax = axes[1, 1]
    params = [pipe['n_params_M'][b] for b in order]
    size = np.array(gpu_mb) / max(gpu_mb) * 700 + 100
    for i, b in enumerate(order):
        ax.scatter(params[i], sem_ms[i], s=size[i], color=bcolors[i],
                   alpha=0.8, edgecolors='white', linewidths=1.5, zorder=3)
        ax.annotate(blabels[i], (params[i], sem_ms[i]), xytext=(8, 8),
                    textcoords='offset points', fontsize=9, fontweight='bold',
                    color=bcolors[i])
    ax.set_xlabel('Parameters (M)', fontsize=10)
    ax.set_ylabel('Semantic inference time (ms)', fontsize=10)
    ax.set_title('(d) Semantic latency vs. parameters', fontsize=11.5, fontweight='bold')
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.text(0.5, 0.06, 'bubble size = GPU memory', transform=ax.transAxes,
            fontsize=8.5, color='#718096', ha='center')

    fig.suptitle('ADPV: Efficiency & Cost Analysis (RTX 5070, watershed_3d)',
                 fontsize=14.5, fontweight='bold')
    fig.savefig(OUTPUT, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f'已保存: {OUTPUT}')


if __name__ == '__main__':
    main()
