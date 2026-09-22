#!/usr/bin/env python3
"""将综合效率图拆分为 4 张独立图片 (保持各子图原有布局与样式)。

输出:
  output/fig_a_latency.png        端到端耗时构成
  output/fig_b_memory.png         内存占用 (GPU / ADPV-Framework CPU)
  output/fig_c_scalability.png    场景级可扩展性
  output/fig_d_accuracy_gain.png  ADPV 成本 vs 精度增益
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'output')
PIPE = json.load(open(os.path.join(OUT, 'efficiency_full_pipeline.json'), encoding='utf-8'))
GIDM = json.load(open(os.path.join(OUT, 'efficiency_analysis.json'), encoding='utf-8'))

BACKBONES = ['randla', 'pointgroup', 'pointnext']
B_LABELS = {'randla': 'RandLA-Net', 'pointgroup': 'PointGroup', 'pointnext': 'PointNeXt-L'}
B_COLORS = {'randla': '#2b6cb0', 'pointgroup': '#168a5b', 'pointnext': '#7650a8'}
METHODS = ['watershed_3d', 'meanshift', 'hdbscan']
M_LABELS = {'watershed_3d': 'Watershed 3D', 'meanshift': 'MeanShift', 'hdbscan': 'HDBSCAN'}
M_COLORS = {'watershed_3d': '#c53b3b', 'meanshift': '#168a5b', 'hdbscan': '#7650a8'}
F1 = {'watershed_3d': (0.5303, 0.8251), 'meanshift': (0.856, 0.888), 'hdbscan': (0.114, 0.780)}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.color': '#e5e7eb',
    'grid.linewidth': 0.7,
    'grid.alpha': 1.0,
})


def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=180, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print('已保存:', path)


def fig_a_latency(pipe_summary):
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor='white')
    x = np.arange(len(BACKBONES))
    sem = np.array([pipe_summary['backbones'][b]['sem_ms'] for b in BACKBONES])
    clu = np.array([pipe_summary['backbones'][b]['cluster_ms'] for b in BACKBONES])
    gidm = np.array([pipe_summary['backbones'][b]['gidm_ms'] for b in BACKBONES])

    width = 0.26
    ax.bar(x - width, sem, width, color='#2b6cb0', label='Semantic inference')
    ax.bar(x, clu, width, color='#d49a24', label='Clustering')
    ax.bar(x + width, gidm, width, color='#c53b3b', label='ADPV-Framework')

    # 每根柱顶标注毫秒值
    for i in range(len(BACKBONES)):
        for xoff, val in [(-width, sem[i]), (0, clu[i]), (width, gidm[i])]:
            ax.text(x[i] + xoff, val + max(sem) * 0.02, f'{val:.0f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x, [B_LABELS[b] for b in BACKBONES])
    ax.set_ylabel('Runtime (ms)')
    ax.legend(frameon=False, fontsize=8.5, ncol=1, loc='upper left')
    ax.grid(axis='y')
    ax.grid(axis='x', visible=False)
    ax.set_ylim(0, max(sem) * 1.15)
    _save(fig, 'fig_a_latency.png')


def fig_b_memory(pipe_summary):
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor='white')
    mem_x = np.arange(len(BACKBONES))
    gpu = np.array([pipe_summary['backbones'][b]['gpu_mb'] for b in BACKBONES])
    cpu = np.array([pipe_summary['backbones'][b]['cpu_mb'] for b in BACKBONES])
    width = 0.34
    ax.bar(mem_x - width / 2, gpu, width, color=[B_COLORS[b] for b in BACKBONES],
           alpha=0.88, label='GPU peak')
    ax2 = ax.twinx()
    ax2.bar(mem_x + width / 2, cpu, width, color='#9ca3af', alpha=0.95, label='ADPV-Framework CPU peak')
    ax.set_xticks(mem_x, [B_LABELS[b] for b in BACKBONES])
    ax.set_ylabel('GPU memory (MB)', color='#374151')
    ax2.set_ylabel('CPU memory (MB)', color='#6b7280')
    ax2.spines['right'].set_visible(True)
    ax.set_ylim(0, max(gpu) * 1.24)
    ax2.set_ylim(0, max(cpu) * 1.65)
    ax.grid(axis='y')
    ax2.grid(False)
    for i, value in enumerate(gpu):
        ax.text(i - width / 2, value + max(gpu) * 0.025, f'{value:.0f}',
                ha='center', fontsize=9)
    for i, value in enumerate(cpu):
        ax2.text(i + width / 2, value + max(cpu) * 0.05, f'{value:.1f}',
                 ha='center', fontsize=9, color='#4b5563')
    # legend for both axes
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, ncol=1,
              loc='upper right')
    _save(fig, 'fig_b_memory.png')


def fig_c_scalability(pipe_rows):
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor='white')
    line_styles = {'randla': '-', 'pointgroup': '--', 'pointnext': '-.'}
    for b in BACKBONES:
        xs = np.array([r['n_points'] for r in pipe_rows]) / 1000.0
        ys = np.array([r[f'{b}_sem_ms'] + r[f'{b}_cluster_ms'] + r[f'{b}_gidm_ms'] for r in pipe_rows])
        order = np.argsort(xs)
        ax.plot(xs[order], ys[order], marker='o', markersize=3.8, linewidth=1.7,
                linestyle=line_styles[b], color=B_COLORS[b], label=B_LABELS[b], alpha=0.9)
    ax.set_xlabel('Non-ground points per scene (K)')
    ax.set_ylabel('Full pipeline runtime (ms)')
    ax.grid(True)
    ax.set_yscale('log')
    ax.text(0.98, 0.04, 'log scale', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9, color='#6b7280')
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    _save(fig, 'fig_c_scalability.png')


def fig_d_accuracy_gain(gidm_summary):
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor='white')
    for m in METHODS:
        if m not in gidm_summary['methods']:
            continue
        cost = gidm_summary['methods'][m]['gidm_increase_ms']
        gain = (F1[m][1] - F1[m][0]) * 100
        ax.scatter(cost, gain, s=160, color=M_COLORS[m], edgecolors='white',
                   linewidths=1.5, zorder=3)
        ax.annotate(f'{M_LABELS[m]}  (+{gain:.1f} pp)', (cost, gain),
                    xytext=(8, 0), textcoords='offset points', va='center',
                    fontsize=10, color=M_COLORS[m], fontweight='bold')
    ax.set_xlabel('ADPV latency increment (ms)')
    ax.set_ylabel('Instance F1 gain (percentage points)')
    ax.grid(True)
    ax.set_xlim(0, max(gidm_summary['methods'][m]['gidm_increase_ms'] for m in METHODS) * 1.5)
    ax.set_ylim(0, max((F1[m][1] - F1[m][0]) * 100 for m in METHODS) * 1.25)
    ax.text(0.03, 0.95, 'GPU increment = 0 MB', transform=ax.transAxes,
            va='top', fontsize=10, color='#4b5563')
    _save(fig, 'fig_d_accuracy_gain.png')


def main():
    pipe_summary = PIPE['summary']
    pipe_rows = PIPE['per_file']
    gidm_summary = GIDM['summary']

    fig_a_latency(pipe_summary)
    fig_b_memory(pipe_summary)
    fig_c_scalability(pipe_rows)
    fig_d_accuracy_gain(gidm_summary)


if __name__ == '__main__':
    main()
