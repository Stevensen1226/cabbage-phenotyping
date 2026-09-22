#!/usr/bin/env python3
"""Publication-style efficiency figure based on the measured JSON files."""

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
GRID = '#e5e7eb'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.color': '#e5e7eb',
    'grid.linewidth': 0.7,
    'grid.alpha': 1.0,
})


def main():
    pipe_summary = PIPE['summary']
    pipe_rows = PIPE['per_file']
    gidm_summary = GIDM['summary']

    fig = plt.figure(figsize=(13.4, 9.0), facecolor='white')
    grid = fig.add_gridspec(2, 2, hspace=0.44, wspace=0.32)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    # (a) Full pipeline latency: one clean horizontal stacked bar per backbone.
    y = np.arange(len(BACKBONES))
    sem = np.array([pipe_summary['backbones'][b]['sem_ms'] for b in BACKBONES])
    clu = np.array([pipe_summary['backbones'][b]['cluster_ms'] for b in BACKBONES])
    gidm = np.array([pipe_summary['backbones'][b]['gidm_ms'] for b in BACKBONES])
    ax_a.barh(y, sem, color='#2b6cb0', height=0.48, label='Semantic inference')
    ax_a.barh(y, clu, left=sem, color='#d49a24', height=0.48, label='Clustering')
    ax_a.barh(y, gidm, left=sem + clu, color='#c53b3b', height=0.48, label='ADPV / GIDM')
    totals = sem + clu + gidm
    for i, value in enumerate(totals):
        ax_a.text(value + max(totals) * 0.018, i, f'{value:.0f} ms', va='center', fontsize=9, fontweight='bold')
    ax_a.set_yticks(y, [B_LABELS[b] for b in BACKBONES])
    ax_a.invert_yaxis()
    ax_a.set_xlabel('Runtime per scene (ms)')
    ax_a.set_title('(a) End-to-end latency', loc='left', fontweight='bold')
    ax_a.legend(frameon=False, fontsize=8.5, ncol=3, loc='upper left', bbox_to_anchor=(0, 1.02))
    ax_a.grid(axis='x')
    ax_a.grid(axis='y', visible=False)
    ax_a.set_xlim(0, max(totals) * 1.22)

    # (b) Memory footprint: separate axes so PointNeXt/RandLA do not flatten CPU memory.
    mem_x = np.arange(len(BACKBONES))
    gpu = np.array([pipe_summary['backbones'][b]['gpu_mb'] for b in BACKBONES])
    cpu = np.array([pipe_summary['backbones'][b]['cpu_mb'] for b in BACKBONES])
    width = 0.34
    ax_b.bar(mem_x - width / 2, gpu, width, color=[B_COLORS[b] for b in BACKBONES], alpha=0.88, label='GPU peak')
    ax_b2 = ax_b.twinx()
    ax_b2.bar(mem_x + width / 2, cpu, width, color='#9ca3af', alpha=0.95, label='GIDM CPU peak')
    ax_b.set_xticks(mem_x, [B_LABELS[b] for b in BACKBONES])
    ax_b.set_ylabel('GPU memory (MB)', color='#374151')
    ax_b2.set_ylabel('GIDM CPU memory (MB)', color='#6b7280')
    ax_b.set_title('(b) Memory footprint', loc='left', fontweight='bold')
    ax_b.set_ylim(0, max(gpu) * 1.24)
    ax_b2.set_ylim(0, max(cpu) * 1.65)
    ax_b.grid(axis='y')
    ax_b2.grid(False)
    for i, value in enumerate(gpu):
        ax_b.text(i - width / 2, value + max(gpu) * 0.025, f'{value:.0f}', ha='center', fontsize=8.5)
    for i, value in enumerate(cpu):
        ax_b2.text(i + width / 2, value + max(cpu) * 0.05, f'{value:.1f}', ha='center', fontsize=8.5, color='#4b5563')

    # (c) Per-scene scalability: clean log-scale comparison.
    ax_c.clear()
    line_styles = {'randla': '-', 'pointgroup': '--', 'pointnext': '-.'}
    endpoint_labels = []
    for b in BACKBONES:
        xs = np.array([r['n_points'] for r in pipe_rows]) / 1000.0
        ys = np.array([r[f'{b}_sem_ms'] + r[f'{b}_cluster_ms'] + r[f'{b}_gidm_ms'] for r in pipe_rows])
        order = np.argsort(xs)
        ax_c.plot(xs[order], ys[order], marker='o', markersize=3.8, linewidth=1.7,
                  linestyle=line_styles[b], color=B_COLORS[b], label=B_LABELS[b], alpha=0.9)
        endpoint_labels.append((xs[order][-1], ys[order][-1], b))
    ax_c.set_xlabel('Non-ground points per scene (K)')
    ax_c.set_ylabel('Full pipeline runtime (ms)')
    ax_c.set_title('(c) Scene-level scalability', loc='left', fontweight='bold')
    ax_c.grid(True)
    ax_c.set_yscale('log')
    ax_c.text(0.98, 0.04, 'log scale', transform=ax_c.transAxes,
              ha='right', va='bottom', fontsize=8, color='#6b7280')
    for px, py, b in endpoint_labels:
        ax_c.annotate(B_LABELS[b], (px, py), xytext=(7, 0),
                      textcoords='offset points', va='center', fontsize=8.5,
                      fontweight='bold', color=B_COLORS[b],
                      bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                edgecolor='none', alpha=0.85))

    # (d) ADPV cost versus measured F1 gain: fewer labels, more readable.
    ax_d.clear()
    for m in METHODS:
        if m not in gidm_summary['methods']:
            continue
        cost = gidm_summary['methods'][m]['gidm_increase_ms']
        gain = (F1[m][1] - F1[m][0]) * 100
        cpu = gidm_summary['methods'][m]['cpu_mb']
        ax_d.scatter(cost, gain, s=140, color=M_COLORS[m], edgecolors='white', linewidths=1.5, zorder=3)
        ax_d.annotate(M_LABELS[m], (cost, gain), xytext=(8, 7), textcoords='offset points',
                      fontsize=9, color=M_COLORS[m], fontweight='bold')
        ax_d.annotate(f'+{gain:.1f} pp', (cost, gain), xytext=(8, -16), textcoords='offset points',
                      fontsize=8, color='#4b5563')
    ax_d.set_xlabel('ADPV latency increment (ms)')
    ax_d.set_ylabel('Instance F1 gain (percentage points)')
    ax_d.set_title('(d) Accuracy gain per post-processing cost', loc='left', fontweight='bold')
    ax_d.grid(True)
    ax_d.text(0.03, 0.95, 'GPU increment = 0 MB', transform=ax_d.transAxes,
              va='top', fontsize=9, color='#4b5563')

    fig.suptitle('ADPV / GIDM: Efficiency, Memory, and Accuracy Trade-offs',
                 fontsize=15, fontweight='bold', y=0.98)
    # Do not use bbox_inches='tight': inset axes can produce an enormous
    # calculated bounding box and an unreadable PNG.
    fig.savefig(os.path.join(OUT, 'efficiency_comprehensive.png'), dpi=180,
                facecolor='white')
    plt.close(fig)
    print('已保存:', os.path.join(OUT, 'efficiency_comprehensive.png'))


if __name__ == '__main__':
    main()
