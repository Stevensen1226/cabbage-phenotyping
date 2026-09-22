#!/usr/bin/env python3
"""多 backbone 效率对比图 (RandLA / PointGroup / PointNeXt)。

三张子图:
  (a) 推理耗时柱状图 (ms)
  (b) GPU 峰值显存柱状图 (MB)
  (c) 耗时 vs 参数量 散点 (气泡=显存)

用法:
  $HOME/miniconda3/envs/bgpseg/bin/python tools/plot_backbones.py
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'output')
INPUT = os.path.join(OUT, 'efficiency_backbones.json')
OUTPUT = os.path.join(OUT, 'efficiency_backbones.png')

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
GRID = '#e2e8f0'


def main():
    data = json.load(open(INPUT, encoding='utf-8'))
    summary = data['summary']
    order = ['randla', 'pointgroup', 'pointnext']
    labels = [BACKBONE_LABELS[b] for b in order]
    colors = [BACKBONE_COLORS[b] for b in order]
    x = np.arange(len(order))

    ms = [summary['backbones'][b]['ms'] for b in order]
    gpu = [summary['backbones'][b]['gpu_mb'] for b in order]
    params = [summary['n_params_M'][b] for b in order]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), gridspec_kw={'wspace': 0.30})

    # (a) 推理耗时
    ax = axes[0]
    bars = ax.bar(x, ms, width=0.55, color=colors, edgecolor='white', linewidth=1)
    for i, v in enumerate(ms):
        ax.text(i, v + max(ms) * 0.03, f'{v:.0f} ms', ha='center',
                va='bottom', fontsize=9, fontweight='bold', color=colors[i])
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel('Inference time (ms)', fontsize=10)
    ax.set_title('(a) Latency', fontsize=12, fontweight='bold')
    ax.grid(axis='y', color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(ms) * 1.18)

    # (b) GPU 显存
    ax = axes[1]
    ax.bar(x, gpu, width=0.55, color=colors, edgecolor='white', linewidth=1)
    for i, v in enumerate(gpu):
        ax.text(i, v + max(gpu) * 0.03, f'{v:.0f} MB', ha='center',
                va='bottom', fontsize=9, fontweight='bold', color=colors[i])
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel('Peak GPU memory (MB)', fontsize=10)
    ax.set_title('(b) GPU memory', fontsize=12, fontweight='bold')
    ax.grid(axis='y', color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(gpu) * 1.18)

    # (c) 耗时 vs 参数量 (气泡=显存)
    ax = axes[2]
    size = np.array(gpu) / max(gpu) * 700 + 100
    for i, b in enumerate(order):
        ax.scatter(params[i], ms[i], s=size[i], color=colors[i],
                   alpha=0.8, edgecolors='white', linewidths=1.5, zorder=3)
        ax.annotate(labels[i], (params[i], ms[i]), xytext=(8, 8),
                    textcoords='offset points', fontsize=9, fontweight='bold',
                    color=colors[i])
    ax.set_xlabel('Parameters (M)', fontsize=10)
    ax.set_ylabel('Inference time (ms)', fontsize=10)
    ax.set_title('(c) Latency vs. parameters', fontsize=12, fontweight='bold')
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.text(0.5, 0.06, 'bubble size = GPU memory', transform=ax.transAxes,
            fontsize=8.5, color='#718096', ha='center')

    fig.suptitle('Backbone Efficiency Comparison (RTX 5070)',
                 fontsize=14, fontweight='bold')
    fig.savefig(OUTPUT, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f'已保存: {OUTPUT}')


if __name__ == '__main__':
    main()
