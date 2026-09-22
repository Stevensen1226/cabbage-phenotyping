#!/usr/bin/env python3
"""
对比实验论文图表自动生成脚本
用法: python generate_all_charts.py
输出: tools/figures/ 目录下的 PNG 图片和 LaTeX 表格
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ========== 全局设置 ==========
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'font.family': 'sans-serif',
})

OUT_DIR = Path(__file__).parent
DATA_PATH = OUT_DIR / 'experiment_data.csv'

# 配色方案
COLORS = {
    'RandLA-Net': '#3498db',
    'PointGroup-TD': '#e74c3c',
    'PointGroup-NMS': '#2ecc71',
    'PointGroup-IACH': '#9b59b6',
    'No': '#bdc3c7',
    'Yes': '#e67e22',
    'meanshift': '#1abc9c',
    'hdbscan': '#3498db',
    'watershed': '#e74c3c',
}

BACKBONE_ORDER = ['RandLA-Net', 'PointGroup-TD', 'PointGroup-NMS', 'PointGroup-IACH']
CLUSTER_ORDER = ['None', 'meanshift', 'hdbscan', 'watershed']


def load_data():
    df = pd.read_csv(DATA_PATH)
    df['SARR_SS'] = pd.Categorical(df['SARR_SS'], categories=['No', 'Yes'], ordered=True)
    df['Backbone'] = pd.Categorical(df['Backbone'], categories=BACKBONE_ORDER, ordered=True)
    return df


# ============================================================
# Figure 1: SARR-SS 消融柱状图 (2×2 子图, 每子图对比F1和MAE)
# ============================================================
def plot_sarr_ablation(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    backbones = BACKBONE_ORDER
    metrics_info = [
        ('Instance_F1', 'Instance F1-score ↑'),
        ('MAE_Count', 'MAE (Count) ↓'),
    ]

    for i, backbone in enumerate(backbones):
        ax = axes[i // 2][i % 2]
        sub = df[df['Backbone'] == backbone].copy()
        sub = sub[sub['Clustering'] != 'None']  # 排除无聚类baseline

        x_labels = []
        f1_no, f1_yes = [], []
        mae_no, mae_yes = [], []

        for clust in ['meanshift', 'hdbscan', 'watershed']:
            row_no = sub[(sub['Clustering'] == clust) & (sub['SARR_SS'] == 'No')]
            row_yes = sub[(sub['Clustering'] == clust) & (sub['SARR_SS'] == 'Yes')]
            if len(row_no) and len(row_yes):
                x_labels.append(clust)
                f1_no.append(row_no['Instance_F1'].values[0])
                f1_yes.append(row_yes['Instance_F1'].values[0])
                mae_no.append(row_no['MAE_Count'].values[0])
                mae_yes.append(row_yes['MAE_Count'].values[0])

        x = np.arange(len(x_labels))
        w = 0.2

        # F1 bars
        bars1 = ax.bar(x - w, f1_no, w * 0.9, color=COLORS['No'], edgecolor='white', label='w/o SARR-SS')
        bars2 = ax.bar(x, f1_yes, w * 0.9, color=COLORS['Yes'], edgecolor='white', label='w/ SARR-SS (Ours)')

        # 标注数值
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold',
                    color=COLORS['Yes'])

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_ylabel('Instance F1-score', fontsize=11)
        ax.set_title(f'{backbone}', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Figure 1: Ablation Study of SARR-SS on Instance F1-score',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'fig1_sarr_ablation_f1.png', dpi=300)
    plt.close()
    print('[OK] fig1_sarr_ablation_f1.png')


# Figure 1b: MAE 版本
def plot_sarr_ablation_mae(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    backbones = BACKBONE_ORDER

    for i, backbone in enumerate(backbones):
        ax = axes[i // 2][i % 2]
        sub = df[df['Backbone'] == backbone]
        sub = sub[sub['Clustering'] != 'None']

        x_labels = []
        mae_no, mae_yes = [], []
        for clust in ['meanshift', 'hdbscan', 'watershed']:
            row_no = sub[(sub['Clustering'] == clust) & (sub['SARR_SS'] == 'No')]
            row_yes = sub[(sub['Clustering'] == clust) & (sub['SARR_SS'] == 'Yes')]
            if len(row_no) and len(row_yes):
                x_labels.append(clust)
                mae_no.append(row_no['MAE_Count'].values[0])
                mae_yes.append(row_yes['MAE_Count'].values[0])

        x = np.arange(len(x_labels))
        w = 0.2
        bars1 = ax.bar(x - w, mae_no, w * 0.9, color=COLORS['No'], edgecolor='white', label='w/o SARR-SS')
        bars2 = ax.bar(x, mae_yes, w * 0.9, color=COLORS['Yes'], edgecolor='white', label='w/ SARR-SS (Ours)')

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=7)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold',
                    color=COLORS['Yes'])

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_ylabel('MAE (Count Error)', fontsize=11)
        ax.set_title(f'{backbone}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Figure 1b: Ablation Study of SARR-SS on Count MAE',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'fig1b_sarr_ablation_mae.png', dpi=300)
    plt.close()
    print('[OK] fig1b_sarr_ablation_mae.png')


# ============================================================
# Figure 2: 全方法 × 指标热力图
# ============================================================
def plot_full_heatmap(df):
    metrics = ['Instance_Precision', 'Instance_Recall', 'Instance_F1',
               'Instance_mIoU', 'Semantic_mIoU', 'MAE_Count']
    metric_labels = ['Inst. Prec↑', 'Inst. Rec↑', 'Inst. F1↑',
                     'Inst. mIoU↑', 'Sem. mIoU↑', 'MAE Count↓']

    # 构建 pivot
    pivot_data = df.pivot_table(index='Method', values=metrics)

    # 对 MAE 做反转（越小越好 → 越大越好，用于颜色统一）
    heat_data = pivot_data[metrics].copy()
    heat_data['MAE_Count'] = 1.0 / (heat_data['MAE_Count'] + 0.01)  # 倒数映射

    # 按 F1 降序排列
    heat_data = heat_data.sort_values('Instance_F1', ascending=False)

    fig, ax = plt.subplots(figsize=(12, 14))
    sns.heatmap(heat_data, annot=pivot_data.loc[heat_data.index].round(3).values,
                fmt='', cmap='RdYlGn', center=0.5,
                linewidths=0.5, linecolor='white',
                xticklabels=metric_labels,
                cbar_kws={'label': 'Score (normalized)', 'shrink': 0.6},
                ax=ax)
    ax.set_title('Figure 2: Comprehensive Comparison of All Methods', fontsize=14, fontweight='bold')
    ax.set_ylabel('')
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'fig2_full_heatmap.png', dpi=300)
    plt.close()
    print('[OK] fig2_full_heatmap.png')


# ============================================================
# Figure 3: F1 排名横向条形图
# ============================================================
def plot_f1_ranking(df):
    ranked = df.sort_values('Instance_F1', ascending=True)

    colors = [COLORS[b] for b in ranked['Backbone']]
    edge_colors = ['#333' if s == 'Yes' else '#aaa' for s in ranked['SARR_SS']]
    linewidths = [2 if s == 'Yes' else 0.5 for s in ranked['SARR_SS']]

    fig, ax = plt.subplots(figsize=(10, 9))
    bars = ax.barh(range(len(ranked)), ranked['Instance_F1'], color=colors,
                   edgecolor=edge_colors, linewidth=linewidths)

    # 标注数值
    for i, (_, row) in enumerate(ranked.iterrows()):
        marker = ' ★' if row['SARR_SS'] == 'Yes' else ''
        ax.text(row['Instance_F1'] + 0.005, i,
                f"{row['Instance_F1']:.3f}{marker}", va='center', fontsize=7.5)

    ax.set_yticks(range(len(ranked)))
    ax.set_yticklabels(ranked['Method'], fontsize=7.5)
    ax.set_xlabel('Instance F1-score', fontsize=12)
    ax.set_title('Figure 3: Instance F1-score Ranking (★ = with SARR-SS)', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.grid(axis='x', alpha=0.3)

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS[b], label=b) for b in BACKBONE_ORDER]
    legend_elements += [Patch(facecolor='white', edgecolor='#333', linewidth=2, label='w/ SARR-SS'),
                        Patch(facecolor='white', edgecolor='#aaa', linewidth=0.5, label='w/o SARR-SS')]
    ax.legend(handles=legend_elements, fontsize=8, loc='lower right', ncol=2)

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'fig3_f1_ranking.png', dpi=300)
    plt.close()
    print('[OK] fig3_f1_ranking.png')


# ============================================================
# Figure 4: Precision-Recall 散点图
# ============================================================
def plot_pr_scatter(df):
    fig, ax = plt.subplots(figsize=(10, 8))

    for backbone in BACKBONE_ORDER:
        sub = df[df['Backbone'] == backbone]
        # w/o SARR-SS
        sub_no = sub[sub['SARR_SS'] == 'No']
        ax.scatter(sub_no['Instance_Recall'], sub_no['Instance_Precision'],
                   c=COLORS[backbone], marker='o', s=60, alpha=0.7,
                   edgecolors='white', linewidth=0.5, label=f'{backbone} (w/o)')
        # w/ SARR-SS
        sub_yes = sub[sub['SARR_SS'] == 'Yes']
        ax.scatter(sub_yes['Instance_Recall'], sub_yes['Instance_Precision'],
                   c=COLORS[backbone], marker='D', s=100, alpha=0.9,
                   edgecolors='black', linewidth=1.2, label=f'{backbone} (w/ SARR-SS)')

    # F1 等值线
    f1_levels = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95]
    for f1 in f1_levels:
        r = np.linspace(0.01, 1, 100)
        p = f1 * r / (2 * r - f1)
        valid = (p >= 0) & (p <= 1)
        ax.plot(r[valid], p[valid], 'grey', linestyle='--', linewidth=0.5, alpha=0.5)
        # 标注 F1 值
        idx = np.argmin(np.abs(r - 0.85))
        if idx < len(r) and valid[idx]:
            ax.annotate(f'F1={f1}', xy=(r[idx], p[idx]), fontsize=6, color='grey', alpha=0.7)

    ax.set_xlabel('Instance Recall', fontsize=12)
    ax.set_ylabel('Instance Precision', fontsize=12)
    ax.set_title('Figure 4: Precision-Recall Scatter Plot', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, loc='lower left', ncol=2)
    ax.axline((0, 0), (1, 1), color='black', linewidth=0.3, linestyle=':')

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'fig4_pr_scatter.png', dpi=300)
    plt.close()
    print('[OK] fig4_pr_scatter.png')


# ============================================================
# Figure 5: 聚类方法雷达图 (以 PointGroup-TD 为例)
# ============================================================
def plot_radar(df):
    backbone = 'PointGroup-TD'
    sub = df[df['Backbone'] == backbone]

    # 归一化函数
    metrics_radar = ['Instance_Precision', 'Instance_Recall', 'Instance_F1',
                     'Instance_mIoU', 'Semantic_mIoU']
    metric_labels_radar = ['Inst. Prec', 'Inst. Recall', 'Inst. F1',
                           'Inst. mIoU', 'Sem. mIoU']

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw=dict(polar=True))

    for ax_idx, sarr_status in enumerate(['No', 'Yes']):
        ax = axes[ax_idx]
        sarr_label = 'w/o SARR-SS' if sarr_status == 'No' else 'w/ SARR-SS (Ours)'

        angles = np.linspace(0, 2 * np.pi, len(metrics_radar), endpoint=False).tolist()
        angles += angles[:1]

        for clust in ['meanshift', 'hdbscan', 'watershed']:
            row = sub[(sub['Clustering'] == clust) & (sub['SARR_SS'] == sarr_status)]
            if len(row) == 0:
                continue
            values = row[metrics_radar].values[0].tolist()
            values += values[:1]
            ax.fill(angles, values, alpha=0.1, color=COLORS[clust])
            ax.plot(angles, values, 'o-', linewidth=2, color=COLORS[clust], label=clust, markersize=4)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels_radar, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(f'{sarr_label}', fontsize=12, fontweight='bold', pad=20)
        ax.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.3, 1.0))

    fig.suptitle(f'Figure 5: Clustering Method Comparison ({backbone})',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'fig5_radar_clustering.png', dpi=300)
    plt.close()
    print('[OK] fig5_radar_clustering.png')


# ============================================================
# Figure 6: Semantic mIoU 一致性
# ============================================================
def plot_semantic_consistency(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 箱线图：按 Backbone
    ax = axes[0]
    bp = df.boxplot(column='Semantic_mIoU', by='Backbone', ax=ax, patch_artist=True,
                    boxprops=dict(facecolor='#ecf0f1'),
                    medianprops=dict(color='#e74c3c', linewidth=2))
    for i, backbone in enumerate(BACKBONE_ORDER):
        bp.findobj(match=plt.Line2D)  # no-op, just triggering
    ax.set_title('Semantic mIoU by Backbone', fontsize=11)
    ax.set_ylabel('Semantic mIoU')
    ax.set_xlabel('')
    fig.suptitle('')  # remove auto-title from boxplot

    # 散点图：全部27个点
    ax = axes[1]
    for i, backbone in enumerate(BACKBONE_ORDER):
        sub = df[df['Backbone'] == backbone]
        jitter = np.random.uniform(-0.15, 0.15, len(sub))
        ax.scatter([i] * len(sub) + jitter, sub['Semantic_mIoU'],
                   c=COLORS[backbone], s=30, alpha=0.7, label=backbone)
    ax.set_xticks(range(len(BACKBONE_ORDER)))
    ax.set_xticklabels(BACKBONE_ORDER, fontsize=9)
    ax.set_ylabel('Semantic mIoU')
    ax.set_title('Per-Method Scatter', fontsize=11)
    ax.set_ylim(0.81, 0.825)
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Figure 6: Semantic mIoU Consistency (σ < 0.002 across 27 methods)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'fig6_semantic_consistency.png', dpi=300)
    plt.close()
    print('[OK] fig6_semantic_consistency.png')


# ============================================================
# Figure 7: 主对比表精选柱状图 (Representative methods comparison)
# ============================================================
def plot_representative_comparison(df):
    """精选代表性方法做主对比图"""
    selected = [
        'RandLA-Net + meanshift',
        'RandLA-Net + meanshift + SARR-SS',
        'PointGroup (Top-Down)',
        'PointGroup (Top-Down) + hdbscan',
        'PointGroup (Top-Down) + hdbscan + SARR-SS',
    ]
    sub = df[df['Method'].isin(selected)].set_index('Method').loc[selected]

    metrics_plot = ['Instance_Precision', 'Instance_Recall', 'Instance_F1',
                    'Instance_mIoU', 'Semantic_mIoU']
    metric_labels = ['Inst. Prec', 'Inst. Recall', 'Inst. F1', 'Inst. mIoU', 'Sem. mIoU']

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(selected))
    w = 0.15
    colors_bar = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    for j, (met, label) in enumerate(zip(metrics_plot, metric_labels)):
        vals = sub[met].values
        bars = ax.bar(x + j * w - w * 2, vals, w, label=label, color=colors_bar[j], edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=6.5, rotation=45)

    ax.set_xticks(x)
    short_names = ['RL+MS', 'RL+MS+SARR', 'PG(TD)', 'PG(TD)+HDB', 'PG(TD)+HDB+SARR']
    ax.set_xticklabels(short_names, fontsize=8, rotation=15)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.15)
    ax.set_title('Figure 7: Representative Methods Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right', ncol=3)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'fig7_representative_comparison.png', dpi=300)
    plt.close()
    print('[OK] fig7_representative_comparison.png')


# ============================================================
# 导出 LaTeX 表格
# ============================================================
def export_latex_table(df):
    """导出精选对比表"""
    selected = [
        'RandLA-Net + meanshift',
        'RandLA-Net + meanshift + SARR-SS',
        'RandLA-Net + hdbscan',
        'RandLA-Net + hdbscan + SARR-SS',
        'RandLA-Net + watershed',
        'RandLA-Net + watershed + SARR-SS',
        'PointGroup (Top-Down) + meanshift',
        'PointGroup (Top-Down) + meanshift + SARR-SS',
        'PointGroup (Top-Down) + hdbscan',
        'PointGroup (Top-Down) + hdbscan + SARR-SS',
        'PointGroup (Top-Down) + watershed',
        'PointGroup (Top-Down) + watershed + SARR-SS',
    ]
    sub = df[df['Method'].isin(selected)].set_index('Method').loc[selected]

    cols = ['Instance_Precision', 'Instance_Recall', 'Instance_F1',
            'Instance_mIoU', 'Semantic_mIoU', 'MAE_Count']
    col_map = {
        'Instance_Precision': 'Inst. Prec.',
        'Instance_Recall': 'Inst. Recall',
        'Instance_F1': 'Inst. F1',
        'Instance_mIoU': 'Inst. mIoU',
        'Semantic_mIoU': 'Sem. mIoU',
        'MAE_Count': 'MAE$\\downarrow$',
    }
    table = sub[cols].rename(columns=col_map)

    latex = table.to_latex(float_format="%.3f", column_format='l' + 'c' * len(cols),
                           caption='Comparison of different instance segmentation methods.',
                           label='tab:comparison',
                           escape=False)

    with open(OUT_DIR / 'table_comparison.tex', 'w') as f:
        f.write(latex)
    print('[OK] table_comparison.tex')

    # 同时导出完整CSV供Excel打开
    df.to_csv(OUT_DIR / 'all_results_formatted.csv', index=False, float_format='%.3f')
    print('[OK] all_results_formatted.csv')


# ============================================================
# Main
# ============================================================
def main():
    print('=' * 60)
    print('  甘蓝点云实例分割 — 对比实验图表生成')
    print('=' * 60)

    df = load_data()
    print(f'Loaded {len(df)} methods, {len(df.columns)} columns')

    plot_sarr_ablation(df)
    plot_sarr_ablation_mae(df)
    plot_full_heatmap(df)
    plot_f1_ranking(df)
    plot_pr_scatter(df)
    plot_radar(df)
    plot_semantic_consistency(df)
    plot_representative_comparison(df)
    export_latex_table(df)

    print('=' * 60)
    print(f'所有图表已生成至: {OUT_DIR}')
    print('文件列表:')
    for f in sorted(OUT_DIR.glob('*.png')):
        print(f'  {f.name}')
    for f in sorted(OUT_DIR.glob('*.tex')):
        print(f'  {f.name}')
    for f in sorted(OUT_DIR.glob('*.csv')):
        print(f'  {f.name}')


if __name__ == '__main__':
    main()
