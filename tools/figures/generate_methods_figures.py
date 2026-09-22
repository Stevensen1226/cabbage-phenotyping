#!/usr/bin/env python3
"""
论文方法部分 (Methods) 图表自动生成脚本
用法: python generate_methods_figures.py
输出: tools/figures/ 目录下的 PNG 图片
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, Polygon, Rectangle, Ellipse
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'font.family': 'sans-serif',
})

OUT_DIR = Path(__file__).parent

# ============================================================
# 配色方案
# ============================================================
C = {
    'bg': '#FAFAFA',
    'input': '#3498db',       # 蓝色 - 输入
    'preprocess': '#1abc9c',  # 青色 - 预处理
    'semantic': '#9b59b6',    # 紫色 - 语义分割
    'cluster': '#e67e22',     # 橙色 - 聚类
    'sarr': '#e74c3c',        # 红色 - SARR-SS (核心)
    'trait': '#2ecc71',       # 绿色 - 表型
    'output': '#34495e',      # 深灰 - 输出
    'arrow': '#7f8c8d',
    'box_bg': 'white',
    'text': '#2c3e50',
    'highlight': '#FFF3CD',
    'stage1': '#FFEAA7',
    'stage2': '#FDCB6E',
    'stage3': '#E17055',
    'stage4': '#D63031',
}


def draw_rounded_box(ax, x, y, width, height, color, text='', text_color='white',
                     fontsize=9, fontweight='bold', edgecolor=None, linewidth=1.5, alpha=1.0):
    """绘制圆角矩形框"""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.08", facecolor=color,
                         edgecolor=edgecolor or color, linewidth=linewidth, alpha=alpha)
    ax.add_patch(box)
    if text:
        ax.text(x + width / 2, y + height / 2, text,
                ha='center', va='center', color=text_color, fontsize=fontsize,
                fontweight=fontweight)


def draw_arrow(ax, start, end, color='#7f8c8d', lw=1.5, style='simple'):
    """绘制箭头"""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle='arc3,rad=0'))


def draw_dashed_box(ax, x, y, width, height, color, label='', fontsize=8):
    """绘制虚线框（分组）"""
    rect = Rectangle((x, y), width, height, fill=False, edgecolor=color,
                     linestyle='--', linewidth=1.2, alpha=0.7)
    ax.add_patch(rect)
    if label:
        ax.text(x + width / 2, y + height + 0.03, label,
                ha='center', va='bottom', fontsize=fontsize, color=color,
                fontweight='bold', fontstyle='italic')


# ============================================================
# Figure M1: 整体 Pipeline 流程图
# ============================================================
def plot_pipeline_overview():
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_facecolor(C['bg'])

    # --- 第0行: 标题 ---
    ax.text(8, 7.6, 'Overall Pipeline of Cabbage Phenotyping from 3D Point Cloud',
            ha='center', fontsize=15, fontweight='bold', color=C['text'])

    # === 第一行: 输入 & 预处理 ===
    y1 = 6.0
    draw_dashed_box(ax, 0.15, 4.8, 3.7, 2.2, '#bdc3c7', 'Input & Preprocessing')

    draw_rounded_box(ax, 0.3, y1 + 0.8, 1.5, 0.8, C['input'], 'Raw Point\nCloud (.ply)', fontsize=8)
    draw_arrow(ax, (1.8, y1 + 1.2), (2.3, y1 + 1.2))
    draw_rounded_box(ax, 2.3, y1 + 0.8, 1.3, 0.8, C['preprocess'], 'SOR\nDenoising', fontsize=8)
    draw_arrow(ax, (3.6, y1 + 1.2), (4.05, y1 + 1.2))
    draw_arrow(ax, (4.05, y1 + 1.6), (4.05, y1 + 0.1))
    draw_rounded_box(ax, 2.8, y1 - 0.3, 2.5, 0.7, C['preprocess'], 'RANSAC Ground Removal', fontsize=8)

    # === 第二行: 语义分割 ===
    y2 = 3.5
    draw_dashed_box(ax, 0.15, 3.2, 3.7, 2.2, '#bdc3c7', 'Semantic Segmentation')
    draw_arrow(ax, (4.05, y2 + 2.1), (4.05, y2 + 1.2))

    draw_rounded_box(ax, 0.8, y2 + 0.8, 1.8, 0.8, C['semantic'], 'PointGroup\n(Sem + Offset)', fontsize=8)
    ax.text(1.7, y2 + 0.45, 'U-Net backbone\n→ semantic logits\n→ offset vectors',
            ha='center', fontsize=6.5, color=C['text'], style='italic')

    # === 第三行: 实例提案 ===
    y3 = 1.8
    draw_rounded_box(ax, 1.0, y3 + 0.1, 1.5, 0.7, C['cluster'], 'Instance\nProposals', fontsize=8)
    draw_arrow(ax, (1.75, y2 + 0.8), (1.75, y3 + 0.8))

    # === SARR-SS 核心区域 (右侧大框) ===
    draw_dashed_box(ax, 4.5, 1.75, 7.0, 4.8, C['sarr'], 'SARR-SS Refinement (Ours)')

    # Stage 1
    sx, sy = 4.8, 5.5
    draw_rounded_box(ax, sx, sy, 2.8, 0.9, C['stage1'], '', alpha=0.15, edgecolor=C['stage1'])
    ax.text(sx + 1.4, sy + 0.7, 'Stage 1: Watershed-3D', fontsize=9, fontweight='bold', ha='center', color='#8B6914')
    ax.text(sx + 1.4, sy + 0.35, 'Voxelize → EDT → Peaks → Watershed', fontsize=6.5, ha='center', color=C['text'])

    # Stage 2
    draw_rounded_box(ax, sx, sy - 1.1, 2.8, 0.9, C['stage2'], '', alpha=0.15, edgecolor=C['stage2'])
    ax.text(sx + 1.4, sy - 0.4, 'Stage 2: Pre-cluster Filter', fontsize=9, fontweight='bold', ha='center', color='#8B6914')
    ax.text(sx + 1.4, sy - 0.75, 'Outlier removal + Density check', fontsize=6.5, ha='center', color=C['text'])

    # Stage 3
    draw_rounded_box(ax, sx + 3.2, sy, 2.8, 0.9, C['stage3'], '', alpha=0.15, edgecolor=C['stage3'])
    ax.text(sx + 4.6, sy + 0.7, 'Stage 3: Skeleton Split', fontsize=9, fontweight='bold', ha='center', color='#8B0000')
    ax.text(sx + 4.6, sy + 0.35, 'PCA → Project → Density peaks\n→ Valley cut (recursive)', fontsize=6.5, ha='center', color=C['text'])

    # Stage 4
    draw_rounded_box(ax, sx + 3.2, sy - 1.1, 2.8, 0.9, C['stage4'], '', alpha=0.15, edgecolor=C['stage4'])
    ax.text(sx + 4.6, sy - 0.4, 'Stage 4: Fragment Voting', fontsize=9, fontweight='bold', ha='center', color='#8B0000')
    ax.text(sx + 4.6, sy - 0.75, 'Discard <500pts, Merge <1500pts', fontsize=6.5, ha='center', color=C['text'])

    # 箭头连接
    draw_arrow(ax, (2.5, y3 + 0.45), (sx + 0.5, sy + 0.45), C['sarr'], lw=2)
    draw_arrow(ax, (sx + 1.4, sy), (sx + 1.4, sy - 0.2), C['arrow'], lw=1.2)
    draw_arrow(ax, (sx + 2.8, sy + 0.45), (sx + 3.2, sy + 0.45), C['arrow'], lw=1.2)
    draw_arrow(ax, (sx + 4.6, sy), (sx + 4.6, sy - 0.2), C['arrow'], lw=1.2)

    # === 第五行: 表型 ===
    y4 = 0.5
    draw_rounded_box(ax, 8.5, y4 - 0.3, 2.0, 0.7, C['trait'], 'Trait Calculation\n(Height, Width, Volume...)', fontsize=7.5)
    draw_arrow(ax, (sx + 2.8, sy - 1.1 + 0.45), (8.5, y4 + 0.05), C['arrow'], lw=1.5)
    draw_arrow(ax, (sx + 6.0, sy - 1.1 + 0.45), (8.5, y4 + 0.05), C['arrow'], lw=1.5)

    # 输出
    draw_rounded_box(ax, 11.0, y4 - 0.3, 1.5, 0.7, C['output'], 'plants.csv\nplants.json', fontsize=8)
    draw_arrow(ax, (10.5, y4 + 0.05), (11.0, y4 + 0.05))

    # === 标注关键数据流 ===
    # 语义标签
    ax.annotate('semantic\nlabels', xy=(2.5, 3.8), fontsize=6, color=C['semantic'],
                ha='center', style='italic')
    # 偏移向量
    ax.annotate('offset\nvectors', xy=(3.0, 3.8), fontsize=6, color=C['semantic'],
                ha='center', style='italic')

    # 右侧说明
    ax.text(12.5, 5.8, 'Key Innovation:', fontsize=9, fontweight='bold', color=C['sarr'])
    ax.text(12.5, 5.3, '• Watershed-3D for\n  coarse clustering', fontsize=7.5, color=C['text'])
    ax.text(12.5, 4.7, '• PCA skeleton split\n  for touching plants', fontsize=7.5, color=C['text'])
    ax.text(12.5, 4.1, '• Fragment voting\n  to eliminate noise', fontsize=7.5, color=C['text'])
    ax.text(12.5, 3.5, '• Density-aware\n  quality control', fontsize=7.5, color=C['text'])

    fig.savefig(OUT_DIR / 'method_fig1_pipeline_overview.png', dpi=300)
    plt.close()
    print('[OK] method_fig1_pipeline_overview.png')


# ============================================================
# Figure M2: Watershed-3D 聚类示意图
# ============================================================
def plot_watershed_illustration():
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.8))
    fig.suptitle('Stage 1: Watershed-3D Clustering Pipeline', fontsize=13, fontweight='bold', y=1.02)

    steps = [
        ('(a) Input Points', 'point_cloud'),
        ('(b) Voxelization', 'voxel'),
        ('(c) Distance Transform', 'edt'),
        ('(d) Peak Detection', 'peaks'),
        ('(e) Watershed Result', 'result'),
    ]

    np.random.seed(42)
    for idx, (title, step) in enumerate(steps):
        ax = axes[idx]
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')

        if step == 'point_cloud':
            # 两组点模拟两株植物
            n = 200
            # 植物1
            r1 = np.random.randn(n) * 0.15 + 0.5
            theta1 = np.random.rand(n) * 2 * np.pi
            x1 = r1 * np.cos(theta1) - 0.6
            y1 = r1 * np.sin(theta1)
            # 植物2
            r2 = np.random.randn(n) * 0.15 + 0.5
            theta2 = np.random.rand(n) * 2 * np.pi
            x2 = r2 * np.cos(theta2) + 0.6
            y2 = r2 * np.sin(theta2)

            ax.scatter(x1, y1, s=2, c='#3498db', alpha=0.6)
            ax.scatter(x2, y2, s=2, c='#e74c3c', alpha=0.6)
            ax.text(-0.6, -1.3, 'Plant A', ha='center', fontsize=8, color='#3498db')
            ax.text(0.6, -1.3, 'Plant B', ha='center', fontsize=8, color='#e74c3c')

        elif step == 'voxel':
            # 体素网格
            grid = np.zeros((30, 30))
            # 两团
            for i in range(30):
                for j in range(30):
                    cx, cy = (i - 15) / 10, (j - 15) / 10
                    dist_a = np.sqrt((cx + 0.6)**2 + cy**2)
                    dist_b = np.sqrt((cx - 0.6)**2 + cy**2)
                    if dist_a < 0.45 or dist_b < 0.45:
                        grid[i, j] = 1
            ax.imshow(grid.T, origin='lower', cmap='Blues', extent=[-1.5, 1.5, -1.5, 1.5], alpha=0.8)
            # 网格线
            for i in range(-15, 16, 5):
                ax.axvline(i / 10, color='white', linewidth=0.3, alpha=0.5)
                ax.axhline(i / 10, color='white', linewidth=0.3, alpha=0.5)

        elif step == 'edt':
            # 距离变换（用热力图画）
            xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 60), np.linspace(-1.5, 1.5, 60))
            dist_a = np.sqrt((xx + 0.6)**2 + yy**2)
            dist_b = np.sqrt((xx - 0.6)**2 + yy**2)
            edt_val = np.minimum(dist_a, dist_b)
            edt_val = np.clip(0.5 - edt_val, 0, 0.5) / 0.5  # 反转：内部值高
            # 只在"植物内部"显示
            mask = (dist_a < 0.45) | (dist_b < 0.45)
            edt_val[~mask] = 0
            ax.imshow(edt_val, origin='lower', cmap='hot', extent=[-1.5, 1.5, -1.5, 1.5], alpha=0.9)
            ax.text(-0.6, -1.3, 'bright = far from\nboundary', ha='center', fontsize=6.5, style='italic')

        elif step == 'peaks':
            # EDT + 峰值标记
            xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 60), np.linspace(-1.5, 1.5, 60))
            dist_a = np.sqrt((xx + 0.6)**2 + yy**2)
            dist_b = np.sqrt((xx - 0.6)**2 + yy**2)
            edt_val = np.minimum(dist_a, dist_b)
            edt_val = np.clip(0.5 - edt_val, 0, 0.5) / 0.5
            mask = (dist_a < 0.45) | (dist_b < 0.45)
            edt_val[~mask] = 0
            ax.imshow(edt_val, origin='lower', cmap='hot', extent=[-1.5, 1.5, -1.5, 1.5], alpha=0.9)
            # 标记峰值
            ax.plot(-0.6, 0, 'D', color='cyan', markersize=10, markeredgecolor='white', markeredgewidth=1.5)
            ax.plot(0.6, 0, 'D', color='cyan', markersize=10, markeredgecolor='white', markeredgewidth=1.5)
            ax.annotate('Peak A', (-0.6, 0.2), fontsize=7, color='cyan', ha='center', fontweight='bold')
            ax.annotate('Peak B', (0.6, 0.2), fontsize=7, color='cyan', ha='center', fontweight='bold')

        elif step == 'result':
            # 分水岭结果
            xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 100), np.linspace(-1.5, 1.5, 100))
            dist_a = np.sqrt((xx + 0.6)**2 + yy**2)
            dist_b = np.sqrt((xx - 0.6)**2 + yy**2)
            mask = (dist_a < 0.45) | (dist_b < 0.45)
            # 分割线在 x=0
            label = np.zeros_like(xx)
            label[(dist_a < 0.45) & (xx < 0)] = 1
            label[(dist_b < 0.45) & (xx > 0)] = 2
            # 中间区域分配给更近的
            between = mask & (label == 0)
            label[between] = np.where(xx[between] < 0, 1, 2)

            cmap = plt.cm.colors.ListedColormap(['white', '#3498db', '#e74c3c'])
            ax.imshow(label, origin='lower', cmap=cmap, extent=[-1.5, 1.5, -1.5, 1.5], alpha=0.8)
            # 分割边界
            ax.axvline(0, color='black', linewidth=2, linestyle='--')
            ax.text(-0.6, -1.3, 'Instance 1', ha='center', fontsize=8, color='#3498db', fontweight='bold')
            ax.text(0.6, -1.3, 'Instance 2', ha='center', fontsize=8, color='#e74c3c', fontweight='bold')

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'method_fig2_watershed_pipeline.png', dpi=300)
    plt.close()
    print('[OK] method_fig2_watershed_pipeline.png')


# ============================================================
# Figure M3: PCA Skeleton Split 示意图
# ============================================================
def plot_skeleton_split():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Stage 3: PCA Skeleton-Based Adaptive Split', fontsize=15, fontweight='bold', y=0.98)

    np.random.seed(123)

    # 生成两个粘连的椭圆
    n = 300
    x1 = np.random.randn(n) * 0.5 - 1.2
    y1 = np.random.randn(n) * 0.3 + 0.0
    x2 = np.random.randn(n) * 0.5 + 0.5
    y2 = np.random.randn(n) * 0.3 + 0.5
    x3 = np.random.uniform(-0.6, 0.1, 80)
    y3 = np.random.normal(0.25, 0.15, 80)
    x_all = np.concatenate([x1, x2, x3])
    y_all = np.concatenate([y1, y2, y3])

    titles = [
        '(a) Input Cluster (two touching plants)',
        '(b) PCA & Skeleton Projection',
        '(c) Density Profile & Valley Detection',
        '(d) Split Result',
    ]
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for (row, col), title in zip(positions, titles):
        ax = axes[row][col]
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

        if title.startswith('(a)'):
            ax.scatter(x_all, y_all, s=6, c='#e74c3c', alpha=0.6, edgecolors='none')
            ax.set_xlim(-3, 3)
            ax.set_ylim(-2, 2)
            ax.set_aspect('equal')
            ellipse = Ellipse((-0.3, 0.25), 3.5, 1.2, angle=10,
                              fill=False, edgecolor='grey', linestyle='--', linewidth=1.2)
            ax.add_patch(ellipse)
            ax.set_xlabel('X (m)', fontsize=10)
            ax.set_ylabel('Y (m)', fontsize=10)

        elif title.startswith('(b)'):
            pts = np.column_stack([x_all, y_all])
            mean = pts.mean(axis=0)
            centered = pts - mean
            cov = np.cov(centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            v1 = eigvecs[:, -1]

            ax.scatter(x_all, y_all, s=6, c='#e74c3c', alpha=0.5, edgecolors='none')
            ax.arrow(mean[0], mean[1], v1[0] * 2, v1[1] * 2,
                     head_width=0.15, head_length=0.15, fc='#2c3e50', ec='#2c3e50', linewidth=2.5)
            ax.arrow(mean[0], mean[1], -v1[0] * 2, -v1[1] * 2,
                     head_width=0.15, head_length=0.15, fc='#2c3e50', ec='#2c3e50', linewidth=2.5)
            # 垂直偏移 label 避免和箭头/散点重叠
            label_x = mean[0] + v1[0] * 2.6
            label_y = mean[1] + v1[1] * 2.6 + 0.25
            ax.text(label_x, label_y, r'$\mathbf{v}_1$ (PCA axis)',
                    fontsize=11, color='#2c3e50', fontweight='bold')
            for pt in pts[::15]:
                proj = np.dot(pt - mean, v1)
                proj_pt = mean + proj * v1
                ax.plot([pt[0], proj_pt[0]], [pt[1], proj_pt[1]],
                        color='grey', alpha=0.25, linewidth=0.4)
            ax.set_xlim(-3, 3)
            ax.set_ylim(-2, 2)
            ax.set_aspect('equal')
            ax.set_xlabel('X (m)', fontsize=10)
            ax.set_ylabel('Y (m)', fontsize=10)

        elif title.startswith('(c)'):
            pts = np.column_stack([x_all, y_all])
            mean = pts.mean(axis=0)
            v1 = np.linalg.eigh(np.cov((pts - mean).T))[1][:, -1]
            projections = np.array([np.dot(pt - mean, v1) for pt in pts])

            # 不设置 aspect='equal'，让直方图自然填充
            counts, bins, patches = ax.hist(projections, bins=15, color='#e74c3c',
                                              edgecolor='white', alpha=0.65, linewidth=1)

            from scipy.ndimage import gaussian_filter1d
            from scipy.signal import find_peaks
            bin_centers = (bins[:-1] + bins[1:]) / 2
            smoothed = gaussian_filter1d(counts.astype(float), sigma=0.8)
            ax.plot(bin_centers, smoothed, 'b-', linewidth=3, label='Smoothed density', zorder=5)

            peaks, props = find_peaks(smoothed, distance=3)
            ymax = max(counts.max(), smoothed.max()) * 1.35
            ax.set_ylim(0, ymax)

            for p in peaks:
                ax.plot(bin_centers[p], smoothed[p], 'D', color='#2ecc71', markersize=14,
                         markeredgecolor='white', markeredgewidth=2, zorder=6)
                # Peak 标注在图表顶部，远离直方图
                ax.annotate('Peak', (bin_centers[p], smoothed[p]),
                             xytext=(bin_centers[p], ymax * 0.97),
                             fontsize=10, ha='center', color='#2ecc71', fontweight='bold',
                             arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=1.5))

            if len(peaks) >= 2:
                valley_idx = np.argmin(smoothed[peaks[0]:peaks[1]]) + peaks[0]
                ax.axvline(bin_centers[valley_idx], color='#c0392b', linewidth=2.5,
                            linestyle='--', zorder=4)
                # Valley 标注放在右中区域，水平和垂直都远离 Peak，clip_on=False 防止被裁切
                vx = bin_centers[valley_idx]
                ax.annotate('Valley → Cut Plane',
                             xy=(vx, smoothed[valley_idx]),
                             xytext=(bins[-1] * 0.78, ymax * 0.65),
                             fontsize=10, ha='center', color='#c0392b', fontweight='bold',
                             arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5,
                                             connectionstyle='arc3,rad=-0.35'),
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3CD',
                                       edgecolor='#c0392b', alpha=0.85),
                             clip_on=False)

            ax.set_xlabel('Projection along $v_1$ (m)', fontsize=10)
            ax.set_ylabel('Point Density', fontsize=10)
            ax.legend(fontsize=9, loc='upper left')

        elif title.startswith('(d)'):
            split_line_x = -0.25
            ax.scatter(x_all[x_all < split_line_x], y_all[x_all < split_line_x],
                       s=8, c='#3498db', alpha=0.7, edgecolors='none', label='Plant 1')
            ax.scatter(x_all[x_all >= split_line_x], y_all[x_all >= split_line_x],
                       s=8, c='#e74c3c', alpha=0.7, edgecolors='none', label='Plant 2')
            ax.axvline(split_line_x, color='black', linewidth=3, linestyle='--', label='Cut plane')
            ax.arrow(split_line_x, 1.5, 0.0, -0.5,
                     head_width=0.12, head_length=0.12, fc='black', ec='black', linewidth=2.5)
            ax.text(split_line_x + 0.2, 1.6, r'Cut normal $\mathbf{n}$', fontsize=11, color='black')

            ax.set_xlim(-3, 3)
            ax.set_ylim(-2, 2)
            ax.set_aspect('equal')
            ax.set_xlabel('X (m)', fontsize=10)
            ax.set_ylabel('Y (m)', fontsize=10)
            ax.legend(fontsize=10, loc='upper left')

    plt.tight_layout(pad=2.0)
    fig.savefig(OUT_DIR / 'method_fig3_skeleton_split.png', dpi=300)
    plt.close()
    print('[OK] method_fig3_skeleton_split.png')


# ============================================================
# Figure M4: Fragment Voting 示意图
# ============================================================
def plot_fragment_voting():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Stage 4: Fragment Neighbor Voting', fontsize=13, fontweight='bold', y=1.02)

    np.random.seed(456)
    n = 200
    # 大簇
    x_big = np.random.randn(n) * 0.4 + 0.0
    y_big = np.random.randn(n) * 0.4 + 0.0
    # 中型碎片
    x_mid = np.random.randn(30) * 0.1 + 1.2
    y_mid = np.random.randn(30) * 0.1 + 0.8
    # 微型碎片
    x_tiny1 = np.random.randn(8) * 0.05 + -0.8
    y_tiny1 = np.random.randn(8) * 0.05 + -1.0
    x_tiny2 = np.random.randn(5) * 0.04 + 1.5
    y_tiny2 = np.random.randn(5) * 0.04 + -0.5

    titles = ['(a) Before Voting', '(b) Voting Process', '(c) After Voting']

    for idx, title in enumerate(titles):
        ax = axes[idx]
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlim(-2, 2.5)
        ax.set_ylim(-2, 2)

        if idx == 0:
            ax.scatter(x_big, y_big, s=5, c='#3498db', alpha=0.7, edgecolors='none', label='Main cluster (≥3000 pts)')
            ax.scatter(x_mid, y_mid, s=12, c='#f39c12', alpha=0.8, edgecolors='black', linewidth=0.5,
                       label='Mid fragment (500-1500 pts)')
            ax.scatter(x_tiny1, y_tiny1, s=18, c='#e74c3c', alpha=0.9, edgecolors='black', linewidth=1,
                       marker='X', label='Tiny fragment (<500 pts)')
            ax.scatter(x_tiny2, y_tiny2, s=18, c='#e74c3c', alpha=0.9, edgecolors='black', linewidth=1, marker='X')
            ax.legend(fontsize=7, loc='lower right')

        elif idx == 1:
            ax.scatter(x_big, y_big, s=5, c='#3498db', alpha=0.7, edgecolors='none')
            ax.scatter(x_mid, y_mid, s=12, c='#f39c12', alpha=0.8, edgecolors='black', linewidth=0.5)
            ax.scatter(x_tiny1, y_tiny1, s=18, c='#e74c3c', alpha=0.9, edgecolors='black', linewidth=1, marker='X')
            ax.scatter(x_tiny2, y_tiny2, s=18, c='#e74c3c', alpha=0.9, edgecolors='black', linewidth=1, marker='X')

            # 质心
            centroid_big = np.array([0, 0])
            centroid_tiny1 = np.array([-0.8, -1.0])

            ax.plot(centroid_big[0], centroid_big[1], 'D', color='#2ecc71', markersize=10,
                    markeredgecolor='white', markeredgewidth=1)
            ax.plot(centroid_tiny1[0], centroid_tiny1[1], 'D', color='#2ecc71', markersize=8,
                    markeredgecolor='white', markeredgewidth=1)

            # 连接线 → merge
            ax.annotate('', xy=centroid_big, xytext=centroid_tiny1,
                        arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=2,
                                        connectionstyle='arc3,rad=0.2'))
            ax.text(-0.4, -0.6, 'Merge\n(d<0.2m)', fontsize=7, color='#2ecc71',
                    ha='center', fontweight='bold')

            # X 标记 → 丢弃
            ax.annotate('', xy=(1.5, -0.1), xytext=(1.5, -0.5),
                        arrowprops=dict(arrowstyle='->', color='#95a5a6', lw=1.5))
            ax.text(2.0, -0.3, 'Discard\n(isolated)', fontsize=7, color='#95a5a6', ha='center')

        elif idx == 2:
            ax.scatter(x_big, y_big, s=5, c='#3498db', alpha=0.7, edgecolors='none', label='Main cluster')
            ax.scatter(x_mid, y_mid, s=12, c='#3498db', alpha=0.5, edgecolors='none', label='Merged fragment')
            ax.scatter(x_tiny1, y_tiny1, s=5, c='#3498db', alpha=0.5, edgecolors='none')
            # tiny2 消失（丢弃）
            ax.text(1.5, -0.5, '✗ Removed', fontsize=7, color='#95a5a6', ha='center')
            ax.legend(fontsize=8, loc='lower right')

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'method_fig4_fragment_voting.png', dpi=300)
    plt.close()
    print('[OK] method_fig4_fragment_voting.png')


# ============================================================
# Figure M5: SARR-SS 算法伪代码 / 结构化流程图
# ============================================================
def plot_sarr_algorithm_flow():
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor(C['bg'])

    ax.text(6, 9.5, 'SARR-SS Algorithm: Structure-Aware Refinement & Re-clustering',
            ha='center', fontsize=14, fontweight='bold', color=C['sarr'])

    # 输入
    y_start = 8.5
    draw_rounded_box(ax, 3.5, y_start, 5, 0.6, '#ecf0f1',
                     'Input: Initial instance labels $L_0$ + Point cloud $P$',
                     text_color=C['text'], fontsize=9, fontweight='normal')

    # === Stage 1 ===
    y = 7.3
    ax.text(0.5, y + 0.7, 'Stage 1', fontsize=11, fontweight='bold', color='#8B6914')
    draw_rounded_box(ax, 2.0, y, 8.5, 1.0, C['stage1'], '', alpha=0.12, edgecolor=C['stage1'])
    draw_rounded_box(ax, 2.3, y + 0.15, 7.9, 0.7, 'white',
                     'Watershed-3D: Voxelize → EDT → Gaussian smooth → Peak detection → Watershed',
                     text_color=C['text'], fontsize=8, fontweight='normal', edgecolor='#ddd')
    draw_arrow(ax, (6, y_start), (6, y + 1.0), C['arrow'])

    # === Stage 2 ===
    y = 5.7
    ax.text(0.5, y + 0.7, 'Stage 2', fontsize=11, fontweight='bold', color='#8B6914')
    draw_rounded_box(ax, 2.0, y, 8.5, 1.0, C['stage2'], '', alpha=0.12, edgecolor=C['stage2'])
    draw_rounded_box(ax, 2.3, y + 0.15, 7.9, 0.7, 'white',
                     'Pre-cluster filter: Radius outlier removal (r=3cm, k=10)\n→ Post-density check (min 500pts/m, min 100pts)',
                     text_color=C['text'], fontsize=8, fontweight='normal', edgecolor='#ddd')
    draw_arrow(ax, (6, y + 1.7), (6, y + 1.0), C['arrow'])

    # === Stage 3 ===
    y = 3.5
    ax.text(0.5, y + 0.7, 'Stage 3', fontsize=11, fontweight='bold', color='#8B0000')
    draw_rounded_box(ax, 2.0, y, 8.5, 1.8, C['stage3'], '', alpha=0.10, edgecolor=C['stage3'])
    draw_rounded_box(ax, 2.3, y + 0.85, 7.9, 0.8, 'white',
                     'PCA Skeleton Split (recursive, depth≤3):\n'
                     '  if diam>0.6m or (aspect>1.6 & diam>0.25m):\n'
                     '    Project→Bin(15)→Smooth→Find peaks→Find valley→Cut plane→Recurse',
                     text_color=C['text'], fontsize=7.5, fontweight='normal', edgecolor='#ddd')
    # 触发条件标注
    ax.annotate('Trigger:\nD>0.6m or\nAR>1.6',
                xy=(10.8, y + 1.2), fontsize=6.5, color=C['stage3'],
                ha='center', style='italic',
                bbox=dict(boxstyle='round', facecolor='#FFF3CD', alpha=0.8))
    draw_arrow(ax, (6, y + 2.7), (6, y + 1.65), C['arrow'])

    # === Stage 4 ===
    y = 1.8
    ax.text(0.5, y + 0.7, 'Stage 4', fontsize=11, fontweight='bold', color='#8B0000')
    draw_rounded_box(ax, 2.0, y, 8.5, 1.2, C['stage4'], '', alpha=0.10, edgecolor=C['stage4'])
    draw_rounded_box(ax, 2.3, y + 0.25, 7.9, 0.8, 'white',
                     'Fragment Neighbor Voting:\n'
                     '  Discard <500pts | Merge 500~1500pts to nearest centroid (d<0.2m)\n'
                     '  → Final renumbering',
                     text_color=C['text'], fontsize=8, fontweight='normal', edgecolor='#ddd')
    draw_arrow(ax, (6, y + 2.2), (6, y + 1.05), C['arrow'])

    # 输出
    y_out = 0.6
    draw_rounded_box(ax, 3.5, y_out, 5, 0.6, '#2ecc71',
                     'Output: Refined instance labels $L_{final}$',
                     text_color='white', fontsize=9, fontweight='bold')
    draw_arrow(ax, (6, y + 0.25), (6, y_out + 0.6), C['arrow'])

    # 右侧注释
    ax.text(11.5, 8.5, 'Key Design\nChoices:', fontsize=8, fontweight='bold', color=C['sarr'])
    ax.text(11.5, 7.8, '• Recursive split\n  (depth≤3)', fontsize=7, color=C['text'])
    ax.text(11.5, 7.2, '• Density peaks\n  not distance', fontsize=7, color=C['text'])
    ax.text(11.5, 6.6, '• Skeleton-aware\n  cut plane', fontsize=7, color=C['text'])
    ax.text(11.5, 6.0, '• Centroid voting\n  not point-level', fontsize=7, color=C['text'])

    fig.savefig(OUT_DIR / 'method_fig5_sarr_algorithm.png', dpi=300)
    plt.close()
    print('[OK] method_fig5_sarr_algorithm.png')


# ============================================================
# Figure M6: PointGroup 语义+偏移双分支示意图
# ============================================================
def plot_pointgroup_dual_branch():
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_facecolor(C['bg'])

    ax.text(6, 6.7, 'PointGroup Backbone: Dual-Branch Prediction',
            ha='center', fontsize=14, fontweight='bold', color=C['semantic'])

    # 输入
    draw_rounded_box(ax, 4.2, 5.8, 3.6, 0.6, C['input'], 'Input: Voxelized Point Cloud', fontsize=9)
    draw_arrow(ax, (6, 5.8), (6, 5.3), C['arrow'])

    # U-Net 骨干
    draw_rounded_box(ax, 3.5, 4.5, 5.0, 0.8, C['semantic'], '3D U-Net Backbone (SparseConv)', fontsize=9, alpha=0.5, edgecolor=C['semantic'])
    draw_arrow(ax, (6, 4.5), (6, 3.8), C['arrow'])

    # 分支
    draw_rounded_box(ax, 1.0, 2.5, 3.5, 1.0, C['semantic'], 'Semantic Branch\n→ Per-point class scores\n→ $L_{sem}$ (cabbage/ground)', fontsize=7.5, alpha=0.3, edgecolor=C['semantic'])
    draw_rounded_box(ax, 7.5, 2.5, 3.5, 1.0, '#8e44ad', r'Offset Branch\n→ Per-point shift vectors\n→ $\Delta_{xyz}$ (to instance center)', fontsize=7.5, alpha=0.3, edgecolor='#8e44ad')

    # 分叉箭头
    ax.annotate('', xy=(2.75, 3.5), xytext=(5.5, 3.8),
                arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=1.5, connectionstyle='arc3,rad=-0.3'))
    ax.annotate('', xy=(9.25, 3.5), xytext=(6.5, 3.8),
                arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=1.5, connectionstyle='arc3,rad=0.3'))

    # 合并
    draw_arrow(ax, (2.75, 2.5), (5.0, 1.8), C['arrow'])
    draw_arrow(ax, (9.25, 2.5), (7.0, 1.8), C['arrow'])
    draw_rounded_box(ax, 2.5, 0.8, 7.0, 1.0, C['cluster'],
                     'Instance Proposal: Shift points by offset → Clustering (NMS/Top-Down/IACH)\n→ Initial instance labels',
                     fontsize=7.5, alpha=0.5, edgecolor=C['cluster'])

    # 连接到SARR-SS
    draw_arrow(ax, (6, 0.8), (6, 0.3), C['sarr'], lw=2)
    draw_rounded_box(ax, 3.5, 0.0, 5.0, 0.5, C['sarr'], '→ SARR-SS Refinement', fontsize=9, fontweight='bold')

    # 损失标注
    ax.text(0.3, 4.9, 'Loss:', fontsize=8, fontweight='bold', color='#7f8c8d')
    ax.text(0.3, 4.5, r'$\mathcal{L}=\mathcal{L}_{sem}+\mathcal{L}_{offset}$', fontsize=9, color='#7f8c8d')
    ax.text(0.3, 4.1, '(cross-entropy + L1 reg)', fontsize=7, color='#7f8c8d')

    fig.savefig(OUT_DIR / 'method_fig6_pointgroup_dual_branch.png', dpi=300)
    plt.close()
    print('[OK] method_fig6_pointgroup_dual_branch.png')


# ============================================================
# Main
# ============================================================
def main():
    print('=' * 60)
    print('  论文方法部分 (Methods) 图表生成')
    print('=' * 60)

    plot_pipeline_overview()
    plot_watershed_illustration()
    plot_skeleton_split()
    plot_fragment_voting()
    plot_sarr_algorithm_flow()
    plot_pointgroup_dual_branch()

    print('=' * 60)
    print(f'所有图表已生成至: {OUT_DIR}')
    pngs = sorted(OUT_DIR.glob('method_fig*.png'))
    for f in pngs:
        size_kb = f.stat().st_size / 1024
        print(f'  {f.name}  ({size_kb:.0f} KB)')
    print(f'共 {len(pngs)} 张图')


if __name__ == '__main__':
    main()
