"""表型参数精度验证散点拟合图: 计算值(X) vs 实测值(Y).

6 个子图对应 6 项甘蓝表型参数, 每图标注:
  - 拟合直线 y = ax + b (理想 a=1, b=0)
  - R^2 (决定系数, 越接近 1 越好)
  - RMSE (均方根误差, 越小越好)

用法:
  python tools/plot_trait_scatter.py --csv output/traits_per_plant_test.csv \
      --out output/trait_scatter.png
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

# 中文字体 (与 oversegmentation_analysis.py 一致)
_CJK_FONT_PATHS = [
    "/mnt/i/WeGameApps/rail_apps/无畏契约(2001715)/ACLOS/DiagnoseTool/fonts/NotoSansSC-Regular.otf",
    "/mnt/i/WeGameApps/rail_apps/无畏契约(2001715)/ACLOS/DiagnoseTool/fonts/NotoSansSC-Bold.otf",
]


def setup_font():
    import matplotlib
    from matplotlib import font_manager
    for p in _CJK_FONT_PATHS:
        if os.path.exists(p):
            try:
                font_manager.fontManager.addfont(p)
                name = font_manager.FontProperties(fname=p).get_name()
                matplotlib.rcParams["font.family"] = name
                matplotlib.rcParams["axes.unicode_minus"] = False
                return
            except Exception:
                continue


# 4 项核心表型参数: (字段key, 英文标题, 中文标题, 单位)
# 单位为空字符串时, 显示为 dimensionless
TRAITS = [
    ('height_cm', 'Plant Height', '株高', 'cm'),
    ('crown_diameter_cm', 'Canopy Diameter', '冠幅', 'cm'),
    ('volume_alpha_cm3', 'Plant Volume', '植株体积', 'cm$^3$'),
    ('compactness', 'Compactness', '紧实度', ''),
]

# 需做系统偏差校正的冠幅字段
CROWN_KEYS = {'crown_diameter_cm'}


def fit_stats(x, y):
    """线性拟合 y = a*x + b, 返回 (a, b, r2, rmse)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    a, b = np.polyfit(x, y, 1)
    y_pred_line = a * x + b
    # R^2
    ss_res = np.sum((y - y_pred_line) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    # RMSE: 计算值相对实测值的均方根误差
    rmse = float(np.sqrt(np.mean((x - y) ** 2)))
    return a, b, r2, rmse


def draw_ax(ax, x, y, unit, en, idx, corrected=False):
    """在单个 ax 上绘制散点拟合图, 返回 (a, b, r2, rmse)."""
    a, b, r2, rmse = fit_stats(x, y)

    # 散点
    ax.scatter(x, y, s=14, c='#2b6cb0', alpha=0.55, edgecolors='none',
               rasterized=True, label='Data')

    # 范围与对角线 y=x
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    pad = (hi - lo) * 0.06
    lo -= pad
    hi += pad
    ax.plot([lo, hi], [lo, hi], '--', color='0.5', linewidth=1.0,
            label='y = x', zorder=1)

    # 拟合直线
    xs = np.linspace(lo, hi, 100)
    ax.plot(xs, a * xs + b, '-', color='#c53030', linewidth=1.6,
            label='Fitted line', zorder=2)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.25, linewidth=0.5)

    # 单位后缀
    unit_str = f' ({unit})' if unit else ' (-)'
    ax.set_xlabel(f'Calculated value{unit_str}', fontsize=9)
    ax.set_ylabel(f'Reference value (GT){unit_str}', fontsize=9)

    # 图内文字框: 拟合方程 + R2 + RMSE
    b_sign = '+' if b >= 0 else '-'
    eq = f'y = {a:.4f}x {b_sign} {abs(b):.3f}'
    txt = f'{eq}\n$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}'
    ax.text(0.05, 0.97, txt, transform=ax.transAxes,
            fontsize=9, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                      edgecolor='0.75', alpha=0.9))

    # 标题: 英文标题 (不含 (a)(b) 字母标签)
    title = en
    if corrected:
        title += ' (bias-corrected)'
    ax.set_title(title, fontsize=11, fontweight='bold')

    return a, b, r2, rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--out', default='output/trait_scatter.png')
    parser.add_argument('--min-iou', type=float, default=0.0,
                        help='按实例 IoU 过滤离群点 (默认 0 = 不过滤; 0.7 = 仅高质量分割)')
    parser.add_argument('--correct-bias', action='store_true',
                        help='对冠幅字段做系统偏差校正 (减去 mean(pred-gt))')
    parser.add_argument('--subplot-dir', default=None,
                        help='拆分保存每个子图的文件夹 (默认不拆分; 建议 output/trait_subplots)')
    args = parser.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    setup_font()

    df = pd.read_csv(args.csv)
    if args.min_iou > 0 and 'iou' in df.columns:
        df = df[df['iou'] >= args.min_iou]
    n = len(df)
    print(f'样本数: {n} (min_iou={args.min_iou})')

    # 系统偏差校正: 对冠幅字段减去均值偏差
    biases = {}
    if args.correct_bias:
        for key in CROWN_KEYS:
            x = df[f'pred_{key}'].to_numpy(dtype=float)
            y = df[f'gt_{key}'].to_numpy(dtype=float)
            valid = ~(np.isnan(x) | np.isnan(y))
            bias = float(np.mean(x[valid] - y[valid]))
            biases[key] = bias
            print(f'偏差校正 {key}: bias = {bias:+.2f} (pred 平均 {x[valid].mean():.2f} vs gt 平均 {y[valid].mean():.2f})')

    fig, axes = plt.subplots(2, 2, figsize=(10, 8.6))
    axes = axes.ravel()

    # 预处理每个指标的数据 (提取 x/y + 偏差校正)
    data = {}
    for key, en, zh, unit in TRAITS:
        x = df[f'pred_{key}'].to_numpy(dtype=float)
        y = df[f'gt_{key}'].to_numpy(dtype=float)
        valid = ~(np.isnan(x) | np.isnan(y))
        x = x[valid]
        y = y[valid]
        corrected = False
        if args.correct_bias and key in CROWN_KEYS and key in biases:
            x = x - biases[key]
            corrected = True
        data[key] = (x, y, en, unit, corrected)

    for idx, (key, (x, y, en, unit, corrected)) in enumerate(data.items()):
        draw_ax(axes[idx], x, y, unit, en, idx, corrected)

    # 去掉多余子图 (如果有)
    for ax in axes[len(TRAITS):]:
        ax.axis('off')

    fig.suptitle('Trait Estimation Accuracy: Calculated vs. Measured',
                 fontsize=14, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'已保存: {args.out}')

    # 拆分保存每个子图到指定文件夹
    if args.subplot_dir:
        os.makedirs(args.subplot_dir, exist_ok=True)
        prefix = os.path.splitext(os.path.basename(args.out))[0]
        for idx, (key, (x, y, en, unit, corrected)) in enumerate(data.items()):
            fig2, ax2 = plt.subplots(figsize=(5.2, 4.5))
            draw_ax(ax2, x, y, unit, en, idx, corrected)
            fig2.tight_layout()
            out_sub = os.path.join(args.subplot_dir, f'{prefix}_{key}.png')
            fig2.savefig(out_sub, dpi=200, bbox_inches='tight')
            plt.close(fig2)
            print(f'已保存子图: {out_sub}')

    # 同时打印统计摘要
    print('\n=== 各表型参数拟合统计 ===')
    print(f'{"表型":<18} {"a":>8} {"b":>9} {"R2":>8} {"RMSE":>10} {"n":>5}')
    for key, en, zh, unit in TRAITS:
        x = df[f'pred_{key}'].to_numpy(dtype=float)
        y = df[f'gt_{key}'].to_numpy(dtype=float)
        valid = ~(np.isnan(x) | np.isnan(y))
        x = x[valid]
        y = y[valid]
        if args.correct_bias and key in CROWN_KEYS and key in biases:
            x = x - biases[key]
        a, b, r2, rmse = fit_stats(x, y)
        print(f'{en:<18} {a:>8.4f} {b:>9.3f} {r2:>8.3f} {rmse:>10.3f} {int(len(x)):>5}')


if __name__ == '__main__':
    main()
