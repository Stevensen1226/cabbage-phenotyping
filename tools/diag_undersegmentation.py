"""欠分割诊断: 可视化预测实例吞并了哪些真值株。

针对 alpha 体积误差大的匹配对 (如 cloudR3 pred_id=15, cloudR17 pred_id=9),
定位预测实例覆盖的真值株构成, 确认是哪几株粘连未切开。

用法:
  python tools/diag_undersegmentation.py --input evalaute_test --config configs/default.yaml \
      --file cloudR3.ply --pred-id 15
"""
import argparse
import os
import sys

import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.dump_traits_per_plant import process_file
from cabbage_pheno.service.evaluation import align_gt_to_pred

# 复用 oversegmentation_analysis 的中文字体设置
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oversegmentation_analysis import _setup_font, _CJK_FONT_PATHS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--file', required=True, help='文件名, 如 cloudR3.ply')
    parser.add_argument('--pred-id', type=int, required=True, help='预测实例 ID')
    parser.add_argument('--out', default=None, help='输出 PNG 路径')
    args = parser.parse_args()

    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    fp = os.path.join(args.input, args.file)
    if not os.path.exists(fp):
        # 尝试直接匹配 basename
        cands = [os.path.join(args.input, args.file)]
        print(f'未找到 {fp}', flush=True)
        sys.exit(1)

    res = process_file(fp, cfg)
    if res is None:
        print('process_file 返回 None')
        sys.exit(1)

    points_clean = res['points_clean']
    full_instance_pred = res['full_instance_pred']
    gt_points_raw = res['gt_points_raw']
    gt_inst_raw = res['gt_inst_raw']

    # 对齐 GT 到 pred (points_clean 上)
    _, gt_inst_clean = align_gt_to_pred(gt_points_raw, None, gt_inst_raw, points_clean)
    gt_inst_clean = gt_inst_clean.copy()
    gt_inst_clean[gt_inst_clean <= 0] = 0

    # 目标预测实例
    pred_mask = full_instance_pred == args.pred_id
    n_pred = int(pred_mask.sum())
    print(f'\n=== 欠分割诊断: {args.file} pred_id={args.pred_id} ===')
    print(f'预测实例点数: {n_pred:,}')

    if n_pred == 0:
        print('该 pred_id 不存在于当前分割结果')
        sys.exit(1)

    # 统计该预测实例覆盖的真值株分布
    gt_under = gt_inst_clean[pred_mask]
    cnt = Counter(gt_under.tolist())
    cnt.pop(0, None)  # 去掉背景
    total_labeled = sum(cnt.values())

    print(f'\n预测实例 {args.pred_id} 覆盖的真值株构成 (按点数):')
    print(f'{"gt_id":>6} {"点数":>10} {"占比":>8}')
    for gid, c in cnt.most_common():
        print(f'{gid:>6} {c:>10,} {c/total_labeled*100:>7.1f}%')

    # 多数真值株 = 匹配上的 gt_id
    if cnt:
        majority_gt = cnt.most_common(1)[0][0]
        majority_frac = cnt.most_common(1)[0][1] / total_labeled
        print(f'\n多数真值株: gt_id={majority_gt} ({majority_frac*100:.1f}%)')
        print(f'被吞并的真值株: {[g for g in cnt if g != majority_gt]}')
        print(f'欠分割结论: 预测实例由 {len(cnt)} 株真值构成, '
              f'主体 gt_id={majority_gt} 占 {majority_frac*100:.1f}%, '
              f'吞并了 {len(cnt)-1} 株相邻甘蓝')

    # ---- 可视化 ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _setup_font()

    xy = points_clean[:, :2]
    # 只画预测实例附近的区域
    pc = xy[pred_mask]
    cx, cy = pc[:, 0].mean(), pc[:, 1].mean()
    half = max(np.ptp(pc[:, 0]), np.ptp(pc[:, 1])) / 2.0 + 0.15

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))

    # (a) 真值着色: 被吞并的株高亮
    ax = axes[0]
    involved_gt = set(cnt.keys())
    gt_ids = np.unique(gt_inst_clean)
    gt_ids = gt_ids[gt_ids > 0]
    palette = plt.get_cmap('tab20')
    for i, gid in enumerate(gt_ids):
        m = gt_inst_clean == gid
        if gid in involved_gt:
            color = palette((i % 20) / 20.0)
            alpha = 0.9
            s = 0.8
        else:
            color = '0.85'
            alpha = 0.35
            s = 0.3
        ax.scatter(xy[m, 0], xy[m, 1], s=s, c=[color], alpha=alpha, rasterized=True,
                   label=f'GT {gid}' if gid in involved_gt else None)
    ax.scatter(xy[pred_mask, 0], xy[pred_mask, 1], s=0.1, facecolors='none',
               edgecolors='red', linewidths=0.4, alpha=0.6, rasterized=True)
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
    ax.set_aspect('equal')
    ax.set_title(f'(a) 真值着色 (红圈=pred {args.pred_id} 覆盖范围)\n'
                 f'被吞并株: {sorted(involved_gt)}', fontsize=10)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.legend(fontsize=7, loc='best')
    ax.grid(alpha=0.2)

    # (b) 预测着色: 目标实例高亮
    ax = axes[1]
    pred_ids = np.unique(full_instance_pred)
    pred_ids = pred_ids[pred_ids > 0]
    for i, pid in enumerate(pred_ids):
        m = full_instance_pred == pid
        if pid == args.pred_id:
            ax.scatter(xy[m, 0], xy[m, 1], s=0.8, c='red', alpha=0.9, rasterized=True)
        else:
            ax.scatter(xy[m, 0], xy[m, 1], s=0.3, c='0.80', alpha=0.4, rasterized=True)
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
    ax.set_aspect('equal')
    ax.set_title(f'(b) 预测着色 (红=pred {args.pred_id})\n'
                 f'共 {len(pred_ids)} 个预测实例', fontsize=10)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.grid(alpha=0.2)

    # (c) 预测实例内部真值构成占比
    ax = axes[2]
    items = cnt.most_common()
    labels = [f'GT {g}' for g, _ in items]
    vals = [c / total_labeled * 100 for _, c in items]
    colors = [palette((g % 20) / 20.0) for g, _ in items]
    bars = ax.barh(range(len(items))[::-1], vals[::-1], color=colors[::-1], alpha=0.85)
    ax.set_yticks(range(len(items))[::-1])
    ax.set_yticklabels(labels[::-1])
    ax.set_xlabel('占比 (%)')
    ax.set_title(f'(c) pred {args.pred_id} 内部真值构成\n'
                 f'(若 >1 株 → 欠分割)', fontsize=10)
    ax.set_xlim(0, 100)
    for b, v in zip(bars, vals[::-1]):
        ax.text(b.get_width() + 1, b.get_y() + b.get_height()/2,
                f'{v:.1f}%', va='center', fontsize=8)
    ax.grid(alpha=0.2, axis='x')

    fig.suptitle(f'欠分割诊断: {args.file}  pred_id={args.pred_id}'
                 f'  (预测实例吞并 {len(cnt)} 株真值)', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = args.out or os.path.join('output', f'underseg_{os.path.splitext(args.file)[0]}_pred{args.pred_id}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'\n图已保存: {out}')


if __name__ == '__main__':
    main()
