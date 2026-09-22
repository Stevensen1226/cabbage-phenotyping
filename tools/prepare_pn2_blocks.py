#!/usr/bin/env python3
"""准备真正的 PointNet2 (PointNet++) 分块训练数据。

从 evalaute_test 的 21 块田地生成:
- 输入: evalaute_test/cloudR*.ply (xyz + rgb) + cloudR*_gt.txt (修复后的语义标签)
- 输出: data/pn2_blocks/train/*.npy, 每个块 (N, 7) = xyz(3) + rgb(3) + sem_label(1)
- rgb 归一化到 [0, 1]

用法:
  python tools/prepare_pn2_blocks.py --block-size 1.0 --stride 0.5
"""
import os
import glob
import numpy as np
import argparse

def read_ply_xyz_rgb(path):
    """读取 ascii ply 的 xyz + rgb, 返回 (N,3) float, (N,3) float[0,1]"""
    with open(path) as f:
        lines = f.readlines()
    # 找 end_header
    start = 0
    props = []
    for i, line in enumerate(lines):
        if line.startswith('property'):
            props.append(line.split()[-1])
        if line.strip() == 'end_header':
            start = i + 1
            break
    data = np.loadtxt(lines[start:])
    # 找到 xyz rgb 列
    xyz = data[:, :3].astype(np.float32)
    # 找 red/green/blue 列
    rgb_idx = [props.index(c) for c in ['red', 'green', 'blue']]
    rgb = data[:, rgb_idx].astype(np.float32) / 255.0  # -> [0,1]
    return xyz, rgb


def read_gt_sem(path):
    """读取 gt.txt 的语义标签 (第7列, index 6), 返回 (N,) int"""
    data = np.loadtxt(path)
    sem = (data[:, -2] > 0).astype(np.int64)  # 倒数第2列 = 语义标签
    return sem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--block-size', type=float, default=1.0)
    parser.add_argument('--stride', type=float, default=0.5)
    parser.add_argument('--min-points', type=int, default=100)
    parser.add_argument('--output', default='data/pn2_blocks')
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output, 'train'), exist_ok=True)

    ply_files = sorted(glob.glob('evalaute_test/cloudR*.ply'))
    ply_files = [f for f in ply_files if 'step' not in os.path.basename(f)
                 and 'sor' not in os.path.basename(f)]

    total_blocks = 0
    block_idx = 0
    for ply_path in ply_files:
        base = os.path.splitext(os.path.basename(ply_path))[0]
        gt_path = os.path.join('evalaute_test', base + '_gt.txt')
        if not os.path.exists(gt_path):
            print(f'[SKIP] {base}: 无 GT')
            continue

        xyz, rgb = read_ply_xyz_rgb(ply_path)
        sem = read_gt_sem(gt_path)
        assert len(xyz) == len(sem), f'{base}: 点数不一致 {len(xyz)} vs {len(sem)}'

        # 网格切块 (只按 XY 平面切)
        min_bound = xyz[:, :2].min(0)
        max_bound = xyz[:, :2].max(0)

        n_file_blocks = 0
        x0 = min_bound[0]
        while x0 < max_bound[0]:
            y0 = min_bound[1]
            while y0 < max_bound[1]:
                x1 = x0 + args.block_size
                y1 = y0 + args.block_size
                mask = (xyz[:, 0] >= x0) & (xyz[:, 0] < x1) & \
                       (xyz[:, 1] >= y0) & (xyz[:, 1] < y1)
                if mask.sum() >= args.min_points:
                    block = np.column_stack([xyz[mask], rgb[mask], sem[mask].astype(np.float32)])
                    out = os.path.join(args.output, 'train', f'{base}_block_{block_idx:05d}.npy')
                    np.save(out, block)
                    block_idx += 1
                    n_file_blocks += 1
                y0 += args.stride
            x0 += args.stride

        total_blocks += n_file_blocks
        print(f'{base}: {n_file_blocks} blocks ({len(xyz)} pts)')

    print(f'\n总计: {total_blocks} blocks -> {args.output}/train/')


if __name__ == '__main__':
    main()
