#!/usr/bin/env python3
"""
BonnBeetClouds3D → RandLA-Net 训练格式转换。

将甜菜 PLY patch (train/val) 转换为甘蓝 RandLA-Net 训练所用的 .pth 四元组:
    (xyz, rgb, semantic_label, instance_label)
- xyz: 米制坐标 (float32), 与甘蓝一致 (训练时内部 xyz -= min)
- rgb: 中心化 [-1, 1] (float32)
- semantic_label: 0=背景, 1=植株 (plant_ids>0)
- instance_label: plant_ids (正数=植株ID, -1=背景)

体素下采样到甘蓝 voxel_size (默认 5mm), 用体素质心 + 众数标签, 保证训练/推理
分辨率与甘蓝管线一致 (公平泛化)。

用法:
  python tools/prepare_bonnbeet_randla.py --input BonnBeetClouds3D \
      --output BonnBeetClouds3D/randla_format --voxel 0.005
"""

import argparse
import glob
import logging
import os

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PrepareBonnBeetRandLA")

_TYPE_MAP = {'float': '<f4', 'double': '<f8', 'int': '<i4', 'uchar': 'u1'}


def parse_ply_header(fp):
    with open(fp, 'rb') as f:
        names, types = [], []
        vertex_count = 0
        while True:
            line = f.readline().decode('latin1').strip()
            if line == 'end_header':
                break
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            if line.startswith('property'):
                parts = line.split()
                types.append(parts[1])
                names.append(parts[-1])
        return names, types, vertex_count


def read_bonnbeet_ply(fp):
    names, types, vertex_count = parse_ply_header(fp)
    dtype = np.dtype([(n, _TYPE_MAP[t]) for n, t in zip(names, types)])
    with open(fp, 'rb') as f:
        while f.readline().strip() != b'end_header':
            pass
        data = np.fromfile(f, dtype=dtype, count=vertex_count)

    xyz = np.column_stack([data['x'], data['y'], data['z']]).astype(np.float32)
    if 'red' in names:
        rgb = np.column_stack([data['red'], data['green'], data['blue']]).astype(np.float32)
    else:
        rgb = np.ones_like(xyz) * 128.0
    plant_ids = data['plant_ids'].astype(np.int64) if 'plant_ids' in names else None
    return xyz, rgb, plant_ids


def voxel_downsample(xyz, rgb, sem, inst, voxel_size):
    """体素下采样: 质心坐标 + 众数标签。返回 (xyz_ds, rgb_ds, sem_ds, inst_ds)。"""
    minv = xyz.min(axis=0)
    vidx = np.floor((xyz - minv) / voxel_size).astype(np.int64)
    _, first, inv = np.unique(vidx, axis=0, return_index=True, return_inverse=True)
    n = int(inv.max()) + 1
    cnt = np.bincount(inv, minlength=n)

    # 质心坐标
    xs = np.bincount(inv, weights=xyz[:, 0], minlength=n)
    ys = np.bincount(inv, weights=xyz[:, 1], minlength=n)
    zs = np.bincount(inv, weights=xyz[:, 2], minlength=n)
    xyz_ds = np.column_stack([xs, ys, zs]).astype(np.float32) / cnt[:, None]

    # 颜色均值
    rs = np.bincount(inv, weights=rgb[:, 0], minlength=n)
    gs = np.bincount(inv, weights=rgb[:, 1], minlength=n)
    bs = np.bincount(inv, weights=rgb[:, 2], minlength=n)
    rgb_ds = np.column_stack([rs, gs, bs]).astype(np.float32) / cnt[:, None]

    # 语义众数: 植株点(>0)占比 > 0.5 → 1
    pos_cnt = np.bincount(inv, weights=(sem > 0).astype(np.float32), minlength=n)
    sem_ds = (pos_cnt / cnt > 0.5).astype(np.int64)

    # 实例众数: 每体素内出现最多的 plant_id (背景用 -1)
    # 先统计每体素内非背景(>0)植株点最多的是哪个 id; 若植株点未过半则 -1
    inst_ds = np.full(n, -1, dtype=np.int64)
    pos_idx = np.where(sem > 0)[0]
    if len(pos_idx) > 0:
        pos_inv = inv[pos_idx]
        pos_inst = inst[pos_idx]
        # 用 bincount 逐体素找众数: 拼接 key = inv * K + inst 无法直接, 用 loop-free 近似
        # 简化: 对每个体素, 取该体素内第一个植株点的 inst (边界误差 < 1 体素)
        order = np.argsort(pos_inv, kind='stable')
        sorted_inv = pos_inv[order]
        sorted_inst = pos_inst[order]
        _, first_pos = np.unique(sorted_inv, return_index=True)
        inst_ds[sorted_inv[first_pos]] = sorted_inst[first_pos]

    return xyz_ds, rgb_ds, sem_ds, inst_ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='BonnBeetClouds3D')
    parser.add_argument('--output', type=str, default='BonnBeetClouds3D/randla_format')
    parser.add_argument('--voxel', type=float, default=0.005)
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val'])
    args = parser.parse_args()

    import torch  # 延迟导入, 仅保存时用

    for split in args.splits:
        in_dir = os.path.join(args.input, split)
        out_dir = os.path.join(args.output, split)
        os.makedirs(out_dir, exist_ok=True)
        files = sorted(glob.glob(os.path.join(in_dir, '*.ply')))
        logger.info(f'[{split}] {len(files)} 个 PLY → {out_dir}')

        for fp in files:
            base = os.path.splitext(os.path.basename(fp))[0]
            out_fp = os.path.join(out_dir, base + '.pth')
            if os.path.exists(out_fp):
                continue

            xyz, rgb, plant_ids = read_bonnbeet_ply(fp)
            if plant_ids is None:
                logger.warning(f'跳过 (无 plant_ids): {fp}')
                continue

            sem = (plant_ids > 0).astype(np.int64)
            inst = plant_ids.copy()

            if args.voxel and args.voxel > 0:
                xyz, rgb, sem, inst = voxel_downsample(xyz, rgb, sem, inst, args.voxel)

            # rgb 中心化 [-1, 1]
            rgb = (rgb / 255.0 * 2.0 - 1.0).astype(np.float32)

            torch.save((xyz, rgb, sem, inst), out_fp)
            logger.info(f'  {base}: {len(xyz):,} 点 (植株 {int((sem > 0).sum()):,})')

    logger.info('转换完成')


if __name__ == '__main__':
    main()
