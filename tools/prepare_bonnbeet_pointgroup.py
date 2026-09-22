#!/usr/bin/env python3
"""
BonnBeetClouds3D → PointGroup 训练格式转换。

PointGroup 的 .pth 格式要求: (xyz, rgb, semantic_label, instance_label)
- xyz: (N,3) float32, 米制 (训练时内部 scale 到体素空间)
- rgb: (N,3) float32, [-1, 1] 中心化
- semantic_label: 0=背景, 1=植株
- instance_label: 0-based 连续编号 (每 patch 内重新编号), 背景 = -100

与 RandLA 不同, PointGroup 需要 instance_label 参与训练 (offset + score 分支),
所以必须正确重映射 plant_ids 为 patch 内 0-based 连续编号。

用法:
  python tools/prepare_bonnbeet_pointgroup.py --input BonnBeetClouds3D \
      --output BonnBeetClouds3D/pointgroup_format --voxel 0.005
"""

import argparse
import glob
import logging
import os

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PrepareBonnBeetPG")

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
    """体素下采样: 质心坐标 + 均值颜色 + 众数语义 + 众数实例。"""
    minv = xyz.min(axis=0)
    vidx = np.floor((xyz - minv) / voxel_size).astype(np.int64)
    _, first, inv = np.unique(vidx, axis=0, return_index=True, return_inverse=True)
    cnt = np.bincount(inv)
    n = len(cnt)
    w = cnt.astype(np.float32)[:, None]

    xyz_ds = np.column_stack([
        np.bincount(inv, weights=xyz[:, 0], minlength=n),
        np.bincount(inv, weights=xyz[:, 1], minlength=n),
        np.bincount(inv, weights=xyz[:, 2], minlength=n),
    ]).astype(np.float32) / w
    rgb_ds = np.column_stack([
        np.bincount(inv, weights=rgb[:, 0], minlength=n),
        np.bincount(inv, weights=rgb[:, 1], minlength=n),
        np.bincount(inv, weights=rgb[:, 2], minlength=n),
    ]).astype(np.float32) / w

    pos_cnt = np.bincount(inv, weights=(sem > 0).astype(np.float32), minlength=n)
    sem_ds = (pos_cnt / cnt > 0.5).astype(np.int64)

    # 实例: 每体素内众数 (植株点)
    inst_ds = np.full(n, -1, dtype=np.int64)
    pos_idx = np.where(sem > 0)[0]
    if len(pos_idx) > 0:
        pos_inv = inv[pos_idx]
        pos_inst = inst[pos_idx]
        order = np.argsort(pos_inv, kind='stable')
        sorted_inv = pos_inv[order]
        sorted_inst = pos_inst[order]
        _, first_pos = np.unique(sorted_inv, return_index=True)
        inst_ds[sorted_inv[first_pos]] = sorted_inst[first_pos]

    return xyz_ds, rgb_ds, sem_ds, inst_ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='BonnBeetClouds3D')
    parser.add_argument('--output', type=str, default='BonnBeetClouds3D/pointgroup_format')
    parser.add_argument('--voxel', type=float, default=0.005)
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val'])
    args = parser.parse_args()

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
                continue

            sem = (plant_ids > 0).astype(np.int64)
            inst = plant_ids.copy()

            if args.voxel and args.voxel > 0:
                xyz, rgb, sem, inst = voxel_downsample(xyz, rgb, sem, inst, args.voxel)

            # rgb 中心化 [-1, 1]
            rgb = (rgb / 255.0 * 2.0 - 1.0).astype(np.float32)

            # PointGroup instance 要求 0-based 连续编号 + 背景 = -100
            inst_pg = np.full(len(inst), -100, dtype=np.int64)
            plant_mask = sem > 0
            if plant_mask.sum() > 0:
                uniq = np.unique(inst[plant_mask])
                remap = {pid: i for i, pid in enumerate(uniq)}
                inst_pg[plant_mask] = np.array([remap[p] for p in inst[plant_mask]])

            torch.save((xyz, rgb, sem, inst_pg), out_fp)
            logger.info(f'  {base}: {len(xyz):,} 点, {len(np.unique(inst_pg[plant_mask]))} 株')

    logger.info('转换完成')


if __name__ == '__main__':
    main()
