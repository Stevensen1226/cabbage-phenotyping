#!/usr/bin/env python3
"""
BonnBeetClouds3D 株级实例分割泛化实验。

将甘蓝(大场地)的几何实例分割方法(GIDM 骨架切割 + 聚类)迁移到甜菜数据集。
语义分割用 GT (plant_ids > 0) 作为 oracle, 只泛化几何实例分割层, 评估株级
实例分割的跨物种泛化能力。

用法:
  # 全量 (train + val, 共 174 个有标签 patch)
  python tools/bonnbeet_generalize.py --input BonnBeetClouds3D --config configs/default.yaml

  # 只跑 val (25 patch, 快速验证)
  python tools/bonnbeet_generalize.py --input BonnBeetClouds3D --config configs/default.yaml --split val

  # 只跑 train
  python tools/bonnbeet_generalize.py --input BonnBeetClouds3D --config configs/default.yaml --split train

  # 指定方法 (默认 watershed_3d meanshift hdbscan)
  python tools/bonnbeet_generalize.py ... --methods watershed_3d meanshift

  # 限制文件数 (调试)
  python tools/bonnbeet_generalize.py ... --max-files 5
"""

import argparse
import copy
import glob
import json
import logging
import os
import sys
import time

import numpy as np

# 项目根加入 sys.path, 以便 import cabbage_pheno
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import open3d as o3d
import yaml

from cabbage_pheno.instance import InstanceClusterer
from cabbage_pheno.service.evaluation import evaluate_from_predictions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BonnBeetGeneralize")

_TYPE_MAP = {'float': '<f4', 'double': '<f8', 'int': '<i4', 'uchar': 'u1'}


def parse_ply_header(fp):
    """解析 PLY 头部, 返回 (property_names, property_types, vertex_count)。"""
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
    """读取 BonnBeet PLY, 返回 (xyz, rgb, plant_ids)。plant_ids 不存在时返回 None。"""
    names, types, vertex_count = parse_ply_header(fp)
    dtype = np.dtype([(n, _TYPE_MAP[t]) for n, t in zip(names, types)])

    with open(fp, 'rb') as f:
        while f.readline().strip() != b'end_header':
            pass
        data = np.fromfile(f, dtype=dtype, count=vertex_count)

    xyz = np.column_stack([data['x'], data['y'], data['z']]).astype(np.float32)

    if 'red' in names:
        rgb = np.column_stack([data['red'], data['green'], data['blue']]).astype(np.float32) / 255.0
    else:
        rgb = np.ones_like(xyz)

    plant_ids = data['plant_ids'].astype(np.int64) if 'plant_ids' in names else None
    return xyz, rgb, plant_ids


def voxel_downsample(xyz, labels, voxel_size):
    """体素下采样: 返回 (xyz_ds, labels_ds)。xyz 取体素质心, labels 取体素内第一个点。

    甜菜 1mm GSD 的密度约为甘蓝 10-30 倍, 需下采样到甘蓝 voxel_size(0.005m)
    以做公平的密度归一化(而非改方法参数)。
    """
    minv = xyz.min(axis=0)
    vidx = np.floor((xyz - minv) / voxel_size).astype(np.int64)
    _, first, inv = np.unique(vidx, axis=0, return_index=True, return_inverse=True)
    cnt = np.bincount(inv)
    n = len(cnt)
    xs = np.bincount(inv, weights=xyz[:, 0], minlength=n)
    ys = np.bincount(inv, weights=xyz[:, 1], minlength=n)
    zs = np.bincount(inv, weights=xyz[:, 2], minlength=n)
    xyz_ds = np.column_stack([xs, ys, zs]).astype(np.float32) / cnt[:, None]
    labels_ds = labels[first]
    return xyz_ds, labels_ds


def build_clusterer(cfg, method):
    """用指定实例聚类方法构造 InstanceClusterer。"""
    cfg2 = copy.deepcopy(cfg)
    cfg2['instance']['method'] = method
    return InstanceClusterer(cfg2)


def run_file(fp, cfg, methods, iou_thresh, voxel_size):
    """对单个 patch 用多种聚类方法跑株级实例分割, 返回 per-method records。"""
    xyz, rgb, plant_ids = read_bonnbeet_ply(fp)
    if plant_ids is None:
        logger.warning(f'无 plant_ids 标签, 跳过: {fp}')
        return {}

    N_raw = len(xyz)

    # 密度归一化: 甜菜 1mm GSD 密度为甘蓝 ~10-30 倍, 体素下采样到甘蓝 voxel_size
    if voxel_size and voxel_size > 0:
        xyz, plant_ids = voxel_downsample(xyz, plant_ids, voxel_size)

    N = len(xyz)
    plant_mask = plant_ids > 0

    # oracle 语义: plant_ids > 0 即为植株
    gt_sem = plant_mask.astype(int)
    gt_inst = plant_ids.copy()
    gt_inst[gt_inst < 0] = 0

    xyz_plant = xyz[plant_mask]
    n_plant = int(plant_mask.sum())
    logger.info(f'{os.path.basename(fp)}: {N_raw:,}→{N:,} 点, 植株 {n_plant:,} 点, '
                f'{len(np.unique(gt_inst[gt_inst > 0]))} 株')

    records = {}
    if n_plant == 0:
        return records

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_plant)

    for method in methods:
        clusterer = build_clusterer(cfg, method)
        t0 = time.time()
        try:
            labels, num_inst = clusterer.cluster(pcd)
        except Exception as e:
            logger.error(f'[{method}] 聚类失败 {fp}: {e}')
            continue
        runtime = time.time() - t0

        # cluster 返回 -1=背景/噪声, 0..n-1=实例 → 转评估格式: 0=背景, >=1=实例
        pred_inst = np.zeros(N, dtype=int)
        plant_labels = labels + 1  # -1 -> 0, 0 -> 1, 1 -> 2 ...
        pred_inst[plant_mask] = plant_labels

        pred_sem = plant_mask.astype(int)  # oracle 语义

        records[method] = {
            'fname': os.path.basename(fp),
            'pred_sem': pred_sem,
            'pred_inst': pred_inst,
            'gt_sem': gt_sem,
            'gt_inst': gt_inst,
            'runtime': runtime,
        }
        logger.info(f'  [{method}] 预测 {num_inst} 株, 用时 {runtime:.1f}s')

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='BonnBeetClouds3D')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--split', type=str, default=None, choices=['train', 'val', 'test', 'all'])
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['watershed_3d', 'meanshift', 'hdbscan'])
    parser.add_argument('--voxel', type=float, default=0.005,
                        help='体素下采样(m), 密度归一化到甘蓝水平, 0=不下采样')
    parser.add_argument('--max-files', type=int, default=0, help='限制处理文件数(调试)')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding='utf-8'))
    iou_thresh = float(cfg.get('evaluation', {}).get('iou_thresh', 0.5))

    # 收集文件
    if args.split and args.split != 'all':
        splits = [args.split]
    else:
        splits = ['train', 'val']

    files = []
    for sp in splits:
        d = os.path.join(args.input, sp)
        if os.path.isdir(d):
            files.extend(sorted(glob.glob(os.path.join(d, '*.ply'))))
    if args.max_files > 0:
        files = files[:args.max_files]

    logger.info(f'方法: {args.methods} | IoU阈值: {iou_thresh} | 文件数: {len(files)} '
                f'| 体素下采样: {args.voxel}m')

    # 每个方法累积 records
    per_method_records = {m: [] for m in args.methods}
    for fp in files:
        recs = run_file(fp, cfg, args.methods, iou_thresh, args.voxel)
        for m, rec in recs.items():
            per_method_records[m].append(rec)

    results = {}
    for m in args.methods:
        recs = per_method_records[m]
        if not recs:
            results[m] = {'label': m, 'processed_files': [], 'per_file': [], 'summary': {}}
            continue
        results[m] = evaluate_from_predictions(recs, iou_thresh=iou_thresh, label=m)

    # 保存
    if args.output is None:
        split_tag = args.split or 'trainval'
        args.output = os.path.join('output', f'bonnbeet_generalize_{split_tag}.json')
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else 'output',
                exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f'结果已保存: {args.output}')

    # 打印对比表
    print('\n' + '=' * 90)
    print(f'BonnBeetClouds3D 株级实例分割泛化结果 (iou_thresh={iou_thresh}, {len(files)} 文件)')
    print('=' * 90)
    header = (f'{"方法":<16}{"Inst Prec":>11}{"Inst Rec":>11}{"Inst F1":>11}'
              f'{"Inst mIoU":>11}{"Sem mIoU":>11}{"MAE(Count)":>12}')
    print(header)
    print('-' * 90)
    for m in args.methods:
        s = results[m].get('summary', {})
        if not s:
            print(f'{m:<16}{"N/A":>11}')
            continue
        print(f'{m:<16}{s["avg_inst_prec"]:>11.3f}{s["avg_inst_rec"]:>11.3f}'
              f'{s["avg_inst_f1"]:>11.3f}{s["avg_inst_miou"]:>11.3f}'
              f'{s["avg_sem_miou"]:>11.3f}{s["mae_count"]:>12.3f}')
    print('=' * 90)


if __name__ == '__main__':
    main()
