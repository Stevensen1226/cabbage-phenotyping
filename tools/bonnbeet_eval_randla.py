#!/usr/bin/env python3
"""
BonnBeetClouds3D 完整方法迁移评估: RandLA-Net 语义 + 几何实例分割。

与 tools/bonnbeet_generalize.py 的区别:
  语义分割不再用 GT oracle, 而是用远程训练的 RandLA-Net 模型推理。
  几何实例分割层 (watershed/meanshift + GIDM) 与 oracle 实验完全一致。

用法:
  python tools/bonnbeet_eval_randla.py --input BonnBeetClouds3D \
      --config configs/default.yaml --split val \
      --ckpt models/randlanet_bonnbeet_best.pth \
      --methods watershed_3d meanshift

环境: 需 torch + open3d (bgpseg)
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

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import open3d as o3d
import yaml

from cabbage_pheno.instance import InstanceClusterer
from cabbage_pheno.service.evaluation import evaluate_from_predictions
from tools.bonnbeet_generalize import (
    read_bonnbeet_ply, build_clusterer,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BonnBeetEvalRandLA")


def voxel_downsample3(xyz, rgb, plant_ids, voxel_size):
    """体素下采样 xyz(质心) + rgb(均值) + plant_ids(众数)。"""
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
    ids_ds = plant_ids[first]
    return xyz_ds, rgb_ds, ids_ds


def load_randla(ckpt):
    import torch
    sys.path.insert(0, os.path.join(PROJ_ROOT, 'RandLANet_Ours'))
    from randlanet import RandLANet
    model = RandLANet(d_in=6, num_classes=2).cuda().eval()
    sd = torch.load(ckpt, map_location='cuda', weights_only=False)
    model.load_state_dict(sd['model'])
    logger.info(f'RandLA-Net loaded (epoch {sd.get("epoch", "?")}), '
                f'{sum(p.numel() for p in model.parameters())/1e6:.2f}M params')
    return model


def randla_infer(model, xyz, rgb, chunk=120000):
    """RandLA-Net 分块语义推理。xyz=(N,3)米, rgb=(N,3)∈[0,1]。返回 (N,) 0/1。"""
    import torch
    N = len(xyz)
    rgb_norm = (rgb * 2.0 - 1.0).astype(np.float32)
    xyz_shift = (xyz - xyz.min(0)).astype(np.float32)
    mask = np.zeros(N, dtype=bool)
    with torch.no_grad():
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            x = torch.from_numpy(xyz_shift[start:end]).unsqueeze(0).cuda()
            r = torch.from_numpy(rgb_norm[start:end]).unsqueeze(0).cuda()
            pred = model(x, r)
            mask[start:end] = pred[0].max(1)[1].cpu().numpy() == 1
    return mask.astype(int)


def run_file(fp, cfg, methods, iou_thresh, voxel_size, model):
    xyz, rgb, plant_ids = read_bonnbeet_ply(fp)
    if plant_ids is None:
        return {}

    if voxel_size and voxel_size > 0:
        xyz, rgb, plant_ids = voxel_downsample3(xyz, rgb, plant_ids, voxel_size)

    N = len(xyz)
    gt_sem = (plant_ids > 0).astype(int)
    gt_inst = plant_ids.copy()
    gt_inst[gt_inst < 0] = 0

    # RandLA-Net 语义推理
    t0 = time.time()
    pred_sem = randla_infer(model, xyz, rgb)
    sem_time = time.time() - t0

    plant_mask = pred_sem > 0
    n_plant = int(plant_mask.sum())
    logger.info(f'{os.path.basename(fp)}: {N:,} 点, RandLA预测植株 {n_plant:,} '
                f'(GT {(gt_sem>0).sum():,}) | 语义推理 {sem_time:.1f}s')

    records = {}
    if n_plant == 0:
        return records

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[plant_mask])

    for method in methods:
        clusterer = build_clusterer(cfg, method)
        t1 = time.time()
        try:
            labels, num_inst = clusterer.cluster(pcd)
        except Exception as e:
            logger.error(f'[{method}] 聚类失败 {fp}: {e}')
            continue
        runtime = sem_time + (time.time() - t1)

        pred_inst = np.zeros(N, dtype=int)
        pred_inst[plant_mask] = labels + 1

        records[method] = {
            'fname': os.path.basename(fp),
            'pred_sem': pred_sem,
            'pred_inst': pred_inst,
            'gt_sem': gt_sem,
            'gt_inst': gt_inst,
            'runtime': runtime,
        }
        logger.info(f'  [{method}] 预测 {num_inst} 株, 总用时 {runtime:.1f}s')

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='BonnBeetClouds3D')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'])
    parser.add_argument('--ckpt', type=str, default='models/randlanet_bonnbeet_best.pth')
    parser.add_argument('--methods', type=str, nargs='+', default=['watershed_3d', 'meanshift'])
    parser.add_argument('--voxel', type=float, default=0.005)
    parser.add_argument('--max-files', type=int, default=0)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    import torch
    cfg = yaml.safe_load(open(args.config, encoding='utf-8'))
    iou_thresh = float(cfg.get('evaluation', {}).get('iou_thresh', 0.5))
    model = load_randla(args.ckpt)

    files = sorted(glob.glob(os.path.join(args.input, args.split, '*.ply')))
    if args.max_files > 0:
        files = files[:args.max_files]

    logger.info(f'方法: {args.methods} | 语义: RandLA-Net | 文件数: {len(files)}')

    per_method_records = {m: [] for m in args.methods}
    for fp in files:
        recs = run_file(fp, cfg, args.methods, iou_thresh, args.voxel, model)
        for m, rec in recs.items():
            per_method_records[m].append(rec)

    results = {}
    for m in args.methods:
        recs = per_method_records[m]
        results[m] = (evaluate_from_predictions(recs, iou_thresh=iou_thresh, label=m)
                      if recs else {'label': m, 'processed_files': [], 'per_file': [], 'summary': {}})

    if args.output is None:
        args.output = os.path.join('output', f'bonnbeet_randla_{args.split}.json')
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else 'output', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f'结果已保存: {args.output}')

    print('\n' + '=' * 100)
    print(f'BonnBeetClouds3D 完整方法迁移 (RandLA-Net 语义 + 几何实例) | iou_thresh={iou_thresh}, {len(files)} 文件')
    print('=' * 100)
    print(f'{"方法":<16}{"Inst Prec":>11}{"Inst Rec":>11}{"Inst F1":>11}{"Inst mIoU":>11}{"Sem mIoU":>11}{"MAE(Count)":>12}')
    print('-' * 100)
    for m in args.methods:
        s = results[m].get('summary', {})
        if not s:
            print(f'{m:<16}{"N/A":>11}')
            continue
        print(f'{m:<16}{s["avg_inst_prec"]:>11.3f}{s["avg_inst_rec"]:>11.3f}'
              f'{s["avg_inst_f1"]:>11.3f}{s["avg_inst_miou"]:>11.3f}'
              f'{s["avg_sem_miou"]:>11.3f}{s["mae_count"]:>12.3f}')
    print('=' * 100)


if __name__ == '__main__':
    main()
