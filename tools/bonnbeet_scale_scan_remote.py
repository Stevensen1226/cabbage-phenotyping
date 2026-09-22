#!/usr/bin/env python3
"""
BonnBeetClouds3D oracle 几何实验 — 尺度适配扫描 (远程版)。

直接从 randla_format 的 .pth 读取 (已体素下采样到 0.005m 的 xyz + 原始 plant_ids),
语义用 GT (sem>0) 作为 oracle, 只测几何实例分割层跨物种泛化。

扫描关键尺度参数:
  - watershed_3d.min_seed_distance  (甘蓝默认 0.30m -> 甜菜株间距 18cm)
  - meanshift.bandwidth             (甘蓝默认 0.25m)

用法 (远程):
  LD_LIBRARY_PATH=.../torch/lib python bonnbeet_scale_scan_remote.py \
      --data /root/autodl-tmp/bonnbeet_randla/val \
      --config /root/autodl-tmp/configs/default.yaml
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
import torch

# 让 cabbage_pheno 可导入 (远程在 /root/autodl-tmp/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import open3d as o3d
import yaml

from cabbage_pheno.instance import InstanceClusterer
from cabbage_pheno.service.evaluation import evaluate_from_predictions

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BonnBeetScaleScanRemote")


def build_clusterer_with(cfg, method, overrides):
    cfg2 = copy.deepcopy(cfg)
    cfg2['instance']['method'] = method
    for sub, params in overrides.items():
        cfg2['instance'].setdefault(sub, {}).update(params)
    return InstanceClusterer(cfg2)


def run_param_point(fp, cfg, method, overrides, iou_thresh):
    """单参数点单文件: 从 pth 读 xyz + plant_ids。"""
    xyz, rgb, sem, inst = torch.load(fp, weights_only=False)
    if hasattr(xyz, 'numpy'):
        xyz = xyz.numpy()
    if hasattr(inst, 'numpy'):
        inst = inst.numpy()
    xyz = xyz.astype(np.float32)

    N = len(xyz)
    plant_mask = inst > 0
    gt_sem = plant_mask.astype(int)
    gt_inst = inst.copy().astype(np.int64)
    gt_inst[gt_inst < 0] = 0

    xyz_plant = xyz[plant_mask]
    if len(xyz_plant) == 0:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_plant)

    clusterer = build_clusterer_with(cfg, method, overrides)
    try:
        labels, num_inst = clusterer.cluster(pcd)
    except Exception as e:
        logger.error(f'[{method}] {overrides} 聚类失败 {fp}: {e}')
        return None

    pred_inst = np.zeros(N, dtype=int)
    pred_inst[plant_mask] = labels + 1
    pred_sem = plant_mask.astype(int)

    return {
        'fname': os.path.basename(fp),
        'pred_sem': pred_sem,
        'pred_inst': pred_inst,
        'gt_sem': gt_sem,
        'gt_inst': gt_inst,
        'runtime': 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/root/autodl-tmp/bonnbeet_randla/val')
    parser.add_argument('--config', type=str, default='/root/autodl-tmp/configs/default.yaml')
    parser.add_argument('--methods', type=str, nargs='+', default=['watershed_3d', 'meanshift'])
    parser.add_argument('--max-files', type=int, default=0)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding='utf-8'))
    iou_thresh = float(cfg.get('evaluation', {}).get('iou_thresh', 0.5))

    files = sorted(glob.glob(os.path.join(args.data, '*.pth')))
    if args.max_files > 0:
        files = files[:args.max_files]
    logger.info(f'扫描 {len(files)} 个文件')

    scan_points = {
        'watershed_3d': [
            {'watershed_3d': {'min_seed_distance': v}}
            for v in [0.30, 0.25, 0.20, 0.18, 0.15, 0.12, 0.10, 0.08]
        ],
        'meanshift': [
            {'meanshift': {'bandwidth': v}}
            for v in [0.25, 0.20, 0.18, 0.15, 0.12, 0.10, 0.08]
        ],
    }

    results = {}
    for method in args.methods:
        results[method] = {}
        for overrides in scan_points.get(method, []):
            recs = []
            for fp in files:
                rec = run_param_point(fp, cfg, method, overrides, iou_thresh)
                if rec is not None:
                    recs.append(rec)
            if not recs:
                continue
            label = list(overrides.values())[0]
            param_str = ','.join(f'{k}={v}' for k, v in label.items())
            res = evaluate_from_predictions(recs, iou_thresh=iou_thresh, label=f'{method}:{param_str}')
            results[method][param_str] = res['summary']
            s = res['summary']
            logger.info(f'  [{method}] {param_str}: F1={s["avg_inst_f1"]:.3f}, '
                        f'Prec={s["avg_inst_prec"]:.3f}, Rec={s["avg_inst_rec"]:.3f}, '
                        f'MAE={s["mae_count"]:.3f}')

    if args.output is None:
        args.output = os.path.join(ROOT, 'output', 'bonnbeet_scale_scan_val.json')
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f'结果已保存: {args.output}')

    for method in args.methods:
        print(f'\n{"=" * 80}')
        print(f'{method} 尺度扫描结果 ({len(files)} 文件, iou_thresh={iou_thresh})')
        print(f'{"=" * 80}')
        print(f'{"参数":<30}{"Inst Prec":>11}{"Inst Rec":>11}{"Inst F1":>11}'
              f'{"Inst mIoU":>11}{"MAE(Count)":>12}')
        print('-' * 80)
        for param_str, s in results[method].items():
            print(f'{param_str:<30}{s["avg_inst_prec"]:>11.3f}{s["avg_inst_rec"]:>11.3f}'
                  f'{s["avg_inst_f1"]:>11.3f}{s["avg_inst_miou"]:>11.3f}'
                  f'{s["mae_count"]:>12.3f}')


if __name__ == '__main__':
    main()
