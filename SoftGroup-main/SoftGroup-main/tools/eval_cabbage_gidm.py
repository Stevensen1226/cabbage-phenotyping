#!/usr/bin/env python
"""SoftGroup + GIDM 对照实验。

对 SoftGroup 端到端输出 (欠分割的粘连大簇) 应用 GIDM (PCA 骨架波谷切割 + 碎片回并),
量化 GIDM 能否挽救端到端检测式方法的欠分割。

流程:
  1. SoftGroup 推理 -> pred_sem (N,), pred_inst (N,) 0=背景, 1..K=实例
  2. 提取甘蓝点, 对其上的实例标签应用 GIDM (_apply_skeleton_split + _merge_fragments)
  3. 与无 GIDM 的 SoftGroup 端到端结果对比

用法:
  cd /root/autodl-tmp/SoftGroup-main
  python tools/eval_cabbage_gidm.py [checkpoint]
"""
import os
import sys
import json
import time
import logging

import numpy as np
import torch
import open3d as o3d
import yaml

sys.path.insert(0, '/root/autodl-tmp/SoftGroup-main')
sys.path.insert(0, '/root/autodl-tmp')

from softgroup.model import SoftGroup
from softgroup.util import get_root_logger, load_checkpoint, rle_decode
from softgroup.ops import voxelization_idx


def build_single_batch(xyz, rgb, scale, spatial_shape_min):
    """构建 SoftGroup forward_test 所需的单个 batch。"""
    N = xyz.shape[0]
    xyz_middle = xyz.copy()
    xyz_voxel = (xyz_middle * scale)
    xyz_voxel = xyz_voxel - xyz_voxel.min(0)
    coords = torch.cat([
        torch.zeros(N, 1, dtype=torch.long),
        torch.from_numpy(xyz_voxel).long()
    ], dim=1)
    batch_idxs = coords[:, 0].int()
    voxel_coords, v2p_map, p2v_map = voxelization_idx(coords, 1)
    spatial_shape = np.clip(
        coords.max(0)[0][1:].numpy() + 1, spatial_shape_min, None)
    batch = {
        'scan_ids': ['scene'],
        'coords': coords,
        'batch_idxs': batch_idxs,
        'voxel_coords': voxel_coords,
        'p2v_map': p2v_map,
        'v2p_map': v2p_map,
        'coords_float': torch.from_numpy(xyz_middle).float(),
        'feats': torch.from_numpy(rgb).float(),
        'semantic_labels': torch.zeros(N, dtype=torch.long),
        'instance_labels': torch.zeros(N, dtype=torch.long),
        'instance_pointnum': torch.tensor([], dtype=torch.int),
        'instance_cls': torch.tensor([], dtype=torch.long),
        'pt_offset_labels': torch.zeros(N, 3).float(),
        'spatial_shape': spatial_shape,
        'batch_size': 1,
    }
    return batch


def decode_instances(pred_instances, num_points):
    """把 SoftGroup 输出的 RLE 实例列表解码为点级 instance label。"""
    inst_label = np.zeros(num_points, dtype=np.int64)
    for i, inst in enumerate(pred_instances):
        mask = rle_decode(inst['pred_mask'])
        inst_label[mask == 1] = i + 1
    return inst_label


def apply_gidm(pred_inst, cabbage_xyz):
    """对甘蓝点上的实例标签应用 GIDM (骨架切割 + 碎片回并)。
    pred_inst: (N,) 0=背景, 1..K=实例 (欠分割大簇)
    cabbage_xyz: (N,3) 甘蓝点坐标
    返回: 细化后的 pred_inst (0=背景, 1..K'=实例)
    """
    from cabbage_pheno.instance.clustering import InstanceClusterer

    # 构造 GIDM 配置 (与论文消融一致)
    cfg = {
        'instance': {
            'method': 'watershed_3d',
            'min_cluster_points': 2000,
            'max_cluster_points': 100000,
            'pca_split': {
                'enable': True,
                'abnormal_diameter': 0.6,
                'abnormal_aspect_ratio': 1.6,
                'skeleton_bins': 15,
                'min_peak_dist_m': 0.3,
                'valley_depth_rel': 0.9,
                'merge_min_diameter': 0.15,
                'merge_max_dist': 0.2,
                'merge_max_angle': 30,
                # 消融开关 (默认 True = 完整 GIDM)
                'abnormal_detect': True,
                'skeleton_cut': True,
                'adaptive_thresh': True,
                'merge_back': True,
            },
            'fragment_voting': {
                'enable': False,  # 骨架切割 + 回并已含碎片处理, 关闭额外投票
                'discard_threshold': 500,
                'merge_threshold': 1500,
            },
        }
    }
    clusterer = InstanceClusterer(cfg)

    # SoftGroup 的 0=背景 -> GIDM 的 -1=背景
    labels = pred_inst.copy()
    labels[labels == 0] = -1

    # GIDM: PCA 骨架波谷切割
    labels = clusterer._apply_skeleton_split(labels, cabbage_xyz)
    # GIDM: 碎片回并
    labels = clusterer._merge_fragments(labels, cabbage_xyz)

    # -1=背景 -> 0=背景
    labels[labels == -1] = 0
    return labels


def main():
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else \
        'work_dirs/softgroup_cabbage/epoch_160.pth'

    logging.basicConfig(level=logging.INFO)
    logger = get_root_logger()

    from munch import Munch
    cfg = Munch.fromDict(yaml.safe_load(open('configs/softgroup/softgroup_cabbage.yaml')))

    # 模型
    model = SoftGroup(**cfg.model).cuda()
    load_checkpoint(checkpoint, logger, model)
    model.eval()
    logger.info(f'Loaded {checkpoint}')

    split = json.load(open('/root/autodl-tmp/evalaute_test/split.json'))
    test_files = split['test']
    logger.info(f'Evaluating {len(test_files)} test files')

    scale = cfg.data.test.voxel_cfg.scale
    spatial_shape_min = cfg.data.test.voxel_cfg.spatial_shape[0]

    file_records = []          # 无 GIDM (SoftGroup 端到端)
    file_records_gidm = []     # 有 GIDM

    for name in test_files:
        ply_path = f'/root/autodl-tmp/evalaute_test/{name}.ply'
        gt_path = f'/root/autodl-tmp/evalaute_test/{name}_gt.txt'

        pcd = o3d.io.read_point_cloud(ply_path)
        xyz = np.asarray(pcd.points, dtype=np.float32)
        if len(pcd.colors) > 0:
            rgb = (np.asarray(pcd.colors) * 2.0 - 1.0).astype(np.float32)
        else:
            rgb = np.ones_like(xyz)

        gt_arr = np.loadtxt(gt_path)
        gt_sem = np.round(gt_arr[:, -2]).astype(np.int64)
        gt_inst = np.round(gt_arr[:, -1]).astype(np.int64)

        # 推理
        t0 = time.time()
        batch = build_single_batch(xyz, rgb, scale, spatial_shape_min)
        with torch.no_grad():
            result = model(batch)
        pred_sem = result['semantic_preds']
        pred_inst = decode_instances(result['pred_instances'], len(xyz))
        runtime = time.time() - t0

        # 提取甘蓝点
        cabbage_mask = pred_sem == 1
        cabbage_xyz = xyz[cabbage_mask]

        # GIDM 后处理 (仅对甘蓝点)
        pred_inst_gidm = pred_inst.copy()
        if cabbage_mask.sum() > 0:
            cabbage_inst_gidm = apply_gidm(pred_inst[cabbage_mask], cabbage_xyz)
            pred_inst_gidm[cabbage_mask] = cabbage_inst_gidm

        n_inst = pred_inst.max()
        n_inst_gidm = pred_inst_gidm.max()
        logger.info(f'{name}: 端到端 {n_inst} 实例 -> +GIDM {n_inst_gidm} 实例 '
                    f'(甘蓝点 {cabbage_mask.sum():,})')

        file_records.append({
            'fname': name + '.ply', 'pred_sem': pred_sem, 'pred_inst': pred_inst,
            'gt_sem': gt_sem, 'gt_inst': gt_inst, 'runtime': runtime,
        })
        file_records_gidm.append({
            'fname': name + '.ply', 'pred_sem': pred_sem, 'pred_inst': pred_inst_gidm,
            'gt_sem': gt_sem, 'gt_inst': gt_inst, 'runtime': runtime,
        })

    # 统一指标评估
    from cabbage_pheno.service.evaluation import evaluate_from_predictions
    res_wo = evaluate_from_predictions(file_records, iou_thresh=0.5, label='SoftGroup-end2end')
    res_gidm = evaluate_from_predictions(file_records_gidm, iou_thresh=0.5, label='SoftGroup-GIDM')

    print('\n========== SoftGroup vs SoftGroup+GIDM ==========')
    print('指标           | SoftGroup端到端 | SoftGroup+GIDM')
    print('---------------|----------------|---------------')
    s0 = res_wo['summary']
    s1 = res_gidm['summary']
    for key, label in [('avg_inst_prec', 'Inst Prec'), ('avg_inst_rec', 'Inst Rec'),
                       ('avg_inst_f1', 'Inst F1'), ('avg_inst_miou', 'Inst mIoU'),
                       ('avg_sem_miou', 'Sem mIoU'), ('mae_count', 'Count MAE')]:
        print(f'{label:14s} | {s0[key]:14.4f} | {s1[key]:14.4f}')
    print('================================================')

    # 保存
    out = '/root/autodl-tmp/output/softgroup_gidm_test.json'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'softgroup_end2end': res_wo, 'softgroup_gidm': res_gidm}, f, indent=2)
    logger.info(f'Results saved to {out}')


if __name__ == '__main__':
    main()
