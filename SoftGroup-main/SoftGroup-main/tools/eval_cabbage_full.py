#!/usr/bin/env python
"""SoftGroup 完整对照实验 (对齐 PointGroup 的未分配点处理流程)。

4 种配置:
  1. end2end        : SoftGroup 端到端 (丢弃未分配点)
  2. expand         : SoftGroup + 未分配点吸附 (pg_expand radius=999, 全部吸附)
  3. gidm           : SoftGroup + GIDM (丢弃未分配点)
  4. expand_gidm    : SoftGroup + 未分配点吸附 + GIDM (完整流程)

未分配点定义: 语义判为甘蓝 (pred_sem==1) 但未被任何 proposal 覆盖 (inst==0) 的点。

用法:
  cd /root/autodl-tmp/SoftGroup-main
  python tools/eval_cabbage_full.py [checkpoint]
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
from munch import Munch

sys.path.insert(0, '/root/autodl-tmp/SoftGroup-main')
sys.path.insert(0, '/root/autodl-tmp')

from softgroup.model import SoftGroup
from softgroup.util import get_root_logger, load_checkpoint, rle_decode
from softgroup.ops import voxelization_idx


def build_single_batch(xyz, rgb, scale, spatial_shape_min):
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
    spatial_shape = np.clip(coords.max(0)[0][1:].numpy() + 1, spatial_shape_min, None)
    batch = {
        'scan_ids': ['scene'],
        'coords': coords, 'batch_idxs': batch_idxs,
        'voxel_coords': voxel_coords, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
        'coords_float': torch.from_numpy(xyz_middle).float(),
        'feats': torch.from_numpy(rgb).float(),
        'semantic_labels': torch.zeros(N, dtype=torch.long),
        'instance_labels': torch.zeros(N, dtype=torch.long),
        'instance_pointnum': torch.tensor([], dtype=torch.int),
        'instance_cls': torch.tensor([], dtype=torch.long),
        'pt_offset_labels': torch.zeros(N, 3).float(),
        'spatial_shape': spatial_shape, 'batch_size': 1,
    }
    return batch


def decode_instances(pred_instances, num_points):
    inst_label = np.zeros(num_points, dtype=np.int64)
    for i, inst in enumerate(pred_instances):
        mask = rle_decode(inst['pred_mask'])
        inst_label[mask == 1] = i + 1
    return inst_label


def expand_unassigned(inst_cabbage, cabbage_xyz, expand_radius=999.0):
    """把未分配点 (inst==0) 吸附到最近的已分配点 (inst>0)。
    expand_radius=999 表示全部吸附 (对齐 PointGroup 的 pg_expand 流程)。
    返回: 吸附后的实例标签 (0=背景, 1..K=实例)
    """
    from scipy.spatial import cKDTree
    labels = inst_cabbage.copy()
    assigned = labels > 0
    if assigned.sum() == 0:
        return labels  # 无已分配点, 无法吸附
    if (~assigned).sum() == 0:
        return labels  # 无未分配点

    pg_pts = cabbage_xyz[assigned]
    pg_labels = labels[assigned]
    un_pts = cabbage_xyz[~assigned]

    tree = cKDTree(pg_pts)
    dists, nn_idx = tree.query(un_pts, k=1)
    nearby = dists < expand_radius
    un_global = np.where(~assigned)[0]
    for i in np.where(nearby)[0]:
        labels[un_global[i]] = pg_labels[nn_idx[i]]
    return labels


def apply_gidm(inst_cabbage, cabbage_xyz):
    """对甘蓝点上的实例标签应用 GIDM (骨架切割 + 碎片回并)。
    inst_cabbage: (N,) 0=背景, 1..K=实例 (可能是欠分割大簇)
    返回: 细化后的实例标签
    """
    from cabbage_pheno.instance.clustering import InstanceClusterer

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
                'abnormal_detect': True,
                'skeleton_cut': True,
                'adaptive_thresh': True,
                'merge_back': True,
            },
            'fragment_voting': {'enable': False, 'discard_threshold': 500, 'merge_threshold': 1500},
        }
    }
    clusterer = InstanceClusterer(cfg)

    labels = inst_cabbage.copy()
    labels[labels == 0] = -1
    labels = clusterer._apply_skeleton_split(labels, cabbage_xyz)
    labels = clusterer._merge_fragments(labels, cabbage_xyz)
    labels[labels == -1] = 0
    return labels


def main():
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else \
        'work_dirs/softgroup_cabbage/epoch_160.pth'

    logging.basicConfig(level=logging.INFO)
    logger = get_root_logger()

    cfg = Munch.fromDict(yaml.safe_load(open('configs/softgroup/softgroup_cabbage.yaml')))
    model = SoftGroup(**cfg.model).cuda()
    load_checkpoint(checkpoint, logger, model)
    model.eval()
    logger.info(f'Loaded {checkpoint}')

    split = json.load(open('/root/autodl-tmp/evalaute_test/split.json'))
    test_files = split['test']
    logger.info(f'Evaluating {len(test_files)} test files')

    scale = cfg.data.test.voxel_cfg.scale
    spatial_shape_min = cfg.data.test.voxel_cfg.spatial_shape[0]

    records = {'end2end': [], 'expand': [], 'gidm': [], 'expand_gidm': []}

    for name in test_files:
        ply_path = f'/root/autodl-tmp/evalaute_test/{name}.ply'
        gt_path = f'/root/autodl-tmp/evalaute_test/{name}_gt.txt'

        pcd = o3d.io.read_point_cloud(ply_path)
        xyz = np.asarray(pcd.points, dtype=np.float32)
        rgb = (np.asarray(pcd.colors) * 2.0 - 1.0).astype(np.float32) if len(pcd.colors) > 0 else np.ones_like(xyz)

        gt_arr = np.loadtxt(gt_path)
        gt_sem = np.round(gt_arr[:, -2]).astype(np.int64)
        gt_inst = np.round(gt_arr[:, -1]).astype(np.int64)

        t0 = time.time()
        batch = build_single_batch(xyz, rgb, scale, spatial_shape_min)
        with torch.no_grad():
            result = model(batch)
        pred_sem = result['semantic_preds']
        pred_inst = decode_instances(result['pred_instances'], len(xyz))
        runtime = time.time() - t0

        cabbage_mask = pred_sem == 1
        cabbage_xyz = xyz[cabbage_mask]
        inst_cabbage = pred_inst[cabbage_mask]  # 0=未分配, 1..K=实例

        # 4 种配置
        inst_wo = inst_cabbage.copy()                                    # 端到端 (丢弃未分配)
        inst_expand = expand_unassigned(inst_cabbage, cabbage_xyz)       # expand
        inst_gidm = apply_gidm(inst_cabbage, cabbage_xyz)                # GIDM
        inst_expand_gidm = apply_gidm(inst_expand, cabbage_xyz)          # expand + GIDM

        def to_full(inst_c):
            full = np.zeros(len(xyz), dtype=np.int64)
            full[cabbage_mask] = inst_c
            return full

        n0 = inst_wo.max(); n1 = inst_expand.max(); n2 = inst_gidm.max(); n3 = inst_expand_gidm.max()
        logger.info(f'{name}: end2end={n0} expand={n1} gidm={n2} expand_gidm={n3} '
                    f'(甘蓝 {cabbage_mask.sum():,}, 未分配 {(inst_cabbage==0).sum():,})')

        base = {'fname': name + '.ply', 'pred_sem': pred_sem, 'gt_sem': gt_sem, 'gt_inst': gt_inst, 'runtime': runtime}
        records['end2end'].append(dict(base, pred_inst=to_full(inst_wo)))
        records['expand'].append(dict(base, pred_inst=to_full(inst_expand)))
        records['gidm'].append(dict(base, pred_inst=to_full(inst_gidm)))
        records['expand_gidm'].append(dict(base, pred_inst=to_full(inst_expand_gidm)))

    from cabbage_pheno.service.evaluation import evaluate_from_predictions
    results = {}
    for k, recs in records.items():
        results[k] = evaluate_from_predictions(recs, iou_thresh=0.5, label=k)

    print('\n========== SoftGroup 4 种配置对比 ==========')
    print('指标           | end2end | expand  | gidm    | expand+gidm')
    print('---------------|---------|---------|---------|-----------')
    for key, label in [('avg_inst_prec', 'Inst Prec'), ('avg_inst_rec', 'Inst Rec'),
                       ('avg_inst_f1', 'Inst F1'), ('avg_inst_miou', 'Inst mIoU'),
                       ('avg_sem_miou', 'Sem mIoU'), ('mae_count', 'Count MAE')]:
        row = f'{label:14s} |'
        for k in ['end2end', 'expand', 'gidm', 'expand_gidm']:
            row += f' {results[k]["summary"][key]:7.4f} |'
        print(row)
    print('==============================================')

    out = '/root/autodl-tmp/output/softgroup_full_test.json'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'Results saved to {out}')


if __name__ == '__main__':
    main()
