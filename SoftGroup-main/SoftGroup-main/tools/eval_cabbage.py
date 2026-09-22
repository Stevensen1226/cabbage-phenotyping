#!/usr/bin/env python
"""完整 SoftGroup 端到端推理 + 评估脚本 (甘蓝)。

对 evalaute_test 的 test 划分 (14 块田地) 推理, 输出语义 + 实例分割,
并用项目统一指标 (Prec/Rec/F1/mIoU/MAE) 评估。

用法:
  cd /root/autodl-tmp/SoftGroup-main
  python tools/eval_cabbage.py [checkpoint]
"""
import os
import sys
import json
import time
import glob
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
    """构建 SoftGroup forward_test 所需的单个 batch (不做旋转/增强)。"""
    N = xyz.shape[0]
    xyz_middle = xyz.copy()                      # 原始坐标 (float32)
    xyz_voxel = (xyz_middle * scale)             # 缩放到体素空间
    xyz_voxel = xyz_voxel - xyz_voxel.min(0)     # 平移到原点
    coords = torch.cat([
        torch.zeros(N, 1, dtype=torch.long),
        torch.from_numpy(xyz_voxel).long()
    ], dim=1)                                    # (N, 1+3), 第0维 = batch_idx

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
    """把 SoftGroup 输出的 RLE 实例列表解码为点级 instance label (0=背景, 1+=实例)。"""
    inst_label = np.zeros(num_points, dtype=np.int64)
    for i, inst in enumerate(pred_instances):
        mask = rle_decode(inst['pred_mask'])
        inst_label[mask == 1] = i + 1
    return inst_label


def main():
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else \
        'work_dirs/softgroup_cabbage/epoch_160.pth'

    logging.basicConfig(level=logging.INFO)
    logger = get_root_logger()

    cfg = Munch.fromDict(yaml.safe_load(
        open('configs/softgroup/softgroup_cabbage.yaml')))

    # 模型
    model = SoftGroup(**cfg.model).cuda()
    load_checkpoint(checkpoint, logger, model)
    model.eval()
    logger.info(f'Loaded {checkpoint}')

    # test 划分
    split = json.load(open('/root/autodl-tmp/evalaute_test/split.json'))
    test_files = split['test']
    logger.info(f'Evaluating {len(test_files)} test files: {test_files}')

    file_records = []
    scale = cfg.data.test.voxel_cfg.scale
    spatial_shape_min = cfg.data.test.voxel_cfg.spatial_shape[0]

    for name in test_files:
        ply_path = f'/root/autodl-tmp/evalaute_test/{name}.ply'
        gt_path = f'/root/autodl-tmp/evalaute_test/{name}_gt.txt'

        # 读点云
        pcd = o3d.io.read_point_cloud(ply_path)
        xyz = np.asarray(pcd.points, dtype=np.float32)
        if len(pcd.colors) > 0:
            rgb = (np.asarray(pcd.colors) * 2.0 - 1.0).astype(np.float32)
        else:
            rgb = np.ones_like(xyz)

        # 读 GT
        gt_arr = np.loadtxt(gt_path)
        gt_sem = np.round(gt_arr[:, -2]).astype(np.int64)
        gt_inst = np.round(gt_arr[:, -1]).astype(np.int64)

        # 推理
        t0 = time.time()
        batch = build_single_batch(xyz, rgb, scale, spatial_shape_min)
        with torch.no_grad():
            result = model(batch)
        pred_sem = result['semantic_preds']            # (N,)
        pred_inst = decode_instances(result['pred_instances'], len(xyz))
        runtime = time.time() - t0

        logger.info(f'{name}: sem={pred_sem.sum():,}/{len(xyz):,} '
                    f'inst={len(result["pred_instances"])} runtime={runtime:.1f}s')

        file_records.append({
            'fname': name + '.ply',
            'pred_sem': pred_sem,
            'pred_inst': pred_inst,
            'gt_sem': gt_sem,
            'gt_inst': gt_inst,
            'runtime': runtime,
        })

    # 统一指标评估 (iou_thresh 0.5, 与项目一致)
    from cabbage_pheno.service.evaluation import evaluate_from_predictions
    result = evaluate_from_predictions(file_records, iou_thresh=0.5, label='SoftGroup-end2end')

    print('\n========== SoftGroup 端到端评估结果 ==========')
    s = result['summary']
    print(f"  Sem mIoU:       {s['avg_sem_miou']:.4f}")
    print(f"  Inst Prec:      {s['avg_inst_prec']:.4f}")
    print(f"  Inst Rec:       {s['avg_inst_rec']:.4f}")
    print(f"  Inst F1:        {s['avg_inst_f1']:.4f}")
    print(f"  Inst mIoU:      {s['avg_inst_miou']:.4f}")
    print(f"  Count MAE:      {s['mae_count']:.4f}")
    print(f"  Avg runtime:    {s['avg_runtime']:.1f}s")
    print('===============================================')

    # 保存结果
    out_path = '/root/autodl-tmp/output/softgroup_end2end_test.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f'Results saved to {out_path}')


if __name__ == '__main__':
    main()
