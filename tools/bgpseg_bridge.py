#!/usr/bin/env python3
"""
BGPSeg 桥接脚本：经过与 Cabbage 相同的预处理后，用 BGPSeg 模型推理，
再通过 Cabbage 统一的 evaluate_from_predictions 计算指标。

用法:
  cd /home/stevensen/Cabbage
  python tools/bgpseg_bridge.py \
      --data e_data/train \
      --config configs/default.yaml \
      --output output/bgpseg_result.json
"""
import argparse
import json
import os
import sys
from time import time

import numpy as np
import torch

# ---- BGPSeg imports ----
BGPSEG_ROOT = '/home/stevensen/BGPSeg-main'
sys.path.insert(0, BGPSEG_ROOT)

from model.BGPSeg import BoundaryPredictor, BGPSeg
from util.Cabbage import generate_knn_edges
from util.loss_util import mean_shift_gpu
from util.cabbage_cluster import boundary_guided_primitive_clustering

# ---- Cabbage evaluation imports ----
CABBAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CABBAGE_ROOT)

from cabbage_pheno.io import read_point_cloud
from cabbage_pheno.service.pipeline import collect_input_files, preprocess_point_cloud
from cabbage_pheno.service.evaluation import (
    align_gt_to_pred,
    load_config,
    load_ground_truth,
    evaluate_from_predictions,
)

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def inference_bgpseg_on_pcd(pcd, boundary_model, model, device,
                             bandwidth=1.31, cluster_threshold=0.99, cluster_batch=50):
    """在 Open3D PointCloud 上跑 BGPSeg 推理 → semantic + instance labels (N,)."""
    xyz = np.asarray(pcd.points, dtype=np.float32)
    rgb = np.asarray(pcd.colors, dtype=np.float32) if len(pcd.colors) > 0 else np.ones_like(xyz) * 0.5
    N = len(xyz)

    c = torch.FloatTensor(xyz).to(device)
    f = torch.FloatTensor(rgb).to(device)
    o = torch.IntTensor([N]).to(device)
    c = c - c.min(0)[0]
    knn = generate_knn_edges(c.cpu().numpy(), k=16)
    knn_g = torch.IntTensor(knn).to(device)

    with torch.no_grad():
        bp = boundary_model([c, f, o])
        emb, tp = model([c, f, o], knn_g,
                        boundary_gt=torch.zeros(N, dtype=torch.long).to(device),
                        boundary_pred=bp, is_train=False)

    semantic = tp.max(1)[1].cpu().numpy()
    embedding = emb.cpu().numpy()
    bprob = torch.softmax(bp, dim=1)[:, 1].cpu().numpy()

    cabbage_idx = np.where(semantic == 1)[0]
    n_cab = len(cabbage_idx)
    instance = np.full(N, -1, dtype=np.int32)

    if n_cab > 0:
        emb_all = torch.FloatTensor(embedding[cabbage_idx]).to(device)
        off_all = torch.IntTensor([n_cab]).to(device)
        _, shifted = mean_shift_gpu(emb_all, off_all, bandwidth=bandwidth, batch_size=cluster_batch)
        shifted_np = shifted.cpu().numpy().astype('float32')

        b_pred = (bprob > 0.5).astype('int32')
        instance_sub = np.full(n_cab, -1, dtype=np.int32)
        mask_sub = np.zeros(n_cab, dtype=np.int32)
        boundary_guided_primitive_clustering(
            xyz[cabbage_idx], shifted_np, b_pred[cabbage_idx],
            instance_sub, mask_sub, threshold=cluster_threshold)
        valid = instance_sub >= 0
        if valid.sum() > 0:
            for ni, oi in enumerate(np.unique(instance_sub[valid]), 1):
                instance[cabbage_idx[instance_sub == oi]] = ni

    n_inst = len(np.unique(instance[instance > 0]))
    return semantic, instance, n_inst


def main():
    parser = argparse.ArgumentParser(description='BGPSeg bridge → Cabbage evaluation (shared preprocessing)')
    parser.add_argument('--data', default='e_data/train', help='Directory containing PLY + _gt.txt files')
    parser.add_argument('--config', default='configs/default.yaml', help='Cabbage config for preprocessing settings')
    parser.add_argument('--model', default=f'{BGPSEG_ROOT}/checkpoints/cabbage_best_v2.pth')
    parser.add_argument('--boundary', default=f'{BGPSEG_ROOT}/checkpoints/boundary_best.pth')
    parser.add_argument('--output', default='output/bgpseg_result.json', help='JSON result file')
    parser.add_argument('--bandwidth', type=float, default=1.31)
    parser.add_argument('--cluster-threshold', type=float, default=0.99)
    parser.add_argument('--iou-thresh', type=float, default=0.5, help='Instance matching IoU threshold')
    parser.add_argument('--gt-label-col', type=int, default=-2)
    parser.add_argument('--gt-instance-col', type=int, default=-1)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    cfg = load_config(args.config)
    seg_method = cfg.get('segmentation', {}).get('method', 'hybrid')
    print(f'Config: {args.config}  (seg_method={seg_method})')

    print('Loading BGPSeg models...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    boundary_model = BoundaryPredictor(in_channels=6).to(device).eval()
    model = BGPSeg(in_channels=7, num_classes=2).to(device).eval()

    for path, mdl in [(args.boundary, boundary_model), (args.model, model)]:
        if os.path.exists(path):
            ck = torch.load(path, map_location='cpu', weights_only=False)
            sd = {k.replace('module.', ''): v for k, v in ck['state_dict'].items()}
            mdl.load_state_dict(sd, strict=False)
            print(f'  Loaded: {path}')
        else:
            print(f'  WARNING: not found: {path}')

    input_files = collect_input_files(args.data)
    print(f'Found {len(input_files)} input files')

    file_records = []

    for file_path in input_files:
        name = os.path.splitext(os.path.basename(file_path))[0]
        print(f'  Processing {name}...')
        t_start = time()

        # ---- 1. 读取原始点云 ----
        pcd = read_point_cloud(file_path)
        if pcd is None:
            print(f'    SKIP: cannot read PLY')
            continue

        # ---- 2. 加载 GT (原始点) ----
        gt_points_raw, gt_sem_raw, gt_inst_raw = load_ground_truth(
            file_path, label_col=args.gt_label_col,
            instance_col=args.gt_instance_col, binarize_sem=True)
        if gt_inst_raw is None:
            print(f'    SKIP: no GT')
            continue

        # ---- 3. 与 Cabbage 相同的预处理 (SOR + 地面去除) ----
        preprocess_result = preprocess_point_cloud(pcd, cfg)
        points_clean = preprocess_result.points_clean       # SOR + 地面去除后的点
        non_ground_pcd = preprocess_result.non_ground_pcd   # 同上

        # ---- 4. 构建 KDTree: clean 点 → 非地面点索引 ----
        ng_indices = []
        if HAS_SCIPY and len(non_ground_pcd.points) > 0:
            tree = cKDTree(points_clean)
            _, ng_indices = tree.query(np.asarray(non_ground_pcd.points), k=1)

        # ---- 5. BGPSeg 推理 (在 non_ground_pcd 上) ----
        pred_sem_ng, pred_inst_ng, n_inst = inference_bgpseg_on_pcd(
            non_ground_pcd, boundary_model, model, device,
            bandwidth=args.bandwidth, cluster_threshold=args.cluster_threshold)

        # ---- 6. 映射回 points_clean 全集 ----
        full_semantic_pred = np.zeros(len(points_clean), dtype=int)
        full_instance_pred = np.full(len(points_clean), -1, dtype=int)

        if len(ng_indices) > 0:
            # BGPSeg 语义: 1=cabbage → 映射到 clean
            cabbage_mask_on_ng = pred_sem_ng == 1
            cab_to_clean = np.array(ng_indices)[cabbage_mask_on_ng]
            full_semantic_pred[cab_to_clean] = 1

            # BGPSeg 实例 → 映射到 clean
            inst_valid = pred_inst_ng >= 0
            inst_to_clean = np.array(ng_indices)[inst_valid]
            full_instance_pred[inst_to_clean] = pred_inst_ng[inst_valid]

        # ---- 7. GT 对齐 (原始点 → 清洗后的点) ----
        gt_sem_clean, gt_inst_clean = align_gt_to_pred(
            gt_points_raw, gt_sem_raw, gt_inst_raw, points_clean)
        if gt_inst_clean is not None:
            gt_inst_clean = gt_inst_clean.copy()
            gt_inst_clean[gt_inst_clean <= 0] = 0

        runtime = time() - t_start
        print(f'    {n_inst} instances, {runtime:.1f}s')

        file_records.append({
            'fname': name,
            'pred_sem': full_semantic_pred,
            'pred_inst': full_instance_pred,
            'gt_sem': gt_sem_clean,
            'gt_inst': gt_inst_clean,
            'runtime': runtime,
        })

    # ---- 8. 统一指标计算 ----
    result = evaluate_from_predictions(file_records, iou_thresh=args.iou_thresh, label='BGPSeg')

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    summary = result['summary']
    print(f"\nBGPSeg result ({len(file_records)} files):")
    print(f"  Semantic mIoU:  {summary.get('avg_sem_miou', 0):.4f}")
    print(f"  Instance Prec:  {summary.get('avg_inst_prec', 0):.4f}")
    print(f"  Instance Rec:   {summary.get('avg_inst_rec', 0):.4f}")
    print(f"  Instance F1:    {summary.get('avg_inst_f1', 0):.4f}")
    print(f"  Instance mIoU:  {summary.get('avg_inst_miou', 0):.4f}")
    print(f"  MAE Count:      {summary.get('mae_count', 0):.2f}")
    print(f"  Avg Runtime:    {summary.get('avg_runtime', 0):.1f}s")
    print(f'Saved → {args.output}')


if __name__ == '__main__':
    main()
