"""逐株 dump 表型参数对比 (预测 vs 真值).

对 val/test 集每个匹配的实例对, 计算并输出 alpha-shape 体积、体素体积、
株高、冠幅等表型参数的预测值 / 真值 / 绝对误差 / 相对误差。

复用 main.py 与 evaluation.py 的完整分割流程, 保证与 evaluate.py 结果一致。
用法:
  python tools/dump_traits_per_plant.py --input evalaute_test --config configs/default.yaml --split val
"""
import argparse
import json
import os
import sys

import numpy as np
import open3d as o3d
import yaml
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cabbage_pheno.io import read_point_cloud
from cabbage_pheno.instance import InstanceClusterer
from cabbage_pheno.service.pipeline import collect_input_files, preprocess_point_cloud
from cabbage_pheno.service.evaluation import (
    align_gt_to_pred,
    get_instance_matches,
    load_ground_truth,
    TRAIT_MAE_FIELDS,
)
from cabbage_pheno.traits import TraitCalculator
from main import pointgroup_segment_blocks


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def process_file(file_path, cfg):
    """复现 evaluate.py 的完整流程, 返回逐株匹配对的表型明细."""
    seg_method = cfg.get('segmentation', {}).get('method', 'hybrid')
    if seg_method == 'clustering':
        return process_file_clustering(file_path, cfg)
    pcd = read_point_cloud(file_path)
    if pcd is None:
        return None

    gt_points_raw, gt_sem_raw, gt_inst_raw = load_ground_truth(file_path)
    if gt_inst_raw is None:
        return None

    # 确定性
    seed = cfg.get('pipeline', {}).get('seed', None)
    if seed is not None:
        np.random.seed(int(seed))
        try:
            import torch as _t
            _t.manual_seed(int(seed))
            _t.cuda.manual_seed_all(int(seed))
        except Exception:
            pass

    # 预处理
    preprocess_result = preprocess_point_cloud(pcd, cfg)
    points_clean = preprocess_result.points_clean
    non_ground_pcd = preprocess_result.non_ground_pcd
    plane_model = preprocess_result.plane_model

    # 语义 + 实例提案 (hybrid + nms)
    cabbage_pcd, inst_init = pointgroup_segment_blocks(non_ground_pcd, cfg, 'hybrid', selection='nms')
    cabbage_pts = np.asarray(cabbage_pcd.points)

    # 映射 cabbage -> clean (最近邻)
    tree_clean = cKDTree(points_clean)
    _, ng_indices = tree_clean.query(np.asarray(non_ground_pcd.points), k=1)
    cab_to_clean = None
    if len(cabbage_pts) > 0:
        _, cab_to_ng = cKDTree(np.asarray(non_ground_pcd.points)).query(cabbage_pts, k=1)
        cab_to_clean = np.array(ng_indices)[cab_to_ng]

    clusterer = InstanceClusterer(cfg)
    full_instance_pred = np.zeros(len(points_clean), dtype=int) - 1

    if len(cabbage_pts) > 0:
        # ---- Stage2: PG 保留 + 未分配点聚类 ----
        assigned_mask = inst_init > 0
        if (~assigned_mask).sum() > 100:
            expand_radius = cfg.get('instance', {}).get('pg_expand', {}).get('radius', 0.05)
            un_pts_all = cabbage_pts[~assigned_mask]
            pg_pts = cabbage_pts[assigned_mask]
            pg_labels = inst_init[assigned_mask]
            if len(pg_pts) > 0:
                dists, nn_idx = cKDTree(pg_pts).query(un_pts_all, k=1)
                nearby = dists < expand_radius
                un_global = np.where(~assigned_mask)[0]
                for i in np.where(nearby)[0]:
                    inst_init[un_global[i]] = pg_labels[nn_idx[i]]
                assigned_mask = inst_init > 0

            un_pts = cabbage_pts[~assigned_mask]
            prefilter = cfg.get('instance', {}).get('precluster_filter', {})
            dense_mask = np.ones(len(un_pts), dtype=bool)
            if prefilter.get('enable', True) and len(un_pts) > 0:
                ror_nb = prefilter.get('min_points', 10)
                ror_radius = prefilter.get('eps', 0.03)
                tmp_pcd = o3d.geometry.PointCloud()
                tmp_pcd.points = o3d.utility.Vector3dVector(un_pts)
                _, dense_idx = tmp_pcd.remove_radius_outlier(nb_points=ror_nb, radius=ror_radius)
                dense_idx = np.asarray(dense_idx, dtype=np.int64)
                dense_mask = np.zeros(len(un_pts), dtype=bool)
                if len(dense_idx) > 0:
                    dense_mask[dense_idx] = True
                un_pts = un_pts[dense_mask]

            if len(un_pts) > 0:
                un_pcd = o3d.geometry.PointCloud()
                un_pcd.points = o3d.utility.Vector3dVector(un_pts)
                try:
                    un_labels, _ = clusterer.cluster(un_pcd)
                except Exception:
                    un_labels = np.array([], dtype=np.int64)
            else:
                un_labels = np.array([], dtype=np.int64)

            dense_cfg = prefilter.get('post_density_check', {})
            if dense_cfg.get('enable', True):
                min_lin = dense_cfg.get('min_linear_density', 500)
                min_abs = dense_cfg.get('min_pts_absolute', 100)
                valid_clusters = set()
                for ul in sorted(set(un_labels)):
                    if ul < 0:
                        continue
                    cl_pts = un_pts[un_labels == ul]
                    if len(cl_pts) < min_abs:
                        continue
                    span = np.max(cl_pts, axis=0) - np.min(cl_pts, axis=0)
                    max_span = np.max(span)
                    if max_span > 0 and len(cl_pts) / max_span >= min_lin:
                        valid_clusters.add(ul)
            else:
                valid_clusters = set(ul for ul in set(un_labels) if ul >= 0)

            labels_s2 = np.full(len(cabbage_pts), -1, dtype=np.int64)
            labels_s2[assigned_mask] = inst_init[assigned_mask]
            next_id = inst_init.max() + 1
            unassigned_global = np.where(~assigned_mask)[0]
            dense_global = unassigned_global[dense_mask]
            for ul in sorted(valid_clusters):
                labels_s2[dense_global[un_labels == ul]] = next_id
                next_id += 1
        else:
            labels_s2 = inst_init.copy()

        # ---- Stage3: 骨架拆分 + 碎片回并 ----
        labels_s3 = labels_s2.copy()
        if cfg.get('instance', {}).get('pca_split', {}).get('enable', False):
            if clusterer.skeleton_cut:
                try:
                    labels_s3 = clusterer._apply_skeleton_split(labels_s3, cabbage_pts)
                except Exception:
                    pass
            if clusterer.merge_back:
                try:
                    labels_s3 = clusterer._merge_fragments(labels_s3, cabbage_pts)
                except Exception:
                    pass

        # ---- Stage4: 碎片清理 + 噪声重归入 + 重编号 ----
        discard_thresh = cfg.get('instance', {}).get('fragment_voting', {}).get('discard_threshold', 300)
        min_pts = cfg.get('instance', {}).get('min_cluster_points', 2000)

        uniq, cnts = np.unique(labels_s3, return_counts=True)
        for lbl, c in zip(uniq, cnts):
            if lbl >= 0 and c < discard_thresh:
                labels_s3[labels_s3 == lbl] = -1

        if cfg.get('instance', {}).get('fragment_voting', {}).get('enable', False):
            try:
                labels_s3 = clusterer._cleanup_tiny_fragments(labels_s3, cabbage_pts)
            except Exception:
                pass

        uniq, cnts = np.unique(labels_s3, return_counts=True)
        for lbl, c in zip(uniq, cnts):
            if lbl >= 0 and c < min_pts:
                labels_s3[labels_s3 == lbl] = -1

        noise_mask = labels_s3 == -1
        if noise_mask.sum() > 0 and labels_s3.max() >= 0:
            valid_mask = labels_s3 >= 0
            tree = cKDTree(cabbage_pts[valid_mask])
            dists, nn_idx = tree.query(cabbage_pts[noise_mask], k=1)
            reassign_radius = cfg.get('instance', {}).get('noise_reassign_radius', 0.10)
            nearby = dists < reassign_radius
            noise_idx = np.where(noise_mask)[0][nearby]
            labels_s3[noise_idx] = labels_s3[valid_mask][nn_idx[nearby]]

        valid = labels_s3 >= 0
        new_labels = np.full_like(labels_s3, -1)
        next_id = 1
        for old_lbl in sorted(set(labels_s3[valid].tolist())):
            new_labels[labels_s3 == old_lbl] = next_id
            next_id += 1
        inst_labels = new_labels

        if cab_to_clean is not None:
            full_instance_pred[cab_to_clean] = inst_labels

    # 对齐 GT 到 pred
    gt_sem_clean, gt_inst_clean = align_gt_to_pred(gt_points_raw, gt_sem_raw, gt_inst_raw, points_clean)
    if gt_inst_clean is not None:
        gt_inst_clean = gt_inst_clean.copy()
        gt_inst_clean[gt_inst_clean <= 0] = 0

    iou_thresh = cfg.get('evaluation', {}).get('iou_thresh', 0.5)
    matches, n_pred, n_gt = get_instance_matches(full_instance_pred, gt_inst_clean, iou_thresh)

    return {
        'file': os.path.basename(file_path),
        'points_clean': points_clean,
        'full_instance_pred': full_instance_pred,
        'gt_points_raw': gt_points_raw,
        'gt_inst_raw': gt_inst_raw,
        'plane_model': plane_model,
        'matches': matches,
    }


_RLA_MODEL = None


def _get_randla_model():
    """懒加载 RandLA-Net 模型 (只加载一次, 跨文件复用)."""
    global _RLA_MODEL
    if _RLA_MODEL is None:
        import torch as _torch
        proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, os.path.join(proj_root, 'RandLANet_Ours'))
        from randlanet import RandLANet as RLAModel
        _RLA_MODEL = RLAModel(d_in=6, num_classes=2).cuda()
        ckpt = os.path.join(proj_root, 'RandLANet_Ours', 'exp', 'randlanet', 'best.pth')
        state = _torch.load(ckpt, map_location='cuda')
        _RLA_MODEL.load_state_dict(state['model'])
        _RLA_MODEL.eval()
    return _RLA_MODEL


def process_file_clustering(file_path, cfg):
    """clustering 模式 (RandLA/PointGroup 语义 + 传统聚类; GIDM 已在 clusterer.cluster 内部)."""
    pcd = read_point_cloud(file_path)
    if pcd is None:
        return None

    gt_points_raw, gt_sem_raw, gt_inst_raw = load_ground_truth(file_path)
    if gt_inst_raw is None:
        return None

    # 确定性
    seed = cfg.get('pipeline', {}).get('seed', None)
    if seed is not None:
        np.random.seed(int(seed))
        try:
            import torch as _t
            _t.manual_seed(int(seed))
            _t.cuda.manual_seed_all(int(seed))
        except Exception:
            pass

    # 预处理
    preprocess_result = preprocess_point_cloud(pcd, cfg)
    points_clean = preprocess_result.points_clean
    non_ground_pcd = preprocess_result.non_ground_pcd
    plane_model = preprocess_result.plane_model

    backbone = cfg.get('segmentation', {}).get('backbone', 'pointnet2')

    # ---- 语义分割 ----
    if backbone == 'randlanet':
        import torch as _torch
        rla_model = _get_randla_model()
        xyz = np.asarray(non_ground_pcd.points, dtype=np.float32)
        rgb_orig = np.asarray(non_ground_pcd.colors, dtype=np.float32) if len(non_ground_pcd.colors) > 0 else np.ones_like(xyz)
        rgb = (rgb_orig * 2 - 1).astype(np.float32)
        xyz_shift = xyz - xyz.min(0)
        CHUNK = 120000
        N = len(xyz)
        mask = np.zeros(N, dtype=bool)
        for start in range(0, N, CHUNK):
            end = min(start + CHUNK, N)
            with _torch.no_grad():
                pred = rla_model(_torch.from_numpy(xyz_shift[start:end]).unsqueeze(0).cuda(),
                                 _torch.from_numpy(rgb[start:end]).unsqueeze(0).cuda())
            mask[start:end] = pred[0].max(1)[1].cpu().numpy() == 1
        cabbage_pts = xyz[mask]
        cabbage_pcd = o3d.geometry.PointCloud()
        cabbage_pcd.points = o3d.utility.Vector3dVector(cabbage_pts)
        cabbage_pcd.colors = o3d.utility.Vector3dVector(rgb_orig[mask])
    else:
        # PointGroup 语义 (clustering 模式)
        from main import pointgroup_segment
        cabbage_pcd, _ = pointgroup_segment(non_ground_pcd, cfg, 'clustering')
        cabbage_pts = np.asarray(cabbage_pcd.points)

    # 映射 cabbage -> clean (最近邻)
    tree_clean = cKDTree(points_clean)
    _, ng_indices = tree_clean.query(np.asarray(non_ground_pcd.points), k=1)
    cab_to_clean = None
    if len(cabbage_pts) > 0:
        _, cab_to_ng = cKDTree(np.asarray(non_ground_pcd.points)).query(cabbage_pts, k=1)
        cab_to_clean = np.array(ng_indices)[cab_to_ng]

    clusterer = InstanceClusterer(cfg)
    full_instance_pred = np.zeros(len(points_clean), dtype=int) - 1

    if len(cabbage_pts) > 0:
        # clustering: clusterer.cluster() 内部已含 GIDM (骨架切割 + 碎片回并 + 邻域投票)
        un_pcd = o3d.geometry.PointCloud()
        un_pcd.points = o3d.utility.Vector3dVector(cabbage_pts)
        labels_s3, _ = clusterer.cluster(un_pcd)

        # ---- Stage4: 碎片清理 + 噪声重归入 + 重编号 (与 evaluate.py 一致) ----
        discard_thresh = cfg.get('instance', {}).get('fragment_voting', {}).get('discard_threshold', 300)
        min_pts = cfg.get('instance', {}).get('min_cluster_points', 2000)

        uniq, cnts = np.unique(labels_s3, return_counts=True)
        for lbl, c in zip(uniq, cnts):
            if lbl >= 0 and c < discard_thresh:
                labels_s3[labels_s3 == lbl] = -1

        if cfg.get('instance', {}).get('fragment_voting', {}).get('enable', False):
            try:
                labels_s3 = clusterer._cleanup_tiny_fragments(labels_s3, cabbage_pts)
            except Exception:
                pass

        uniq, cnts = np.unique(labels_s3, return_counts=True)
        for lbl, c in zip(uniq, cnts):
            if lbl >= 0 and c < min_pts:
                labels_s3[labels_s3 == lbl] = -1

        noise_mask = labels_s3 == -1
        if noise_mask.sum() > 0 and labels_s3.max() >= 0:
            valid_mask = labels_s3 >= 0
            tree = cKDTree(cabbage_pts[valid_mask])
            dists, nn_idx = tree.query(cabbage_pts[noise_mask], k=1)
            reassign_radius = cfg.get('instance', {}).get('noise_reassign_radius', 0.10)
            nearby = dists < reassign_radius
            noise_idx = np.where(noise_mask)[0][nearby]
            labels_s3[noise_idx] = labels_s3[valid_mask][nn_idx[nearby]]

        valid = labels_s3 >= 0
        new_labels = np.full_like(labels_s3, -1)
        next_id = 1
        for old_lbl in sorted(set(labels_s3[valid].tolist())):
            new_labels[labels_s3 == old_lbl] = next_id
            next_id += 1
        inst_labels = new_labels

        if cab_to_clean is not None:
            full_instance_pred[cab_to_clean] = inst_labels

    # 对齐 GT 到 pred
    gt_sem_clean, gt_inst_clean = align_gt_to_pred(gt_points_raw, gt_sem_raw, gt_inst_raw, points_clean)
    if gt_inst_clean is not None:
        gt_inst_clean = gt_inst_clean.copy()
        gt_inst_clean[gt_inst_clean <= 0] = 0

    iou_thresh = cfg.get('evaluation', {}).get('iou_thresh', 0.5)
    matches, n_pred, n_gt = get_instance_matches(full_instance_pred, gt_inst_clean, iou_thresh)

    return {
        'file': os.path.basename(file_path),
        'points_clean': points_clean,
        'full_instance_pred': full_instance_pred,
        'gt_points_raw': gt_points_raw,
        'gt_inst_raw': gt_inst_raw,
        'plane_model': plane_model,
        'matches': matches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--split', default=None, choices=['val', 'test', 'all'])
    parser.add_argument('--output', default=None,
                        help='CSV 输出路径 (默认 output/traits_per_plant_{split}.csv)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    trait_calc = TraitCalculator(cfg)
    input_files = collect_input_files(args.input)

    if args.split and args.split != 'all':
        split_path = os.path.join(args.input, 'split.json')
        if not os.path.exists(split_path):
            split_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      'evalaute_test', 'split.json')
        with open(split_path) as f:
            _split = json.load(f)
        allowed = set(_split.get(args.split, []))
        input_files = [f for f in input_files
                       if os.path.splitext(os.path.basename(f))[0] in allowed]

    rows = []
    for fp in input_files:
        print(f'--- {os.path.basename(fp)} ---', flush=True)
        res = process_file(fp, cfg)
        if res is None:
            continue
        for pid, gid, iou in res['matches']:
            p_mask = res['full_instance_pred'] == pid
            pts_p = res['points_clean'][p_mask]
            g_mask = res['gt_inst_raw'] == gid
            pts_g = res['gt_points_raw'][g_mask]
            if len(pts_p) < 4 or len(pts_g) < 4:
                continue
            pcd_p = o3d.geometry.PointCloud()
            pcd_p.points = o3d.utility.Vector3dVector(pts_p)
            pcd_g = o3d.geometry.PointCloud()
            pcd_g.points = o3d.utility.Vector3dVector(pts_g)
            t_p = trait_calc.calculate_traits(pcd_p, pid, res['plane_model'])
            t_g = trait_calc.calculate_traits(pcd_g, gid, res['plane_model'])
            if not (t_p and t_g):
                continue
            row = {
                'file': res['file'],
                'pred_id': int(pid),
                'gt_id': int(gid),
                'iou': float(iou),
            }
            for key, _, _ in TRAIT_MAE_FIELDS:
                vp = t_p.get(key, None)
                vg = t_g.get(key, None)
                row[f'pred_{key}'] = None if vp is None else float(vp)
                row[f'gt_{key}'] = None if vg is None else float(vg)
                if vp is not None and vg is not None:
                    row[f'err_{key}'] = float(abs(vp - vg))
                    row[f'rel_{key}_pct'] = float(abs(vp - vg) / abs(vg) * 100.0) if abs(vg) > 1e-9 else None
                else:
                    row[f'err_{key}'] = None
                    row[f'rel_{key}_pct'] = None
            rows.append(row)

    import pandas as pd
    df = pd.DataFrame(rows)
    out = args.output or os.path.join('output', f'traits_per_plant_{args.split or "all"}.csv')
    df.round(4).to_csv(out, index=False)
    print(f'\nSaved {len(df)} 对匹配实例 -> {out}')
    print(df[['file', 'pred_id', 'gt_id', 'iou',
              'gt_volume_alpha_cm3', 'pred_volume_alpha_cm3',
              'err_volume_alpha_cm3', 'rel_volume_alpha_cm3_pct']].round(2).to_string())


if __name__ == '__main__':
    main()
