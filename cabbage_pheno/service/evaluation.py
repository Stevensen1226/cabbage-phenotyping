import argparse
import logging
import os
import sys
import time
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import open3d as o3d
import yaml

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from cabbage_pheno.io import read_point_cloud
from cabbage_pheno.instance import InstanceClusterer
from cabbage_pheno.service.pipeline import collect_input_files, preprocess_point_cloud
from cabbage_pheno.traits import TraitCalculator

logger = logging.getLogger("Evaluator")


# 表型字段对比配置: (字段key, 中文名, 单位)
# 评估时对每个匹配实例对计算预测 vs 真值的绝对/相对误差
TRAIT_MAE_FIELDS = [
    ('height_cm', '株高', 'cm'),
    ('height_max_cm', '最大株高', 'cm'),
    ('crown_diameter_cm', '冠幅-等面积圆', 'cm'),
    ('crown_diameter_hull_cm', '冠幅-凸包直径', 'cm'),
    ('crown_diameter_circle_cm', '冠幅-最小外接圆', 'cm'),
    ('crown_diameter_ew_cm', '冠幅-东西向', 'cm'),
    ('crown_diameter_ns_cm', '冠幅-南北向', 'cm'),
    ('crown_diameter_mean_cm', '冠幅-均值', 'cm'),
    ('volume_alpha_cm3', 'alpha体积', 'cm3'),
    ('compactness', '紧实度', '-'),
]


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def tqdm(iterable, desc=""):
    try:
        from tqdm import tqdm as _tqdm
        return _tqdm(iterable, desc=desc)
    except ImportError:
        logger.info(f"Start: {desc}")
        return iterable


def load_ground_truth(file_path, label_col=-2, instance_col=-1, binarize_sem=True):
    dir_name = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    candidates = [
        os.path.join(dir_name, base_name + ".txt"),
        os.path.join(dir_name, base_name + "_gt.txt"),
        file_path.replace(".ply", ".txt"),
    ]

    gt_path = None
    for p in candidates:
        if os.path.exists(p) and os.path.isfile(p):
            gt_path = p
            break

    if gt_path is None:
        logger.warning(f"No GT file found for {base_name}. Checked: {candidates}")
        return None, None, None

    try:
        data = np.loadtxt(gt_path)
        points = data[:, 0:3]
        sem_labels = data[:, label_col].astype(int)
        if binarize_sem:
            sem_labels = (sem_labels > 0).astype(int)

        inst_labels = data[:, instance_col].astype(int)
        inst_labels[inst_labels < 0] = 0
        return points, sem_labels, inst_labels
    except Exception as e:
        logger.error(f"Failed to load GT from {gt_path}: {e}")
        return None, None, None


def align_gt_to_pred(gt_points, gt_sem, gt_inst, pred_points):
    if len(gt_points) == len(pred_points):
        return gt_sem, gt_inst

    logger.info(f"Aligning GT ({len(gt_points)}) to Pred ({len(pred_points)}) via Nearest Neighbor...")

    if HAS_SCIPY:
        tree = cKDTree(gt_points)
        _, indices = tree.query(pred_points, k=1)
    else:
        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(gt_points)
        tree = o3d.geometry.KDTreeFlann(pcd_gt)
        indices = []
        for pt in pred_points:
            _, idx, _ = tree.search_knn_vector_3d(pt, 1)
            indices.append(idx[0])
        indices = np.array(indices)

    aligned_sem = gt_sem[indices] if gt_sem is not None else None
    aligned_inst = gt_inst[indices] if gt_inst is not None else None
    return aligned_sem, aligned_inst


def compute_iou_semantic(pred, gt, num_classes=2):
    ious = []
    for c in range(num_classes):
        pred_mask = pred == c
        gt_mask = gt == c
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(intersection / union)
    return ious


def get_instance_matches(pred_inst, gt_inst, iou_thresh=0.5):
    pred_ids = np.unique(pred_inst)
    pred_ids = pred_ids[pred_ids > 0]

    gt_ids = np.unique(gt_inst)
    gt_ids = gt_ids[gt_ids > 0]

    matches = []
    matched_gt = set()

    for pid in pred_ids:
        p_mask = pred_inst == pid
        p_area = p_mask.sum()
        best_iou = 0
        best_gid = -1

        gt_under_pred = gt_inst[p_mask]
        candidate_gids = np.unique(gt_under_pred)
        candidate_gids = candidate_gids[candidate_gids > 0]

        for gid in candidate_gids:
            g_mask = gt_inst == gid
            g_area = g_mask.sum()
            intersection = np.sum(gt_under_pred == gid)
            union = p_area + g_area - intersection
            iou = intersection / union
            if iou > best_iou:
                best_iou = iou
                best_gid = gid

        if best_iou >= iou_thresh and best_gid not in matched_gt:
            matches.append((pid, best_gid, best_iou))
            matched_gt.add(best_gid)

    return matches, len(pred_ids), len(gt_ids)


def compute_instance_metrics_from_matches(matches, num_pred, num_gt):
    tp = len(matches)
    total_iou = sum([m[2] for m in matches])

    precision = tp / num_pred if num_pred > 0 else 0.0
    recall = tp / num_gt if num_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    inst_miou = total_iou / tp if tp > 0 else 0.0

    return precision, recall, f1, inst_miou, tp


def calculate_mae_for_matches(matches, points_clean, full_inst_pred,
                              gt_points_raw, gt_inst_raw,
                              plane_model, trait_calc):
    """对每个匹配实例对, 计算所有表型字段的绝对误差与相对误差.

    返回 dict:
      'abs': {field_key: [绝对误差, ...], ...}
      'rel': {field_key: [相对误差百分比, ...], ...}
    """
    abs_errors = defaultdict(list)
    rel_errors = defaultdict(list)

    for pid, gid, _ in matches:
        p_mask = full_inst_pred == pid
        pts_p = points_clean[p_mask]
        if len(pts_p) < 4:
            continue

        pcd_p = o3d.geometry.PointCloud()
        pcd_p.points = o3d.utility.Vector3dVector(pts_p)

        g_mask = gt_inst_raw == gid
        pts_g = gt_points_raw[g_mask]
        if len(pts_g) < 4:
            continue

        pcd_g = o3d.geometry.PointCloud()
        pcd_g.points = o3d.utility.Vector3dVector(pts_g)

        t_p = trait_calc.calculate_traits(pcd_p, pid, plane_model)
        t_g = trait_calc.calculate_traits(pcd_g, gid, plane_model)

        if not (t_p and t_g):
            continue

        for key, _, _ in TRAIT_MAE_FIELDS:
            vp = t_p.get(key, None)
            vg = t_g.get(key, None)
            if vp is None or vg is None:
                continue
            abs_errors[key].append(abs(vp - vg))
            if abs(vg) > 1e-9:
                rel_errors[key].append(abs(vp - vg) / abs(vg) * 100.0)

    return {'abs': abs_errors, 'rel': rel_errors}


def run_evaluation(args) -> Dict[str, Any]:
    cfg = load_config(args.config)
    seg_method = cfg.get('segmentation', {}).get('method', 'hybrid')
    skip_clustering = getattr(args, 'skip_clustering', False)
    logger.info(f'分割模式: {seg_method}')
    if skip_clustering:
        logger.info('  [消融] 跳过中间聚类算法, 语义分割甘蓝点云整体进入 GIDM 骨架切割+碎片合并')

    # --- 确定性设置: 保证评估可复现 ---
    # 读取 pipeline.seed / pipeline.deterministic (此前被忽略)
    pipeline_cfg = cfg.get('pipeline', {})
    seed = pipeline_cfg.get('seed', None)
    if seed is not None:
        seed = int(seed)
        np.random.seed(seed)
        try:
            import torch as _torch
            _torch.manual_seed(seed)
            _torch.cuda.manual_seed_all(seed)
            if pipeline_cfg.get('deterministic', False):
                _torch.backends.cudnn.deterministic = True
                _torch.backends.cudnn.benchmark = False
            logger.info(f'确定性已启用: seed={seed}, deterministic={pipeline_cfg.get("deterministic", False)}')
        except Exception:
            pass

    clusterer = InstanceClusterer(cfg)
    trait_calc = TraitCalculator(cfg)
    input_files = collect_input_files(args.input)

    # --- Split filtering ---
    split_name = getattr(args, 'split', None)
    if split_name and split_name != 'all':
        import json as _json
        split_path = os.path.join(args.input, 'split.json')
        if not os.path.exists(split_path):
            # 也尝试在项目根目录查找
            split_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      'evalaute_test', 'split.json')
        if os.path.exists(split_path):
            with open(split_path) as _f:
                _split = _json.load(_f)
            allowed = set(_split.get(split_name, []))
            input_files = [f for f in input_files
                           if os.path.splitext(os.path.basename(f))[0] in allowed]
            logger.info(f'Split [{split_name}]: {len(input_files)} files selected from {len(allowed)} allowed')
        else:
            logger.warning(f'Split config not found at {split_path}, using all files')

    logger.info(f'Evaluating on {len(input_files)} files: {[os.path.basename(f) for f in input_files]}')

    global_metrics = {
        'sem_miou': [],
        'sem_acc': [],
        'inst_prec': [],
        'inst_rec': [],
        'inst_f1': [],
        'inst_miou': [],
        'runtime': [],
        'count_error': [],
        'time_prep': [],
        'time_sem': [],
        'time_coarse': [],
        'time_refine': [],
        'time_post': [],
    }
    for _key, _, _ in TRAIT_MAE_FIELDS:
        global_metrics[f'trait_mae_{_key}'] = []
        global_metrics[f'trait_rel_{_key}'] = []
        global_metrics[f'trait_mae_{_key}_perfile'] = []
        global_metrics[f'trait_rel_{_key}_perfile'] = []

    global_sem_intersection = np.zeros(2, dtype=float)
    global_sem_union = np.zeros(2, dtype=float)

    total_gt_instances = 0
    total_pred_instances = 0
    total_tp = 0
    processed_files_list = []

    seg_method = cfg.get('segmentation', {}).get('method', 'hybrid')
    backbone = cfg.get('segmentation', {}).get('backbone', 'pointnet2')
    _needs_pg = seg_method in ('hybrid', 'softgroup', 'pointgroup', 'iach')
    _needs_hais = seg_method == 'hais'
    _needs_randlanet = (backbone == 'randlanet' and seg_method == 'clustering')
    pointgroup_segment = pointgroup_segment_blocks = _iach_cluster = None
    _pointgroup_forward = _load_pointgroup = _post_semantic_filter = None
    hais_segment = hais_segment_blocks = None
    randlanet_model = None

    if _needs_pg:
        try:
            from main import (pointgroup_segment, pointgroup_segment_blocks,
                              _iach_cluster, _pointgroup_forward, _load_pointgroup,
                              _post_semantic_filter)
        except Exception as e:
            pointgroup_segment = None
            pointgroup_segment_blocks = None
            _iach_cluster = None
            _pointgroup_forward = None
            _load_pointgroup = None
            _post_semantic_filter = None
            logger.warning(f'Could not import pointgroup helpers: {e}')

    if _needs_hais:
        try:
            from main import hais_segment, hais_segment_blocks
        except Exception as e:
            hais_segment = None
            hais_segment_blocks = None
            logger.warning(f'Could not import hais helpers: {e}')

    if _needs_randlanet:
        try:
            import torch
            proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            sys.path.insert(0, os.path.join(proj_root, 'RandLANet_Ours'))
            from randlanet import RandLANet as RLAModel
            randlanet_model = RLAModel(d_in=6, num_classes=2).cuda()
            ckpt = os.path.join(proj_root, 'RandLANet_Ours', 'exp', 'randlanet', 'best.pth')
            state = torch.load(ckpt, map_location='cuda')
            randlanet_model.load_state_dict(state['model'])
            randlanet_model.eval()
            logger.info('RandLA-Net model loaded for evaluation')
        except Exception as e:
            logger.error(f'Failed to load RandLA-Net: {e}')
            randlanet_model = None

    for file_path in tqdm(input_files, desc='Processing'):
        logger.info(f'--- {os.path.basename(file_path)} ---')

        pcd = read_point_cloud(file_path)
        if pcd is None:
            continue

        gt_points_raw, gt_sem_raw, gt_inst_raw = load_ground_truth(
            file_path,
            args.gt_label_col,
            args.gt_instance_col,
            args.gt_binarize,
        )

        if gt_inst_raw is None:
            logger.warning('Skipping file due to missing GT.')
            continue

        t_start_prep = time.time()
        preprocess_result = preprocess_point_cloud(pcd, cfg)
        clean_pcd = preprocess_result.pcd_clean
        points_clean = preprocess_result.points_clean
        non_ground_pcd = preprocess_result.non_ground_pcd
        t_prep = time.time() - t_start_prep

        full_semantic_pred = np.zeros(len(points_clean), dtype=int)
        full_instance_pred = np.zeros(len(points_clean), dtype=int) - 1

        t_prep_extra_start = time.time()
        if HAS_SCIPY:
            tree = cKDTree(points_clean)
        else:
            tree = None

        ng_indices = []
        if len(non_ground_pcd.points) > 0 and tree:
            _, ng_indices = tree.query(np.asarray(non_ground_pcd.points), k=1)
        t_prep += time.time() - t_prep_extra_start

        t_start_sem = time.time()
        t_coarse = 0.0
        t_refine = 0.0

        if len(ng_indices) > 0:
            seg_method = cfg.get('segmentation', {}).get('method', 'hybrid')

            # --- 不需要 PointGroup 导入的模式 ---
            if seg_method in ('clustering', 'none'):
                if backbone == 'randlanet' and randlanet_model is not None:
                    # RandLA-Net 语义推理 (分块，每块最多 120K 点防 OOM)
                    import torch as _torch
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
                            pred = randlanet_model(
                                _torch.from_numpy(xyz_shift[start:end]).unsqueeze(0).cuda(),
                                _torch.from_numpy(rgb[start:end]).unsqueeze(0).cuda())
                        mask[start:end] = pred[0].max(1)[1].cpu().numpy() == 1
                    cabbage_pts = xyz[mask]
                    cabbage_pcd = o3d.geometry.PointCloud()
                    cabbage_pcd.points = o3d.utility.Vector3dVector(cabbage_pts)
                    cabbage_pcd.colors = o3d.utility.Vector3dVector(rgb_orig[mask])
                    logger.info(f'RandLA-Net semantic: {mask.sum():,}/{N:,} cabbage pts ({N//CHUNK+1} chunks)')
                else:
                    cabbage_pcd = non_ground_pcd
                inst_init = np.array([], dtype=int)

            # --- 需要 PointGroup 导入的模式 ---
            elif seg_method in ('hybrid', 'softgroup', 'pointgroup', 'iach'):
                if (seg_method in ('hybrid', 'softgroup')
                        and cfg.get('segmentation', {}).get('proposal_selection', 'nms') == 'iach'
                        and _pointgroup_forward and _load_pointgroup and _iach_cluster):
                    # IACH: PointGroup offset 移位 + DBSCAN (与 main.py 一致)
                    import torch as _torch
                    model, model_fn, pg_cfg, _seg_cfg, pgo = _load_pointgroup(cfg)
                    xyz_all = np.asarray(non_ground_pcd.points, dtype=np.float32)
                    rgb_all_orig = np.asarray(non_ground_pcd.colors, dtype=np.float32) if len(non_ground_pcd.colors) > 0 else np.ones_like(xyz_all)
                    rgb_all = (rgb_all_orig * 2 - 1).astype(np.float32)
                    sem_scores, _, _, _, pt_offsets = _pointgroup_forward(
                        xyz_all - xyz_all.min(0), rgb_all, model, model_fn, pg_cfg, pgo)
                    mask = sem_scores.max(1)[1].cpu().numpy() == 1
                    if _post_semantic_filter is not None:
                        mask = _post_semantic_filter(mask, xyz_all, cfg)
                    cabbage_pcd = o3d.geometry.PointCloud()
                    cabbage_pcd.points = o3d.utility.Vector3dVector(xyz_all[mask])
                    cabbage_pcd.colors = o3d.utility.Vector3dVector(rgb_all_orig[mask])
                    iach_l, _ = _iach_cluster(xyz_all[mask], pt_offsets[mask], cfg)
                    inst_init = iach_l
                elif seg_method in ('hybrid', 'softgroup') and pointgroup_segment_blocks:
                    sel = 'greedy' if (seg_method == 'softgroup' or cfg.get('segmentation', {}).get('proposal_selection', 'nms') == 'greedy') else 'nms'
                    cabbage_pcd, inst_init = pointgroup_segment_blocks(non_ground_pcd, cfg, 'hybrid', selection=sel)
                elif seg_method == 'pointgroup' and pointgroup_segment_blocks:
                    cabbage_pcd, inst_init = pointgroup_segment_blocks(non_ground_pcd, cfg, 'pointgroup')
                elif pointgroup_segment:
                    cabbage_pcd, inst_init = pointgroup_segment(non_ground_pcd, cfg, seg_method)
                else:
                    cabbage_pcd = non_ground_pcd
                    inst_init = np.array([], dtype=int)
            elif seg_method == 'hais' and hais_segment_blocks:
                cabbage_pcd, inst_init = hais_segment_blocks(non_ground_pcd, cfg)
            elif seg_method == 'hais' and hais_segment:
                cabbage_pcd, inst_init = hais_segment(non_ground_pcd, cfg)
            else:
                cabbage_pcd = non_ground_pcd
                inst_init = np.array([], dtype=int)

            cabbage_pts = np.asarray(cabbage_pcd.points)

            cab_to_clean = []
            if len(cabbage_pts) > 0 and HAS_SCIPY:
                _, cab_to_ng = cKDTree(np.asarray(non_ground_pcd.points)).query(cabbage_pts, k=1)
                cab_to_clean = np.array(ng_indices)[cab_to_ng]
                full_semantic_pred[cab_to_clean] = 1

            t_sem = time.time() - t_start_sem
            t_start_coarse = time.time()

            if len(cabbage_pts) > 0:
                if skip_clustering:
                    # 跳过中间实例聚类算法: 语义分割得到的甘蓝点云整体作为单一簇,
                    # 直接进入后续 GIDM 骨架切割 + 碎片合并框架
                    labels_s2 = np.zeros(len(cabbage_pts), dtype=np.int64)
                elif seg_method in ('hybrid', 'hais') and inst_init is not None and len(inst_init) > 0:
                    assigned_mask = inst_init > 0
                    if (~assigned_mask).sum() > 100:
                        expand_radius = cfg.get('instance', {}).get('pg_expand', {}).get('radius', 0.05)
                        un_pts_all = cabbage_pts[~assigned_mask]
                        pg_pts = cabbage_pts[assigned_mask]
                        pg_labels = inst_init[assigned_mask]

                        if len(pg_pts) > 0 and HAS_SCIPY:
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
                            t_coarse += getattr(clusterer, 'last_exec_time', {}).get('coarse', 0.0)
                            t_refine += getattr(clusterer, 'last_exec_time', {}).get('refinement', 0.0)
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
                elif seg_method == 'pointgroup' and inst_init is not None and len(inst_init) > 0:
                    # PointGroup 提案输出: 0=未分配/背景, 1+=实例 → 统一转为 -1=背景
                    labels_s2 = np.full(len(cabbage_pts), -1, dtype=np.int64)
                    assigned = inst_init > 0
                    labels_s2[assigned] = inst_init[assigned]
                elif seg_method == 'clustering':
                    un_pcd = o3d.geometry.PointCloud()
                    un_pcd.points = o3d.utility.Vector3dVector(cabbage_pts)
                    labels_s2, _ = clusterer.cluster(un_pcd)
                    t_coarse += getattr(clusterer, 'last_exec_time', {}).get('coarse', 0.0)
                    t_refine += getattr(clusterer, 'last_exec_time', {}).get('refinement', 0.0)
                else:
                    labels_s2 = np.zeros(len(cabbage_pts), dtype=np.int64)

                t_start_refine = time.time()
                labels_s3 = labels_s2.copy()
                # GIDM: PCA 骨架切割 + 碎片回并 (对整个 labels_s2 生效, 包括 PointGroup 提案点)
                # 注意: 之前遗漏了此步, 导致 hybrid 模式下 GIDM 未作用于提案点
                # 修复: (1) clustering 模式已在 clusterer.cluster() 内部完成 GIDM, 跳过以避免重复执行;
                #       (2) 尊重 skeleton_cut / merge_back 消融开关 (此前被无条件绕过, 导致开关失效)
                if (cfg.get('instance', {}).get('pca_split', {}).get('enable', False)
                        and (seg_method != 'clustering' or skip_clustering)):
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
                t_refine += time.time() - t_start_refine

                t_start_post = time.time()
                discard_thresh = cfg.get('instance', {}).get('fragment_voting', {}).get('discard_threshold', 300)
                merge_thresh = cfg.get('instance', {}).get('fragment_voting', {}).get('merge_threshold', 1500)
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
                if noise_mask.sum() > 0 and labels_s3.max() >= 0 and HAS_SCIPY:
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
                t_post = time.time() - t_start_post
                full_instance_pred[cab_to_clean] = inst_labels
            else:
                t_post = 0.0
        else:
            t_sem = time.time() - t_start_sem
            t_post = 0.0

        runtime = t_prep + t_sem + t_coarse + t_refine + t_post
        global_metrics['runtime'].append(runtime)
        global_metrics['time_prep'].append(t_prep)
        global_metrics['time_sem'].append(t_sem)
        global_metrics['time_coarse'].append(t_coarse)
        global_metrics['time_refine'].append(t_refine)
        global_metrics['time_post'].append(t_post)

        gt_sem_clean, gt_inst_clean = align_gt_to_pred(gt_points_raw, gt_sem_raw, gt_inst_raw, points_clean)
        if gt_inst_clean is not None:
            gt_inst_clean = gt_inst_clean.copy()
            gt_inst_clean[gt_inst_clean <= 0] = 0

        ious = compute_iou_semantic(full_semantic_pred, gt_sem_clean, num_classes=2)
        miou = np.nanmean(ious)
        acc = np.mean(full_semantic_pred == gt_sem_clean)

        global_metrics['sem_miou'].append(miou)
        global_metrics['sem_acc'].append(acc)

        for c in range(2):
            p_mask = full_semantic_pred == c
            g_mask = gt_sem_clean == c
            global_sem_intersection[c] += np.logical_and(p_mask, g_mask).sum()
            global_sem_union[c] += np.logical_or(p_mask, g_mask).sum()

        iou_thresh = cfg.get('evaluation', {}).get('iou_thresh', 0.3)
        matches, n_pred, n_gt = get_instance_matches(full_instance_pred, gt_inst_clean, iou_thresh)
        prec, rec, f1, inst_miou, tp = compute_instance_metrics_from_matches(matches, n_pred, n_gt)

        global_metrics['inst_prec'].append(prec)
        global_metrics['inst_rec'].append(rec)
        global_metrics['inst_f1'].append(f1)
        global_metrics['inst_miou'].append(inst_miou)

        count_err = abs(n_pred - n_gt)
        global_metrics['count_error'].append(count_err)

        # 表型参数对比: 对每个匹配实例对计算各表型字段误差
        try:
            mae_result = calculate_mae_for_matches(
                matches, points_clean, full_instance_pred,
                gt_points_raw, gt_inst_raw,
                getattr(preprocess_result, 'plane_model', None), trait_calc)
        except Exception as e:
            logger.warning(f'Trait MAE failed for {os.path.basename(file_path)}: {e}')
            mae_result = {'abs': {}, 'rel': {}}
        for _key, _, _ in TRAIT_MAE_FIELDS:
            abs_errs = mae_result['abs'].get(_key, [])
            rel_errs = mae_result['rel'].get(_key, [])
            global_metrics[f'trait_mae_{_key}'].extend(abs_errs)
            global_metrics[f'trait_rel_{_key}'].extend(rel_errs)
            global_metrics[f'trait_mae_{_key}_perfile'].append(
                float(np.mean(abs_errs)) if abs_errs else float('nan'))
            global_metrics[f'trait_rel_{_key}_perfile'].append(
                float(np.mean(rel_errs)) if rel_errs else float('nan'))

        total_gt_instances += n_gt
        total_pred_instances += n_pred
        total_tp += tp
        processed_files_list.append(os.path.basename(file_path))

    if len(processed_files_list) == 0:
        return {
            'config': args.config,
            'input': args.input,
            'processed_files': [],
            'per_file': [],
            'summary': {},
        }

    per_file_rows = []
    for i, fname in enumerate(processed_files_list):
        t = global_metrics['runtime'][i]
        t1 = global_metrics['time_prep'][i]
        ts = global_metrics['time_sem'][i]
        t2 = global_metrics['time_coarse'][i]
        t3 = global_metrics['time_refine'][i]
        t4 = global_metrics['time_post'][i]
        p = global_metrics['inst_prec'][i]
        r = global_metrics['inst_rec'][i]
        f = global_metrics['inst_f1'][i]
        m = global_metrics['sem_miou'][i]
        ce = global_metrics['count_error'][i]
        row = {
            'file': fname,
            'runtime': float(t),
            'time_prep': float(t1),
            'time_sem': float(ts),
            'time_coarse': float(t2),
            'time_refine': float(t3),
            'time_post': float(t4),
            'inst_prec': float(p),
            'inst_rec': float(r),
            'inst_f1': float(f),
            'sem_miou': float(m),
            'count_error': float(ce),
        }
        for _key, _, _ in TRAIT_MAE_FIELDS:
            _mae = global_metrics[f'trait_mae_{_key}_perfile'][i]
            _rel = global_metrics[f'trait_rel_{_key}_perfile'][i]
            row[f'mae_{_key}'] = None if _mae != _mae else float(_mae)
            row[f'rel_{_key}_pct'] = None if _rel != _rel else float(_rel)
        per_file_rows.append(row)

    avg_prec = np.mean(global_metrics['inst_prec'])
    avg_rec = np.mean(global_metrics['inst_rec'])
    avg_f1 = np.mean(global_metrics['inst_f1'])
    avg_inst_miou = np.mean(global_metrics['inst_miou'])
    avg_sem_miou = np.nanmean(global_metrics['sem_miou'])
    avg_runtime = np.mean(global_metrics['runtime'])
    avg_t1 = np.mean(global_metrics['time_prep'])
    avg_ts = np.mean(global_metrics['time_sem'])
    avg_t2 = np.mean(global_metrics['time_coarse'])
    avg_t3 = np.mean(global_metrics['time_refine'])
    avg_t4 = np.mean(global_metrics['time_post'])
    mae_count = np.mean(global_metrics['count_error'])

    n_trait_pairs = int(len(global_metrics.get(f'trait_mae_{TRAIT_MAE_FIELDS[0][0]}', [])))
    trait_mae_summary = {}
    for _key, _, _ in TRAIT_MAE_FIELDS:
        _mae_list = global_metrics.get(f'trait_mae_{_key}', [])
        _rel_list = global_metrics.get(f'trait_rel_{_key}', [])
        trait_mae_summary[f'mae_{_key}'] = float(np.nanmean(_mae_list)) if _mae_list else None
        trait_mae_summary[f'rel_{_key}_pct'] = float(np.nanmean(_rel_list)) if _rel_list else None

    summary = {
        'avg_runtime': float(avg_runtime),
        'avg_time_prep': float(avg_t1),
        'avg_time_sem': float(avg_ts),
        'avg_time_coarse': float(avg_t2),
        'avg_time_refine': float(avg_t3),
        'avg_time_post': float(avg_t4),
        'avg_inst_prec': float(avg_prec),
        'avg_inst_rec': float(avg_rec),
        'avg_inst_f1': float(avg_f1),
        'avg_inst_miou': float(avg_inst_miou),
        'avg_sem_miou': float(avg_sem_miou),
        'mae_count': float(mae_count),
        'n_trait_matched_pairs': n_trait_pairs,
        'processed_files': int(len(processed_files_list)),
    }
    summary.update(trait_mae_summary)

    return {
        'config': args.config,
        'input': args.input,
        'split': split_name,
        'processed_files': processed_files_list,
        'per_file': per_file_rows,
        'summary': summary,
    }


def evaluate_from_predictions(file_records, iou_thresh=0.3, label='external'):
    """接受外部模型的语义/实例预测，用 Cabbage 指标计算并返回结构化结果。

    file_records: list of dict, 每项需包含:
        fname       (str)  文件名
        pred_sem    (N,)   int 语义预测 (0=背景, 1=甘蓝)
        pred_inst   (N,)   int 实例预测 (0/-1=背景, >=1=实例)
        gt_sem      (N,)   int GT 语义
        gt_inst     (N,)   int GT 实例
        runtime     (float) 可选, 总用时 (秒)
    """
    global_metrics = defaultdict(list)
    processed_files_list = []

    for rec in file_records:
        fname = rec['fname']
        pred_sem = np.asarray(rec['pred_sem'], dtype=int)
        pred_inst = np.asarray(rec['pred_inst'], dtype=int)
        gt_sem = np.asarray(rec['gt_sem'], dtype=int)
        gt_inst = np.asarray(rec['gt_inst'], dtype=int)
        runtime = rec.get('runtime', 0.0)

        gt_inst = gt_inst.copy()
        gt_inst[gt_inst <= 0] = 0

        ious = compute_iou_semantic(pred_sem, gt_sem, num_classes=2)
        miou = np.nanmean(ious)
        acc = np.mean(pred_sem == gt_sem)

        matches, n_pred, n_gt = get_instance_matches(pred_inst, gt_inst, iou_thresh)
        prec, rec, f1, inst_miou, tp = compute_instance_metrics_from_matches(matches, n_pred, n_gt)
        count_err = abs(n_pred - n_gt)

        global_metrics['sem_miou'].append(miou)
        global_metrics['sem_acc'].append(acc)
        global_metrics['inst_prec'].append(prec)
        global_metrics['inst_rec'].append(rec)
        global_metrics['inst_f1'].append(f1)
        global_metrics['inst_miou'].append(inst_miou)
        global_metrics['runtime'].append(runtime)
        global_metrics['count_error'].append(count_err)
        processed_files_list.append(fname)

    if not processed_files_list:
        return {'label': label, 'processed_files': [], 'per_file': [], 'summary': {}}

    per_file_rows = []
    for i, fname in enumerate(processed_files_list):
        per_file_rows.append({
            'file': fname,
            'runtime': float(global_metrics['runtime'][i]),
            'inst_prec': float(global_metrics['inst_prec'][i]),
            'inst_rec': float(global_metrics['inst_rec'][i]),
            'inst_f1': float(global_metrics['inst_f1'][i]),
            'sem_miou': float(global_metrics['sem_miou'][i]),
            'count_error': float(global_metrics['count_error'][i]),
        })

    summary = {
        'avg_runtime': float(np.mean(global_metrics['runtime'])),
        'avg_inst_prec': float(np.mean(global_metrics['inst_prec'])),
        'avg_inst_rec': float(np.mean(global_metrics['inst_rec'])),
        'avg_inst_f1': float(np.mean(global_metrics['inst_f1'])),
        'avg_inst_miou': float(np.mean(global_metrics['inst_miou'])),
        'avg_sem_miou': float(np.nanmean(global_metrics['sem_miou'])),
        'mae_count': float(np.mean(global_metrics['count_error'])),
        'processed_files': int(len(processed_files_list)),
    }

    return {
        'label': label,
        'processed_files': processed_files_list,
        'per_file': per_file_rows,
        'summary': summary,
    }


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to test folder or file')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--split', type=str, default=None,
                        choices=['val', 'test', 'all'],
                        help='Evaluate on val/test split (reads evalaute_test/split.json)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save JSON results (auto-generated if --split is used)')
    parser.add_argument('--gt-label-col', type=int, default=-2)
    parser.add_argument('--gt-instance-col', type=int, default=-1)
    parser.add_argument('--gt-binarize', action='store_true', help='Treat all label > 0 as class 1')
    parser.add_argument('--skip-clustering', action='store_true',
                        help='跳过中间实例聚类算法, 语义分割甘蓝点云整体进入 GIDM 骨架切割+碎片合并')
    return parser
