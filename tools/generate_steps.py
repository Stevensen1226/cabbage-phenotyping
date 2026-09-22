#!/usr/bin/env python3
"""生成单块田地点云的全流程中间步骤 PLY 文件。

用法:
  # 只做预处理 (SOR + RANSAC)
  python tools/generate_steps.py --input evalaute_test/cloudR4.ply --steps preprocess

  # 预处理 + 语义分割 (PointNet2/PointGroup)
  python tools/generate_steps.py --input evalaute_test/cloudR4.ply --steps semantic

  # 预处理 + 语义 + 实例聚类
  python tools/generate_steps.py --input evalaute_test/cloudR4.ply --steps cluster

  # 全流程
  python tools/generate_steps.py --input evalaute_test/cloudR4.ply --steps all

输出文件 (存放在 INPUT 同目录):
  {name}_step1_sor.ply               SOR 去噪
  {name}_step2_ground.ply            被 RANSAC 移除的地面点
  {name}_step2_non_ground.ply        去地面后的非地面点
  {name}_step3_semantic_cabbage.ply  语义分割后仅甘蓝点
  {name}_step3_semantic_all.ply      语义着色全点云
  {name}_step4a_cluster_coarse.ply    粗聚类结果
  {name}_step4b_skeleton_split.ply    仅骨架拆分后的结果
  {name}_step4c_fragment_merged.ply   仅碎片回并后的结果
  {name}_step4d_cluster_cleaned.ply   碎片清理后的结果
  {name}_step4_cluster_coarse.ply     兼容旧脚本的粗聚类别名
  {name}_step4_cluster_refined.ply    兼容旧脚本的碎片回并别名
  {name}_step5_final.ply              最终实例着色输出
"""
import argparse, os, sys, copy, numpy as np, open3d as o3d, yaml, time

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.getcwd())

from cabbage_pheno.io import read_point_cloud
from cabbage_pheno.service.pipeline import preprocess_point_cloud


def load_config(path='configs/default.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_colors_from_labels(labels, colormap='tab20'):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(colormap)
    colors = np.zeros((len(labels), 3))
    noise_mask = (labels == -1)
    colors[noise_mask] = [0.1, 0.1, 0.1]
    valid_mask = ~noise_mask
    if np.any(valid_mask):
        norm = (labels[valid_mask] % 20) / 20.0
        colors[valid_mask] = cmap(norm)[:, :3]
    return colors


# ═══════════════════════════════════════════════
# Step 1-2: 预处理 (SOR + RANSAC 地面去除)
# ═══════════════════════════════════════════════
def write_label_cloud(pcd, labels, path):
    """Write one stage of instance labels as an RGB-colored PLY."""
    out = copy.deepcopy(pcd)
    out.colors = o3d.utility.Vector3dVector(get_colors_from_labels(labels))
    o3d.io.write_point_cloud(path, out)
    return path


def run_preprocess(input_path, cfg, out_dir):
    name = os.path.splitext(os.path.basename(input_path))[0]
    pcd = read_point_cloud(input_path)
    print(f'[Step 0] 原始: {len(pcd.points):,} pts')

    # Step 1: 纯 SOR 去噪 (不含地面去除)
    from cabbage_pheno.preprocess.cleaning import PointCloudCleaner
    cleaner = PointCloudCleaner(cfg)
    pcd_sor = cleaner.remove_outliers(pcd)
    o3d.io.write_point_cloud(os.path.join(out_dir, f'{name}_step1_sor.ply'), pcd_sor)
    print(f'[Step 1] SOR 后: {len(pcd_sor.points):,} → {name}_step1_sor.ply')

    # Step 2: 地面去除 (复用 pipeline 统一逻辑, 与 evaluate/train 完全一致)
    result = preprocess_point_cloud(pcd, cfg)
    pcd_g = result.ground_pcd
    pcd_ng = result.non_ground_pcd

    if pcd_g is not None:
        o3d.io.write_point_cloud(os.path.join(out_dir, f'{name}_step2_ground.ply'), pcd_g)
    o3d.io.write_point_cloud(os.path.join(out_dir, f'{name}_step2_non_ground.ply'), pcd_ng)

    n_sor = len(pcd_sor.points)
    n_g = len(pcd_g.points) if pcd_g is not None else 0
    print(f'[Step 2] 地面: {n_g:,} ({100*n_g/n_sor:.1f}%) → {name}_step2_ground.ply')
    print(f'[Step 2] 非地面: {len(pcd_ng.points):,} → {name}_step2_non_ground.ply')
    return pcd_ng


# ═══════════════════════════════════════════════
# Step 3: 语义分割 (PointNeXt-L)
# ═══════════════════════════════════════════════
def _run_semantic_randlanet(xyz, rgb, non_ground_pcd, name, out_dir):
    import torch
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(proj_root, 'RandLANet_Ours'))
    from randlanet import RandLANet
    print(f'[Step 3] 加载 RandLA-Net ({len(xyz):,} pts)...')

    model = RandLANet(d_in=6, num_classes=2).cuda().eval()
    ckpt = os.path.join(proj_root, 'RandLANet_Ours', 'exp', 'randlanet', 'best.pth')
    sd = torch.load(ckpt, map_location='cuda')
    model.load_state_dict(sd['model'] if isinstance(sd, dict) and 'model' in sd else sd, strict=False)

    rgb_n = (rgb * 2 - 1).astype(np.float32)   # [0,1] -> [-1,1]
    xyz_shift = xyz - xyz.min(0)
    N = len(xyz)
    CHUNK = 120000
    prob = np.zeros((N, 2), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, N, CHUNK):
            end = min(start + CHUNK, N)
            out = model(torch.from_numpy(xyz_shift[start:end]).unsqueeze(0).cuda(),
                        torch.from_numpy(rgb_n[start:end]).unsqueeze(0).cuda())
            prob[start:end] = torch.softmax(out.squeeze(0), dim=-1).cpu().numpy()
            torch.cuda.empty_cache()

    sem = prob.argmax(1).astype(int)
    sp = prob[:, 1] / np.maximum(prob.sum(1), 1e-8)
    nc = (sem == 1).sum()
    print(f'  甘蓝={nc:,}, 背景={N-nc:,} ({100*nc/N:.1f}%)')

    cp = o3d.geometry.PointCloud()
    cp.points = o3d.utility.Vector3dVector(xyz[sem == 1])
    cp.colors = o3d.utility.Vector3dVector(rgb[sem == 1])
    o3d.io.write_point_cloud(os.path.join(out_dir, f'{name}_step3_semantic_cabbage.ply'), cp)

    ap = o3d.geometry.PointCloud()
    ap.points = non_ground_pcd.points
    ap.colors = o3d.utility.Vector3dVector(np.column_stack([np.zeros(N), sp, np.zeros(N)]))
    o3d.io.write_point_cloud(os.path.join(out_dir, f'{name}_step3_semantic_all.ply'), ap)
    print(f'  → {name}_step3_semantic_cabbage.ply + all.ply')

    cabbage_pcd = o3d.geometry.PointCloud()
    cabbage_pcd.points = o3d.utility.Vector3dVector(xyz[sem == 1])
    cabbage_pcd.colors = o3d.utility.Vector3dVector(rgb[sem == 1])
    return cabbage_pcd


def _run_semantic_pointgroup(xyz, rgb, non_ground_pcd, name, out_dir, cfg):
    import torch
    import main as pg_main

    print(f'[Step 3] 加载 PointGroup ({len(xyz):,} pts)...')
    model, model_fn, pg_cfg, seg_cfg, pgo = pg_main._load_pointgroup(cfg)
    model.eval()

    xyz_mid = xyz - xyz.min(0)
    rgb_n = (rgb * 2 - 1).astype(np.float32)

    semantic_scores, _, _, _, _ = pg_main._pointgroup_forward(
        xyz_mid, rgb_n, model, model_fn, pg_cfg, pgo)

    prob = torch.softmax(semantic_scores, dim=1).cpu().numpy()  # (N, 2)
    sem = prob.argmax(1).astype(np.int32)
    sp = prob[:, 1]

    N = len(xyz)
    nc = int((sem == 1).sum())
    print(f'  甘蓝={nc:,}, 背景={N-nc:,} ({100*nc/N:.1f}%)')

    cp = o3d.geometry.PointCloud()
    cp.points = o3d.utility.Vector3dVector(xyz[sem == 1])
    cp.colors = o3d.utility.Vector3dVector(rgb[sem == 1])
    o3d.io.write_point_cloud(os.path.join(out_dir, f'{name}_step3_semantic_cabbage.ply'), cp)

    ap = o3d.geometry.PointCloud()
    ap.points = non_ground_pcd.points
    ap.colors = o3d.utility.Vector3dVector(np.column_stack([np.zeros(N), sp, np.zeros(N)]))
    o3d.io.write_point_cloud(os.path.join(out_dir, f'{name}_step3_semantic_all.ply'), ap)
    print(f'  → {name}_step3_semantic_cabbage.ply + all.ply')

    cabbage_pcd = o3d.geometry.PointCloud()
    cabbage_pcd.points = o3d.utility.Vector3dVector(xyz[sem == 1])
    cabbage_pcd.colors = o3d.utility.Vector3dVector(rgb[sem == 1])
    return cabbage_pcd


def run_semantic(non_ground_pcd, name, out_dir, cfg):
    import torch
    from scipy.spatial import cKDTree

    xyz = np.asarray(non_ground_pcd.points, dtype=np.float32)
    rgb = np.asarray(non_ground_pcd.colors) if len(non_ground_pcd.colors) > 0 \
          else np.ones((len(xyz), 3), dtype=np.float32)

    backbone = cfg.get('segmentation', {}).get('backbone', 'pointnet2').lower()
    if backbone == 'randlanet':
        return _run_semantic_randlanet(xyz, rgb, non_ground_pcd, name, out_dir)

    seg_method = cfg.get('segmentation', {}).get('method', 'hybrid')
    if seg_method in ('hybrid', 'pointgroup', 'softgroup', 'iach'):
        return _run_semantic_pointgroup(xyz, rgb, non_ground_pcd, name, out_dir, cfg)

    sys.path.insert(0, '/home/stevensen/PointNeXt-master')
    from openpoints.models import build_model_from_cfg
    from openpoints.utils import EasyConfig
    print(f'[Step 3] 加载 PointNeXt-L ({len(xyz):,} pts)...')

    pn_cfg = EasyConfig()
    pn_cfg.load('/home/stevensen/PointNeXt-master/cfgs/cabbage.yaml', recursive=True)
    model = build_model_from_cfg(pn_cfg.model).cuda().eval()
    sd = torch.load('/home/stevensen/PointNeXt-master/best_model.pth', map_location='cpu')
    model.load_state_dict(sd, strict=False)

    GM = np.array([-0.10488096, 0.18573543, 0.00078407], dtype=np.float32)
    GS = np.array([0.8321599, 1.4525476, 0.09171805], dtype=np.float32)

    def infer_once(x, r):
        xn = (x - GM) / (GS + 1e-8)
        xt = torch.FloatTensor(xn).unsqueeze(0).cuda()
        f = np.concatenate([xn, r], axis=1).astype(np.float32)
        ft = torch.FloatTensor(f).unsqueeze(0).permute(0, 2, 1).cuda()
        ot = torch.tensor([len(x)], dtype=torch.int32).unsqueeze(0).cuda()
        with torch.no_grad():
            out = model({'pos': xt, 'x': ft, 'offset': ot})
        return torch.softmax(out.squeeze(0), dim=0).cpu().numpy().T

    N = len(xyz)
    if N <= 25000:
        prob = infer_once(xyz, rgb)
    else:
        SZ = 20480
        nr = max(5, min(30, int(np.ceil(3.0*N/SZ))))
        print(f'  分块推理: {nr} rounds...', end='', flush=True)
        ap = np.zeros((N, 2), dtype=np.float32)
        for rnd in range(nr):
            ch = np.random.choice(N, min(N, SZ), replace=False)
            ps = infer_once(xyz[ch], rgb[ch])
            _, nn = cKDTree(xyz[ch]).query(xyz, k=1)
            ap += ps[nn]
            torch.cuda.empty_cache()
        prob = ap

    sem = prob.argmax(1).astype(int)
    sp = prob[:, 1] / np.maximum(prob.sum(1), 1e-8)
    nc = (sem == 1).sum()
    print(f'  甘蓝={nc:,}, 背景={N-nc:,} ({100*nc/N:.1f}%)')

    cp = o3d.geometry.PointCloud()
    cp.points = o3d.utility.Vector3dVector(xyz[sem == 1])
    cp.colors = o3d.utility.Vector3dVector(rgb[sem == 1])
    o3d.io.write_point_cloud(os.path.join(out_dir, f'{name}_step3_semantic_cabbage.ply'), cp)

    ap = o3d.geometry.PointCloud()
    ap.points = non_ground_pcd.points
    ap.colors = o3d.utility.Vector3dVector(np.column_stack([np.zeros(N), sp, np.zeros(N)]))
    o3d.io.write_point_cloud(os.path.join(out_dir, f'{name}_step3_semantic_all.ply'), ap)
    print(f'  → {name}_step3_semantic_cabbage.ply + all.ply')

    # Return only cabbage points
    cabbage_pcd = o3d.geometry.PointCloud()
    cabbage_pcd.points = o3d.utility.Vector3dVector(xyz[sem == 1])
    cabbage_pcd.colors = o3d.utility.Vector3dVector(rgb[sem == 1])
    return cabbage_pcd


# ═══════════════════════════════════════════════
# Step 4: 实例聚类 + GIDM 骨架拆分
# ═══════════════════════════════════════════════
def run_pointgroup_full(ng_pcd, name, out_dir, cfg):
    """PointGroup 完整流程 (与统一评估 evaluation.py 的 hybrid 模式完全一致):
    语义 + offset 提案 + NMS → 未分配点 watershed 补全 → GIDM 骨架切割+碎片回并 → 后处理
    """
    import main as pg_main
    from cabbage_pheno.instance.clustering import InstanceClusterer
    from scipy.spatial import cKDTree

    clusterer = InstanceClusterer(cfg)

    # Step 3: PointGroup 语义 + offset 提案 + NMS
    print(f'[Step 3] PointGroup 语义 + offset 提案 + NMS...')
    cabbage_pcd, inst_init = pg_main.pointgroup_segment_blocks(
        ng_pcd, cfg, 'hybrid', selection='nms')
    cabbage_pts = np.asarray(cabbage_pcd.points)
    rgb_cab = np.asarray(cabbage_pcd.colors) if len(cabbage_pcd.colors) > 0 \
        else np.ones((len(cabbage_pts), 3))

    # 保存语义甘蓝点
    cp = copy.deepcopy(cabbage_pcd)
    o3d.io.write_point_cloud(os.path.join(out_dir, f'{name}_step3_semantic_cabbage.ply'), cp)
    # 语义全点云: 甘蓝点标绿(实例着色用提案结果)
    n_all = len(ng_pcd.points)
    all_colors = np.zeros((n_all, 3))
    # 用最近邻把 cabbage 点映射回 ng
    ng_pts = np.asarray(ng_pcd.points)
    _, cab_to_ng = cKDTree(ng_pts).query(cabbage_pts, k=1)
    all_colors[cab_to_ng] = [0.0, 1.0, 0.0]
    ap = copy.deepcopy(ng_pcd)
    ap.colors = o3d.utility.Vector3dVector(all_colors)
    o3d.io.write_point_cloud(os.path.join(out_dir, f'{name}_step3_semantic_all.ply'), ap)
    print(f'  甘蓝={len(cabbage_pts):,}, 背景={n_all-len(cabbage_pts):,} ({100*len(cabbage_pts)/n_all:.1f}%)')

    # Step 3.5: 保存纯 PointGroup 实例分割结果 (offset 提案 + NMS, 未补全)
    # inst_init: 0=未分配/背景, 1+=实例 → 转成 -1=未分配 用于着色
    proposal_labels = np.where(inst_init > 0, inst_init, -1).astype(np.int64)
    cpp = copy.deepcopy(cabbage_pcd)
    cpp.colors = o3d.utility.Vector3dVector(get_colors_from_labels(proposal_labels))
    o3d.io.write_point_cloud(os.path.join(out_dir, f'{name}_step3.5_pointgroup_proposal.ply'), cpp)
    n_prop = int(len(np.unique(inst_init[inst_init > 0]))) if (inst_init > 0).any() else 0
    n_prop_assigned = int((inst_init > 0).sum())
    print(f'[Step 3.5] PointGroup 提案 (offset+NMS): {n_prop} instances, {n_prop_assigned:,}/{len(cabbage_pts):,} 点已分配 → {name}_step3.5_pointgroup_proposal.ply')

    # Step 4a: 提案 + 未分配点补全 (labels_s2)
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

    # 保存 step4a (提案+补全, GIDM 前)
    write_label_cloud(
        cabbage_pcd, labels_s2,
        os.path.join(out_dir, f'{name}_step4a_cluster_coarse.ply'))
    # Backward-compatible alias used by older scripts.
    write_label_cloud(
        cabbage_pcd, labels_s2,
        os.path.join(out_dir, f'{name}_step4_cluster_coarse.ply'))
    n_s2 = len(np.unique(labels_s2[labels_s2 >= 0]))
    print(f'[Step 4a] Cluster coarse: {n_s2} instances → {name}_step4a_cluster_coarse.ply')

    # Step 4b: skeleton split only (fragment merge has not run yet)
    labels_split = labels_s2.copy()
    if cfg.get('instance', {}).get('pca_split', {}).get('enable', False):
        if clusterer.skeleton_cut:
            try:
                labels_split = clusterer._apply_skeleton_split(labels_split, cabbage_pts)
            except Exception:
                pass

    write_label_cloud(
        cabbage_pcd, labels_split,
        os.path.join(out_dir, f'{name}_step4b_skeleton_split.ply'))
    n_split = len(np.unique(labels_split[labels_split >= 0]))
    print(f'[Step 4b] Skeleton split: {n_split} instances → {name}_step4b_skeleton_split.ply')

    # Step 4c: fragment merge only (cleanup has not run yet)
    labels_merged = labels_split.copy()
    if cfg.get('instance', {}).get('pca_split', {}).get('enable', False):
        if clusterer.merge_back:
            try:
                labels_merged = clusterer._merge_fragments(labels_merged, cabbage_pts)
            except Exception:
                pass

    write_label_cloud(
        cabbage_pcd, labels_merged,
        os.path.join(out_dir, f'{name}_step4c_fragment_merged.ply'))
    # Backward-compatible alias: the old refined file denoted post-merge state.
    write_label_cloud(
        cabbage_pcd, labels_merged,
        os.path.join(out_dir, f'{name}_step4_cluster_refined.ply'))
    n_r = len(np.unique(labels_merged[labels_merged >= 0]))
    print(f'[Step 4c] Fragment merge: {n_r} instances → {name}_step4c_fragment_merged.ply')

    labels_s3 = labels_merged.copy()

    # Step 5: 后处理 (discard → fragment_voting → min_pts → noise reassign → renumber)
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
        _, nn_idx = cKDTree(cabbage_pts[valid_mask]).query(cabbage_pts[noise_mask], k=1)
        labels_s3[noise_mask] = labels_s3[valid_mask][nn_idx]

    valid = labels_s3 >= 0
    new_labels = np.full_like(labels_s3, -1)
    next_id = 1
    for old_lbl in sorted(set(labels_s3[valid].tolist())):
        new_labels[labels_s3 == old_lbl] = next_id
        next_id += 1
    inst_labels = new_labels

    write_label_cloud(
        cabbage_pcd, inst_labels,
        os.path.join(out_dir, f'{name}_step4d_cluster_cleaned.ply'))
    n_clean = len(np.unique(inst_labels[inst_labels >= 0]))
    print(f'[Step 4d] Cluster cleanup: {n_clean} instances → {name}_step4d_cluster_cleaned.ply')

    fp = copy.deepcopy(cabbage_pcd)
    fp.colors = o3d.utility.Vector3dVector(get_colors_from_labels(inst_labels))
    o3d.io.write_point_cloud(os.path.join(out_dir, f'{name}_step5_final.ply'), fp)
    n_f = len(np.unique(inst_labels[inst_labels >= 0]))
    print(f'[Step 5] 最终: {n_f} instances → {name}_step5_final.ply')
    return inst_labels


def run_clustering(cabbage_pcd, name, out_dir, cfg):
    from cabbage_pheno.instance.clustering import InstanceClusterer

    clusterer = InstanceClusterer(cfg)
    points = np.asarray(cabbage_pcd.points)
    rgb = np.asarray(cabbage_pcd.colors) if len(cabbage_pcd.colors) > 0 \
          else np.ones((len(points), 3))

    # Coarse clustering (根据 instance.method 选择算法)
    method = cfg.get('instance', {}).get('method', 'watershed_3d')
    method_fn = {
        'watershed_3d': clusterer._cluster_watershed_3d,
        'meanshift': clusterer._cluster_meanshift,
        'hdbscan': clusterer._cluster_hdbscan,
        'dbscan': clusterer._cluster_dbscan,
        'euclidean': clusterer._cluster_euclidean,
        'graph_based': clusterer._cluster_graph_based,
    }.get(method, clusterer._cluster_watershed_3d)
    coarse_labels, _ = method_fn(cabbage_pcd)
    write_label_cloud(
        cabbage_pcd, coarse_labels,
        os.path.join(out_dir, f'{name}_step4a_cluster_coarse.ply'))
    # Backward-compatible alias used by older scripts.
    write_label_cloud(
        cabbage_pcd, coarse_labels,
        os.path.join(out_dir, f'{name}_step4_cluster_coarse.ply'))
    n_c = len(set(coarse_labels)) - (1 if -1 in coarse_labels else 0)
    print(f'[Step 4a] Cluster coarse ({method}): {n_c} clusters → {name}_step4a_cluster_coarse.ply')

    # Step 4b: skeleton split only.
    labels_split = clusterer._apply_skeleton_split(coarse_labels, points)
    write_label_cloud(
        cabbage_pcd, labels_split,
        os.path.join(out_dir, f'{name}_step4b_skeleton_split.ply'))
    n_split = len(set(labels_split)) - (1 if -1 in labels_split else 0)
    print(f'[Step 4b] Skeleton split: {n_split} instances → {name}_step4b_skeleton_split.ply')

    # Step 4c: fragment merge only.
    labels_merged = clusterer._merge_fragments(labels_split, points)
    write_label_cloud(
        cabbage_pcd, labels_merged,
        os.path.join(out_dir, f'{name}_step4c_fragment_merged.ply'))
    # Backward-compatible alias: old refined file denotes post-merge state.
    write_label_cloud(
        cabbage_pcd, labels_merged,
        os.path.join(out_dir, f'{name}_step4_cluster_refined.ply'))
    n_r = len(set(labels_merged)) - (1 if -1 in labels_merged else 0)
    print(f'[Step 4c] Fragment merge: {n_r} instances → {name}_step4c_fragment_merged.ply')

    # Step 4d: fragment cleanup + renumbering.
    refined = clusterer._cleanup_tiny_fragments(labels_merged, points)
    valid = refined >= 0
    new_lbl = np.full_like(refined, -1)
    nid = 0
    for ol in sorted(set(refined[valid])):
        new_lbl[refined == ol] = nid
        nid += 1
    refined = new_lbl
    write_label_cloud(
        cabbage_pcd, refined,
        os.path.join(out_dir, f'{name}_step4d_cluster_cleaned.ply'))
    n_clean = len(set(refined)) - (1 if -1 in refined else 0)
    print(f'[Step 4d] Cluster cleanup: {n_clean} instances → {name}_step4d_cluster_cleaned.ply')

    # Final (min filter then noise reassign, 与 evaluation.py 顺序一致)
    min_pts = cfg.get('instance', {}).get('min_cluster_points', 2000)
    uni, cnts = np.unique(refined, return_counts=True)
    for lbl, cnt in zip(uni, cnts):
        if lbl < 0: continue
        if cnt < min_pts:
            refined[refined == lbl] = -1

    # 最后把 -1 点重新分配给最近有效簇 (避免黑色残留)
    noise = refined == -1
    if noise.sum() > 0 and refined.max() >= 0:
        from scipy.spatial import cKDTree
        vm = refined >= 0
        _, nn = cKDTree(points[vm]).query(points[noise], k=1)
        refined[noise] = refined[vm][nn]

    fp = copy.deepcopy(cabbage_pcd)
    fp.colors = o3d.utility.Vector3dVector(get_colors_from_labels(refined))
    o3d.io.write_point_cloud(os.path.join(out_dir, f'{name}_step5_final.ply'), fp)
    n_f = len(set(refined)) - (1 if -1 in refined else 0)
    print(f'[Step 5] 最终: {n_f} instances → {name}_step5_final.ply')
    return refined


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成点云处理流程各步骤 PLY 文件')
    parser.add_argument('--input', required=True, help='输入 .ply 文件路径')
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--steps', default='all',
                        choices=['preprocess', 'semantic', 'cluster', 'all'],
                        help='处理到哪一步 (default: all)')
    parser.add_argument('--out_dir', default=None,
                        help='输出目录 (默认与输入文件同目录)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    name = os.path.splitext(os.path.basename(args.input))[0]
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(out_dir, exist_ok=True)

    # 固定随机种子 (保证可复现)
    seed = cfg.get('pipeline', {}).get('seed', 0)
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    try:
        import torch as _torch
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    t0 = time.time()

    # Steps 1-2: Preprocess
    ng_pcd = run_preprocess(args.input, cfg, out_dir)
    if args.steps == 'preprocess':
        print(f'Done in {time.time()-t0:.1f}s')
        sys.exit(0)

    # PointGroup 模式: 走完整流程 (语义+offset提案+NMS → 补全 → GIDM → 后处理)
    seg_method = cfg.get('segmentation', {}).get('method', 'hybrid')
    if seg_method in ('hybrid', 'pointgroup', 'softgroup', 'iach'):
        run_pointgroup_full(ng_pcd, name, out_dir, cfg)
        print(f'\n全部完成! 耗时 {time.time()-t0:.1f}s, 文件保存在 {out_dir}/')
        sys.exit(0)

    # Step 3: Semantic
    cabbage_pcd = run_semantic(ng_pcd, name, out_dir, cfg)
    if args.steps == 'semantic':
        print(f'Done in {time.time()-t0:.1f}s')
        sys.exit(0)

    # Step 4-5: Clustering + GIDM
    run_clustering(cabbage_pcd, name, out_dir, cfg)
    print(f'\n全部完成! 耗时 {time.time()-t0:.1f}s, 文件保存在 {out_dir}/')

