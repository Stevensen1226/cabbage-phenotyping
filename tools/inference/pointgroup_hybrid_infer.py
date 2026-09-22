#!/usr/bin/env python3
"""
PointGroup + 聚类 + 骨架拆分 三阶段混合推理

Stage 1: PointGroup 语义分割 + 实例 proposal
Stage 2: 对 PointGroup 未充分分割的大簇，用聚类算法 (DBSCAN/Meanshift) 二次拆分
Stage 3: 对仍过大的簇，用基于局部骨架曲线的自适应拆分 (PCA Split)
"""
import os, sys, glob, argparse, yaml
import numpy as np
import torch
import open3d as o3d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'PointGroup_Ours'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='输入 .ply 文件或通配符')
    parser.add_argument('--output', default='output/hybrid', help='输出目录')
    parser.add_argument('--config', default='config/pointgroup_cabbage.yaml', help='PointGroup 配置')
    parser.add_argument('--cluster_config', default='../configs/default.yaml', help='二次聚类配置 (cabbage_pheno)')
    parser.add_argument('--score_thresh', type=float, default=0.1)
    parser.add_argument('--npoint_thresh', type=int, default=30)
    parser.add_argument('--nms_thresh', type=float, default=0.5)
    parser.add_argument('--cluster_radius', type=float, default=0.025)
    # 二次拆分参数
    parser.add_argument('--split_method', default='meanshift', choices=['meanshift','dbscan','euclidean','watershed_3d'])
    parser.add_argument('--split_diameter', type=float, default=0.35, help='簇直径超过此值(m)则触发二次拆分')
    parser.add_argument('--split_min_points', type=int, default=500, help='簇点数超过此值才考虑二次拆分')
    parser.add_argument('--bandwidth', type=float, default=0.2, help='Meanshift/DBSCAN 带宽 (m)')
    # 骨架拆分参数 (Stage 3)
    parser.add_argument('--skeleton_split', action='store_true', default=True, help='启用基于骨架曲线的自适应拆分')
    parser.add_argument('--skeleton_diameter', type=float, default=0.40, help='骨架拆分的簇直径阈值(m), 正常甘蓝0.25~0.35')
    parser.add_argument('--skeleton_ar', type=float, default=1.50, help='骨架拆分的纵横比阈值')
    parser.add_argument('--skeleton_valley', type=float, default=0.85, help='骨架拆分的波谷深度(0~1, 越小越容易切)')
    return parser.parse_args()

def load_pointgroup(cfg):
    from model.pointgroup.pointgroup import PointGroup as Network, model_fn_decorator
    import util.utils as utils
    model = Network(cfg).cuda()
    utils.checkpoint_restore(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], True, cfg.test_epoch)
    model_fn = model_fn_decorator(test=True)
    return model, model_fn

def ply_to_pointgroup_input(ply_path, cfg):
    from lib.pointgroup_ops.functions import pointgroup_ops
    pcd = o3d.io.read_point_cloud(ply_path)
    xyz = np.asarray(pcd.points, dtype=np.float32)
    rgb = (np.asarray(pcd.colors)*2-1).astype(np.float32) if len(pcd.colors)>0 else np.zeros_like(xyz)
    xyz_mid = xyz - xyz.min(0)
    xyz_s = xyz_mid * cfg.scale
    N = xyz_s.shape[0]
    locs = torch.cat([torch.LongTensor(N,1).fill_(0), torch.from_numpy(xyz_s).long()], 1)
    voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, 1, cfg.mode)
    batch = {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
             'locs_float': torch.from_numpy(xyz_mid), 'feats': torch.from_numpy(rgb),
             'offsets': torch.tensor([0,N], dtype=torch.int),
             'spatial_shape': np.clip((locs.max(0)[0][1:]+1).numpy(), cfg.full_scale[0], None),
             'id': torch.tensor([0])}
    return batch, xyz_mid, pcd

def pointgroup_proposals(batch, model, model_fn, cfg, N, score_thresh, npoint_thresh, nms_thresh):
    """Stage 1: PointGroup 实例 proposals"""
    with torch.no_grad():
        preds = model_fn(batch, model, cfg.test_epoch)
    semantic_pred = preds['semantic'].max(1)[1]
    proposals_idx, proposals_offset = preds['proposals']
    scores_pred = torch.sigmoid(preds['score'].view(-1))
    nProp = proposals_offset.shape[0]-1
    proposals_mask = torch.zeros((nProp,N), dtype=torch.int, device=scores_pred.device)
    proposals_mask[proposals_idx[:,0].long(), proposals_idx[:,1].long()] = 1

    # Filter
    score_mask = scores_pred > score_thresh
    proposals_mask, scores_pred = proposals_mask[score_mask], scores_pred[score_mask]
    if proposals_mask.shape[0]==0: return semantic_pred, np.zeros(N,dtype=np.int64)
    npoint_mask = proposals_mask.sum(1) > npoint_thresh
    proposals_mask, scores_pred = proposals_mask[npoint_mask], scores_pred[npoint_mask]
    if proposals_mask.shape[0]==0: return semantic_pred, np.zeros(N,dtype=np.int64)

    # NMS
    proposals_f = proposals_mask.float()
    inter = torch.mm(proposals_f, proposals_f.t())
    pn = proposals_f.sum(1)
    ious = inter / (pn.unsqueeze(-1)+pn.unsqueeze(0)-inter+1e-6)
    ixs = scores_pred.cpu().numpy().argsort()[::-1]
    pick, ious_np = [], ious.cpu().numpy()
    while len(ixs)>0:
        i=ixs[0]; pick.append(i)
        rm=np.where(ious_np[i,ixs[1:]]>nms_thresh)[0]+1
        ixs=np.delete(ixs,rm); ixs=np.delete(ixs,0)
    clusters = proposals_mask[np.array(pick,dtype=np.int32)].cpu().numpy()
    inst_label = np.zeros(N,dtype=np.int64)
    for i in range(len(pick)): inst_label[clusters[i]==1] = i+1
    return semantic_pred, inst_label

def skeleton_split_cluster(indices, all_xyz, args, depth=0, debug=False):
    """Stage 3: 基于局部骨架曲线的自适应递归拆分 (PCA Split)"""
    prefix = "  " * depth + f"[深度{depth}]"
    if len(indices) < 50 or depth >= 3:
        if debug: print(f"{prefix} 跳过: 点数{len(indices)}<50")
        return [indices]

    pts = all_xyz[indices]
    center = pts.mean(0)
    centered = pts - center
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eig(cov)
    sort_idx = np.argsort(evals)[::-1]
    evecs, evals = evecs[:, sort_idx], evals[sort_idx]
    v1 = evecs[:, 0]

    proj = np.dot(centered, evecs)
    extents = proj.max(0) - proj.min(0)
    L1, L2, L3 = extents[0], extents[1], extents[2]
    aspect_ratio = L1 / (L2 + 1e-6)
    bbox = pts.max(0) - pts.min(0)
    diameter = np.linalg.norm(bbox)

    is_huge = diameter > args.skeleton_diameter or L1 > args.skeleton_diameter
    is_elongated = aspect_ratio > args.skeleton_ar

    if debug:
        print(f"{prefix} 点数={len(indices):,} 直径={diameter:.3f}m L1={L1:.3f}m AR={aspect_ratio:.2f}")
        print(f"{prefix}   huge={is_huge} (阈值{args.skeleton_diameter}) elongated={is_elongated} (阈值{args.skeleton_ar})")

    if not (is_huge or is_elongated) or L1 < 0.15:
        if debug: print(f"{prefix} → 不拆分 ({'太小' if L1<0.15 else '不合格'})")
        return [indices]

    scalars = np.dot(pts - center, v1)
    s_min, s_max = scalars.min(), scalars.max()
    if s_max - s_min < 0.1:
        if debug: print(f"{prefix} → 不拆分 (投影长度{s_max-s_min:.3f}m < 0.1)")
        return [indices]

    nbins = min(15, max(5, int((s_max - s_min) / 0.03)))
    bins = np.linspace(s_min, s_max, nbins + 1)
    bin_idx = np.digitize(scalars, bins) - 1

    density = np.zeros(nbins)
    for b in range(nbins):
        density[b] = (bin_idx == b).sum()

    try:
        from scipy.ndimage import gaussian_filter
        from scipy.signal import find_peaks
        density_s = gaussian_filter(density, sigma=1.0)
        peaks, _ = find_peaks(density_s, distance=max(1, nbins//5))
    except ImportError:
        return [indices]

    if debug:
        print(f"{prefix}   密度曲线: {density.astype(int).tolist()}")
        print(f"{prefix}   平滑后:    {[round(x) for x in density_s.tolist()]}")
        print(f"{prefix}   峰值位置:  {peaks.tolist()}")

    if len(peaks) < 2:
        if debug: print(f"{prefix} → 不拆分 (峰值<2个)")
        return [indices]

    first_p, last_p = peaks[0], peaks[-1]
    if first_p >= last_p:
        return [indices]

    valley_rel = first_p + np.argmin(density_s[first_p:last_p+1])
    valley_h = density_s[valley_rel]
    peak_h = max(density_s[first_p], density_s[last_p])
    ratio = valley_h / (peak_h + 1e-6)

    current_thresh = args.skeleton_valley
    if L1 > 0.5:
        current_thresh = max(current_thresh, 0.95)
    # 递归越深要求波谷越深才切，避免把单株叶片波动当植株间隙
    current_thresh = min(current_thresh, 0.85 - depth * 0.20)  # depth0:0.85, depth1:0.65, depth2:0.45
    # 递归深度越深，越难再切（防止过分割：同一株的叶片波动不应切）
    if depth >= 1:
        current_thresh = max(current_thresh, 0.75)

    if debug:
        print(f"{prefix}   最深波谷: bin#{valley_rel} 谷值={valley_h:.0f} 峰值={peak_h:.0f} ratio={ratio:.3f} 阈值={current_thresh}")

    if ratio >= current_thresh:
        if debug: print(f"{prefix} → 不拆分 (ratio {ratio:.3f} >= {current_thresh})")
        return [indices]

    # 执行切割
    split_scalar = bins[valley_rel] + (bins[1]-bins[0]) * 0.5
    prev_c = next_c = None
    for off in range(1, nbins):
        if prev_c is None and valley_rel - off >= 0:
            prev_c = pts[bin_idx == valley_rel - off].mean(0) if (bin_idx == valley_rel - off).sum() > 0 else None
        if next_c is None and valley_rel + off < nbins:
            next_c = pts[bin_idx == valley_rel + off].mean(0) if (bin_idx == valley_rel + off).sum() > 0 else None
        if prev_c is not None and next_c is not None:
            break

    if prev_c is not None and next_c is not None:
        cut_normal = next_c - prev_c
        cut_normal /= np.linalg.norm(cut_normal) + 1e-6
        cut_center = pts[bin_idx == valley_rel].mean(0) if (bin_idx == valley_rel).sum() > 0 else (prev_c + next_c) * 0.5
    else:
        cut_normal = v1
        cut_center = center + split_scalar * v1

    # 以波谷处切割面为界，正常应该沿主轴横向切割
    dists = np.dot(pts - cut_center, cut_normal)
    idx1 = indices[dists < 0]
    idx2 = indices[dists >= 0]

    if len(idx1) > 10 and len(idx2) > 10:
        # 🔥 不对称检测：碎片比例 < 15% → 拒绝切割
        total_pts = len(idx1) + len(idx2)
        ratio_small = min(len(idx1), len(idx2)) / total_pts
        if ratio_small < 0.15:
            if debug: print(f"{prefix} → 拒绝切割: 碎片 {ratio_small:.1%} < 15%")
            return [indices]
        if debug: print(f"{prefix} 🔪 切割! 左={len(idx1):,} 右={len(idx2):,} → 递归...")
        return (skeleton_split_cluster(idx1, all_xyz, args, depth+1, debug) +
                skeleton_split_cluster(idx2, all_xyz, args, depth+1, debug))

    if debug: print(f"{prefix} → 切割后子簇太小, 保留")
    return [indices]
def cluster_split_stage2(xyz, inst_label, cluster_id, args):
    """Stage 2: 对大簇用聚类算法二次拆分"""
    mask = inst_label == cluster_id
    idxs = np.where(mask)[0]
    points = xyz[idxs]

    if points.shape[0] < args.split_min_points:
        return [idxs]  # too small, keep as is

    # 计算直径
    bbox = points.max(0) - points.min(0)
    diameter = np.linalg.norm(bbox)
    if diameter < args.split_diameter:
        return [idxs]  # small enough

    # 构建点云并聚类
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if args.split_method == 'meanshift':
        from sklearn.cluster import MeanShift
        ms = MeanShift(bandwidth=args.bandwidth, bin_seeding=True, min_bin_freq=10, n_jobs=-1)
        sub_labels = ms.fit_predict(points)
    elif args.split_method == 'dbscan':
        from sklearn.cluster import DBSCAN
        db = DBSCAN(eps=args.bandwidth, min_samples=10, n_jobs=-1)
        sub_labels = db.fit_predict(points)
    elif args.split_method == 'euclidean':
        sub_labels = np.array(pcd.cluster_dbscan(eps=args.bandwidth, min_points=10, print_progress=False))
    else:
        # watershed_3d via cabbage_pheno
        from cabbage_pheno.instance import InstanceClusterer
        cfg_dict = yaml.safe_load(open(args.cluster_config))
        clusterer = InstanceClusterer(cfg_dict)
        sub_labels, _ = clusterer.cluster(pcd)
        sub_labels = sub_labels - 1  # 转 0-indexed

    # 分配子簇
    sub_clusters = []
    unique_labels = sorted(set(sub_labels))
    for ul in unique_labels:
        if ul < 0: continue
        sub_idxs = idxs[sub_labels == ul]
        if len(sub_idxs) >= args.npoint_thresh:
            sub_clusters.append(sub_idxs)

    return sub_clusters if sub_clusters else [idxs]

def hybrid_split(xyz, semantic_pred, inst_label_pg, args):
    """三阶段流水线 + 遗落甘蓝点补充分割"""
    # 确保是 numpy
    if hasattr(semantic_pred, 'cpu'):
        semantic_pred = semantic_pred.cpu().numpy()
    final_label = np.zeros_like(inst_label_pg)
    next_id = 1
    pg_count = inst_label_pg.max()
    total_skeleton_splits = 0

    # 1. 处理 PointGroup 已分配的实例
    for cid in range(1, pg_count+1):
        sub_clusters = cluster_split_stage2(xyz, inst_label_pg, cid, args)

        if args.skeleton_split:
            skeleton_refined = []
            for sub in sub_clusters:
                parts = skeleton_split_cluster(sub, xyz, args)
                skeleton_refined.extend(parts)
                if len(parts) > 1:
                    total_skeleton_splits += 1
            sub_clusters = skeleton_refined

        for sub in sub_clusters:
            final_label[sub] = next_id
            next_id += 1

    # 2. 🔥 处理 PointGroup 实例分割遗漏的甘蓝点
    mask_cabbage = (semantic_pred == 1)
    mask_assigned = (inst_label_pg > 0)
    mask_leftover = mask_cabbage & (~mask_assigned)
    leftover_count = mask_leftover.sum()

    if leftover_count > args.npoint_thresh:
        leftover_indices = np.where(mask_leftover)[0]
        leftover_pts = xyz[leftover_indices]

        # 对遗落甘蓝点做独立聚类
        pcd_left = o3d.geometry.PointCloud()
        pcd_left.points = o3d.utility.Vector3dVector(leftover_pts)

        if args.split_method == 'meanshift':
            from sklearn.cluster import MeanShift
            ms = MeanShift(bandwidth=args.bandwidth, bin_seeding=True, min_bin_freq=10, n_jobs=-1)
            sub_labels = ms.fit_predict(leftover_pts)
        elif args.split_method == 'dbscan':
            from sklearn.cluster import DBSCAN
            db = DBSCAN(eps=args.bandwidth, min_samples=10, n_jobs=-1)
            sub_labels = db.fit_predict(leftover_pts)
        elif args.split_method == 'euclidean':
            sub_labels = np.array(pcd_left.cluster_dbscan(eps=args.bandwidth, min_points=10, print_progress=False))
        else:
            sub_labels = np.array(pcd_left.cluster_dbscan(eps=args.bandwidth, min_points=10, print_progress=False))

        leftover_clusters = []
        # 遗落点恢复要求最小簇更大，避免碎片
        leftover_min_pts = max(args.npoint_thresh * 3, 300)
        for ul in sorted(set(sub_labels)):
            if ul < 0: continue
            idxs_sub = leftover_indices[sub_labels == ul]
            if len(idxs_sub) >= leftover_min_pts:
                leftover_clusters.append(idxs_sub)

        if leftover_clusters:
            # 对遗落簇也做骨架拆分
            if args.skeleton_split:
                skeleton_refined = []
                for lc in leftover_clusters:
                    parts = skeleton_split_cluster(lc, xyz, args)
                    skeleton_refined.extend(parts)
                    if len(parts) > 1:
                        total_skeleton_splits += 1
                leftover_clusters = skeleton_refined

            for lc in leftover_clusters:
                final_label[lc] = next_id
                next_id += 1

        print(f"    🔥 遗落甘蓝点: {leftover_count:,} → {len(leftover_clusters)} 额外实例")
    else:
        print(f"    遗落甘蓝点: {leftover_count:,} (< {args.npoint_thresh}, 忽略)")

    n_original = pg_count
    n_final = final_label.max()
    msg = f"    PG原始: {n_original} 实例"
    if leftover_count > args.npoint_thresh:
        msg += f" + 遗落补回"
    if total_skeleton_splits > 0:
        msg += f" | 骨架拆分了 {total_skeleton_splits} 个簇"
    msg += f" → 最终: {n_final} 实例"
    print(msg)
    return final_label

def save_results(xyz, semantic_pred, inst_label_pg, inst_label_hybrid, output_dir, scene_name):
    """保存 PointGroup 原始结果 + 混合结果"""
    os.makedirs(output_dir, exist_ok=True)

    def color_save(label, suffix):
        np.random.seed(42)
        palette = np.random.rand(label.max()+1, 3)
        palette[0] = [0.5,0.5,0.5]
        colors = palette[label]
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(xyz)
        pc.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(os.path.join(output_dir, f'{scene_name}_{suffix}.ply'), pc)

    # 语义
    sem_c = np.zeros((xyz.shape[0],3))
    sem_c[semantic_pred.cpu().numpy()==0] = [0.5,0.5,0.5]
    sem_c[semantic_pred.cpu().numpy()==1] = [0,1,0]
    sp = o3d.geometry.PointCloud(); sp.points=o3d.utility.Vector3dVector(xyz); sp.colors=o3d.utility.Vector3dVector(sem_c)
    o3d.io.write_point_cloud(os.path.join(output_dir,f'{scene_name}_semantic.ply'), sp)

    color_save(inst_label_pg, 'instance_pg')
    color_save(inst_label_hybrid, 'instance_hybrid')

    print(f"    结果: {output_dir}/{scene_name}_instance_pg.ply (PointGroup)")
    print(f"    结果: {output_dir}/{scene_name}_instance_hybrid.ply (混合)")

def main():
    args = parse_args()
    input_files = sorted(glob.glob(args.input)) if '*' in args.input else [args.input]
    print(f"找到 {len(input_files)} 个文件")

    # Init PointGroup
    config_path = os.path.join(os.path.dirname(__file__), '..', 'PointGroup_Ours', args.config) if not os.path.isabs(args.config) else args.config
    sys.argv = ['infer', '--config', config_path]
    from util.config import cfg; cfg.task = 'test'
    if args.cluster_radius: cfg.cluster_radius = args.cluster_radius
    model, model_fn = load_pointgroup(cfg)
    print(f"模型加载完成, cluster_radius={cfg.cluster_radius}")

    for ply_path in input_files:
        scene = os.path.splitext(os.path.basename(ply_path))[0]
        print(f"\n{'='*50}\n处理: {scene}")

        batch, xyz, _ = ply_to_pointgroup_input(ply_path, cfg)
        N = xyz.shape[0]
        print(f"  点数: {N:,}")

        # Stage 1: PointGroup
        semantic_pred, inst_pg = pointgroup_proposals(
            batch, model, model_fn, cfg, N,
            args.score_thresh, args.npoint_thresh, args.nms_thresh)

        n_c = (semantic_pred==1).sum().item()
        print(f"  语义: 甘蓝={n_c:,}, 背景={N-n_c:,}")
        print(f"  PointGroup 原始: {inst_pg.max()} 实例")

        # Stage 2: 混合拆分
        inst_hybrid = hybrid_split(xyz, semantic_pred, inst_pg, args)

        # Save
        save_results(xyz, semantic_pred, inst_pg, inst_hybrid, args.output, scene)

    print(f"\n完成! 结果: {args.output}/")


if __name__ == '__main__':
    main()
