import os, sys, argparse, logging, json, copy
import numpy as np
import pandas as pd
import open3d as o3d
import random, torch, yaml
from scipy.spatial import cKDTree

from cabbage_pheno.io import read_point_cloud
from cabbage_pheno.service.pipeline import preprocess_point_cloud
from cabbage_pheno.instance import InstanceClusterer
from cabbage_pheno.traits import TraitCalculator
from cabbage_pheno.viz import Visualizer


def save_instance_pcd(pcd, labels, path):
    """保存按实例上色的点云"""
    if labels is None or len(labels) == 0:
        return
    valid = labels >= 0
    n_inst = labels.max() + 1 if labels.max() >= 0 else 0
    if n_inst == 0:
        return
    np.random.seed(42)
    palette = np.random.rand(n_inst + 1, 3)
    palette[0] = [0.5, 0.5, 0.5]
    colors = palette[labels - labels.min()] if labels.min() < 0 else palette[labels]
    pc = o3d.geometry.PointCloud()
    pc.points = pcd.points
    pc.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pc)
    return n_inst


def cluster_partial(cabbage_pcd, cfg, method, pca_enable=None, fragment_enable=None):
    """使用临时配置运行聚类。
    pca_enable/fragment_enable: None = 使用配置文件的值, True/False = 强制覆盖"""
    import copy
    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp['instance']['method'] = method
    if pca_enable is not None:
        cfg_tmp['instance']['pca_split']['enable'] = pca_enable
    if fragment_enable is not None:
        cfg_tmp['instance']['fragment_voting']['enable'] = fragment_enable
    clusterer = InstanceClusterer(cfg_tmp)
    return clusterer.cluster(cabbage_pcd)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CabbagePipeline")

def set_determinism(cfg):
    pipeline_cfg = cfg.get('pipeline', {}) if isinstance(cfg, dict) else {}
    seed = pipeline_cfg.get('seed', None)
    if seed is None: return
    try: seed = int(seed)
    except: return
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def _load_pointgroup(cfg):
    """加载 PointGroup 模型 (只执行一次, 供分块推理复用).
    Returns (model, model_fn, pg_cfg, seg_cfg, pointgroup_ops)
    """
    seg_cfg = cfg.get('segmentation', {})
    pg_cfg_path = seg_cfg.get('pointgroup_config', 'config/pointgroup_cabbage.yaml')
    if pg_cfg_path.startswith('PointGroup_Ours/'):
        pg_cfg_path = pg_cfg_path[len('PointGroup_Ours/'):]

    pg_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PointGroup_Ours')
    sys.path.insert(0, pg_root)
    sys.argv = ['main', '--config', os.path.join(pg_root, pg_cfg_path)]

    from util.config import cfg as pg_cfg
    pg_cfg.task = 'test'
    pg_cfg.exp_path = os.path.join(pg_root, pg_cfg.exp_path)
    from model.pointgroup.pointgroup import PointGroup as Network, model_fn_decorator
    import util.utils as utils
    from lib.pointgroup_ops.functions import pointgroup_ops

    model = Network(pg_cfg).cuda()
    model.eval()
    start_epoch = utils.checkpoint_restore(model, pg_cfg.exp_path,
        pg_cfg.config.split('/')[-1][:-5], True, pg_cfg.test_epoch)
    logger.info(f"PointGroup loaded: epoch {start_epoch-1}")
    model_fn = model_fn_decorator(test=True)
    return model, model_fn, pg_cfg, seg_cfg, pointgroup_ops


def _pointgroup_forward(xyz_shifted, rgb, model, model_fn, pg_cfg, pgo):
    """单次 PointGroup 前向传播 (坐标已平移到原点).
    xyz_shifted: (N,3) 已减去 min, 匹配训练坐标范围
    rgb: (N,3) 归一化到 [-1,1]
    Returns: (semantic_scores, proposals_idx, proposals_offset, scores_pred)
      semantic_scores: (N, 2) 原始 logits
    """
    xyz_scaled = xyz_shifted * pg_cfg.scale
    N = xyz_scaled.shape[0]
    locs = torch.cat([torch.LongTensor(N, 1).fill_(0), torch.from_numpy(xyz_scaled).long()], 1)
    voxel_locs, p2v_map, v2p_map = pgo.voxelization_idx(locs, 1, pg_cfg.mode)

    # 确定 spatial_shape: 体素坐标上限, 至少 full_scale[0], 最大 full_scale[1]
    raw_shape = (locs.max(0)[0][1:] + 1).numpy()
    spatial_shape = np.clip(raw_shape, pg_cfg.full_scale[0], pg_cfg.full_scale[1])

    batch = {
        'locs': locs, 'voxel_locs': voxel_locs,
        'p2v_map': p2v_map, 'v2p_map': v2p_map,
        'locs_float': torch.from_numpy(xyz_shifted),
        'feats': torch.from_numpy(rgb),
        'offsets': torch.tensor([0, N], dtype=torch.int),
        'spatial_shape': spatial_shape,
        'id': torch.tensor([0]),
    }

    with torch.no_grad():
        preds = model_fn(batch, model, pg_cfg.test_epoch)

    semantic_scores = preds['semantic']  # (N, 2) raw logits
    pt_offsets = preds['pt_offsets'].cpu().numpy()  # (N, 3) offset vectors
    proposals_idx = preds['proposals'][0]
    proposals_offset = preds['proposals'][1]
    scores_pred = torch.sigmoid(preds['score'].view(-1))
    return semantic_scores, proposals_idx, proposals_offset, scores_pred, pt_offsets


def _extract_instances_from_proposals(proposals_idx, proposals_offset, scores_pred,
                                        N, seg_cfg):
    """从原始 proposals 提取实例标签 (含 NMS 过滤).
    Returns: inst_labels (N,) 0=背景/unassigned, 1+=instance
    """
    nProp = proposals_offset.shape[0] - 1
    if nProp == 0:
        return np.zeros(N, dtype=np.int64)

    proposals_mask = torch.zeros((nProp, N), dtype=torch.int, device=scores_pred.device)
    proposals_mask[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1

    # Score threshold
    score_mask = scores_pred > seg_cfg.get('pg_score_thresh', 0.1)
    proposals_mask, scores_pred = proposals_mask[score_mask], scores_pred[score_mask]
    if proposals_mask.shape[0] > 0:
        npoint_mask = proposals_mask.sum(1) > seg_cfg.get('pg_npoint_thresh', 30)
        proposals_mask, scores_pred = proposals_mask[npoint_mask], scores_pred[npoint_mask]

    # NMS by IoU
    if proposals_mask.shape[0] > 1:
        proposals_f = proposals_mask.float()
        inter = torch.mm(proposals_f, proposals_f.t())
        pn = proposals_f.sum(1)
        ious = inter / (pn.unsqueeze(-1) + pn.unsqueeze(0) - inter + 1e-6)
        ixs = scores_pred.cpu().numpy().argsort()[::-1]
        pick = []
        ious_np = ious.cpu().numpy()
        while len(ixs) > 0:
            i = ixs[0]
            pick.append(i)
            rm = np.where(ious_np[i, ixs[1:]] > 0.5)[0] + 1
            ixs = np.delete(ixs, rm)
            ixs = np.delete(ixs, 0)
        proposals_mask = proposals_mask[np.array(pick)].cpu().numpy()
    elif proposals_mask.shape[0] == 1:
        proposals_mask = proposals_mask.cpu().numpy()
    else:
        proposals_mask = np.zeros((0, N), dtype=np.int64)

    inst = np.zeros(N, dtype=np.int64)
    for i in range(len(proposals_mask)):
        inst[proposals_mask[i] == 1] = i + 1
    return inst


def _extract_instances_greedy(proposals_idx, proposals_offset, scores_pred,
                               N, seg_cfg):
    """SoftGroup 风格: 自顶向下贪心选择 (替代 NMS).
    高分提案先选并独占其点，低分提案在剩余点中竞争。
    """
    nProp = proposals_offset.shape[0] - 1
    if nProp == 0:
        return np.zeros(N, dtype=np.int64)

    # Score & npoint filter
    score_mask = scores_pred > seg_cfg.get('pg_score_thresh', 0.05)
    if score_mask.sum() == 0:
        return np.zeros(N, dtype=np.int64)

    proposals_mask = torch.zeros((nProp, N), dtype=torch.int, device=scores_pred.device)
    proposals_mask[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1
    proposals_mask = proposals_mask[score_mask]
    scores_filtered = scores_pred[score_mask]

    if proposals_mask.shape[0] > 0:
        npoint_mask = proposals_mask.sum(1) > seg_cfg.get('pg_npoint_thresh', 20)
        proposals_mask = proposals_mask[npoint_mask]
        scores_filtered = scores_filtered[npoint_mask]

    nProp_f = proposals_mask.shape[0]
    if nProp_f == 0:
        return np.zeros(N, dtype=np.int64)

    # Greedy top-down selection
    order = scores_filtered.cpu().numpy().argsort()[::-1]
    remaining = torch.ones(N, dtype=torch.bool)
    keep = np.zeros(nProp_f, dtype=bool)

    for pi in order:
        pts = proposals_mask[pi] == 1
        avail = (remaining & pts).sum().float()
        total = pts.sum().float()
        if avail / max(total, 1) > 0.5:  # >50% of proposal still available
            keep[pi] = True
            remaining[pts] = False

    proposals_mask = proposals_mask[keep].cpu().numpy()
    inst = np.zeros(N, dtype=np.int64)
    for i in range(len(proposals_mask)):
        inst[proposals_mask[i] == 1] = i + 1
    return inst


def _iach_cluster(xyz, pt_offsets, cfg):
    """IACH 风格聚类: PointGroup offset 移位 + DBSCAN.
    核心思想 (IACH): 用模型预测的 offset 把点移向实例中心, 然后 DBSCAN 聚类.
    不需要 BFS 和 score 分支, 比 PointGroup 更轻量.
    
    Returns: inst_labels (N,) 0=background, 1+=instance
    """
    from sklearn.cluster import DBSCAN
    iach_cfg = cfg.get('segmentation', {}).get('iach', {})
    eps = iach_cfg.get('eps', 0.05)
    min_samples = iach_cfg.get('min_samples', 100)

    # Shift points toward instance centers
    shifted = xyz + pt_offsets
    logger.info(f"IACH: DBSCAN(eps={eps}m, min_samples={min_samples}) on {len(xyz):,} shifted points")
    
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(shifted)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    logger.info(f"IACH found {n_clusters} clusters")
    return labels, n_clusters


def _post_semantic_filter(mask_cabbage, xyz, cfg):
    """语义后几何过滤: 去除语义误判的地面凸起.
    - Z 高度: 去掉最低 z_percentile 以下的点 (地面较低)
    - 密度: 去掉孤立稀疏点簇 (地面凸起通常分散)
    Returns: refined mask (bool array)
    """
    seg_cfg = cfg.get('segmentation', {})
    post_cfg = seg_cfg.get('post_semantic_filter', {})
    if not post_cfg.get('enable', True):
        return mask_cabbage

    mask = mask_cabbage.copy()
    n_before = mask.sum()
    z = xyz[:, 2]  # height axis

    # 1. Z 高度过滤: 去掉最低 z_percentile 的点
    z_pct = post_cfg.get('z_percentile', 3)
    if z_pct > 0:
        z_thresh = np.percentile(z[mask], z_pct) if mask.sum() > 0 else 0
        mask[mask & (z < z_thresh)] = False
        n_z = mask.sum()
        if n_before - n_z > 0:
            logger.info(f"  Z过滤 (p{z_pct}<{z_thresh:.3f}m): {n_before - n_z:,} 点移除")
    else:
        n_z = n_before

    # 2. 密度过滤: 小半径邻居 < min_neighbors → 移除
    ror_radius = post_cfg.get('radius_outlier_radius', 0.03)
    ror_nb = post_cfg.get('radius_outlier_min_neighbors', 8)
    if ror_radius > 0 and mask.sum() > 100:
        tmp_pcd = o3d.geometry.PointCloud()
        tmp_pcd.points = o3d.utility.Vector3dVector(xyz[mask])
        _, dense_idx = tmp_pcd.remove_radius_outlier(nb_points=ror_nb, radius=ror_radius)
        dense_idx = np.asarray(dense_idx, dtype=np.int64)
        new_mask = np.zeros(mask.sum(), dtype=bool)
        if len(dense_idx) > 0:
            new_mask[dense_idx] = True
        global_idx = np.where(mask)[0]
        mask[global_idx[~new_mask]] = False
        n_remain = mask.sum()
        if n_z - n_remain > 0:
            logger.info(f"  密度过滤 (r={ror_radius}m, min_nb={ror_nb}): {n_z - n_remain:,} 点移除")

    n_after = mask.sum()
    if n_before - n_after > 0:
        logger.info(f"  语义后过滤: {n_before:,} → {n_after:,} (-{n_before-n_after:,})")
    return mask


def pointgroup_segment_blocks(pcd, cfg, method='hybrid', selection='nms'):
    """分块 PointGroup 推理 + 提案筛选策略.
    selection: 'nms' (PointGroup) 或 'greedy' (SoftGroup 贪心)
    """
    model, model_fn, pg_cfg, seg_cfg, pgo = _load_pointgroup(cfg)
    extract_fn = _extract_instances_greedy if selection == 'greedy' else _extract_instances_from_proposals

    block_cfg = seg_cfg.get('block_inference', {})
    block_size = float(block_cfg.get('block_size', 2.0))
    overlap = float(block_cfg.get('overlap', 0.5))
    max_single = float(block_cfg.get('max_single_extent', 8.0))
    merge_iou = float(block_cfg.get('merge_iou_thresh', 0.25))

    xyz = np.asarray(pcd.points, dtype=np.float32)
    rgb_orig = np.asarray(pcd.colors, dtype=np.float32) if len(pcd.colors) > 0 else np.ones_like(xyz)
    rgb = (rgb_orig * 2 - 1).astype(np.float32)

    extent = xyz.max(0) - xyz.min(0)
    logger.info(f"田地范围: X={extent[0]:.1f}m Y={extent[1]:.1f}m Z={extent[2]:.2f}m")

    # 小田地不需要分块
    if max(extent) <= max_single and block_cfg.get('enable', True):
        logger.info(f"田地 ≤{max_single}m, 无需分块, 使用单次推理")
        return pointgroup_segment(pcd, cfg, method)

    # --- 切块 ---
    stride = max(block_size - overlap, 0.1)
    x_min, y_min = xyz[:, 0].min(), xyz[:, 1].min()
    x_max, y_max = xyz[:, 0].max(), xyz[:, 1].max()

    blocks = []
    xs = x_min
    while xs < x_max:
        ys = y_min
        while ys < y_max:
            mask = (xyz[:, 0] >= xs) & (xyz[:, 0] < xs + block_size) & \
                   (xyz[:, 1] >= ys) & (xyz[:, 1] < ys + block_size)
            if mask.sum() >= 100:
                idx = np.where(mask)[0]
                blocks.append({
                    'global_idx': idx,
                    'origin': xyz[idx].min(0),  # 块内坐标原点
                    'x_range': (xs, xs + block_size),
                    'y_range': (ys, ys + block_size),
                })
            ys += stride
        xs += stride

    logger.info(f"切分为 {len(blocks)} 个块 (block={block_size}m, overlap={overlap}m)")

    # --- 逐块推理 ---
    N_total = len(xyz)
    # 语义: 每点记录被预测为甘蓝的次数和总预测次数
    cabbage_votes = np.zeros(N_total, dtype=np.int32)
    total_votes = np.zeros(N_total, dtype=np.int32)

    # 实例: 收集所有 proposal (全局点索引集合)
    all_proposals = []   # list of np.array (global indices)
    next_inst_offset = 0
    sem_thresh = seg_cfg.get('pg_sem_thresh', 0)

    for bi, block in enumerate(blocks):
        gidx = block['global_idx']
        block_xyz = xyz[gidx] - block['origin']
        block_rgb = rgb[gidx]

        sem_scores, prop_idx, prop_off, scores, _ = _pointgroup_forward(
            block_xyz, block_rgb, model, model_fn, pg_cfg, pgo)

        inst_local = extract_fn(
            prop_idx, prop_off, scores, len(gidx), seg_cfg)

        # 记录语义投票 (使用概率阈值代替 argmax)
        for i, gi in enumerate(gidx):
            total_votes[gi] += 1
            if sem_thresh > 0:
                prob = float(torch.softmax(sem_scores, dim=1)[i, 1])
                if prob > sem_thresh:
                    cabbage_votes[gi] += 1
            else:
                if sem_scores.max(1)[1][i] == 1:
                    cabbage_votes[gi] += 1

        # 收集 proposals (转为全局点索引)
        n_inst = inst_local.max()
        for inst_id in range(1, int(n_inst) + 1):
            pts_local = np.where(inst_local == inst_id)[0]
            if len(pts_local) >= seg_cfg.get('pg_npoint_thresh', 30):
                all_proposals.append(gidx[pts_local])

        if (bi + 1) % max(1, len(blocks) // 5) == 0 or bi == len(blocks) - 1:
            logger.info(f"  块 {bi+1}/{len(blocks)}: {len(gidx):,}pts, {n_inst} proposals")

    # --- 语义合并: 多数投票 ---
    mask_cabbage = np.zeros(N_total, dtype=bool)
    voted = total_votes > 0
    mask_cabbage[voted] = cabbage_votes[voted] >= (total_votes[voted] * 0.5)
    mask_cabbage = _post_semantic_filter(mask_cabbage, xyz, cfg)
    n_cabbage = mask_cabbage.sum()
    logger.info(f"语义合并: {n_cabbage:,}/{N_total:,} 甘蓝点 ({100*n_cabbage/N_total:.1f}%)")

    # --- 实例合并: Union-Find 跨块合并重叠 proposal ---
    n_prop = len(all_proposals)
    logger.info(f"共 {n_prop} 个 proposals, 开始跨块合并...")

    if n_prop > 0:
        parent = list(range(n_prop))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        # 构建 proposal→blocks 映射, 只比较不同块的 proposal
        prop_to_block = {}
        prop_sizes = []
        for pi, pts in enumerate(all_proposals):
            prop_sizes.append(len(pts))
            # 找到这个 proposal 属于哪个块 (通过第一个点的块归属)
            for bi, block in enumerate(blocks):
                if pts[0] in set(block['global_idx']):
                    prop_to_block[pi] = bi
                    break
            else:
                prop_to_block[pi] = -1

        # 用哈希加速: point → list of proposal indices
        point_to_props = {}
        for pi, pts in enumerate(all_proposals):
            for p in pts:
                point_to_props.setdefault(int(p), []).append(pi)

        # 只合并不同块的 proposal (同块的已被 NMS 处理)
        merged_pairs = set()
        for point, prop_list in point_to_props.items():
            if len(prop_list) < 2:
                continue
            for i in range(len(prop_list)):
                for j in range(i + 1, len(prop_list)):
                    pa, pb = prop_list[i], prop_list[j]
                    if prop_to_block.get(pa, -1) == prop_to_block.get(pb, -1):
                        continue  # 同块不合并
                    pair = (min(pa, pb), max(pa, pb))
                    if pair in merged_pairs:
                        continue
                    merged_pairs.add(pair)
                    # 计算重叠度
                    set_a = set(all_proposals[pa])
                    set_b = set(all_proposals[pb])
                    inter = len(set_a & set_b)
                    union_size = len(set_a | set_b)
                    iou = inter / union_size if union_size > 0 else 0
                    if iou >= merge_iou:
                        union(pa, pb)

        # 分配全局实例 ID
        root_to_id = {}
        inst_global = np.zeros(N_total, dtype=np.int64)
        next_id = 1

        for pi, pts in enumerate(all_proposals):
            root = find(pi)
            if root not in root_to_id:
                root_to_id[root] = next_id
                next_id += 1
            inst_global[pts] = root_to_id[root]

        n_merged = next_id - 1
        logger.info(f"实例合并: {n_prop} proposals → {n_merged} instances")
    else:
        inst_global = np.zeros(N_total, dtype=np.int64)
        n_merged = 0

    # --- 构建输出 ---
    cabbage_xyz = xyz[mask_cabbage]
    cabbage_pcd = o3d.geometry.PointCloud()
    cabbage_pcd.points = o3d.utility.Vector3dVector(cabbage_xyz)
    cabbage_pcd.colors = o3d.utility.Vector3dVector(rgb_orig[mask_cabbage])

    inst_cabbage = inst_global[mask_cabbage]
    logger.info(f"PointGroup(分块): cabbage={n_cabbage:,}pts, {n_merged} instances")
    return cabbage_pcd, inst_cabbage


def pointgroup_segment(pcd, cfg, method='hybrid', selection='nms'):
    """Stage 1: PointGroup semantic + instance segmentation (单次推理).
    method: 'hybrid' | 'pointgroup' | 'clustering' — 用于决定是否归一化坐标
    selection: 'nms' (PointGroup) 或 'greedy' (SoftGroup 贪心)
    注意: hybrid/pointgroup 模式建议使用 pointgroup_segment_blocks 进行分块推理
    """
    model, model_fn, pg_cfg, seg_cfg, pgo = _load_pointgroup(cfg)
    extract_fn = _extract_instances_greedy if selection == 'greedy' else _extract_instances_from_proposals

    xyz = np.asarray(pcd.points, dtype=np.float32)
    rgb_orig = np.asarray(pcd.colors, dtype=np.float32) if len(pcd.colors) > 0 else np.ones_like(xyz)
    rgb = (rgb_orig * 2 - 1).astype(np.float32)
    xyz_mid = xyz - xyz.min(0)
    # 坐标归一化: 仅 clustering 模式安全 (只用语义, 不影响 offset/聚类参数)
    # hybrid/pointgroup 模式需要原始尺度, 否则 offset 预测和 cluster_radius 会失效
    if method == 'clustering':
        max_extent_m = float(xyz_mid.max())
        voxel_extent = max_extent_m * pg_cfg.scale
        if voxel_extent > 400:
            scale_factor = 400.0 / voxel_extent
            logger.info(f"坐标归一化(clustering): 原始范围 {voxel_extent:.0f} 体素 → 缩放至 ~400 (因子 {scale_factor:.3f})")
            xyz_mid = xyz_mid * scale_factor

    semantic_scores, proposals_idx, proposals_offset, scores_pred, _ = _pointgroup_forward(
        xyz_mid, rgb, model, model_fn, pg_cfg, pgo)

    N = len(xyz)
    # 语义阈值: 甘蓝 softmax 概率 > pg_sem_thresh
    sem_thresh = seg_cfg.get('pg_sem_thresh', 0)
    if sem_thresh > 0:
        cabbage_prob = torch.softmax(semantic_scores, dim=1)[:, 1].cpu().numpy()
        mask_cabbage = cabbage_prob > sem_thresh
        logger.info(f"语义阈值: prob>{sem_thresh}, 甘蓝={mask_cabbage.sum():,}/{N:,} pts")
    else:
        mask_cabbage = semantic_scores.max(1)[1].cpu().numpy() == 1
    mask_cabbage = _post_semantic_filter(mask_cabbage, xyz, cfg)
    cabbage_xyz = xyz[mask_cabbage]
    cabbage_pcd = o3d.geometry.PointCloud()
    cabbage_pcd.points = o3d.utility.Vector3dVector(cabbage_xyz)
    cabbage_pcd.colors = o3d.utility.Vector3dVector(rgb_orig[mask_cabbage])

    inst_pg = extract_fn(proposals_idx, proposals_offset, scores_pred, N, seg_cfg)
    cabbage_indices = np.where(mask_cabbage)[0]
    inst_cabbage = inst_pg[cabbage_indices]

    n_sem = mask_cabbage.sum()
    n_inst = (inst_cabbage > 0).sum()
    logger.info(f"PointGroup: cabbage={n_sem:,}pts, assigned={n_inst:,}pts, {inst_cabbage.max()} proposals")
    return cabbage_pcd, inst_cabbage


# ============================================================
# SoftGroup 推理 (自顶向下贪心选择, 对比 PointGroup NMS)
# ============================================================
def _load_softgroup(cfg):
    """加载 SoftGroup 模型. Returns (model, model_fn, sg_cfg, seg_cfg, pgo)"""
    seg_cfg = cfg.get('segmentation', {})
    sg_cfg_path = seg_cfg.get('softgroup_config', 'SoftGroup_Ours/config/softgroup_cabbage.yaml')
    if sg_cfg_path.startswith('SoftGroup_Ours/'):
        sg_cfg_path = sg_cfg_path[len('SoftGroup_Ours/'):]

    sg_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SoftGroup_Ours')
    sys.path.insert(0, sg_root)
    sys.argv = ['main', '--config', os.path.join(sg_root, sg_cfg_path)]

    from util.config import cfg as sg_cfg
    sg_cfg.task = 'test'
    sg_cfg.exp_path = os.path.join(sg_root, sg_cfg.exp_path) if hasattr(sg_cfg, 'exp_path') else os.path.join(sg_root, 'exp/softgroup')
    from model.softgroup.softgroup import SoftGroup as Network, model_fn_decorator, extract_instances_softgroup
    import util.utils as utils
    from lib.pointgroup_ops.functions import pointgroup_ops

    model = Network(sg_cfg).cuda()
    start_epoch = utils.checkpoint_restore(model, sg_cfg.exp_path,
        sg_cfg.config.split('/')[-1][:-5], True, sg_cfg.test_epoch)
    logger.info(f"SoftGroup loaded: epoch {start_epoch-1}")
    model_fn = model_fn_decorator(test=True)
    return model, model_fn, sg_cfg, seg_cfg, pointgroup_ops, extract_instances_softgroup


def softgroup_segment(pcd, cfg):
    """单次 SoftGroup 推理 (自顶向下贪心选择)."""
    model, model_fn, sg_cfg, seg_cfg, pgo, extract_fn = _load_softgroup(cfg)

    xyz = np.asarray(pcd.points, dtype=np.float32)
    rgb_orig = np.asarray(pcd.colors, dtype=np.float32) if len(pcd.colors) > 0 else np.ones_like(xyz)
    rgb = (rgb_orig * 2 - 1).astype(np.float32)
    xyz_mid = xyz - xyz.min(0)

    semantic_scores, prop_idx, prop_off, scores, _ = _pointgroup_forward(
        xyz_mid, rgb, model, model_fn, sg_cfg, pgo)

    N = len(xyz)
    sem_thresh = seg_cfg.get('pg_sem_thresh', 0)
    if sem_thresh > 0:
        cabbage_prob = torch.softmax(semantic_scores, dim=1)[:, 1].cpu().numpy()
        mask_cabbage = cabbage_prob > sem_thresh
    else:
        mask_cabbage = semantic_scores.max(1)[1].cpu().numpy() == 1

    # SoftGroup 自顶向下选择 (替代 PointGroup NMS)
    inst_pg = extract_fn(prop_idx, prop_off, scores, N, sg_cfg)
    cabbage_indices = np.where(mask_cabbage)[0]
    inst_cabbage = inst_pg[cabbage_indices]

    cabbage_xyz = xyz[mask_cabbage]
    cabbage_pcd = o3d.geometry.PointCloud()
    cabbage_pcd.points = o3d.utility.Vector3dVector(cabbage_xyz)
    cabbage_pcd.colors = o3d.utility.Vector3dVector(rgb_orig[mask_cabbage])

    n_sem = mask_cabbage.sum()
    n_inst = (inst_cabbage > 0).sum()
    logger.info(f"SoftGroup: cabbage={n_sem:,}pts, {n_inst} instances")
    return cabbage_pcd, inst_cabbage


def softgroup_segment_blocks(pcd, cfg):
    """分块 SoftGroup 推理."""
    model, model_fn, sg_cfg, seg_cfg, pgo, extract_fn = _load_softgroup(cfg)

    block_cfg = seg_cfg.get('block_inference', {})
    block_size = float(block_cfg.get('block_size', 2.0))
    overlap = float(block_cfg.get('overlap', 0.5))
    max_single = float(block_cfg.get('max_single_extent', 8.0))
    merge_iou = float(block_cfg.get('merge_iou_thresh', 0.25))

    xyz = np.asarray(pcd.points, dtype=np.float32)
    rgb_orig = np.asarray(pcd.colors, dtype=np.float32) if len(pcd.colors) > 0 else np.ones_like(xyz)
    rgb = (rgb_orig * 2 - 1).astype(np.float32)
    extent = xyz.max(0) - xyz.min(0)

    if max(extent) <= max_single and block_cfg.get('enable', True):
        logger.info(f"田地 ≤{max_single}m, 单次 SoftGroup 推理")
        return softgroup_segment(pcd, cfg)

    stride = max(block_size - overlap, 0.1)
    x_min, y_min = xyz[:, 0].min(), xyz[:, 1].min()
    x_max, y_max = xyz[:, 0].max(), xyz[:, 1].max()

    blocks = []
    xs = x_min
    while xs < x_max:
        ys = y_min
        while ys < y_max:
            mask = (xyz[:, 0] >= xs) & (xyz[:, 0] < xs + block_size) & \
                   (xyz[:, 1] >= ys) & (xyz[:, 1] < ys + block_size)
            if mask.sum() >= 100:
                idx = np.where(mask)[0]
                blocks.append({'global_idx': idx, 'origin': xyz[idx].min(0)})
            ys += stride
        xs += stride

    logger.info(f"SoftGroup 分块: {len(blocks)} blocks")

    N_total = len(xyz)
    cabbage_votes = np.zeros(N_total, dtype=np.int32)
    total_votes = np.zeros(N_total, dtype=np.int32)
    all_proposals = []
    sem_thresh = seg_cfg.get('pg_sem_thresh', 0)

    for bi, block in enumerate(blocks):
        gidx = block['global_idx']
        block_xyz = xyz[gidx] - block['origin']
        block_rgb = rgb[gidx]

        sem_scores, prop_idx, prop_off, scores, _ = _pointgroup_forward(
            block_xyz, block_rgb, model, model_fn, sg_cfg, pgo)

        inst_local = extract_fn(prop_idx, prop_off, scores, len(gidx), sg_cfg)

        for i, gi in enumerate(gidx):
            total_votes[gi] += 1
            if sem_thresh > 0:
                prob = float(torch.softmax(sem_scores, dim=1)[i, 1])
                if prob > sem_thresh:
                    cabbage_votes[gi] += 1
            elif sem_scores.max(1)[1][i] == 1:
                cabbage_votes[gi] += 1

        n_inst = inst_local.max()
        for inst_id in range(1, int(n_inst) + 1):
            pts_local = np.where(inst_local == inst_id)[0]
            if len(pts_local) >= seg_cfg.get('pg_npoint_thresh', 20):
                all_proposals.append(gidx[pts_local])

        if (bi + 1) % max(1, len(blocks) // 5) == 0:
            logger.info(f"  SoftGroup block {bi+1}/{len(blocks)}: {len(gidx):,}pts, {n_inst} inst")

    # Merge semantic
    mask_cabbage = np.zeros(N_total, dtype=bool)
    voted = total_votes > 0
    mask_cabbage[voted] = cabbage_votes[voted] >= (total_votes[voted] * 0.5)
    mask_cabbage = _post_semantic_filter(mask_cabbage, xyz, cfg)

    # Merge instances (same Union-Find as PointGroup blocks)
    n_prop = len(all_proposals)
    inst_global = np.zeros(N_total, dtype=np.int64)
    if n_prop > 0:
        parent = list(range(n_prop))
        def find(x):
            while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry: parent[rx] = ry

        point_to_props = {}
        for pi, pts in enumerate(all_proposals):
            for p in pts: point_to_props.setdefault(int(p), []).append(pi)

        merged_pairs = set()
        for point, prop_list in point_to_props.items():
            if len(prop_list) < 2: continue
            for i in range(len(prop_list)):
                for j in range(i+1, len(prop_list)):
                    pa, pb = prop_list[i], prop_list[j]
                    pair = (min(pa, pb), max(pa, pb))
                    if pair in merged_pairs: continue
                    merged_pairs.add(pair)
                    inter = len(set(all_proposals[pa]) & set(all_proposals[pb]))
                    union_size = len(set(all_proposals[pa]) | set(all_proposals[pb]))
                    if union_size > 0 and inter / union_size >= merge_iou:
                        union(pa, pb)

        root_to_id = {}
        next_id = 1
        for pi, pts in enumerate(all_proposals):
            root = find(pi)
            if root not in root_to_id: root_to_id[root] = next_id; next_id += 1
            inst_global[pts] = root_to_id[root]
        logger.info(f"SoftGroup merge: {n_prop} proposals → {next_id-1} instances")

    cabbage_xyz = xyz[mask_cabbage]
    cabbage_pcd = o3d.geometry.PointCloud()
    cabbage_pcd.points = o3d.utility.Vector3dVector(cabbage_xyz)
    cabbage_pcd.colors = o3d.utility.Vector3dVector(rgb_orig[mask_cabbage])
    inst_cabbage = inst_global[mask_cabbage]
    logger.info(f"SoftGroup(分块): cabbage={mask_cabbage.sum():,}pts, {inst_cabbage.max()} instances")
    return cabbage_pcd, inst_cabbage


# ============================================================
# HAIS 推理 (层次聚合: 点聚合 + 集合聚合, ICCV 2021)
# ============================================================
def _load_hais(cfg):
    """加载 HAIS 模型 (只执行一次). Returns (model, model_fn, hais_cfg, seg_cfg, hais_ops)"""
    seg_cfg = cfg.get('segmentation', {})
    hais_cfg_path = seg_cfg.get('hais_config', 'HAIS-main/config/hais_cabbage.yaml')
    if hais_cfg_path.startswith('HAIS-main/'):
        hais_cfg_path = hais_cfg_path[len('HAIS-main/'):]

    hais_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'HAIS-main')
    sys.path.insert(0, hais_root)
    sys.argv = ['main', '--config', os.path.join(hais_root, hais_cfg_path)]

    # 清除 util 模块缓存, 避免与 PointGroup 的 util.config/utils 冲突
    for _m in list(sys.modules):
        if _m == 'util' or _m.startswith('util.'):
            del sys.modules[_m]

    from util.config import cfg as hais_cfg
    hais_cfg.task = 'test'
    hais_cfg.exp_path = os.path.join(hais_root, hais_cfg.exp_path)
    from model.hais.hais import HAIS as Network, model_fn_decorator
    import util.utils as utils
    from lib.hais_ops.functions import hais_ops

    model = Network(hais_cfg).cuda()
    start_epoch = utils.checkpoint_restore(hais_cfg, model, None, hais_cfg.exp_path,
        hais_cfg.config.split('/')[-1][:-5], True, hais_cfg.test_epoch, dist=False, f='')
    logger.info(f"HAIS loaded: epoch {start_epoch-1}")
    model_fn = model_fn_decorator(test=True)
    return model, model_fn, hais_cfg, seg_cfg, hais_ops


def _hais_forward(xyz_block, rgb_block, model, model_fn, hais_cfg, hais_ops):
    """单块 HAIS 前向 (坐标已平移到块内原点).
    Returns: (semantic_scores, proposals)
      semantic_scores: (N, 2) 原始 logits
      proposals: list of np.array (块内点索引)
    """
    N = xyz_block.shape[0]
    xyz_voxel = (xyz_block * hais_cfg.scale)
    xyz_voxel = xyz_voxel - xyz_voxel.min(0)
    coords = torch.cat([torch.LongTensor(N, 1).fill_(0), torch.from_numpy(xyz_voxel).long()], 1)
    voxel_coords, p2v_map, v2p_map = hais_ops.voxelization_idx(coords, 1)
    spatial_shape = np.clip(coords.max(0)[0][1:].numpy() + 1, hais_cfg.full_scale[0], None)
    batch = {
        'locs': coords, 'voxel_locs': voxel_coords, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
        'locs_float': torch.from_numpy(xyz_block).float(),
        'feats': torch.from_numpy(rgb_block).float(),
        'offsets': torch.tensor([0, N], dtype=torch.int),
        'spatial_shape': spatial_shape,
        'batch_size': 1,
        'id': torch.tensor([0]),
    }
    with torch.no_grad():
        preds = model_fn(batch, model, hais_cfg.test_epoch)

    semantic_scores = preds['semantic']  # (N, 2)
    proposals = []
    if 'score' in preds:
        scores_pred = torch.sigmoid(preds['score'].view(-1))
        proposals_idx, proposals_offset, mask_scores = preds['proposals']
        n_prop = proposals_offset.shape[0] - 1
        if n_prop > 0:
            proposals_pred = torch.zeros((n_prop, N), dtype=torch.int, device=scores_pred.device)
            _mask = mask_scores.squeeze(1) > hais_cfg.test_mask_score_thre
            proposals_idx_gpu = proposals_idx.to(_mask.device)
            proposals_pred[proposals_idx_gpu[_mask][:, 0].long(),
                           proposals_idx_gpu[_mask][:, 1].long()] = 1
            score_mask = scores_pred > hais_cfg.TEST_SCORE_THRESH
            proposals_pred = proposals_pred[score_mask]
            proposals_pointnum = proposals_pred.sum(1)
            npoint_mask = proposals_pointnum >= hais_cfg.TEST_NPOINT_THRESH
            proposals_pred = proposals_pred[npoint_mask]
            for i in range(proposals_pred.shape[0]):
                proposals.append(np.where(proposals_pred[i].cpu().numpy() == 1)[0])
    return semantic_scores, proposals


def hais_segment(pcd, cfg):
    """单次 HAIS 推理 (端到端, 不分块)."""
    model, model_fn, hais_cfg, seg_cfg, hais_ops = _load_hais(cfg)

    xyz = np.asarray(pcd.points, dtype=np.float32)
    rgb_orig = np.asarray(pcd.colors, dtype=np.float32) if len(pcd.colors) > 0 else np.ones_like(xyz)
    rgb = (rgb_orig * 2 - 1).astype(np.float32)
    xyz_mid = xyz - xyz.min(0)

    semantic_scores, proposals = _hais_forward(xyz_mid, rgb, model, model_fn, hais_cfg, hais_ops)

    N = len(xyz)
    sem_thresh = seg_cfg.get('pg_sem_thresh', 0)
    if sem_thresh > 0:
        cabbage_prob = torch.softmax(semantic_scores, dim=1)[:, 1].cpu().numpy()
        mask_cabbage = cabbage_prob > sem_thresh
    else:
        mask_cabbage = semantic_scores.max(1)[1].cpu().numpy() == 1
    mask_cabbage = _post_semantic_filter(mask_cabbage, xyz, cfg)

    inst_pg = np.zeros(N, dtype=np.int64)
    for i, pts in enumerate(proposals):
        inst_pg[pts] = i + 1

    cabbage_indices = np.where(mask_cabbage)[0]
    inst_cabbage = inst_pg[cabbage_indices]

    cabbage_xyz = xyz[mask_cabbage]
    cabbage_pcd = o3d.geometry.PointCloud()
    cabbage_pcd.points = o3d.utility.Vector3dVector(cabbage_xyz)
    cabbage_pcd.colors = o3d.utility.Vector3dVector(rgb_orig[mask_cabbage])

    n_sem = mask_cabbage.sum()
    n_inst = (inst_cabbage > 0).sum()
    logger.info(f"HAIS: cabbage={n_sem:,}pts, {len(proposals)} proposals, {inst_cabbage.max()} instances")
    return cabbage_pcd, inst_cabbage


def hais_segment_blocks(pcd, cfg):
    """分块 HAIS 推理 (对齐 PointGroup 分块口径, 大田地避免集合聚合滚雪球)."""
    model, model_fn, hais_cfg, seg_cfg, hais_ops = _load_hais(cfg)

    block_cfg = seg_cfg.get('block_inference', {})
    block_size = float(block_cfg.get('block_size', 2.0))
    overlap = float(block_cfg.get('overlap', 0.5))
    max_single = float(block_cfg.get('max_single_extent', 8.0))
    merge_iou = float(block_cfg.get('merge_iou_thresh', 0.25))

    xyz = np.asarray(pcd.points, dtype=np.float32)
    rgb_orig = np.asarray(pcd.colors, dtype=np.float32) if len(pcd.colors) > 0 else np.ones_like(xyz)
    rgb = (rgb_orig * 2 - 1).astype(np.float32)
    extent = xyz.max(0) - xyz.min(0)

    if max(extent) <= max_single and block_cfg.get('enable', True):
        logger.info(f"田地 ≤{max_single}m, 单次 HAIS 推理")
        return hais_segment(pcd, cfg)

    stride = max(block_size - overlap, 0.1)
    x_min, y_min = xyz[:, 0].min(), xyz[:, 1].min()
    x_max, y_max = xyz[:, 0].max(), xyz[:, 1].max()

    blocks = []
    xs = x_min
    while xs < x_max:
        ys = y_min
        while ys < y_max:
            mask = (xyz[:, 0] >= xs) & (xyz[:, 0] < xs + block_size) & \
                   (xyz[:, 1] >= ys) & (xyz[:, 1] < ys + block_size)
            if mask.sum() >= 100:
                idx = np.where(mask)[0]
                blocks.append({'global_idx': idx, 'origin': xyz[idx].min(0)})
            ys += stride
        xs += stride

    logger.info(f"HAIS 分块: {len(blocks)} blocks")

    N_total = len(xyz)
    cabbage_votes = np.zeros(N_total, dtype=np.int32)
    total_votes = np.zeros(N_total, dtype=np.int32)
    all_proposals = []
    sem_thresh = seg_cfg.get('pg_sem_thresh', 0)

    for bi, block in enumerate(blocks):
        gidx = block['global_idx']
        block_xyz = xyz[gidx] - block['origin']
        block_rgb = rgb[gidx]

        sem_scores, proposals = _hais_forward(block_xyz, block_rgb, model, model_fn, hais_cfg, hais_ops)

        for i, gi in enumerate(gidx):
            total_votes[gi] += 1
            if sem_thresh > 0:
                prob = float(torch.softmax(sem_scores, dim=1)[i, 1])
                if prob > sem_thresh:
                    cabbage_votes[gi] += 1
            elif sem_scores.max(1)[1][i] == 1:
                cabbage_votes[gi] += 1

        for pts in proposals:
            if len(pts) >= seg_cfg.get('pg_npoint_thresh', 20):
                all_proposals.append(gidx[pts])

        if (bi + 1) % max(1, len(blocks) // 5) == 0:
            logger.info(f"  HAIS block {bi+1}/{len(blocks)}: {len(gidx):,}pts, {len(proposals)} proposals")

    # 语义合并
    mask_cabbage = np.zeros(N_total, dtype=bool)
    voted = total_votes > 0
    mask_cabbage[voted] = cabbage_votes[voted] >= (total_votes[voted] * 0.5)
    mask_cabbage = _post_semantic_filter(mask_cabbage, xyz, cfg)

    # 实例跨块合并 (Union-Find)
    n_prop = len(all_proposals)
    inst_global = np.zeros(N_total, dtype=np.int64)
    if n_prop > 0:
        parent = list(range(n_prop))
        def find(x):
            while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry: parent[rx] = ry

        point_to_props = {}
        for pi, pts in enumerate(all_proposals):
            for p in pts: point_to_props.setdefault(int(p), []).append(pi)

        merged_pairs = set()
        for point, prop_list in point_to_props.items():
            if len(prop_list) < 2: continue
            for i in range(len(prop_list)):
                for j in range(i+1, len(prop_list)):
                    pa, pb = prop_list[i], prop_list[j]
                    pair = (min(pa, pb), max(pa, pb))
                    if pair in merged_pairs: continue
                    merged_pairs.add(pair)
                    inter = len(set(all_proposals[pa]) & set(all_proposals[pb]))
                    union_size = len(set(all_proposals[pa]) | set(all_proposals[pb]))
                    if union_size > 0 and inter / union_size >= merge_iou:
                        union(pa, pb)

        root_to_id = {}
        next_id = 1
        for pi, pts in enumerate(all_proposals):
            root = find(pi)
            if root not in root_to_id: root_to_id[root] = next_id; next_id += 1
            inst_global[pts] = root_to_id[root]
        logger.info(f"HAIS merge: {n_prop} proposals → {next_id-1} instances")

    cabbage_xyz = xyz[mask_cabbage]
    cabbage_pcd = o3d.geometry.PointCloud()
    cabbage_pcd.points = o3d.utility.Vector3dVector(cabbage_xyz)
    cabbage_pcd.colors = o3d.utility.Vector3dVector(rgb_orig[mask_cabbage])
    inst_cabbage = inst_global[mask_cabbage]
    logger.info(f"HAIS(分块): cabbage={mask_cabbage.sum():,}pts, {inst_cabbage.max()} instances")
    return cabbage_pcd, inst_cabbage


def run_pipeline(args):
    cfg = load_config(args.config)
    set_determinism(cfg)
    
    output_dir = cfg['pipeline'].get('output_dir', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load + Preprocess
    logger.info(f"Loading: {args.input}")
    pcd = read_point_cloud(args.input)
    preprocess_result = preprocess_point_cloud(pcd, cfg)
    pcd_clean = preprocess_result.pcd_clean
    non_ground_pcd = preprocess_result.non_ground_pcd
    
    # 2. 语义/实例分割: 根据 backbone + method 选择管道
    backbone = cfg.get('segmentation', {}).get('backbone', 'pointnet2')
    seg_method = cfg.get('segmentation', {}).get('method', 'hybrid')
    logger.info(f"分割: backbone={backbone}, method={seg_method}")
    
    # RandLA-Net 独立路由 (无 offset, 只支持聚类)
    if backbone == 'randlanet':
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RandLANet_Ours'))
        from randlanet import RandLANet as RLAModel
        model = RLAModel(d_in=6, num_classes=2).cuda()
        ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RandLANet_Ours/exp/randlanet/best.pth')
        if os.path.exists(ckpt):
            state = torch.load(ckpt, map_location='cuda')
            model.load_state_dict(state['model'])
        model.eval()
        xyz = np.asarray(non_ground_pcd.points, dtype=np.float32)
        rgb_orig = np.asarray(non_ground_pcd.colors, dtype=np.float32) if len(non_ground_pcd.colors) > 0 else np.ones_like(xyz)
        rgb = rgb_orig.copy()
        with torch.no_grad():
            pred = model(torch.from_numpy(xyz - xyz.min(0)).unsqueeze(0).cuda(),
                        torch.from_numpy(rgb * 2 - 1).unsqueeze(0).cuda())
        mask = pred[0].max(1)[1].cpu().numpy() == 1
        mask = _post_semantic_filter(mask, xyz, cfg)
        cabbage_pcd = o3d.geometry.PointCloud()
        cabbage_pcd.points = o3d.utility.Vector3dVector(xyz[mask])
        cabbage_pcd.colors = o3d.utility.Vector3dVector(rgb_orig[mask])
        o3d.io.write_point_cloud(os.path.join(output_dir, "cabbage_points.ply"), cabbage_pcd)
        clusterer = InstanceClusterer(cfg)
        inst_labels, n = clusterer.cluster(cabbage_pcd)
        logger.info(f"RandLA-Net: {n} instances")
        # 跳过后面的 if/elif
    
    # ================================================================
    # Mode 1: hybrid / softgroup
    # proposal_selection: "nms" | "greedy" | "iach"
    # ================================================================
    if seg_method in ('hybrid', 'softgroup'):
        sel = cfg.get('segmentation', {}).get('proposal_selection', 'nms')
        if seg_method == 'softgroup':
            sel = 'greedy'
        
        if sel == 'iach':
            # IACH: offset移位 + DBSCAN, 不需要 pointgroup_segment_blocks
            model, model_fn, pg_cfg, seg_cfg, pgo = _load_pointgroup(cfg)
            xyz_all = np.asarray(non_ground_pcd.points, dtype=np.float32)
            rgb_all_orig = np.asarray(non_ground_pcd.colors, dtype=np.float32) if len(non_ground_pcd.colors) > 0 else np.ones_like(xyz_all)
            rgb_all = (rgb_all_orig * 2 - 1).astype(np.float32)
            sem_scores, _, _, _, pt_offsets = _pointgroup_forward(
                xyz_all - xyz_all.min(0), rgb_all, model, model_fn, pg_cfg, pgo)
            sem_thresh = seg_cfg.get('pg_sem_thresh', 0)
            if sem_thresh > 0:
                p = torch.softmax(sem_scores, dim=1)[:, 1].cpu().numpy()
                mask = p > sem_thresh
            else:
                mask = sem_scores.max(1)[1].cpu().numpy() == 1
            mask = _post_semantic_filter(mask, xyz_all, cfg)
            cabbage_pcd = o3d.geometry.PointCloud()
            cabbage_pcd.points = o3d.utility.Vector3dVector(xyz_all[mask])
            cabbage_pcd.colors = o3d.utility.Vector3dVector(rgb_all_orig[mask])
            o3d.io.write_point_cloud(os.path.join(output_dir, "cabbage_points.ply"), cabbage_pcd)
            iach_l, _ = _iach_cluster(xyz_all[mask], pt_offsets[mask], cfg)
            inst_init = iach_l
            sel_label = 'iach'
        else:
            cabbage_pcd, inst_init = pointgroup_segment_blocks(non_ground_pcd, cfg, 'hybrid', selection=sel)
            sel_label = sel
        cabbage_pts = np.asarray(cabbage_pcd.points)
        o3d.io.write_point_cloud(os.path.join(output_dir, "cabbage_points.ply"), cabbage_pcd)
        
        stage1_name = f"stage1_hybrid_{sel_label}.ply"
        n1 = save_instance_pcd(cabbage_pcd, inst_init, os.path.join(output_dir, stage1_name))
        logger.info(f"Stage1 (hybrid+{sel_label}): {n1} instances")
        
        # PG已分配保留, 只对未分配点做聚类
        assigned_mask = inst_init > 0
        n_assigned = assigned_mask.sum()
        n_unassigned = (~assigned_mask).sum()
        
        if n_unassigned > 100:
            logger.info(f"   已分配: {n_assigned:,}点 → 保留PG提案")
            logger.info(f"   未分配: {n_unassigned:,}点 → 邻近归入 + 密度过滤后聚类")
            
            # 邻近归入 (缩小半径, 避免吸走漏检甘蓝的点)
            expand_cfg = cfg.get('instance', {}).get('pg_expand', {})
            expand_radius = expand_cfg.get('radius', 0.02)  # 5cm→2cm
            un_pts_all = cabbage_pts[~assigned_mask]
            pg_pts = cabbage_pts[assigned_mask]
            pg_labels = inst_init[assigned_mask]
            
            pg_tree = cKDTree(pg_pts)
            dists, nn_idx = pg_tree.query(un_pts_all, k=1)
            nearby = dists < expand_radius
            
            n_reassign = nearby.sum()
            if n_reassign > 0:
                un_global = np.where(~assigned_mask)[0]
                for i in np.where(nearby)[0]:
                    inst_init[un_global[i]] = pg_labels[nn_idx[i]]
                assigned_mask = inst_init > 0
                logger.info(f"   邻近归入: {n_reassign:,} 个未分配点 → 最近提案 (r<{expand_radius}m)")
            
            un_pts = cabbage_pts[~assigned_mask]
            logger.info(f"   剩余未分配: {len(un_pts):,}点 → 密度过滤后聚类")
            
            # 密度预过滤
            prefilter = cfg.get('instance', {}).get('precluster_filter', {})
            dense_mask = np.ones(len(un_pts), dtype=bool)
            if prefilter.get('enable', True) and len(un_pts) > 0:
                ror_nb = prefilter.get('min_points', 10)
                ror_radius = prefilter.get('eps', 0.03)
                tmp_pcd = o3d.geometry.PointCloud()
                tmp_pcd.points = o3d.utility.Vector3dVector(un_pts)
                _, dense_idx = tmp_pcd.remove_radius_outlier(nb_points=ror_nb, radius=ror_radius)
                dense_idx = np.asarray(dense_idx, dtype=np.int64)
                n_discard = len(un_pts) - len(dense_idx)
                if n_discard > 0:
                    logger.info(f"   密度过滤: 丢弃 {n_discard:,} 个稀疏点")
                dense_mask = np.zeros(len(un_pts), dtype=bool)
                if len(dense_idx) > 0:
                    dense_mask[dense_idx] = True
                un_pts = un_pts[dense_mask]
            
            logger.info(f"   有效未分配点: {len(un_pts):,}")
            
            # 聚类 (先用配置方法, 无结果时用 DBSCAN 兜底)
            if len(un_pts) > 0:
                un_pcd = o3d.geometry.PointCloud()
                un_pcd.points = o3d.utility.Vector3dVector(un_pts)
                clusterer = InstanceClusterer(cfg)
                try:
                    un_labels, n_un_clusters = clusterer.cluster(un_pcd)
                except Exception:
                    un_labels = np.array([], dtype=np.int64)
                    n_un_clusters = 0
                # DBSCAN 兜底: 主聚类没找到实例时, 用更激进参数重试
                if n_un_clusters == 0 and len(un_pts) >= 2000:
                    logger.info(f"   主聚类无结果, DBSCAN兜底 (eps=0.08, min_samples=500)...")
                    from sklearn.cluster import DBSCAN
                    db = DBSCAN(eps=0.08, min_samples=500).fit(un_pts)
                    un_labels = db.labels_
                    n_un_clusters = len(set(un_labels)) - (1 if -1 in un_labels else 0)
                    logger.info(f"   DBSCAN兜底: {n_un_clusters} clusters")
            else:
                un_labels = np.array([], dtype=np.int64)
                n_un_clusters = 0
            
            # 聚类后密度校验
            dense_cfg = prefilter.get('post_density_check', {})
            if dense_cfg.get('enable', True):
                min_linear_density = dense_cfg.get('min_linear_density', 500)
                min_pts_abs = dense_cfg.get('min_pts_absolute', 100)
                valid_clusters = set()
                for ul in sorted(set(un_labels)):
                    if ul < 0: continue
                    cl_pts = un_pts[un_labels == ul]
                    n_pt = len(cl_pts)
                    if n_pt < min_pts_abs: continue
                    span = np.max(cl_pts, axis=0) - np.min(cl_pts, axis=0)
                    max_span = np.max(span)
                    if max_span > 0 and n_pt / max_span >= min_linear_density:
                        valid_clusters.add(ul)
                n_reject = len(set(un_labels)) - len(valid_clusters) - (1 if -1 in un_labels else 0)
                if n_reject > 0:
                    logger.info(f"   密度校验: 丢弃 {n_reject} 个稀疏簇")
            else:
                valid_clusters = set(ul for ul in set(un_labels) if ul >= 0)
            
            # 合并: PG分配 + 新聚类
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
        
        n2 = save_instance_pcd(cabbage_pcd, labels_s2,
            os.path.join(output_dir, "stage2_clustering.ply"))
        logger.info(f"Stage2 (PG保留+未分配聚类): {n2} instances")
        
        # Stage 3: 骨架拆分
        from cabbage_pheno.instance import InstanceClusterer as IC
        tmp_clusterer = IC(cfg)
        if cfg.get('instance', {}).get('pca_split', {}).get('enable', False):
            labels_s3 = tmp_clusterer._apply_skeleton_split(labels_s2, cabbage_pts)
        else:
            labels_s3 = labels_s2.copy()
        try:
            labels_s3 = tmp_clusterer._merge_fragments(labels_s3, cabbage_pts)
        except: pass
        
        n3 = save_instance_pcd(cabbage_pcd, labels_s3,
            os.path.join(output_dir, "stage3_skeleton.ply"))
        logger.info(f"Stage3 (+Skeleton split): {n3} instances")
        
        # Stage 4: 碎片清理
        discard_thresh = cfg.get('instance', {}).get('fragment_voting', {}).get('discard_threshold', 300)
        merge_thresh = cfg.get('instance', {}).get('fragment_voting', {}).get('merge_threshold', 1500)
        
        unique, counts = np.unique(labels_s3, return_counts=True)
        for lbl, cnt in zip(unique, counts):
            if lbl < 0: continue
            if cnt < discard_thresh:
                labels_s3[labels_s3 == lbl] = -1
        
        if cfg.get('instance', {}).get('fragment_voting', {}).get('enable', False):
            try:
                labels_s3 = tmp_clusterer._cleanup_tiny_fragments(labels_s3, cabbage_pts, min_size=merge_thresh)
            except: pass
        
        # 最终过滤 min_cluster_points
        min_pts = cfg.get('instance', {}).get('min_cluster_points', 2000)
        unique, counts = np.unique(labels_s3, return_counts=True)
        for lbl, cnt in zip(unique, counts):
            if lbl < 0: continue
            if cnt < min_pts:
                labels_s3[labels_s3 == lbl] = -1
        # 重新编号
        valid = labels_s3 >= 0
        new_labels = np.full_like(labels_s3, -1)
        next_id = 0
        for old_lbl in sorted(set(labels_s3[valid].tolist())):
            new_labels[labels_s3 == old_lbl] = next_id
            next_id += 1
        inst_labels = new_labels
    
    # ================================================================
    # Mode 2: pointgroup — 仅 PointGroup 提案, 不做二次聚类
    # ================================================================
    elif seg_method == 'pointgroup':
        cabbage_pcd, inst_init = pointgroup_segment_blocks(non_ground_pcd, cfg, 'pointgroup')
        cabbage_pts = np.asarray(cabbage_pcd.points)
        o3d.io.write_point_cloud(os.path.join(output_dir, "cabbage_points.ply"), cabbage_pcd)
        
        n1 = save_instance_pcd(cabbage_pcd, inst_init,
            os.path.join(output_dir, "stage1_pointgroup.ply"))
        logger.info(f"Stage1 (PointGroup blocks): {n1} instances")
        
        inst_labels = inst_init.copy()
    
    # ================================================================
    # Mode: pointnet2 — PointNet2纯语义 + 传统聚类 (不同backbone对比)
    # ================================================================
    elif seg_method == 'pointnet2':
        logger.info("PointNet2 纯语义 + MeanShift聚类 (对比 backbone)")
        pn2_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PointNet2_Ours')
        sys.path.insert(0, pn2_root); sys.path.insert(0, os.path.join(pn2_root, 'model'))
        from pointnet2_sem import PointNet2Sem, model_fn_decorator as pn2_fn
        from util.config import cfg as pn2_cfg
        import util.utils as pn2_utils
        pn2_cfg.task = 'test'
        # Load model
        model = PointNet2Sem(pn2_cfg).cuda()
        exp_dir = os.path.join(pn2_root, 'exp/cabbage_dataset/pointnet2_sem/pointnet2_cabbage')
        pn2_utils.checkpoint_restore(model, exp_dir, 'pointnet2_cabbage', True, pn2_cfg.test_epoch)
        mfn = pn2_fn(test=True)
        
        from lib.pointgroup_ops.functions import pointgroup_ops
        xyz = np.asarray(non_ground_pcd.points, dtype=np.float32)
        rgb_orig = np.asarray(non_ground_pcd.colors, dtype=np.float32) if len(non_ground_pcd.colors) > 0 else np.ones_like(xyz)
        rgb = (rgb_orig * 2 - 1).astype(np.float32)
        xm = xyz - xyz.min(0); xs = xm * pn2_cfg.scale; N = len(xyz)
        locs = torch.cat([torch.LongTensor(N,1).fill_(0), torch.from_numpy(xs).long()], 1)
        vl, p2v, v2p = pointgroup_ops.voxelization_idx(locs, 1, pn2_cfg.mode)
        batch = {'locs': locs, 'voxel_locs': vl, 'p2v_map': p2v, 'v2p_map': v2p,
                 'locs_float': torch.from_numpy(xm), 'feats': torch.from_numpy(rgb),
                 'offsets': torch.tensor([0,N], dtype=torch.int),
                 'spatial_shape': np.clip((locs.max(0)[0][1:]+1).numpy(), pn2_cfg.full_scale[0], None)}
        with torch.no_grad():
            preds = mfn(batch, model, pn2_cfg.test_epoch)
        ss = preds['semantic']
        mask = ss.max(1)[1].cpu().numpy() == 1
        mask = _post_semantic_filter(mask, xyz, cfg)
        cabbage_pcd = o3d.geometry.PointCloud()
        cabbage_pcd.points = o3d.utility.Vector3dVector(xyz[mask])
        cabbage_pcd.colors = o3d.utility.Vector3dVector(rgb_orig[mask])
        o3d.io.write_point_cloud(os.path.join(output_dir, "cabbage_points.ply"), cabbage_pcd)
        clusterer = InstanceClusterer(cfg)
        inst_labels, n = clusterer.cluster(cabbage_pcd)
        logger.info(f"PointNet2+MeanShift: {n} instances")
    
    # ================================================================
    # Mode: randlanet — RandLA-Net语义 + 传统聚类 (不同backbone对比)
    # ================================================================
    elif seg_method == 'randlanet':
        logger.info("RandLA-Net 语义 + MeanShift聚类")
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RandLANet_Ours'))
        from randlanet import RandLANet as RLAModel
        model = RLAModel(d_in=6, num_classes=2).cuda()
        ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RandLANet_Ours/exp/randlanet/best.pth')
        state = torch.load(ckpt, map_location='cuda')
        model.load_state_dict(state['model'])
        model.eval()
        
        xyz = np.asarray(non_ground_pcd.points, dtype=np.float32)
        rgb_orig = np.asarray(non_ground_pcd.colors, dtype=np.float32) if len(non_ground_pcd.colors) > 0 else np.ones_like(xyz)
        rgb = rgb_orig.copy()
        xyz_t = torch.from_numpy(xyz - xyz.min(0)).unsqueeze(0).cuda()
        rgb_t = torch.from_numpy(rgb * 2 - 1).unsqueeze(0).cuda()
        
        with torch.no_grad():
            pred = model(xyz_t, rgb_t)
        mask = pred[0].max(1)[1].cpu().numpy() == 1
        mask = _post_semantic_filter(mask, xyz, cfg)
        
        cabbage_pcd = o3d.geometry.PointCloud()
        cabbage_pcd.points = o3d.utility.Vector3dVector(xyz[mask])
        cabbage_pcd.colors = o3d.utility.Vector3dVector(rgb_orig[mask])
        o3d.io.write_point_cloud(os.path.join(output_dir, "cabbage_points.ply"), cabbage_pcd)
        clusterer = InstanceClusterer(cfg)
        inst_labels, n = clusterer.cluster(cabbage_pcd)
        logger.info(f"RandLA-Net+MeanShift: {n} instances")
    
    # ================================================================
    # Mode 3: clustering — PointGroup 语义分割 + 传统聚类
    # ================================================================
    elif seg_method == 'clustering':
        logger.info("PointGroup 语义分割筛选甘蓝点 → 传统聚类")
        cabbage_pcd, _ = pointgroup_segment(non_ground_pcd, cfg, 'clustering')
        cabbage_pts = np.asarray(cabbage_pcd.points)
        o3d.io.write_point_cloud(os.path.join(output_dir, "cabbage_points.ply"), cabbage_pcd)
        
        logger.info(f"甘蓝点数: {len(cabbage_pts):,}, 开始聚类...")
        clusterer = InstanceClusterer(cfg)
        inst_labels, n_clusters = clusterer.cluster(cabbage_pcd)
        logger.info(f"传统聚类完成: {n_clusters} instances")
        
        # 骨架拆分 + 碎片清理
        from cabbage_pheno.instance import InstanceClusterer as IC
        tmp_clusterer = IC(cfg)
        if cfg.get('instance', {}).get('pca_split', {}).get('enable', False):
            inst_labels = tmp_clusterer._apply_skeleton_split(inst_labels, cabbage_pts)
        try:
            inst_labels = tmp_clusterer._merge_fragments(inst_labels, cabbage_pts)
        except: pass
        
        # 碎片清理
        discard_thresh = cfg.get('instance', {}).get('fragment_voting', {}).get('discard_threshold', 300)
        merge_thresh = cfg.get('instance', {}).get('fragment_voting', {}).get('merge_threshold', 1500)
        unique, counts = np.unique(inst_labels, return_counts=True)
        for lbl, cnt in zip(unique, counts):
            if lbl < 0: continue
            if cnt < discard_thresh:
                inst_labels[inst_labels == lbl] = -1
        if cfg.get('instance', {}).get('fragment_voting', {}).get('enable', False):
            try:
                inst_labels = tmp_clusterer._cleanup_tiny_fragments(inst_labels, cabbage_pts, min_size=merge_thresh)
            except: pass
        
        min_pts = cfg.get('instance', {}).get('min_cluster_points', 2000)
        unique, counts = np.unique(inst_labels, return_counts=True)
        for lbl, cnt in zip(unique, counts):
            if lbl < 0: continue
            if cnt < min_pts:
                inst_labels[inst_labels == lbl] = -1
        valid = inst_labels >= 0
        new_labels = np.full_like(inst_labels, -1)
        next_id = 0
        for old_lbl in sorted(set(inst_labels[valid].tolist())):
            new_labels[inst_labels == old_lbl] = next_id
            next_id += 1
        inst_labels = new_labels
        
        save_instance_pcd(cabbage_pcd, inst_labels,
            os.path.join(output_dir, "stage_clustering.ply"))
    
    # ================================================================
    # Mode 4: none — 跳过分割
    # ================================================================
    else:  # seg_method == 'none'
        logger.info("跳过分割, 所有点作为一个实例")
        cabbage_pcd = non_ground_pcd
        cabbage_pts = np.asarray(cabbage_pcd.points)
        o3d.io.write_point_cloud(os.path.join(output_dir, "cabbage_points.ply"), cabbage_pcd)
        inst_labels = np.zeros(len(cabbage_pts), dtype=np.int64)
    
    # === 后处理: noise → 最近邻 (带距离上限, 防止远噪声点被硬塞进植株拉大冠幅) ===
    noise_mask = inst_labels == -1
    n_noise = noise_mask.sum()
    if n_noise > 0 and inst_labels.max() >= 0:
        cabbage_pts = np.asarray(cabbage_pcd.points)
        valid_mask = inst_labels >= 0
        tree = cKDTree(cabbage_pts[valid_mask])
        dists, nn_idx = tree.query(cabbage_pts[noise_mask], k=1)
        reassign_radius = cfg.get('instance', {}).get('noise_reassign_radius', 0.10)
        nearby = dists < reassign_radius
        reassign_idx = np.where(noise_mask)[0][nearby]
        inst_labels[reassign_idx] = inst_labels[valid_mask][nn_idx[nearby]]
        logger.info(f"Noise reassign: {nearby.sum():,}/{n_noise:,} points → nearest cluster (r<{reassign_radius}m), {n_noise - nearby.sum():,} dropped")
    
    # === 最终过滤: 丢弃 < min_cluster_points 的碎片 ===
    min_pts = cfg.get('instance', {}).get('min_cluster_points', 1000)
    unique, counts = np.unique(inst_labels, return_counts=True)
    n_discarded = 0
    for lbl, cnt in zip(unique, counts):
        if lbl < 0: continue
        if cnt < min_pts:
            inst_labels[inst_labels == lbl] = -1
            n_discarded += 1
    if n_discarded > 0:
        valid = inst_labels >= 0
        new_labels = np.full_like(inst_labels, -1)
        next_id = 0
        for old_lbl in sorted(set(inst_labels[valid].tolist())):
            new_labels[inst_labels == old_lbl] = next_id
            next_id += 1
        inst_labels = new_labels
        logger.info(f"Discarded {n_discarded} tiny clusters (< {min_pts} pts)")
    
    save_instance_pcd(cabbage_pcd, inst_labels, os.path.join(output_dir, "stage4_final.ply"))
    logger.info(f"Stage4 Final: {inst_labels.max()+1} instances")
    
    # 4. Traits
    calculator = TraitCalculator(cfg)
    results = []
    for lbl in sorted(set(inst_labels)):
        if lbl < 0: continue
        idxs = np.where(inst_labels == lbl)[0]
        plant_pcd = cabbage_pcd.select_by_index(idxs)
        traits = calculator.calculate_traits(plant_pcd, lbl, None)
        if traits: results.append(traits)
    logger.info(f"Extracted traits for {len(results)} plants")
    
    # 5. Save
    with open(os.path.join(output_dir, 'plants.json'), 'w') as f:
        json.dump(results, f, indent=2)
    if results:
        pd.DataFrame(results).round(2).to_csv(os.path.join(output_dir, 'plants.csv'), index=False)
    
    # 6. Final visualize
    save_instance_pcd(cabbage_pcd, inst_labels, os.path.join(output_dir, "stage4_final.ply"))
    logger.info(f"Done! Results in {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cabbage Pipeline (PointGroup + Clustering)")
    parser.add_argument("--input", required=True, help="Input .ply/.pcd")
    parser.add_argument("--config", default="configs/default.yaml", help="Config path")
    args = parser.parse_args()
    run_pipeline(args)
