#!/usr/bin/env python3
"""PointNet2 (PointNet++) 语义分割 + 三种聚类统一评估。

- 语义 backbone: 真正的 PointNet2SemSeg (SA+FP, 含 rgb)
- 推理: 分块 (block_size) + 重叠 + 投票融合
- 数据: evalaute_test (21文件, split.json 划分 val/test)
- instance.method 决定聚类, pca_split.enable 决定 GIDM

用法:
  python tools/pointnet2_real_eval.py --config configs/pn_meanshift.yaml --split test
"""
import sys, os, time, json, glob, warnings, argparse, gc
import numpy as np

os.chdir('/root/autodl-tmp')
sys.path.insert(0, '/root/autodl-tmp')
warnings.filterwarnings('ignore')

import torch, open3d as o3d
from scipy.spatial import cKDTree

from cabbage_pheno.segmentation.pointnet2_model import PointNet2SemSeg
from cabbage_pheno.io import read_point_cloud
from cabbage_pheno.service.pipeline import preprocess_point_cloud
from cabbage_pheno.service.evaluation import (
    load_config, load_ground_truth, align_gt_to_pred, evaluate_from_predictions,
)
from cabbage_pheno.instance import InstanceClusterer


def load_model():
    model = PointNet2SemSeg(num_classes=2, additional_channel=3).cuda().eval()
    ckpt = '/root/autodl-tmp/models/pointnet2_sem_best.pth'
    sd = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(sd if not isinstance(sd, dict) or 'model_state_dict' not in sd else sd['model_state_dict'])
    print(f'PointNet2 loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params')
    return model


def block_inference(xyz, rgb, model, block_size=1.0, overlap=0.5):
    """分块推理 + 重叠投票。xyz=(N,3) 米尺度, rgb=(N,3)∈[0,1]。返回 (N,) 语义标签"""
    N = len(xyz)
    npoints = 4096
    rgb_norm = rgb * 2.0 - 1.0  # -> [-1,1]

    # 收集所有块的投票
    votes = np.zeros((N, 2), dtype=np.float32)
    counts = np.zeros(N, dtype=np.float32)

    xmin, ymin = xyz[:, 0].min(), xyz[:, 1].min()
    xmax, ymax = xyz[:, 0].max(), xyz[:, 1].max()
    stride = block_size * (1 - overlap)

    xs = np.arange(xmin, xmax, stride)
    ys = np.arange(ymin, ymax, stride)

    for x0 in xs:
        for y0 in ys:
            x1 = x0 + block_size; y1 = y0 + block_size
            mask = (xyz[:, 0] >= x0) & (xyz[:, 0] < x1) & \
                   (xyz[:, 1] >= y0) & (xyz[:, 1] < y1)
            if mask.sum() < 50:
                continue
            idx = np.where(mask)[0]
            blk_xyz = xyz[idx]
            blk_rgb = rgb_norm[idx]
            # 采样 npoints
            if len(idx) >= npoints:
                choice = np.random.choice(len(idx), npoints, replace=False)
            else:
                choice = np.random.choice(len(idx), npoints, replace=True)
            # 居中: 用全块均值 (与训练一致: xyz - xyz.mean(0))
            blk_center = blk_xyz.mean(0)
            sub_xyz = blk_xyz[choice] - blk_center
            sub_rgb = blk_rgb[choice]

            xyz_t = torch.from_numpy(sub_xyz.T).unsqueeze(0).cuda()  # (1,3,N)
            rgb_t = torch.from_numpy(sub_rgb.T).unsqueeze(0).cuda()  # (1,3,N)
            with torch.no_grad():
                pred, _ = model(xyz_t, rgb_t)  # (1,2,N) log_softmax
            prob = torch.exp(pred).squeeze(0).cpu().numpy()  # (2,N)
            # 最近邻传播回全块: 采样点(居中) -> 全块点(居中)
            sub_xyz_centered = blk_xyz[choice] - blk_center
            blk_xyz_centered = blk_xyz - blk_center
            tree = cKDTree(sub_xyz_centered)
            _, nn = tree.query(blk_xyz_centered, k=1)
            votes[idx] += prob[:, nn].T
            counts[idx] += 1

    # 无投票的点 -> 全背景
    sem = np.zeros(N, dtype=np.int32)
    valid = counts > 0
    sem[valid] = votes[valid].argmax(1)
    return sem


def run_one_file(fp, cfg, model, clusterer, skip_clustering=False):
    nm = os.path.splitext(os.path.basename(fp))[0]
    t0 = time.time()
    pcd = read_point_cloud(fp)
    if pcd is None:
        return None
    gt_pts, gt_sem, gt_inst = load_ground_truth(fp, -2, -1, True)
    if gt_inst is None:
        return None

    pr = preprocess_point_cloud(pcd, cfg)
    xyz_clean = pr.points_clean
    ng_pcd = pr.non_ground_pcd
    xyz_ng = np.asarray(ng_pcd.points, np.float32)
    rgb_ng = np.asarray(ng_pcd.colors, np.float32) if len(ng_pcd.colors) > 0 \
             else np.ones((len(xyz_ng), 3), np.float32) * 0.5
    n_all = len(xyz_clean)
    n_ng = len(xyz_ng)
    print(f'  Points: {n_all} total → {n_ng} non-ground', flush=True)

    if n_ng < 32:
        return None

    # 语义推理
    t_sem = time.time()
    sem_ng = block_inference(xyz_ng, rgb_ng, model, block_size=1.0, overlap=0.5)
    t_sem = time.time() - t_sem
    n_cab = int(sem_ng.sum())
    print(f'  Sem: {n_cab} cabbage / {n_ng} ({t_sem:.1f}s)', flush=True)

    full_sem = np.zeros(n_all, dtype=int)
    full_inst = np.full(n_all, -1, dtype=int)
    cab_mask_ng = sem_ng == 1
    cab_idx_ng = np.where(cab_mask_ng)[0]
    ng2all = None
    if n_all > 0 and n_ng > 0:
        t_ng = cKDTree(xyz_ng)
        _, ng2all = t_ng.query(xyz_clean, k=1)
        full_sem = sem_ng[ng2all]

    if n_cab >= 50:
        cab_xyz = xyz_ng[cab_mask_ng]
        cab_pcd = o3d.geometry.PointCloud()
        cab_pcd.points = o3d.utility.Vector3dVector(cab_xyz)
        if skip_clustering:
            inst_ng = np.zeros(n_cab, dtype=np.int64)
            if cfg.get('instance', {}).get('pca_split', {}).get('enable', False):
                if clusterer.skeleton_cut:
                    try:
                        inst_ng = clusterer._apply_skeleton_split(inst_ng, cab_xyz)
                    except Exception:
                        pass
                if clusterer.merge_back:
                    try:
                        inst_ng = clusterer._merge_fragments(inst_ng, cab_xyz)
                    except Exception:
                        pass
        else:
            try:
                inst_ng, _ = clusterer.cluster(cab_pcd)
            except Exception as e:
                print(f'  Clustering error: {e}', flush=True)
                inst_ng = np.full(n_cab, -1, dtype=np.int64)

        discard_thresh = cfg.get('instance', {}).get('fragment_voting', {}).get('discard_threshold', 300)
        min_pts = cfg.get('instance', {}).get('min_cluster_points', 2000)

        uniq, cnts = np.unique(inst_ng, return_counts=True)
        for lbl, c in zip(uniq, cnts):
            if lbl >= 0 and c < discard_thresh:
                inst_ng[inst_ng == lbl] = -1
        if cfg.get('instance', {}).get('fragment_voting', {}).get('enable', False):
            try:
                inst_ng = clusterer._cleanup_tiny_fragments(inst_ng, cab_xyz)
            except Exception:
                pass
        uniq, cnts = np.unique(inst_ng, return_counts=True)
        for lbl, c in zip(uniq, cnts):
            if lbl >= 0 and c < min_pts:
                inst_ng[inst_ng == lbl] = -1

        noise_mask = inst_ng == -1
        if noise_mask.sum() > 0 and (inst_ng >= 0).sum() > 0:
            valid_mask = inst_ng >= 0
            tree_ng = cKDTree(cab_xyz[valid_mask])
            _, nn_idx = tree_ng.query(cab_xyz[noise_mask], k=1)
            inst_ng[noise_mask] = inst_ng[valid_mask][nn_idx]

        valid = inst_ng >= 0
        if valid.sum() > 0:
            new_labels = np.full_like(inst_ng, -1)
            next_id = 1
            for old_lbl in sorted(set(inst_ng[valid].tolist())):
                new_labels[inst_ng == old_lbl] = next_id
                next_id += 1
            inst_ng = new_labels

        valid_inst = inst_ng > 0
        if valid_inst.sum() > 0 and n_all > 0 and ng2all is not None:
            cab2all = ng2all[cab_idx_ng]
            full_inst[cab2all[valid_inst]] = inst_ng[valid_inst]

    n_inst = len(np.unique(full_inst[full_inst > 0]))
    rt = time.time() - t0
    print(f'  → {n_inst} instances | {rt:.1f}s total', flush=True)

    gs, gi = align_gt_to_pred(gt_pts, gt_sem, gt_inst, xyz_clean)
    if gi is not None:
        gi = gi.copy(); gi[gi <= 0] = 0

    torch.cuda.empty_cache(); gc.collect()
    return {'fname': nm, 'pred_sem': full_sem, 'pred_inst': full_inst,
            'gt_sem': gs, 'gt_inst': gi, 'runtime': rt}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/pn_meanshift.yaml')
    parser.add_argument('--split', default='all', choices=['val', 'test', 'all'])
    parser.add_argument('--output', default=None)
    parser.add_argument('--skip-clustering', action='store_true',
                        help='跳过中间实例聚类, 语义甘蓝点云整体进 GIDM')
    args = parser.parse_args()

    cfg = load_config(args.config)
    np.random.seed(cfg.get('pipeline', {}).get('seed', 42))

    files = sorted(glob.glob('evalaute_test/cloudR*.ply'))
    files = [f for f in files if '_gt' not in os.path.basename(f)
             and 'step' not in os.path.basename(f) and 'sor' not in os.path.basename(f)]
    if args.split != 'all':
        with open('evalaute_test/split.json') as f:
            split_map = json.load(f)
        allowed = set(split_map.get(args.split, []))
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in allowed]
    print(f'Config: {args.config} | Split: {args.split} | Files: {len(files)}')

    model = load_model()
    clusterer = InstanceClusterer(cfg)
    recs = []
    for idx, fp in enumerate(files):
        nm = os.path.splitext(os.path.basename(fp))[0]
        print(f'\n[{idx+1}/{len(files)}] {nm}', flush=True)
        try:
            rec = run_one_file(fp, cfg, model, clusterer, args.skip_clustering)
        except Exception as e:
            print(f'  [ERROR] {nm}: {e}', flush=True)
            torch.cuda.empty_cache(); gc.collect()
            rec = None
        if rec is not None:
            recs.append(rec)

    if not recs:
        print('No valid records!')
        return

    pca_enabled = cfg.get('instance', {}).get('pca_split', {}).get('enable', False)
    method_label = cfg.get('instance', {}).get('method', 'watershed_3d')
    if args.output:
        out_json = args.output
        label = os.path.splitext(os.path.basename(args.output))[0]
    else:
        pca_suffix = '_pca' if pca_enabled else '_nopca'
        nocl_suffix = '_nocluster' if getattr(args, 'skip_clustering', False) else ''
        out_name = f'pointnet2_{method_label}{pca_suffix}{nocl_suffix}_{args.split}'
        out_json = f'output/{out_name}.json'
        label = f'PointNet2 ({method_label} {"+GIDM" if pca_enabled else "no-GIDM"} {args.split})'

    res = evaluate_from_predictions(recs, iou_thresh=0.5, label=label)
    os.makedirs('output', exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print('\n' + '=' * 70)
    print(f'=== {label} ===')
    s = res['summary']
    for k, v in sorted(s.items()):
        print(f'  {k}: {v}')
    print(f'结果 → {out_json}')


if __name__ == '__main__':
    main()
