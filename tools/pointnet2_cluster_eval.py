#!/usr/bin/env python3
"""PointNet2 语义分割 + 三种聚类统一评估 (远程 A100)

用法:
  python tools/pointnet2_cluster_eval.py --config configs/pn_meanshift.yaml --split test
  python tools/pointnet2_cluster_eval.py --config configs/pn_meanshift_noPCA.yaml --split test

说明:
  - 语义 backbone: PointNet2_Ours 的 PointNet2Sem (spconv, 384轮, 纯语义无offset)
  - 数据: evalaute_test (21文件, split.json 划分 val/test)
  - instance.method 决定聚类 (meanshift/hdbscan/watershed_3d)
  - instance.pca_split.enable 决定 GIDM
  - 聚类后处理与 evaluation.py 的 clustering 分支对齐
"""
import sys, os, time, json, glob, warnings, argparse, gc
import numpy as np

os.chdir('/root/autodl-tmp')
sys.path.insert(0, '/root/autodl-tmp')
warnings.filterwarnings('ignore')

import torch, open3d as o3d
from scipy.spatial import cKDTree

from cabbage_pheno.io import read_point_cloud
from cabbage_pheno.service.pipeline import preprocess_point_cloud
from cabbage_pheno.service.evaluation import (
    load_config, load_ground_truth, align_gt_to_pred, evaluate_from_predictions,
)
from cabbage_pheno.instance import InstanceClusterer

# --- 先解析用户参数 (避免被 sys.argv 覆盖) ---
_user_parser = argparse.ArgumentParser()
_user_parser.add_argument('--config', default='configs/pn_meanshift.yaml')
_user_parser.add_argument('--split', default='all', choices=['val', 'test', 'all'])
_user_parser.add_argument('--output', default=None)
_user_parser.add_argument('--skip-clustering', action='store_true',
                          help='跳过中间实例聚类, 语义甘蓝点云整体进 GIDM')
USER_ARGS = _user_parser.parse_args()

# --- 加载 PointNet2 配置 (改 sys.argv 后 import) ---
sys.argv = ['eval', '--config', 'PointNet2_Ours/config/pointnet2_cabbage.yaml']
sys.path.insert(0, '/root/autodl-tmp/PointNet2_Ours')
sys.path.insert(0, '/root/autodl-tmp/PointNet2_Ours/model')
from pointnet2_sem import PointNet2Sem, model_fn_decorator as pn2_fn
from util.config import cfg as pn2_cfg
from lib.pointgroup_ops.functions import pointgroup_ops

pn2_cfg.task = 'test'


def load_model():
    model = PointNet2Sem(pn2_cfg).cuda().eval()
    n_params = sum(p.numel() for p in model.parameters())
    exp_dir = '/root/autodl-tmp/PointNet2_Ours/exp/cabbage_dataset/pointnet2_sem/pointnet2_cabbage'
    # 优先 best.pth (epoch 200, val_loss 更低), 否则回退 ckpt_384
    best = os.path.join(exp_dir, 'best.pth')
    ckpt384 = os.path.join(exp_dir, 'pointnet2_cabbage-000000384.pth')
    if os.path.exists(best):
        sd = torch.load(best, map_location='cpu')
        sd = sd['model'] if isinstance(sd, dict) and 'model' in sd else sd
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f'Loaded best.pth (epoch 200), params={n_params/1e6:.2f}M, missing={len(missing)}, unexpected={len(unexpected)}')
    elif os.path.exists(ckpt384):
        sd = torch.load(ckpt384, map_location='cpu')
        sd = sd['model'] if isinstance(sd, dict) and 'model' in sd else sd
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f'Loaded ckpt_384.pth, params={n_params/1e6:.2f}M, missing={len(missing)}, unexpected={len(unexpected)}')
    else:
        raise FileNotFoundError('No PointNet2 checkpoint found')
    mfn = pn2_fn(test=True)
    return model, mfn


def semantic_infer(xyz_ng, rgb_ng, model, mfn):
    """PointNet2 语义推理 (spconv, 一次性)。返回 (N,) {0,1} 标签"""
    N = len(xyz_ng)
    rgb = (rgb_ng * 2 - 1).astype(np.float32)
    xm = xyz_ng - xyz_ng.min(0)
    xs = xm * pn2_cfg.scale
    locs = torch.cat([torch.LongTensor(N, 1).fill_(0), torch.from_numpy(xs).long()], 1)
    vl, p2v, v2p = pointgroup_ops.voxelization_idx(locs, 1, pn2_cfg.mode)
    batch = {
        'locs': locs, 'voxel_locs': vl, 'p2v_map': p2v, 'v2p_map': v2p,
        'locs_float': torch.from_numpy(xm), 'feats': torch.from_numpy(rgb),
        'offsets': torch.tensor([0, N], dtype=torch.int),
        'spatial_shape': np.clip((locs.max(0)[0][1:] + 1).numpy(), pn2_cfg.full_scale[0], None),
    }
    with torch.no_grad():
        preds = mfn(batch, model, pn2_cfg.test_epoch)
    ss = preds['semantic']
    mask = ss.max(1)[1].cpu().numpy() == 1
    del locs, vl, p2v, v2p, batch, preds, ss
    torch.cuda.empty_cache()
    return mask.astype(np.int32)


def run_one_file(fp, cfg, model, mfn, clusterer, skip_clustering=False):
    nm = os.path.splitext(os.path.basename(fp))[0]
    t0 = time.time()

    pcd = read_point_cloud(fp)
    if pcd is None:
        return None
    gt_pts, gt_sem, gt_inst = load_ground_truth(fp, -2, -1, True)
    if gt_inst is None:
        print(f'  [SKIP] {nm}: 无 GT', flush=True)
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
        return {
            'fname': nm, 'pred_sem': np.zeros(n_all, dtype=int),
            'pred_inst': np.full(n_all, -1, dtype=int),
            'gt_sem': np.zeros(0), 'gt_inst': np.zeros(0), 'runtime': time.time() - t0,
        }

    # 语义推理
    t_sem = time.time()
    sem_ng = semantic_infer(xyz_ng, rgb_ng, model, mfn)
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

    # 聚类 (skip_clustering 时跳过中间聚类, 语义甘蓝点云整体进 GIDM)
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

        try:
            inst_ng = clusterer._merge_fragments(inst_ng.copy(), cab_xyz)
        except Exception:
            pass

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
    return {
        'fname': nm, 'pred_sem': full_sem, 'pred_inst': full_inst,
        'gt_sem': gs, 'gt_inst': gi, 'runtime': rt,
    }


def main():
    args = USER_ARGS

    cfg = load_config(args.config)
    np.random.seed(cfg.get('pipeline', {}).get('seed', 42))

    files = sorted(glob.glob('evalaute_test/cloudR*.ply'))
    files = [f for f in files if '_gt' not in os.path.basename(f)
             and 'step' not in os.path.basename(f)]
    if args.split != 'all':
        with open('evalaute_test/split.json') as f:
            split_map = json.load(f)
        allowed = set(split_map.get(args.split, []))
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in allowed]
    print(f'Config: {args.config} | Split: {args.split} | Files: {len(files)}')

    model, mfn = load_model()
    clusterer = InstanceClusterer(cfg)
    recs = []

    for idx, fp in enumerate(files):
        nm = os.path.splitext(os.path.basename(fp))[0]
        print(f'\n[{idx+1}/{len(files)}] {nm}', flush=True)
        try:
            rec = run_one_file(fp, cfg, model, mfn, clusterer, USER_ARGS.skip_clustering)
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
        pca_tag = "+GIDM" if pca_enabled else "no-GIDM"
        label = f'PointNet2 ({method_label} {pca_tag} {args.split})'

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
