#!/usr/bin/env python3
"""PointNeXt 语义分割 + 三种聚类统一评估 (本地 RTX 5070, 分块推理防显存爆炸)

用法:
  # 单配置单 split (MeanShift + GIDM, test)
  python tools/pointnext_cluster_eval.py --config configs/pn_meanshift.yaml --split test

  # 一次跑三种聚类 (有 GIDM)
  for m in meanshift hdbscan watershed; do
    python tools/pointnext_cluster_eval.py --config configs/pn_${m}.yaml --split test
  done

说明:
  - 数据: evalaute_test (21 文件, split.json 划分 val/test)
  - backbone 由 config 决定 (非 randla 都加载 PointNeXt)
  - instance.method 决定聚类 (meanshift/hdbscan/watershed_3d)
  - instance.pca_split.enable 决定 GIDM
  - 分块推理: 单次最多 20480 点, 多轮采样+最近邻传播, 防显存 OOM
"""
import sys, os, time, json, glob, warnings, argparse, gc
import numpy as np

os.chdir('/home/stevensen/Cabbage')
sys.path.insert(0, '/home/stevensen/Cabbage')
sys.path.insert(0, '/home/stevensen/PointNeXt-master')
warnings.filterwarnings('ignore')

import torch, open3d as o3d
from scipy.spatial import cKDTree

from openpoints.models import build_model_from_cfg
from openpoints.utils import EasyConfig
from cabbage_pheno.io import read_point_cloud
from cabbage_pheno.service.pipeline import preprocess_point_cloud
from cabbage_pheno.service.evaluation import (
    load_config, load_ground_truth, align_gt_to_pred,
    evaluate_from_predictions,
)
from cabbage_pheno.instance import InstanceClusterer
from RandLANet_Ours.randlanet import RandLANet

# 训练时归一化统计量
GLOBAL_MEAN = np.array([-0.10488096, 0.18573543, 0.00078407], dtype=np.float32)
GLOBAL_STD  = np.array([0.8321599,  1.4525476,  0.09171805], dtype=np.float32)

SAMPLE_SIZE = 20480   # 单次推理点数 (匹配训练)
GPU_MAX     = 25000   # GPU 安全上限 (超此点数走分块多轮采样)


def _single_infer_prob(xyz, rgb, model, backbone):
    """返回 softmax 概率 (N, 2)。xyz=(N,3), rgb=(N,3)∈[0,1]"""
    N = len(xyz)
    if backbone.startswith('randla'):
        xyz_norm = xyz - xyz.min(0, keepdims=True)
        xyz_t = torch.FloatTensor(xyz_norm).unsqueeze(0).cuda()
        rgb_norm = rgb.astype(np.float32) * 2.0 - 1.0
        feat_t = torch.FloatTensor(rgb_norm).unsqueeze(0).cuda()
        with torch.no_grad():
            out = model(xyz_t, feat_t)
        prob = torch.softmax(out.squeeze(0), dim=-1).cpu().numpy()
        del xyz_t, feat_t, out
        return prob

    xyz_norm = (xyz - GLOBAL_MEAN) / (GLOBAL_STD + 1e-8)
    xyz_t = torch.FloatTensor(xyz_norm).unsqueeze(0).cuda()
    feat = np.concatenate([xyz_norm, rgb], axis=1).astype(np.float32)
    feat_t = torch.FloatTensor(feat).unsqueeze(0).permute(0, 2, 1).cuda()
    off_t = torch.tensor([N], dtype=torch.int32).unsqueeze(0).cuda()
    with torch.no_grad():
        out = model({'pos': xyz_t, 'x': feat_t, 'offset': off_t})
    prob = torch.softmax(out.squeeze(0), dim=0).cpu().numpy().T
    del xyz_t, feat_t, off_t, out
    return prob


def block_inference(points_xyz, points_rgb, model, backbone):
    """多轮随机采样 + 概率累加 + 最近邻传播。防显存 OOM。"""
    N = len(points_xyz)
    if N <= GPU_MAX:
        prob = _single_infer_prob(points_xyz, points_rgb, model, backbone)
        return prob.argmax(1).astype(np.int32)

    n_rounds = max(5, int(np.ceil(3.0 * N / SAMPLE_SIZE)))
    n_rounds = min(n_rounds, 30)
    print(f'  [{N} pts → {n_rounds} rounds]', end='', flush=True)

    all_prob = np.zeros((N, 2), dtype=np.float32)
    for r in range(n_rounds):
        choice = np.random.choice(N, min(N, SAMPLE_SIZE), replace=False)
        prob_sub = _single_infer_prob(points_xyz[choice], points_rgb[choice], model, backbone)
        tree = cKDTree(points_xyz[choice])
        _, nn = tree.query(points_xyz, k=1)
        all_prob += prob_sub[nn]
        if (r + 1) % 5 == 0:
            torch.cuda.empty_cache()

    return all_prob.argmax(1).astype(np.int32)


def run_one_file(fp, cfg, model, backbone, clusterer, skip_clustering=False):
    """对单个文件执行: 预处理 → 语义推理 → 聚类 → 后处理 → 返回评估记录"""
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
            'gt_sem': np.zeros(0), 'gt_inst': np.zeros(0),
            'runtime': time.time() - t0,
        }

    # 语义推理
    t_sem = time.time()
    sem_ng = block_inference(xyz_ng, rgb_ng, model, backbone)
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

    torch.cuda.empty_cache()
    gc.collect()
    return {
        'fname': nm, 'pred_sem': full_sem, 'pred_inst': full_inst,
        'gt_sem': gs, 'gt_inst': gi, 'runtime': rt,
    }


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

    # 收集数据 (evalaute_test) + split 过滤
    files = sorted(glob.glob('evalaute_test/cloudR*.ply'))
    files = [f for f in files if '_gt' not in os.path.basename(f)
             and 'step' not in os.path.basename(f)]
    if args.split != 'all':
        with open('evalaute_test/split.json') as f:
            split_map = json.load(f)
        allowed = set(split_map.get(args.split, []))
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in allowed]
    print(f'Config: {args.config} | Split: {args.split} | Files: {len(files)}')

    # 加载 backbone
    backbone = cfg.get('segmentation', {}).get('backbone', 'pointnet2').lower()
    if backbone.startswith('randla'):
        print('  Loading RandLA-Net...')
        model = RandLANet(d_in=6, num_classes=2).cuda().eval()
        ck = None
        for p in sorted(glob.glob('RandLANet_Ours/exp/randlanet/*.pth')):
            if 'best' in p or 'ckpt' in p:
                ck = p; break
        if ck:
            sd = torch.load(ck, map_location='cpu')
            model.load_state_dict(sd['model'] if isinstance(sd, dict) and 'model' in sd else sd, strict=False)
        print('  RandLA-Net loaded')
    else:
        print('  Loading PointNeXt...')
        pn_cfg = EasyConfig()
        pn_cfg.load('/home/stevensen/PointNeXt-master/cfgs/cabbage.yaml', recursive=True)
        model = build_model_from_cfg(pn_cfg.model).cuda().eval()
        sd = torch.load('/home/stevensen/PointNeXt-master/best_model.pth', map_location='cpu')
        model.load_state_dict(sd, strict=False)
        print(f'  PointNeXt-L: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params')

    clusterer = InstanceClusterer(cfg)
    recs = []

    for idx, fp in enumerate(files):
        nm = os.path.splitext(os.path.basename(fp))[0]
        print(f'\n[{idx+1}/{len(files)}] {nm}', flush=True)
        try:
            rec = run_one_file(fp, cfg, model, backbone, clusterer, args.skip_clustering)
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
        out_name = f'pointnext_{method_label}{pca_suffix}{nocl_suffix}_{args.split}'
        out_json = f'output/{out_name}.json'
        pca_tag = "+GIDM" if pca_enabled else "no-GIDM"
        label = f'PointNeXt-L ({method_label} {pca_tag} {args.split})'

    res = evaluate_from_predictions(recs, iou_thresh=0.5, label=label)
    os.makedirs('output', exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print('\n' + '=' * 70)
    print(f'=== {label} ===')
    print('=' * 70)
    s = res['summary']
    for k, v in sorted(s.items()):
        print(f'  {k}: {v}')
    print(f'结果 → {out_json}')


if __name__ == '__main__':
    main()
