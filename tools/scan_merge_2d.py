#!/usr/bin/env python3
"""二维扫描 merge_back 参数 (merge_min_diameter × merge_max_dist)。

语义推理只做一次，聚类+合并对参数网格分别执行。

用法:
  python tools/scan_merge_2d.py --config configs/pn_watershed_3d.yaml --split test
"""
import sys, os, json, glob, warnings, argparse, gc, copy
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
    load_config, load_ground_truth, align_gt_to_pred, evaluate_from_predictions,
)
from cabbage_pheno.instance import InstanceClusterer

GLOBAL_MEAN = np.array([-0.10488096, 0.18573543, 0.00078407], dtype=np.float32)
GLOBAL_STD  = np.array([0.8321599,  1.4525476,  0.09171805], dtype=np.float32)
SAMPLE_SIZE = 20480
GPU_MAX     = 25000

MD_VALUES    = [0.15, 0.30, 0.50, 0.70]      # merge_min_diameter
MDIST_VALUES = [0.20, 0.30, 0.40, 0.50]      # merge_max_dist


def _single_infer_prob(xyz, rgb, model):
    N = len(xyz)
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


def block_inference(points_xyz, points_rgb, model):
    N = len(points_xyz)
    if N <= GPU_MAX:
        return _single_infer_prob(points_xyz, points_rgb, model).argmax(1).astype(np.int32)
    n_rounds = max(5, int(np.ceil(3.0 * N / SAMPLE_SIZE)))
    n_rounds = min(n_rounds, 30)
    all_prob = np.zeros((N, 2), dtype=np.float32)
    for r in range(n_rounds):
        choice = np.random.choice(N, min(N, SAMPLE_SIZE), replace=False)
        prob_sub = _single_infer_prob(points_xyz[choice], points_rgb[choice], model)
        tree = cKDTree(points_xyz[choice])
        _, nn = tree.query(points_xyz, k=1)
        all_prob += prob_sub[nn]
        if (r + 1) % 5 == 0:
            torch.cuda.empty_cache()
    return all_prob.argmax(1).astype(np.int32)


def postprocess_instances(inst_ng, cab_xyz, cfg, clusterer):
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
    return inst_ng


def load_model():
    pn_cfg = EasyConfig()
    pn_cfg.load('/home/stevensen/PointNeXt-master/cfgs/cabbage.yaml', recursive=True)
    model = build_model_from_cfg(pn_cfg.model).cuda().eval()
    sd = torch.load('/home/stevensen/PointNeXt-master/best_model.pth', map_location='cpu')
    model.load_state_dict(sd, strict=False)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/pn_watershed_3d.yaml')
    parser.add_argument('--split', default='test', choices=['val', 'test', 'all'])
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    np.random.seed(base_cfg.get('pipeline', {}).get('seed', 42))

    files = sorted(glob.glob('evalaute_test/cloudR*.ply'))
    files = [f for f in files if '_gt' not in os.path.basename(f)
             and 'step' not in os.path.basename(f)]
    if args.split != 'all':
        with open('evalaute_test/split.json') as f:
            split_map = json.load(f)
        allowed = set(split_map.get(args.split, []))
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in allowed]

    model = load_model()
    combos = [(md, mdst) for md in MD_VALUES for mdst in MDIST_VALUES]
    print(f'二维扫描 {len(combos)} 组合 (md × max_dist), 文件数={len(files)}', flush=True)

    records_by_combo = {c: [] for c in combos}

    for idx, fp in enumerate(files):
        nm = os.path.splitext(os.path.basename(fp))[0]
        pcd = read_point_cloud(fp)
        if pcd is None:
            continue
        gt_pts, gt_sem, gt_inst = load_ground_truth(fp, -2, -1, True)
        if gt_inst is None:
            continue
        pr = preprocess_point_cloud(pcd, base_cfg)
        xyz_clean = pr.points_clean
        ng_pcd = pr.non_ground_pcd
        xyz_ng = np.asarray(ng_pcd.points, np.float32)
        rgb_ng = np.asarray(ng_pcd.colors, np.float32) if len(ng_pcd.colors) > 0 \
                 else np.ones((len(xyz_ng), 3), np.float32) * 0.5
        n_all = len(xyz_clean)
        n_ng = len(xyz_ng)

        sem_ng = block_inference(xyz_ng, rgb_ng, model)
        n_cab = int(sem_ng.sum())
        full_sem = np.zeros(n_all, dtype=int)
        cab_mask_ng = sem_ng == 1
        cab_idx_ng = np.where(cab_mask_ng)[0]
        ng2all = None
        if n_all > 0 and n_ng > 0:
            t_ng = cKDTree(xyz_ng)
            _, ng2all = t_ng.query(xyz_clean, k=1)
            full_sem = sem_ng[ng2all]

        gs, gi = align_gt_to_pred(gt_pts, gt_sem, gt_inst, xyz_clean)
        if gi is not None:
            gi = gi.copy(); gi[gi <= 0] = 0

        cab_xyz = xyz_ng[cab_mask_ng] if n_cab >= 50 else None

        for md, mdst in combos:
            full_inst = np.full(n_all, -1, dtype=int)
            if cab_xyz is not None and len(cab_xyz) >= 50:
                cfg_tmp = copy.deepcopy(base_cfg)
                cfg_tmp['instance']['pca_split']['merge_min_diameter'] = md
                cfg_tmp['instance']['pca_split']['merge_max_dist'] = mdst
                clusterer = InstanceClusterer(cfg_tmp)
                cab_pcd = o3d.geometry.PointCloud()
                cab_pcd.points = o3d.utility.Vector3dVector(cab_xyz)
                try:
                    inst_ng, _ = clusterer.cluster(cab_pcd)
                except Exception:
                    inst_ng = np.full(len(cab_xyz), -1, dtype=np.int64)
                inst_ng = postprocess_instances(inst_ng, cab_xyz, cfg_tmp, clusterer)
                valid_inst = inst_ng > 0
                if valid_inst.sum() > 0 and n_all > 0 and ng2all is not None:
                    cab2all = ng2all[cab_idx_ng]
                    full_inst[cab2all[valid_inst]] = inst_ng[valid_inst]
            records_by_combo[(md, mdst)].append({
                'fname': nm, 'pred_sem': full_sem, 'pred_inst': full_inst,
                'gt_sem': gs, 'gt_inst': gi, 'runtime': 0.0,
            })
        print(f'[{idx+1}/{len(files)}] {nm} 完成 ({n_cab} pts)', flush=True)
        torch.cuda.empty_cache(); gc.collect()

    print('\n' + '=' * 100)
    print('merge_back 二维扫描 (merge_min_diameter × merge_max_dist) — F1 / MAE')
    print('=' * 100)
    header = '            ' + ''.join(f'md_dist={m:.2f}m'.center(12) for m in MDIST_VALUES)
    print(header)
    results = {}
    for md in MD_VALUES:
        line = f'md={md:.2f}m    '
        for mdst in MDIST_VALUES:
            res = evaluate_from_predictions(records_by_combo[(md, mdst)], iou_thresh=0.5, label='')
            s = res['summary']
            results[f'{md}_{mdst}'] = s
            line += f'{s["avg_inst_f1"]:.3f}/{s["mae_count"]:.1f}'.center(12)
        print(line)

    print('\n完整指标 (Prec/Rec/F1/Inst-mIoU/Sem-mIoU/MAE):')
    for md in MD_VALUES:
        for mdst in MDIST_VALUES:
            s = results[f'{md}_{mdst}']
            print(f'  md={md:.2f} dist={mdst:.2f}: prec={s["avg_inst_prec"]:.3f} rec={s["avg_inst_rec"]:.3f} '
                  f'f1={s["avg_inst_f1"]:.3f} imiou={s["avg_inst_miou"]:.3f} sem={s["avg_sem_miou"]:.3f} mae={s["mae_count"]:.2f}')

    out = {'md_values': MD_VALUES, 'mdist_values': MDIST_VALUES,
           'results': {k: v for k, v in results.items()}}
    os.makedirs('output', exist_ok=True)
    with open('output/merge_2d_scan.json', 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('\n结果 → output/merge_2d_scan.json')


if __name__ == '__main__':
    main()
