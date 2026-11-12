#!/usr/bin/env python3
"""Whole-scene block inference for PointGroup (semantic mask + native instances)."""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import yaml
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.crops3d_whole_common import PRED_ROOT, TEST_NAMES, VAL_NAMES, load_gt, write_json  # noqa: E402
from main import pointgroup_segment_blocks  # noqa: E402


def base_cfg(crop='maize'):
    cfg = yaml.safe_load((ROOT / 'configs/randla_watershed_dpvis_tuned.yaml').read_text(encoding='utf-8'))
    seg = cfg.setdefault('segmentation', {})
    seg['pointgroup_config'] = 'PointGroup_Ours/config/pointgroup_cabbage_paper.yaml'
    seg['pointgroup_pretrain'] = str(ROOT / 'server_backup/autodl-tmp/runs/pointgroup_final/best.pth')
    seg['block_inference'] = {
        'enable': True,
        'block_size': 2.0,
        'overlap': 1.0,
        'max_single_extent': 0.0,
        'merge_iou_thresh': 0.25,
    }
    seg['pg_score_thresh'] = 0.09
    seg['pg_npoint_thresh'] = 100
    seg['pg_sem_thresh'] = 0
    # Crop-specific cross-block merge / global NMS thresholds selected on
    # official-train validation plots.
    if crop == 'maize':
        seg['pg_block_overlap_thresh'] = 0.85
        seg['pg_global_nms_iou'] = 0.40
        seg['pg_global_nms_overlap'] = 0.80
        seg['pg_global_nms_centroid'] = 0.00
    else:
        seg['pg_block_overlap_thresh'] = 0.35
        seg['pg_global_nms_iou'] = 0.30
        seg['pg_global_nms_overlap'] = 0.65
        seg['pg_global_nms_centroid'] = 0.08
    return cfg


def infer_plot(name):
    xyz, rgb, _, _ = load_gt(name)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    crop = 'maize' if name.startswith('maize') else 'rapeseed'
    cpcd, inst_c, mask_c, inst_global = pointgroup_segment_blocks(
        pcd, base_cfg(crop), method='hybrid', selection='nms', return_meta=True
    )
    # pred is a dense whole-scene semantic mask; native instances are also
    # restored to the full plot length (0 = background).
    pred = mask_c.astype(np.int8)
    return pred, inst_global, len(cpcd.points), int(inst_global.max())


def run(split, overwrite=False):
    names = TEST_NAMES if split == 'test' else VAL_NAMES
    outdir = PRED_ROOT / 'pointgroup'; outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for crop, plots in names.items():
        for name in plots:
            p = outdir / f'{name}_pred.npy'
            if p.exists() and not overwrite:
                print('SKIP', name, flush=True); continue
            t0 = time.time()
            pred, inst, n_fg, n_native = infer_plot(name)
            np.save(p, pred)
            np.save(outdir / f'{name}_inst.npy', inst)
            row = {'name': name, 'crop': crop, 'n': int(len(pred)),
                   'fg': int(pred.sum()), 'native_instances': n_native,
                   'native_points': int((inst > 0).sum()), 'sec': round(time.time() - t0, 2)}
            rows.append(row)
            print('DONE pointgroup', row, flush=True)
    write_json(outdir / f'protocol_{split}.json', {
        'model': 'pointgroup',
        'protocol': 'whole-scene 2 m blocks / 1 m stride; 50% semantic voting + native proposal merge',
        'split': split, 'plots': rows,
    })


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['train', 'test', 'all'], default='train')
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    if a.split in ('train', 'all'):
        run('train', a.overwrite)
    if a.split in ('test', 'all'):
        run('test', a.overwrite)
