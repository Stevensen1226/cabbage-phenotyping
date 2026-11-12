#!/usr/bin/env python3
"""Coordinate-search clustering parameters on validation plots and evaluate the test plots."""
from __future__ import annotations
import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cabbage_pheno.instance.clustering import InstanceClusterer  # noqa: E402
from tools.crops3d_whole_common import (  # noqa: E402
    OUT_ROOT, TEST_NAMES, VAL_NAMES, load_gt, load_pred, metrics, aggregate, write_json,
)

MODELS = ['randla', 'pointnet2_rgb', 'pointnext_l', 'pointgroup']
METHODS = ['watershed_3d', 'meanshift', 'hdbscan']
BASE_CFG = ROOT / 'configs/randla_watershed_dpvis_tuned.yaml'
TUNE_SAMPLE = 100000
TEST_SAMPLE = None  # None => cluster the full foreground of every test plot
TUNE_CROP_PLOTS = {
    'maize': ['maize-1', 'maize-6'],
    'rapeseed': ['rapeseed-5', 'rapeseed-7'],
}


class Pcd:
    def __init__(self, points):
        self.points = np.asarray(points, dtype=np.float32)


def _seed_for(model, name):
    return int(1000 + (sum((i + 1) * ord(c) for i, c in enumerate(model + ':' + name)) % 100000))


def cluster_indices(model, name, pred, max_points):
    fg = np.flatnonzero(pred > 0)
    if max_points is None or len(fg) <= max_points:
        return fg
    rng = np.random.default_rng(_seed_for(model, name))
    return np.sort(fg[rng.choice(len(fg), max_points, replace=False)])


def make_cfg(method, crop, params):
    cfg = yaml.safe_load(BASE_CFG.read_text(encoding='utf-8'))
    inst = cfg.setdefault('instance', {})
    inst['method'] = method
    inst['drop_oversized_clusters'] = False
    inst['max_cluster_points'] = 10_000_000
    inst['noise_reassign_radius'] = 0.03
    inst['noise_reassign_k'] = 5
    inst['noise_reassign_min_votes'] = 3
    # Parameter fractions are applied to the point set actually clustered.
    min_frac = float(params.get('min_cluster_frac', 0.003))
    n_cluster = int(params.get('_n_cluster', 100000))
    inst['min_cluster_points'] = max(10, int(round(min_frac * n_cluster)))
    ps = inst.setdefault('pca_split', {})
    ps.update({
        'enable': bool(params.get('dpvis', False)),
        'horizontal_split_only': True,
        'abnormal_diameter': 0.60 if crop == 'maize' else 0.55,
        'abnormal_aspect_ratio': 1.7,
        'skeleton_bins': 25,
        'min_peak_dist_m': float(params.get('dpvis_peak', 0.25)),
        'valley_depth_rel': 0.85,
        'merge_min_diameter': 0.40,
        'merge_max_dist': 0.30,
        'merge_max_angle': 50,
        'dynamic_min_peak_dist': {'enable': False},
        'vertical_merge': {'enable': False},
    })
    fv = inst.setdefault('fragment_voting', {})
    fv.update({
        'enable': bool(params.get('dpvis', False)),
        'discard_threshold': max(10, int(round(0.0005 * n_cluster))),
        'merge_threshold': max(30, int(round(0.004 * n_cluster))),
        'fragment_merge_max_dist': 0.25,
    })
    if method == 'watershed_3d':
        w = inst.setdefault('watershed_3d', {})
        w.update({
            'voxel_resolution': 0.01,
            'min_seed_distance': float(params.get('min_seed_distance', 0.24)),
            'smoothing_sigma': 1.0,
            'dilation_iters': 1,
            'peak_threshold_abs_m': 0.006,
            'peak_threshold_rel': 0.05,
            'max_voxels': 50_000_000,
        })
    elif method == 'meanshift':
        inst['meanshift'] = {
            'bandwidth': float(params.get('bandwidth', 0.20)),
            'bin_seeding': True,
            'min_bin_freq': 10,
        }
    elif method == 'hdbscan':
        inst['hdbscan'] = {
            'min_cluster_size': float(params.get('min_cluster_size_frac', 0.004)),
            'min_samples': int(params.get('min_samples', 5)),
            'cluster_selection_method': 'eom',
            'fallback_leaf': True,
            'allow_single_cluster': False,
            'max_cluster_fraction': 0.60,
        }
    return cfg


def run_one(model, crop, method, params, names, split, max_points):
    rows = []
    for name in names:
        xyz, rgb, sem, inst = load_gt(name)
        pred = load_pred(model, name)
        idx = cluster_indices(model, name, pred, max_points)
        cfg = copy.deepcopy(params); cfg['_n_cluster'] = len(idx)
        labels, _ = InstanceClusterer(make_cfg(method, crop, cfg)).cluster(Pcd(xyz[idx]))
        if split == 'train':
            rows.append(metrics(labels, inst[idx]))
        else:
            fg_idx = np.flatnonzero(pred > 0)
            if len(idx) == len(fg_idx):
                full = np.full(len(xyz), -1, dtype=np.int64)
                full[fg_idx] = labels
            else:
                tree = cKDTree(xyz[idx])
                _, nn = tree.query(xyz[fg_idx], k=1, workers=-1)
                full = np.full(len(xyz), -1, dtype=np.int64)
                full[fg_idx] = labels[nn]
            r = metrics(full, inst); r['name'] = name; rows.append(r)
    return rows


def tune_model_crop_method(model, crop, method):
    names = TUNE_CROP_PLOTS[crop]
    cache = {}

    def evaluate(params):
        key = tuple(sorted((k, v) for k, v in params.items() if k != 'dpvis'))
        if key in cache:
            return cache[key]
        p_base = dict(params); p_base['dpvis'] = False
        p_dp = dict(params); p_dp['dpvis'] = True
        rows_b = run_one(model, crop, method, p_base, names, 'train', TUNE_SAMPLE)
        rows_d = run_one(model, crop, method, p_dp, names, 'train', TUNE_SAMPLE)
        ab, ad = aggregate(rows_b), aggregate(rows_d)
        score = ad['F1'] + 0.15 * ab['F1'] - 0.01 * ad['MAE']
        rec = {'score': score, 'base': ab, 'dpvis': ad, 'params': {k: v for k, v in params.items() if k != 'dpvis'}}
        cache[key] = rec
        print(f'TUNE {model} {crop} {method} {rec["params"]} base={ab} dpvis={ad} score={score:.4f}', flush=True)
        return rec

    if method == 'watershed_3d':
        params = {'min_seed_distance': 0.24, 'min_cluster_frac': 0.003, 'dpvis_peak': 0.25}
        axes = [
            ('min_seed_distance', [0.18, 0.22, 0.26, 0.30]),
            ('min_cluster_frac', [0.0001, 0.0003, 0.001]),
            ('dpvis_peak', [0.20, 0.30]),
        ]
    elif method == 'meanshift':
        params = {'bandwidth': 0.20, 'min_cluster_frac': 0.003, 'dpvis_peak': 0.25}
        axes = [
            ('bandwidth', [0.12, 0.16, 0.20, 0.26, 0.32]),
            ('min_cluster_frac', [0.0001, 0.0003, 0.001]),
            ('dpvis_peak', [0.20, 0.30]),
        ]
    else:
        params = {'min_cluster_size_frac': 0.004, 'min_samples': 5,
                  'min_cluster_frac': 0.002, 'dpvis_peak': 0.25}
        axes = [
            ('min_cluster_size_frac', [0.002, 0.005, 0.01, 0.02]),
            ('min_samples', [3, 5, 10]),
            ('min_cluster_frac', [0.0001, 0.0003]),
            ('dpvis_peak', [0.20, 0.30]),
        ]

    history = []
    best = evaluate(params)
    for axis, values in axes:
        cand_records = []
        for value in values:
            p = dict(params); p[axis] = value
            rec = evaluate(p)
            cand_records.append((rec['score'], value, rec))
        _, value, rec = max(cand_records, key=lambda x: x[0])
        params[axis] = value
        best = rec
    return {'model': model, 'crop': crop, 'method': method,
            'selected': params, 'validation': best}


def run_tuning(models, methods, crops):
    out = {}
    for model in models:
        for crop in crops:
            for method in methods:
                key = f'{model}|{crop}|{method}'
                out[key] = tune_model_crop_method(model, crop, method)
                write_json(OUT_ROOT / 'tuning.json', out)
    return out


def evaluate_test(models, methods, crops, tuning):
    result = {'protocol': {
        'scene': 'whole scene per plot',
        'clustering_sample_max': None if TEST_SAMPLE is None else TEST_SAMPLE,
        'label_propagation': 'nearest neighbour in 3D from clustering sample to all foreground points',
        'parameters': 'selected on official-train validation plots; base and +DPVIS share clustering parameters',
    }, 'models': {}}
    for model in models:
        result['models'][model] = {}
        for crop in crops:
            result['models'][model][crop] = {}
            for method in methods:
                key = f'{model}|{crop}|{method}'
                selected = tuning[key]['selected']
                for condition, dpvis in (('base', False), ('dpvis', True)):
                    p = dict(selected); p['dpvis'] = dpvis
                    rows = run_one(model, crop, method, p, TEST_NAMES[crop], 'test', TEST_SAMPLE)
                    result['models'][model][crop][f'{method}_{condition}'] = {
                        'params': {k: v for k, v in p.items() if k != 'dpvis'},
                        'summary': aggregate(rows),
                        'plots': rows,
                    }
                    print('TEST', model, crop, method, condition,
                          result['models'][model][crop][f'{method}_{condition}']['summary'], flush=True)
        write_json(OUT_ROOT / 'unified_results.json', result)
    return result


def evaluate_pointgroup_native():
    out = {'protocol': 'PointGroup native whole-scene proposals merged across 2 m / 1 m blocks',
           'crops': {}}
    for crop, names in TEST_NAMES.items():
        rows = []
        for name in names:
            xyz, rgb, sem, inst = load_gt(name)
            pred = np.load(OUT_ROOT / 'preds/pointgroup' / f'{name}_inst.npy')
            r = metrics(pred, inst, zero_is_background=True); r['name'] = name; rows.append(r)
        out['crops'][crop] = {'summary': aggregate(rows), 'plots': rows}
    write_json(OUT_ROOT / 'pointgroup_native_whole.json', out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['tune', 'eval', 'native', 'all'], default='all')
    ap.add_argument('--models', nargs='+', default=MODELS, choices=MODELS)
    ap.add_argument('--methods', nargs='+', default=METHODS, choices=METHODS)
    ap.add_argument('--crops', nargs='+', default=['maize', 'rapeseed'], choices=['maize', 'rapeseed'])
    a = ap.parse_args()
    tp = OUT_ROOT / 'tuning.json'
    tuning = json.loads(tp.read_text(encoding='utf-8')) if tp.exists() else {}
    if a.phase in ('tune', 'all'):
        tuning = run_tuning(a.models, a.methods, a.crops)
    if a.phase in ('eval', 'all'):
        if not tuning:
            tuning = json.loads(tp.read_text(encoding='utf-8'))
        evaluate_test(a.models, a.methods, a.crops, tuning)
    if a.phase in ('native', 'all'):
        evaluate_pointgroup_native()


if __name__ == '__main__':
    main()
