#!/usr/bin/env python3
"""Common utilities for the unified Crops3D whole-scene comparison."""
from __future__ import annotations
from pathlib import Path
import json
import os
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / 'data/external/crops3d_is_converted'
OUT_ROOT = ROOT / 'output/crops3d_whole_scene'
PRED_ROOT = OUT_ROOT / 'preds'
CACHE_ROOT = OUT_ROOT / 'gt_cache'

TEST_NAMES = {
    'maize': ['maize-11', 'maize-12', 'maize-14', 'maize-2', 'maize-3'],
    'rapeseed': ['rapeseed-11', 'rapeseed-3', 'rapeseed-4', 'rapeseed-6'],
}
# A small official-train subset used only for clustering-parameter selection.
VAL_NAMES = {
    'maize': ['maize-1', 'maize-4', 'maize-6'],
    'rapeseed': ['rapeseed-5', 'rapeseed-7', 'rapeseed-8'],
}
ALL_NAMES = {k: TEST_NAMES[k] + VAL_NAMES[k] for k in TEST_NAMES}


def plot_split(name: str) -> str:
    for split, groups in (('test', TEST_NAMES), ('train', VAL_NAMES)):
        if any(name in v for v in groups.values()):
            return split
    raise KeyError(name)


def gt_path(name: str) -> Path:
    return DATA_ROOT / plot_split(name) / f'{name}_gt.txt'


def load_gt(name: str, use_cache: bool = True):
    """Return xyz, rgb (0..1), semantic, instance, using a .npz cache."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache = CACHE_ROOT / f'{name}.npz'
    if use_cache and cache.exists():
        z = np.load(cache)
        return z['xyz'], z['rgb'], z['sem'], z['inst']
    p = gt_path(name)
    b = p.read_bytes().replace(b'\n', b' ')
    a = np.fromstring(b, dtype=np.float64, sep=' ')
    a = a[: a.size - (a.size % 8)].reshape(-1, 8)
    xyz = np.ascontiguousarray(a[:, :3], dtype=np.float32)
    rgb = np.ascontiguousarray(a[:, 3:6] / 255.0, dtype=np.float32)
    sem = a[:, 6].astype(np.int8)
    inst = a[:, 7].astype(np.int32)
    if use_cache:
        np.savez_compressed(cache, xyz=xyz, rgb=rgb, sem=sem, inst=inst)
    return xyz, rgb, sem, inst


def metrics(pred, gt, iou_thresh: float = 0.5, zero_is_background: bool = False):
    """Greedy one-to-one IoU matching; background is instance id <= 0."""
    valid = gt > 0
    p = np.asarray(pred)[valid].astype(np.int64)
    g = np.asarray(gt)[valid].astype(np.int64)
    pairs, counts = np.unique(np.stack([p, g], axis=1), axis=0, return_counts=True)
    pt, gtt, inter = {}, {}, {}
    for (a, b), c in zip(pairs, counts):
        a, b, c = int(a), int(b), int(c)
        gtt[b] = gtt.get(b, 0) + c
        if (a <= 0 if zero_is_background else a < 0):
            continue
        pt[a] = pt.get(a, 0) + c
        inter[(a, b)] = c
    cands = []
    for (a, b), c in inter.items():
        iou = c / (pt[a] + gtt[b] - c)
        if iou >= iou_thresh:
            cands.append((iou, a, b))
    cands.sort(reverse=True)
    mp, mg = set(), set()
    for _, a, b in cands:
        if a in mp or b in mg:
            continue
        mp.add(a); mg.add(b)
    tp = len(mp); npr = len(pt); ngt = len(gtt)
    precision = tp / npr if npr else 0.0
    recall = tp / ngt if ngt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        'tp': tp, 'n_pred': npr, 'n_gt': ngt,
        'P': precision, 'R': recall, 'F1': f1,
        'MAE': abs(npr - ngt),
    }


def aggregate(rows):
    tp = sum(int(r['tp']) for r in rows)
    npr = sum(int(r['n_pred']) for r in rows)
    ngt = sum(int(r['n_gt']) for r in rows)
    p = tp / npr if npr else 0.0
    r = tp / ngt if ngt else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {
        'plots': len(rows), 'TP': tp, 'n_pred': npr, 'n_gt': ngt,
        'P': round(float(p), 4), 'R': round(float(r), 4),
        'F1': round(float(f1), 4),
        'MAE': round(float(np.mean([r['MAE'] for r in rows])), 4),
    }


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def load_pred(model: str, name: str) -> np.ndarray:
    p = PRED_ROOT / model / f'{name}_pred.npy'
    if not p.exists():
        raise FileNotFoundError(p)
    return np.load(p)
