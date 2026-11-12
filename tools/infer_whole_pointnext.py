#!/usr/bin/env python3
"""Whole-scene sliding-window semantic inference for PointNeXt-L."""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
PN_ROOT = ROOT / 'server_backup/autodl-tmp/PointNeXt'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PN_ROOT))
sys.path.insert(0, str(ROOT / 'server_backup/autodl-tmp/PointNeXt/openpoints/cpp/pointnet2_batch'))
from tools.crops3d_whole_common import (  # noqa: E402
    PRED_ROOT, TEST_NAMES, VAL_NAMES, load_gt, write_json,
)
from openpoints.utils import EasyConfig  # noqa: E402
from openpoints.models import build_model_from_cfg  # noqa: E402

CFG = PN_ROOT / 'cfgs/crops3d/pointnext-l.yaml'
CKPT = (PN_ROOT / 'examples/segmentation/log/crops3d/'
        'crops3d-train-pointnext-l-ngpus1-20261001-033052-GzuQHJgqe4CL7A2EWCHXYv/'
        'checkpoint/'
        'crops3d-train-pointnext-l-ngpus1-20261001-033052-GzuQHJgqe4CL7A2EWCHXYv_ckpt_best.pth')
VOXEL = 0.02
MAXPTS = 24000


def load_model():
    c = EasyConfig(); c.load(str(CFG), recursive=True)
    model = build_model_from_cfg(c.model).cuda().eval()
    sd = torch.load(CKPT, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'PointNeXt load: missing={len(missing)} unexpected={len(unexpected)}', flush=True)
    return model


def _predict_window(model, xyz, rgb):
    center = xyz.mean(0, keepdims=True)
    q = (xyz - center).astype(np.float32)
    # Match the 2 cm voxel preparation used by the PointNeXt Crops3D data pipeline.
    keys = np.floor(q / VOXEL).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    keep = np.sort(keep)
    qk = q[keep]
    ck = rgb[keep]
    # Very thin boundary windows can collapse to <512 unique voxels; the
    # OpenPoints CUDA grouping/FPS kernels are not defined for such tiny sets.
    if len(qk) < 512:
        return None
    if len(qk) > MAXPTS:
        take = np.linspace(0, len(qk) - 1, MAXPTS).astype(np.int64)
        qk, ck = qk[take], ck[take]
        keep = keep[take]
    # PointCloudXYZAlign: xyz centering + gravity-axis minimum alignment.
    qk = qk - qk.mean(0, keepdims=True)
    qk[:, 2] -= qk[:, 2].min()
    pos = torch.from_numpy(qk[None]).float().cuda()
    feats = torch.from_numpy(ck.T[None]).float().cuda()
    with torch.no_grad():
        logits = model({'pos': pos, 'x': feats})
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return logits.argmax(1)[0].cpu().numpy().astype(np.int8), qk, keep


def infer_plot(model, name):
    xyz, rgb01, _, _ = load_gt(name)
    rgb01 = (rgb01 * 2.0 - 1.0).astype(np.float32)
    n = len(xyz)
    votes = np.zeros((n, 2), dtype=np.int32)
    seen = np.zeros(n, dtype=np.int32)
    lo = xyz[:, :2].min(0); hi = xyz[:, :2].max(0)
    for x0 in np.arange(lo[0], hi[0] + 1e-6, 1.0):
        for y0 in np.arange(lo[1], hi[1] + 1e-6, 1.0):
            m = ((xyz[:, 0] >= x0) & (xyz[:, 0] < x0 + 2.0) &
                 (xyz[:, 1] >= y0) & (xyz[:, 1] < y0 + 2.0))
            idx = np.flatnonzero(m)
            if len(idx) < 2000:
                continue
            res = _predict_window(model, xyz[idx], rgb01[idx])
            if res is None:
                continue
            lab, qk, keep = res
            # Vote only on the retained 2 cm voxel points; uncovered points
            # are filled once after all overlapping windows are aggregated.
            sel_idx = idx[keep]
            votes[sel_idx, 0] += (lab == 0)
            votes[sel_idx, 1] += (lab == 1)
            seen[sel_idx] += 1
    pred = votes.argmax(1).astype(np.int8)
    covered = seen > 0
    if covered.any() and not covered.all():
        center = xyz[covered].mean(0, keepdims=True)
        tree = cKDTree(xyz[covered][:, :2] - center[:, :2])
        _, nn = tree.query(xyz[~covered][:, :2] - center[:, :2], k=1)
        pred[~covered] = pred[covered][nn]
    return pred, int(covered.sum()), int((seen > 1).sum())


def run(split, overwrite=False):
    names = TEST_NAMES if split == 'test' else VAL_NAMES
    outdir = PRED_ROOT / 'pointnext_l'; outdir.mkdir(parents=True, exist_ok=True)
    model = load_model()
    rows = []
    for crop, plots in names.items():
        for name in plots:
            out = outdir / f'{name}_pred.npy'
            if out.exists() and not overwrite:
                print('SKIP', name, flush=True); continue
            t0 = time.time()
            pred, covered, overlap = infer_plot(model, name)
            np.save(out, pred)
            row = {'name': name, 'crop': crop, 'n': len(pred),
                   'fg': int((pred == 1).sum()), 'covered': covered,
                   'overlap_points': overlap, 'sec': round(time.time() - t0, 2)}
            rows.append(row)
            print('DONE pointnext_l', row, flush=True)
    write_json(outdir / f'protocol_{split}.json', {
        'model': 'pointnext_l',
        'protocol': 'whole-scene 2 m sliding windows / 1 m stride; 2 cm voxelised point budget 24000; pointwise voting',
        'split': split, 'plots': rows,
    })
    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['train', 'test', 'all'], default='all')
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    if a.split in ('train', 'all'):
        run('train', a.overwrite)
    if a.split in ('test', 'all'):
        run('test', a.overwrite)
