#!/usr/bin/env python3
"""Whole-scene sliding-window semantic inference for RandLA-Net and PointNet2-RGB."""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'RandLANet_Ours'))
from tools.crops3d_whole_common import (  # noqa: E402
    PRED_ROOT, TEST_NAMES, VAL_NAMES, load_gt, write_json,
)
from randlanet import RandLANet  # noqa: E402
from cabbage_pheno.segmentation.pointnet2_model import PointNet2SemSeg  # noqa: E402

CKPT = {
    'randla': ROOT / 'models/paper_training_20260923/randlanet_final/best.pth',
    'pointnet2_rgb': ROOT / 'models/paper_training_20260923/pointnet2_rgb/best_model.pth',
}


def _vote_windows(model_name, model, xyz, rgb, block, stride, npts, min_pts):
    """Dense point votes from overlapping XY windows."""
    lo = xyz[:, :2].min(0)
    hi = xyz[:, :2].max(0)
    votes = np.zeros((len(xyz), 2), dtype=np.int32)
    seen = np.zeros(len(xyz), dtype=np.int32)
    x_steps = np.arange(lo[0], hi[0] + 1e-6, stride)
    y_steps = np.arange(lo[1], hi[1] + 1e-6, stride)
    for x0 in x_steps:
        for y0 in y_steps:
            m = ((xyz[:, 0] >= x0) & (xyz[:, 0] < x0 + block) &
                 (xyz[:, 1] >= y0) & (xyz[:, 1] < y0 + block))
            idx = np.flatnonzero(m)
            if len(idx) < min_pts:
                continue
            # Uniform deterministic point budget, same convention as prior
            # whole-scene RandLA/PointNet2 evaluation.
            if len(idx) >= npts:
                sel = idx[np.linspace(0, len(idx) - 1, npts).astype(np.int64)]
            else:
                sel = idx
            q = xyz[sel].copy()
            center = q.mean(0, keepdims=True)
            q -= center
            c = rgb[sel]
            if model_name == 'randla':
                out = model(torch.from_numpy(q).unsqueeze(0).cuda(),
                            torch.from_numpy(c).unsqueeze(0).cuda())
                lab = out.argmax(-1)[0].cpu().numpy()
            else:
                out = model(torch.from_numpy(q.T).unsqueeze(0).cuda(),
                            torch.from_numpy(c.T).unsqueeze(0).cuda())
                if isinstance(out, (tuple, list)):
                    out = out[0]
                lab = out.argmax(1)[0].cpu().numpy()
            votes[sel, 0] += (lab == 0)
            votes[sel, 1] += (lab == 1)
            seen[sel] += 1
    pred = votes.argmax(1).astype(np.int8)
    covered = seen > 0
    if covered.any() and not covered.all():
        tree = cKDTree(xyz[covered, :2])
        _, nn = tree.query(xyz[~covered, :2], k=1)
        pred[~covered] = pred[covered][nn]
    return pred, int(covered.sum()), int((seen > 1).sum())


def run_model(model_name, split, overwrite=False):
    names = (TEST_NAMES if split == 'test' else VAL_NAMES)
    outdir = PRED_ROOT / model_name
    outdir.mkdir(parents=True, exist_ok=True)
    if model_name == 'randla':
        ck = torch.load(CKPT['randla'], map_location='cpu', weights_only=False)
        model = RandLANet(d_in=6, num_classes=2).cuda().eval()
        model.load_state_dict(ck['model'])
        block, stride, npts, minpts = 1.0, 0.5, 65536, 100
    else:
        ck = torch.load(CKPT['pointnet2_rgb'], map_location='cpu', weights_only=False)
        model = PointNet2SemSeg(2, additional_channel=3).cuda().eval()
        model.load_state_dict(ck['model'])
        block, stride, npts, minpts = 1.0, 0.5, 4096, 100

    rows = []
    for crop, plots in names.items():
        for name in plots:
            out = outdir / f'{name}_pred.npy'
            if out.exists() and not overwrite:
                print(f'SKIP {model_name} {name}', flush=True)
                continue
            t0 = time.time()
            xyz, rgb01, _, _ = load_gt(name)
            rgb = (rgb01 * 2.0 - 1.0).astype(np.float32)
            with torch.no_grad():
                pred, covered, overlap = _vote_windows(
                    model_name, model, xyz, rgb, block, stride, npts, minpts
                )
            np.save(out, pred)
            row = {'name': name, 'crop': crop, 'n': int(len(xyz)),
                   'fg': int((pred == 1).sum()), 'covered': covered,
                   'overlap_points': overlap, 'sec': round(time.time() - t0, 2)}
            rows.append(row)
            print(f'DONE {model_name} {name} {row}', flush=True)
    write_json(outdir / f'protocol_{split}.json', {
        'model': model_name,
        'protocol': f'unified whole-scene sliding-window inference, block={block} m, stride={stride} m',
        'split': split,
        'plots': rows,
    })
    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['randla', 'pointnet2_rgb', 'all'], required=True)
    ap.add_argument('--split', choices=['train', 'test', 'all'], default='all')
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    models = ['randla', 'pointnet2_rgb'] if a.model == 'all' else [a.model]
    for model in models:
        if a.split == 'all':
            run_model(model, 'train', a.overwrite); run_model(model, 'test', a.overwrite)
        else:
            run_model(model, a.split, a.overwrite)


if __name__ == '__main__':
    main()
