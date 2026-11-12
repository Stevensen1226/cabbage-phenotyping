#!/usr/bin/env python3
"""PointNeXt-L + Watershed examples on the same point clouds as the RandLA examples."""
from __future__ import annotations
import copy, json, sys, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

ROOT = Path(r'E:\Cabbage')
sys.path.insert(0, str(ROOT))
from cabbage_pheno.instance.clustering import InstanceClusterer
from tools.tune_eval_whole_clustering import make_cfg, Pcd
from tools.infer_whole_pointnext import load_model, _predict_window

LOG = ROOT / 'server_backup/autodl-tmp/logs'
FIG = ROOT / 'output/crops3d_generalization/figures'
FIG.mkdir(parents=True, exist_ok=True)


def infer_tile(model, xyz, rgb01):
    rgb = (rgb01 * 2.0 - 1.0).astype(np.float32)
    res = _predict_window(model, xyz.astype(np.float32), rgb)
    if res is None:
        raise RuntimeError('PointNeXt inference returned no window')
    lab, _, keep = res
    pred = np.zeros(len(xyz), dtype=np.int8)
    pred[keep] = lab
    seen = np.zeros(len(xyz), dtype=bool)
    seen[keep] = True
    if seen.any() and not seen.all():
        tree = cKDTree(xyz[seen][:, :2])
        _, nn = tree.query(xyz[~seen][:, :2], k=1, workers=-1)
        pred[~seen] = pred[seen][nn]
    return pred


def fill_foreground(labels, xyz, sem, radius=0.05, k=5, max_passes=50):
    out = labels.copy()
    fg = sem == 1
    for _ in range(max_passes):
        noise = np.where(fg & (out < 0))[0]
        valid = np.where(fg & (out >= 0))[0]
        if len(noise) == 0 or len(valid) == 0:
            break
        tree = cKDTree(xyz[valid])
        kk = min(k, len(valid))
        d, nn = tree.query(xyz[noise], k=kk, distance_upper_bound=radius, workers=-1)
        if kk == 1:
            d = d[:, None]
            nn = nn[:, None]
        idx, labs = [], []
        for j in range(len(noise)):
            ok = np.isfinite(d[j])
            if not np.any(ok):
                continue
            vals, cnt = np.unique(out[valid[nn[j][ok]]], return_counts=True)
            idx.append(noise[j])
            labs.append(vals[np.argmax(cnt)])
        if not idx:
            break
        out[np.asarray(idx)] = np.asarray(labs)
    noise = np.where(fg & (out < 0))[0]
    valid = np.where(fg & (out >= 0))[0]
    if len(noise) and len(valid):
        _, nn = cKDTree(xyz[valid]).query(xyz[noise], k=1, workers=-1)
        out[noise] = out[valid][nn]
    return out


def ground_filter(xyz, pred, low_h=0.03, high_h=0.10, high_r=0.08, k=50):
    xy = xyz[:, :2]
    tree = cKDTree(xy)
    kk = min(k, len(xyz))
    _, nn = tree.query(xy, k=kk, workers=-1)
    if kk == 1:
        nn = nn[:, None]
    gz = np.percentile(xyz[nn, 2], 5, axis=1)
    h = xyz[:, 2] - gz
    fg = pred == 1
    high = fg & (h > high_h)
    out = pred.copy()
    if high.sum() > 0:
        dh, _ = cKDTree(xyz[high]).query(xyz, k=1, workers=-1)
        keep = fg & ((h > low_h) | (dh < high_r))
    else:
        keep = fg & (h > low_h)
    out[~keep] = 0
    return out


def run_clustering(xyz, sem, crop, params):
    fg = sem == 1
    labels = {}
    for cond, dpvis in [('base', False), ('dp', True)]:
        p = dict(params)
        p['dpvis'] = dpvis
        cfg = make_cfg('watershed_3d', crop, {**p, '_n_cluster': int(fg.sum())})
        lab, _ = InstanceClusterer(cfg).cluster(Pcd(xyz[fg]))
        full = np.full(len(xyz), -1, dtype=np.int64)
        full[np.flatnonzero(fg)] = np.asarray(lab).astype(np.int64)
        labels[cond] = fill_foreground(full, xyz, sem)
    return labels['base'], labels['dp']


def gt_palette(gt):
    vals = np.unique(gt[gt >= 0])
    cols = plt.get_cmap('turbo')(np.linspace(0.04, 0.96, max(1, len(vals))))
    return {int(v): cols[i] for i, v in enumerate(vals)}


def gt_colors(vals, pal):
    c = np.zeros((len(vals), 4)); c[:, :3] = 0.82; c[:, 3] = 1.0
    for v, col in pal.items():
        c[vals == v] = col
    return c


def match_pred_to_gt(pred, gt):
    m = gt >= 0
    p, g = pred[m], gt[m]
    pairs, counts = np.unique(np.stack([p, g], 1), axis=0, return_counts=True)
    inter = {}
    for (a, b), n in zip(pairs, counts):
        a, b, n = int(a), int(b), int(n)
        if a >= 0:
            inter[(a, b)] = n
    best = {}
    for (a, b), n in inter.items():
        if n > best.get(a, (-1, None))[0]:
            best[a] = (n, b)
    return {a: b for a, (_, b) in best.items()}


def pred_colors(pred, gt, pal):
    c = np.zeros((len(pred), 4)); c[:, :3] = 0.82; c[:, 3] = 1.0
    mp = match_pred_to_gt(pred, gt)
    for p in np.unique(pred):
        p = int(p)
        if p >= 0:
            c[pred == p] = pal.get(mp.get(p, -1), np.array([0.82, 0.82, 0.82, 1.0]))
    return c


def sem_colors(sem):
    c = np.zeros((len(sem), 4)); c[:, :3] = 0.82; c[:, 3] = 1.0
    c[sem > 0] = [0.08, 0.72, 0.25, 1.0]
    return c


def style(ax, xyz, title):
    lo = np.percentile(xyz, 1, axis=0)
    hi = np.percentile(xyz, 99, axis=0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.view_init(24, -62); ax.set_proj_type('ortho'); ax.set_axis_off()
    ax.set_title(title, fontsize=24, pad=2)


def render(rows, outpath):
    fig = plt.figure(figsize=(25, 5.4 * len(rows)), dpi=300)
    for r, (label, xyz, gt, rgb, sem, base, dp) in enumerate(rows):
        pal = gt_palette(gt)
        rng = np.random.default_rng(0)
        sel = rng.choice(len(xyz), size=min(70000, len(xyz)), replace=False)
        panels = [
            ('Original RGB', np.clip(rgb, 0, 1)),
            ('GT instances', gt_colors(gt, pal)),
            ('PointNeXt-L semantic', sem_colors(sem)),
            ('PointNeXt-L +\nwatershed', pred_colors(base, gt, pal)),
            ('PointNeXt-L +\nwatershed + DPVIS', pred_colors(dp, gt, pal)),
        ]
        for c, (title, col) in enumerate(panels):
            ax = fig.add_subplot(len(rows), 5, r * 5 + c + 1, projection='3d')
            ax.scatter(xyz[sel, 0], xyz[sel, 1], xyz[sel, 2], c=col[sel],
                       s=0.35, alpha=1.0, depthshade=False, rasterized=True)
            style(ax, xyz[sel], title)
            if c == 0:
                ax.text2D(-0.04, 0.5, label, transform=ax.transAxes,
                          rotation=90, va='center', ha='right', fontsize=20)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', outpath, flush=True)


def main():
    pgm = np.load(LOG / 'pg_maize_tall_screen.npz')
    rap = np.load(LOG / 'pg_rapeseed_hybrid_vis_rapsplit_058.npz')
    rlex = np.load(LOG / 'randla_examples_data.npz', allow_pickle=True)
    tuning = json.loads((ROOT / 'output/crops3d_whole_scene/tuning.json').read_text(encoding='utf-8'))

    model = load_model()
    rows = []

    # RandLA example rows use these exact two rapeseed tiles.
    for key, pgkey in [('r0', 'rapeseed-4_t001'), ('r1', 'rapeseed-4_t000')]:
        xyz = rap[f'{pgkey}_xyz'].astype(np.float32)
        gt = rap[f'{pgkey}_gt'].astype(np.int64)
        rgb = rlex[f'{key}_rgb'].astype(np.float32)
        sem0 = infer_tile(model, xyz, rgb)
        sem = ground_filter(xyz, sem0)
        base, dp = run_clustering(xyz, sem, 'rapeseed', tuning['pointnext_l|rapeseed|watershed_3d']['selected'])
        rows.append((pgkey, xyz, gt, rgb, sem, base, dp))
        print('inferred', pgkey, 'fg', int((sem == 1).sum()), flush=True)
    render(rows, FIG / 'pointnext_rapeseed_examples.png')

    rows = []
    # Same native clusters as the maize RandLA examples: maize-12 cluster 4 and maize-14 cluster 2.
    for pgkey, lbl in [('maize-12_t001', 4), ('maize-14_t001', 2)]:
        xyz_all = pgm[f'{pgkey}_xyz'].astype(np.float32)
        gt_all = pgm[f'{pgkey}_gt'].astype(np.int64)
        rgb_all = pgm[f'{pgkey}_rgb'].astype(np.float32)
        rgb01 = np.clip((rgb_all + 1.0) * 0.5, 0, 1)
        sem_all = infer_tile(model, xyz_all, rgb01)
        m = pgm[f'{pgkey}_native'] == lbl
        xyz, gt, rgb = xyz_all[m], gt_all[m], rgb01[m]
        sem = ground_filter(xyz, sem_all[m])
        base, dp = run_clustering(xyz, sem, 'maize', tuning['pointnext_l|maize|watershed_3d']['selected'])
        rows.append((f'{pgkey}\nnative cluster {lbl}', xyz, gt, rgb, sem, base, dp))
        print('inferred', pgkey, 'cluster', lbl, 'n', len(xyz), 'fg', int((sem == 1).sum()), flush=True)
    render(rows, FIG / 'pointnext_maize_examples.png')

    del model
    import torch
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
