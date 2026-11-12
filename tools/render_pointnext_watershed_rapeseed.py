#!/usr/bin/env python3
"""Render PointNeXt-L + Watershed rapeseed panels with the final whole-scene parameters."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(r'E:\Cabbage')
sys.path.insert(0, str(ROOT))
from cabbage_pheno.instance.clustering import InstanceClusterer
from tools.crops3d_whole_common import load_gt, load_pred
from tools.tune_eval_whole_clustering import make_cfg, Pcd


def tile_indices(xyzc, k):
    lo = xyzc[:, :2].min(0)
    hi = xyzc[:, :2].max(0)
    kk = 0
    for x0 in np.arange(lo[0], hi[0], 1.0):
        for y0 in np.arange(lo[1], hi[1], 1.0):
            m = ((xyzc[:, 0] >= x0) & (xyzc[:, 0] < x0 + 2) &
                 (xyzc[:, 1] >= y0) & (xyzc[:, 1] < y0 + 2))
            if int(m.sum()) < 2000:
                continue
            if kk == k:
                return np.flatnonzero(m)
            kk += 1
    raise ValueError(k)


def gt_palette(inst):
    vals = np.unique(inst[inst > 0])
    cols = plt.get_cmap('turbo')(np.linspace(0.04, 0.96, max(1, len(vals))))
    return {int(v): cols[i] for i, v in enumerate(vals)}


def gt_colors(inst, pal):
    c = np.zeros((len(inst), 4), dtype=float)
    c[:, :3] = 0.82
    c[:, 3] = 1.0
    for v, col in pal.items():
        c[inst == v] = col
    return c


def semantic_colors(sem):
    c = np.zeros((len(sem), 4), dtype=float)
    c[:, :3] = 0.82
    c[:, 3] = 1.0
    c[sem > 0] = [0.08, 0.72, 0.25, 1.0]
    return c


def match_to_gt(labels, inst):
    valid = inst > 0
    p = np.asarray(labels)[valid]
    g = np.asarray(inst)[valid]
    pairs, counts = np.unique(np.stack([p, g], axis=1), axis=0, return_counts=True)
    best = {}
    for (a, b), n in zip(pairs, counts):
        a, b, n = int(a), int(b), int(n)
        if a < 0:
            continue
        if n > best.get(a, (-1, None))[0]:
            best[a] = (n, b)
    return {a: b for a, (_, b) in best.items()}


def prediction_colors(labels, inst, pal):
    c = np.zeros((len(labels), 4), dtype=float)
    c[:, :3] = 0.82
    c[:, 3] = 1.0
    mp = match_to_gt(labels, inst)
    for lab in np.unique(labels):
        lab = int(lab)
        if lab >= 0:
            c[labels == lab] = pal.get(mp.get(lab, -1), np.array([0.25, 0.25, 0.25, 1.0]))
    return c


def draw_ax(ax, xyz, col, size):
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=col, s=size,
               alpha=1.0, depthshade=False, rasterized=True)
    lo = np.percentile(xyz, 1, axis=0)
    hi = np.percentile(xyz, 99, axis=0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.view_init(24, -62)
    ax.set_proj_type('ortho')
    ax.set_axis_off()


def save_panel(path, xyz, col, size):
    fig = plt.figure(figsize=(5.2, 5.2), dpi=400)
    ax = fig.add_subplot(111, projection='3d')
    draw_ax(ax, xyz, col, size)
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=400, bbox_inches='tight', pad_inches=0.02, facecolor='white')
    plt.close(fig)


def main():
    name = 'rapeseed-4'
    tile_k = 2
    outdir = ROOT / 'output/crops3d_generalization/figures/split_subplots_fullres/rapeseed/pointnext_watershed'
    outdir.mkdir(parents=True, exist_ok=True)
    composite = ROOT / 'output/crops3d_generalization/figures/pointnext_watershed_rapeseed_examples.png'

    xyz, rgb, _, inst = load_gt(name)
    pred = load_pred('pointnext_l', name).astype(np.int64)
    labels_path = ROOT / 'output/crops3d_whole_scene/preds/pointnext_l' / f'{name}_watershed.npy'
    params_path = ROOT / 'output/crops3d_whole_scene/preds/pointnext_l' / f'{name}_watershed_params.json'

    tuning = json.loads((ROOT / 'output/crops3d_whole_scene/tuning.json').read_text(encoding='utf-8'))
    params = dict(tuning['pointnext_l|rapeseed|watershed_3d']['selected'])
    params['dpvis'] = False

    if labels_path.exists():
        full_labels = np.load(labels_path)
        print('loaded', labels_path)
    else:
        fg = np.flatnonzero(pred > 0)
        cfg = make_cfg('watershed_3d', 'rapeseed', {**params, '_n_cluster': len(fg)})
        t0 = time.time()
        labels, _ = InstanceClusterer(cfg).cluster(Pcd(xyz[fg]))
        full_labels = np.full(len(xyz), -1, dtype=np.int64)
        full_labels[fg] = labels.astype(np.int64)
        np.save(labels_path, full_labels)
        params_path.write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'clustered {len(fg)} foreground points in {time.time()-t0:.1f}s', flush=True)

    xyzc = xyz - xyz.mean(0)
    idx = tile_indices(xyzc, tile_k)
    q = xyzc[idx].copy()
    q -= q.mean(0, keepdims=True)
    q_rgb = rgb[idx]
    q_inst = inst[idx]
    q_sem = pred[idx]
    q_lab = full_labels[idx]
    pal = gt_palette(q_inst)

    panels = [
        ('original_rgb', q_rgb, 0.038),
        ('gt_instances', gt_colors(q_inst, pal), 0.045),
        ('pointnext_semantic', semantic_colors(q_sem), 0.045),
        ('pointnext_watershed', prediction_colors(q_lab, q_inst, pal), 0.05),
    ]
    paths = []
    for panel, col, size in panels:
        p = outdir / f'{name}_t{tile_k:03d}__{panel}.png'
        save_panel(p, q, col, size)
        paths.append(p)
        print('saved', p)

    fig = plt.figure(figsize=(10.4, 10.4), dpi=400)
    for i, (_, col, size) in enumerate(panels):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        draw_ax(ax, q, col, size)
    fig.tight_layout(pad=0)
    fig.savefig(composite, dpi=400, bbox_inches='tight', pad_inches=0.02, facecolor='white')
    plt.close(fig)
    print('saved', composite)


if __name__ == '__main__':
    main()
