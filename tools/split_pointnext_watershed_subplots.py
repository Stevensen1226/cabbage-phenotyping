#!/usr/bin/env python3
"""Split PointNeXt-L + Watershed example panels into title-free high-resolution subplots."""
from __future__ import annotations
import csv, json, re, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(r'E:\Cabbage')
sys.path.insert(0, str(ROOT))
import tools.render_pointnext_watershed_same_examples as rws

FIG = ROOT / 'output/crops3d_generalization/figures'
OUT = FIG / 'split_subplots_fullres'
MANIFEST = OUT / 'manifest_pointnext_watershed.csv'


def san(s):
    s = str(s).strip().replace('\n', '_').replace(' ', '_')
    s = re.sub(r'[^0-9A-Za-z._-]+', '_', s)
    return re.sub(r'_+', '_', s).strip('_')


def panel_style(ax, xyz):
    lo = np.percentile(xyz, 1, axis=0)
    hi = np.percentile(xyz, 99, axis=0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.view_init(24, -62); ax.set_proj_type('ortho'); ax.set_axis_off()


def save_panel(path, xyz, col, size):
    fig = plt.figure(figsize=(5.2, 5.2), dpi=400)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=col, s=size,
               alpha=1.0, depthshade=False, rasterized=True)
    panel_style(ax, xyz)
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=400, bbox_inches='tight', pad_inches=0.02, facecolor='white')
    plt.close(fig)


def save_row_manifest(manifest, crop, scene_id, xyz, gt, rgb, sem, base, dp):
    pal = rws.gt_palette(gt)
    panels = [
        ('original_rgb', np.clip(rgb, 0, 1), 0.035),
        ('gt_instances', rws.gt_colors(gt, pal), 0.045),
        ('pointnext_semantic', rws.sem_colors(sem), 0.045),
        ('pointnext_watershed', rws.pred_colors(base, gt, pal), 0.050),
        ('pointnext_watershed_dpvis', rws.pred_colors(dp, gt, pal), 0.050),
    ]
    outdir = OUT / crop / 'pointnext_watershed'
    outdir.mkdir(parents=True, exist_ok=True)
    scene_s = san(scene_id)
    for panel, col, size in panels:
        path = outdir / f'{scene_s}__{panel}.png'
        save_panel(path, xyz, col, size)
        manifest.append({'crop': crop, 'method': 'pointnext_watershed',
                         'scene': scene_id, 'panel': panel,
                         'path': str(path.relative_to(ROOT))})
        print('saved', path, flush=True)


def main():
    rws.SUBPLOTS = OUT
    pgm = np.load(rws.LOG / 'pg_maize_tall_screen.npz')
    rap = np.load(rws.LOG / 'pg_rapeseed_hybrid_vis_rapsplit_058.npz')
    rlex = np.load(rws.LOG / 'randla_examples_data.npz', allow_pickle=True)
    tuning = json.loads((ROOT / 'output/crops3d_whole_scene/tuning.json').read_text(encoding='utf-8'))
    manifest = []
    model = rws.load_model()

    for key, pgkey in [('r0', 'rapeseed-4_t001'), ('r1', 'rapeseed-4_t000')]:
        xyz = rap[f'{pgkey}_xyz'].astype(np.float32)
        gt = rap[f'{pgkey}_gt'].astype(np.int64)
        rgb = rlex[f'{key}_rgb'].astype(np.float32)
        sem = rws.ground_filter(xyz, rws.infer_tile(model, xyz, rgb))
        base, dp = rws.run_clustering(xyz, sem, 'rapeseed', tuning['pointnext_l|rapeseed|watershed_3d']['selected'])
        save_row_manifest(manifest, 'rapeseed', pgkey, xyz, gt, rgb, sem, base, dp)

    for pgkey, lbl in [('maize-12_t001', 4), ('maize-14_t001', 2)]:
        xyz_all = pgm[f'{pgkey}_xyz'].astype(np.float32)
        gt_all = pgm[f'{pgkey}_gt'].astype(np.int64)
        rgb_all = pgm[f'{pgkey}_rgb'].astype(np.float32)
        rgb01 = np.clip((rgb_all + 1.0) * 0.5, 0, 1)
        sem_all = rws.infer_tile(model, xyz_all, rgb01)
        m = pgm[f'{pgkey}_native'] == lbl
        xyz, gt, rgb = xyz_all[m], gt_all[m], rgb01[m]
        sem = rws.ground_filter(xyz, sem_all[m])
        base, dp = rws.run_clustering(xyz, sem, 'maize', tuning['pointnext_l|maize|watershed_3d']['selected'])
        save_row_manifest(manifest, 'maize', f'{pgkey}_native_cluster_{lbl}', xyz, gt, rgb, sem, base, dp)

    OUT.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['crop', 'method', 'scene', 'panel', 'path'])
        w.writeheader(); w.writerows(manifest)
    print('saved', len(manifest), 'subplots to', OUT)


if __name__ == '__main__':
    main()
