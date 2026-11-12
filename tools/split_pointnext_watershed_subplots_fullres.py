#!/usr/bin/env python3
"""PointNeXt-L + Watershed full-resolution subplots on the same raw 2 m tiles as before."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(r'E:\Cabbage')
sys.path.insert(0, str(ROOT))
import tools.split_example_subplots_fullres as full
import tools.render_pointnext_watershed_same_examples as rws

PGM = np.load(full.LOG / 'pg_maize_tall_screen.npz')
RAP = np.load(full.LOG / 'pg_rapeseed_hybrid_vis_rapsplit_058.npz')
RLEX = np.load(full.LOG / 'randla_examples_data.npz', allow_pickle=True)
TUNING = json.loads((ROOT / 'output/crops3d_whole_scene/tuning.json').read_text(encoding='utf-8'))
OUT = full.OUT
MANIFEST = OUT / 'manifest_pointnext_watershed.csv'


def pointnext_example(model, key, pgkey, crop):
    if crop == 'rapeseed':
        xyz = RAP[f'{pgkey}_xyz'].astype(np.float32)
        rgb = RLEX[f'{key}_rgb'].astype(np.float32)
        sem = rws.ground_filter(xyz, rws.infer_tile(model, xyz, rgb))
        base, dp = rws.run_clustering(xyz, sem, crop, TUNING[f'pointnext_l|{crop}|watershed_3d']['selected'])
        return xyz, sem, base, dp
    xyz_all = PGM[f'{pgkey}_xyz'].astype(np.float32)
    rgb_all = PGM[f'{pgkey}_rgb'].astype(np.float32)
    rgb01 = np.clip((rgb_all + 1.0) * 0.5, 0, 1)
    sem_all = rws.infer_tile(model, xyz_all, rgb01)
    lbl = {'maize-12_t001': 4, 'maize-14_t001': 2}[pgkey]
    m = PGM[f'{pgkey}_native'] == lbl
    xyz = xyz_all[m]
    sem = rws.ground_filter(xyz, sem_all[m])
    base, dp = rws.run_clustering(xyz, sem, crop, TUNING[f'pointnext_l|{crop}|watershed_3d']['selected'])
    return xyz, sem, base, dp


def save_panels(crop, scene, key, raw_plot, tile_k, native_lbl, model, manifest, pg_scene=None):
    pg_scene = pg_scene or scene
    q, rgb, gt = full.load_tile(raw_plot, tile_k)
    if native_lbl is not None:
        nat = full.transfer(q, PGM[f'{pg_scene}_xyz'], PGM[f'{pg_scene}_native'])
        keep = nat == native_lbl
        q, rgb, gt = q[keep], rgb[keep], gt[keep]
    ex_xyz, ex_sem, ex_base, ex_dp = pointnext_example(model, key, pg_scene, crop)
    sem = full.transfer(q, ex_xyz, ex_sem)
    base = full.transfer(q, ex_xyz, ex_base)
    dp = full.transfer(q, ex_xyz, ex_dp)
    pal = full.gt_pal(gt)
    panels = [
        ('original_rgb', rgb, 0.038),
        ('gt_instances', full.gt_col(gt, pal), 0.045),
        ('pointnext_semantic', full.sem_col(sem), 0.045),
        ('pointnext_watershed', full.pred_col(base, gt, pal), 0.050),
        ('pointnext_watershed_dpvis', full.pred_col(dp, gt, pal), 0.050),
    ]
    outdir = OUT / crop / 'pointnext_watershed'
    outdir.mkdir(parents=True, exist_ok=True)
    for panel, col, size in panels:
        p = outdir / f'{full.san(scene)}__{panel}.png'
        if not p.exists():
            full.render(p, q, col, size=size)
        manifest.append({'crop': crop, 'method': 'pointnext_watershed', 'scene': scene,
                         'panel': panel, 'path': str(p.relative_to(ROOT))})
        print('saved', p, 'points', len(q), flush=True)


def main():
    manifest = []
    model = rws.load_model()
    save_panels('rapeseed', 'rapeseed-4_t001', 'r0', 'rapeseed-4', 1, None, model, manifest)
    save_panels('rapeseed', 'rapeseed-4_t000', 'r1', 'rapeseed-4', 0, None, model, manifest)
    save_panels('maize', 'maize-12_t001_native_cluster_4', 'm0', 'maize-12', 1, 4, model, manifest, pg_scene='maize-12_t001')
    save_panels('maize', 'maize-14_t001_native_cluster_2', 'm1', 'maize-14', 1, 2, model, manifest, pg_scene='maize-14_t001')
    with MANIFEST.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['crop', 'method', 'scene', 'panel', 'path'])
        w.writeheader(); w.writerows(manifest)
    print('saved', len(manifest), 'full-resolution subplots to', OUT)


if __name__ == '__main__':
    main()
