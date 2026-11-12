#!/usr/bin/env python3
"""Refine PointGroup native instance labels with DPVIS and visualize."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib import cm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cabbage_pheno.instance.clustering import InstanceClusterer
from tools.eval_tiles_instances import inst_metrics


def refine_labels(clusterer, xyz, labels):
    out = labels.copy()
    if clusterer.pca_enable:
        out = clusterer._apply_skeleton_split(out, xyz)
        if clusterer.merge_back:
            out = clusterer._merge_fragments(out, xyz)
        if getattr(clusterer, 'vertical_merge_enable', False):
            out = clusterer._merge_vertical_fragments(out, xyz)
    if clusterer.vote_enable:
        out = clusterer._cleanup_tiny_fragments(out, xyz)
    out = clusterer._simple_filter(out, n_points=len(xyz))
    out, n = clusterer._renumber_labels(out)
    return out, n


def instance_colors(vals):
    colors = np.zeros((len(vals), 4), dtype=float)
    colors[:, :3] = 0.82
    colors[:, 3] = 1.0
    pos = vals >= 0
    if pos.any():
        uniq = np.unique(vals[pos])
        palette = cm.turbo(np.linspace(0.04, 0.96, max(1, len(uniq))))
        colors[pos] = palette[np.searchsorted(uniq, vals[pos])]
    return colors


def style_3d(ax, xyz, title):
    lo = np.percentile(xyz, 1, axis=0)
    hi = np.percentile(xyz, 99, axis=0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((hi - lo))
    ax.view_init(elev=24, azim=-62)
    ax.set_proj_type('ortho')
    ax.set_axis_off()
    ax.set_title(title, fontsize=10, pad=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', nargs='+', required=True, help='NPZ files with xyz, gt, pred')
    ap.add_argument('--config', default=str(ROOT / 'configs/default.yaml'))
    ap.add_argument('--output', default=str(ROOT / 'output/crops3d_generalization/figures/pointgroup_native_dpvis.png'))
    ap.add_argument('--json', default=str(ROOT / 'output/crops3d_generalization/unified_fixed/pointgroup_native_dpvis_metrics.json'))
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    clusterer = InstanceClusterer(cfg)
    items = []
    for f in a.npz:
        p = Path(f)
        d = np.load(p)
        xyz = d['xyz'].astype(np.float32)
        gt = d['gt'].astype(np.int64)
        pred = d['pred'].astype(np.int64)
        refined, n = refine_labels(clusterer, xyz, pred)
        before = inst_metrics(pred, gt)
        after = inst_metrics(refined, gt)
        items.append((p.stem, xyz, gt, pred, refined, before, after))
        print(p.stem, 'native', before, 'dpvis', after, 'clusters', n, flush=True)

    rows = len(items)
    fig = plt.figure(figsize=(15, 4.5 * rows), dpi=300)
    for r, (name, xyz, gt, pred, refined, before, after) in enumerate(items):
        rng = np.random.default_rng(0)
        sel = rng.choice(len(xyz), size=min(70000, len(xyz)), replace=False)
        panels = [('GT instances', gt), ('PointGroup native', pred), ('PointGroup + DPVIS', refined)]
        for c, (title, vals) in enumerate(panels):
            ax = fig.add_subplot(rows, 3, r * 3 + c + 1, projection='3d')
            v = vals[sel]
            ax.scatter(xyz[sel, 0], xyz[sel, 1], xyz[sel, 2], c=instance_colors(v), s=0.22, alpha=0.82, depthshade=False, rasterized=True)
            style_3d(ax, xyz[sel], title)
            if c == 0:
                ax.text2D(-0.03, 0.5, name, transform=ax.transAxes, rotation=90, va='center', ha='right', fontsize=9)
    fig.suptitle('PointGroup native instance segmentation with DPVIS refinement', fontsize=12, y=1.01)
    fig.tight_layout()
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    result = {name: {'native': before, 'dpvis': after} for name, _, _, _, _, before, after in items}
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('saved', a.output, a.json)


if __name__ == '__main__':
    main()