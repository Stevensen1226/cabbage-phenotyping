"""Scene-adaptive watershed seed selection for instance segmentation."""
from copy import deepcopy
import numpy as np


def cluster_with_adaptive_seed(clusterer_cls, cfg, pcd, candidates=(0.20, 0.22, 0.25, 0.28, 0.30), max_unassigned_frac=0.10):
    """Run the full clustering/DPVIS pipeline at multiple seed distances.

    The largest candidate whose foreground points remain assigned at a rate of
    at least ``1 - max_unassigned_frac`` is selected.  If none qualifies, the
    candidate with the lowest unassigned fraction is returned.
    """
    best = None
    for seed in sorted((float(x) for x in candidates), reverse=True):
        c = deepcopy(cfg)
        c.setdefault('instance', {}).setdefault('watershed_3d', {})
        c['instance']['watershed_3d']['min_seed_distance'] = float(seed)
        labels, n_inst = clusterer_cls(c).cluster(pcd)
        labels = np.asarray(labels)
        unassigned = float((labels < 0).mean()) if labels.size else 1.0
        rec = {'seed': float(seed), 'unassigned_frac': unassigned, 'n_instances': int(n_inst)}
        if unassigned <= float(max_unassigned_frac):
            return labels, int(n_inst), rec
        if best is None or unassigned < best[0]:
            best = (unassigned, labels, int(n_inst), rec)
    return best[1], best[2], best[3]
