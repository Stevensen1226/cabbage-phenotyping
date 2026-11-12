import numpy as np
import logging
import time
from collections import Counter
from sklearn.cluster import MeanShift, AgglomerativeClustering, DBSCAN, MiniBatchKMeans, OPTICS, Birch
from sklearn.neighbors import kneighbors_graph

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

try:
    from scipy.stats import mode
except ImportError:
    mode = None

try:
    from scipy.ndimage import distance_transform_edt, gaussian_filter, binary_dilation
    from scipy.signal import find_peaks
except ImportError:
    distance_transform_edt = None
    gaussian_filter = None
    binary_dilation = None
    find_peaks = None

try:
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
except ImportError:
    peak_local_max = None
    watershed = None

logger = logging.getLogger(__name__)

class InstanceClusterer:
    def __init__(self, cfg):
        self.inst_cfg = cfg.get('instance', {})
        self.method = self.inst_cfg.get('method', 'watershed_3d')

        # Watershed 3D params
        self.ws3d_cfg = self.inst_cfg.get('watershed_3d', {})
        self.ws3d_res = self.ws3d_cfg.get('voxel_resolution', 0.01)
        self.ws3d_min_seed_dist = self.ws3d_cfg.get('min_seed_distance', 0.22)
        self.ws3d_sigma = self.ws3d_cfg.get('smoothing_sigma', 0.8)
        self.ws3d_dilation_iters = int(self.ws3d_cfg.get('dilation_iters', 1))
        self.ws3d_max_voxels = int(self.ws3d_cfg.get('max_voxels', 50_000_000))
        self.ws3d_peak_threshold_abs_m = self.ws3d_cfg.get('peak_threshold_abs_m', None)
        self.ws3d_peak_threshold_rel = float(self.ws3d_cfg.get('peak_threshold_rel', 0.0) or 0.0)
        self.ws3d_max_seeds = int(self.ws3d_cfg.get('max_seeds', 0) or 0)

        # Mean Shift params
        self.ms_cfg = self.inst_cfg.get('meanshift', {})
        self.ms_bandwidth = self.ms_cfg.get('bandwidth', 0.25)
        self.ms_bin_seeding = bool(self.ms_cfg.get('bin_seeding', True))
        self.ms_min_bin_freq = int(self.ms_cfg.get('min_bin_freq', 10))

        # Euclidean Clustering params (using Open3D or DBSCAN backend)
        self.ec_cfg = self.inst_cfg.get('euclidean', {})
        self.ec_tolerance = float(self.ec_cfg.get('tolerance', 0.05))
        self.ec_min_size = int(self.ec_cfg.get('min_cluster_size', 50))

        # DBSCAN params
        self.db_cfg = self.inst_cfg.get('dbscan', {})
        self.db_eps = float(self.db_cfg.get('eps', 0.05))
        self.db_min_samples = int(self.db_cfg.get('min_samples', 10))

        # HDBSCAN params
        self.hdb_cfg = self.inst_cfg.get('hdbscan', {})
        # Values in (0, 1) are interpreted as fractions of the foreground
        # point count at fit time.  Integers retain the historical absolute
        # behaviour, so existing cabbage configs remain valid.
        self.hdb_min_cluster_size = self.hdb_cfg.get('min_cluster_size', 500)
        self.hdb_min_samples = self.hdb_cfg.get('min_samples', 10)
        self.hdb_selection_method = self.hdb_cfg.get('cluster_selection_method', 'eom')
        self.hdb_fallback_leaf = bool(self.hdb_cfg.get('fallback_leaf', True))
        self.hdb_allow_single_cluster = bool(self.hdb_cfg.get('allow_single_cluster', False))
        self.hdb_max_cluster_fraction = self.hdb_cfg.get('max_cluster_fraction', None)
        self.dpc_cfg = self.inst_cfg.get('dpc', {})
        self.dpc_k = int(self.dpc_cfg.get('k', 20))
        self.dpc_center_fraction = float(self.dpc_cfg.get('center_fraction', 1e-4))
        self.dpc_min_centers = int(self.dpc_cfg.get('min_centers', 4))
        self.dpc_max_centers = int(self.dpc_cfg.get('max_centers', 80))
        self.dpc_min_sep = float(self.dpc_cfg.get('min_center_sep_m', 0.25))
        self.dpc_assignment = str(self.dpc_cfg.get('assignment', 'nearest_center')).lower()

        # Grid-seeded MeanShift++-style mode seeking.
        self.mspp_cfg = self.inst_cfg.get('meanshift_pp', {})
        self.mspp_bandwidth = float(self.mspp_cfg.get('bandwidth', 0.25))
        self.mspp_grid_size = float(self.mspp_cfg.get('grid_size_m', 0.10))
        self.mspp_max_seeds = int(self.mspp_cfg.get('max_seeds', 5000))

        # MiniBatch KMeans baseline parameters.
        self.km_cfg = self.inst_cfg.get('kmeans', {})
        self.km_n_clusters = int(self.km_cfg.get('n_clusters', 18))
        self.km_feature_mode = str(self.km_cfg.get('feature_mode', 'xy')).lower()
        self.km_batch_size = int(self.km_cfg.get('batch_size', 4096))
        self.km_n_init = int(self.km_cfg.get('n_init', 10))
        self.km_seed = int(cfg.get('pipeline', {}).get('seed', 0))

        # OPTICS and BIRCH comparison baselines.
        self.optics_cfg = self.inst_cfg.get('optics', {})
        self.optics_min_samples = int(self.optics_cfg.get('min_samples', 20))
        self.optics_xi = float(self.optics_cfg.get('xi', 0.05))
        self.optics_min_cluster_size = int(self.optics_cfg.get('min_cluster_size', 500))
        self.optics_max_eps = self.optics_cfg.get('max_eps', None)
        if self.optics_max_eps is not None:
            self.optics_max_eps = float(self.optics_max_eps)
        self.birch_cfg = self.inst_cfg.get('birch', {})
        self.birch_threshold = float(self.birch_cfg.get('threshold', 0.05))
        self.birch_branching_factor = int(self.birch_cfg.get('branching_factor', 50))
        self.birch_n_clusters = int(self.birch_cfg.get('n_clusters', 15))




        # Graph-Based params (for comparative experiments)
        self.gb_cfg = self.inst_cfg.get('graph_based', {})
        self.gb_neighbors = int(self.gb_cfg.get('n_neighbors', 15))
        self.gb_linkage = self.gb_cfg.get('linkage', 'average')
        self.gb_dist_threshold = float(self.gb_cfg.get('distance_threshold', 0.15))

        # Enhanced PCA/Skeleton Split Params
        self.pca_cfg = self.inst_cfg.get('pca_split', {})
        self.pca_enable = bool(self.pca_cfg.get('enable', False))

        # 消融实验开关 (默认 True = 完整 GIDM)
        self.abnormal_detect = bool(self.pca_cfg.get('abnormal_detect', True))   # 异常簇检测
        self.skeleton_cut = bool(self.pca_cfg.get('skeleton_cut', True))         # 骨架切割
        self.adaptive_thresh = bool(self.pca_cfg.get('adaptive_thresh', True))   # 自适应阈值
        self.merge_back = bool(self.pca_cfg.get('merge_back', True))             # 碎片回并
        
        self.abn_d = float(self.pca_cfg.get('abnormal_diameter', 0.45))
        self.abn_ar = float(self.pca_cfg.get('abnormal_aspect_ratio', 1.5))
        
        self.skel_bins = int(self.pca_cfg.get('skeleton_bins', 15))
        self.min_peak_dist_m = float(self.pca_cfg.get('min_peak_dist_m', 0.35))
        self.valley_depth_rel = float(self.pca_cfg.get('valley_depth_rel', 0.92))
        # Crop-specific switch: use an XY-ground-plane splitter instead of the
        # original 3D PCA/skeleton splitter.  This prevents a single tall plant
        # from being cut into upper/lower fragments while still separating
        # neighbouring plants that touch in the ground-plane projection.
        self.horizontal_split_only = bool(self.pca_cfg.get('horizontal_split_only', False))

        # Optional second-pass fallback for abnormal clusters that are not split
        # by the primary min_peak_dist_m rule.
        self.dynamic_peak_cfg = self.pca_cfg.get('dynamic_min_peak_dist', {})
        self.dynamic_peak_enable = bool(self.dynamic_peak_cfg.get('enable', False))
        self.dynamic_peak_factor = float(self.dynamic_peak_cfg.get('length_factor', 0.30))
        self.dynamic_peak_min_m = float(self.dynamic_peak_cfg.get('min_m', 0.12))
        self.dynamic_peak_max_m = float(self.dynamic_peak_cfg.get('max_m', 0.20))
        self.dynamic_peak_min_child_points = int(self.dynamic_peak_cfg.get('min_child_points', 100))
        self.dynamic_peak_min_child_fraction = float(self.dynamic_peak_cfg.get('min_child_fraction', 0.01))
        self.dynamic_peak_max_children = int(self.dynamic_peak_cfg.get('max_children', 6))
        self.dynamic_peak_allow_elongated = bool(self.dynamic_peak_cfg.get('allow_elongated', True))
        self.dynamic_peak_once_per_branch = bool(self.dynamic_peak_cfg.get('once_per_branch', True))
        self.dynamic_peak_single_cut = bool(self.dynamic_peak_cfg.get('single_cut', True))
        
        # Merge Back parameters
        self.merge_min_d = float(self.pca_cfg.get('merge_min_diameter', 0.15))
        self.merge_max_dist = float(self.pca_cfg.get('merge_max_dist', 0.20))
        self.merge_max_angle = float(self.pca_cfg.get('merge_max_angle', 25.0))

        # Optional post-split repair for vertically stacked fragments that
        # belong to the same plant.  It is disabled by default and used by the
        # maize configuration after horizontal-only splitting.
        self.vertical_merge_cfg = self.pca_cfg.get('vertical_merge', {})
        self.vertical_merge_enable = bool(self.vertical_merge_cfg.get('enable', False))
        self.vertical_merge_max_xy = float(self.vertical_merge_cfg.get('max_xy_dist_m', 0.20))
        self.vertical_merge_min_z_sep = float(self.vertical_merge_cfg.get('min_z_sep_m', 0.20))
        self.vertical_merge_max_diag = float(self.vertical_merge_cfg.get('max_combined_xy_diag_m', 0.65))
        self.vertical_merge_min_overlap = float(self.vertical_merge_cfg.get('min_xy_overlap', 0.15))
        self.vertical_merge_cell_size = float(self.vertical_merge_cfg.get('cell_size_m', 0.03))

        # Fragment Voting Optimization (Neighbor Voting)
        self.vote_cfg = self.inst_cfg.get('fragment_voting', {})
        self.vote_enable = bool(self.vote_cfg.get('enable', True))
        self.vote_discard = int(self.vote_cfg.get('discard_threshold', 300))
        self.vote_merge = int(self.vote_cfg.get('merge_threshold', 1500))
        self.vote_merge_max_dist = float(self.vote_cfg.get('fragment_merge_max_dist', 0.15))
        self.vote_mode = str(self.vote_cfg.get('mode', 'centroid')).lower()
        self.vote_knn_k = int(self.vote_cfg.get('knn_k', 10))
        self.vote_fragment_length = float(self.vote_cfg.get('fragment_length_m', 0.15))
        self.vote_knn_eps = float(self.vote_cfg.get('knn_epsilon', 1e-6))
        self.vote_knn_max_dist = float(self.vote_cfg.get('knn_max_dist', 0.0))

        # Basic Filtering params
        self.min_points = self.inst_cfg.get('min_cluster_points', 50)
        self.max_points = self.inst_cfg.get('max_cluster_points', 100000)
        # Optional post-filter noise reassignment.  Disabled by default so
        # existing pipelines retain their original behaviour unless requested.
        self.noise_reassign_radius = float(self.inst_cfg.get('noise_reassign_radius', 0.0) or 0.0)
        self.noise_reassign_k = int(self.inst_cfg.get('noise_reassign_k', 5))
        self.noise_reassign_min_votes = int(self.inst_cfg.get('noise_reassign_min_votes', 3))
        # Legacy behaviour drops oversized clusters for watershed / meanshift.
        # HDBSCAN keeps its scale-aware large-cluster behaviour.  The option can
        # still be overridden explicitly in a config.
        default_drop_oversized = (self.method != 'hdbscan')
        self.drop_oversized_clusters = bool(
            self.inst_cfg.get('drop_oversized_clusters', default_drop_oversized)
        )

    def cluster(self, pcd):
        """
        Perform instance clustering on the vegetation point cloud.
        """
        self.last_exec_time = {'coarse': 0.0, 'refinement': 0.0}
        t0 = time.time()

        if len(pcd.points) == 0:
            logger.warning("No points provided for clustering.")
            return np.array([]), 0

        # 1. Base Segmentation
        if self.method == 'meanshift':
            labels, num_seeds = self._cluster_meanshift(pcd)
        elif self.method == 'graph_based':
            labels, num_seeds = self._cluster_graph_based(pcd)
        elif self.method == 'dbscan':
            labels, num_seeds = self._cluster_dbscan(pcd)
        elif self.method == 'hdbscan':
            labels, num_seeds = self._cluster_hdbscan(pcd)
        elif self.method == 'dpc':
            labels, num_seeds = self._cluster_dpc(pcd)
        elif self.method == 'meanshift_pp':
            labels, num_seeds = self._cluster_meanshift_pp(pcd)
        elif self.method == 'kmeans':
            labels, num_seeds = self._cluster_kmeans(pcd)
        elif self.method == 'optics':
            labels, num_seeds = self._cluster_optics(pcd)
        elif self.method == 'birch':
            labels, num_seeds = self._cluster_birch(pcd)
        elif self.method == 'euclidean':
            labels, num_seeds = self._cluster_euclidean(pcd)
        elif self.method == 'none':
            labels, num_seeds = np.zeros(len(pcd.points), dtype=np.int64), 0
        else:
            labels, num_seeds = self._cluster_watershed_3d(pcd)
        
        t1 = time.time()
        self.last_exec_time['coarse'] = t1 - t0

        # Watershed sometimes returns 0 for background if not careful, but my impl usually returns >=0 or -1.
        # Ensure points is numpy
        points = np.asarray(pcd.points)
        
        # 2. Enhanced Split: Skeleton-based Density Analysis
        if self.pca_enable and self.skeleton_cut and find_peaks is not None:
             labels = self._apply_skeleton_split(labels, points)
        
        # 3. Quality Control: Merge Back Over-segmented Fragments
        if self.pca_enable and self.merge_back and cKDTree is not None:
             labels = self._merge_fragments(labels, points)
        
        # New: Repair vertically split fragments before tiny-fragment cleanup.
        if self.vertical_merge_enable and cKDTree is not None:
             labels = self._merge_vertical_fragments(labels, points)

        # New: Merge tiny fragments (<3000 pts) to nearest neighbor
        if self.vote_enable and cKDTree is not None:
             labels = self._cleanup_tiny_fragments(labels, points)

        t2 = time.time()
        self.last_exec_time['refinement'] = t2 - t1

        # 4. Final Basic Filter & Renumber
        labels = self._simple_filter(labels, n_points=len(pcd.points))
        if self.noise_reassign_radius > 0:
            labels = self._reassign_noise(labels, np.asarray(pcd.points))
        final_labels, num_instances = self._renumber_labels(labels)
        
        logger.info(f"Final instances: {num_instances}")
        return final_labels, num_instances

    # -------------------------------------------------------------------------
    # Euclidean Clustering (Wrapper for Open3D's cluster_dbscan or DBSCAN)
    # -------------------------------------------------------------------------

    def _cluster_euclidean(self, pcd):
        logger.info(f"Clustering using Euclidean (tolerance={self.ec_tolerance}, min_size={self.ec_min_size})...")
        
        # Prefer Open3D implementation if available on the object
        if hasattr(pcd, 'cluster_dbscan'):
             # Open3D's cluster_dbscan is essentially Euclidean clustering
             # labels are 0-indexed, noise is -1
             labels = np.array(pcd.cluster_dbscan(eps=self.ec_tolerance, min_points=self.ec_min_size, print_progress=False))
             n_clusters = labels.max() + 1
             logger.info(f"Euclidean (Open3D) found {n_clusters} clusters")
             return labels, n_clusters
        else:
             # Fallback to sklearn
             points = np.asarray(pcd.points)
             db = DBSCAN(eps=self.ec_tolerance, min_samples=self.ec_min_size, n_jobs=-1)
             db.fit(points)
             labels = db.labels_
             
             n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
             logger.info(f"Euclidean (sklearn) found {n_clusters} clusters")
             return labels, n_clusters

    # -------------------------------------------------------------------------
    # DBSCAN Logic
    # -------------------------------------------------------------------------

    def _cluster_dbscan(self, pcd):
        logger.info(f"Clustering using DBSCAN (eps={self.db_eps}, min_samples={self.db_min_samples})...")
        points = np.asarray(pcd.points)
        
        db = DBSCAN(eps=self.db_eps, min_samples=self.db_min_samples, n_jobs=-1)
        db.fit(points)
        labels = db.labels_
        
        # DBSCAN labels noise as -1
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info(f"DBSCAN found {num_clusters} clusters")
        
        return labels, num_clusters

    # -------------------------------------------------------------------------
    # HDBSCAN Logic (不需要预设eps, 自适应密度)
    # -------------------------------------------------------------------------

    def _cluster_hdbscan(self, pcd):
        """Scale-aware HDBSCAN.

        Config values in (0, 1) for ``min_cluster_size`` / ``min_samples`` are
        fractions of the current foreground point count.  The primary selection
        method is tried first; if it yields a single dominant cluster (the usual
        whole-plot failure mode), a leaf selection fallback is evaluated and the
        candidate with more valid clusters / less dominance is kept.
        """
        points = np.asarray(pcd.points)
        n_points = len(points)
        if n_points == 0:
            return np.full(0, -1, dtype=int), 0

        def _resolve(value, default, minimum):
            if value is None:
                value = default
            if isinstance(value, float) and 0.0 < value < 1.0:
                return max(minimum, int(round(value * n_points)))
            return max(minimum, int(value))

        min_cluster_size = _resolve(self.hdb_min_cluster_size, 500, 2)
        min_samples = _resolve(self.hdb_min_samples, 10, 1)

        logger.info(
            "Clustering using HDBSCAN "
            f"(n_points={n_points}, min_cluster_size={min_cluster_size}, "
            f"min_samples={min_samples}, primary={self.hdb_selection_method})..."
        )

        try:
            from hdbscan import HDBSCAN as HDBSCANImpl
            backend = 'hdbscan'
        except ImportError:
            try:
                from sklearn.cluster import HDBSCAN as HDBSCANImpl
                backend = 'sklearn'
            except ImportError:
                logger.error("hdbscan is not installed (neither hdbscan nor sklearn HDBSCAN)")
                return np.full(n_points, -1, dtype=int), 0

        def _fit(selection_method):
            kwargs = {
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
                'cluster_selection_method': selection_method,
                'allow_single_cluster': self.hdb_allow_single_cluster,
                'metric': 'euclidean',
            }
            if backend == 'hdbscan':
                kwargs['core_dist_n_jobs'] = -1
            else:
                # scikit-learn's HDBSCAN accepts copy; the external package does not.
                kwargs['copy'] = True
            model = HDBSCANImpl(**kwargs)
            return np.asarray(model.fit_predict(points), dtype=int)

        methods = [self.hdb_selection_method]
        if self.hdb_fallback_leaf and self.hdb_selection_method != 'leaf':
            methods.append('leaf')

        best_labels = None
        best_score = None
        for selection_method in methods:
            labels = _fit(selection_method)
            valid = labels[labels >= 0]
            n_clusters = int(len(np.unique(valid)))
            if n_clusters:
                _, counts = np.unique(valid, return_counts=True)
                largest = int(counts.max())
            else:
                largest = 0
            largest_fraction = largest / max(1, n_points)
            # Prefer more separated clusters and, among equal counts, avoid a
            # single cluster consuming most of the foreground.
            score = (n_clusters, -largest_fraction)
            logger.info(
                f"HDBSCAN[{selection_method}] found {n_clusters} clusters; "
                f"largest={largest} ({largest_fraction:.1%}), "
                f"noise={int((labels < 0).sum())}"
            )
            if best_score is None or score > best_score:
                best_score = score
                best_labels = labels
            if n_clusters >= 2 and largest_fraction <= 0.75:
                break

        if best_labels is None:
            return np.full(n_points, -1, dtype=int), 0
        num_clusters = len(np.unique(best_labels[best_labels >= 0]))
        logger.info(f"HDBSCAN selected {num_clusters} clusters")
        return best_labels, num_clusters

    def _cluster_dpc(self, pcd):
        """Density Peak Clustering for spatially separated plant instances."""
        points = np.asarray(pcd.points)
        n_points = len(points)
        if n_points == 0:
            return np.full(0, -1, dtype=int), 0
        if cKDTree is None:
            return np.full(n_points, -1, dtype=int), 0

        k = max(2, min(int(self.dpc_k), n_points - 1))
        tree = cKDTree(points)
        dists, nbrs = tree.query(points, k=k + 1, workers=-1)
        dists = np.asarray(dists, dtype=np.float64)
        nbrs = np.asarray(nbrs)
        self_mask = nbrs == np.arange(n_points, dtype=nbrs.dtype)[:, None]
        if self_mask.any(axis=1).all():
            dists = np.delete(dists, 0, axis=1)
            nbrs = np.delete(nbrs, 0, axis=1)
            k -= 1
        rho = 1.0 / (dists.mean(axis=1) + 1e-6)

        order = np.argsort(rho)
        rank = np.empty(n_points, dtype=np.int64)
        rank[order] = np.arange(n_points)
        higher = rank[nbrs] > rank[:, None]
        candidate_d = np.where(higher, dists, np.inf)
        best_j = np.argmin(candidate_d, axis=1)
        best_d = candidate_d[np.arange(n_points), best_j]
        has_higher = np.isfinite(best_d)
        delta = np.where(has_higher, best_d, dists.max(axis=1))
        gamma = rho * delta

        target = int(round(n_points * max(0.0, self.dpc_center_fraction)))
        target = max(max(2, self.dpc_min_centers), min(target, self.dpc_max_centers))
        target = min(target, n_points)
        candidates = np.argsort(gamma)[::-1]
        centers = []
        for idx in candidates:
            if len(centers) >= target:
                break
            if not centers or np.min(np.linalg.norm(points[centers] - points[idx], axis=1)) >= self.dpc_min_sep:
                centers.append(int(idx))
        if len(centers) < max(2, self.dpc_min_centers):
            for idx in candidates:
                if int(idx) not in centers:
                    centers.append(int(idx))
                if len(centers) >= max(2, self.dpc_min_centers):
                    break
        if not centers:
            return np.full(n_points, -1, dtype=int), 0
        center_indices = np.asarray(centers, dtype=np.int64)
        center_tree = cKDTree(points[center_indices])
        _, nearest_center = center_tree.query(points, k=1, workers=-1)
        nearest_center = np.asarray(nearest_center, dtype=np.int64)
        labels = nearest_center.copy()

        if self.dpc_assignment == 'density_chain':
            # Standard DPC assignment: follow the nearest higher-density point
            # instead of assigning every sample to the geometrically nearest
            # centre. This keeps connected leaf surfaces in the same basin.
            labels = np.full(n_points, -1, dtype=np.int64)
            labels[center_indices] = np.arange(len(center_indices), dtype=np.int64)
            for idx in order[::-1]:
                if labels[idx] >= 0:
                    continue
                if has_higher[idx]:
                    parent = int(nbrs[idx, best_j[idx]])
                    if labels[parent] >= 0:
                        labels[idx] = labels[parent]
                        continue
                # Points without a reachable higher-density parent fall back to
                # the nearest selected density peak.
                labels[idx] = nearest_center[idx]

        logger.info(
            f"DPC selected {len(centers)} centres from {n_points} points "
            f"(k={k}, center_fraction={self.dpc_center_fraction}, "
            f"min_sep={self.dpc_min_sep:.2f} m, assignment={self.dpc_assignment})."
        )
        return labels, len(centers)

    def _cluster_meanshift_pp(self, pcd):
        """Grid-seeded mean shift using flat-kernel mode seeking."""
        points = np.asarray(pcd.points)
        n_points = len(points)
        if n_points == 0:
            return np.full(0, -1, dtype=int), 0
        if n_points == 1:
            return np.zeros(1, dtype=int), 1

        grid_size = max(float(self.mspp_grid_size), 1e-6)
        keys = np.floor((points - points.min(axis=0)) / grid_size).astype(np.int64)
        _, inverse, counts = np.unique(
            keys, axis=0, return_inverse=True, return_counts=True
        )
        sums = np.zeros((len(counts), 3), dtype=np.float64)
        np.add.at(sums, inverse, points)
        seeds = sums / counts[:, None]
        if self.mspp_max_seeds > 0 and len(seeds) > self.mspp_max_seeds:
            idx = np.linspace(0, len(seeds) - 1, self.mspp_max_seeds, dtype=np.int64)
            seeds = seeds[idx]

        model = MeanShift(
            bandwidth=self.mspp_bandwidth,
            seeds=seeds,
            bin_seeding=False,
            cluster_all=True,
            n_jobs=-1,
        )
        labels = np.asarray(model.fit_predict(points), dtype=np.int64)
        num_clusters = len(np.unique(labels))
        logger.info(
            f"MeanShift++ found {num_clusters} clusters from {n_points} points "
            f"(bandwidth={self.mspp_bandwidth:.2f}, grid={grid_size:.2f}, "
            f"seeds={len(seeds)})."
        )
        return labels, int(num_clusters)


    def _cluster_kmeans(self, pcd):
        """MiniBatch KMeans baseline with an explicit cluster count."""
        points = np.asarray(pcd.points)
        n_points = len(points)
        if n_points == 0:
            return np.full(0, -1, dtype=int), 0
        if self.km_feature_mode == 'xy':
            features = points[:, :2]
        elif self.km_feature_mode == 'xyz':
            features = points[:, :3]
        else:
            raise ValueError(f"Unsupported kmeans.feature_mode: {self.km_feature_mode}")
        n_clusters = max(2, min(int(self.km_n_clusters), n_points))
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=max(n_clusters, int(self.km_batch_size)),
            n_init=self.km_n_init,
            random_state=self.km_seed,
        )
        labels = np.asarray(model.fit_predict(features), dtype=np.int64)
        logger.info(
            f"MiniBatchKMeans found {n_clusters} clusters from {n_points} points "
            f"(features={self.km_feature_mode})."
        )
        return labels, int(n_clusters)


    def _cluster_optics(self, pcd):
        """OPTICS clustering with xi-steepness extraction."""
        points = np.asarray(pcd.points)
        n_points = len(points)
        if n_points == 0:
            return np.full(0, -1, dtype=int), 0
        model = OPTICS(
            min_samples=max(2, min(self.optics_min_samples, n_points - 1)),
            xi=self.optics_xi,
            min_cluster_size=max(2, min(self.optics_min_cluster_size, n_points)),
            max_eps=np.inf if self.optics_max_eps is None else self.optics_max_eps,
            cluster_method='xi',
            metric='euclidean',
            n_jobs=-1,
        )
        labels = np.asarray(model.fit_predict(points), dtype=np.int64)
        num_clusters = len(np.unique(labels[labels >= 0]))
        logger.info(f"OPTICS found {num_clusters} clusters from {n_points} points.")
        return labels, int(num_clusters)

    def _cluster_birch(self, pcd):
        """BIRCH hierarchical clustering with a target cluster count."""
        points = np.asarray(pcd.points)
        n_points = len(points)
        if n_points == 0:
            return np.full(0, -1, dtype=int), 0
        n_clusters = max(2, min(self.birch_n_clusters, n_points))
        model = Birch(
            threshold=self.birch_threshold,
            branching_factor=self.birch_branching_factor,
            n_clusters=n_clusters,
        )
        labels = np.asarray(model.fit_predict(points), dtype=np.int64)
        logger.info(
            f"BIRCH found {n_clusters} clusters from {n_points} points "
            f"(threshold={self.birch_threshold:.2f})."
        )
        return labels, int(n_clusters)


    # -------------------------------------------------------------------------
    # Mean Shift Logic
    # -------------------------------------------------------------------------

    def _cluster_meanshift(self, pcd):
        logger.info(f"Clustering using Mean Shift (bandwidth={self.ms_bandwidth}, bin_seeding={self.ms_bin_seeding})...")
        points = np.asarray(pcd.points)
        
        # Use sklearn MeanShift
        # If bandwidth is None, it could be estimated, but fixed is preferred for consistency/speed if tuned.
        ms = MeanShift(bandwidth=self.ms_bandwidth, bin_seeding=self.ms_bin_seeding, min_bin_freq=self.ms_min_bin_freq, n_jobs=-1)
        ms.fit(points)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        
        num_clusters = len(np.unique(labels))
        logger.info(f"Mean Shift found {num_clusters} clusters")
        return labels, num_clusters

    # -------------------------------------------------------------------------
    # Graph-Based Region Merging Logic
    # -------------------------------------------------------------------------

    def _cluster_graph_based(self, pcd):
        logger.info(f"Clustering using Graph-Based Merging (linkage={self.gb_linkage}, threshold={self.gb_dist_threshold}, k={self.gb_neighbors})...")
        points = np.asarray(pcd.points)
        n_points = len(points)
        final_labels = np.full(n_points, -1, dtype=int)

        # 1. Build Connectivity (KNN Graph)
        connectivity = kneighbors_graph(points, n_neighbors=self.gb_neighbors, include_self=False)

        # 2. Decompose into connected components to avoid O(N^2) memory usage in AgglomerativeClustering
        # when the graph is not fully connected. Sklearn attempts to "fix" connectivity by computing
        # all-pairs distances if the graph is disconnected, which causes MemoryError on large clouds.
        from scipy.sparse.csgraph import connected_components
        n_components, component_labels = connected_components(connectivity, directed=False)
        
        logger.info(f"  - Graph has {n_components} connected components. Processing independently...")
        
        current_max_label = 0
        
        for i in range(n_components):
            mask = (component_labels == i)
            # count = np.sum(mask) # Optimization: len(item) is faster
            sub_points = points[mask]
            
            # If component is very small, treat as one cluster
            if len(sub_points) < 5:
                final_labels[mask] = current_max_label
                current_max_label += 1
                continue
            
            # Extract sub-graph
            sub_connectivity = connectivity[mask][:, mask]

            # 3. Agglomerative Clustering on Sub-Component
            try:
                model = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=self.gb_dist_threshold,
                    linkage=self.gb_linkage,
                    connectivity=sub_connectivity,
                    metric='euclidean'  # New sklearn
                )
                model.fit(sub_points)
            except TypeError:
                # Fallback for older sklearn (affinity instead of metric)
                model = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=self.gb_dist_threshold,
                    linkage=self.gb_linkage,
                    connectivity=sub_connectivity,
                    affinity='euclidean' 
                )
                model.fit(sub_points)

            # Assign global labels
            final_labels[mask] = model.labels_ + current_max_label
            current_max_label += model.n_clusters_
            
            # Debug huge over-segmentation
            if i < 5 and model.n_clusters_ > 10:
                logger.info(f"    [Debug] Component {i} (pts={len(sub_points)}) split into {model.n_clusters_} clusters. "
                            f"Avg dist within connectivity graph might be > {self.gb_dist_threshold}.")

        logger.info(f"Graph Merging found {current_max_label} clusters (from {n_components} connected components). "
                    f"Params: K={self.gb_neighbors}, Thresh={self.gb_dist_threshold:.2f}, Link={self.gb_linkage}")
        return final_labels, current_max_label

    # -------------------------------------------------------------------------
    # Watershed Only Logic
    # -------------------------------------------------------------------------

    def _cluster_watershed_3d(self, pcd):
        logger.info(f"Clustering using 3D EDT Watershed (res={self.ws3d_res})...")
        points = np.asarray(pcd.points)
        labels, num_seeds = self._compute_watershed3d_labels(points, self.ws3d_min_seed_dist)

        if labels is None:
            return np.full(len(points), -1), 0
        
        # Initial cleanup
        final_labels = labels - 1
        return final_labels, num_seeds

    def _compute_watershed3d_labels(self, points, min_seed_dist):
        if distance_transform_edt is None:
            logger.error("scipy/skimage missing.")
            return None, 0

        min_xyz = points.min(axis=0)
        max_xyz = points.max(axis=0)
        
        # Adaptive resolution
        voxel_res = float(self.ws3d_res)
        while True:
            dims = np.ceil((max_xyz - min_xyz) / voxel_res).astype(int) + 1
            total = np.prod(dims)
            if total <= self.ws3d_max_voxels: break
            voxel_res *= 1.1

        idx = ((points - min_xyz) / voxel_res).astype(int)
        idx = np.clip(idx, 0, dims - 1)

        occ = np.zeros(dims, dtype=bool)
        occ[idx[:,0], idx[:,1], idx[:,2]] = True

        if self.ws3d_dilation_iters > 0:
            occ = binary_dilation(occ, iterations=self.ws3d_dilation_iters)

        edt = distance_transform_edt(occ)
        if self.ws3d_sigma > 0:
            edt = gaussian_filter(edt, sigma=self.ws3d_sigma)

        min_dist_vox = max(1, int(min_seed_dist / voxel_res))
        
        # Peak threshold logic
        if self.ws3d_peak_threshold_abs_m:
             abs_thresh = self.ws3d_peak_threshold_abs_m / voxel_res
        else:
             abs_thresh = 0
             
        peaks = peak_local_max(edt, min_distance=min_dist_vox, threshold_abs=abs_thresh, labels=occ.astype(int), exclude_border=False)
        
        if len(peaks) == 0:
            return np.zeros(len(points), dtype=int), 0

        markers = np.zeros_like(edt, dtype=np.int32)
        markers[tuple(peaks.T)] = np.arange(1, len(peaks)+1, dtype=np.int32)
        
        labels_grid = watershed(-edt, markers, mask=occ)
        labels = labels_grid[idx[:,0], idx[:,1], idx[:,2]]
        
        return labels, int(len(peaks))

    # -------------------------------------------------------------------------
    # Improved Structure-Aware Splitting (Local PCA / Skeleton)
    # -------------------------------------------------------------------------

    def _apply_skeleton_split(self, labels, points):
        """Iteratively split clusters using 'Spine' density analysis until convergence."""
        current_labels = labels.copy()
        max_label = current_labels.max() if len(current_labels) > 0 else 0
        total_splits = 0
        dynamic_locked = set()
        
        for iteration in range(5):  # safety: max 5 iterations
            unique_labels = np.unique(current_labels)
            unique_labels = unique_labels[unique_labels != -1]
            new_labels = current_labels.copy()
            next_label_id = max_label + 1
            any_split = False
            
            for lbl in unique_labels:
                mask = (current_labels == lbl)
                lbl_points = points[mask]
                
                if len(lbl_points) < 50: continue

                if self.horizontal_split_only:
                    feat = self._compute_horizontal_features(lbl_points)
                    mins = lbl_points[:, :2].min(axis=0)
                    maxs = lbl_points[:, :2].max(axis=0)
                else:
                    feat = self._compute_geometric_features(lbl_points)
                    mins = lbl_points.min(axis=0)
                    maxs = lbl_points.max(axis=0)
                dims = maxs - mins
                max_axis = dims.max()
                
                is_huge = (feat['L1'] > self.abn_d) or (max_axis > self.abn_d)
                is_elongated = (feat['r'] > self.abn_ar) and (feat['L1'] > 0.25)

                # 消融 -GD: 关闭异常检测时, 所有簇(除极小簇)都进入骨架切割
                if not self.abnormal_detect:
                    is_abnormal = True
                else:
                    is_abnormal = is_huge or is_elongated
                
                if is_abnormal:
                    indices = np.where(mask)[0]
                    if self.horizontal_split_only:
                        sub_labels_list = self._recursive_horizontal_split(
                            points, indices, depth=0, min_peak_dist_m=self.min_peak_dist_m)
                    else:
                        sub_labels_list = self._recursive_skeleton_split(
                            points, indices, depth=0, min_peak_dist_m=self.min_peak_dist_m)

                    fallback_used = False
                    if (not self.horizontal_split_only and len(sub_labels_list) == 1 and self.dynamic_peak_enable
                            and (is_huge or (self.dynamic_peak_allow_elongated and is_elongated))
                            and (not self.dynamic_peak_once_per_branch or lbl not in dynamic_locked)):
                        dynamic_dist = self._dynamic_peak_distance(feat['L1'])
                        if dynamic_dist < self.min_peak_dist_m - 1e-9:
                            candidate = self._recursive_skeleton_split(
                                points, indices, depth=0, min_peak_dist_m=dynamic_dist)
                            if self._valid_dynamic_split(candidate, len(indices)):
                                sub_labels_list = candidate
                                fallback_used = True
                                if self.dynamic_peak_once_per_branch:
                                    dynamic_locked.add(int(lbl))

                    if len(sub_labels_list) > 1:
                        mode = 'dynamic fallback' if fallback_used else 'primary rule'
                        logger.info(
                            f"Cluster {lbl} (L1={feat['L1']:.2f}, r={feat['r']:.2f}) "
                            f"split into {len(sub_labels_list)} parts via {mode} (iter {iteration}).")
                        any_split = True
                        
                        for k, sub_idx in enumerate(sub_labels_list):
                            if k == 0:
                                new_labels[sub_idx] = lbl
                            else:
                                new_labels[sub_idx] = next_label_id
                                if fallback_used and self.dynamic_peak_once_per_branch:
                                    dynamic_locked.add(int(next_label_id))
                                next_label_id += 1
            
            total_splits += (1 if any_split else 0)
            current_labels = new_labels
            max_label = next_label_id - 1
            
            if not any_split:
                break
                    
        if total_splits > 0:
            logger.info(f"Structure Split completed after {total_splits} iteration(s).")
        return current_labels

    def _dynamic_peak_distance(self, length):
        """Return a bounded local peak spacing for second-pass splitting."""
        upper = min(self.dynamic_peak_max_m, self.min_peak_dist_m)
        lower = min(self.dynamic_peak_min_m, upper)
        return float(np.clip(self.dynamic_peak_factor * length, lower, upper))

    def _valid_dynamic_split(self, split_indices, parent_points):
        """Reject fallback splits that create too many or excessively small parts."""
        if len(split_indices) < 2:
            return False
        if len(split_indices) > self.dynamic_peak_max_children:
            return False
        min_required = max(
            self.dynamic_peak_min_child_points,
            int(np.ceil(self.dynamic_peak_min_child_fraction * parent_points)),
        )
        return min(len(part) for part in split_indices) >= min_required

    def _compute_horizontal_features(self, points):
        """Return PCA-like features in the XY ground plane.

        The returned ``v1`` is guaranteed to be horizontal, so a cut plane
        derived from it is a vertical wall through the ground-plane extent.
        """
        pts = np.asarray(points)
        xy = pts[:, :3].copy()
        z = xy[:, 2].copy()
        xy[:, 2] = 0.0
        feat = self._compute_geometric_features(xy)
        feat['center'] = pts[:, :3].mean(axis=0)
        feat['z_extent'] = float(np.percentile(z, 99) - np.percentile(z, 1)) if len(z) else 0.0
        return feat

    def _recursive_horizontal_split(self, all_points, indices, depth, min_peak_dist_m=None):
        """Recursively split a cluster with a horizontal 1-D density profile.

        This is the maize-safe counterpart of ``_recursive_skeleton_split``:
        peaks are searched on the XY projection, and the cut normal is forced
        to lie in the XY plane.  Therefore it separates touching neighbours
        laterally but never cuts a single plant into top and bottom pieces.
        """
        if len(indices) < 50 or depth >= 3:
            return [indices]
        if min_peak_dist_m is None:
            min_peak_dist_m = self.min_peak_dist_m

        subset_pts = all_points[indices]
        xy = subset_pts[:, :2]
        center_xy = xy.mean(axis=0)
        centered = xy - center_xy
        cov = np.cov(centered.T)
        if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
            return [indices]
        evals, evecs = np.linalg.eigh(cov)
        v1 = evecs[:, int(np.argmax(evals))]
        v1 = np.asarray(v1, dtype=float)
        # Deterministic orientation.
        if v1[0] < 0 or (abs(v1[0]) < 1e-12 and v1[1] < 0):
            v1 = -v1
        norm = np.linalg.norm(v1)
        if norm < 1e-12:
            return [indices]
        v1 /= norm

        scalars = centered @ v1
        length = float(scalars.max() - scalars.min())
        # A single maize plant has a small XY footprint.  Do not split unless
        # the horizontal extent is at least the configured peak spacing.
        if length < max(float(min_peak_dist_m), 0.20):
            return [indices]

        nbins = self.skel_bins
        if length / nbins < 0.02:
            nbins = max(5, int(length / 0.02))
        counts, edges = np.histogram(scalars, bins=nbins)
        density_smooth = gaussian_filter(counts.astype(float), sigma=0.5)
        bin_width = length / nbins
        min_dist_bins = max(1, int(round(min_peak_dist_m / bin_width)))
        peaks, _ = find_peaks(
            density_smooth,
            distance=min_dist_bins,
            height=np.max(density_smooth) * 0.10,
        )
        if len(peaks) < 2:
            return [indices]

        first_peak = int(peaks[0])
        last_peak = int(peaks[-1])
        valley_idx = first_peak + int(np.argmin(density_smooth[first_peak:last_peak + 1]))
        valley_h = float(density_smooth[valley_idx])
        peak_h = float(min(density_smooth[first_peak], density_smooth[last_peak]))
        ratio = valley_h / (peak_h + 1e-6)
        valley_thresh = self.valley_depth_rel
        if self.adaptive_thresh and length > 0.60:
            valley_thresh = max(valley_thresh, 0.98)
        if ratio >= valley_thresh:
            return [indices]

        cut_scalar = 0.5 * (edges[valley_idx] + edges[valley_idx + 1])
        cut_center_xy = center_xy + cut_scalar * v1
        signed = (xy - cut_center_xy) @ v1
        left = signed < 0
        idx1 = indices[left]
        idx2 = indices[~left]
        if len(idx1) < 10 or len(idx2) < 10:
            return [indices]
        return (
            self._recursive_horizontal_split(all_points, idx1, depth + 1, min_peak_dist_m)
            + self._recursive_horizontal_split(all_points, idx2, depth + 1, min_peak_dist_m)
        )

    def _recursive_skeleton_split(self, all_points, indices, depth, min_peak_dist_m=None):
        if len(indices) < 50 or depth >= 3: 
            return [indices]

        if min_peak_dist_m is None:
            min_peak_dist_m = self.min_peak_dist_m

        subset_pts = all_points[indices]
        feat = self._compute_geometric_features(subset_pts)
        
        # Check if still abnormal enough to warrant split check even at depth>0
        # Relax constraints slightly for children?
        if feat['L1'] < 0.20: # Start getting too small
            return [indices]

        # 3. Backbone/Skeleton Analysis
        # Slice points along major axis
        v1 = feat['v1']
        center = feat['center']
        
        # Project to 1D
        scalars = np.dot(subset_pts - center, v1)
        s_min, s_max = scalars.min(), scalars.max()
        length = s_max - s_min
        
        if length < min_peak_dist_m: 
             return [indices]
             
        # Create bins
        nbins = self.skel_bins
        # Ensure bin size isn't too small (e.g. < 2cm)
        if length / nbins < 0.02:
            nbins = max(5, int(length / 0.02))
            
        # compute histograms or slice centroids
        # We need "Density" and "Slice Centroids"
        # Bins:
        bins = np.linspace(s_min, s_max, nbins + 1)
        bin_indices = np.digitize(scalars, bins) - 1 # 0 to nbins-1
        
        # Compute slice stats
        slice_densities = []
        slice_centers = []
        valid_bins = []
        
        for b in range(nbins):
            mask_b = (bin_indices == b)
            count = np.sum(mask_b)
            if count > 0:
                # Centroid of this slice - this approximates the "Curve" or "Skeleton"
                slice_center = subset_pts[mask_b].mean(axis=0)
                slice_densities.append(count)
                slice_centers.append(slice_center)
                valid_bins.append(b)
            else:
                # Interpolate gap or skip? 
                # For density, gap is density=0 (strong valley)
                # But typically we just skip for now or fill with 0
                slice_densities.append(0)
                slice_centers.append(None) # Can't compute
                
        # Analyze Density Profile for Valleys (Necks)
        density_arr = np.array(slice_densities, dtype=float)
        # Smooth density
        density_smooth = gaussian_filter(density_arr, sigma=0.5)
        
        # Find Peaks
        # bin_width
        bin_width = length / nbins
        min_dist_bins = max(1, int(min_peak_dist_m / bin_width))
        
        peaks, _ = find_peaks(density_smooth, distance=min_dist_bins, height=np.max(density_smooth)*0.1)
        
        if len(peaks) >= 2:
            # Find deepest valley between peaks
            # We want the valley index in 'bins'
             
            # Consider only the two strongest peaks for simplicity, or iteratively cut?
            # Let's simple split at strongest valley between *any* two significant peaks
            
            # Simple approach: Find global min between first and last peak
            first_peak = peaks[0]
            last_peak = peaks[-1]
            if first_peak >= last_peak: 
                 return [indices]
                 
            # Search for valley in smoothed signal
            segment = density_smooth[first_peak:last_peak+1]
            local_min_idx = np.argmin(segment)
            valley_idx = first_peak + local_min_idx
            
            # Check Prominence
            valley_h = density_smooth[valley_idx]
            peak_h = min(density_smooth[first_peak], density_smooth[last_peak]) # Conservative: compare to boundary peaks
            
            ratio = valley_h / (peak_h + 1e-6)
            
            # Adaptive Threshold:
            # If the object is HUGE (> 0.6m), it is almost certainly multiple plants.
            # In this case, we relax the valley depth requirement (allow shallower valleys e.g. 0.98).
            # Otherwise, we use the strict config value.
            
            current_valley_thresh = self.valley_depth_rel
            if self.adaptive_thresh and length > 0.60:
                # Relax threshold for huge clusters
                current_valley_thresh = max(current_valley_thresh, 0.98) 
                
            logger.debug(f"Skeleton Check: L={length:.2f}m Peaks:{len(peaks)} ValleyRatio:{ratio:.2f} Thresh:{current_valley_thresh}")

            if ratio < current_valley_thresh:
                # CONFIRMED CUT
                # 4. Perform Cut using Skeleton geometry
                # Ideally cut plane should be normal to the curve at valley_idx
                # curve point: slice_centers[valley_idx]
                
                # If slice_centers[valley_idx] is None (gap), that's an even better cut!
                # But let's handle the case where we have a point.
                
                cut_center = None
                cut_normal = None
                
                # Check neighbors for normal
                c_curr = slice_centers[valley_idx]
                
                # If this bin is empty, finding the 'scalar' split value is enough
                # split_scalar = (bins[valley_idx] + bins[valley_idx+1]) / 2.0
                split_scalar = bins[valley_idx] + (bins[1]-bins[0])*0.5
                
                # But if we want a curved cut, we need the normal.
                # v1 is the global normal. Local normal is better.
                # Try to get local tangent: C[v+1] - C[v-1]
                prev_c = self._find_nearest_valid(slice_centers, valley_idx, -1)
                next_c = self._find_nearest_valid(slice_centers, valley_idx, 1)
                
                if prev_c is not None and next_c is not None:
                     cut_normal = next_c - prev_c
                     cut_normal /= (np.linalg.norm(cut_normal) + 1e-6)
                     if c_curr is not None:
                         cut_center = c_curr
                     else:
                         cut_center = (prev_c + next_c) * 0.5
                else:
                     # Fallback to global v1
                     cut_normal = v1
                     cut_center = center + split_scalar * v1 # Approx
                     
                if cut_center is None: cut_center = center
                
                # Execute Cut: Dot product with local normal
                # (p - cut_center) . cut_normal
                
                dists = np.dot(subset_pts - cut_center, cut_normal)
                mask_left = dists < 0
                mask_right = ~mask_left
                
                idx1 = indices[mask_left]
                idx2 = indices[mask_right]
                
                if len(idx1) > 10 and len(idx2) > 10:
                    # A dynamic fallback performs one corrective cut only;
                    # subsequent iterations return to the primary rule.
                    if (self.dynamic_peak_single_cut
                            and min_peak_dist_m < self.min_peak_dist_m - 1e-9):
                        return [idx1, idx2]

                    return (self._recursive_skeleton_split(
                                all_points, idx1, depth+1, min_peak_dist_m) +
                            self._recursive_skeleton_split(
                                all_points, idx2, depth+1, min_peak_dist_m))
                            
        return [indices]

    def _find_nearest_valid(self, lst, idx, direction):
        curr = idx + direction
        while 0 <= curr < len(lst):
            if lst[curr] is not None:
                return lst[curr]
            curr += direction
        return None

    def _compute_geometric_features(self, points):
        if len(points) < 4:
            return {'v1': np.array([1,0,0]), 'center': np.mean(points,0), 'L1':0, 'r':1, 'f':1}
            
        center = points.mean(axis=0)
        centered = points - center
        cov = np.cov(centered.T)
        evals, evecs = np.linalg.eigh(cov)
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]
        
        # Real extent
        proj = np.dot(centered, evecs)
        mins = proj.min(axis=0)
        maxs = proj.max(axis=0)
        extents = maxs - mins
        
        return {
            'L1': extents[0], # Major
            'L2': extents[1],
            'L3': extents[2], # Height approx
            'r': extents[0]/extents[1] if extents[1]>0 else 1,
            'v1': evecs[:, 0],
            'center': center
        }

    def _cleanup_tiny_fragments(self, labels, points):
        """
        Discard noise (< discard_threshold) and merge small fragments (< merge_threshold) to nearest large cluster.
        """
        unique_labels, counts = np.unique(labels, return_counts=True)
        valid_mask = (unique_labels != -1)
        unique_labels = unique_labels[valid_mask]
        counts = counts[valid_mask]
        
        if len(unique_labels) == 0:
            return labels

        new_labels = labels.copy()

        # Step A: Discard tiny noise (< discard_threshold)
        discard_labels = unique_labels[counts < self.vote_discard]
        for dl in discard_labels:
            new_labels[new_labels == dl] = -1

        # Recompute after discard
        unique_labels, counts = np.unique(new_labels, return_counts=True)
        valid_mask = (unique_labels != -1)
        unique_labels = unique_labels[valid_mask]
        counts = counts[valid_mask]

        if len(unique_labels) < 2:
            return new_labels

        # Step B: Identify fragments to merge (< merge_threshold)
        is_tiny = counts < self.vote_merge
        tiny_labels = unique_labels[is_tiny]
        target_labels = unique_labels[~is_tiny]
        
        if len(tiny_labels) == 0:
            return new_labels
            
        if len(target_labels) == 0:
            return new_labels
             
        target_mask = np.isin(new_labels, target_labels)
        target_points = points[target_mask]
        target_point_labels = new_labels[target_mask]
        
        if len(target_points) == 0:
            return new_labels
        
        # 用簇质心算距离
        centroids = []
        centroid_lbls = []
        for t_lbl in target_labels:
            centroids.append(points[new_labels == t_lbl].mean(0))
            centroid_lbls.append(t_lbl)
        centroid_pts = np.array(centroids)
        centroid_lbls = np.array(centroid_lbls)
        
        merged_count = 0
        
        for t_lbl in tiny_labels:
            mask = (new_labels == t_lbl)
            frag_center = points[mask].mean(0)
            dists = np.linalg.norm(centroid_pts - frag_center, axis=1)
            nearest = int(np.argmin(dists))
            # 距离上限: 碎片质心距最近大簇质心过远则丢弃 (防远噪声点被硬塞进植株)
            if dists[nearest] <= self.vote_merge_max_dist:
                new_labels[mask] = centroid_lbls[nearest]
                merged_count += 1
                
        if merged_count > 0:
            logger.info(f"Merged {merged_count} tiny fragments (<{self.vote_merge} pts) and discarded {len(discard_labels)} noise (<{self.vote_discard} pts).")
            
        return new_labels

    def _reassign_fragments_knn(self, labels, points):
        """Reassign fragments using distance-weighted k-NN voting."""
        if cKDTree is None:
            return labels

        unique_labels, counts = np.unique(labels, return_counts=True)
        valid = unique_labels >= 0
        unique_labels = unique_labels[valid]
        counts = counts[valid]
        if len(unique_labels) < 2:
            return labels

        candidate_labels = []
        target_labels = []
        length_thresh = max(0.0, float(self.vote_fragment_length))
        for lbl, count in zip(unique_labels, counts):
            mask = labels == lbl
            if count < self.vote_merge:
                candidate_labels.append(lbl)
                continue
            feat = self._compute_geometric_features(points[mask])
            major_length = float(feat['L1']) if feat is not None else 0.0
            if major_length < length_thresh:
                candidate_labels.append(lbl)
            else:
                target_labels.append(lbl)

        if not candidate_labels or not target_labels:
            return labels

        candidate_mask = np.isin(labels, candidate_labels)
        target_mask = np.isin(labels, target_labels)
        candidate_points = points[candidate_mask]
        target_points = points[target_mask]
        target_point_labels = labels[target_mask]
        if len(candidate_points) == 0 or len(target_points) == 0:
            return labels

        k = max(1, min(int(self.vote_knn_k), len(target_points)))
        tree = cKDTree(target_points)
        if self.vote_knn_max_dist > 0:
            dists, nbrs = tree.query(
                candidate_points, k=k,
                distance_upper_bound=float(self.vote_knn_max_dist),
            )
        else:
            dists, nbrs = tree.query(candidate_points, k=k)
        if k == 1:
            dists = np.asarray(dists)[:, None]
            nbrs = np.asarray(nbrs)[:, None]
        else:
            dists = np.asarray(dists)
            nbrs = np.asarray(nbrs)

        valid_nbr = np.isfinite(dists)
        weights = np.zeros_like(dists, dtype=np.float64)
        weights[valid_nbr] = 1.0 / (dists[valid_nbr] + self.vote_knn_eps)
        neighbor_labels = np.full(nbrs.shape, -1, dtype=labels.dtype)
        if valid_nbr.any():
            neighbor_labels[valid_nbr] = target_point_labels[nbrs[valid_nbr]]

        target_labels_arr = np.asarray(target_labels, dtype=labels.dtype)
        label_to_col = {int(lbl): i for i, lbl in enumerate(target_labels_arr)}
        scores = np.zeros((len(candidate_points), len(target_labels_arr)), dtype=np.float64)
        rows = np.arange(len(candidate_points))
        for j in range(k):
            labels_j = neighbor_labels[:, j]
            valid_j = labels_j >= 0
            if not valid_j.any():
                continue
            cols = np.fromiter(
                (label_to_col[int(lbl)] for lbl in labels_j[valid_j]),
                dtype=np.int64,
                count=int(valid_j.sum()),
            )
            np.add.at(scores, (rows[valid_j], cols), weights[valid_j, j])

        best_col = scores.argmax(axis=1)
        best_score = scores[rows, best_col]
        assigned = best_score > 0
        out = labels.copy()
        candidate_idx = np.flatnonzero(candidate_mask)
        out[candidate_idx[assigned]] = target_labels_arr[best_col[assigned]]
        logger.info(
            f"kNN-R reassigned {int(assigned.sum())}/{len(candidate_points)} points "
            f"from {len(candidate_labels)} fragments to {len(target_labels)} complete instances "
            f"(k={k}, theta_L={length_thresh:.3f} m)."
        )
        return out


    # -------------------------------------------------------------------------
    # Merge Back Logic
    # -------------------------------------------------------------------------
    def _merge_vertical_fragments(self, labels, points):
        """Merge same-plant fragments separated mainly along the Z axis.

        Unlike ``_merge_fragments`` this test is deliberately conservative:
        fragments must be close in XY, separated in Z, overlap in their XY
        occupancy grids, and remain inside a plausible single-plant footprint.
        This repairs PointGroup-native vertical fragmentation without merging
        neighbouring maize plants that only happen to have close centroids.
        """
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels >= 0]
        if len(unique_labels) < 2:
            return labels

        stats = {}
        cell = max(float(self.vertical_merge_cell_size), 1e-6)
        for lbl in unique_labels:
            mask = labels == lbl
            pts = points[mask]
            if len(pts) < 10:
                continue
            xy = pts[:, :2]
            cells = set(map(tuple, np.floor(xy / cell).astype(np.int64)))
            stats[int(lbl)] = {
                'center_xy': xy.mean(axis=0),
                'center_z': float(pts[:, 2].mean()),
                'cells': cells,
                'bbox_min': xy.min(axis=0),
                'bbox_max': xy.max(axis=0),
            }

        if len(stats) < 2:
            return labels

        parent = {lbl: lbl for lbl in stats}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        keys = list(stats)
        for i, a in enumerate(keys):
            sa = stats[a]
            for b in keys[i + 1:]:
                sb = stats[b]
                d_xy = float(np.linalg.norm(sa['center_xy'] - sb['center_xy']))
                if d_xy > self.vertical_merge_max_xy:
                    continue
                d_z = abs(sa['center_z'] - sb['center_z'])
                if d_z < self.vertical_merge_min_z_sep:
                    continue
                denom = min(len(sa['cells']), len(sb['cells']))
                overlap = len(sa['cells'].intersection(sb['cells'])) / denom if denom else 0.0
                if overlap < self.vertical_merge_min_overlap:
                    continue
                lo = np.minimum(sa['bbox_min'], sb['bbox_min'])
                hi = np.maximum(sa['bbox_max'], sb['bbox_max'])
                if float(np.linalg.norm(hi - lo)) > self.vertical_merge_max_diag:
                    continue
                union(a, b)

        merged = labels.copy()
        for lbl in keys:
            root = find(lbl)
            if root != lbl:
                merged[labels == lbl] = root
        return merged

    def _merge_fragments(self, labels, points):
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]
        
        if len(unique_labels) < 2: return labels
        
        stats = {}
        centroids = []
        valid_lbls = []
        
        for lbl in unique_labels:
            mask = (labels==lbl)
            pts = points[mask]
            feat = self._compute_geometric_features(pts)
            stats[lbl] = {'feat':feat, 'count':len(pts)}
            centroids.append(feat['center'])
            valid_lbls.append(lbl)
            
        tree = cKDTree(centroids)
        parent = {l:l for l in valid_lbls}
        def find(x):
            if parent[x]!=x: parent[x]=find(parent[x])
            return parent[x]
        def union(a,b):
            ra,rb = find(a),find(b)
            if ra!=rb: parent[rb]=ra
            
        sorted_lbls = sorted(valid_lbls, key=lambda l: stats[l]['count'])

        for lbl in sorted_lbls:
            s_feat = stats[lbl]['feat']
            # Fragment condition
            if s_feat['L1'] > self.merge_min_d:
                continue 
                
            nbrs = tree.query_ball_point(s_feat['center'], r=self.merge_max_dist)
            for j in nbrs:
                tl = valid_lbls[j]
                if tl == lbl: continue
                # Angle check
                v1s = s_feat['v1']
                v1t = stats[tl]['feat']['v1']
                dp = abs(np.dot(v1s, v1t))
                angle = np.degrees(np.arccos(min(dp, 1.0)))
                
                if angle < self.merge_max_angle:
                    union(tl, lbl)
                    break
        
        new_labels = labels.copy()
        for lbl in valid_lbls:
            root = find(lbl)
            if root != lbl:
                new_labels[labels==lbl] = root
        return new_labels

    def _reassign_noise(self, labels, points):
        """Reassign -1 points to nearby valid instances by k-NN voting.

        A noise point is assigned only when at least
        ``noise_reassign_min_votes`` among its ``noise_reassign_k`` nearest
        assigned points lie within ``noise_reassign_radius`` and share the same
        label.  Isolated noise points remain labelled -1.
        """
        if cKDTree is None or mode is None:
            return labels
        noise_idx = np.flatnonzero(labels == -1)
        assigned_idx = np.flatnonzero(labels >= 0)
        if len(noise_idx) == 0 or len(assigned_idx) == 0:
            return labels

        k = max(1, min(int(self.noise_reassign_k), len(assigned_idx)))
        radius = float(self.noise_reassign_radius)
        tree = cKDTree(points[assigned_idx])
        dist, nbr = tree.query(points[noise_idx], k=k, distance_upper_bound=radius)
        if k == 1:
            dist = np.asarray(dist)[:, None]
            nbr = np.asarray(nbr)[:, None]
        else:
            dist = np.asarray(dist)
            nbr = np.asarray(nbr)

        valid = np.isfinite(dist) & (dist <= radius)
        neigh = np.full(dist.shape, -1, dtype=labels.dtype)
        if valid.any():
            neigh[valid] = labels[assigned_idx][nbr[valid]]

        min_votes = max(1, min(int(self.noise_reassign_min_votes), k))
        modes, counts = mode(neigh, axis=1, keepdims=False)
        assign = (counts >= min_votes) & (modes >= 0)
        out = labels.copy()
        out[noise_idx[assign]] = modes[assign]
        logger.info(
            f"Noise reassignment: {int(assign.sum())}/{len(noise_idx)} points "
            f"assigned (radius={radius} m, k={k}, min_votes={min_votes})."
        )
        return out

    def _simple_filter(self, labels, n_points=None):
        """Remove clusters outside the configured size range.

        For HDBSCAN, ``min_cluster_points`` may be a fraction in (0, 1), and
        ``max_cluster_fraction`` (under ``instance.hdbscan``) scales the upper
        bound with the current foreground size.  The upper bound is therefore
        never smaller than the configured absolute max.
        """
        counts = Counter(labels)
        new_labels = labels.copy()

        min_points = self.min_points
        if (self.method == 'hdbscan' and n_points is not None
                and isinstance(min_points, float) and 0.0 < min_points < 1.0):
            min_points = max(1, int(round(min_points * n_points)))

        max_points = self.max_points
        if (self.method == 'hdbscan' and n_points is not None
                and self.hdb_max_cluster_fraction is not None):
            frac = float(self.hdb_max_cluster_fraction)
            if 0.0 < frac < 1.0:
                scaled_max = max(1, int(round(frac * n_points)))
                max_points = scaled_max if max_points is None else max(int(max_points), scaled_max)

        large_kept = 0
        for lbl, count in counts.items():
            if lbl == -1:
                continue
            too_small = count < int(min_points)
            too_large = (max_points is not None and int(max_points) > 0 and count > int(max_points))
            if too_small:
                new_labels[labels == lbl] = -1
            elif too_large:
                if self.drop_oversized_clusters:
                    new_labels[labels == lbl] = -1
                else:
                    large_kept += 1
        if large_kept:
            logger.info(f"Kept {large_kept} oversized clusters; max_cluster_points is not used as a deletion threshold.")
        return new_labels

    def _renumber_labels(self, labels):
        final_labels = np.full_like(labels, -1)
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]
        for i, lbl in enumerate(unique_labels):
            final_labels[labels == lbl] = i
        return final_labels, len(unique_labels)
