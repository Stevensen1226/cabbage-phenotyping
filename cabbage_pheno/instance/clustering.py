import numpy as np
import logging
import time
from collections import Counter
from sklearn.cluster import MeanShift, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import kneighbors_graph

try:
    from scipy.ndimage import distance_transform_edt, gaussian_filter, binary_dilation
    from scipy.signal import find_peaks
    from scipy.spatial import cKDTree
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
except ImportError:
    distance_transform_edt = None
    gaussian_filter = None
    binary_dilation = None
    find_peaks = None
    cKDTree = None
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
        self.hdb_min_cluster_size = int(self.hdb_cfg.get('min_cluster_size', 500))
        self.hdb_min_samples = int(self.hdb_cfg.get('min_samples', 10))

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

        # Fragment Voting Optimization (Neighbor Voting)
        self.vote_cfg = self.inst_cfg.get('fragment_voting', {})
        self.vote_enable = bool(self.vote_cfg.get('enable', True))
        self.vote_discard = int(self.vote_cfg.get('discard_threshold', 300))
        self.vote_merge = int(self.vote_cfg.get('merge_threshold', 1500))
        self.vote_merge_max_dist = float(self.vote_cfg.get('fragment_merge_max_dist', 0.15))

        # Basic Filtering params
        self.min_points = self.inst_cfg.get('min_cluster_points', 50)
        self.max_points = self.inst_cfg.get('max_cluster_points', 100000)

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
        
        # New: Merge tiny fragments (<3000 pts) to nearest neighbor
        if self.vote_enable and cKDTree is not None:
             labels = self._cleanup_tiny_fragments(labels, points)

        t2 = time.time()
        self.last_exec_time['refinement'] = t2 - t1

        # 4. Final Basic Filter & Renumber
        labels = self._simple_filter(labels)
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
        logger.info(f"Clustering using HDBSCAN (min_cluster_size={self.hdb_min_cluster_size}, min_samples={self.hdb_min_samples})...")
        points = np.asarray(pcd.points)
        
        try:
            from hdbscan import HDBSCAN
        except ImportError:
            logger.error("hdbscan not installed. Run: pip install hdbscan")
            return np.full(len(points), -1, dtype=int), 0
        
        clusterer = HDBSCAN(
            min_cluster_size=self.hdb_min_cluster_size,
            min_samples=self.hdb_min_samples,
            cluster_selection_method='eom',
            metric='euclidean',
            core_dist_n_jobs=-1)
        labels = clusterer.fit_predict(points)
        
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info(f"HDBSCAN found {num_clusters} clusters")
        return labels, num_clusters

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
                    sub_labels_list = self._recursive_skeleton_split(
                        points, indices, depth=0, min_peak_dist_m=self.min_peak_dist_m)

                    fallback_used = False
                    if (len(sub_labels_list) == 1 and self.dynamic_peak_enable
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


    # -------------------------------------------------------------------------
    # Merge Back Logic
    # -------------------------------------------------------------------------
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

    def _simple_filter(self, labels):
        counts = Counter(labels)
        new_labels = labels.copy()
        for lbl, count in counts.items():
            if lbl == -1: continue
            if count < self.min_points or count > self.max_points:
                new_labels[labels == lbl] = -1
        return new_labels

    def _renumber_labels(self, labels):
        final_labels = np.full_like(labels, -1)
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]
        for i, lbl in enumerate(unique_labels):
            final_labels[labels == lbl] = i
        return final_labels, len(unique_labels)
