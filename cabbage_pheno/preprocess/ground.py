import open3d as o3d
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:  # pragma: no cover
    cKDTree = None

try:
    from scipy.ndimage import gaussian_filter  # type: ignore
except Exception:  # pragma: no cover
    gaussian_filter = None

class GroundSegmentor:
    def __init__(self, cfg):
        self.ground_cfg = cfg.get('preprocess', {}).get('ground_estimation', {})
        self.distance_threshold = self.ground_cfg.get('distance_threshold', 0.02)
        self.method = self.ground_cfg.get('method', 'ransac')
        # Optional: fit the plane using only the lowest-Z percentile points.
        # This helps prevent RANSAC from locking onto a large, flat leaf canopy.
        # Set to e.g. 10~30; null/omitted = use all points.
        self.fit_low_percentile = self.ground_cfg.get('fit_low_percentile', None)

        # Optional: when classifying on the full cloud, restrict candidates to low-Z points.
        # This can reduce misclassified low canopy/leaf points when the plane fit is imperfect.
        # null/omitted = do not restrict.
        self.classify_max_z_percentile = self.ground_cfg.get('classify_max_z_percentile', None)

        # DTM-grid (local ground surface) parameters
        self.dtm_cfg = self.ground_cfg.get('dtm_grid', {}) if isinstance(self.ground_cfg.get('dtm_grid', {}), dict) else {}

        # Optional: after fitting the plane, classify ground on the full cloud by distance-to-plane.
        # This fixes the common issue where RANSAC inliers (especially when fitting on a subset)
        # yield too few ground points.
        self.classify_all_points = bool(self.ground_cfg.get('classify_all_points', False))

    def _segment_ransac_plane(self, pcd):
        """RANSAC plane + (optional) full-cloud classification."""
        pcd_fit = pcd
        fit_idx = None
        if self.fit_low_percentile is not None:
            try:
                p = float(self.fit_low_percentile)
                if 0.0 < p < 100.0:
                    if p > 40.0:
                        logger.warning(
                            f"fit_low_percentile={p:.1f} is quite high and may include cabbage points in plane fitting. "
                            "Consider 10~30 for cleaner ground separation."
                        )
                    pts = np.asarray(pcd.points)
                    z_thr = float(np.percentile(pts[:, 2], p))
                    fit_idx = np.where(pts[:, 2] <= z_thr)[0]
                    if fit_idx.size >= 3:
                        pcd_fit = pcd.select_by_index(fit_idx.tolist())
                        logger.info(f"RANSAC fitting on lowest {p:.1f}%% Z points: {fit_idx.size} / {len(pts)}")
                    else:
                        fit_idx = None
            except Exception:
                fit_idx = None

        plane_model, inliers = pcd_fit.segment_plane(
            distance_threshold=float(self.distance_threshold),
            ransac_n=3,
            num_iterations=1000,
        )

        a, b, c, d = plane_model
        if c < 0:
            a, b, c, d = -a, -b, -c, -d
            plane_model = [a, b, c, d]

        nrm = float(np.sqrt(a * a + b * b + c * c))
        if nrm <= 0:
            nrm = 1.0

        if self.classify_all_points:
            pts_all = np.asarray(pcd.points)
            dists = (pts_all[:, 0] * a + pts_all[:, 1] * b + pts_all[:, 2] * c + d) / nrm
            mask = np.abs(dists) < float(self.distance_threshold)

            if self.classify_max_z_percentile is not None:
                try:
                    pz = float(self.classify_max_z_percentile)
                    if 0.0 < pz < 100.0:
                        z_max = float(np.percentile(pts_all[:, 2], pz))
                        mask &= (pts_all[:, 2] <= z_max)
                        logger.info(f"Classify ground candidates restricted to lowest {pz:.1f}%% Z points (z<= {z_max:.3f})")
                except Exception:
                    pass

            inliers_full = np.where(mask)[0].tolist()
            ground_pcd = pcd.select_by_index(inliers_full)
            non_ground_pcd = pcd.select_by_index(inliers_full, invert=True)
        else:
            if fit_idx is not None:
                inliers = fit_idx[np.asarray(inliers, dtype=int)]
                inliers = inliers.tolist()
            ground_pcd = pcd.select_by_index(inliers)
            non_ground_pcd = pcd.select_by_index(inliers, invert=True)

        logger.info(f"Ground plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
        logger.info(f"Ground points: {len(ground_pcd.points)}, Non-ground points: {len(non_ground_pcd.points)}")
        return ground_pcd, non_ground_pcd, plane_model

    def _segment_dtm_grid(self, pcd):
        """DTM grid: build a local ground surface by XY gridding and taking low-percentile Z in each cell."""
        pts = np.asarray(pcd.points)
        if pts.shape[0] < 10:
            return self._segment_ransac_plane(pcd)

        cell_size = float(self.dtm_cfg.get('cell_size', 0.05))
        z_percentile = float(self.dtm_cfg.get('z_percentile', 5.0))
        min_points_per_cell = int(self.dtm_cfg.get('min_points_per_cell', 5))
        smooth_sigma_m = float(self.dtm_cfg.get('smooth_sigma_m', 0.0))
        max_above_ground = float(self.dtm_cfg.get('max_above_ground', 0.03))
        below_margin = float(self.dtm_cfg.get('below_margin', 0.02))
        max_cells = int(self.dtm_cfg.get('max_cells', 2000000))
        fit_low_percentile = self.dtm_cfg.get('fit_low_percentile', None)

        if cell_size <= 0:
            return self._segment_ransac_plane(pcd)

        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        xmin, xmax = float(x.min()), float(x.max())
        ymin, ymax = float(y.min()), float(y.max())
        nx = int(np.floor((xmax - xmin) / cell_size)) + 1
        ny = int(np.floor((ymax - ymin) / cell_size)) + 1

        if nx <= 1 or ny <= 1 or nx * ny > max_cells:
            logger.warning(f"DTM grid size {nx}x{ny} too large/degenerate; falling back to RANSAC.")
            return self._segment_ransac_plane(pcd)

        ix = np.clip(((x - xmin) / cell_size).astype(np.int32), 0, nx - 1)
        iy = np.clip(((y - ymin) / cell_size).astype(np.int32), 0, ny - 1)

        # Build ground surface using only globally low-Z points to avoid canopy/plant contamination
        surface_mask = np.ones(z.shape[0], dtype=bool)
        if fit_low_percentile is not None:
            try:
                p = float(fit_low_percentile)
                if 0.0 < p < 100.0:
                    z_thr = float(np.percentile(z, p))
                    surface_mask = z <= z_thr
                    logger.info(f"DTM surface built on lowest {p:.1f}%% Z points: {int(surface_mask.sum())} / {z.shape[0]}")
            except Exception:
                surface_mask = np.ones(z.shape[0], dtype=bool)

        lin = iy.astype(np.int64) * int(nx) + ix.astype(np.int64)
        lin_surface = lin[surface_mask]
        z_surface = z[surface_mask]
        if lin_surface.size < 10:
            logger.warning("Too few points for DTM surface after low-Z filtering; falling back to RANSAC.")
            return self._segment_ransac_plane(pcd)

        order = np.argsort(lin_surface)
        lin_s = lin_surface[order]
        z_s = z_surface[order]

        grid = np.full((ny, nx), np.nan, dtype=np.float32)
        filled = np.zeros((ny, nx), dtype=bool)

        start = 0
        while start < lin_s.size:
            end = start + 1
            while end < lin_s.size and lin_s[end] == lin_s[start]:
                end += 1
            count = end - start
            if count >= min_points_per_cell:
                lid = int(lin_s[start])
                cy = lid // nx
                cx = lid - cy * nx
                grid[cy, cx] = np.percentile(z_s[start:end], z_percentile)
                filled[cy, cx] = True
            start = end

        filled_count = int(filled.sum())
        logger.info(
            f"DTM grid built: cell_size={cell_size:.3f}m, filled_cells={filled_count}/{nx*ny}, "
            f"z_percentile={z_percentile:.1f}, min_pts/cell={min_points_per_cell}"
        )
        if filled_count < 10:
            logger.warning("Too few filled DTM cells; falling back to RANSAC.")
            return self._segment_ransac_plane(pcd)

        # Fill holes by nearest filled cell (fast with KDTree if available)
        if np.any(~filled):
            ys, xs = np.where(filled)
            centers = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
            values = grid[ys, xs]
            yq, xq = np.where(~filled)
            queries = np.stack([xq.astype(np.float32), yq.astype(np.float32)], axis=1)
            if cKDTree is not None:
                tree = cKDTree(centers)
                _, nn = tree.query(queries, k=1)
                grid[yq, xq] = values[nn]
            else:
                # Fallback: brute-force nearest (ok for small grids)
                for qi, (qx, qy) in enumerate(zip(xq, yq)):
                    d2 = (centers[:, 0] - float(qx)) ** 2 + (centers[:, 1] - float(qy)) ** 2
                    grid[qy, qx] = values[int(np.argmin(d2))]

        # Optional smoothing to reduce staircase artifacts
        if smooth_sigma_m > 0 and gaussian_filter is not None:
            sigma_cells = max(0.0, float(smooth_sigma_m) / float(cell_size))
            if sigma_cells > 0:
                grid = gaussian_filter(grid, sigma=sigma_cells, mode='nearest')

        # Classify each point by height above local ground surface
        z0 = grid[iy, ix].astype(np.float32)
        dz = z.astype(np.float32) - z0
        ground_mask = (dz <= max_above_ground) & (dz >= -below_margin)
        inliers_full = np.where(ground_mask)[0].tolist()
        ground_pcd = pcd.select_by_index(inliers_full)
        non_ground_pcd = pcd.select_by_index(inliers_full, invert=True)

        # Still return a plane model for downstream alignment/traits (fit a plane on the ground points)
        plane_model = [0.0, 0.0, 1.0, 0.0]
        if len(ground_pcd.points) >= 3:
            try:
                plane_fit_thr = float(self.dtm_cfg.get('plane_fit_distance_threshold', 0.02))
                plane_model, _ = ground_pcd.segment_plane(
                    distance_threshold=plane_fit_thr,
                    ransac_n=3,
                    num_iterations=500,
                )
                a, b, c, d = plane_model
                if c < 0:
                    plane_model = [-a, -b, -c, -d]
            except Exception:
                pass

        logger.info(
            f"Ground points: {len(ground_pcd.points)}, Non-ground points: {len(non_ground_pcd.points)} "
            f"(DTM max_above_ground={max_above_ground:.3f}, below_margin={below_margin:.3f})"
        )
        # Store mask for advanced users/evaluation
        self._last_ground_mask = ground_mask
        self._last_ground_indices = inliers_full
        
        return ground_pcd, non_ground_pcd, plane_model

    def segment_ground(self, pcd):
        """最简地面分割：Open3D RANSAC 平面 + inliers 作为 ground。"""
        logger.info(f"Segmenting ground using {self.method} with threshold {self.distance_threshold}")
        
        # Reset debug info
        self._last_ground_mask = None
        self._last_ground_indices = None
        
        if self.method == 'ransac':
            return self._segment_ransac_plane(pcd)

        if self.method == 'dtm_grid':
            return self._segment_dtm_grid(pcd)

        if self.method == 'none':
            self._last_ground_mask = np.zeros(len(pcd.points), dtype=bool)
            import open3d as o3d
            return o3d.geometry.PointCloud(), pcd, None

        raise NotImplementedError("Supported ground methods: 'ransac', 'dtm_grid', 'none'.")
    
    def get_last_ground_mask(self, full_size=None):
        """
        Returns the boolean mask of ground points from the last segmentation.
        If the last method was RANSAC on indices, it converts indices to mask.
        """
        if hasattr(self, '_last_ground_mask') and self._last_ground_mask is not None:
            return self._last_ground_mask
            
        if hasattr(self, '_last_ground_indices') and self._last_ground_indices is not None and full_size is not None:
             mask = np.zeros(full_size, dtype=bool)
             mask[self._last_ground_indices] = True
             return mask
             
        return None

