import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull
import logging

logger = logging.getLogger(__name__)

class TraitCalculator:
    def __init__(self, cfg):
        self.traits_cfg = cfg.get('traits', {})
        self.voxel_size = cfg.get('pipeline', {}).get('voxel_size', 0.005)
        
    def _filter_outliers(self, points):
        """基于质心距离的 MAD 稳健离群过滤, 防止远噪声点拉大冠幅/体积。

        返回 (过滤后的点, 被移除的点数)。
        """
        cfg = self.traits_cfg.get('outlier_filter', {})
        if not cfg.get('enable', True):
            return points, 0
        min_pts = int(cfg.get('min_points', 100))
        if len(points) < min_pts:
            return points, 0
        thresh = float(cfg.get('mad_thresh', 3.5))
        centroid = np.median(points, axis=0)
        dists = np.linalg.norm(points - centroid, axis=1)
        med = float(np.median(dists))
        mad = float(np.median(np.abs(dists - med)))
        if mad < 1e-9:
            return points, 0
        cutoff = med + thresh * 1.4826 * mad
        keep = dists <= cutoff
        n_removed = int((~keep).sum())
        return points[keep], n_removed

    def _min_enclosing_circle(self, points_2d):
        """最小外接圆 (在凸包顶点上枚举两点圆/三点圆).

        返回 (圆心, 半径). 凸包顶点数通常 <100, 枚举复杂度完全可接受.
        """
        try:
            hull = ConvexHull(points_2d)
            pts = points_2d[hull.vertices]
        except Exception:
            pts = points_2d
        pts = np.unique(np.round(pts, 6), axis=0)
        n = len(pts)
        if n == 0:
            return np.zeros(2), 0.0
        if n == 1:
            return pts[0].copy(), 0.0
        if n == 2:
            c = (pts[0] + pts[1]) / 2.0
            return c, float(np.linalg.norm(pts[0] - pts[1]) / 2.0)

        best_r = np.inf
        best_c = np.zeros(2)

        def covers(c, r):
            return bool(np.all(np.linalg.norm(pts - c, axis=1) <= r + 1e-9))

        # 两点圆 (直径两端)
        for i in range(n):
            for j in range(i + 1, n):
                c = (pts[i] + pts[j]) / 2.0
                r = float(np.linalg.norm(pts[i] - pts[j]) / 2.0)
                if r < best_r and covers(c, r):
                    best_r, best_c = r, c

        # 三点圆 (外接圆)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    ax, ay = pts[i]
                    bx, by = pts[j]
                    cx, cy = pts[k]
                    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
                    if abs(d) < 1e-12:
                        continue
                    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
                    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
                    c = np.array([ux, uy])
                    r = float(np.linalg.norm(c - pts[i]))
                    if r < best_r and covers(c, r):
                        best_r, best_c = r, c

        return best_c, best_r

    @staticmethod
    def _tetra_volume(tetra):
        a, b, c, d = tetra
        return abs(float(np.dot(b - a, np.cross(c - a, d - a)))) / 6.0

    @staticmethod
    def _circumradius(tetra):
        a, b, c, d = tetra
        A = 2.0 * (b - a)
        B = 2.0 * (c - a)
        C = 2.0 * (d - a)
        M = np.array([A, B, C])
        rhs = np.array([
            float(b @ b - a @ a),
            float(c @ c - a @ a),
            float(d @ d - a @ a),
        ])
        try:
            center = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            return np.inf
        return float(np.linalg.norm(center - a))

    def _alpha_shape_volume(self, points):
        """alpha-shape 体积 (3D Delaunay 四面体化 + 外接球半径过滤).

        保留外接球半径 <= alpha 的四面体, 体积求和; 相比凸包能反映表面凹陷.
        返回体积 (m^3).
        """
        from scipy.spatial import Delaunay
        alpha = float(self.traits_cfg.get('alpha_shape_radius', 0.03))
        if alpha <= 0:
            return 0.0
        try:
            tri = Delaunay(points)
        except Exception:
            return 0.0
        vol = 0.0
        for simplex in tri.simplices:
            tetra = points[simplex]
            if self._circumradius(tetra) <= alpha:
                vol += self._tetra_volume(tetra)
        return vol

    def calculate_traits(self, pcd, plant_id, plane_model=None):
        """
        Calculate phenotype traits for a single plant point cloud.
        
        Args:
            pcd: Open3D PointCloud of the single plant.
            plant_id: Integer ID of the plant.
            plane_model: [a, b, c, d] ground plane equation for height correction.
            
        Returns:
            dict: Dictionary of calculated traits.
        """
        points = np.asarray(pcd.points)
        if len(points) < 4: # Need at least 4 points for 3D hull/volume usually
            return None

        # 离群过滤: 防止远噪声点拉大冠幅/体积
        raw_count = len(points)
        points, n_removed = self._filter_outliers(points)
        if len(points) < 4:
            return None
            
        traits = {
            "plant_id": int(plant_id),
            "point_count": raw_count,
            "outlier_removed": n_removed
        }
        
        # 1. Centroid
        centroid = np.mean(points, axis=0)
        traits["centroid"] = centroid.tolist()
        
        # 2. Bounding Box (Axis Aligned)
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        traits["bbox"] = {
            "min": min_bound.tolist(),
            "max": max_bound.tolist()
        }
        
        # 3. Height
        # If plane_model is provided, calculate height relative to plane
        # Otherwise assume Z is up and ground is roughly at min Z (or use raw Z range)
        if plane_model is not None:
            [a, b, c, d] = plane_model
            # Signed distance to plane
            heights = (points[:, 0] * a + points[:, 1] * b + points[:, 2] * c + d)
            # Ensure positive heights (if normal points up)
            heights = np.abs(heights) 
        else:
            # Fallback: relative to min Z of the cluster
            heights = points[:, 2] - np.min(points[:, 2])
            
        p_low = self.traits_cfg.get('height_percentile_low', 1)
        p_high = self.traits_cfg.get('height_percentile_high', 99)
        
        h_min = np.percentile(heights, p_low)
        h_max = np.percentile(heights, p_high)
        traits["height_cm"] = float(h_max - h_min) * 100.0
        traits["height_max_cm"] = float(np.max(heights)) * 100.0
        
        # 4. Projected Area & Crown Diameter (2D Analysis)
        points_2d = points[:, :2] # XY projection
        try:
            hull_2d = ConvexHull(points_2d)
            projected_area = hull_2d.volume # In 2D, volume is area
            traits["projected_area_cm2"] = float(projected_area) * 10000.0
            
            # 稳健冠幅 = 等面积圆直径 2*sqrt(area/pi): 对远噪声点敏感度低于凸包最大距离
            equivalent_diameter = 2.0 * np.sqrt(projected_area / np.pi)
            traits["crown_diameter_cm"] = float(equivalent_diameter) * 100.0

            # 原始凸包最大直径 (参考值, 对噪声敏感)
            hull_points = points_2d[hull_2d.vertices]
            max_dist = 0
            for i in range(len(hull_points)):
                for j in range(i+1, len(hull_points)):
                    dist = np.linalg.norm(hull_points[i] - hull_points[j])
                    if dist > max_dist:
                        max_dist = dist
            traits["crown_diameter_hull_cm"] = float(max_dist) * 100.0

            # 最小外接圆直径 (论文"凸包最小外接圆"口径)
            try:
                _, mec_r = self._min_enclosing_circle(points_2d)
                traits["crown_diameter_circle_cm"] = float(2.0 * mec_r) * 100.0
            except Exception:
                traits["crown_diameter_circle_cm"] = 0.0

            # 方向冠幅: 沿东西(X轴)/南北(Y轴)的凸包跨度及其均值
            ew_span = hull_points[:, 0].max() - hull_points[:, 0].min()
            ns_span = hull_points[:, 1].max() - hull_points[:, 1].min()
            traits["crown_diameter_ew_cm"] = float(ew_span) * 100.0
            traits["crown_diameter_ns_cm"] = float(ns_span) * 100.0
            traits["crown_diameter_mean_cm"] = float((ew_span + ns_span) / 2.0) * 100.0
            
        except Exception as e:
            logger.warning(f"Failed to compute 2D hull for plant {plant_id}: {e}")
            traits["projected_area_cm2"] = 0.0
            traits["crown_diameter_hull_cm"] = 0.0
            traits["crown_diameter_circle_cm"] = 0.0
            traits["crown_diameter_cm"] = 0.0
            traits["crown_diameter_ew_cm"] = 0.0
            traits["crown_diameter_ns_cm"] = 0.0
            traits["crown_diameter_mean_cm"] = 0.0

        # 5. Volume (Alpha Shape): 唯一体积口径, 反映植株真实空间占用
        # alpha-shape: 3D Delaunay 四面体化 + 外接球半径过滤, 能反映表面凹陷
        alpha_volume_m3 = 0.0
        try:
            alpha_volume_m3 = float(self._alpha_shape_volume(points))
            traits["volume_alpha_cm3"] = alpha_volume_m3 * 1000000.0
        except Exception as e:
            logger.warning(f"Failed to compute alpha shape volume for plant {plant_id}: {e}")
            traits["volume_alpha_cm3"] = 0.0

        # 6. Compactness
        # compactness = alpha体积 / (包围盒体积 + eps)
        bbox_dx = max_bound[0] - min_bound[0]
        bbox_dy = max_bound[1] - min_bound[1]
        bbox_dz = max_bound[2] - min_bound[2]
        bbox_vol_m3 = bbox_dx * bbox_dy * bbox_dz
        
        if bbox_vol_m3 > 1e-6:
            traits["compactness"] = alpha_volume_m3 / bbox_vol_m3
        else:
            traits["compactness"] = 0.0

        return traits
