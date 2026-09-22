import glob
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import open3d as o3d

from cabbage_pheno.preprocess import PointCloudCleaner, GroundSegmentor

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:  # pragma: no cover
    cKDTree = None

logger = logging.getLogger(__name__)


@dataclass
class PreprocessResult:
    pcd_clean: o3d.geometry.PointCloud
    points_clean: np.ndarray
    non_ground_pcd: o3d.geometry.PointCloud
    ground_pcd: Optional[o3d.geometry.PointCloud]
    plane_model: Optional[list] = None


def collect_input_files(input_path: str) -> List[str]:
    if os.path.exists(input_path) and os.path.isdir(input_path):
        input_files = []
        for ext in ['*.ply', '*.txt', '*.pcd', '*.npy']:
            input_files.extend(glob.glob(os.path.join(input_path, '**', ext), recursive=True))
        return sorted([f for f in set(input_files) if '_gt' not in f])

    if '*' in input_path:
        return sorted(glob.glob(input_path))

    return [input_path]


def _fit_plane_ransac(points, distance_threshold, num_iterations, rng):
    """
    确定性 RANSAC 平面拟合。

    替代 Open3D 的 segment_plane (CUDA 版不支持 rng 参数), 用固定种子的
    numpy Generator 采样三点拟合平面, 保证评估可复现。

    返回 [a, b, c, d], 满足 a*x + b*y + c*z + d = 0, 其中 (a, b, c) 为单位法向量。
    与 Open3D segment_plane 返回格式一致。
    """
    n = len(points)
    if n < 3:
        return None

    best_normal = None
    best_d = 0.0
    best_inlier_count = -1

    for _ in range(num_iterations):
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal = normal / norm
        d = -float(np.dot(normal, p0))
        dist = np.abs(points @ normal + d)
        inlier_count = int(np.count_nonzero(dist < distance_threshold))
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_normal = normal
            best_d = d

    if best_normal is None:
        return None

    # 用 inliers 重新拟合 (协方差最小特征向量 = 平面法向量, 最小二乘)
    dist = np.abs(points @ best_normal + best_d)
    inlier_pts = points[dist < distance_threshold]
    if len(inlier_pts) >= 3:
        centroid = inlier_pts.mean(axis=0)
        centered = inlier_pts - centroid
        cov = centered.T @ centered  # (3, 3)
        _, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]  # 最小特征值对应特征向量 = 平面法向量
        d = -float(np.dot(normal, centroid))

    return [float(normal[0]), float(normal[1]), float(normal[2]), d]


def preprocess_point_cloud(pcd, cfg, cleaner=None):
    """Shared preprocessing entry for training/inference/evaluation."""
    cleaner = cleaner or PointCloudCleaner(cfg)

    # 确定性: 从 pipeline.seed 读取随机种子, 固定 RANSAC 地面拟合结果
    pipeline_cfg = cfg.get('pipeline', {})
    seed = pipeline_cfg.get('seed', None)
    rng = np.random.default_rng(int(seed)) if seed is not None else None

    t0 = None
    clean_pcd = cleaner.remove_outliers(pcd)
    points_clean = np.asarray(clean_pcd.points)
    ground_pcd = None
    plane_model = None

    ground_cfg = cfg.get('preprocess', {}).get('ground_removal', {})
    if ground_cfg.get('enable', False) and len(points_clean) > 0:
        method = ground_cfg.get('method', 'ransac')

        if method == 'dtm_grid':
            # DTM 局部地表模型: 跟随地形起伏, 区分贴地甘蓝底部与地面。
            # 用 GroundSegmentor 的 dtm_grid 实现, 将 ground_removal.dtm_grid 参数映射过去。
            dtm_cfg = ground_cfg.get('dtm_grid', {})
            gs_cfg = {'preprocess': {'ground_estimation': {
                'method': 'dtm_grid',
                'dtm_grid': dtm_cfg,
            }}}
            try:
                seg = GroundSegmentor(gs_cfg)
                ground_pcd, non_ground_pcd, plane_model = seg.segment_ground(clean_pcd)
                clean_pcd = non_ground_pcd
                points_clean = np.asarray(clean_pcd.points)
            except Exception as e:
                logger.warning(f'DTM ground segmentation failed ({e}); falling back to RANSAC.')
                method = 'ransac'

        if method == 'ransac':
            plane_model = None
            pts = np.asarray(clean_pcd.points)
            z_thresh = np.percentile(pts[:, 2], ground_cfg.get('fit_percentile', 20))
            low_pts = pts[pts[:, 2] <= z_thresh]
            if len(low_pts) >= 3:
                ransac_thresh = ground_cfg.get('ransac_thresh', 0.02)
                if rng is not None:
                    plane_model = _fit_plane_ransac(low_pts, ransac_thresh, 200, rng)
                else:
                    # 无种子时回退到 Open3D (保持原有行为)
                    plane_pcd = o3d.geometry.PointCloud()
                    plane_pcd.points = o3d.utility.Vector3dVector(low_pts)
                    plane_model, _ = plane_pcd.segment_plane(
                        distance_threshold=ransac_thresh,
                        ransac_n=3,
                        num_iterations=200,
                    )
                if plane_model is None:
                    a = b = c = d = 0.0
                else:
                    a, b, c, d = plane_model
                dist = np.abs(a * pts[:, 0] + b * pts[:, 1] + c * pts[:, 2] + d) / np.sqrt(a * a + b * b + c * c)
                ground_mask = dist < ground_cfg.get('remove_thresh', 0.03)
                ground_pcd = clean_pcd.select_by_index(np.where(ground_mask)[0].tolist())
                clean_pcd = clean_pcd.select_by_index(np.where(~ground_mask)[0].tolist())
                points_clean = np.asarray(clean_pcd.points)

    non_ground_pcd = clean_pcd
    return PreprocessResult(
        pcd_clean=clean_pcd,
        points_clean=points_clean,
        non_ground_pcd=non_ground_pcd,
        ground_pcd=ground_pcd,
        plane_model=plane_model,
    )
