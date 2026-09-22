import torch
import numpy as np
import open3d as o3d
import logging
import os
from .pointnet2_model import PointNet2SemSeg

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

logger = logging.getLogger(__name__)


def _rotation_matrix_from_vectors(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """Return rotation matrix that rotates vec1 to vec2."""
    a = vec1 / (np.linalg.norm(vec1) + 1e-12)
    b = vec2 / (np.linalg.norm(vec2) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        # Parallel (or anti-parallel)
        if c > 0:
            return np.eye(3, dtype=np.float64)
        # 180-degree rotation: pick any orthogonal axis
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        v = np.cross(a, axis)
        v = v / (np.linalg.norm(v) + 1e-12)
        # Rodrigues for pi: R = -I + 2 vv^T
        return (-np.eye(3, dtype=np.float64) + 2.0 * np.outer(v, v))

    # Rodrigues' rotation formula
    vx = np.array(
        [[0.0, -v[2], v[1]],
         [v[2], 0.0, -v[0]],
         [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )
    r = np.eye(3, dtype=np.float64) + vx + (vx @ vx) * ((1.0 - c) / (s * s + 1e-12))
    return r


def _maybe_adapt_threshold(
    cabbage_probs: np.ndarray,
    base_threshold: float,
    enable: bool,
    min_keep_ratio: float,
    max_keep_ratio: float,
    min_threshold: float,
    max_threshold: float,
) -> float:
    base_threshold = float(base_threshold)
    min_threshold = float(min_threshold)
    max_threshold = float(max_threshold)

    if not enable:
        return float(np.clip(base_threshold, min_threshold, max_threshold))
    if cabbage_probs.size == 0:
        return float(np.clip(base_threshold, min_threshold, max_threshold))

    min_keep_ratio = float(min_keep_ratio)
    max_keep_ratio = float(max_keep_ratio)

    min_keep_ratio = max(min_keep_ratio, 0.0)
    max_keep_ratio = min(max_keep_ratio, 1.0)
    if min_keep_ratio <= 0 and max_keep_ratio >= 1:
        return base_threshold

    keep_ratio = float(np.mean(cabbage_probs > base_threshold))
    thr = float(np.clip(base_threshold, min_threshold, max_threshold))

    # If too few points pass the threshold (common under domain shift), lower it.
    if min_keep_ratio > 0 and keep_ratio < min_keep_ratio:
        thr_min = float(np.quantile(cabbage_probs, 1.0 - min_keep_ratio))
        thr = min(thr, thr_min)

    # If too many points pass (optional safeguard), raise it.
    keep_ratio2 = float(np.mean(cabbage_probs > thr))
    if max_keep_ratio < 1 and keep_ratio2 > max_keep_ratio:
        thr_max = float(np.quantile(cabbage_probs, 1.0 - max_keep_ratio))
        thr = max(thr, thr_max)

    return float(np.clip(thr, min_threshold, max_threshold))

class PointNetSegmentor:
    def __init__(self, cfg):
        self.cfg = cfg if isinstance(cfg, dict) else {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        pipeline_cfg = self.cfg.get('pipeline', {})
        seed = pipeline_cfg.get('seed', None)
        self._rng = None
        try:
            if seed is not None:
                self._rng = np.random.default_rng(int(seed))
        except (TypeError, ValueError):
            self._rng = None
        
        seg_cfg = self.cfg.get('segmentation', {})
        self.classes = seg_cfg.get('classes', {})
        self.num_classes = len(self.classes) if self.classes else 4 # Default 4
        self.model_path = seg_cfg.get('model_path', None)
        
        # Initialize Model
        # Assuming input is just XYZ for now, so additional_channel=0
        self.model = PointNet2SemSeg(num_classes=self.num_classes, additional_channel=0)
        self.model.to(self.device)
        
        if self.model_path and os.path.exists(self.model_path):
            logger.info(f"Loading PointNet++ model from {self.model_path}")
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
                self.model.eval()
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                logger.warning("Using initialized random weights (Results will be garbage!)")
        else:
            logger.warning(f"Model path {self.model_path} not found. Using random weights.")

    def _sample_points(self, points, plane_model, max_points):
        N = len(points)
        sampling_cfg = self.cfg.get('segmentation', {}).get('sampling', {})
        sampling_method = str(sampling_cfg.get('method', 'random')).lower()

        # Optional: bias sampling toward higher-above-ground points
        if sampling_method == 'height_biased' and plane_model is not None and len(plane_model) >= 4:
            try:
                a, b, c, d = plane_model
                heights = points[:, 0] * a + points[:, 1] * b + points[:, 2] * c + d
                top_percentile = float(sampling_cfg.get('top_percentile', 80.0))
                top_percentile = min(max(top_percentile, 0.0), 100.0)
                thr = float(np.percentile(heights, top_percentile))
                high_idx = np.where(heights >= thr)[0]
                low_idx = np.where(heights < thr)[0]

                high_fraction = float(sampling_cfg.get('high_fraction', 0.7))
                high_fraction = min(max(high_fraction, 0.0), 1.0)
                n_high = int(round(max_points * high_fraction))
                n_high = min(n_high, len(high_idx))
                n_low = max_points - n_high
                n_low = min(n_low, len(low_idx))

                rng = self._rng if self._rng is not None else np.random.default_rng()
                chosen = []
                if n_high > 0:
                    chosen.append(rng.choice(high_idx, n_high, replace=False))
                if n_low > 0:
                    chosen.append(rng.choice(low_idx, n_low, replace=False))
                
                if chosen:
                    choice = np.concatenate(chosen, axis=0)
                else:
                    choice = rng.choice(N, max_points, replace=False)

                # Fill if needed
                if choice.size < max_points:
                    remaining = np.setdiff1d(np.arange(N), choice, assume_unique=False)
                    need = max_points - choice.size
                    if remaining.size >= need:
                        choice = np.concatenate([choice, rng.choice(remaining, need, replace=False)], axis=0)

                logger.info(
                    f"Sampling(height_biased): top_p={top_percentile:.1f}, high_frac={high_fraction:.2f}, "
                    f"n_high={n_high}, n_low={n_low}"
                )
                return choice
            except Exception:
                pass # Fallback to random

        # Random sampling
        if self._rng is not None:
            return self._rng.choice(N, max_points, replace=False)
        else:
            return np.random.choice(N, max_points, replace=False)

    def segment(self, pcd, plane_model=None):
        """
        Run inference on the point cloud.
        Note: PointNet++ usually requires fixed number of points (e.g. 4096, 8192) or block processing.
        Here we implement a simplified sliding window or sampling approach.
        """
        logger.info("Running PointNet++ inference...")
        
        points = np.asarray(pcd.points)
        N = len(points)

        # Optional: align point cloud so that estimated ground normal becomes +Z.
        # This can reduce domain shift on sloped/uneven terrain without retraining.
        align_to_ground = bool(self.cfg.get('segmentation', {}).get('align_to_ground', False))
        if align_to_ground and plane_model is not None and len(plane_model) >= 3:
            try:
                a, b, c, _d = plane_model
                n = np.array([a, b, c], dtype=np.float64)
                if np.linalg.norm(n) > 1e-8:
                    if n[2] < 0:
                        n = -n
                    r = _rotation_matrix_from_vectors(n, np.array([0.0, 0.0, 1.0], dtype=np.float64))
                    points = (r @ points.T).T.astype(points.dtype, copy=False)
            except Exception:
                pass
        
        # Preprocessing for network
        # 1. Normalize (Centering only, to match training)
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        # m = np.max(np.sqrt(np.sum(points_centered ** 2, axis=1)))
        # points_norm = points_centered / m
        points_norm = points_centered # Do NOT scale if training didn't scale
        
        # 2. Prepare Tensor
        sampling_cfg = self.cfg.get('segmentation', {}).get('sampling', {})
        MAX_POINTS = int(sampling_cfg.get('max_points', 50000))
        
        # New Strategy: If N is huge and max_points is huge (meaning user wants full resolution), we use Block Inference
        # If N is huge but max_points is small, we use Downsampling (Legacy)
        
        ENABLE_BLOCK_INFERENCE = (N > 30000) and (MAX_POINTS > N or MAX_POINTS > 100000)
        
        if ENABLE_BLOCK_INFERENCE:
            # Load block settings from config
            blk_size = float(sampling_cfg.get('block_size', 2.0))
            blk_stride = float(sampling_cfg.get('stride', 2.0))
            blk_limit = int(sampling_cfg.get('batch_limit', 25000))

            logger.info(f"Input too large ({N} > 30k) and downsampling disabled. Switching to Block-wise Inference (Sliding Window)...")
            pred_choice = self._predict_in_blocks(points, block_size=blk_size, stride=blk_stride, batch_limit=blk_limit)
            full_labels = pred_choice
        else:
            if N > MAX_POINTS:
                logger.info(f"Downsampling from {N} to {MAX_POINTS} for inference speed...")
                choice = self._sample_points(points, plane_model, MAX_POINTS)
                points_input = points_norm[choice, :]
            else:
                choice = np.arange(N)
                points_input = points_norm
                
            input_tensor = torch.from_numpy(points_input).float().unsqueeze(0).permute(0, 2, 1).to(self.device) # [1, 3, N]
            
            with torch.no_grad():
                pred, _ = self.model(input_tensor)
                
                # Use score threshold if available and binary classification
                score_thresh = self.cfg.get('segmentation', {}).get('score_threshold', 0.5)
                if self.num_classes == 2:
                    probs = torch.nn.functional.softmax(pred, dim=1) # [1, 2, N]
                    cabbage_probs = probs[0, 1, :].cpu().numpy() # [N]

                    # Optional: adapt threshold to keep a reasonable fraction of points under domain shift.
                    auto_cfg = self.cfg.get('segmentation', {}).get('auto_threshold', {})
                    thr_used = _maybe_adapt_threshold(
                        cabbage_probs=cabbage_probs,
                        base_threshold=float(score_thresh),
                        enable=bool(auto_cfg.get('enable', False)),
                        min_keep_ratio=float(auto_cfg.get('min_keep_ratio', 0.01)),
                        max_keep_ratio=float(auto_cfg.get('max_keep_ratio', 0.60)),
                        min_threshold=float(auto_cfg.get('min_threshold', 0.05)),
                        max_threshold=float(auto_cfg.get('max_threshold', 0.95)),
                    )
                    if bool(auto_cfg.get('enable', False)):
                        logger.info(
                            f"Auto-threshold enabled: base={float(score_thresh):.3f}, used={thr_used:.3f}, "
                            f"keep={float(np.mean(cabbage_probs > thr_used)):.3f}"
                        )

                    pred_choice = (cabbage_probs > thr_used).astype(int)
                else:
                    pred_choice = pred.data.max(1)[1].cpu().numpy()[0] # [N]
            
            # Map back to original points
            full_labels = np.zeros(N, dtype=int)
            if N > MAX_POINTS:
                logger.info("Propagating labels to full cloud using Nearest Neighbor...")
                from scipy.spatial import cKDTree
                tree = cKDTree(points[choice])
                dists, indices = tree.query(points, k=1)
                full_labels = pred_choice[indices]
            else:
                full_labels = pred_choice

        # Extract cabbage points (Class 1)
        cabbage_mask = (full_labels == 1)
        cabbage_pcd = pcd.select_by_index(np.where(cabbage_mask)[0])
        
        logger.info(f"Inference done. Found {len(cabbage_pcd.points)} cabbage points.")
        
        return full_labels, cabbage_pcd

    def _predict_in_blocks(self, points, block_size=2.0, stride=2.0, batch_limit=25000):
        """
        Split points into spatial blocks (XY grid) and run inference on each block to avoid OOM.
        """
        N = len(points)
        full_preds = np.zeros(N, dtype=int)
        
        # Calculate bounding box
        min_p = np.min(points, axis=0)
        max_p = np.max(points, axis=0)
        
        # Grid steps
        x_steps = np.arange(min_p[0], max_p[0], stride)
        y_steps = np.arange(min_p[1], max_p[1], stride)
        
        logger.info(f"Splitting cloud into grid blocks ({len(x_steps)}x{len(y_steps)}) size={block_size}m limit={batch_limit}...")
        
        total_blocks = len(x_steps) * len(y_steps)
        processed_blocks = 0
        
        score_thresh = self.cfg.get('segmentation', {}).get('score_threshold', 0.5)

        for x in x_steps:
            for y in y_steps:
                # bounding box mask
                x_mask = (points[:, 0] >= x) & (points[:, 0] < x + block_size)
                y_mask = (points[:, 1] >= y) & (points[:, 1] < y + block_size)
                mask = x_mask & y_mask
                
                indices = np.where(mask)[0]
                if len(indices) == 0:
                    continue
                
                block_points = points[indices]
                
                # If block is too large, we must randomly downsample it to fit GPU
                # (Or further split, but simple downsample is usually okay for blocks > 25k)
                if len(block_points) > batch_limit:
                    choice = np.random.choice(len(block_points), batch_limit, replace=False)
                    proc_points = block_points[choice]
                else:
                    choice = np.arange(len(block_points))
                    proc_points = block_points

                # Center the block locally!
                # PointNet++ works on relative coordinates. 
                # If we use global coordinates, they are huge. 
                # If the trained model expects centered objects, we center the block.
                # Assuming standard PointNet normalization:
                blk_centroid = np.mean(proc_points, axis=0)
                proc_points_norm = proc_points - blk_centroid
                
                # FIX: Handle small point count (padding) to avoid PointNet++ crash (N < nsample)
                num_points = len(proc_points_norm)
                min_points_needed = 1024 # Standard PointNet++ NPOINT
                
                if num_points < min_points_needed:
                    needed = min_points_needed - num_points
                    dup_idx = np.random.choice(num_points, needed, replace=True)
                    proc_points_padded = np.concatenate([proc_points_norm, proc_points_norm[dup_idx]], axis=0)
                    input_points = proc_points_padded
                else:
                    input_points = proc_points_norm
                
                # Inference
                input_tensor = torch.from_numpy(input_points).float().unsqueeze(0).permute(0, 2, 1).to(self.device)
                
                with torch.no_grad():
                    pred, _ = self.model(input_tensor)
                    
                    if self.num_classes == 2:
                        probs = torch.nn.functional.softmax(pred, dim=1)
                        cabbage_probs = probs[0, 1, :].cpu().numpy() # [N_padded]
                        
                        # Un-pad
                        cabbage_probs = cabbage_probs[:num_points]
                        
                        batch_preds = (cabbage_probs > score_thresh).astype(int)
                    else:
                        cls_pred = pred.data.max(1)[1].cpu().numpy()[0] # [N_padded]
                        cls_pred = cls_pred[:num_points]
                        batch_preds = cls_pred
                
                # Map back to full block
                if len(block_points) > batch_limit:
                     # Propagate back to full block neighbors
                     from scipy.spatial import cKDTree
                     tree = cKDTree(proc_points) # Tree on 25k points is fast
                     dists, n_indices = tree.query(block_points, k=1)
                     full_block_preds = batch_preds[n_indices]
                else:
                     full_block_preds = batch_preds
                
                # Store globally
                full_preds[indices] = full_block_preds
                
                processed_blocks += 1
                if processed_blocks % 10 == 0:
                     logger.debug(f"Processed block {processed_blocks}/{total_blocks}")

        return full_preds
        
        return full_labels, cabbage_pcd
