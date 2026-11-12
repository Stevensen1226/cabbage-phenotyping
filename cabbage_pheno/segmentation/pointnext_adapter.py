"""PointNeXt-L semantic adapter for full-scene Cabbage evaluation."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch


class PointNeXtAdapter:
    def __init__(self, cfg: dict, device: str = "cuda"):
        seg_cfg = cfg.get("segmentation", {})
        root = Path(seg_cfg.get("pointnext_root", "PointNeXt")).expanduser().resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from openpoints.models import build_model_from_cfg
        from openpoints.utils import EasyConfig

        model_cfg_path = Path(seg_cfg["pointnext_config"])
        if not model_cfg_path.is_absolute():
            model_cfg_path = root / model_cfg_path
        ckpt_path = Path(seg_cfg["pointnext_checkpoint"])
        if not ckpt_path.is_absolute():
            ckpt_path = Path.cwd() / ckpt_path

        model_cfg = EasyConfig()
        model_cfg.load(str(model_cfg_path), recursive=True)
        self.model = build_model_from_cfg(model_cfg.model).to(device).eval()
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        self.model.load_state_dict(state)
        self.device = device
        self.block_size = float(seg_cfg.get("pointnext_block_size", 2.0))
        self.overlap = float(seg_cfg.get("pointnext_overlap", 0.5))
        self.max_points = int(seg_cfg.get("pointnext_max_points", 60000))
        self.batch_points = int(seg_cfg.get("pointnext_batch_points", 24000))

    def _predict_block(self, xyz: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        n = len(xyz)
        if n == 0:
            return np.zeros((0, 2), dtype=np.float32)
        if n > self.max_points:
            sel = np.random.choice(n, self.max_points, replace=False)
            xyz_use, rgb_use = xyz[sel], rgb[sel]
        else:
            sel = np.arange(n)
            xyz_use, rgb_use = xyz, rgb

        pos = xyz_use.astype(np.float32, copy=True)
        pos -= pos.mean(axis=0, keepdims=True)
        pos[:, 2] -= pos[:, 2].min()
        feat = (rgb_use.astype(np.float32) * 2.0 - 1.0).T[None]

        pos_t = torch.from_numpy(pos[None]).to(self.device)
        feat_t = torch.from_numpy(feat).to(self.device)
        off_t = torch.tensor([len(pos)], dtype=torch.int32, device=self.device)[None]
        with torch.no_grad():
            logits = self.model({"pos": pos_t, "x": feat_t, "offset": off_t})
            prob = torch.softmax(logits[0], dim=0).cpu().numpy().T

        if len(sel) < n:
            full = np.zeros((n, 2), dtype=np.float32)
            full[sel] = prob
            return full
        return prob

    def predict_mask(self, xyz: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=np.float32)
        rgb = np.asarray(rgb, dtype=np.float32)
        if len(xyz) == 0:
            return np.zeros(0, dtype=bool)

        stride = max(0.25, self.block_size * (1.0 - self.overlap))
        lo = xyz[:, :2].min(axis=0)
        hi = xyz[:, :2].max(axis=0)
        xs = np.arange(lo[0], max(lo[0] + 1e-6, hi[0] - self.block_size + 1e-6), stride)
        ys = np.arange(lo[1], max(lo[1] + 1e-6, hi[1] - self.block_size + 1e-6), stride)
        xs = np.unique(np.append(xs, hi[0] - self.block_size))
        ys = np.unique(np.append(ys, hi[1] - self.block_size))

        prob_sum = np.zeros((len(xyz), 2), dtype=np.float32)
        count = np.zeros(len(xyz), dtype=np.float32)
        for x0 in xs:
            for y0 in ys:
                mask = ((xyz[:, 0] >= x0) & (xyz[:, 0] < x0 + self.block_size)
                        & (xyz[:, 1] >= y0) & (xyz[:, 1] < y0 + self.block_size))
                idx = np.flatnonzero(mask)
                if len(idx) < 32:
                    continue
                prob = self._predict_block(xyz[idx], rgb[idx])
                prob_sum[idx] += prob
                count[idx] += 1.0
        covered = count > 0
        result = np.zeros(len(xyz), dtype=bool)
        if covered.any():
            result[covered] = (prob_sum[covered] / count[covered, None]).argmax(axis=1) == 1
        return result
