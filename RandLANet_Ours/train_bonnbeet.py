#!/usr/bin/env python3
"""
RandLA-Net 语义分割训练 — BonnBeetClouds3D (甜菜株级, 二分类)。

使用官方 train/val 划分 (不是随机 80/20):
  --data-train  BonnBeetClouds3D/randla_format/train
  --data-val    BonnBeetClouds3D/randla_format/val

模型与甘蓝一致: RandLANet(d_in=6, num_classes=2)。语义标签: 0=背景, 1=植株。

性能优化 (跑满 A100):
  - DataLoader num_workers=8, pin_memory
  - 混合精度 (AMP) + GradScaler
  - 大 batch (默认 8, 可调)
  - 缓存 KNN 图到 GPU 加速 (RandLANet 的 LocalEnc 已用 cKDTree, 这里保持兼容)
"""

import argparse
import glob
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RandLABonnBeet")


class BonnBeetDataset(Dataset):
    """从 .pth 四元组 (xyz, rgb, sem, inst) 加载, 随机采样 num_points。"""

    def __init__(self, data_root, num_points=65536, augment=False):
        self.files = sorted(glob.glob(os.path.join(data_root, '*.pth')))
        self.num_points = num_points
        self.augment = augment
        logger.info(f"Dataset [{os.path.basename(data_root)}]: {len(self.files)} scans, {num_points} pts")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=False)
        xyz, rgb, sem = data[0], data[1], data[2]
        if hasattr(xyz, 'numpy'):
            xyz = xyz.numpy()
        if hasattr(rgb, 'numpy'):
            rgb = rgb.numpy()
        if hasattr(sem, 'numpy'):
            sem = sem.numpy()
        sem = sem.astype(np.int64)
        xyz, rgb = xyz.astype(np.float32), rgb.astype(np.float32)

        n = len(xyz)
        if n >= self.num_points:
            sel = np.random.choice(n, self.num_points, replace=False)
        else:
            sel = np.random.choice(n, self.num_points, replace=True)
        xyz = xyz[sel]
        rgb = rgb[sel]
        sem = sem[sel]

        xyz -= xyz.min(0)

        if self.augment:
            if np.random.random() > 0.5:
                theta = np.random.rand() * 2 * np.pi
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=np.float32)
                xyz = xyz @ R.T
            xyz += np.random.randn(*xyz.shape).astype(np.float32) * 0.005

        return (torch.from_numpy(xyz), torch.from_numpy(rgb),
                torch.from_numpy(sem).long())


def collate_fn(batch):
    xyz, rgb, sem = zip(*batch)
    return torch.stack(xyz), torch.stack(rgb), torch.stack(sem)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-train', default='BonnBeetClouds3D/randla_format/train')
    parser.add_argument('--data-val', default='BonnBeetClouds3D/randla_format/val')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num-points', type=int, default=65536)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--save-dir', default='exp/randlanet_bonnbeet')
    args = parser.parse_args()

    ds_train = BonnBeetDataset(args.data_train, args.num_points, augment=True)
    ds_val = BonnBeetDataset(args.data_val, args.num_points, augment=False)
    dl_train = DataLoader(ds_train, args.batch, True, collate_fn=collate_fn,
                          num_workers=args.workers, pin_memory=True, drop_last=True)
    dl_val = DataLoader(ds_val, 1, False, collate_fn=collate_fn, num_workers=2)

    from randlanet import RandLANet
    model = RandLANet(d_in=6, num_classes=2).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f'模型参数: {n_params/1e6:.2f}M')

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    class_w = torch.tensor([0.4, 1.0], device='cuda')
    crit = nn.CrossEntropyLoss(ignore_index=-100, weight=class_w)

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    os.makedirs(args.save_dir, exist_ok=True)
    logger.info(f'训练: {len(ds_train)} train / {len(ds_val)} val, batch={args.batch}')

    best_val = float('inf')
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        for xyz, rgb, sem in dl_train:
            xyz, rgb, sem = xyz.cuda(non_blocking=True), rgb.cuda(non_blocking=True), sem.cuda(non_blocking=True)
            sem[sem < 0] = -100
            with torch.cuda.amp.autocast(enabled=True):
                pred = model(xyz, rgb)
                loss = crit(pred.reshape(-1, 2), sem.reshape(-1))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        dt = time.time() - t0

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xyz, rgb, sem in dl_val:
                xyz, rgb, sem = xyz.cuda(), rgb.cuda(), sem.cuda()
                sem[sem < 0] = -100
                with torch.cuda.amp.autocast(enabled=True):
                    pred = model(xyz, rgb)
                    vl += crit(pred.reshape(-1, 2), sem.reshape(-1)).item()
        vl /= max(len(dl_val), 1)

        gpu_mem = torch.cuda.max_memory_allocated() / 1e9
        logger.info(f'Epoch {ep:3d}/{args.epochs} train={dt:.1f}s val_loss={vl:.4f} '
                    f'GPU峰值={gpu_mem:.1f}GB')

        if ep % 20 == 0:
            torch.save({'epoch': ep, 'model': model.state_dict()},
                       os.path.join(args.save_dir, f'ckpt_{ep}.pth'))
        if vl < best_val:
            best_val = vl
            torch.save({'epoch': ep, 'model': model.state_dict()},
                       os.path.join(args.save_dir, 'best.pth'))
            logger.info(f'  [best] val_loss={best_val:.4f} saved')

    logger.info(f'完成. best val_loss={best_val:.4f}')


if __name__ == '__main__':
    main()
