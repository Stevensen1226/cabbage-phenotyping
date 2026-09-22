#!/usr/bin/env python3
"""训练真正的 PointNet2 (PointNet++) 语义分割模型。

- 模型: cabbage_pheno.segmentation.pointnet2_model.PointNet2SemSeg (SA+FP, 含 rgb)
- 数据: data/pn2_blocks/train/*.npy (N,7) = xyz(3)+rgb(3)+label(1)
- 反频率 class_weight
- 数据增强: Z轴旋转 + 缩放 + 抖动
- 每块采样 npoints, 居中 (保留真实米尺度, SA 半径用米)

用法:
  python tools/train_pn2.py --data data/pn2_blocks --epochs 200 --npoints 4096
"""
import os
import sys
import glob
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cabbage_pheno.segmentation.pointnet2_model import PointNet2SemSeg

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('PointNet2')


class BlockDataset(Dataset):
    def __init__(self, root, npoints=4096, split='train'):
        self.npoints = npoints
        files = sorted(glob.glob(os.path.join(root, 'train', '*.npy')))
        # 简单 90/10 划分
        np.random.seed(42)
        idx = np.random.permutation(len(files))
        n_val = max(1, int(len(files) * 0.1))
        if split == 'train':
            self.files = [files[i] for i in idx[n_val:]]
        else:
            self.files = [files[i] for i in idx[:n_val]]
        logger.info(f'BlockDataset[{split}]: {len(self.files)} blocks')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        xyz = d[:, :3].astype(np.float32)
        rgb = d[:, 3:6].astype(np.float32)
        label = d[:, 6].astype(np.int64)

        # 采样 npoints (replace=True 处理不足)
        if len(xyz) >= self.npoints:
            choice = np.random.choice(len(xyz), self.npoints, replace=False)
        else:
            choice = np.random.choice(len(xyz), self.npoints, replace=True)
        xyz = xyz[choice]
        rgb = rgb[choice]
        label = label[choice]

        # 居中 (保留真实米尺度)
        xyz = xyz - xyz.mean(0)

        # rgb 归一化到 [-1, 1] (与训练分布匹配)
        rgb = rgb * 2.0 - 1.0

        # 转 (3, N) 和 (3, N), label (N,)
        return torch.from_numpy(xyz.T), torch.from_numpy(rgb.T), torch.from_numpy(label)


def augment(xyz):
    """xyz: (B, 3, N) 增强: Z轴旋转 + 缩放 + 抖动"""
    B, C, N = xyz.shape
    device = xyz.device
    theta = torch.rand(B, device=device) * 2 * np.pi
    cos, sin = torch.cos(theta), torch.sin(theta)
    R = torch.zeros(B, 3, 3, device=device)
    R[:, 0, 0] = cos; R[:, 0, 1] = -sin
    R[:, 1, 0] = sin; R[:, 1, 1] = cos
    R[:, 2, 2] = 1
    xyz = torch.bmm(R, xyz)
    scales = torch.rand(B, 1, 1, device=device) * 0.45 + 0.8  # 0.8~1.25
    xyz = xyz * scales
    xyz = xyz + torch.randn(B, C, N, device=device) * 0.005
    return xyz


def compute_class_weights(files, num_classes=2):
    counts = np.zeros(num_classes)
    for f in files:
        d = np.load(f)
        label = d[:, 6].astype(np.int64)
        for c in range(num_classes):
            counts[c] += (label == c).sum()
    total = counts.sum()
    w = total / (num_classes * counts + 1e-6)
    w = w / w.sum() * num_classes
    logger.info(f'类别计数: {counts}, 权重: {w}')
    return torch.from_numpy(w.astype(np.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/pn2_blocks')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--npoints', type=int, default=4096)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save', default='models/pointnet2_sem_best.pth')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device={device}')

    train_ds = BlockDataset(args.data, npoints=args.npoints, split='train')
    val_ds = BlockDataset(args.data, npoints=args.npoints, split='val')
    nw = min(16, os.cpu_count() or 4)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=nw, drop_last=True, pin_memory=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=True, persistent_workers=True)
    logger.info(f'num_workers={nw}, batch_size={args.batch_size}')

    model = PointNet2SemSeg(num_classes=2, additional_channel=3).to(device)
    logger.info(f'模型参数: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

    class_weights = compute_class_weights(train_ds.files).to(device)
    criterion = nn.NLLLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_miou = 0.0
    os.makedirs(os.path.dirname(args.save), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tl = 0.0
        for xyz, rgb, label in train_dl:
            xyz = xyz.to(device); rgb = rgb.to(device); label = label.to(device)
            xyz = augment(xyz)
            pred, _ = model(xyz, rgb)  # pred: (B, 2, N) log_softmax
            loss = criterion(pred, label)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tl += loss.item()
        tl /= len(train_dl)
        scheduler.step()

        # 验证
        model.eval()
        total_iou = []
        with torch.no_grad():
            for xyz, rgb, label in val_dl:
                xyz = xyz.to(device); rgb = rgb.to(device); label = label.to(device)
                pred, _ = model(xyz, rgb)
                pred_cls = pred.max(1)[1]
                ious = []
                for c in range(2):
                    pi = pred_cls == c; ti = label == c
                    inter = (pi & ti).sum().float()
                    union = (pi | ti).sum().float()
                    ious.append((inter / (union + 1e-8)).item())
                total_iou.append(np.nanmean(ious))
        miou = float(np.mean(total_iou))

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f'Epoch {epoch}/{args.epochs} train_loss={tl:.4f} val_mIoU={miou:.4f}')

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), args.save)
            logger.info(f'  保存 best (mIoU={miou:.4f}) -> {args.save}')

    logger.info(f'完成, best_mIoU={best_miou:.4f}')


if __name__ == '__main__':
    main()
