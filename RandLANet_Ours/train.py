"""RandLA-Net 训练 — 甘蓝数据集."""
import os, sys, torch, torch.nn as nn, logging, random, numpy as np, glob
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RandLANet")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class CabbageDatasetRLA(Dataset):
    """RandLA-Net 点云数据集 (非体素, 随机采样固定点数)."""
    def __init__(self, data_root, split='train', num_points=65536):
        self.files = sorted(glob.glob(os.path.join(data_root, '*.pth')))
        np.random.seed(123)
        idx = np.random.permutation(len(self.files))
        split_n = max(1, int(len(self.files) * 0.8))
        self.files = [self.files[i] for i in (idx[:split_n] if split == 'train' else idx[split_n:])]
        self.num_points = num_points
        logger.info(f"RandLA Dataset [{split}]: {len(self.files)} scans, {num_points} pts each")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=False)
        xyz, rgb, sem = data[0], data[1], data[2]
        # Handle both torch tensor and numpy
        if hasattr(xyz, 'numpy'): xyz = xyz.numpy()
        if hasattr(rgb, 'numpy'): rgb = rgb.numpy()
        if hasattr(sem, 'numpy'): sem = sem.numpy()
        sem = sem.astype(np.int64)
        xyz, rgb = xyz.astype(np.float32), rgb.astype(np.float32)

        # Random sampling or padding
        n = len(xyz)
        if n >= self.num_points:
            sel = np.random.choice(n, self.num_points, replace=False)
        else:
            sel = np.random.choice(n, self.num_points, replace=True)
        xyz = xyz[sel].astype(np.float32)
        rgb = rgb[sel].astype(np.float32)
        sem = sem[sel]

        # Center and scale
        xyz -= xyz.min(0)

        # Data augment (train only)
        if np.random.random() > 0.5:
            # Random rotation around Z
            theta = np.random.rand() * 2 * np.pi
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=np.float32)
            xyz = xyz @ R.T
        xyz += np.random.randn(*xyz.shape).astype(np.float32) * 0.005  # small jitter

        return (torch.from_numpy(xyz), torch.from_numpy(rgb),
                torch.from_numpy(sem).long())


def collate_fn(batch):
    """Stack into batch: xyz (B,N,3), rgb (B,N,3), sem (B,N)."""
    xyz, rgb, sem = zip(*batch)
    return torch.stack(xyz), torch.stack(rgb), torch.stack(sem)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/home/stevensen/Cabbage/e_data/pointgroup_format/train')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_points', type=int, default=65536)
    parser.add_argument('--save_dir', default='exp/randlanet')
    args = parser.parse_args()

    ds_train = CabbageDatasetRLA(args.data, 'train', args.num_points)
    ds_val = CabbageDatasetRLA(args.data, 'val', args.num_points)
    dl_train = DataLoader(ds_train, args.batch, True, collate_fn=collate_fn, drop_last=True)
    dl_val = DataLoader(ds_val, 1, False, collate_fn=collate_fn)

    from randlanet import RandLANet
    model = RandLANet(d_in=6, num_classes=2).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss(ignore_index=-100)
    # Class weight
    class_w = torch.tensor([0.4, 1.0], device='cuda')
    crit = nn.CrossEntropyLoss(ignore_index=-100, weight=class_w)

    os.makedirs(args.save_dir, exist_ok=True)
    logger.info(f"Training RandLA-Net: {len(ds_train)} train, {len(ds_val)} val")

    best_val = float('inf')
    for ep in range(1, args.epochs + 1):
        model.train()
        for xyz, rgb, sem in dl_train:
            xyz, rgb, sem = xyz.cuda(), rgb.cuda(), sem.cuda()
            sem[sem < 0] = -100  # background as ignore
            pred = model(xyz, rgb)
            loss = crit(pred.view(-1, 2), sem.view(-1))
            opt.zero_grad(); loss.backward(); opt.step()

        # Validation
        model.eval(); vl = 0.0
        with torch.no_grad():
            for xyz, rgb, sem in dl_val:
                xyz, rgb, sem = xyz.cuda(), rgb.cuda(), sem.cuda()
                sem[sem < 0] = -100
                pred = model(xyz, rgb)
                vl += crit(pred.view(-1, 2), sem.view(-1)).item()
        vl /= max(len(dl_val), 1)
        logger.info(f"Epoch {ep:3d} val_loss={vl:.4f}")
        if ep % 20 == 0:
            torch.save({'epoch': ep, 'model': model.state_dict()}, os.path.join(args.save_dir, f'ckpt_{ep}.pth'))
        if vl < best_val:
            best_val = vl
            torch.save({'epoch': ep, 'model': model.state_dict()}, os.path.join(args.save_dir, 'best.pth'))
    logger.info(f"Done. Best val={best_val:.4f}")


if __name__ == '__main__':
    main()
