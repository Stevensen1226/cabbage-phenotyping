#!/usr/bin/env python3
"""Few-shot domain adaptation: cabbage RGB PointNet2 -> Crops3D_IS.

Protocol
  * validation plots are FIXED (last --val-per-crop plots per crop, sorted)
    and never used for training;
  * the support set is --k plots per crop drawn from the remaining train pool
    (seeded);
  * the official Crops3D_IS test split is never touched here - it is scored
    once, afterwards, by eval_crops3d_zeroshot.py.

Recipe is identical to the source cabbage training (4096 points per 1 m block,
mean-centred xyz, rgb in [-1,1], NLLLoss weights [1.1, 0.9]); only the data
changes, which keeps the adaptation comparison clean.
"""
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path('/root/autodl-tmp/cabbage-phenotyping')
import sys
sys.path.insert(0, str(ROOT))
from cabbage_pheno.segmentation.pointnet2_model import PointNet2SemSeg

CROPS = ('maize', 'potato', 'rapeseed')


class DS(Dataset):
    def __init__(self, files, n=4096, train=False):
        self.files = list(files); self.n = n; self.train = train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        a = np.load(self.files[i]).astype(np.float32)
        xyz = a[:, :3]; rgb = a[:, 3:6]; lab = a[:, 6].astype(np.int64)
        n = len(xyz)
        idx = np.random.choice(n, self.n, replace=True) if self.train else np.linspace(0, n - 1, self.n, dtype=np.int64)
        xyz = xyz[idx].copy(); rgb = rgb[idx].copy(); lab = lab[idx].copy()
        xyz -= xyz.mean(0, keepdims=True)
        if self.train:
            t = np.random.rand() * 2 * np.pi; c, s = np.cos(t), np.sin(t)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)
            xyz = xyz @ R.T
            xyz = xyz * np.random.uniform(.9, 1.1) + np.random.randn(*xyz.shape).astype(np.float32) * .002
        return torch.from_numpy(xyz), torch.from_numpy(rgb), torch.from_numpy(lab)


def collate(b):
    return tuple(torch.stack(x) for x in zip(*b))


def ev(model, dl, dev, crit):
    model.eval(); inter = np.zeros(2); union = np.zeros(2); ls = 0.0; correct = total = 0
    with torch.no_grad():
        for xyz, rgb, lab in dl:
            xyz = xyz.to(dev); rgb = rgb.to(dev); lab = lab.to(dev)
            out = model(xyz.permute(0, 2, 1), rgb.permute(0, 2, 1))[0]
            ls += float(crit(out, lab)); pred = out.argmax(1)
            for c in range(2):
                pi = pred == c; ti = lab == c
                inter[c] += float((pi & ti).sum()); union[c] += float((pi | ti).sum())
            correct += int((pred == lab).sum()); total += int(lab.numel())
    iou = np.divide(inter, union, out=np.full(2, np.nan), where=union > 0)
    return {'loss': ls / max(len(dl), 1), 'iou_background': float(iou[0]),
            'iou_foreground': float(iou[1]), 'miou': float(np.nanmean(iou)),
            'accuracy': correct / max(total, 1)}


def crop_of(name):
    return name.split('-')[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--blocks', default='/root/autodl-tmp/data/external/crops3d_blocks/train')
    ap.add_argument('--init', default='/root/autodl-tmp/runs/pointnet2_rgb_paper/best_model.pth')
    ap.add_argument('--out', required=True)
    ap.add_argument('--k', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--val-per-crop', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--npoints', type=int, default=4096)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    bd = Path(a.blocks)
    plots = sorted({p.name.rsplit('_block_', 1)[0] for p in bd.glob('*_block_*.npy')})
    by_crop = {c: sorted([p for p in plots if crop_of(p) == c]) for c in CROPS}
    val_plots, pool = [], []
    for c in CROPS:
        lst = by_crop[c]
        val_plots += lst[-a.val_per_crop:]
        pool.append(lst[:-a.val_per_crop])
    rng = np.random.default_rng(a.seed)
    sup_plots = []
    for c, p in zip(CROPS, pool):
        take = min(a.k, len(p))
        sup_plots += list(rng.permutation(p)[:take])   # nested: k=1 subset of k=2 subset of k=4
    def files_of(names):
        s = set(names)
        return [f for f in sorted(bd.glob('*_block_*.npy')) if f.name.rsplit('_block_', 1)[0] in s]
    trf, vaf = files_of(sup_plots), files_of(val_plots)
    print(f'k={a.k} seed={a.seed} support={sup_plots}', flush=True)
    print(f'val={val_plots}', flush=True)
    print(f'train blocks={len(trf)} val blocks={len(vaf)}', flush=True)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = 'cuda'
    tr = DS(trf, a.npoints, True); va = DS(vaf, a.npoints, False)
    tl = DataLoader(tr, a.batch, True, num_workers=a.workers, collate_fn=collate, drop_last=True)
    vl = DataLoader(va, a.batch, False, num_workers=a.workers, collate_fn=collate)

    model = PointNet2SemSeg(2, additional_channel=3).to(dev)
    ck0 = torch.load(a.init, map_location='cpu', weights_only=False)
    model.load_state_dict(ck0['model'])
    print('initialised from', a.init, 'epoch', ck0.get('epoch'), flush=True)

    crit = nn.NLLLoss(weight=torch.tensor([1.1, .9], device=dev))
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs, eta_min=1e-5)
    best = -1; stale = 0; hist = []
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); ls = 0.0
        for xyz, rgb, lab in tl:
            xyz = xyz.to(dev); rgb = rgb.to(dev); lab = lab.to(dev)
            logit = model(xyz.permute(0, 2, 1), rgb.permute(0, 2, 1))[0]
            loss = crit(logit, lab)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); ls += float(loss)
        sched.step()
        m = ev(model, vl, dev, crit); m['epoch'] = ep; m['train_loss'] = ls / max(len(tl), 1); m['seconds'] = time.time() - t0
        hist.append(m)
        print(f"ep={ep:3d} train_loss={m['train_loss']:.4f} val_loss={m['loss']:.4f} "
              f"IoU_bg={m['iou_background']:.4f} IoU_fg={m['iou_foreground']:.4f} mIoU={m['miou']:.4f} "
              f"acc={m['accuracy']:.4f} ({m['seconds']:.1f}s)", flush=True)
        state = {'epoch': ep, 'model': model.state_dict(), 'val_metrics': m, 'support': sup_plots, 'val': val_plots, 'k': a.k, 'seed': a.seed}
        torch.save(state, out / 'last.pth')
        if m['miou'] > best:
            best = m['miou']; stale = 0; torch.save(state, out / 'best_model.pth')
            (out / 'best_epoch.txt').write_text(str(ep))
        else:
            stale += 1
        if stale >= a.patience:
            print('early stop at epoch', ep, flush=True); break
    with (out / 'history.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=hist[0].keys()); w.writeheader(); w.writerows(hist)
    (out / 'summary.json').write_text(json.dumps(
        {'k': a.k, 'seed': a.seed, 'support_plots': sup_plots, 'val_plots': val_plots,
         'train_blocks': len(trf), 'val_blocks': len(vaf),
         'best_miou': best, 'best': max(hist, key=lambda x: x['miou'])}, indent=2))
    print('BEST mIoU', best, flush=True)


if __name__ == '__main__':
    main()