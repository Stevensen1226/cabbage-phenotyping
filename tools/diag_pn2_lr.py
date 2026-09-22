#!/usr/bin/env python3
"""PointNet2 快速收敛验证：对比不同 lr 下 loss 能否跌破随机平台 0.693。"""
import sys, os, torch, numpy as np, time

os.chdir('/root/autodl-tmp')
sys.path.insert(0, '/root/autodl-tmp/PointNet2_Ours')
sys.path.insert(0, '/root/autodl-tmp/PointNet2_Ours/model')
sys.argv = ['x', '--config', '/root/autodl-tmp/PointNet2_Ours/config/pointnet2_cabbage.yaml']

from torch.utils.data import DataLoader
from dataset.cabbage_dataset import CabbageDataset, cabbage_collate_fn
from model.pointnet2_sem import PointNet2Sem, model_fn_decorator
from util.config import cfg
cfg.task = 'train'

ds_train = CabbageDataset(cfg, 'train')
ds_val = CabbageDataset(cfg, 'val')
dl_train = DataLoader(ds_train, cfg.batch_size, True, num_workers=4, collate_fn=cabbage_collate_fn, drop_last=True)
dl_val = DataLoader(ds_val, 1, False, num_workers=4, collate_fn=cabbage_collate_fn)

def train_with_lr(lr, epochs=60):
    model = PointNet2Sem(cfg).cuda()
    mfn = model_fn_decorator(test=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    t0 = time.time()
    first_loss = None
    last_loss = None
    for ep in range(1, epochs + 1):
        model.train()
        for batch in dl_train:
            loss, _, _, _ = mfn(batch, model, ep)
            opt.zero_grad(); loss.backward(); opt.step()
            if first_loss is None:
                first_loss = loss.item()
        # val
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for batch in dl_val:
                loss, _, _, _ = mfn(batch, model, ep)
                vl += loss.item()
        vl /= max(len(dl_val), 1)
        last_loss = vl
        if ep % 10 == 0 or ep == 1:
            print(f'  lr={lr}: ep {ep:3d}/{epochs} val_loss={vl:.4f}', flush=True)
    return first_loss, last_loss, time.time() - t0

for lr in [0.01, 0.005, 0.002]:
    print(f'=== 测试 lr={lr} ===', flush=True)
    f, l, dt = train_with_lr(lr, epochs=60)
    print(f'  -> first={f:.4f} last={l:.4f} 耗时={dt:.0f}s\n', flush=True)
