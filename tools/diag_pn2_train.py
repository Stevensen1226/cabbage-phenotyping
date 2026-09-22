#!/usr/bin/env python3
"""PointNet2 训练诊断：验证梯度回传 + 数据增强是否正常。

在远程 /root/autodl-tmp 下运行。
"""
import sys, os, torch, numpy as np, yaml

os.chdir('/root/autodl-tmp')
sys.path.insert(0, '/root/autodl-tmp/PointNet2_Ours')
sys.path.insert(0, '/root/autodl-tmp/PointNet2_Ours/model')

# 关键：在 import util.config 之前设置 sys.argv，避免 argparse 报错
sys.argv = ['diag', '--config', '/root/autodl-tmp/PointNet2_Ours/config/pointnet2_cabbage.yaml']

from torch.utils.data import DataLoader
from dataset.cabbage_dataset import CabbageDataset, cabbage_collate_fn
from model.pointnet2_sem import PointNet2Sem, model_fn_decorator
from util.config import cfg

# 覆盖 task 为 train
cfg.task = 'train'

ds = CabbageDataset(cfg, 'train')
dl = DataLoader(ds, cfg.batch_size, True, num_workers=0, collate_fn=cabbage_collate_fn, drop_last=True)

model = PointNet2Sem(cfg).cuda()
mfn = model_fn_decorator(test=False)
opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

print(f'模型参数: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
print(f'训练集样本: {len(ds)}, batch_size: {cfg.batch_size}')

for step, batch in enumerate(dl):
    if step >= 5:
        break
    loss, _, meter, _ = mfn(batch, model, 1)
    opt.zero_grad()
    loss.backward()
    gn = sum((p.grad.norm() ** 2).item() for p in model.parameters() if p.grad is not None) ** 0.5
    opt.step()

    labels = batch['labels'].numpy()
    n_bg = int((labels == 0).sum())
    n_fg = int((labels == 1).sum())
    print(f'step {step}: loss={loss.item():.4f} grad_norm={gn:.4f} | label 背景={n_bg} 甘蓝={n_fg}')
    # 打印类别0/1的预测概率（看模型初始偏向）
    if step == 0:
        with torch.no_grad():
            c = batch['locs'].cuda(); f = batch['feats'].cuda()
            if cfg.use_coords:
                f = torch.cat((f, batch['locs_float'].cuda()), 1)
            from lib.pointgroup_ops.functions import pointgroup_ops
            vf = pointgroup_ops.voxelization(f, batch['v2p_map'].cuda(), cfg.mode)
            import spconv.pytorch as spconv
            inp = spconv.SparseConvTensor(vf, batch['voxel_locs'].cuda().int(), batch['spatial_shape'], cfg.batch_size)
            ret = model(inp, batch['p2v_map'].cuda(), batch['locs_float'].cuda(), c[:, 0].int(), batch['offsets'].cuda(), 1)
            probs = torch.softmax(ret['semantic_scores'], dim=1)
            print(f'  初始预测: 背景概率均值={probs[:,0].mean():.3f} 甘蓝概率均值={probs[:,1].mean():.3f}')

print('\n诊断完成。若 loss 能下降且 grad_norm 正常，说明梯度回传 OK，问题在数据增强或学习率。')
