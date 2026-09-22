import sys, os, torch, numpy as np
os.chdir('/root/autodl-tmp')
sys.path.insert(0, '/root/autodl-tmp/PointNet2_Ours')
sys.path.insert(0, '/root/autodl-tmp/PointNet2_Ours/model')
sys.argv = ['x', '--config', '/root/autodl-tmp/PointNet2_Ours/config/pointnet2_cabbage.yaml']

from torch.utils.data import DataLoader
from dataset.cabbage_dataset import CabbageDataset, cabbage_collate_fn
from model.pointnet2_sem import PointNet2Sem
from util.config import cfg
cfg.task = 'train'

ds = CabbageDataset(cfg, 'train')
dl = DataLoader(ds, cfg.batch_size, True, num_workers=0, collate_fn=cabbage_collate_fn, drop_last=True)
model = PointNet2Sem(cfg).cuda()

from lib.pointgroup_ops.functions import pointgroup_ops
import spconv.pytorch as spconv

for batch in dl:
    c = batch['locs'].cuda(); f = batch['feats'].cuda()
    if cfg.use_coords:
        f = torch.cat((f, batch['locs_float'].cuda()), 1)
    vf = pointgroup_ops.voxelization(f, batch['v2p_map'].cuda(), cfg.mode)
    inp = spconv.SparseConvTensor(vf, batch['voxel_locs'].cuda().int(), batch['spatial_shape'], cfg.batch_size)
    ret = model(inp, batch['p2v_map'].cuda(), batch['locs_float'].cuda(), c[:, 0].int(), batch['offsets'].cuda(), 1)
    ss = ret['semantic_scores']
    print('logits 统计: min={:.2f} max={:.2f} mean={:.2f} std={:.2f}'.format(
        ss.min().item(), ss.max().item(), ss.mean().item(), ss.std().item()))
    print('logits[:,0] 范围: [{:.2f}, {:.2f}]'.format(ss[:, 0].min().item(), ss[:, 0].max().item()))
    print('logits[:,1] 范围: [{:.2f}, {:.2f}]'.format(ss[:, 1].min().item(), ss[:, 1].max().item()))
    diff = ss[:, 0] - ss[:, 1]
    print('logit差 (ch0-ch1): mean={:.4f} std={:.4f} min={:.4f} max={:.4f}'.format(
        diff.mean().item(), diff.std().item(), diff.min().item(), diff.max().item()))
    # 检查模型各层输入输出量级
    print('\n--- 逐层输出量级 ---')
    o = model.input_conv(inp)
    print(f'input_conv 输出: mean={o.features.mean().item():.4f} std={o.features.std().item():.4f}')
    o2 = model.unet(o)
    print(f'unet 输出: mean={o2.features.mean().item():.4f} std={o2.features.std().item():.4f}')
    o3 = model.out(o2)
    print(f'out 输出: mean={o3.features.mean().item():.4f} std={o3.features.std().item():.4f}')
    of = o3.features[inp.p2v_map.long() if hasattr(inp, "p2v_map") else batch["p2v_map"].cuda().long()]
    print(f'output_feats: mean={of.mean().item():.4f} std={of.std().item():.4f}')
    break
