"""
PointNet2 纯语义分割模型 (无 offset/score 分支).
与 PointGroup 对比: 去掉了实例相关的 offset 和 score 分支, 仅保留语义.
"""
import torch, torch.nn as nn, spconv.pytorch as spconv, functools
from collections import OrderedDict
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class ResidualBlock(spconv.SparseModule):
    def __init__(self, in_c, out_c, norm_fn, indice_key=None):
        super().__init__()
        if in_c == out_c:
            self.i_branch = spconv.SparseSequential(nn.Identity())
        else:
            self.i_branch = spconv.SparseSequential(spconv.SubMConv3d(in_c, out_c, 1, bias=False))
        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_c), nn.ReLU(),
            spconv.SubMConv3d(in_c, out_c, 3, padding=1, bias=False, indice_key=indice_key),
            norm_fn(out_c), nn.ReLU(),
            spconv.SubMConv3d(out_c, out_c, 3, padding=1, bias=False, indice_key=indice_key))

    def forward(self, input):
        idt = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)
        o = self.conv_branch(input)
        return o.replace_feature(o.features + self.i_branch(idt).features)


class UBlock(nn.Module):
    def __init__(self, nP, norm_fn, reps, block, kid=1):
        super().__init__()
        self.nP = nP
        self.blocks = spconv.SparseSequential(OrderedDict(
            {f'b{i}': block(nP[0], nP[0], norm_fn, f"subm{kid}") for i in range(reps)}))
        if len(nP) > 1:
            self.conv = spconv.SparseSequential(norm_fn(nP[0]), nn.ReLU(),
                spconv.SparseConv3d(nP[0], nP[1], 2, stride=2, bias=False, indice_key=f"spconv{kid}"))
            self.u = UBlock(nP[1:], norm_fn, reps, block, kid + 1)
            self.deconv = spconv.SparseSequential(norm_fn(nP[1]), nn.ReLU(),
                spconv.SparseInverseConv3d(nP[1], nP[0], 2, bias=False, indice_key=f"spconv{kid}"))
            self.tail = spconv.SparseSequential(OrderedDict(
                {f'b{i}': block(nP[0], nP[0], norm_fn, f"subm{kid}") for i in range(reps)}))

    def forward(self, x):
        o = self.blocks(x)
        if len(self.nP) > 1:
            d = self.conv(o); u = self.deconv(self.u(d))
            o = o.replace_feature(o.features + u.features); o = self.tail(o)
        return o


class PointNet2Sem(nn.Module):
    """纯语义 PointNet2, 无 offset/score 分支"""
    def __init__(self, cfg):
        super().__init__()
        ic = cfg.input_channel; m = cfg.m
        if cfg.use_coords: ic += 3
        nf = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(ic, m, 3, padding=1, bias=False, indice_key="subm1"), nf(m), nn.ReLU())
        self.unet = UBlock([m, 2*m, 3*m, 4*m, 5*m], nf, cfg.block_reps, ResidualBlock)
        self.out = spconv.SparseSequential(
            spconv.SparseConv3d(m, m, 3, padding=1, bias=False, indice_key="subm_out"), nf(m), nn.ReLU())
        self.linear = nn.Linear(m, cfg.classes)

    def forward(self, input, input_map, coords, batch_idxs, batch_offsets, epoch):
        o = self.input_conv(input); o = self.unet(o); o = self.out(o)
        return {'semantic_scores': self.linear(o.features[input_map.long()])}


def model_fn_decorator(test=False):
    from util.config import cfg
    crit = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).cuda()
    if getattr(cfg, 'class_weight', None):
        crit = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label,
            weight=torch.tensor(cfg.class_weight, dtype=torch.float32).cuda()).cuda()

    def test_fn(batch, model, epoch):
        from lib.pointgroup_ops.functions import pointgroup_ops
        c = batch['locs'].cuda(); f = batch['feats'].cuda()
        if cfg.use_coords: f = torch.cat((f, batch['locs_float'].cuda()), 1)
        vf = pointgroup_ops.voxelization(f, batch['v2p_map'].cuda(), cfg.mode)
        inp = spconv.SparseConvTensor(vf, batch['voxel_locs'].cuda().int(), batch['spatial_shape'], cfg.batch_size)
        ret = model(inp, batch['p2v_map'].cuda(), batch['locs_float'].cuda(), c[:, 0].int(), batch['offsets'].cuda(), epoch)
        return {'semantic': ret['semantic_scores'],
                'proposals': (torch.zeros(0, 2, dtype=torch.long), torch.tensor([0], dtype=torch.long)),
                'score': torch.zeros(0, 1)}

    def train_fn(batch, model, epoch):
        from lib.pointgroup_ops.functions import pointgroup_ops
        c = batch['locs'].cuda(); f = batch['feats'].cuda()
        if cfg.use_coords: f = torch.cat((f, batch['locs_float'].cuda()), 1)
        vf = pointgroup_ops.voxelization(f, batch['v2p_map'].cuda(), cfg.mode)
        inp = spconv.SparseConvTensor(vf, batch['voxel_locs'].cuda().int(), batch['spatial_shape'], cfg.batch_size)
        ret = model(inp, batch['p2v_map'].cuda(), batch['locs_float'].cuda(), c[:, 0].int(), batch['offsets'].cuda(), epoch)
        loss = crit(ret['semantic_scores'], batch['labels'].cuda())
        return loss, {}, {'semantic': loss.item()}, {}

    return test_fn if test else train_fn
