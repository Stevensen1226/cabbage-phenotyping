"""
SoftGroup-lite: PointGroup backbone + 自顶向下贪心选择 (no learned score branch).

与 PointGroup 的核心区别:
- PointGroup: BFS聚类 + learned score + NMS
- SoftGroup-lite: BFS聚类 + heuristic score + 自顶向下贪心选择
"""

import torch
import torch.nn as nn
import spconv.pytorch as spconv
import numpy as np
import functools
from collections import OrderedDict
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class ResidualBlock(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()
        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(nn.Identity())
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False))
        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels), nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
            norm_fn(out_channels), nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key))

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)
        output = self.conv_branch(input)
        output = output.replace_feature(output.features + self.i_branch(identity).features)
        return output


class UBlock(nn.Module):
    def __init__(self, nPlanes, norm_fn, block_reps, block, indice_key_id=1):
        super().__init__()
        self.nPlanes = nPlanes
        blocks = {f'block{i}': block(nPlanes[0], nPlanes[0], norm_fn, indice_key=f"subm{indice_key_id}") for i in range(block_reps)}
        self.blocks = spconv.SparseSequential(OrderedDict(blocks))
        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]), nn.ReLU(),
                spconv.SparseConv3d(nPlanes[0], nPlanes[1], kernel_size=2, stride=2, bias=False, indice_key=f"spconv{indice_key_id}"))
            self.u = UBlock(nPlanes[1:], norm_fn, block_reps, block, indice_key_id + 1)
            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]), nn.ReLU(),
                spconv.SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2, bias=False, indice_key=f"spconv{indice_key_id}"))
            tail_blocks = {f'block{i}': block(nPlanes[0], nPlanes[0], norm_fn, indice_key=f"subm{indice_key_id}") for i in range(block_reps)}
            self.blocks_tail = spconv.SparseSequential(OrderedDict(tail_blocks))

    def forward(self, x):
        output = self.blocks(x)
        if len(self.nPlanes) > 1:
            x_down = self.conv(output)
            x_up = self.deconv(self.u(x_down))
            output = output.replace_feature(output.features + x_up.features)
            output = self.blocks_tail(output)
        return output


class SoftGroup(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        input_c = cfg.input_channel
        m = cfg.m
        if cfg.use_coords: input_c += 3
        self.prepare_epochs = cfg.prepare_epochs
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(input_c, m, kernel_size=3, padding=1, bias=False, indice_key="subm1"),
            norm_fn(m), nn.ReLU())
        self.unet = UBlock([m, 2*m, 3*m, 4*m, 5*m], norm_fn, cfg.block_reps, ResidualBlock)
        self.output_layer = spconv.SparseSequential(
            spconv.SparseConv3d(m, m, kernel_size=3, padding=1, bias=False, indice_key="subm_out"),
            norm_fn(m), nn.ReLU())
        self.linear = nn.Linear(m, cfg.classes)
        self.offset = nn.Sequential(nn.Linear(m, m, bias=True), norm_fn(m), nn.ReLU(True))
        self.offset_linear = nn.Linear(m, 3)

        self.cluster_radius = cfg.cluster_radius
        self.cluster_meanActive = cfg.cluster_meanActive
        self.cluster_shift_meanActive = cfg.cluster_shift_meanActive
        self.cluster_npoint_thre = cfg.cluster_npoint_thre

    def forward(self, input, input_map, coords, batch_idxs, batch_offsets, epoch):
        import util.utils as utils
        from lib.pointgroup_ops.functions import pointgroup_ops
        ret = {}

        output = self.input_conv(input)
        output = self.unet(output)
        output = self.output_layer(output)
        output_feats = output.features[input_map.long()]

        ret['semantic_scores'] = self.linear(output_feats)
        semantic_preds = ret['semantic_scores'].max(1)[1]
        ret['pt_offsets'] = self.offset_linear(self.offset(output_feats))

        if epoch > self.prepare_epochs:
            obj = torch.nonzero(semantic_preds > 0).view(-1)
            if len(obj) == 0:
                ret['proposal_scores'] = (torch.zeros(0, 1),
                    torch.zeros(0, 2, dtype=torch.long),
                    torch.tensor([0], dtype=torch.long))
                return ret

            bidx = batch_idxs[obj]; boff = utils.get_batch_offsets(bidx, input.batch_size)
            crd = coords[obj]; off = ret['pt_offsets'][obj]
            scpu = semantic_preds[obj].int().cpu()

            ish, slsh = pointgroup_ops.ballquery_batch_p(crd + off, bidx, boff, self.cluster_radius, self.cluster_shift_meanActive)
            pish, posh = pointgroup_ops.bfs_cluster(scpu, ish.cpu(), slsh.cpu(), self.cluster_npoint_thre)
            pish[:, 1] = obj[pish[:, 1].long()].int()

            i0, sl0 = pointgroup_ops.ballquery_batch_p(crd, bidx, boff, self.cluster_radius, self.cluster_meanActive)
            pi0, po0 = pointgroup_ops.bfs_cluster(scpu, i0.cpu(), sl0.cpu(), self.cluster_npoint_thre)
            pi0[:, 1] = obj[pi0[:, 1].long()].int()

            pish[:, 0] += (po0.size(0) - 1); posh += po0[-1]
            pi = torch.cat((pi0, pish), dim=0); po = torch.cat((po0, posh[1:]))

            nP = po.shape[0] - 1
            if nP > 0:
                cnt = (po[1:] - po[:-1]).float()
                sc = torch.clamp(cnt / max(cnt.max().item(), 1), 0.0, 1.0).unsqueeze(1)
            else:
                sc = torch.zeros(0, 1)
            ret['proposal_scores'] = (sc, pi, po)

        return ret


def model_fn_decorator(test=False):
    from util.config import cfg
    semantic_criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).cuda()
    if getattr(cfg, 'class_weight', None):
        semantic_criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label,
            weight=torch.tensor(cfg.class_weight, dtype=torch.float32).cuda()).cuda()

    def test_model_fn(batch, model, epoch):
        from lib.pointgroup_ops.functions import pointgroup_ops
        coords = batch['locs'].cuda()
        feats = batch['feats'].cuda()
        if cfg.use_coords: feats = torch.cat((feats, batch['locs_float'].cuda()), 1)
        vf = pointgroup_ops.voxelization(feats, batch['v2p_map'].cuda(), cfg.mode)
        inp = spconv.SparseConvTensor(vf, batch['voxel_locs'].cuda().int(), batch['spatial_shape'], cfg.batch_size)
        ret = model(inp, batch['p2v_map'].cuda(), batch['locs_float'].cuda(), coords[:, 0].int(), batch['offsets'].cuda(), epoch)
        preds = {k: ret[k] for k in ['semantic_scores', 'pt_offsets']}
        if epoch > cfg.prepare_epochs:
            preds['score'], preds['proposals'] = ret['proposal_scores'][0], ret['proposal_scores'][1:3]
        return preds

    def model_fn(batch, model, epoch):
        from lib.pointgroup_ops.functions import pointgroup_ops
        coords = batch['locs'].cuda(); feats = batch['feats'].cuda()
        if cfg.use_coords: feats = torch.cat((feats, batch['locs_float'].cuda()), 1)
        vf = pointgroup_ops.voxelization(feats, batch['v2p_map'].cuda(), cfg.mode)
        inp = spconv.SparseConvTensor(vf, batch['voxel_locs'].cuda().int(), batch['spatial_shape'], cfg.batch_size)
        ret = model(inp, batch['p2v_map'].cuda(), batch['locs_float'].cuda(), coords[:, 0].int(), batch['offsets'].cuda(), epoch)

        sem_loss = semantic_criterion(ret['semantic_scores'], batch['labels'].cuda())

        off, c, ii, il = ret['pt_offsets'], batch['locs_float'].cuda(), batch['instance_info'].cuda(), batch['instance_labels'].cuda()
        gt_off = ii[:, 0:3] - c; diff = off - gt_off
        valid = (il != cfg.ignore_label).float()
        off_norm = (torch.sum(torch.abs(diff), -1) * valid).sum() / (valid.sum() + 1e-6)
        gt_n = torch.norm(gt_off, p=2, dim=1); off_n = torch.norm(off, p=2, dim=1)
        dir_loss = -(gt_off / (gt_n.unsqueeze(-1)+1e-8) * off / (off_n.unsqueeze(-1)+1e-8)).sum(-1)
        off_dir = (dir_loss * valid).sum() / (valid.sum() + 1e-6)

        loss = cfg.loss_weight[0] * sem_loss + cfg.loss_weight[1] * off_norm + cfg.loss_weight[2] * off_dir
        return loss, {}, {'semantic': sem_loss.item(), 'offset_norm': off_norm.item(), 'offset_dir': off_dir.item()}, {}

    return test_model_fn if test else model_fn


def softgroup_topdown_selection(proposals_idx, proposals_offset, scores_pred, N, score_thresh=0.0, overlap_thresh=0.5):
    nP = proposals_offset.shape[0] - 1
    if nP == 0: return torch.zeros(0, dtype=torch.bool)
    valid = scores_pred.view(-1) > score_thresh
    if valid.sum() == 0: return torch.zeros(nP, dtype=torch.bool)
    order = scores_pred.view(-1).argsort(descending=True)
    order = order[valid[order]]
    remain = torch.ones(N, dtype=torch.bool, device=scores_pred.device)
    selected = torch.zeros(nP, dtype=torch.bool)
    for pi in order:
        s, e = int(proposals_offset[pi]), int(proposals_offset[pi+1])
        pts = proposals_idx[s:e, 1].long()
        avail = remain[pts].sum().float()
        if avail / max(e - s, 1) > overlap_thresh:
            selected[pi] = True; remain[pts] = False
    return selected


def extract_instances_softgroup(proposals_idx, proposals_offset, scores_pred, N, cfg):
    nP = proposals_offset.shape[0] - 1
    if nP == 0: return np.zeros(N, dtype=np.int64)
    selected = softgroup_topdown_selection(proposals_idx, proposals_offset,
        scores_pred.view(-1), N,
        getattr(cfg, 'TEST_SCORE_THRESH', 0.0),
        getattr(cfg, 'topdown_overlap_thresh', 0.5))
    pm = torch.zeros((nP, N), dtype=torch.int, device=scores_pred.device)
    pm[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1
    pm = pm[selected].cpu().numpy()
    inst = np.zeros(N, dtype=np.int64)
    for i in range(len(pm)): inst[pm[i] == 1] = i + 1
    return inst
