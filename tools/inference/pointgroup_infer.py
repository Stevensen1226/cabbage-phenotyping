#!/usr/bin/env python3
"""
PointGroup 推理脚本 - 对未标注点云进行语义分割 + 实例分割
用法:
    python tools/pointgroup_infer.py --input data/unlabel/cloudR1.ply --output output/pointgroup/
    python tools/pointgroup_infer.py --input "data/unlabel/*.ply" --output output/pointgroup/
"""
import os, sys, glob, argparse
import numpy as np
import torch
import open3d as o3d

# PointGroup root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'PointGroup_Ours'))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='输入 .ply 文件或通配符')
    parser.add_argument('--output', default='output/pointgroup', help='输出目录')
    parser.add_argument('--config', default='PointGroup_Ours/config/pointgroup_cabbage.yaml', help='PointGroup 配置文件')
    parser.add_argument('--checkpoint', default=None, help='模型权重路径 (默认自动找最新)')
    parser.add_argument('--epoch', type=int, default=384, help='test_epoch')
    parser.add_argument('--score_thresh', type=float, default=0.1, help='实例得分阈值（越低检出越多）')
    parser.add_argument('--npoint_thresh', type=int, default=30, help='实例最小点数')
    parser.add_argument('--nms_thresh', type=float, default=0.5, help='NMS IoU 阈值')
    parser.add_argument('--cluster_radius', type=float, default=0.025, help='聚类半径 (体素空间)')
    return parser.parse_args()

def load_pointgroup(checkpoint_path, config_path):
    """加载 PointGroup 模型"""
    # 设置 cfg
    from util.config import cfg
    cfg.task = 'test'
    from model.pointgroup.pointgroup import PointGroup as Network
    from model.pointgroup.pointgroup import model_fn_decorator
    import util.utils as utils

    model = Network(cfg)
    model = model.cuda()
    model_fn = model_fn_decorator(test=True)

    if checkpoint_path:
        utils.checkpoint_restore(model, cfg.exp_path, 
            cfg.config.split('/')[-1][:-5], True, cfg.test_epoch,
            dist=False, f=checkpoint_path)
    else:
        utils.checkpoint_restore(model, cfg.exp_path,
            cfg.config.split('/')[-1][:-5], True, cfg.test_epoch)

    return model, model_fn, cfg

def ply_to_pointgroup_input(ply_path, cfg):
    """将 .ply 转为 PointGroup 推理需要的 batch dict"""
    from lib.pointgroup_ops.functions import pointgroup_ops

    pcd = o3d.io.read_point_cloud(ply_path)
    xyz = np.asarray(pcd.points, dtype=np.float32)
    if len(pcd.colors) > 0:
        rgb = (np.asarray(pcd.colors) * 2.0 - 1.0).astype(np.float32)
    else:
        rgb = np.zeros_like(xyz)

    # Center and scale
    xyz_middle = xyz - xyz.min(0)
    xyz_scaled = xyz_middle * cfg.scale

    # Voxelization (like scannet's testMerge)
    locs = torch.cat([
        torch.LongTensor(xyz_scaled.shape[0], 1).fill_(0),
        torch.from_numpy(xyz_scaled).long()
    ], 1)
    locs_float = torch.from_numpy(xyz_middle)
    feats = torch.from_numpy(rgb)

    batch_offsets = torch.tensor([0, xyz_scaled.shape[0]], dtype=torch.int)
    spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), cfg.full_scale[0], None)

    voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, 1, cfg.mode)

    batch = {
        'locs': locs,
        'voxel_locs': voxel_locs,
        'p2v_map': p2v_map,
        'v2p_map': v2p_map,
        'locs_float': locs_float,
        'feats': feats,
        'offsets': batch_offsets,
        'spatial_shape': spatial_shape,
        'id': torch.tensor([0]),
        'scene_name': os.path.basename(ply_path)
    }
    return batch, xyz_middle, pcd

def run_inference(batch, model, model_fn, cfg):
    """运行推理，返回语义和实例结果"""
    with torch.no_grad():
        model.eval()
        preds = model_fn(batch, model, cfg.test_epoch)

    semantic_scores = preds['semantic']  # (N, 2)
    semantic_pred = semantic_scores.max(1)[1]  # (N,)

    pt_offsets = preds['pt_offsets'] if 'pt_offsets' in preds else None

    if 'proposals' in preds:
        proposals_idx, proposals_offset = preds['proposals']
        scores = preds['score'] if 'score' in preds else None
    elif 'proposal_scores' in preds:
        scores, proposals_idx, proposals_offset = preds['proposal_scores']
    else:
        scores = proposals_idx = proposals_offset = None

    return semantic_pred, pt_offsets, scores, proposals_idx, proposals_offset

def get_instance_labels(semantic_pred, scores, proposals_idx, proposals_offset,
                        N, score_thresh=0.1, npoint_thresh=30, nms_thresh=0.5):
    """后处理：NMS + 阈值过滤得到实例标签"""
    if proposals_idx is None or scores is None:
        return None, None

    scores_pred = torch.sigmoid(scores.view(-1))
    nProposal = proposals_offset.shape[0] - 1

    # 构建 proposals mask
    proposals_pred = torch.zeros((nProposal, N), dtype=torch.int, device=scores_pred.device)
    proposals_pred[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1

    # 得分过滤
    score_mask = scores_pred > score_thresh
    scores_pred = scores_pred[score_mask]
    proposals_pred = proposals_pred[score_mask]

    if proposals_pred.shape[0] == 0:
        return np.zeros(N, dtype=np.int64), None

    # 点数过滤
    proposals_pointnum = proposals_pred.sum(1)
    npoint_mask = proposals_pointnum > npoint_thresh
    proposals_pred = proposals_pred[npoint_mask]
    scores_pred = scores_pred[npoint_mask]

    if proposals_pred.shape[0] == 0:
        return np.zeros(N, dtype=np.int64), None

    # NMS
    proposals_f = proposals_pred.float()
    intersection = torch.mm(proposals_f, proposals_f.t())
    proposals_pn = proposals_f.sum(1)
    proposals_pn_h = proposals_pn.unsqueeze(-1).repeat(1, proposals_pn.shape[0])
    proposals_pn_v = proposals_pn.unsqueeze(0).repeat(proposals_pn.shape[0], 1)
    cross_ious = intersection / (proposals_pn_h + proposals_pn_v - intersection)

    # 简易 NMS
    ixs = scores_pred.cpu().numpy().argsort()[::-1]
    pick = []
    ious_np = cross_ious.cpu().numpy()
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        iou = ious_np[i, ixs[1:]]
        remove_ixs = np.where(iou > nms_thresh)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)

    pick = np.array(pick, dtype=np.int32)
    clusters = proposals_pred[pick].cpu().numpy()  # (nCluster, N)

    # 分配实例标签
    inst_label = np.zeros(N, dtype=np.int64)
    for i in range(clusters.shape[0]):
        inst_label[clusters[i] == 1] = i + 1  # 1-based

    return inst_label, clusters

def save_results(xyz, rgb, semantic_pred, inst_label, output_dir, scene_name, pcd_original):
    """保存可视化结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. 语义分割结果 (绿色=甘蓝, 灰色=背景)
    sem_colors = np.zeros((xyz.shape[0], 3))
    sem_colors[semantic_pred == 0] = [0.5, 0.5, 0.5]  # 灰色=背景
    sem_colors[semantic_pred == 1] = [0.0, 1.0, 0.0]  # 绿色=甘蓝

    sem_pcd = o3d.geometry.PointCloud()
    sem_pcd.points = o3d.utility.Vector3dVector(xyz)
    sem_pcd.colors = o3d.utility.Vector3dVector(sem_colors)
    o3d.io.write_point_cloud(os.path.join(output_dir, f'{scene_name}_semantic.ply'), sem_pcd)

    # 2. 实例分割结果 (不同颜色标记不同株)
    if inst_label is not None and inst_label.max() > 0:
        np.random.seed(42)
        n_inst = inst_label.max()
        palette = np.random.rand(n_inst + 1, 3)
        palette[0] = [0.5, 0.5, 0.5]  # 背景=灰色

        inst_colors = palette[inst_label]
        inst_pcd = o3d.geometry.PointCloud()
        inst_pcd.points = o3d.utility.Vector3dVector(xyz)
        inst_pcd.colors = o3d.utility.Vector3dVector(inst_colors)
        o3d.io.write_point_cloud(os.path.join(output_dir, f'{scene_name}_instance.ply'), inst_pcd)

        print(f"  实例数: {n_inst}")
    else:
        print("  未检测到有效实例")

    print(f"  语义结果: {output_dir}/{scene_name}_semantic.ply")
    print(f"  实例结果: {output_dir}/{scene_name}_instance.ply")


def main():
    args = parse_args()

    # 获取输入文件列表
    if '*' in args.input:
        input_files = sorted(glob.glob(args.input))
    else:
        input_files = [args.input]

    if not input_files:
        print(f"未找到输入文件: {args.input}")
        return

    print(f"找到 {len(input_files)} 个文件待处理")

    # 加载模型
    print("正在加载 PointGroup 模型...")
    if args.config.startswith('PointGroup_Ours/'):
        config_path = os.path.join(os.path.dirname(__file__), '..', args.config)
    else:
        config_path = args.config
    sys.argv = ['infer', '--config', config_path]
    if args.checkpoint:
        sys.argv += ['--pretrain', args.checkpoint]

    # 手动设置 cfg + override cluster_radius
    from util.config import cfg
    cfg.task = 'test'
    if args.cluster_radius:
        cfg.cluster_radius = args.cluster_radius
        print(f"cluster_radius override: {args.cluster_radius}")

    from model.pointgroup.pointgroup import PointGroup as Network
    from model.pointgroup.pointgroup import model_fn_decorator
    import util.utils as utils
    from lib.pointgroup_ops.functions import pointgroup_ops

    model = Network(cfg)
    model = model.cuda()
    print(f"模型参数: {sum([x.nelement() for x in model.parameters()]):,}")

    # 加载权重
    if args.checkpoint:
        start_epoch = utils.checkpoint_restore(model, cfg.exp_path,
            cfg.config.split('/')[-1][:-5], True, 0, dist=False, f=args.checkpoint)
        cfg.test_epoch = start_epoch - 1
    else:
        start_epoch = utils.checkpoint_restore(model, cfg.exp_path,
            cfg.config.split('/')[-1][:-5], True, cfg.test_epoch)
    print(f"加载模型: epoch {start_epoch - 1}")

    model_fn = model_fn_decorator(test=True)

    # 逐文件推理
    for ply_path in input_files:
        scene_name = os.path.splitext(os.path.basename(ply_path))[0]
        print(f"\n{'='*50}")
        print(f"处理: {scene_name}")

        # 准备输入
        batch, xyz_middle, pcd = ply_to_pointgroup_input(ply_path, cfg)
        N = xyz_middle.shape[0]
        print(f"  点数: {N:,}")

        # 推理
        semantic_pred, pt_offsets, scores, proposals_idx, proposals_offset = \
            run_inference(batch, model, model_fn, cfg)

        # 获取实例
        inst_label, clusters = get_instance_labels(
            semantic_pred, scores, proposals_idx, proposals_offset,
            N, args.score_thresh, args.npoint_thresh, args.nms_thresh)

        # 统计语义结果
        n_cabbage = (semantic_pred == 1).sum().item()
        n_bg = (semantic_pred == 0).sum().item()
        print(f"  语义: 甘蓝={n_cabbage:,} 点, 背景={n_bg:,} 点")

        # 保存
        save_results(xyz_middle, None, semantic_pred.cpu().numpy(),
                     inst_label, args.output, scene_name, pcd)

    print(f"\n全部完成! 结果保存在: {args.output}/")


if __name__ == '__main__':
    main()
