#!/usr/bin/env python3
"""PointNeXt 分块推理评估脚本 v2

用法:
  # 默认配置 (含 PCA 骨架拆分)
  /home/stevensen/miniconda3/envs/pointgroup_blackwell/bin/python tools/pointnext_eval.py

  # 消融实验 (无 PCA)
  /home/stevensen/miniconda3/envs/pointgroup_blackwell/bin/python tools/pointnext_eval.py --config configs/ablation_no_pca.yaml
"""
import sys, os, time, json, glob, warnings, argparse
import numpy as np

os.chdir('/home/stevensen/Cabbage')
sys.path.insert(0, '/home/stevensen/Cabbage')
sys.path.insert(0, '/home/stevensen/PointNeXt-master')
warnings.filterwarnings('ignore')

import torch, open3d as o3d
from scipy.spatial import cKDTree

from openpoints.models import build_model_from_cfg
from openpoints.utils import EasyConfig
from cabbage_pheno.io import read_point_cloud
from cabbage_pheno.service.pipeline import preprocess_point_cloud
from cabbage_pheno.service.evaluation import (
    load_config, load_ground_truth, align_gt_to_pred,
    evaluate_from_predictions,
)
from cabbage_pheno.instance import InstanceClusterer
from RandLANet_Ours.randlanet import RandLANet

# ═══════════════════════════════════════════════
# 模型加载
# ═══════════════════════════════════════════════
print('Loading semantic backbone...')

# model will be loaded after reading eval config so we can choose backbone
model = None
BACKBONE = 'pointnext'



# ═══════════════════════════════════════════════
# 单次推理
# ═══════════════════════════════════════════════
# 训练时归一化统计量 (来自 /home/stevensen/PointNeXt-master/train/)
GLOBAL_MEAN = np.array([-0.10488096, 0.18573543, 0.00078407], dtype=np.float32)
GLOBAL_STD  = np.array([0.8321599,  1.4525476,  0.09171805], dtype=np.float32)


def _single_infer(xyz, rgb):
    """xyz=(N,3), rgb=(N,3) → sem_pred=(N,) {0,1}
    归一化方式与 train_custom.py 完全一致: (xyz - GLOBAL_MEAN) / GLOBAL_STD"""
    N = len(xyz)
    # 全局归一化 (匹配训练)
    xyz_norm = (xyz - GLOBAL_MEAN) / (GLOBAL_STD + 1e-8)
    if BACKBONE.startswith('randla'):
        # RandLA-Net训练归一化: xyz -= xyz.min(0), rgb in [-1, 1]
        xyz_norm = xyz - xyz.min(0, keepdims=True)
        xyz_t = torch.FloatTensor(xyz_norm).unsqueeze(0).cuda()
        rgb_norm = rgb.astype(np.float32) * 2.0 - 1.0  # [0,1] → [-1,1]
        feat_t = torch.FloatTensor(rgb_norm).unsqueeze(0).cuda()
        with torch.no_grad():
            out = model(xyz_t, feat_t)  # (1,N,classes)
        sem = out.squeeze(0).argmax(-1).cpu().numpy().astype(np.int32)
        del xyz_t, feat_t, out
        return sem

    # Default: PointNeXt style
    xyz_t = torch.FloatTensor(xyz_norm).unsqueeze(0).cuda()
    # 特征: [xyz_norm, rgb] (6 通道, 匹配模型 in_channels=6)
    feat = np.concatenate([xyz_norm, rgb], axis=1).astype(np.float32)
    feat_t = torch.FloatTensor(feat).unsqueeze(0).permute(0, 2, 1).cuda()
    off_t = torch.tensor([N], dtype=torch.int32).unsqueeze(0).cuda()
    with torch.no_grad():
        out = model({'pos': xyz_t, 'x': feat_t, 'offset': off_t})
    sem = out.squeeze(0).argmax(0).cpu().numpy().astype(np.int32)
    del xyz_t, feat_t, off_t, out
    return sem


def _single_infer_prob(xyz, rgb):
    """返回 softmax 概率: (N, 2)"""
    N = len(xyz)
    xyz_norm = (xyz - GLOBAL_MEAN) / (GLOBAL_STD + 1e-8)
    if BACKBONE.startswith('randla'):
        # RandLA-Net训练归一化: xyz -= xyz.min(0), rgb in [-1, 1]
        xyz_norm = xyz - xyz.min(0, keepdims=True)
        xyz_t = torch.FloatTensor(xyz_norm).unsqueeze(0).cuda()
        rgb_norm = rgb.astype(np.float32) * 2.0 - 1.0  # [0,1] → [-1,1]
        feat_t = torch.FloatTensor(rgb_norm).unsqueeze(0).cuda()
        with torch.no_grad():
            out = model(xyz_t, feat_t)  # (1,N,classes)
        prob = torch.softmax(out.squeeze(0), dim=-1).cpu().numpy()  # (N, C)
        del xyz_t, feat_t, out
        return prob

    xyz_t = torch.FloatTensor(xyz_norm).unsqueeze(0).cuda()
    feat = np.concatenate([xyz_norm, rgb], axis=1).astype(np.float32)
    feat_t = torch.FloatTensor(feat).unsqueeze(0).permute(0, 2, 1).cuda()
    off_t = torch.tensor([N], dtype=torch.int32).unsqueeze(0).cuda()
    with torch.no_grad():
        out = model({'pos': xyz_t, 'x': feat_t, 'offset': off_t})
    prob = torch.softmax(out.squeeze(0), dim=0).cpu().numpy().T  # (N, 2)
    del xyz_t, feat_t, off_t, out
    return prob


# ═══════════════════════════════════════════════
# 多轮随机采样推理 (匹配训练分布)
# ═══════════════════════════════════════════════
def block_inference(points_xyz, points_rgb, block_size, overlap):
    """多轮随机采样 + 概率累加 → 匹配训练时 20480 点采样的分布。
    每轮: 随机采 20480 点 → 推理 → 最近邻传播到全点云
    多轮概率累加后 argmax 得到最终语义标签。"""
    N = len(points_xyz)
    SAMPLE_SIZE = 20480   # 匹配训练时的 num_points
    GPU_MAX = 25000       # GPU 安全上限

    # 小点云: 单次推理
    if N <= GPU_MAX:
        return _single_infer(points_xyz, points_rgb)

    # 计算覆盖轮数: 期望每点被采到 ≥ 3 次
    n_rounds = max(5, int(np.ceil(3.0 * N / SAMPLE_SIZE)))
    n_rounds = min(n_rounds, 30)  # 最多 30 轮避免过慢
    print(f'  [{N} pts → {n_rounds} rounds]', end='', flush=True)

    all_prob = np.zeros((N, 2), dtype=np.float32)
    for r in range(n_rounds):
        choice = np.random.choice(N, min(N, SAMPLE_SIZE), replace=False)
        prob_sub = _single_infer_prob(points_xyz[choice], points_rgb[choice])
        # 最近邻传播
        tree = cKDTree(points_xyz[choice])
        _, nn = tree.query(points_xyz, k=1)
        all_prob += prob_sub[nn]
        if (r + 1) % 5 == 0:
            torch.cuda.empty_cache()

    return all_prob.argmax(1).astype(np.int32)


# ═══════════════════════════════════════════════
# 主评估循环
# ═══════════════════════════════════════════════
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/default.yaml', help='配置文件路径')
parser.add_argument('--output', default=None, help='输出 JSON 路径 (默认自动生成)')
args = parser.parse_args()

cfg = load_config(args.config)
print(f'Config: {args.config}')
np.random.seed(cfg.get('pipeline', {}).get('seed', 42))
files = sorted(glob.glob('e_data/train/*.ply'))
files = [f for f in files if '_gt' not in os.path.basename(f)]
print(f'Total files: {len(files)}')

blk_cfg = cfg.get('segmentation', {}).get('block_inference', {})
BLOCK_SIZE = blk_cfg.get('block_size', 2.0)
OVERLAP = blk_cfg.get('overlap', 0.5)

# Load semantic backbone
backbone = cfg.get('segmentation', {}).get('backbone', 'pointnet2')
BACKBONE = backbone.lower()
if BACKBONE.startswith('randla'):
    print('  Loading RandLA-Net...')
    model = RandLANet(d_in=6, num_classes=2).cuda().eval()
    # try common checkpoints
    ck = None
    for p in sorted(glob.glob('RandLANet_Ours/exp/randlanet/*.pth')):
        if 'best' in p or 'ckpt' in p:
            ck = p
            break
    if ck:
        sd = torch.load(ck, map_location='cpu')
        if isinstance(sd, dict) and 'model' in sd:
            model.load_state_dict(sd['model'], strict=False)
        else:
            try:
                model.load_state_dict(sd, strict=False)
            except Exception:
                pass
    print('  RandLA-Net loaded')
else:
    print('  Loading PointNeXt model...')
    pn_cfg = EasyConfig()
    pn_cfg.load('/home/stevensen/PointNeXt-master/cfgs/cabbage.yaml', recursive=True)
    model = build_model_from_cfg(pn_cfg.model).cuda().eval()
    sd = torch.load('/home/stevensen/PointNeXt-master/best_model.pth', map_location='cpu')
    model.load_state_dict(sd, strict=False)
    print(f'  PointNeXt-L: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params')

clusterer = InstanceClusterer(cfg)
recs = []

for idx, fp in enumerate(files):
    nm = os.path.splitext(os.path.basename(fp))[0]
    print(f'\n[{idx+1}/{len(files)}] {nm}', flush=True)
    t0 = time.time()

    # ── Open3D 预处理 (与 evaluate.py/默认管道一致) ──
    pcd = read_point_cloud(fp)
    if pcd is None:
        continue
    pr = preprocess_point_cloud(pcd, cfg)
    xyz_clean = pr.points_clean
    ng_pcd = pr.non_ground_pcd
    xyz_ng = np.asarray(ng_pcd.points, np.float32)
    rgb_ng = np.asarray(ng_pcd.colors, np.float32) if len(ng_pcd.colors) > 0 \
             else np.ones((len(xyz_ng), 3), np.float32) * 0.5
    n_all = len(xyz_clean)
    n_ng = len(xyz_ng)
    print(f'  Points: {n_all} total → {n_ng} non-ground', flush=True)

    # ── GT ──
    gt_pts, gt_sem, gt_inst = load_ground_truth(fp, -2, -1, True)

    if n_ng < 32:
        rt = time.time() - t0
        recs.append({
            'fname': nm, 'pred_sem': np.zeros(n_all, dtype=int),
            'pred_inst': np.full(n_all, -1, dtype=int),
            'gt_sem': np.zeros(0), 'gt_inst': np.zeros(0), 'runtime': rt,
        })
        continue

    # ── 分块语义推理 ──
    t_sem = time.time()
    sem_ng = block_inference(xyz_ng, rgb_ng, BLOCK_SIZE, OVERLAP)
    t_sem = time.time() - t_sem
    n_cab = int(sem_ng.sum())
    print(f'  Sem: {n_cab} cabbage / {n_ng} ({t_sem:.1f}s)', flush=True)

    # ── 映射回全点云 ──
    full_sem = np.zeros(n_all, dtype=int)
    full_inst = np.full(n_all, -1, dtype=int)
    cab_mask_ng = sem_ng == 1
    cab_idx_ng = np.where(cab_mask_ng)[0]
    if n_all > 0 and n_ng > 0:
        t_ng = cKDTree(xyz_ng)
        _, ng2all = t_ng.query(xyz_clean, k=1)
        full_sem = sem_ng[ng2all]

    # ── 聚类 ──
    if n_cab >= 50:
        cab_xyz = xyz_ng[cab_mask_ng]
        cab_pcd = o3d.geometry.PointCloud()
        cab_pcd.points = o3d.utility.Vector3dVector(cab_xyz)
        try:
            inst_ng, _ = clusterer.cluster(cab_pcd)
        except Exception as e:
            print(f'  Clustering error: {e}', flush=True)
            inst_ng = np.full(n_cab, -1, dtype=np.int64)

        # ── 与原始 evaluate.py 完全一致的后处理 ──
        # 0) _merge_fragments (原始 evaluate.py 在 cluster 之后显式调用了此步)
        try:
            inst_ng = clusterer._merge_fragments(inst_ng.copy(), cab_xyz)
        except Exception:
            pass

        # 1) 丢弃点数 < discard_threshold 的碎片
        discard_thresh = cfg.get('instance', {}).get('fragment_voting', {}).get('discard_threshold', 300)
        min_pts = cfg.get('instance', {}).get('min_cluster_points', 2000)
        merge_thresh = cfg.get('instance', {}).get('fragment_voting', {}).get('merge_threshold', 1500)

        uniq, cnts = np.unique(inst_ng, return_counts=True)
        for lbl, c in zip(uniq, cnts):
            if lbl >= 0 and c < discard_thresh:
                inst_ng[inst_ng == lbl] = -1

        # 2) _cleanup_tiny_fragments (匹配原始 evaluate.py)
        if cfg.get('instance', {}).get('fragment_voting', {}).get('enable', False):
            try:
                inst_ng = clusterer._cleanup_tiny_fragments(inst_ng, cab_xyz)
            except Exception:
                pass

        # 3) 丢弃点数 < min_cluster_points 的簇
        uniq, cnts = np.unique(inst_ng, return_counts=True)
        for lbl, c in zip(uniq, cnts):
            if lbl >= 0 and c < min_pts:
                inst_ng[inst_ng == lbl] = -1

        # 4) 噪声重分配: 将 label=-1 的点分配给最近的有效簇
        noise_mask = inst_ng == -1
        if noise_mask.sum() > 0 and (inst_ng >= 0).sum() > 0:
            valid_mask = inst_ng >= 0
            tree_ng = cKDTree(cab_xyz[valid_mask])
            _, nn_idx = tree_ng.query(cab_xyz[noise_mask], k=1)
            inst_ng[noise_mask] = inst_ng[valid_mask][nn_idx]

        # 5) 重编号为 1..N (匹配原始 evaluate.py 和 get_instance_matches 的 >0 过滤)
        valid = inst_ng >= 0
        if valid.sum() > 0:
            new_labels = np.full_like(inst_ng, -1)
            next_id = 1
            for old_lbl in sorted(set(inst_ng[valid].tolist())):
                new_labels[inst_ng == old_lbl] = next_id
                next_id += 1
            inst_ng = new_labels

        valid_inst = inst_ng > 0
        if valid_inst.sum() > 0 and n_all > 0:
            cab2all = ng2all[cab_idx_ng]
            full_inst[cab2all[valid_inst]] = inst_ng[valid_inst]

    n_inst = len(np.unique(full_inst[full_inst > 0]))
    rt = time.time() - t0
    print(f'  → {n_inst} instances | {rt:.1f}s total', flush=True)

    # ── Align GT ──
    gs, gi = align_gt_to_pred(gt_pts, gt_sem, gt_inst, xyz_clean)
    if gi is not None:
        gi = gi.copy(); gi[gi <= 0] = 0

    recs.append({
        'fname': nm,
        'pred_sem': full_sem,
        'pred_inst': full_inst,
        'gt_sem': gs,
        'gt_inst': gi,
        'runtime': rt,
    })
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════
# 评估 & 输出
# ═══════════════════════════════════════════════
pca_enabled = cfg.get('instance', {}).get('pca_split', {}).get('enable', False)
method_label = cfg.get('instance', {}).get('method', 'watershed_3d')
if args.output:
    out_json = args.output
    label = os.path.splitext(os.path.basename(args.output))[0]
else:
    pca_suffix = '_pca' if pca_enabled else '_nopca'
    out_name = f'pointnext_{method_label}{pca_suffix}'
    out_json = f'output/{out_name}.json'
    pca_tag = "+PCA" if pca_enabled else ""
    label = f'PointNeXt-L ({method_label}{pca_tag})'

print('\n' + '=' * 60)
print(f'Evaluating... ({label})')
res = evaluate_from_predictions(recs, iou_thresh=0.5, label=label)
os.makedirs('output', exist_ok=True)
with open(out_json, 'w') as f:
    json.dump(res, f, ensure_ascii=False, indent=2)

print('=' * 60)
print(f'=== {label} ===')
print('=' * 60)
s = res['summary']
for k, v in sorted(s.items()):
    print(f'  {k}: {v}')

# Per-file
HDR = f'\n{"File":<12} {"Sem mIoU":>10} {"Inst Prec":>10} {"Inst Rec":>10} {"Inst F1":>10}'
print(HDR)
print('-' * 54)
for r in res['per_file']:
    fname, sm, ip, ir, fi = r["file"], r["sem_miou"], r["inst_prec"], r["inst_rec"], r["inst_f1"]
    print(f'{fname:<12} {sm:>10.4f} {ip:>10.4f} {ir:>10.4f} {fi:>10.4f}')
print('-' * 54)
AVG = "AVERAGE"
asm, aip, air, afi = s["avg_sem_miou"], s["avg_inst_prec"], s["avg_inst_rec"], s["avg_inst_f1"]
print(f'{AVG:<12} {asm:>10.4f} {aip:>10.4f} {air:>10.4f} {afi:>10.4f}')
print('=' * 60)
print(f'结果 → {out_json}')
