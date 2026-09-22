#!/usr/bin/env python3
"""
多 backbone 效率对比: RandLA / PointGroup / PointNeXt。

统一在 RTX 5070 (pointgroup_blackwell 环境) 上, 对同一个测试集 (evalaute_test)
用相同预处理, 测每个 backbone 的语义推理耗时 + GPU 峰值显存。

用法 (必须设 LD_LIBRARY_PATH):
  export LD_LIBRARY_PATH=$HOME/miniconda3/envs/pointgroup_blackwell/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
  $HOME/miniconda3/envs/pointgroup_blackwell/bin/python tools/efficiency_backbones.py \
      --input evalaute_test --split test
"""

import sys
import os
import time
import argparse
import json

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, 'RandLANet_Ours') not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, 'RandLANet_Ours'))
if '/home/stevensen/PointNeXt-master' not in sys.path:
    sys.path.insert(0, '/home/stevensen/PointNeXt-master')

import open3d as o3d
import yaml

from cabbage_pheno.io import read_point_cloud
from cabbage_pheno.service.pipeline import preprocess_point_cloud

CHUNK = 120000  # RandLA 分块


# ═══════════════════════════════════════════════
# 模型加载
# ═══════════════════════════════════════════════
def load_randla():
    from randlanet import RandLANet
    m = RandLANet(d_in=6, num_classes=2).cuda().eval()
    sd = torch.load(os.path.join(ROOT, 'RandLANet_Ours', 'exp', 'randlanet', 'best.pth'),
                    map_location='cuda')
    m.load_state_dict(sd['model'])
    return m


def load_pointgroup():
    from main import _load_pointgroup
    cfg = {'segmentation': {'pointgroup_config': 'config/pointgroup_cabbage.yaml'}}
    model, model_fn, pg_cfg, seg_cfg, pgo = _load_pointgroup(cfg)
    return model, model_fn, pg_cfg, pgo


def load_pointnext():
    from openpoints.models import build_model_from_cfg
    from openpoints.utils import EasyConfig
    pn_cfg = EasyConfig()
    pn_cfg.load('/home/stevensen/PointNeXt-master/cfgs/cabbage.yaml', recursive=True)
    m = build_model_from_cfg(pn_cfg.model).cuda().eval()
    sd = torch.load('/home/stevensen/PointNeXt-master/best_model.pth', map_location='cpu')
    m.load_state_dict(sd, strict=False)
    return m


# ═══════════════════════════════════════════════
# 语义推理 (各自的标准范式)
# ═══════════════════════════════════════════════
def randla_semantic(model, xyz, rgb):
    rgb_norm = (rgb * 2.0 - 1.0).astype(np.float32)
    xyz_shift = (xyz - xyz.min(0)).astype(np.float32)
    N = len(xyz)
    mask = np.zeros(N, dtype=bool)
    with torch.no_grad():
        for s in range(0, N, CHUNK):
            e = min(s + CHUNK, N)
            pred = model(
                torch.from_numpy(xyz_shift[s:e]).unsqueeze(0).cuda(),
                torch.from_numpy(rgb_norm[s:e]).unsqueeze(0).cuda())
            mask[s:e] = pred[0].max(1)[1].cpu().numpy() == 1
    return mask


def pointgroup_semantic(xyz, rgb, model, model_fn, pg_cfg, pgo):
    from main import _pointgroup_forward
    xyz_shifted = (xyz - xyz.min(0)).astype(np.float32)
    rgb_norm = (rgb * 2.0 - 1.0).astype(np.float32)
    sem_scores, _, _, _, _ = _pointgroup_forward(xyz_shifted, rgb_norm, model, model_fn, pg_cfg, pgo)
    return sem_scores.max(1)[1].cpu().numpy() == 1


PN_GLOBAL_MEAN = np.array([-0.10488096, 0.18573543, 0.00078407], dtype=np.float32)
PN_GLOBAL_STD = np.array([0.8321599, 1.4525476, 0.09171805], dtype=np.float32)
PN_SAMPLE = 20480
PN_GPU_MAX = 25000


def pointnext_semantic(model, xyz, rgb):
    N = len(xyz)
    xyz_norm = (xyz - PN_GLOBAL_MEAN) / (PN_GLOBAL_STD + 1e-8)

    def single_infer(sub_xyz_norm, sub_rgb):
        xyz_t = torch.FloatTensor(sub_xyz_norm).unsqueeze(0).cuda()
        feat = np.concatenate([sub_xyz_norm, sub_rgb], axis=1).astype(np.float32)
        feat_t = torch.FloatTensor(feat).unsqueeze(0).permute(0, 2, 1).cuda()
        off_t = torch.tensor([len(sub_xyz_norm)], dtype=torch.int32).unsqueeze(0).cuda()
        with torch.no_grad():
            out = model({'pos': xyz_t, 'x': feat_t, 'offset': off_t})
        return out.squeeze(0).argmax(0).cpu().numpy().astype(np.int32)

    if N <= PN_GPU_MAX:
        return single_infer(xyz_norm, rgb) == 1

    # 大点云: 多轮采样 (与 pointnext_eval.py 一致)
    from scipy.spatial import cKDTree
    n_rounds = max(5, int(np.ceil(3.0 * N / PN_SAMPLE)))
    n_rounds = min(n_rounds, 30)
    all_prob = np.zeros((N, 2), dtype=np.float32)
    for _ in range(n_rounds):
        choice = np.random.choice(N, min(N, PN_SAMPLE), replace=False)
        sem = single_infer(xyz_norm[choice], rgb[choice])
        # 最近邻传播 (用硬标签近似概率)
        tree = cKDTree(xyz[choice])
        _, nn = tree.query(xyz, k=1)
        all_prob[range(N), sem[nn]] += 1
        torch.cuda.empty_cache()
    return all_prob.argmax(1) == 1


# ═══════════════════════════════════════════════
# 单文件计时
# ═══════════════════════════════════════════════
def measure(xyz, rgb, fn):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn(xyz, rgb)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    gpu_mb = torch.cuda.max_memory_allocated() / 1e6
    return dt, gpu_mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=str, default='evalaute_test')
    ap.add_argument('--split', type=str, default='test', choices=['val', 'test', 'all'])
    ap.add_argument('--config', type=str, default='configs/default.yaml')
    ap.add_argument('--output', type=str, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding='utf-8'))

    # 收集文件
    import glob as _g
    files = sorted(_g.glob(os.path.join(args.input, '*.ply')))
    files = [f for f in files if '_gt' not in f]
    if args.split != 'all':
        sp = os.path.join(args.input, 'split.json')
        if os.path.exists(sp):
            allowed = set(json.load(open(sp))[args.split])
            files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in allowed]

    print(f'加载 3 个 backbone ... ({len(files)} 文件)')
    models = {
        'randla': (load_randla(), None, None, None),
        'pointgroup': load_pointgroup(),
        'pointnext': (load_pointnext(), None, None, None),
    }

    # 参数量
    n_params = {
        'randla': sum(p.numel() for p in models['randla'][0].parameters()) / 1e6,
        'pointgroup': sum(p.numel() for p in models['pointgroup'][0].parameters()) / 1e6,
        'pointnext': sum(p.numel() for p in models['pointnext'][0].parameters()) / 1e6,
    }

    rows = []
    for fp in files:
        pcd = read_point_cloud(fp)
        if pcd is None or len(pcd.points) == 0:
            continue
        pre = preprocess_point_cloud(pcd, cfg)
        xyz = np.asarray(pre.non_ground_pcd.points, dtype=np.float32)
        rgb = np.asarray(pre.non_ground_pcd.colors, dtype=np.float32)
        if len(xyz) == 0:
            continue

        row = {'file': os.path.basename(fp), 'n_points': len(xyz)}

        # RandLA
        m = models['randla'][0]
        dt, gb = measure(xyz, rgb, lambda x, r: randla_semantic(m, x, r))
        row['randla_ms'] = dt * 1000
        row['randla_gpu'] = gb

        # PointGroup
        m, mfn, pgc, pgo = models['pointgroup']
        dt, gb = measure(xyz, rgb, lambda x, r: pointgroup_semantic(x, r, m, mfn, pgc, pgo))
        row['pointgroup_ms'] = dt * 1000
        row['pointgroup_gpu'] = gb

        # PointNeXt
        m = models['pointnext'][0]
        dt, gb = measure(xyz, rgb, lambda x, r: pointnext_semantic(m, x, r))
        row['pointnext_ms'] = dt * 1000
        row['pointnext_gpu'] = gb

        rows.append(row)
        print(f"  {row['file']}: {len(xyz):,}pts | "
              f"RandLA {row['randla_ms']:.0f}ms/{row['randla_gpu']:.0f}MB | "
              f"PG {row['pointgroup_ms']:.0f}ms/{row['pointgroup_gpu']:.0f}MB | "
              f"PN {row['pointnext_ms']:.0f}ms/{row['pointnext_gpu']:.0f}MB")

    # 汇总
    def avg(k):
        return float(np.mean([r[k] for r in rows]))

    summary = {
        'files': len(rows),
        'hardware': 'RTX 5070',
        'n_params_M': n_params,
        'backbones': {
            'randla': {'ms': avg('randla_ms'), 'gpu_mb': avg('randla_gpu')},
            'pointgroup': {'ms': avg('pointgroup_ms'), 'gpu_mb': avg('pointgroup_gpu')},
            'pointnext': {'ms': avg('pointnext_ms'), 'gpu_mb': avg('pointnext_gpu')},
        },
    }

    print('\n' + '=' * 70)
    print(f'多 backbone 效率对比 ({len(rows)} 文件, RTX 5070)')
    print('=' * 70)
    print(f'{"backbone":<14}{"参数(M)":>10}{"耗时(ms)":>12}{"显存(MB)":>12}')
    print('-' * 70)
    for name in ['randla', 'pointgroup', 'pointnext']:
        b = summary['backbones'][name]
        print(f'{name:<14}{n_params[name]:>10.2f}{b["ms"]:>12.1f}{b["gpu_mb"]:>12.0f}')
    print('=' * 70)

    if args.output is None:
        args.output = os.path.join(ROOT, 'output', 'efficiency_backbones.json')
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump({'summary': summary, 'per_file': rows}, open(args.output, 'w'),
              indent=2, default=float)
    print(f'结果已保存: {args.output}')


if __name__ == '__main__':
    main()
