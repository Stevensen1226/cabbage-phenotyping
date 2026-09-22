#!/usr/bin/env python3
"""
完整 pipeline 效率统一测量: 3 backbone × (语义 + watershed 聚类 + GIDM 后处理)。

统一在 RTX 5070 (pointgroup_blackwell 环境), 相同预处理, 逐文件测量:
  - t_sem:      语义 backbone 推理耗时 (ms)
  - t_cluster:  watershed_3d 聚类耗时 (ms)
  - t_gidm:     ADPV/GIDM 后处理耗时 (ms) [骨架切割 + 碎片回并]
  - gpu_mb:     语义推理 GPU 峰值显存 (MB)
  - cpu_mb:     GIDM 后处理 CPU 内存峰值 (MB)

用法:
  export LD_LIBRARY_PATH=$HOME/miniconda3/envs/pointgroup_blackwell/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
  $HOME/miniconda3/envs/pointgroup_blackwell/bin/python tools/efficiency_full_pipeline.py \
      --input evalaute_test --split test
"""

import sys
import os
import time
import argparse
import json
import gc
import tracemalloc

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
from cabbage_pheno.instance import InstanceClusterer

CHUNK = 120000


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
# 语义推理 (返回植株点 mask)
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
    from scipy.spatial import cKDTree
    N = len(xyz)
    xyz_norm = (xyz - PN_GLOBAL_MEAN) / (PN_GLOBAL_STD + 1e-8)

    def single(sub_xyz_norm, sub_rgb):
        xyz_t = torch.FloatTensor(sub_xyz_norm).unsqueeze(0).cuda()
        feat = np.concatenate([sub_xyz_norm, sub_rgb], axis=1).astype(np.float32)
        feat_t = torch.FloatTensor(feat).unsqueeze(0).permute(0, 2, 1).cuda()
        off_t = torch.tensor([len(sub_xyz_norm)], dtype=torch.int32).unsqueeze(0).cuda()
        with torch.no_grad():
            out = model({'pos': xyz_t, 'x': feat_t, 'offset': off_t})
        return out.squeeze(0).argmax(0).cpu().numpy().astype(np.int32)

    if N <= PN_GPU_MAX:
        return single(xyz_norm, rgb) == 1

    n_rounds = max(5, int(np.ceil(3.0 * N / PN_SAMPLE)))
    n_rounds = min(n_rounds, 30)
    all_prob = np.zeros((N, 2), dtype=np.float32)
    for _ in range(n_rounds):
        choice = np.random.choice(N, min(N, PN_SAMPLE), replace=False)
        sem = single(xyz_norm[choice], rgb[choice])
        tree = cKDTree(xyz[choice])
        _, nn = tree.query(xyz, k=1)
        all_prob[range(N), sem[nn]] += 1
        torch.cuda.empty_cache()
    return all_prob.argmax(1) == 1


# ═══════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=str, default='evalaute_test')
    ap.add_argument('--split', type=str, default='test', choices=['val', 'test', 'all'])
    ap.add_argument('--config', type=str, default='configs/default.yaml')
    ap.add_argument('--output', type=str, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding='utf-8'))

    import glob as _g
    files = sorted(_g.glob(os.path.join(args.input, '*.ply')))
    files = [f for f in files if '_gt' not in f]
    if args.split != 'all':
        sp = os.path.join(args.input, 'split.json')
        if os.path.exists(sp):
            allowed = set(json.load(open(sp))[args.split])
            files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in allowed]

    print(f'加载 3 个 backbone ... ({len(files)} 文件)')
    backbones = {
        'randla': {'model': load_randla(), 'fn': None},
        'pointgroup': {'model': load_pointgroup(), 'fn': None},
        'pointnext': {'model': load_pointnext(), 'fn': None},
    }
    # PointGroup 返回 (model, model_fn, pg_cfg, pgo) 元组
    backbones['pointgroup']['model'], backbones['pointgroup']['fn'], \
        backbones['pointgroup']['pg_cfg'], backbones['pointgroup']['pgo'] = backbones['pointgroup']['model']

    n_params = {
        'randla': sum(p.numel() for p in backbones['randla']['model'].parameters()) / 1e6,
        'pointgroup': sum(p.numel() for p in backbones['pointgroup']['model'].parameters()) / 1e6,
        'pointnext': sum(p.numel() for p in backbones['pointnext']['model'].parameters()) / 1e6,
    }

    clusterer = InstanceClusterer(cfg)

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

        for name in ['randla', 'pointgroup', 'pointnext']:
            b = backbones[name]
            # 语义推理计时 + 显存
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if name == 'randla':
                mask = randla_semantic(b['model'], xyz, rgb)
            elif name == 'pointgroup':
                mask = pointgroup_semantic(xyz, rgb, b['model'], b['fn'], b['pg_cfg'], b['pgo'])
            else:
                mask = pointnext_semantic(b['model'], xyz, rgb)
            torch.cuda.synchronize()
            t_sem = (time.perf_counter() - t0) * 1000
            gpu_mb = torch.cuda.max_memory_allocated() / 1e6

            xyz_plant = xyz[mask]
            n_plant = len(xyz_plant)

            # watershed 聚类计时
            if n_plant > 0:
                pcd_p = o3d.geometry.PointCloud()
                pcd_p.points = o3d.utility.Vector3dVector(xyz_plant)
                t0 = time.perf_counter()
                labels, _ = clusterer._cluster_watershed_3d(pcd_p)
                t_cluster = (time.perf_counter() - t0) * 1000
                labels = np.asarray(labels, dtype=np.int64) if labels is not None else None

                # GIDM 计时 + CPU 内存
                t0 = time.perf_counter()
                if labels is not None:
                    lg = clusterer._apply_skeleton_split(labels, xyz_plant)
                    lg = clusterer._merge_fragments(lg, xyz_plant)
                t_gidm = (time.perf_counter() - t0) * 1000

                gc.collect()
                tracemalloc.start()
                if labels is not None:
                    _ = clusterer._apply_skeleton_split(labels, xyz_plant)
                    _ = clusterer._merge_fragments(_, xyz_plant)
                _, cpu_peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                cpu_mb = cpu_peak / 1e6
            else:
                t_cluster = t_gidm = cpu_mb = 0.0

            row[f'{name}_sem_ms'] = t_sem
            row[f'{name}_cluster_ms'] = t_cluster
            row[f'{name}_gidm_ms'] = t_gidm
            row[f'{name}_gpu_mb'] = gpu_mb
            row[f'{name}_cpu_mb'] = cpu_mb
            row[f'{name}_n_plant'] = n_plant

        rows.append(row)
        print(f"  {row['file']}: {len(xyz):,}pts | "
              f"RLA sem {row['randla_sem_ms']:.0f}/clu {row['randla_cluster_ms']:.0f}/gidm {row['randla_gidm_ms']:.0f}ms | "
              f"PG sem {row['pointgroup_sem_ms']:.0f}/gidm {row['pointgroup_gidm_ms']:.0f}ms | "
              f"PN sem {row['pointnext_sem_ms']:.0f}/gidm {row['pointnext_gidm_ms']:.0f}ms")

    def avg(k):
        return float(np.mean([r[k] for r in rows]))

    summary = {
        'files': len(rows),
        'hardware': 'RTX 5070',
        'cluster': 'watershed_3d',
        'n_params_M': n_params,
        'backbones': {},
    }
    for name in ['randla', 'pointgroup', 'pointnext']:
        summary['backbones'][name] = {
            'sem_ms': avg(f'{name}_sem_ms'),
            'cluster_ms': avg(f'{name}_cluster_ms'),
            'gidm_ms': avg(f'{name}_gidm_ms'),
            'gpu_mb': avg(f'{name}_gpu_mb'),
            'cpu_mb': avg(f'{name}_cpu_mb'),
            'n_plant_avg': avg(f'{name}_n_plant'),
        }

    print('\n' + '=' * 78)
    print(f'完整 pipeline 效率 ({len(rows)} 文件, RTX 5070, watershed_3d)')
    print('=' * 78)
    print(f'{"backbone":<12}{"参数M":>7}{"语义ms":>9}{"聚类ms":>9}{"GIDMms":>9}{"显存MB":>9}{"CPU_MB":>9}')
    print('-' * 78)
    for name in ['randla', 'pointgroup', 'pointnext']:
        b = summary['backbones'][name]
        print(f'{name:<12}{n_params[name]:>7.2f}{b["sem_ms"]:>9.0f}{b["cluster_ms"]:>9.0f}'
              f'{b["gidm_ms"]:>9.0f}{b["gpu_mb"]:>9.0f}{b["cpu_mb"]:>9.1f}')
    print('=' * 78)

    if args.output is None:
        args.output = os.path.join(ROOT, 'output', 'efficiency_full_pipeline.json')
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump({'summary': summary, 'per_file': rows}, open(args.output, 'w'),
              indent=2, default=float)
    print(f'结果已保存: {args.output}')


if __name__ == '__main__':
    main()
