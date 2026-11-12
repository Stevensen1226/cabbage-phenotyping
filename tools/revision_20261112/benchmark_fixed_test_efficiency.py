from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / 'revision_data'
OUT = REV / 'final_v5' / 'data'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'RandLANet_Ours'))

from cabbage_pheno.io import read_point_cloud
from cabbage_pheno.instance import InstanceClusterer
from cabbage_pheno.service.pipeline import preprocess_point_cloud

SHORT_TO_MODEL = {'randla': 'RandLA-Net', 'pointnet2': 'PointNet2-RGB', 'pointnext': 'PointNeXt-L', 'pointgroup': 'PointGroup'}
MANIFEST_PATH = REV / 'final_v5' / 'configs' / 'manifest_v5.json'
CONFIG_MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding='utf-8')) if MANIFEST_PATH.exists() else {}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_randla(cfg):
    from randlanet import RandLANet
    model = RandLANet(d_in=6, num_classes=2).cuda().eval()
    ckpt = ROOT / cfg['segmentation']['randlanet_checkpoint']
    state = torch.load(ckpt, map_location='cuda', weights_only=False)
    model.load_state_dict(state['model'])
    return model


def load_pointnet2(cfg):
    from cabbage_pheno.segmentation.pointnet2_model import PointNet2SemSeg
    model = PointNet2SemSeg(2, additional_channel=3).cuda().eval()
    ckpt = ROOT / cfg['segmentation']['pointnet2_checkpoint']
    state = torch.load(ckpt, map_location='cuda', weights_only=False)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    model.load_state_dict(state)
    return model


def load_pointnext(cfg):
    from cabbage_pheno.segmentation.pointnext_adapter import PointNeXtAdapter
    return PointNeXtAdapter(cfg)


def load_pointgroup(cfg):
    from main import _load_pointgroup, _pointgroup_forward, _post_semantic_filter
    return {'bundle': _load_pointgroup(cfg), 'forward': _pointgroup_forward, 'post': _post_semantic_filter}


def semantic_mask(short, model, xyz, rgb):
    if short == 'randla':
        rgb_n = (rgb * 2.0 - 1.0).astype(np.float32)
        shifted = (xyz - xyz.min(0)).astype(np.float32)
        mask = np.zeros(len(xyz), dtype=bool)
        with torch.no_grad():
            for start in range(0, len(xyz), 120000):
                end = min(start + 120000, len(xyz))
                pred = model(torch.from_numpy(shifted[start:end]).unsqueeze(0).cuda(), torch.from_numpy(rgb_n[start:end]).unsqueeze(0).cuda())
                mask[start:end] = pred[0].max(1)[1].cpu().numpy() == 1
        return mask
    if short == 'pointnet2':
        rgb_n = (rgb * 2.0 - 1.0).astype(np.float32)
        mask = np.zeros(len(xyz), dtype=bool)
        with torch.no_grad():
            for start in range(0, len(xyz), 120000):
                end = min(start + 120000, len(xyz))
                out = model(torch.from_numpy(xyz[start:end].T).unsqueeze(0).cuda(), torch.from_numpy(rgb_n[start:end].T).unsqueeze(0).cuda())[0]
                mask[start:end] = out[0].argmax(0).cpu().numpy() == 1
        return mask
    if short == 'pointnext':
        return model.predict_mask(xyz, rgb)
    pg_model, pg_model_fn, pg_cfg, pg_seg_cfg, pgo = model['bundle']
    scores, _, _, _, _ = model['forward']((xyz - xyz.min(0)).astype(np.float32), (rgb * 2.0 - 1.0).astype(np.float32), pg_model, pg_model_fn, pg_cfg, pgo)
    mask = scores.argmax(1).cpu().numpy() == 1
    return mask

def timed_semantic(short, model, xyz, rgb, repeats=2, warmup=False):
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats + (1 if warmup else 0)):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        mask = semantic_mask(short, model, xyz, rgb)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        times.append(dt)
    if warmup:
        times = times[1:]
    return float(min(times)), mask


def timed_cluster(clusterer, pcd, points, repeats=2):
    times = []
    labels = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        labels, n = clusterer.cluster(pcd)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(min(times)), labels, int(n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', required=True, choices=list(SHORT_TO_MODEL))
    parser.add_argument('--input', default=str(REV / 'evalaute_test_v2'))
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    short = args.backbone
    model_name = SHORT_TO_MODEL[short]
    adpv_cfg_path = Path(CONFIG_MANIFEST[f'{model_name}|adpv'])
    base_cfg_path = Path(CONFIG_MANIFEST[f'{model_name}|base'])
    cfg = yaml.safe_load(adpv_cfg_path.read_text(encoding='utf-8'))
    base_cfg = yaml.safe_load(base_cfg_path.read_text(encoding='utf-8'))
    assert cfg['instance']['method'] == 'watershed_3d'
    assert base_cfg['instance']['method'] == 'watershed_3d'

    split = json.loads((Path(args.input) / 'split.json').read_text(encoding='utf-8'))
    scene_ids = split['test']
    files = [Path(args.input) / f'{s}.ply' for s in scene_ids]
    for p in files:
        assert p.exists(), p

    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.reset_peak_memory_stats()

    if short == 'randla':
        model = load_randla(cfg)
    elif short == 'pointnet2':
        model = load_pointnet2(cfg)
    elif short == 'pointnext':
        model = load_pointnext(cfg)
    else:
        model = load_pointgroup(cfg)
    base_clusterer = InstanceClusterer(base_cfg)
    adpv_clusterer = InstanceClusterer(cfg)
    rows = []
    warm = True
    for fp in files:
        pcd = read_point_cloud(str(fp))
        if pcd is None or len(pcd.points) == 0:
            continue
        t0 = time.perf_counter()
        pre = preprocess_point_cloud(pcd, cfg)
        prep_ms = (time.perf_counter() - t0) * 1000.0
        xyz = np.asarray(pre.non_ground_pcd.points, dtype=np.float32)
        rgb = np.asarray(pre.non_ground_pcd.colors, dtype=np.float32)
        if len(xyz) == 0:
            continue

        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.reset_peak_memory_stats()
        sem_ms, mask = timed_semantic(short, model, xyz, rgb, repeats=args.repeats, warmup=warm)
        warm = False
        gpu_mb = torch.cuda.max_memory_allocated() / 1e6
        plant = xyz[mask]
        if len(plant) == 0:
            continue
        import open3d as o3d
        plant_pcd = o3d.geometry.PointCloud()
        plant_pcd.points = o3d.utility.Vector3dVector(plant)

        base_ms, _, base_n = timed_cluster(base_clusterer, plant_pcd, plant, repeats=args.repeats)
        adpv_ms, _, adpv_n = timed_cluster(adpv_clusterer, plant_pcd, plant, repeats=args.repeats)

        gc.collect()
        tracemalloc.start()
        base_clusterer.cluster(plant_pcd)
        _, base_cpu_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        gc.collect()
        tracemalloc.start()
        adpv_clusterer.cluster(plant_pcd)
        _, adpv_cpu_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        overhead_ms = adpv_ms - base_ms
        total_base_ms = prep_ms + sem_ms + base_ms
        total_adpv_ms = prep_ms + sem_ms + adpv_ms
        rows.append({
            'backbone': short,
            'scene_id': fp.stem,
            'n_points': int(len(xyz)),
            'n_plant_points': int(mask.sum()),
            'base_instances': base_n,
            'adpv_instances': adpv_n,
            'preprocess_ms': prep_ms,
            'semantic_ms': sem_ms,
            'base_cluster_ms': base_ms,
            'adpv_cluster_ms': adpv_ms,
            'adpv_overhead_ms': overhead_ms,
            'total_base_ms': total_base_ms,
            'total_adpv_ms': total_adpv_ms,
            'relative_overhead_pct': 100.0 * overhead_ms / total_base_ms,
            'fps_base': 1000.0 / total_base_ms,
            'fps_adpv': 1000.0 / total_adpv_ms,
            'gpu_peak_mb': gpu_mb,
            'base_cpu_peak_mb': base_cpu_peak / 1e6,
            'adpv_cpu_peak_mb': adpv_cpu_peak / 1e6,
        })
        print(f"{fp.stem}: sem={sem_ms:.0f}ms base_clu={base_ms:.0f}ms adpv_clu={adpv_ms:.0f}ms overhead={overhead_ms:.0f}ms", flush=True)

    df = pd.DataFrame(rows)
    def mean_std(col):
        return {'mean': float(df[col].mean()), 'std': float(df[col].std(ddof=1))}
    numeric = ['preprocess_ms', 'semantic_ms', 'base_cluster_ms', 'adpv_cluster_ms', 'adpv_overhead_ms', 'total_base_ms', 'total_adpv_ms', 'relative_overhead_pct', 'fps_base', 'fps_adpv', 'gpu_peak_mb', 'base_cpu_peak_mb', 'adpv_cpu_peak_mb']
    summary = {col: mean_std(col) for col in numeric}
    summary['n_scenes'] = len(df)
    summary['backbone'] = short
    summary['model'] = model_name
    summary['config_adpv'] = str(adpv_cfg_path.relative_to(ROOT)).replace('\\', '/')
    summary['config_base'] = str(base_cfg_path.relative_to(ROOT)).replace('\\', '/')
    summary['config_adpv_sha256'] = sha256(adpv_cfg_path)
    summary['config_base_sha256'] = sha256(base_cfg_path)
    summary['checkpoint_sha256'] = sha256(ROOT / cfg['segmentation'].get(
        {'randla': 'randlanet_checkpoint', 'pointnet2': 'pointnet2_checkpoint',
         'pointnext': 'pointnext_checkpoint', 'pointgroup': 'pointgroup_pretrain'}[short], ''))
    out = Path(args.output) if args.output else OUT / f'efficiency_fixed_test_{short}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'summary': summary, 'per_scene': rows}, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('WROTE', out)
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == '__main__':
    main()


