from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / "revision_data"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REV / "tools"))
sys.path.insert(0, str(ROOT / "RandLANet_Ours"))

from benchmark_fixed_test_efficiency import (
    load_pointgroup,
    load_pointnet2,
    load_pointnext,
    load_randla,
    sha256,
    timed_semantic,
)
from cabbage_pheno.instance import InstanceClusterer
from cabbage_pheno.io import read_point_cloud
from cabbage_pheno.service.pipeline import preprocess_point_cloud

CONFIG = REV / "final_v6_ransac/configs"
RETRAIN = CONFIG / "ransac_retrain_20261111"
OUT = REV / "final_v6_ransac/efficiency_20261111"
SCENES = ["cloudR2", "cloudR5", "cloudR9", "cloudR12", "cloudR14", "cloudR18"]
DATA_DIR = REV / "evalaute_test_v2"
MODEL_NAME = {"randla": "RandLA-Net", "pointnet2": "PointNet2-RGB", "pointnext": "PointNeXt-L", "pointgroup": "PointGroup"}
CONFIGS = {
    "randla": {
        "base": RETRAIN / "randla_net_watershed_base_retrained.yaml",
        "adpv": RETRAIN / "randla_net_watershed_adpv_retrained.yaml",
    },
    "pointnet2": {
        "base": CONFIG / "pointnet2_rgb_watershed_base_v6.yaml",
        "adpv": CONFIG / "pointnet2_rgb_watershed_adpv_v6.yaml",
    },
    "pointnext": {
        "base": RETRAIN / "pointnext_l_watershed_base_retrained.yaml",
        "adpv": RETRAIN / "pointnext_l_watershed_adpv_retrained.yaml",
    },
    "pointgroup": {
        "base": CONFIG / "pointgroup_watershed_base_v6.yaml",
        "adpv": CONFIG / "pointgroup_watershed_adpv_v6.yaml",
    },
}


def cluster_with_post(clusterer: InstanceClusterer, pcd, points: np.ndarray):
    labels, _ = clusterer.cluster(pcd)
    cfg = clusterer.inst_cfg
    discard_threshold = int(cfg.get("fragment_voting", {}).get("discard_threshold", 300))
    min_points = int(cfg.get("min_cluster_points", 2000))
    uniq, counts = np.unique(labels, return_counts=True)
    for label, count in zip(uniq, counts):
        if label >= 0 and count < discard_threshold:
            labels[labels == label] = -1
    voting = cfg.get("fragment_voting", {})
    if voting.get("enable", False):
        if getattr(clusterer, "vote_mode", "centroid") == "knn_weighted":
            labels = clusterer._reassign_fragments_knn(labels, points)
        else:
            labels = clusterer._cleanup_tiny_fragments(labels, points)
    uniq, counts = np.unique(labels, return_counts=True)
    for label, count in zip(uniq, counts):
        if label >= 0 and count < min_points:
            labels[labels == label] = -1
    valid = labels >= 0
    n_instances = int(len(np.unique(labels[valid]))) if valid.any() else 0
    return labels, n_instances


def timed_cluster_actual(clusterer: InstanceClusterer, points: np.ndarray, repeats: int):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    times = []
    labels = None
    n_instances = 0
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        labels, n_instances = cluster_with_post(clusterer, pcd, points)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(min(times)), labels, n_instances


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=list(CONFIGS))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    short = args.backbone
    OUT.mkdir(parents=True, exist_ok=True)
    adpv_path = CONFIGS[short]["adpv"]
    base_path = CONFIGS[short]["base"]
    adpv_cfg = yaml.safe_load(adpv_path.read_text(encoding="utf-8"))
    base_cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if short == "randla":
        new_ckpt = "E:\\Cabbage\\revision_data\\runs\\ransac_retrain_20261111\\randlanet\\best.pth"
        adpv_cfg["segmentation"]["randlanet_checkpoint"] = new_ckpt
        base_cfg["segmentation"]["randlanet_checkpoint"] = new_ckpt
    if short == "pointnext":
        new_ckpt = "E:\\Cabbage\\revision_data\\runs\\ransac_retrain_20261111\\pointnext_final\\best.pth"
        adpv_cfg["segmentation"]["pointnext_checkpoint"] = new_ckpt
        base_cfg["segmentation"]["pointnext_checkpoint"] = new_ckpt

    torch.manual_seed(0)
    np.random.seed(0)
    if short == "randla":
        model = load_randla(adpv_cfg)
    elif short == "pointnet2":
        model = load_pointnet2(adpv_cfg)
    elif short == "pointnext":
        model = load_pointnext(adpv_cfg)
    else:
        model = load_pointgroup(adpv_cfg)

    base_clusterer = InstanceClusterer(base_cfg)
    adpv_clusterer = InstanceClusterer(adpv_cfg)
    rows = []
    warm = True
    for scene in SCENES:
        fp = DATA_DIR / f"{scene}.ply"
        pcd = read_point_cloud(str(fp))
        t0 = time.perf_counter()
        pre = preprocess_point_cloud(pcd, adpv_cfg)
        prep_ms = (time.perf_counter() - t0) * 1000.0
        xyz = np.asarray(pre.non_ground_pcd.points, dtype=np.float32)
        rgb = np.asarray(pre.non_ground_pcd.colors, dtype=np.float32)
        torch.cuda.reset_peak_memory_stats()
        sem_ms, mask = timed_semantic(short, model, xyz, rgb, repeats=args.repeats, warmup=warm)
        warm = False
        plant_points = xyz[mask]
        if len(plant_points) == 0:
            continue
        base_ms, _, base_n = timed_cluster_actual(base_clusterer, plant_points, args.repeats)
        adpv_ms, _, adpv_n = timed_cluster_actual(adpv_clusterer, plant_points, args.repeats)
        gpu_mb = torch.cuda.max_memory_allocated() / 1e6
        overhead_ms = adpv_ms - base_ms
        total_base_ms = prep_ms + sem_ms + base_ms
        total_adpv_ms = prep_ms + sem_ms + adpv_ms
        rows.append({
            "backbone": short,
            "scene_id": scene,
            "n_points": int(len(xyz)),
            "n_plant_points": int(mask.sum()),
            "base_instances": int(base_n),
            "adpv_instances": int(adpv_n),
            "preprocess_ms": prep_ms,
            "semantic_ms": sem_ms,
            "base_cluster_ms": base_ms,
            "adpv_cluster_ms": adpv_ms,
            "adpv_overhead_ms": overhead_ms,
            "total_base_ms": total_base_ms,
            "total_adpv_ms": total_adpv_ms,
            "relative_overhead_pct": 100.0 * overhead_ms / total_base_ms,
            "fps_base": 1000.0 / total_base_ms,
            "fps_adpv": 1000.0 / total_adpv_ms,
            "gpu_peak_mb": gpu_mb,
        })
        print(scene, f"sem={sem_ms:.0f} base={base_ms:.0f} adpv={adpv_ms:.0f} overhead={overhead_ms:.0f}", flush=True)

    frame = pd.DataFrame(rows)
    numeric = [
        "preprocess_ms", "semantic_ms", "base_cluster_ms", "adpv_cluster_ms",
        "adpv_overhead_ms", "total_base_ms", "total_adpv_ms", "relative_overhead_pct",
        "fps_base", "fps_adpv", "gpu_peak_mb",
    ]
    summary = {col: {"mean": float(frame[col].mean()), "std": float(frame[col].std(ddof=1))} for col in numeric}
    summary.update({
        "n_scenes": len(frame),
        "backbone": short,
        "model": MODEL_NAME[short],
        "config_adpv": str(adpv_path.relative_to(ROOT)).replace("\\", "/"),
        "config_base": str(base_path.relative_to(ROOT)).replace("\\", "/"),
        "config_adpv_sha256": sha256(adpv_path),
        "config_base_sha256": sha256(base_path),
    })
    checkpoint_key = {"randla": "randlanet_checkpoint", "pointnet2": "pointnet2_checkpoint", "pointnext": "pointnext_checkpoint", "pointgroup": "pointgroup_pretrain"}[short]
    checkpoint = Path(adpv_cfg["segmentation"][checkpoint_key])
    summary["checkpoint_sha256"] = sha256(checkpoint if checkpoint.is_absolute() else ROOT / checkpoint)
    out = args.output or OUT / f"efficiency_fixed_test_{short}.json"
    out.write_text(json.dumps({"summary": summary, "per_scene": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("WROTE", out, flush=True)
    print(pd.DataFrame([summary]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
