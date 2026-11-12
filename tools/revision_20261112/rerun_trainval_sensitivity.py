from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / "revision_data"
sys.path.insert(0, str(REV))
sys.path.insert(0, str(REV / "tools"))

from tools.density_sensitivity import InstanceClusterer, find_sticky_clusters
from tools.multi_cluster_validation import (
    DENSITY_N_TRIALS,
    HYPER_N_TRIALS,
    MIN_PEAK_DIST_GRID,
    RATIOS,
    VALLEY_DEPTH_GRID,
    run_density_sensitivity,
    run_hyperparam_sensitivity,
)

OUT = REV / "final_v6_ransac/ablation_retrained"
OUT.mkdir(parents=True, exist_ok=True)
SPLIT_PATH = REV / "splits/paper_split_v2.json"


def scene_key(name: str):
    match = re.match(r"cloudR(\d+)$", name)
    return (int(match.group(1)) if match else 10**9, name)


def load_cloud_ply_utf8(path: str):
    with open(path, "rb") as handle:
        header_lines = 0
        n_vertices = 0
        binary = False
        while True:
            line = handle.readline()
            if not line:
                raise RuntimeError(f"Unexpected EOF in PLY header: {path}")
            header_lines += 1
            stripped = line.strip()
            if stripped.startswith(b"element vertex "):
                n_vertices = int(stripped.split()[-1])
            if stripped.startswith(b"format binary"):
                binary = True
            if stripped == b"end_header":
                break
        if binary:
            raise RuntimeError(f"Expected ASCII PLY for sensitivity analysis: {path}")
        numeric = handle.read().decode("ascii")
    values = np.fromstring(numeric, sep=" ")
    if values.size % n_vertices != 0:
        raise RuntimeError(f"PLY numeric block is not rectangular: {path}")
    data = values.reshape(n_vertices, -1)
    xyz = data[:, 0:3].astype(np.float64)
    semantic = np.nan_to_num(data[:, 6], nan=0.0).astype(int)
    instance = np.nan_to_num(data[:, 7], nan=-1.0).astype(int)
    return xyz, semantic, instance

def collect_trainval_clusters(train_scenes, val_scenes, min_plants=2, max_plants=5,
                              min_points=2000, eps=0.03):
    allowed = set(train_scenes) | set(val_scenes)
    data_dir = REV / "evalaute_test"
    files = sorted(
        [p.name for p in data_dir.glob("cloudR*.ply")
         if p.stem in allowed and not p.name.endswith("_sor.ply")],
        key=lambda p: scene_key(Path(p).stem),
    )
    clusters = []
    skipped = []
    for filename in files:
        try:
            xyz, _sem, inst = load_cloud_ply_utf8(str(data_dir / filename))
        except Exception as exc:
            skipped.append({"file": filename, "reason": str(exc)})
            print(f"SKIP_UNREADABLE {filename}: {exc}", flush=True)
            continue
        for cluster in find_sticky_clusters(xyz, inst, eps=eps):
            if min_plants <= cluster["n_instances"] <= max_plants and cluster["n_points"] >= min_points:
                clusters.append({
                    "file": filename,
                    "instances": cluster["instances"],
                    "points": xyz[cluster["idx"]],
                    "k": cluster["n_instances"],
                    "n": cluster["n_points"],
                })
    clusters.sort(key=lambda cluster: (cluster["n"], cluster["file"], cluster["instances"]))
    return files, clusters, skipped


def selection_table(min_peak_mae, valley_mae):
    rows = []
    for parameter, grid, matrix, selected in [
        ("min_peak_dist_m", MIN_PEAK_DIST_GRID, min_peak_mae, 0.30),
        ("valley_depth_rel", VALLEY_DEPTH_GRID, valley_mae, 0.90),
    ]:
        for value, values in zip(grid, matrix):
            rows.append({
                "parameter": parameter,
                "value": value,
                "eta_100": values[0],
                "eta_50": values[1],
                "eta_10": values[2],
                "eta_5": values[3],
                "mean_across_densities": float(np.nanmean(values)),
                "selected_in_deployed_model": bool(np.isclose(value, selected)),
            })
    return pd.DataFrame(rows)


def main():
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    train_scenes = split["train_subregions"]
    val_scenes = split["val_subregions"]
    test_scenes = split["test_subregions"]
    allowed_scenes = sorted(train_scenes + val_scenes, key=scene_key)

    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    cfg["instance"]["pca_split"]["enable"] = True
    cfg["instance"]["pca_split"]["abnormal_detect"] = True
    cfg["instance"]["pca_split"]["skeleton_cut"] = True
    cfg["instance"]["pca_split"]["adaptive_thresh"] = True
    cfg["instance"]["pca_split"]["merge_back"] = False
    cfg["instance"]["pca_split"]["min_peak_dist_m"] = 0.30
    cfg["instance"]["pca_split"]["valley_depth_rel"] = 0.90
    clusterer = InstanceClusterer(cfg)

    files, clusters, skipped = collect_trainval_clusters(train_scenes, val_scenes)
    if set(Path(name).stem for name in files) & set(test_scenes):
        raise RuntimeError("Test scene leaked into parameter-selection data")
    by_k = Counter(cluster["k"] for cluster in clusters)
    print(f"scenes={len(files)} clusters={len(clusters)} by_k={dict(sorted(by_k.items()))}", flush=True)

    clusterer.min_peak_dist_m = 0.30
    clusterer.valley_depth_rel = 0.90
    density_summary, density_by_k, cluster_results = run_density_sensitivity(clusterer, clusters)

    clusterer.valley_depth_rel = 0.90
    cm_mpd, mae_mpd = run_hyperparam_sensitivity(
        clusterer, clusters, "min_peak_dist_m", MIN_PEAK_DIST_GRID
    )
    clusterer.min_peak_dist_m = 0.30
    cm_vdr, mae_vdr = run_hyperparam_sensitivity(
        clusterer, clusters, "valley_depth_rel", VALLEY_DEPTH_GRID
    )

    payload = {
        "protocol": "train+val only; fixed test scenes excluded; rerun 2026-11-11",
        "split_source": str(SPLIT_PATH),
        "allowed_scenes": allowed_scenes,
        "excluded_test_scenes": test_scenes,
        "n_scenes": len(files),
        "readable_scenes": len(files),
        "skipped_scenes": skipped,
        "n_clusters": len(clusters),
        "n_clusters_by_k": {str(k): int(v) for k, v in sorted(by_k.items())},
        "ratios": RATIOS,
        "density_n_trials": DENSITY_N_TRIALS,
        "hyper_n_trials": HYPER_N_TRIALS,
        "fixed_values": {
            "min_peak_dist_m": 0.30,
            "valley_depth_rel": 0.90,
            "merge_back": False,
        },
        "density_summary": {str(ratio): density_summary[ratio] for ratio in RATIOS},
        "min_peak_dist_m": {
            "grid": MIN_PEAK_DIST_GRID,
            "correct_matrix": cm_mpd.tolist(),
            "mae_matrix": mae_mpd.tolist(),
        },
        "valley_depth_rel": {
            "grid": VALLEY_DEPTH_GRID,
            "correct_matrix": cm_vdr.tolist(),
            "mae_matrix": mae_vdr.tolist(),
        },
        "cluster_results": [
            {
                "file": cluster["file"],
                "instances": cluster["instances"],
                "k": cluster["k"],
                "n": cluster["n"],
                "mae_full": cluster["mae_full"],
                "mae_by_ratio": {str(r): cluster["mae_by_ratio"][r] for r in RATIOS},
            }
            for cluster in cluster_results
        ],
    }
    json_path = OUT / "sensitivity_multi_cluster_trainval.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    table = selection_table(mae_mpd, mae_vdr)
    csv_path = OUT / "sensitivity_trainval.csv"
    table.to_csv(csv_path, index=False)

    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)
    print("WROTE", json_path, flush=True)
    print("WROTE", csv_path, flush=True)


if __name__ == "__main__":
    main()






