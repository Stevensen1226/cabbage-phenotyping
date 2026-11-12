from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / "revision_data"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REV / "tools"))
os.chdir(ROOT)

from dump_traits_per_plant import process_file, process_file_clustering
from cabbage_pheno.service.evaluation import (
    align_gt_to_pred,
    compute_instance_metrics_from_matches,
    compute_iou_semantic,
    get_instance_matches,
    load_ground_truth,
)

CONFIG = REV / "final_v6_ransac/configs"
RETRAIN = CONFIG / "ransac_retrain_20261111"
DATA_DIR = REV / "evalaute_test_v2"
SCENES = ["cloudR2", "cloudR5", "cloudR9", "cloudR12", "cloudR14", "cloudR18"]
DEFAULT_OUT = REV / "final_v6_ransac/main_unified_20261111"
REGISTRY = pd.read_csv(REV / "splits/test_plant_registry.csv")
NEW_RANDLA = "E:\\Cabbage\\revision_data\\runs\\ransac_retrain_20261111\\randlanet\\best.pth"
NEW_PNEXT = "E:\\Cabbage\\revision_data\\runs\\ransac_retrain_20261111\\pointnext_final\\best.pth"

MODELS = {
    "RandLA-Net": {
        "watershed": {
            "base": RETRAIN / "randla_net_watershed_base_retrained.yaml",
            "adpv": RETRAIN / "randla_net_watershed_adpv_retrained.yaml",
        },
        "meanshift": {
            "base": CONFIG / "randla_net_meanshift_base_v6.yaml",
            "adpv": CONFIG / "randla_net_meanshift_adpv_v6.yaml",
        },
    },
    "PointNet2-RGB": {
        "watershed": {
            "base": CONFIG / "pointnet2_rgb_watershed_base_v6.yaml",
            "adpv": CONFIG / "pointnet2_rgb_watershed_adpv_v6.yaml",
        },
        "meanshift": {
            "base": CONFIG / "pointnet2_rgb_meanshift_base_v6.yaml",
            "adpv": CONFIG / "pointnet2_rgb_meanshift_adpv_v6.yaml",
        },
    },
    "PointNeXt-L": {
        "watershed": {
            "base": RETRAIN / "pointnext_l_watershed_base_retrained.yaml",
            "adpv": RETRAIN / "pointnext_l_watershed_adpv_retrained.yaml",
        },
        "meanshift": {
            "base": CONFIG / "pointnext_l_meanshift_base_v6.yaml",
            "adpv": CONFIG / "pointnext_l_meanshift_adpv_v6.yaml",
        },
    },
    "PointGroup": {
        "watershed": {
            "base": CONFIG / "pointgroup_watershed_base_v6.yaml",
            "adpv": CONFIG / "pointgroup_watershed_adpv_v6.yaml",
        },
        "meanshift": {
            "base": CONFIG / "pointgroup_meanshift_base_v6.yaml",
            "adpv": CONFIG / "pointgroup_meanshift_adpv_v6.yaml",
        },
        "native": {
            "base": CONFIG / "pointgroup_native_base_v6.yaml",
            "adpv": CONFIG / "pointgroup_native_adpv_v6.yaml",
        },
    },
}


def load_cfg(model: str, clustering: str, refinement: str) -> dict:
    cfg = yaml.safe_load(MODELS[model][clustering][refinement].read_text(encoding="utf-8"))
    if model == "RandLA-Net":
        cfg["segmentation"]["randlanet_checkpoint"] = NEW_RANDLA
    elif model == "PointNeXt-L":
        cfg["segmentation"]["pointnext_checkpoint"] = NEW_PNEXT
    return cfg


def evaluate_result(result: dict, scene: str, runtime: float) -> dict:
    if result is None:
        raise RuntimeError(f"No result for {scene}")
    points = np.asarray(result["points_clean"], dtype=np.float32)
    pred_inst = np.asarray(result["full_instance_pred"], dtype=np.int32)
    pred_sem = np.zeros(len(points), dtype=np.uint8)
    cabbage_to_clean = result.get("cabbage_to_clean")
    if cabbage_to_clean is not None:
        pred_sem[np.asarray(cabbage_to_clean, dtype=np.int64)] = 1

    gt_points = result["gt_points_raw"]
    gt_sem = result["gt_inst_raw"]
    gt_sem_raw, gt_inst_raw = load_ground_truth(str(DATA_DIR / f"{scene}.ply"))[1:]
    aligned_sem, aligned_inst = align_gt_to_pred(gt_points, gt_sem_raw, gt_inst_raw, points)
    aligned_sem = np.asarray(aligned_sem, dtype=np.uint8)
    aligned_inst = np.asarray(aligned_inst, dtype=np.int32)
    aligned_inst[aligned_inst <= 0] = 0

    tp = int(((pred_sem == 1) & (aligned_sem == 1)).sum())
    fp = int(((pred_sem == 1) & (aligned_sem == 0)).sum())
    fn = int(((pred_sem == 0) & (aligned_sem == 1)).sum())
    tn = int(((pred_sem == 0) & (aligned_sem == 0)).sum())
    ious = compute_iou_semantic(pred_sem, aligned_sem, 2)
    matches, n_pred, n_gt = get_instance_matches(pred_inst, aligned_inst, 0.5)
    precision, recall, f1, instance_miou, matched = compute_instance_metrics_from_matches(matches, n_pred, n_gt)
    return {
        "scene_id": scene,
        "points": len(points),
        "semantic_precision": tp / (tp + fp) if tp + fp else np.nan,
        "semantic_recall": tp / (tp + fn) if tp + fn else np.nan,
        "semantic_f1": 2 * tp / (2 * tp + fp + fn) if tp else np.nan,
        "semantic_iou_background": float(ious[0]),
        "semantic_iou_cabbage": float(ious[1]),
        "semantic_miou": float(np.nanmean(ious)),
        "instance_precision": precision,
        "instance_recall": recall,
        "instance_f1": f1,
        "instance_miou": instance_miou,
        "matched_instances": int(matched),
        "pred_instances": n_pred,
        "gt_instances": n_gt,
        "mae_count": abs(n_pred - n_gt),
        "runtime_s": runtime,
        "semantic_tp": tp,
        "semantic_fp": fp,
        "semantic_fn": fn,
        "semantic_tn": tn,
    }


def build_plant_mapping(result: dict, scene: str, scene_row: dict, model: str, clustering: str, refinement: str) -> list[dict]:
    if result is None:
        raise RuntimeError(f"No result for {scene}")
    match_map = {int(gid): (int(pid), float(iou)) for pid, gid, iou in result["matches"]}
    gt_inst = np.asarray(result["gt_inst_raw"], dtype=np.int32)
    pred_inst = np.asarray(result["full_instance_pred"], dtype=np.int32)
    registry = REGISTRY[REGISTRY["scene_id"].eq(scene)]
    rows = []
    for plant in registry.itertuples(index=False):
        gid = int(plant.gt_instance_id)
        matched = gid in match_map
        if matched:
            pred_id, iou = match_map[gid]
            pred_points = int((pred_inst == pred_id).sum())
        else:
            pred_id = None
            iou = 0.0
            pred_points = 0
        rows.append({
            "scene_id": scene,
            "plant_id": plant.plant_id,
            "gt_instance_id": gid,
            "model": model,
            "clustering": clustering,
            "refinement": refinement,
            "matched": int(matched),
            "pred_instance_id": pred_id,
            "iou": iou,
            "gt_points": int((gt_inst == gid).sum()),
            "pred_points": pred_points,
            "scene_instance_f1": float(scene_row["instance_f1"]),
            "scene_instance_miou": float(scene_row["instance_miou"]),
            "scene_mae_count": float(scene_row["mae_count"]),
        })
    if len(rows) != len(registry):
        raise RuntimeError(f"Expected {len(registry)} plants for {scene}, got {len(rows)}")
    return rows


def summarize(frame: pd.DataFrame, out: Path) -> pd.DataFrame:
    metrics = [
        "semantic_precision", "semantic_recall", "semantic_f1", "semantic_miou",
        "instance_precision", "instance_recall", "instance_f1", "instance_miou",
        "mae_count", "runtime_s",
    ]
    macro = frame.groupby(["model", "clustering", "refinement"], observed=True)[metrics].mean().reset_index()
    coverage = frame.groupby(["model", "clustering", "refinement"], observed=True).agg(
        matched_instances=("matched_instances", "sum"),
        gt_instances=("gt_instances", "sum"),
    ).reset_index()
    coverage["coverage"] = coverage["matched_instances"] / coverage["gt_instances"]
    macro = macro.merge(coverage, on=["model", "clustering", "refinement"])
    macro.to_csv(out / "macro_method_metrics.csv", index=False)
    report = macro.rename(columns={
        "precipitation": "precision",
        "instance_precision": "precision",
        "instance_recall": "recall",
        "instance_f1": "f1",
        "mae_count": "mae_count",
    })
    report = report[[
        "model", "clustering", "refinement", "precision", "recall", "f1",
        "instance_miou", "semantic_miou", "mae_count", "coverage", "runtime_s",
    ]]
    report.to_csv(out / "main_results_for_report.csv", index=False)
    return macro


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--models", default="RandLA-Net,PointNet2-RGB,PointNeXt-L,PointGroup")
    parser.add_argument("--clusterings", default="watershed,meanshift,native")
    parser.add_argument("--refs", default="base,adpv")
    parser.add_argument("--scenes", default=",".join(SCENES))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mapping-out", type=Path, default=None)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    requested_models = [x.strip() for x in args.models.split(",") if x.strip()]
    requested_clusterings = [x.strip() for x in args.clusterings.split(",") if x.strip()]
    requested_refs = [x.strip() for x in args.refs.split(",") if x.strip()]
    requested_scenes = [x.strip() for x in args.scenes.split(",") if x.strip()]

    path = args.out_dir / "scene_method_metrics.csv"
    mapping_path = args.mapping_out or (args.out_dir / "scene_plant_method_metrics.csv")
    rows = [] if args.overwrite or not path.exists() else pd.read_csv(path).to_dict("records")
    plant_rows = [] if args.overwrite or not mapping_path.exists() else pd.read_csv(mapping_path).to_dict("records")
    completed = {(r["model"], r["clustering"], r["refinement"], r["scene_id"]) for r in rows}

    for model in requested_models:
        for clustering in requested_clusterings:
            if clustering not in MODELS[model]:
                continue
            for refinement in requested_refs:
                cfg = load_cfg(model, clustering, refinement)
                for scene in requested_scenes:
                    key = (model, clustering, refinement, scene)
                    if key in completed:
                        print("SKIP", key, flush=True)
                        continue
                    print("PROCESS", key, flush=True)
                    t0 = time.time()
                    if clustering == "native":
                        result = process_file(str(DATA_DIR / f"{scene}.ply"), cfg)
                    else:
                        result = process_file_clustering(str(DATA_DIR / f"{scene}.ply"), cfg)
                    runtime = time.time() - t0
                    row = evaluate_result(result, scene, runtime)
                    row.update({"model": model, "clustering": clustering, "refinement": refinement})
                    rows.append(row)
                    plant_rows.extend(build_plant_mapping(result, scene, row, model, clustering, refinement))
                    completed.add(key)
                    pd.DataFrame(rows).to_csv(path, index=False)
                    pd.DataFrame(plant_rows).to_csv(mapping_path, index=False)
                    print("DONE", key, f"F1={row['instance_f1']:.4f}", f"mIoU={row['instance_miou']:.4f}", flush=True)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No main-evaluation rows")
    summarize(frame, args.out_dir)
    (args.out_dir / "RUN_SUMMARY.json").write_text(
        json.dumps({
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_dir": str(DATA_DIR),
            "scenes": SCENES,
            "models": requested_models,
            "clusterings": requested_clusterings,
            "refinements": requested_refs,
            "rows": len(frame),
            "plant_mapping_rows": len(plant_rows),
            "plant_mapping": str(mapping_path),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(pd.read_csv(args.out_dir / "main_results_for_report.csv").to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
