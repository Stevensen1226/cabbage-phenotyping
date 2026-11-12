from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / "revision_data"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REV / "tools"))
os.chdir(ROOT)

from dump_traits_per_plant import process_file_clustering
from cabbage_pheno.service.evaluation import (
    align_gt_to_pred,
    compute_instance_metrics_from_matches,
    compute_iou_semantic,
    get_instance_matches,
    load_ground_truth,
)

CONFIG_DIR = REV / "final_v6_ransac/configs/ransac_retrain_20261111/ablation_v2"
DEFAULT_OUT = REV / "final_v6_ransac/ablation_retrained"
SPLITS = {
    "val": {
        "data_dir": REV / "evalaute_test",
        "scenes": ["cloudR3", "cloudR6", "cloudR16", "cloudR20"],
    },
    "test": {
        "data_dir": REV / "evalaute_test_v2",
        "scenes": ["cloudR2", "cloudR5", "cloudR9", "cloudR12", "cloudR14", "cloudR18"],
    },
}
MODELS = [("PointNeXt-L", "pointnext"), ("PointGroup", "pointgroup")]
STATES = [("base", "Base"), ("split_only", "Split only"), ("full", "Split+merge")]


def evaluate_scene(scene: str, data_dir: Path, cfg: dict) -> dict:
    path = data_dir / f"{scene}.ply"
    gt_points, gt_sem, gt_inst = load_ground_truth(str(path))
    t0 = time.time()
    result = process_file_clustering(str(path), cfg)
    runtime = time.time() - t0
    if result is None:
        raise RuntimeError(f"process_file_clustering returned None for {path}")

    points = np.asarray(result["points_clean"], dtype=np.float32)
    pred_inst = np.asarray(result["full_instance_pred"], dtype=np.int32)
    pred_sem = np.zeros(len(points), dtype=np.uint8)
    cabbage_to_clean = result.get("cabbage_to_clean")
    if cabbage_to_clean is not None:
        pred_sem[np.asarray(cabbage_to_clean, dtype=np.int64)] = 1

    aligned_sem, aligned_inst = align_gt_to_pred(gt_points, gt_sem, gt_inst, points)
    aligned_sem = np.asarray(aligned_sem, dtype=np.uint8)
    aligned_inst = np.asarray(aligned_inst, dtype=np.int32)
    aligned_inst[aligned_inst <= 0] = 0

    tp = int(((pred_sem == 1) & (aligned_sem == 1)).sum())
    fp = int(((pred_sem == 1) & (aligned_sem == 0)).sum())
    fn = int(((pred_sem == 0) & (aligned_sem == 1)).sum())
    tn = int(((pred_sem == 0) & (aligned_sem == 0)).sum())
    ious = compute_iou_semantic(pred_sem, aligned_sem, 2)
    matches, n_pred, n_gt = get_instance_matches(pred_inst, aligned_inst, 0.5)
    precision, recall, f1, instance_miou, matched = compute_instance_metrics_from_matches(
        matches, n_pred, n_gt
    )

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


def write_summary(frame: pd.DataFrame, out_dir: Path) -> None:
    metric_cols = [
        "semantic_precision",
        "semantic_recall",
        "semantic_f1",
        "semantic_iou_background",
        "semantic_iou_cabbage",
        "semantic_miou",
        "instance_precision",
        "instance_recall",
        "instance_f1",
        "instance_miou",
        "mae_count",
        "runtime_s",
    ]
    summary = (
        frame.groupby(["split", "model", "state", "state_label"], observed=True)[metric_cols]
        .mean()
        .reset_index()
    )
    summary["n_scenes"] = (
        frame.groupby(["split", "model", "state", "state_label"], observed=True)
        .size()
        .to_numpy()
    )
    summary.to_csv(out_dir / "ablation_summary.csv", index=False)

    rows = []
    for (split, model), group in summary.groupby(["split", "model"], observed=True):
        indexed = group.set_index("state")
        if not {"base", "split_only", "full"}.issubset(indexed.index):
            continue
        for reference in ("base", "split_only"):
            ref = indexed.loc[reference]
            for state in ("base", "split_only", "full"):
                cur = indexed.loc[state]
                row = {
                    "split": split,
                    "model": model,
                    "reference": reference,
                    "state": state,
                    "delta_instance_f1": cur["instance_f1"] - ref["instance_f1"],
                    "delta_instance_miou": cur["instance_miou"] - ref["instance_miou"],
                    "delta_mae_count": cur["mae_count"] - ref["mae_count"],
                    "delta_semantic_miou": cur["semantic_miou"] - ref["semantic_miou"],
                    "delta_runtime_s": cur["runtime_s"] - ref["runtime_s"],
                }
                rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "ablation_deltas.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--models", default="PointNeXt-L,PointGroup")
    parser.add_argument("--states", default="base,split_only,full")
    parser.add_argument("--scenes", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    requested_splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    requested_models = {x.strip() for x in args.models.split(",") if x.strip()}
    requested_states = {x.strip() for x in args.states.split(",") if x.strip()}
    requested_scenes = {x.strip() for x in args.scenes.split(",") if x.strip()}

    metrics_path = out_dir / "ablation_scene_metrics.csv"
    if metrics_path.exists() and not args.overwrite:
        frame = pd.read_csv(metrics_path)
        rows = frame.to_dict("records")
    else:
        frame = pd.DataFrame()
        rows = []

    completed = {
        (str(r["split"]), str(r["model"]), str(r["state"]), str(r["scene_id"]))
        for r in rows
    }

    for split in requested_splits:
        split_cfg = SPLITS[split]
        scenes = [s for s in split_cfg["scenes"] if not requested_scenes or s in requested_scenes]
        for model, slug in MODELS:
            if model not in requested_models:
                continue
            for state, state_label in STATES:
                if state not in requested_states:
                    continue
                cfg_path = CONFIG_DIR / f"{slug}_{state}.yaml"
                cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
                for scene in scenes:
                    key = (split, model, state, scene)
                    if key in completed:
                        print("SKIP", key, flush=True)
                        continue
                    print("PROCESS", key, flush=True)
                    row = evaluate_scene(scene, split_cfg["data_dir"], cfg)
                    row.update(
                        {
                            "split": split,
                            "model": model,
                            "state": state,
                            "state_label": state_label,
                            "config": str(cfg_path),
                        }
                    )
                    rows.append(row)
                    completed.add(key)
                    frame = pd.DataFrame(rows)
                    frame.to_csv(metrics_path, index=False)
                    print(
                        "DONE",
                        split,
                        model,
                        state,
                        scene,
                        f"F1={row['instance_f1']:.4f}",
                        f"mIoU={row['instance_miou']:.4f}",
                        f"MAE={row['mae_count']:.0f}",
                        f"t={row['runtime_s']:.1f}s",
                        flush=True,
                    )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No ablation rows were produced")
    write_summary(frame, out_dir)

    checkpoints = {}
    for model, slug in MODELS:
        cfg = yaml.safe_load((CONFIG_DIR / f"{slug}_full.yaml").read_text(encoding="utf-8"))
        seg = cfg["segmentation"]
        checkpoints[model] = seg.get("pointnext_checkpoint") or seg.get("pointgroup_pretrain")

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_dir": str(CONFIG_DIR),
        "splits": {k: {"data_dir": str(v["data_dir"]), "scenes": v["scenes"]} for k, v in SPLITS.items()},
        "states": {k: v for k, v in STATES},
        "checkpoints": checkpoints,
        "rows": int(len(frame)),
        "outputs": {
            "scene_metrics": str(metrics_path),
            "summary": str(out_dir / "ablation_summary.csv"),
            "deltas": str(out_dir / "ablation_deltas.csv"),
        },
    }
    (out_dir / "RUN_SUMMARY.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print((out_dir / "ablation_summary.csv").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
