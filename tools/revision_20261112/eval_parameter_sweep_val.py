from __future__ import annotations

from copy import deepcopy
from itertools import product
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / "revision_data"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REV / "tools"))

from eval_retrained_ablation import evaluate_scene

OUT = REV / "final_v6_ransac/ablation_retrained/parameter_sweep"
OUT.mkdir(parents=True, exist_ok=True)
CONFIG_DIR = REV / "final_v6_ransac/configs/ransac_retrain_20261111/ablation_v2"
DATA_DIR = REV / "evalaute_test"
SCENES = ["cloudR3", "cloudR6", "cloudR16", "cloudR20"]
MODELS = [("PointNeXt-L", "pointnext"), ("PointGroup", "pointgroup")]
GRID = list(product([0.30, 0.35], [0.85, 0.90, 0.95]))

rows = []
for model, slug in MODELS:
    base_cfg = yaml.safe_load((CONFIG_DIR / f"{slug}_full.yaml").read_text(encoding="utf-8"))
    for min_peak, valley in GRID:
        cfg = deepcopy(base_cfg)
        cfg["instance"]["pca_split"]["min_peak_dist_m"] = min_peak
        cfg["instance"]["pca_split"]["valley_depth_rel"] = valley
        for scene in SCENES:
            print("PROCESS", model, min_peak, valley, scene, flush=True)
            row = evaluate_scene(scene, DATA_DIR, cfg)
            row.update({
                "model": model,
                "scene_id": scene,
                "min_peak_dist_m": min_peak,
                "valley_depth_rel": valley,
            })
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT / "parameter_sweep_scene_metrics.csv", index=False)

frame = pd.DataFrame(rows)
metrics = ["instance_precision", "instance_recall", "instance_f1", "instance_miou", "mae_count", "runtime_s"]
summary = frame.groupby(["model", "min_peak_dist_m", "valley_depth_rel"])[metrics].mean().reset_index()
summary.to_csv(OUT / "parameter_sweep_summary.csv", index=False)

best = (
    summary.sort_values(["model", "instance_f1", "mae_count"], ascending=[True, False, True])
    .groupby("model", as_index=False)
    .first()
)
best.to_csv(OUT / "parameter_sweep_best_per_model.csv", index=False)
(OUT / "RUN_SUMMARY.json").write_text(
    json.dumps({
        "protocol": "validation-only end-to-end parameter sweep; test scenes excluded",
        "scenes": SCENES,
        "grid": {"min_peak_dist_m": [0.30, 0.35], "valley_depth_rel": [0.85, 0.90, 0.95]},
        "models": [m[0] for m in MODELS],
        "rows": len(frame),
    }, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)
print(best.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)
