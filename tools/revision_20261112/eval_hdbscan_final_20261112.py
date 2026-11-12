from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(r"E:\Cabbage")
REV = ROOT / "revision_data"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REV / "tools"))
import eval_unified_main_20261111 as base

OUT = REV / "final_v6_ransac/hdbscan_20261112"
OUT.mkdir(parents=True, exist_ok=True)
CFG = REV / "final_v6_ransac/configs/hdbscan_20261112"
SCENES = base.SCENES
MODELS = [("RandLA-Net", "randla_net"), ("PointNet2-RGB", "pointnet2_rgb"), ("PointNeXt-L", "pointnext_l"), ("PointGroup", "pointgroup")]
rows = []
plant_rows = []
for model, slug in MODELS:
    for refinement in ("base", "adpv"):
        cfg_path = CFG / f"{slug}_hdbscan_{refinement}.yaml"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        for scene in SCENES:
            print("PROCESS", model, refinement, scene, flush=True)
            t0 = time.time()
            result = base.process_file_clustering(str(base.DATA_DIR / f"{scene}.ply"), cfg)
            runtime = time.time() - t0
            row = base.evaluate_result(result, scene, runtime)
            row.update({"model": model, "clustering": "hdbscan", "refinement": refinement})
            rows.append(row)
            plant_rows.extend(base.build_plant_mapping(result, scene, row, model, "hdbscan", refinement))
            pd.DataFrame(rows).to_csv(OUT / "scene_method_metrics.csv", index=False)
            pd.DataFrame(plant_rows).to_csv(OUT / "scene_plant_method_metrics.csv", index=False)
            print("DONE", model, refinement, scene, row["instance_f1"], row["instance_miou"], row["mae_count"], flush=True)
frame = pd.DataFrame(rows)
base.summarize(frame, OUT)
(OUT / "RUN_SUMMARY.json").write_text(json.dumps({"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "scenes": SCENES, "models": [m[0] for m in MODELS], "clustering": "hdbscan", "hdbscan": {"min_cluster_size": 500, "min_samples": 10, "cluster_selection_method": "eom"}, "rows": len(frame), "plant_mapping_rows": len(plant_rows), "config_dir": str(CFG)}, ensure_ascii=False, indent=2), encoding="utf-8")
print(pd.read_csv(OUT / "main_results_for_report.csv").to_string(index=False, float_format=lambda x: f"{x:.4f}"))
