from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / "revision_data"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REV / "tools"))
os.chdir(ROOT)

from dump_traits_per_plant import process_file_clustering
from cabbage_pheno.service.evaluation import TRAIT_MAE_FIELDS
from cabbage_pheno.traits import TraitCalculator

CONFIG = REV / "final_v6_ransac/configs"
RETRAIN = CONFIG / "ransac_retrain_20261111"
MODEL_CONFIGS = {
    "RandLA-Net": {
        "base": RETRAIN / "randla_net_watershed_base_retrained.yaml",
        "adpv": RETRAIN / "randla_net_watershed_adpv_retrained.yaml",
        "slug": "randla_net",
    },
    "PointNet2-RGB": {
        "base": CONFIG / "pointnet2_rgb_watershed_base_v6.yaml",
        "adpv": CONFIG / "pointnet2_rgb_watershed_adpv_v6.yaml",
        "slug": "pointnet2_rgb",
    },
    "PointNeXt-L": {
        "base": RETRAIN / "pointnext_l_watershed_base_retrained.yaml",
        "adpv": RETRAIN / "pointnext_l_watershed_adpv_retrained.yaml",
        "slug": "pointnext_l",
    },
    "PointGroup": {
        "base": CONFIG / "pointgroup_watershed_base_v6.yaml",
        "adpv": CONFIG / "pointgroup_watershed_adpv_v6.yaml",
        "slug": "pointgroup",
    },
}
SCENES = ["cloudR2", "cloudR5", "cloudR9", "cloudR12", "cloudR14", "cloudR18"]
DATA_DIR = REV / "evalaute_test_v2"
DEFAULT_OUT = REV / "final_v6_ransac/paired_rerun_20261111/raw_traits"


def extract_traits(model: str, ref: str, scenes: list[str], out_dir: Path) -> Path:
    spec = MODEL_CONFIGS[model]
    cfg = yaml.safe_load(spec[ref].read_text(encoding="utf-8"))
    trait_calc = TraitCalculator(cfg)
    rows = []
    for scene in scenes:
        path = DATA_DIR / f"{scene}.ply"
        print("PROCESS", model, ref, scene, flush=True)
        result = process_file_clustering(str(path), cfg)
        if result is None:
            raise RuntimeError(f"No result for {model} {ref} {scene}")
        for pid, gid, iou in result["matches"]:
            p_mask = result["full_instance_pred"] == pid
            g_mask = result["gt_inst_raw"] == gid
            pts_p = result["points_clean"][p_mask]
            pts_g = result["gt_points_raw"][g_mask]
            if len(pts_p) < 4 or len(pts_g) < 4:
                continue
            pcd_p = o3d.geometry.PointCloud()
            pcd_p.points = o3d.utility.Vector3dVector(pts_p)
            pcd_g = o3d.geometry.PointCloud()
            pcd_g.points = o3d.utility.Vector3dVector(pts_g)
            t_p = trait_calc.calculate_traits(pcd_p, pid, result["plane_model"])
            t_g = trait_calc.calculate_traits(pcd_g, gid, result["plane_model"])
            if not (t_p and t_g):
                continue
            row = {
                "file": result["file"],
                "pred_id": int(pid),
                "gt_id": int(gid),
                "iou": float(iou),
            }
            for key, _, _ in TRAIT_MAE_FIELDS:
                vp = t_p.get(key)
                vg = t_g.get(key)
                row[f"pred_{key}"] = None if vp is None else float(vp)
                row[f"gt_{key}"] = None if vg is None else float(vg)
                if vp is not None and vg is not None:
                    row[f"err_{key}"] = float(abs(vp - vg))
                    row[f"rel_{key}_pct"] = (
                        float(abs(vp - vg) / abs(vg) * 100.0) if abs(vg) > 1e-9 else None
                    )
                else:
                    row[f"err_{key}"] = None
                    row[f"rel_{key}_pct"] = None
            rows.append(row)
        print("DONE_SCENE", model, ref, scene, len(rows), flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{spec['slug']}_watershed_{ref}.csv"
    pd.DataFrame(rows).round(4).to_csv(out, index=False)
    print("WROTE", out, len(rows), flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--models", default=",".join(MODEL_CONFIGS))
    parser.add_argument("--refs", default="base,adpv")
    parser.add_argument("--scenes", default=",".join(SCENES))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    refs = [x.strip() for x in args.refs.split(",") if x.strip()]
    scenes = [x.strip() for x in args.scenes.split(",") if x.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    for model in models:
        spec = MODEL_CONFIGS[model]
        for ref in refs:
            out = args.out_dir / f"{spec['slug']}_watershed_{ref}.csv"
            if out.exists() and not args.overwrite:
                print("SKIP", out, flush=True)
                continue
            extract_traits(model, ref, scenes, args.out_dir)
    manifest = {
        "started": started,
        "models": models,
        "refs": refs,
        "scenes": scenes,
        "configs": {
            model: {ref: str(MODEL_CONFIGS[model][ref]) for ref in refs}
            for model in models
        },
        "out_dir": str(args.out_dir),
    }
    (args.out_dir / "RUN_SUMMARY.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
