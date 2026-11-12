from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(r"E:\Cabbage")
SRC = ROOT / "revision_data/final_v6_ransac/main_unified_20261111_plantmap/scene_plant_method_metrics.csv"
OUT = ROOT / "revision_data/final_v6_ransac/main_unified_20261111"
d = pd.read_csv(SRC)
keys = ["scene_id", "plant_id", "gt_instance_id"]
detail_rows = []
summary_rows = []
for (model, clustering), group in d.groupby(["model", "clustering"]):
    base = group[group.refinement.eq("base")].set_index(keys)
    adpv = group[group.refinement.eq("adpv")].set_index(keys)
    common = base.index.intersection(adpv.index)
    b = base.loc[common]
    a = adpv.loc[common]
    detail = pd.DataFrame({
        "scene_id": [k[0] for k in common],
        "plant_id": [k[1] for k in common],
        "gt_instance_id": [k[2] for k in common],
        "base_matched": b.matched.to_numpy(int),
        "adpv_matched": a.matched.to_numpy(int),
        "base_iou": b.iou.to_numpy(float),
        "adpv_iou": a.iou.to_numpy(float),
    })
    detail["model"] = model
    detail["clustering"] = clustering
    detail["newly_matched"] = ((detail.base_matched == 0) & (detail.adpv_matched == 1)).astype(int)
    detail["lost_match"] = ((detail.base_matched == 1) & (detail.adpv_matched == 0)).astype(int)
    detail["both_matched"] = ((detail.base_matched == 1) & (detail.adpv_matched == 1)).astype(int)
    detail["delta_iou"] = detail.adpv_iou - detail.base_iou
    detail_rows.append(detail)
    both = detail[detail.both_matched.eq(1)]
    summary_rows.append({
        "model": model,
        "clustering": clustering,
        "gt_plants": len(detail),
        "base_matched": int(detail.base_matched.sum()),
        "adpv_matched": int(detail.adpv_matched.sum()),
        "newly_matched": int(detail.newly_matched.sum()),
        "lost_matches": int(detail.lost_match.sum()),
        "both_matched": int(detail.both_matched.sum()),
        "mean_base_iou_both": float(both.base_iou.mean()),
        "mean_adpv_iou_both": float(both.adpv_iou.mean()),
        "mean_delta_iou_both": float(both.delta_iou.mean()),
    })
detail_all = pd.concat(detail_rows, ignore_index=True)[[
    "model", "clustering", "scene_id", "plant_id", "gt_instance_id",
    "base_matched", "adpv_matched", "newly_matched", "lost_match", "both_matched",
    "base_iou", "adpv_iou", "delta_iou",
]]
summary = pd.DataFrame(summary_rows)
detail_all.to_csv(OUT / "plant_gain_table.csv", index=False)
summary.to_csv(OUT / "plant_gain_summary.csv", index=False)
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
