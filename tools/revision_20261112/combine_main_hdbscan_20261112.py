from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

ROOT = Path(r"E:\Cabbage")
REV = ROOT / "revision_data"
BASE = REV / "final_v6_ransac"
PRIMARY = BASE / "main_unified_20261111"
HDB = BASE / "hdbscan_20261112"
OUT = BASE / "main_unified_20261112"
OUT.mkdir(parents=True, exist_ok=True)
scene = pd.concat([pd.read_csv(PRIMARY / "scene_method_metrics.csv"), pd.read_csv(HDB / "scene_method_metrics.csv")], ignore_index=True)
main = pd.concat([pd.read_csv(PRIMARY / "main_results_for_report.csv"), pd.read_csv(HDB / "main_results_for_report.csv")], ignore_index=True)
macro = pd.concat([pd.read_csv(PRIMARY / "macro_method_metrics.csv"), pd.read_csv(HDB / "macro_method_metrics.csv")], ignore_index=True)
mapping = pd.concat([pd.read_csv(PRIMARY / "scene_plant_method_metrics.csv"), pd.read_csv(HDB / "scene_plant_method_metrics.csv")], ignore_index=True)
sort_cols = ["model", "clustering", "refinement"]
scene = scene.sort_values(sort_cols + ["scene_id"]).reset_index(drop=True)
main = main.sort_values(sort_cols).reset_index(drop=True)
macro = macro.sort_values(sort_cols).reset_index(drop=True)
mapping = mapping.sort_values(sort_cols + ["scene_id", "gt_instance_id"]).reset_index(drop=True)
scene.to_csv(OUT / "scene_method_metrics.csv", index=False)
main.to_csv(OUT / "main_results_for_report.csv", index=False)
macro.to_csv(OUT / "macro_method_metrics.csv", index=False)
mapping.to_csv(OUT / "scene_plant_method_metrics.csv", index=False)
rows = []
for (model, clustering), group in scene.groupby(["model", "clustering"]):
    b = group[group.refinement.eq("base")].set_index("scene_id"); a = group[group.refinement.eq("adpv")].set_index("scene_id")
    for sid in b.index:
        rb, ra = b.loc[sid], a.loc[sid]
        rows.append({"model": model, "clustering": clustering, "scene_id": sid, "gt_instances": int(rb.gt_instances), "base_f1": rb.instance_f1, "adpv_f1": ra.instance_f1, "delta_f1": ra.instance_f1-rb.instance_f1, "base_instance_miou": rb.instance_miou, "adpv_instance_miou": ra.instance_miou, "delta_instance_miou": ra.instance_miou-rb.instance_miou, "base_mae_count": rb.mae_count, "adpv_mae_count": ra.mae_count, "delta_mae_count": ra.mae_count-rb.mae_count, "base_matched": int(rb.matched_instances), "adpv_matched": int(ra.matched_instances), "delta_coverage": (ra.matched_instances-rb.matched_instances)/rb.gt_instances})
sg = pd.DataFrame(rows); sg.to_csv(OUT / "scene_gain_table.csv", index=False)
ss = []
for (model, clustering), group in sg.groupby(["model", "clustering"]):
    neg = group[group.delta_f1 < -1e-12].sort_values("delta_f1")
    ss.append({"model": model, "clustering": clustering, "scenes_positive": int((group.delta_f1>1e-12).sum()), "scenes_tie": int((group.delta_f1.abs()<=1e-12).sum()), "scenes_negative": int((group.delta_f1<-1e-12).sum()), "min_delta_f1": group.delta_f1.min(), "max_delta_f1": group.delta_f1.max(), "mean_delta_f1": group.delta_f1.mean(), "negative_scenes": "; ".join(f"{r.scene_id}({r.delta_f1:+.3f})" for r in neg.itertuples())})
pd.DataFrame(ss).to_csv(OUT / "scene_gain_summary.csv", index=False)
keys = ["scene_id", "plant_id", "gt_instance_id"]; detail_rows, summary_rows = [], []
for (model, clustering), group in mapping.groupby(["model", "clustering"]):
    b = group[group.refinement.eq("base")].set_index(keys); a = group[group.refinement.eq("adpv")].set_index(keys)
    common = b.index.intersection(a.index); b, a = b.loc[common], a.loc[common]
    detail = pd.DataFrame({"scene_id": [k[0] for k in common], "plant_id": [k[1] for k in common], "gt_instance_id": [k[2] for k in common], "base_matched": b.matched.to_numpy(int), "adpv_matched": a.matched.to_numpy(int), "base_iou": b.iou.to_numpy(float), "adpv_iou": a.iou.to_numpy(float), "model": model, "clustering": clustering})
    detail["newly_matched"] = ((detail.base_matched==0)&(detail.adpv_matched==1)).astype(int)
    detail["lost_match"] = ((detail.base_matched==1)&(detail.adpv_matched==0)).astype(int)
    detail["both_matched"] = ((detail.base_matched==1)&(detail.adpv_matched==1)).astype(int)
    detail["delta_iou"] = detail.adpv_iou-detail.base_iou
    detail_rows.append(detail)
    both = detail[detail.both_matched.eq(1)]
    summary_rows.append({"model": model, "clustering": clustering, "gt_plants": len(detail), "base_matched": int(detail.base_matched.sum()), "adpv_matched": int(detail.adpv_matched.sum()), "newly_matched": int(detail.newly_matched.sum()), "lost_matches": int(detail.lost_match.sum()), "both_matched": int(detail.both_matched.sum()), "mean_base_iou_both": both.base_iou.mean(), "mean_adpv_iou_both": both.adpv_iou.mean(), "mean_delta_iou_both": both.delta_iou.mean()})
pd.concat(detail_rows, ignore_index=True).to_csv(OUT / "plant_gain_table.csv", index=False)
pd.DataFrame(summary_rows).to_csv(OUT / "plant_gain_summary.csv", index=False)
assert len(scene) == 156
assert len(main) == 26
assert len(mapping) == 2652
assert len(sg) == 78 and len(ss) == 13
assert len(pd.read_csv(OUT / "plant_gain_table.csv")) == 1326
(OUT / "RUN_SUMMARY.json").write_text(json.dumps({"generated_at": "2026-11-12", "primary_configs": 18, "hdbscan_configs": 8, "total_configs": 26, "scene_rows": 156, "plant_mapping_rows": 2652, "models": sorted(main.model.unique().tolist()), "clusterings": sorted(main.clustering.unique().tolist())}, ensure_ascii=False, indent=2), encoding="utf-8")
print(main.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
