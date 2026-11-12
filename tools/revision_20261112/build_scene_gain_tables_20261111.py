from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(r"E:\Cabbage")
SRC = ROOT / "revision_data/final_v6_ransac/main_unified_20261111/scene_method_metrics.csv"
OUT = ROOT / "revision_data/final_v6_ransac/main_unified_20261111"
d = pd.read_csv(SRC)
rows = []
for (model, clustering), group in d.groupby(["model", "clustering"]):
    base = group[group.refinement.eq("base")].set_index("scene_id")
    adpv = group[group.refinement.eq("adpv")].set_index("scene_id")
    for scene in sorted(base.index, key=lambda x: (int(x.replace("cloudR", "")), x)):
        b = base.loc[scene]
        a = adpv.loc[scene]
        rows.append({
            "model": model,
            "clustering": clustering,
            "scene_id": scene,
            "gt_instances": int(b.gt_instances),
            "base_f1": float(b.instance_f1),
            "adpv_f1": float(a.instance_f1),
            "delta_f1": float(a.instance_f1 - b.instance_f1),
            "base_instance_miou": float(b.instance_miou),
            "adpv_instance_miou": float(a.instance_miou),
            "delta_instance_miou": float(a.instance_miou - b.instance_miou),
            "base_mae_count": float(b.mae_count),
            "adpv_mae_count": float(a.mae_count),
            "delta_mae_count": float(a.mae_count - b.mae_count),
            "base_matched": int(b.matched_instances),
            "adpv_matched": int(a.matched_instances),
            "delta_coverage": float((a.matched_instances - b.matched_instances) / b.gt_instances),
        })
scene = pd.DataFrame(rows)
scene.to_csv(OUT / "scene_gain_table.csv", index=False)
summary_rows = []
for (model, clustering), group in scene.groupby(["model", "clustering"]):
    negative = group[group.delta_f1 < -1e-12].sort_values("delta_f1")
    summary_rows.append({
        "model": model,
        "clustering": clustering,
        "scenes_positive": int((group.delta_f1 > 1e-12).sum()),
        "scenes_tie": int((group.delta_f1.abs() <= 1e-12).sum()),
        "scenes_negative": int((group.delta_f1 < -1e-12).sum()),
        "min_delta_f1": float(group.delta_f1.min()),
        "max_delta_f1": float(group.delta_f1.max()),
        "mean_delta_f1": float(group.delta_f1.mean()),
        "negative_scenes": "; ".join(f"{r.scene_id}({r.delta_f1:+.3f})" for r in negative.itertuples()),
    })
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "scene_gain_summary.csv", index=False)
print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
