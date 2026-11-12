from pathlib import Path
import json
import pandas as pd

ROOT = Path(r"E:\Cabbage")
P = ROOT / "revision_data/final_v6_ransac/hdbscan_20261112"
scene = pd.read_csv(P / "scene_method_metrics.csv")
plant = pd.read_csv(P / "scene_plant_method_metrics.csv")
summary = pd.read_csv(P / "hdbscan_summary_for_review.csv")
assert len(scene) == 48
assert len(plant) == 816
assert len(summary) == 4
assert not plant.duplicated(["model", "refinement", "plant_id"]).any()
assert plant.groupby(["model", "refinement"]).size().eq(102).all()
for (model, ref), g in scene.groupby(["model", "refinement"]):
    assert len(g) == 6
    assert g.gt_instances.sum() == 102
meta = json.loads((P / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
assert meta["rows"] == 48 and meta["plant_mapping_rows"] == 816
text = (P / "HDBSCAN_RESULTS_CN.md").read_text(encoding="utf-8")
assert text.count(chr(0xfffd)) == 0
for token in ("0.8479", "0.8507", "0.7971", "0.7967", "min_cluster_size=500"):
    assert token in text
print("HDBSCAN_AUDIT_OK")
print(summary.to_string(index=False))
