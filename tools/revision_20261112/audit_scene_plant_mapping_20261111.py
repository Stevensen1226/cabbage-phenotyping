from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(r"E:\Cabbage")
R = ROOT / "revision_data/final_v6_ransac"
MAP = R / "main_unified_20261112/scene_plant_method_metrics.csv"
SCENE = R / "main_unified_20261112/scene_method_metrics.csv"
PLANT = R / "main_unified_20261112/plant_gain_summary.csv"
REG = pd.read_csv(ROOT / "revision_data/splits/test_plant_registry.csv")
mapping = pd.read_csv(MAP)
scene = pd.read_csv(SCENE)
plant = pd.read_csv(PLANT)
assert len(mapping) == 2652
assert len(scene) == 156
assert len(plant) == 13
assert not mapping.duplicated(["model", "clustering", "refinement", "plant_id"]).any()
keys = set(zip(REG.scene_id, REG.gt_instance_id))
assert set(zip(mapping.scene_id, mapping.gt_instance_id)) == keys
for (model, clustering, refinement), group in mapping.groupby(["model", "clustering", "refinement"]):
    assert len(group) == 102, (model, clustering, refinement, len(group))
    assert group.plant_id.nunique() == 102
    sm = scene[(scene.model.eq(model)) & (scene.clustering.eq(clustering)) & (scene.refinement.eq(refinement))]
    assert int(group.matched.sum()) == int(sm.matched_instances.sum())

for (model, clustering), group in mapping.groupby(["model", "clustering"]):
    b = group[group.refinement.eq("base")].set_index("plant_id")
    a = group[group.refinement.eq("adpv")].set_index("plant_id")
    row = plant[(plant.model.eq(model)) & (plant.clustering.eq(clustering))].iloc[0]
    assert int(b.matched.sum()) == int(row.base_matched)
    assert int(a.matched.sum()) == int(row.adpv_matched)
    assert int(((b.matched == 0) & (a.matched == 1)).sum()) == int(row.newly_matched)
    assert int(((b.matched == 1) & (a.matched == 0)).sum()) == int(row.lost_matches)

report = (ROOT / "docs/COMPAG-D-26-10878_Fixed_Cohort_Results_CN.md").read_text(encoding="utf-8-sig")
man = (ROOT / "docs/COMPAG-D-26-10878_Revised_Manuscript_Text.md").read_text(encoding="utf-8-sig")
cover = (ROOT / "docs/COMPAG-D-26-10878_Cover_Letter.md").read_text(encoding="utf-8")
for text in (report, man):
    assert "2652" in text or "2,652" in text
    for scene_id in ("cloudR2", "cloudR9", "cloudR14"):
        assert scene_id in text
assert "per-plant audit" in cover.lower()
assert "scene_plant_method_metrics.csv" in report
print("SCENE_PLANT_MAPPING_AUDIT_OK")
print(plant.to_string(index=False))
