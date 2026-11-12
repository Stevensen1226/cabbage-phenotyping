from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(r"E:\Cabbage")
R = ROOT / "revision_data/final_v6_ransac"
MAIN = R / "main_unified_20261112"
EFF = R / "efficiency_20261111"
SCENES = {"cloudR2", "cloudR5", "cloudR9", "cloudR12", "cloudR14", "cloudR18"}
EXPECTED = {
    "RandLA-Net": {"watershed", "meanshift", "hdbscan"},
    "PointNet2-RGB": {"watershed", "meanshift", "hdbscan"},
    "PointNeXt-L": {"watershed", "meanshift", "hdbscan"},
    "PointGroup": {"watershed", "meanshift", "native", "hdbscan"},
}

scene = pd.read_csv(MAIN / "scene_method_metrics.csv")
assert len(scene) == 156, len(scene)
for model, methods in EXPECTED.items():
    for method in methods:
        for ref in ("base", "adpv"):
            q = scene[(scene.model.eq(model)) & (scene.clustering.eq(method)) & (scene.refinement.eq(ref))]
            assert len(q) == 6, (model, method, ref, len(q))
            assert set(q.scene_id) == SCENES
            assert q.gt_instances.sum() == 102

main = pd.read_csv(MAIN / "main_results_for_report.csv")
assert len(main) == 26
for _, row in main.iterrows():
    q = scene[(scene.model.eq(row.model)) & (scene.clustering.eq(row.clustering)) & (scene.refinement.eq(row.refinement))]
    assert abs(q.instance_f1.mean() - row.f1) < 1e-12
    assert abs(q.instance_miou.mean() - row.instance_miou) < 1e-12
    assert abs(q.matched_instances.sum() / q.gt_instances.sum() - row.coverage) < 1e-12
assert abs(main.f1.max() - 0.9082075057618137) < 1e-12

eff_summary = pd.read_csv(EFF / "efficiency_summary.csv")
assert len(eff_summary) == 4
for short in ("randla", "pointnet2", "pointnext", "pointgroup"):
    payload = json.loads((EFF / f"efficiency_fixed_test_{short}.json").read_text(encoding="utf-8"))
    assert len(payload["per_scene"]) == 6
    for row in payload["per_scene"]:
        assert abs(row["adpv_overhead_ms"] - (row["adpv_cluster_ms"] - row["base_cluster_ms"])) < 1e-9
        assert abs(row["total_adpv_ms"] - (row["preprocess_ms"] + row["semantic_ms"] + row["adpv_cluster_ms"])) < 1e-9

for name in ("fig10_efficiency_fixed_test_rerun_20261111.png", "fig10_efficiency_fixed_test_rerun_20261111.pdf"):
    p = R / "figures" / name
    assert p.exists() and p.stat().st_size > 1000

doc = (ROOT / "docs/COMPAG-D-26-10878_Fixed_Cohort_Results_CN.md").read_text(encoding="utf-8-sig")
man = (ROOT / "docs/COMPAG-D-26-10878_Revised_Manuscript_Text.md").read_text(encoding="utf-8-sig")
for text, name in ((doc, "report"), (man, "manuscript")):
    assert "0.9082" in text, name
    assert "0.8709" in text if name == "report" else True
    assert "176.7" in text and "117.6" in text and "14.8" in text, name
assert "0.9464" not in man
assert "0.8329" not in doc[doc.index("## 3."):doc.index("## 4.")]
table_dir = R / "manuscript_tables_20261111"
for i in range(2, 7):
    assert (table_dir / f"Table{i}_main_results.csv").exists() if i == 2 else True
    assert len(list(table_dir.glob(f"Table{i}_*.csv"))) == 1, i
    assert len(list(table_dir.glob(f"Table{i}_*.tex"))) == 1, i
assert (table_dir / "TableS_scene_plant_method_metric_mapping.csv").exists()
assert (table_dir / "TableS_HDBSCAN_results.csv").exists()

print("MAIN_EFFICIENCY_AUDIT_OK")
print(main.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(eff_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
