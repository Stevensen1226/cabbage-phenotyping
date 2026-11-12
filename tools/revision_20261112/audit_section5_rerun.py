from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(r"E:\Cabbage")
R = ROOT / "revision_data/final_v6_ransac/ablation_retrained"
VAL = ["cloudR3", "cloudR6", "cloudR16", "cloudR20"]
TEST = ["cloudR2", "cloudR5", "cloudR9", "cloudR12", "cloudR14", "cloudR18"]
STATES = {"base", "split_only", "full"}
MODELS = {"PointNeXt-L", "PointGroup"}

scene = pd.read_csv(R / "ablation_scene_metrics.csv")
assert len(scene) == 60, len(scene)
assert set(scene.model) == MODELS
assert set(scene.state) == STATES
for split, scenes in (("val", VAL), ("test", TEST)):
    q = scene[scene.split.eq(split)]
    assert set(q.scene_id) == set(scenes)
    for model in MODELS:
        for state in STATES:
            assert len(q[q.model.eq(model) & q.state.eq(state)]) == len(scenes)

summary = pd.read_csv(R / "ablation_summary.csv")
assert len(summary) == 12
for _, row in summary.iterrows():
    q = scene[scene.split.eq(row.split) & scene.model.eq(row.model) & scene.state.eq(row.state)]
    assert abs(q.instance_f1.mean() - row.instance_f1) < 1e-12
    assert abs(q.instance_miou.mean() - row.instance_miou) < 1e-12
    assert abs(q.mae_count.mean() - row.mae_count) < 1e-12

sens = json.loads((R / "sensitivity_multi_cluster_trainval.json").read_text(encoding="utf-8"))
assert sens["n_scenes"] == 17
assert sens["n_clusters"] == 21
assert not set(sens["allowed_scenes"]) & set(sens["excluded_test_scenes"])
assert set(sens["excluded_test_scenes"]) == set(TEST)
assert {x["file"] for x in sens["skipped_scenes"]} == {"cloudR22.ply", "cloudR23.ply"}

sweep = pd.read_csv(R / "parameter_sweep/parameter_sweep_scene_metrics.csv")
assert len(sweep) == 48
assert set(sweep.scene_id) == set(VAL)
assert set(sweep.min_peak_dist_m) == {0.3, 0.35}
assert set(sweep.valley_depth_rel) == {0.85, 0.9, 0.95}

for name in ("fig8_sensitivity_trainval_rerun_20261111.png", "fig8_sensitivity_trainval_rerun_20261111.pdf"):
    assert (ROOT / "revision_data/final_v6_ransac/figures" / name).stat().st_size > 1000

doc_path = ROOT / "docs/COMPAG-D-26-10878_Fixed_Cohort_Results_CN.md"
raw = doc_path.read_bytes()
doc = raw.decode("utf-8-sig")
assert raw.startswith(b"\xef\xbb\xbf")
assert "?" not in doc and "\ufffd" not in doc
required = [
    "\u53d7\u63a7\u6d88\u878d\u4e0e\u53c2\u6570\u9009\u62e9\uff082026-11-11 \u91cd\u8dd1\uff09",
    "0.9033", "0.7098", "0.8602", "min_peak_dist_m", "valley_depth_rel",
    "fig8_sensitivity_trainval_rerun_20261111.png",
]
for token in required:
    assert token in doc, token
for legacy in ("\u6cbf\u7528\u539f\u5b9e\u9a8c", "0.3526", "0.7340", "0.8786", "0.8121"):
    assert legacy not in doc, legacy

print("SECTION5_AUDIT_OK")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
