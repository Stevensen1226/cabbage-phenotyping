from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(r"E:\Cabbage")
R = ROOT / "revision_data/final_v6_ransac/paired_rerun_20261111"
REG = pd.read_csv(ROOT / "revision_data/splits/test_plant_registry.csv")
FLOW = pd.read_csv(R / "pairing_flow.csv")
PAIRS = pd.read_csv(R / "v6_paired_trait_deltas.csv")
STATS = pd.read_csv(R / "v6_paired_trait_statistics.csv")

expected = {"RandLA-Net": 50, "PointNet2-RGB": 45, "PointNeXt-L": 36, "PointGroup": 36}
assert len(FLOW) == 4
assert dict(zip(FLOW.model, FLOW.paired_plants)) == expected
assert (FLOW.paired_plants == FLOW.paired_plants_in_registry).all()
assert len(STATS) == 16
assert set(STATS.trait) == {"height_cm", "crown_diameter_mean_cm", "volume_alpha_cm3", "compactness"}
assert dict(STATS.groupby("model").size()) == {k: 4 for k in expected}

registry_keys = set(zip(REG.scene_id, REG.gt_instance_id))
keys = set(zip(PAIRS.scene_id, PAIRS.gt_instance_id))
assert keys <= registry_keys, keys - registry_keys
assert not PAIRS.duplicated(["model", "scene_id", "gt_instance_id", "trait"]).any()
for model, n in expected.items():
    q = PAIRS[PAIRS.model.eq(model)]
    for trait in STATS[STATS.model.eq(model)].trait:
        tq = q[q.trait.eq(trait)]
        row = STATS[(STATS.model.eq(model)) & (STATS.trait.eq(trait))].iloc[0]
        assert len(tq) == int(row.n) == n
        assert abs(tq.delta_abs_error.mean() - row.mean_delta_abs_error_adpv_minus_base) < 1e-9
        assert abs(tq.base_abs_error.mean() - row.mean_base_abs_error) < 1e-9
        assert abs(tq.adpv_abs_error.mean() - row.mean_adpv_abs_error) < 1e-9

figs = [
    ROOT / "revision_data/final_v6_ransac/figures/fig7_paired_trait_delta.png",
    ROOT / "revision_data/final_v6_ransac/figures/fig7_paired_trait_delta.pdf",
    ROOT / "revision_data/final_v6_ransac/figures/figS_paired_trait_identity.png",
    ROOT / "revision_data/final_v6_ransac/figures/figS_paired_trait_identity.pdf",
]
stats_mtime = (R / "v6_paired_trait_statistics.csv").stat().st_mtime
for fig in figs:
    assert fig.exists() and fig.stat().st_size > 1000
    assert fig.stat().st_mtime >= stats_mtime - 1

doc_path = ROOT / "docs/COMPAG-D-26-10878_Fixed_Cohort_Results_CN.md"
raw = doc_path.read_bytes()
doc = raw.decode("utf-8-sig")
assert raw.startswith(b"\xef\xbb\xbf")
assert "?" not in doc and "\ufffd" not in doc
required = ["-5.533", "-4.063", "-4.826", "-4.445", "-355.478", "-143.751", "-180.271", "-370.771", "50/45/36/36"]
for token in required:
    assert token in doc, token
for legacy in ["-329.643", "-3.637", "-115.659"]:
    assert legacy not in doc, legacy

print("TRAIT_RERUN_AUDIT_OK")
print(FLOW.to_string(index=False))
print(STATS[["model", "trait", "n", "mean_delta_abs_error_adpv_minus_base", "ci95_low", "ci95_high", "holm_adjusted_p"]].to_string(index=False))
