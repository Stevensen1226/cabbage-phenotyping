from pathlib import Path
from copy import deepcopy
import yaml

ROOT = Path(r"E:\Cabbage")
REV = ROOT / "revision_data"
CFG = REV / "final_v6_ransac/configs"
RETRAIN = CFG / "ransac_retrain_20261111"
OUT = CFG / "hdbscan_20261112"
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "randla_net": {
        "base": RETRAIN / "randla_net_watershed_base_retrained.yaml",
        "adpv": RETRAIN / "randla_net_watershed_adpv_retrained.yaml",
    },
    "pointnet2_rgb": {
        "base": CFG / "pointnet2_rgb_watershed_base_v6.yaml",
        "adpv": CFG / "pointnet2_rgb_watershed_adpv_v6.yaml",
    },
    "pointnext_l": {
        "base": RETRAIN / "pointnext_l_watershed_base_retrained.yaml",
        "adpv": RETRAIN / "pointnext_l_watershed_adpv_retrained.yaml",
    },
    "pointgroup": {
        "base": CFG / "pointgroup_watershed_base_v6.yaml",
        "adpv": CFG / "pointgroup_watershed_adpv_v6.yaml",
    },
}
HDBSCAN = {
    "min_cluster_size": 500,
    "min_samples": 10,
    "cluster_selection_method": "eom",
    "fallback_leaf": False,
    "allow_single_cluster": False,
}
for model, refs in SOURCES.items():
    for ref, src in refs.items():
        cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
        cfg["instance"]["method"] = "hdbscan"
        cfg["instance"]["hdbscan"] = deepcopy(HDBSCAN)
        dst = OUT / f"{model}_hdbscan_{ref}.yaml"
        dst.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(dst)
