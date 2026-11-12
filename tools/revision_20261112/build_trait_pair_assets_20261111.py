from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / "revision_data"
BASE = REV / "final_v6_ransac/paired_rerun_20261111"
RAW = BASE / "raw_traits"
PAIRED = BASE / "paired"
PAIRED.mkdir(parents=True, exist_ok=True)
REGISTRY = REV / "splits/test_plant_registry.csv"

MODELS = {
    "RandLA-Net": "randla_net",
    "PointNet2-RGB": "pointnet2_rgb",
    "PointNeXt-L": "pointnext_l",
    "PointGroup": "pointgroup",
}

pair_frames = []
stat_rows = []
flow_rows = []
for model, slug in MODELS.items():
    base = RAW / f"{slug}_watershed_base.csv"
    adpv = RAW / f"{slug}_watershed_adpv.csv"
    if not base.exists() or not adpv.exists():
        raise FileNotFoundError((model, base.exists(), adpv.exists()))
    prefix = f"v6_{slug}"
    cmd = [
        sys.executable,
        str(REV / "tools/paired_abs_error_stats.py"),
        "--base", str(base),
        "--adpv", str(adpv),
        "--registry", str(REGISTRY),
        "--output-dir", str(PAIRED),
        "--prefix", prefix,
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    pairs = pd.read_csv(PAIRED / f"{prefix}.csv")
    pairs["model"] = model
    pair_frames.append(pairs)
    payload = json.loads((PAIRED / f"{prefix}_statistics.json").read_text(encoding="utf-8"))
    for stat in payload["statistics"]:
        row = {"model": model}
        row.update(stat)
        stat_rows.append(row)
    flow_rows.append({
        "model": model,
        "gt_plants": payload["gt_plants"],
        "base_matched": payload["base_matched"],
        "adpv_matched": payload["adpv_matched"],
        "paired_plants": payload["paired_plants"],
        "paired_plants_in_registry": payload["paired_plants_in_registry"],
    })

all_pairs = pd.concat(pair_frames, ignore_index=True)
all_pairs.to_csv(BASE / "v6_paired_trait_deltas.csv", index=False)
stats = pd.DataFrame(stat_rows)
stats.to_csv(BASE / "v6_paired_trait_statistics.csv", index=False)
flow = pd.DataFrame(flow_rows)
flow.to_csv(BASE / "pairing_flow.csv", index=False)
(BASE / "RUN_SUMMARY.json").write_text(
    json.dumps({
        "raw_traits": str(RAW),
        "paired_deltas": str(BASE / "v6_paired_trait_deltas.csv"),
        "statistics": str(BASE / "v6_paired_trait_statistics.csv"),
        "registry": str(REGISTRY),
        "models": list(MODELS),
    }, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(flow.to_string(index=False))
print(stats[["model", "trait", "n", "mean_delta_abs_error_adpv_minus_base", "ci95_low", "ci95_high", "holm_adjusted_p"]].to_string(index=False))
