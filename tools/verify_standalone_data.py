"""
验证 plot_hyperparam_standalone.py 中硬编码的数据是否与原始
hyperparam_summary.json 一致。

对比项:
  - VDR_GRID / MPD_GRID
  - VDR_MAE / MPD_MAE (MAE(count) 矩阵)
  - VDR_PARTS / MPD_PARTS (平均分块数矩阵)
  - RATIOS

用法:
  python tools/verify_standalone_data.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(ROOT, "output", "density_sensitivity",
                         "single_cluster", "hyperparam", "hyperparam_summary.json")

# 从打包脚本动态导入硬编码常量
sys.path.insert(0, os.path.join(ROOT, "tools"))
from plot_hyperparam_standalone import (
    RATIOS, VDR_GRID, VDR_MAE, VDR_PARTS,
    MPD_GRID, MPD_MAE, MPD_PARTS,
    VDR_DEFAULT, MPD_DEFAULT, K,
)


def close(a, b, tol=1e-4):
    """比较两个数值 (或数组) 是否在容差内一致。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.allclose(a, b, atol=tol, rtol=1e-3)


def main():
    with open(JSON_PATH) as f:
        data = json.load(f)

    print("=" * 60)
    print("验证 plot_hyperparam_standalone.py 硬编码数据 vs 原始 json")
    print("=" * 60)

    errors = []

    # 1. RATIOS
    if not close(RATIOS, data["ratios"]):
        errors.append("RATIOS 不一致")
    print(f"  RATIOS (8 密度): {'✓ 一致' if close(RATIOS, data['ratios']) else '✗ 不一致'}")

    # 2. valley_depth_rel
    vdr = data["valley_depth_rel"]
    print(f"\n  [valley_depth_rel] 默认值 {VDR_DEFAULT:.2f}")
    print(f"    grid: {'✓' if close(VDR_GRID, vdr['grid']) else '✗'}")
    if not close(VDR_GRID, vdr["grid"]):
        errors.append("VDR_GRID 不一致")
        print(f"      脚本: {VDR_GRID}")
        print(f"      json : {vdr['grid']}")

    cm = np.array(vdr["mae_matrix"])
    print(f"    mae_matrix {cm.shape}: "
          f"{'✓' if close(VDR_MAE, cm) else '✗'}")
    if not close(VDR_MAE, cm):
        errors.append("VDR_MAE 不一致")
        diff = np.abs(np.array(VDR_MAE) - cm)
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"      最大差异 {diff.max():.6f} 于 {idx}")

    pm = np.array(vdr["parts_matrix"])
    print(f"    parts_matrix   {pm.shape}: "
          f"{'✓' if close(VDR_PARTS, pm) else '✗'}")
    if not close(VDR_PARTS, pm):
        errors.append("VDR_PARTS 不一致")

    # 3. min_peak_dist_m
    mpd = data["min_peak_dist_m"]
    print(f"\n  [min_peak_dist_m] 默认值 {MPD_DEFAULT:.2f}")
    print(f"    grid: {'✓' if close(MPD_GRID, mpd['grid']) else '✗'}")
    if not close(MPD_GRID, mpd["grid"]):
        errors.append("MPD_GRID 不一致")
        print(f"      脚本: {MPD_GRID}")
        print(f"      json : {mpd['grid']}")

    cm = np.array(mpd["mae_matrix"])
    print(f"    mae_matrix {cm.shape}: "
          f"{'✓' if close(MPD_MAE, cm) else '✗'}")
    if not close(MPD_MAE, cm):
        errors.append("MPD_MAE 不一致")
        diff = np.abs(np.array(MPD_MAE) - cm)
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"      最大差异 {diff.max():.6f} 于 {idx}")

    pm = np.array(mpd["parts_matrix"])
    print(f"    parts_matrix   {pm.shape}: "
          f"{'✓' if close(MPD_PARTS, pm) else '✗'}")
    if not close(MPD_PARTS, pm):
        errors.append("MPD_PARTS 不一致")

    # 4. K 真值
    print(f"\n  K (真值株数) = {K}: {'✓' if K == 4 else '✗ 应等于 4'}")

    print("\n" + "=" * 60)
    if errors:
        print(f"❌ 发现 {len(errors)} 处不一致: {errors}")
        sys.exit(1)
    else:
        print("✅ 所有硬编码数据与原始 json 完全一致")
        sys.exit(0)


if __name__ == "__main__":
    main()
