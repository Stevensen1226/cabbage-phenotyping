# HDBSCAN 指标复核（2026-11-12）

## 设置

- 新模型口径：RandLA-Net 和 PointNeXt-L 使用 RANSAC 新 checkpoint；PointNet2-RGB 和 PointGroup 使用旧 v6 checkpoint。
- 固定 6 个测试场景 / 102 株 registry。
- HDBSCAN：`min_cluster_size=500`、`min_samples=10`、`cluster_selection_method=eom`。
- Base 关闭 PCA 拆分与碎片投票；ADPV 使用最终拆分、回并和 KNN-R。

## 总体指标

| 模型 | HDBSCAN Base F1 | HDBSCAN ADPV F1 | ΔF1 | ADPV mIoU | ADPV MAE | Coverage | Watershed ADPV F1 | MeanShift ADPV F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RandLA-Net | 0.0581 | 0.8479 | +0.7899 | 0.7585 | 2.67 | 76.5% | 0.8777 | 0.9082 |
| PointNet2-RGB | 0.0278 | 0.8507 | +0.8229 | 0.7541 | 3.17 | 73.5% | 0.9079 | 0.8631 |
| PointNeXt-L | 0.0534 | 0.7971 | +0.7437 | 0.7232 | 3.00 | 69.6% | 0.7946 | 0.8586 |
| PointGroup | 0.0581 | 0.7967 | +0.7387 | 0.7406 | 4.33 | 67.6% | 0.9033 | 0.8599 |

## 结论

- HDBSCAN Base 在固定测试集上几乎无法恢复植株，F1 仅 0.0278–0.0581。
- ADPV 对 HDBSCAN 有显著补救作用，F1 提升到 0.7967–0.8507，说明拆分和回并对 HDBSCAN 的合并簇/碎片问题有效。
- 但 HDBSCAN+ADPV 仍低于 Watershed 或 MeanShift 的大部分对应结果；只有 PointNeXt-L 的 HDBSCAN+ADPV F1 略高于 Watershed+ADPV（0.7971 vs 0.7946）。

## 主表与文件

- 主表合并结果：`main_unified_20261112/main_results_for_report.csv`
- `scene_method_metrics.csv`
- `scene_plant_method_metrics.csv`（816 行 = 4 模型 × 2 配置 × 102 株）
- `hdbscan_summary_for_review.csv`
- `hdbscan_vs_other_clustering.csv`
