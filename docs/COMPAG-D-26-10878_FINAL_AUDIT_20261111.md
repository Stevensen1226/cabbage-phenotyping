# COMPAG-D-26-10878 最终一致性审计（2026-11-11；2026-11-12 增补 HDBSCAN）

## 总体状态

实验复核报告、论文替换稿、主表、配对性状、受控消融、参数敏感性、效率、图和投稿材料已统一到同一固定测试队列及修正后的 KNN-R 评估链。

## 关键口径

- 数据：23 场景 / 365 株。
- 划分：13 训练场景 / 203 株；4 验证场景 / 60 株；6 测试场景 / 102 株。
- 测试场景：cloudR2、cloudR5、cloudR9、cloudR12、cloudR14、cloudR18。
- 主表：26 个配置 × 6 场景 = 156 条场景级评价（18 个主要配置 + 8 个 HDBSCAN 配置）。
- 逐株映射：主表 26 个配置 × 102 株 = 2652 行。
- 逐场景/逐株增益：已生成场景 ΔF1、新增匹配、丢失匹配和同株 ΔIoU。
- 性状：4 模型 × 4 性状；配对样本 50/45/36/36。
- 消融：2 模型 × 3 状态 × 2 数据集；共 60 条场景级记录。
- 敏感性：17 个训练/验证场景中的 21 个粘连簇；6 个测试场景全部排除。
- 效率：4 backbone × 6 固定测试场景；ADPV 包含 KNN-R。
- HDBSCAN：已纳入主表，4 backbone × Base/ADPV × 6 测试场景；参数 min_cluster_size=500、min_samples=10、EOM。

## 主要结果

- Watershed-3D 四模型平均 ADPV F1：0.8709；Base：0.4924；提升 +0.3785。
- MeanShift 四模型平均 ADPV F1：0.8725；Base：0.8091；提升 +0.0633。
- 最高 ADPV F1：0.9082（RandLA-Net + MeanShift）；PointNet2-RGB + Watershed-3D 为 0.9079。
- 冠幅直径在四个 backbone 上均显著改善。
- 株高变化均不显著。
- α-volume 在 RandLA-Net、PointNeXt-L、PointGroup 上显著改善；PointNet2-RGB 不显著。
- ADPV 增量：117.6–176.7 ms/场景，5.3%–14.8%；GPU 峰值低于 6.1 GB。
- HDBSCAN Base F1：0.0278–0.0581；HDBSCAN+ADPV F1：0.7967–0.8507。

## 审计结果

- `MAIN_EFFICIENCY_AUDIT_OK`
- `TRAIT_RERUN_AUDIT_OK`
- `SECTION5_AUDIT_OK`
- `SCENE_PLANT_MAPPING_AUDIT_OK`
- `HDBSCAN_AUDIT_OK`
- 报告和论文替换稿：UTF-8 BOM，无 `?` 乱码和 U+FFFD。
- 配对键均在 102 株 registry 内，无重复。
- 旧主表数值、旧性状样本量和旧“株高显著恶化”表述均已清除。

## 核心文件

- 实验复核报告：`docs/COMPAG-D-26-10878_Fixed_Cohort_Results_CN.md`
- 论文替换稿：`docs/COMPAG-D-26-10878_Revised_Manuscript_Text.md`
- Cover letter：`docs/COMPAG-D-26-10878_Cover_Letter.md`
- 主结果与 2652 行映射：`revision_data/final_v6_ransac/main_unified_20261112/`
- 配对性状：`revision_data/final_v6_ransac/paired_rerun_20261111/`
- 消融与敏感性：`revision_data/final_v6_ransac/ablation_retrained/`
- HDBSCAN 补充基线：`revision_data/final_v6_ransac/hdbscan_20261112/`
- 效率：`revision_data/final_v6_ransac/efficiency_20261111/`
- 图表：`revision_data/final_v6_ransac/figures/`
- 投稿表：`revision_data/final_v6_ransac/manuscript_tables_20261111/`

## 仍需人工完成

1. 在最终期刊模板中把 Figure 9 放入 supplementary，并保留 exploratory cross-crop generalization 图注。
2. 将替换稿、表和 Figure 6/7/8/10 放入期刊可编辑模板。
3. 生成最终 PDF 后再次核对图表号、页数和小数位。
