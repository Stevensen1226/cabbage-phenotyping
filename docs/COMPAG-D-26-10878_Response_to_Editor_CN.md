# 致编辑的回复

**稿件题目：** Model-Agnostic Adaptive Density Peak–Valley Refinement for Densely Planted Cabbage Point Cloud Instance Segmentation  
**前稿编号：** COMPAG-D-26-10878  
**日期：** 2026 年 11 月 12 日

尊敬的编辑：

感谢您对稿件的审核意见。我们没有仅以邮件形式进行辩解，而是准备了新的投稿稿件，并重新完成了实验、图表、表格和支持材料，使两项核心意见都可以被直接核查。Cover letter 中已写明前稿编号 **COMPAG-D-26-10878**。

## 一、逐条回复

### 意见 1：验证口径不一致

> 稿件的方法和结果部分使用了不同的测试队列/测试场景，需要建立从测试场景到植株的一致映射，使读者能够核查分割性能提升来自哪些场景和植株。

**回复**

我们已围绕一个冻结的测试队列和统一评价流程重建全部验证实验。

- 数据集共 23 个场景、365 株甘蓝。
- 固定划分如下：
  - 13 个训练场景 / 203 株；
  - 4 个验证场景 / 60 株；
  - 6 个测试场景 / 102 株。
- 6 个测试场景为：cloudR2、cloudR5、cloudR9、cloudR12、cloudR14、cloudR18。
- 全部 26 个模型/配置组合均使用相同的 6 个测试场景、相同的 102 株 registry、相同的标注和统一的 KNN-R 实例评价流程。
- checkpoint、超参数和后处理参数仅使用训练集和验证集确定，测试集不参与模型选择或调参。

我们新增了完整的场景—植株—方法—指标映射，包含：

- 26 个配置 × 102 株 = **2652 行**；
- `scene_id`、`plant_id`、`gt_instance_id`；
- 模型、聚类方法、Base/ADPV 配置；
- 是否匹配、预测实例 ID、逐株 IoU；
- 对应场景的 F1、instance mIoU 和 count MAE。

同时补充了逐场景和逐株增益分析。Watershed-3D 的四个模型平均 F1 从 0.4924 提高到 0.8709；MeanShift 的四个模型平均 F1 从 0.8091 提高到 0.8725。最高 ADPV F1 为 0.9082，来自 RandLA-Net + MeanShift；PointNet2-RGB + Watershed-3D 为 0.9079。

逐场景分析还明确报告了 refinement 无效或负增益的情况：

- PointGroup + MeanShift 在 cloudR9 和 cloudR14 上 F1 下降；
- PointNet2-RGB + MeanShift 在 cloudR2 和 cloudR9 上下降；
- RandLA-Net + MeanShift 在 cloudR9 和 cloudR14 上下降；
- Watershed-3D 在 PointGroup、PointNet2-RGB 和 PointNeXt-L 上均为 6/6 场景提升；RandLA-Net 为 5 个场景提升、1 个持平、无下降。

**支持文件**

- `revision_data/final_v6_ransac/main_unified_20261112/scene_method_metrics.csv`
- `revision_data/final_v6_ransac/main_unified_20261112/scene_plant_method_metrics.csv`
- `revision_data/final_v6_ransac/main_unified_20261112/scene_gain_table.csv`
- `revision_data/final_v6_ransac/main_unified_20261112/plant_gain_table.csv`
- `revision_data/splits/paper_split_v2.json`
- `revision_data/splits/test_plant_registry.csv`

### 意见 2：性状比较前后使用了不同植株组

> 校正前后性状比较使用不同植株组，无法判断性能提升来自 refinement 还是植株组差异。需要在完全相同的植株样本上做校正前后配对比较。

**回复**

我们同意该意见，已删除旧的非配对性状比较。新的分析严格使用相同的 `(scene_id, plant_id, gt_instance_id)` 键进行 Base/ADPV 配对。

纳入标准为：

1. 同一植株在 Base 和 ADPV 下均达到 IoU ≥ 0.5；
2. 四项性状在两侧均有效；
3. 预测植株和参考植株均具有足够点数。

样本流转如下：

| 模型 | Base 匹配 | ADPV 匹配 | 最终配对 |
|---|---:|---:|---:|
| RandLA-Net | 50 | 85 | 50 |
| PointNet2-RGB | 45 | 91 | 45 |
| PointNeXt-L | 36 | 73 | 36 |
| PointGroup | 36 | 87 | 36 |

每一项性状均报告：

- ADPV 减 Base 的平均配对差值；
- bootstrap 95% 置信区间；
- paired t-test；
- Wilcoxon 检验；
- Holm 校正后的 p 值；
- Cohen's dz；
- 明确的配对样本量。

最新结论仅限制在配对子集内：

- 四个 backbone 的冠幅直径误差均显著下降；
- 四个 backbone 的株高变化均不显著；
- α-shape 体积方向均下降，其中 RandLA-Net、PointNeXt-L 和 PointGroup 在 Holm 校正后显著；PointNet2-RGB 不显著；
- 紧凑度仅 RandLA-Net 和 PointNeXt-L 在 Holm 校正后显著改善；
- 不再声称所有性状一致改善，也不再声称 RandLA-Net 株高显著变差。

**支持文件**

- `revision_data/final_v6_ransac/paired_rerun_20261111/v6_paired_trait_statistics.csv`
- `revision_data/final_v6_ransac/paired_rerun_20261111/v6_paired_trait_deltas.csv`
- `revision_data/final_v6_ransac/paired_rerun_20261111/pairing_flow.csv`
- `revision_data/final_v6_ransac/figures/fig7_paired_trait_delta_rerun_20261111.png`
- `revision_data/final_v6_ransac/figures/figS_paired_trait_identity_rerun_20261111.png`

## 二、对编辑建议的其他修改

### 1. 固定数据划分并防止数据泄漏

方法部分已明确说明 13/4/6 的场景级划分，并列出完整场景清单。同一个场景不会同时出现在训练、验证和测试集中，同一场景的全部植株只属于一个划分。测试场景仅用于最终评价。

### 2. 同时报告总体、逐场景和逐株结果

结果部分现已包括：

- 固定测试集的总体指标；
- 每个场景的 ΔF1、Δinstance mIoU 和 ΔMAE；
- 每个模型/聚类配置中提升、持平和下降的场景数量；
- 新增匹配植株、丢失匹配植株和两侧均匹配植株的同株 ΔIoU；
- MeanShift 的明确失效场景。

### 3. 重跑受控消融和参数选择

受控消融已按固定划分重跑，包含 Base、仅拆分、拆分+回并三种状态。参数敏感性使用 17 个训练/验证场景中的 21 个粘连簇，6 个测试场景全部排除。

最终保留 `min_peak_dist_m = 0.30 m`，其依据是完整密度下的计数误差和端到端验证 F1。`valley_depth_rel = 0.85–0.95` 在端到端验证上形成平台，最终保留 0.90。

### 4. 重新测量效率

效率在相同的 6 个测试场景和 102 株 registry 上重新测量，并包含完整的 ADPV 路径和 KNN-R。ADPV 额外耗时为 117.6–176.7 ms/场景，相对完整流水线增加 5.3%–14.8%。

### 5. HDBSCAN 主比较结果

我们还将 HDBSCAN 纳入主比较，使用最终混合 checkpoint，在相同的 6 个测试场景和 102 株 registry 上评价。HDBSCAN Base 的 F1 为 0.0278–0.0581，加入 ADPV 后提高到 0.7967–0.8507。最终主表包含 26 个配置，并提供 2652 行场景—植株—方法—指标映射。

### 6. 图表和表格更新

- Figure 6 使用固定测试场景，并标注 scene ID、cluster ID 和 plant ID；
- Figure 7 改为同株配对性状差值图，包含 95% CI 和 Holm 校正；
- Figure 8 使用仅来自训练集和验证集的参数敏感性结果；
- Figure 9 作为探索性跨作物泛化观察保留，用于展示方法在 maize/rapeseed 上的定性可迁移性；这些样本不属于固定 23 场景 cabbage 队列，也未参与调参或定量模型选择，因此仅作为初步观察，不用于支持广泛泛化结论；
- Figure 10 使用固定测试集效率，并包含完整 ADPV/KNN-R；
- Table 2–6 和补充映射/增益表均已从统一评价结果重新生成。

### 6. Cover letter 与新投稿说明

Cover letter 已写明前稿编号，并逐项说明测试队列、同株配对性状、消融、参数选择、效率和图表修改。本次提交为新稿，不是仅针对审稿意见的邮件回复。

## 三、主要修改总结

1. 所有 26 个配置使用同一个固定 6 场景 / 102 株测试队列。
2. 提供 26 × 102 = 2652 行完整场景—植株—方法—指标映射。
3. 明确报告总体、逐场景和逐株增益，包括负增益案例。
4. 性状比较使用完全相同的配对植株。
5. 报告配对差值、置信区间、统计检验、样本量和纳入/排除标准。
6. 结论限制在已评价的队列、backbone、场景和性状范围内。

我们希望修改后的稿件和支持材料能够直接回应编辑的核心关切，并提供足够的可核查性。

此致

敬礼

作者
