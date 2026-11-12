# COMPAG-D-26-10878 图片修改审计

## 结论

论文共有 Figure 1–10。按当前 v6 固定测试队列和统一 KNN-R 评估口径：

- 已替换：Figure 7、Figure 8、Figure 10。
- 需要最终核查/定稿：Figure 6。
- 保留为探索性跨作物泛化证据：Figure 9。
- 基本不需要修改：Figure 1、Figure 2、Figure 3、Figure 4、Figure 5。

## 逐图判断

| 图号 | 当前内容 | 状态 | 处理说明 |
|---|---|---|---|
| Figure 1 | 实验地点和无人机采集 | 保留 | 地图、坐标和地区标注当前一致。 |
| Figure 2 | 原始点云、去噪、去地面流程 | 保留 | 属于预处理方法图，与测试队列无关。 |
| Figure 3 | ADPV 总流程 | 保留 | 概念图与最终方法一致；核对参数名即可。 |
| Figure 4 | DPVIS 世界坐标、局部坐标、密度曲线、切割面 | 保留 | 概念图保留。 |
| Figure 5 | KNN-R 前后点云示意 | 保留 | 概念图与最终 ADPV 路径一致。 |
| Figure 6 | 粘连场景定性结果 | 已完成 | 已使用固定测试场景 cloudR18/cloudR14，并标注 K3/K4/K5、plant ID、Base/ADPV；正式图注已写入替换稿。 |
| Figure 7 | 同株配对性状差值 | 已替换 | `fig7_paired_trait_delta_rerun_20261111`；n=50/45/36/36，含 95% CI 和 Holm 校正。 |
| Figure 8 | train+val 参数敏感性 | 已替换 | `fig8_sensitivity_trainval_rerun_20261111`；21 个粘连簇，6 个测试场景全部排除。 |
| Figure 9 | maize/rapeseed 跨作物定性结果 | 已完成 | 已定位为 exploratory cross-crop generalization evidence；强调定性、初步可迁移性，不用于定量或广泛泛化结论，最终排版时放入 supplementary。 |
| Figure 10 | 固定测试集效率 | 已替换 | `fig10_efficiency_fixed_test_rerun_20261111`；4 backbone、6 场景、102 株，包含 KNN-R。 |

## 当前替换文件

- `revision_data/final_v6_ransac/figures/fig7_paired_trait_delta_rerun_20261111.png|pdf`
- `revision_data/final_v6_ransac/figures/fig8_sensitivity_trainval_rerun_20261111.png|pdf`
- `revision_data/final_v6_ransac/figures/fig10_efficiency_fixed_test_rerun_20261111.png|pdf`
- `revision_data/final_v6_ransac/figures/figS_paired_trait_identity_rerun_20261111.png|pdf`

## 当前数值口径

- 主表：26 个配置，固定 6 个测试场景、102 株 GT。
- 最高 ADPV F1：0.9082（RandLA-Net + MeanShift）；PointNet2-RGB + Watershed-3D 为 0.9079。
- 性状：4 个模型 × 4 个性状；配对样本 50/45/36/36。
- 效率：ADPV 增量 117.6–176.7 ms/场景，5.3%–14.8%。
- 测试集未参与 checkpoint、超参数或参数平台选择。

## 最优先处理

1. 在最终期刊模板中把 Figure 9 放入 supplementary，并保留 exploratory cross-crop generalization 图注。
2. 将 Figure 6、7、8、10 和更新表格放入期刊模板。
3. 生成最终 PDF 后再次核对图表号、页数和小数位。
