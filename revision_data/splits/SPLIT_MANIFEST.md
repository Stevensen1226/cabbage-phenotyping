# COMPAG-D-26-10878 固定数据划分

## 最终口径

- 总计：23 个场景 / 365 株
- 训练：13 个场景 / 203 株
- 验证：4 个场景 / 60 株
- 测试：6 个场景 / 102 株
- 测试场景：cloudR2、cloudR5、cloudR9、cloudR12、cloudR14、cloudR18

## 文件

- `paper_split_v2.json`：场景划分
- `train_scenes.txt`、`val_scenes.txt`、`test_scenes.txt`：场景清单
- `scene_plant_counts.csv`：逐场景植株数
- `test_plant_registry.csv`：102 株测试植株的唯一 ID、点数和中心位置
- `gt_audit_365.csv`：23 个场景的 GT 计数审计
- `gt_audit_split_summary.csv`：训练/验证/测试汇总

## 规则

1. 所有方法使用相同划分、相同测试场景和相同 GT。
2. cloudR2、cloudR5、cloudR9、cloudR12、cloudR14、cloudR18 仅用于最终测试。
3. 超参数和 checkpoint 只由训练集与验证集确定。
4. 测试集不参与模型选择或参数回选。
5. Base 与 ADPV 使用完全相同的输入、真值和 checkpoint。
