# GT 计数审计（COMPAG-D-26-10878）

## 审计结论

| 划分 | 场景数 | 有效甘蓝实例数 | 无效语义实例点 | GT 来源 |
|---|---:|---:|---:|---|
| 训练 | 13 | 203 | 0 | 人工 GT |
| 验证 | 4 | 60 | 0 | 人工 GT |
| 测试 | 6 | 102 | 0 | 人工 GT |
| 合计 | 23 | 365 | 0 | 人工 GT |

`cloudR22`（44,322 点 / 5 株）和 `cloudR23`（252,127 点 / 27 株）为独立人工 GT，不再使用重复回退文件。

## 固定测试集

cloudR2（8 株）、cloudR5（9）、cloudR9（20）、cloudR12（30）、cloudR14（19）、cloudR18（16），合计 102 株。

逐株 registry：

- `revision_data/splits/test_plant_registry.csv`
- 102 行、102 个唯一 plant_id
- 字段：scene_id、plant_id、gt_instance_id、num_points、center_x、center_y、center_z

## 一致性检查

- 训练、验证和测试场景无重叠。
- 6 个测试场景均有可读的人工 GT。
- 所有 Base/ADPV 和全部方法使用同一测试植株 registry。
- 测试集不参与超参数或 checkpoint 选择。
