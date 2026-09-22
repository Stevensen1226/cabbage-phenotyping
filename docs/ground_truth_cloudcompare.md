# 测试集真值规范（CloudCompare，二分类）

目标：为每个点提供语义真值 `semantic_label`，并且**点顺序与点数量不变**，使 [evaluate.py](../evaluate.py) 能逐点对齐评估。

## 1. 标签约定（二分类）

- `semantic_label = 0`：背景（包含地面、杂草、非卷心菜等）
- `semantic_label = 1`：卷心菜（cabbage）

> 本项目当前的实例评估逻辑也默认“卷心菜类为 1”。如果你未来改成多分类，需要同步修改 [evaluate.py](../evaluate.py)。

## 1.1（C 级）点级实例标签约定（强烈推荐用于“计数+表型”论文）

当你要评估“单株计数误差、实例 Precision/Recall/F1、单株表型误差”时，建议额外制作点级实例真值：

- `instance_id = -1`：背景/非卷心菜点（不参与实例评估）
- `instance_id = 0..K-1`：每棵卷心菜一个唯一 ID（连续更方便，但非必须）

如果你的 CloudCompare 不方便写入 `-1`，也可用：

- `instance_id = 0`：背景
- `instance_id = 1..K`：实例

并在评估时加参数 `--gt-instance-bg-value 0`。

## 2. 文件命名（evaluate.py 会自动查找）

输入点云：
- `data/test/cloudR1.ply`

对应真值（推荐其一）：
- `data/test/cloudR1_gt.txt`
- `data/test/cloudR1_gt.npy`

## 3. 真值文件格式

### 3.1 TXT（最推荐，最易从 CloudCompare 导出）
每行至少 4 列：

```
X Y Z semantic_label
```

其中 `semantic_label` 必须是整数 0/1。

如果包含实例列（C 级），推荐格式：

```
X Y Z semantic_label instance_id
```

但 CloudCompare 常见会导出更多列（例如 `X Y Z R G B semantic_label instance_id`）。此时在评估命令里用 `--gt-label-col` / `--gt-instance-col` 指定对应列即可。

### 3.2 NPY（可选）
二维数组，形状 `(N, 4)`：

- 第 1-3 列：`x, y, z`
- 第 4 列：`semantic_label`（0/1）

## 4. CloudCompare 制作流程（保证点顺序不变）

关键原则：
- 不要在标注过程中对点云做重采样/下采样/体素化/合并/删除点。
- 不要用“裁剪后只保存选中点”的方式做 GT（那会改变点数量）。
- 推荐做法：在**原始点云对象上**新增一个标量场（Scalar Field）作为标签列，然后仅修改其数值。

### 4.1 导入点云
1. 打开 CloudCompare
2. `File -> Open` 导入 `cloudR1.ply`（或你的点云）

### 4.2 创建标签标量场（把所有点先置为 0）
不同版本菜单名字可能略有差异，但思路相同：
- 新增一个 Scalar Field，给全体点赋常数值 0
- 将该 Scalar Field 命名为例如 `semantic_label`

如果你找不到“全体赋值”的菜单，退而求其次：
- 先创建一个 SF（初值任意）
- 之后导出时用脚本把“非 1 的全部当 0”（见第 5 节）

### 4.3 将卷心菜点置为 1
1. 使用 CloudCompare 的分割/选择工具把卷心菜点选出来（常用的是剪刀/多边形分割等交互工具）
2. 对“当前选中点”把 `semantic_label` 设为常数 1

> 你可以分多次选择：每次选中一部分卷心菜点，就把选中点的标签设为 1。

### 4.4 导出（必须导出全体点 + 标签列）
推荐导出为 ASCII：
- `File -> Save`
- 选择 ASCII/TXT
- 勾选/选择导出字段至少包含：`X Y Z semantic_label`

注意：
- 必须导出**全体点**，不要只导出“选中点”。
- 导出后建议用文本编辑器抽查几行，确认第 4 列确实是 0/1。

## 5. 用项目自带脚本把 CloudCompare 导出转成可评估 GT

见脚本 [tools/gt_from_cloudcompare.py](../tools/gt_from_cloudcompare.py)。典型用法：

- 如果导出 TXT 第 4 列就是标签（0/1）：

```
python tools/gt_from_cloudcompare.py --input data/test/cloudR1_cc.txt --output data/test/cloudR1_gt.txt --label-col 3
```

- 如果标签列不是严格 0/1（例如卷心菜标成 100），脚本会把 `!=0` 的都当作 1。

## 6. 评估时的一个重要配置建议

为了确保 GT 与预测严格逐点对齐，评估时建议关闭会“删除点”的预处理（例如 SOR 离群点移除）。否则点数变化会导致无法逐点对比。

如果你希望继续自动评估并保持对齐，有两条路：
- 路线 A（推荐）：评估配置里关闭删点操作
- 路线 B：在评估脚本里做 KDTree 最近邻对齐（会引入匹配误差，论文里要说明）

## 7. 评估命令示例（CloudCompare 导出为 XYZRGB + semantic + instance）

假设导出列为：`X Y Z R G B semantic instance`，则：

- `semantic` 在第 7 列（0-based: 6）
- `instance` 在第 8 列（0-based: 7）

评估命令：

```
python evaluate.py --input data/test --config configs/default.yaml --gt-label-col 6 --gt-instance-col 7 --gt-binarize
```

如果实例背景是 0（不是 -1）：

```
python evaluate.py --input data/test --config configs/default.yaml --gt-label-col 6 --gt-instance-col 7 --gt-binarize --gt-instance-bg-value 0
```
