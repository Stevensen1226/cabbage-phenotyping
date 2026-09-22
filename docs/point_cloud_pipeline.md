# 甘蓝点云处理流程

本项目实现了一套完整的甘蓝（卷心菜）田间点云处理管道，从原始点云输入到最终单株表型参数提取，包含预处理、语义分割、实例分割、后处理与表型计算五个阶段。

## 整体流程图

```
原始点云 (.ply/.pcd)
       │
       ▼
┌──────────────────────────────┐
│  Stage 0: 预处理              │
│  · SOR 统计离群点去除         │
│  · RANSAC 地面点云剔除        │
│  → preprocess_point_cloud()  │
└──────────────┬───────────────┘
               │ 非地面点云
               ▼
┌──────────────────────────────┐
│  Stage 1: 语义分割            │
│  从场景中分离甘蓝植株点        │
│                               │
│  Backbone 可选:               │
│  · PointGroup (U-Net+稀疏卷积)│
│  · PointNet2 (稀疏卷积U-Net)  │
│  · RandLA-Net (随机采样+注意) │
│  → 输出 甘蓝掩码 + offset    │
└──────────────┬───────────────┘
               │ 甘蓝点云
               ▼
┌──────────────────────────────┐
│  Stage 2: 实例分割            │
│  将甘蓝点云聚类为单株实例      │
│                               │
│  根据 seg_method 分为:        │
│  · hybrid: PG提案+聚类+拆分  │
│  · pointgroup: 纯PG提案      │
│  · clustering: 纯传统聚类     │
│  · none: 跳过分割            │
└──────────────┬───────────────┘
               │ 实例标签
               ▼
┌──────────────────────────────┐
│  Stage 3: 后处理              │
│  · 骨架拆分 (PCA split)       │
│  · 碎片合并 (merge fragments) │
│  · 小碎片丢弃 (< min_points)  │
│  · 噪声点最近邻归入           │
└──────────────┬───────────────┘
               │ 最终实例
               ▼
┌──────────────────────────────┐
│  Stage 4: 表型提取            │
│  · 株高 (height_cm)           │
│  · 冠幅 (crown_diameter_cm)   │
│  · 体积 (volume_voxel_cm3)    │
│  · 紧凑度 (compactness)       │
│  → 输出 plants.json / .csv   │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Stage 5: 可视化 & 保存       │
│  · 实例着色点云 (.ply)        │
│  · 各阶段中间结果             │
└──────────────────────────────┘
```

---

## Stage 0: 预处理

**入口函数**: `cabbage_pheno/service/pipeline.py → preprocess_point_cloud()`

| 步骤 | 方法 | 关键参数 |
|------|------|----------|
| **SOR 去噪** | Statistical Outlier Removal | `sor_neighbors=10`, `sor_std_ratio=3.5` |
| **地面去除** | RANSAC 平面拟合 | `fit_percentile=15` (用最低15%点拟合), `ransac_thresh=0.015m`, `remove_thresh=0.02m` |

处理流程：
1. 对原始点云执行 SOR 滤波，去除离群噪声点
2. 取 Z 坐标最低 `fit_percentile%` 的点拟合地平面（RANSAC）
3. 移除距地平面 `< remove_thresh` 的点
4. 输出干净的非地面点云 `non_ground_pcd`

---

## Stage 1: 语义分割

根据配置中的 `segmentation.backbone` 和 `segmentation.method` 选择不同的分割策略。

### 1.1 大田地分块推理 (`block_inference`)

对于大范围田地（最大边长 > `max_single_extent`，默认 8m），自动将点云切分为重叠块进行推理：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `block_size` | 2.0 m | 每块边长 |
| `overlap` | 0.5 m | 块间重叠宽度 |
| `merge_iou_thresh` | 0.25 | 跨块实例合并 IoU 阈值 |

- **语义合并**: 对重叠区域采用多数投票（cabbage 投票 ≥ 50% 即判定为甘蓝）
- **实例合并**: 通过 Union-Find 算法，对 IoU ≥ `merge_iou_thresh` 的跨块 proposal 进行合并

### 1.2 支持的 Backbone 与分割模式

| Backbone | 方法 (method) | 语义输出 | 实例输出 | 说明 |
|----------|--------------|----------|----------|------|
| `pointnet2` | `hybrid` | PointGroup 语义 logits | PG proposals + offset | **默认模式** |
| `pointnet2` | `softgroup` | SoftGroup 语义 logits | SG 贪心选择 proposals | 自顶向下贪心 |
| `pointnet2` | `pointgroup` | PointGroup 语义 logits | NMS proposals | 纯 PG，不做二次聚类 |
| `pointnet2` | `clustering` | PointGroup 语义 logits | 传统聚类 | 语义筛点后聚类 |
| `pointnet2` | `none` | 无 | 无 | 跳过分割，全点作为单实例 |
| `pointnet2` | `pointnet2` | PointNet2 纯语义 | 传统聚类 | backbone 对比实验 |
| `randlanet` | `clustering` | RandLA-Net 语义 | 传统聚类 | RandLA-Net backbone 对比 |
| `randlanet` | `randlanet` | RandLA-Net 语义 | 传统聚类 | 同上 |

**PointGroup 前向传播** (`_pointgroup_forward`):
1. 坐标缩放: `xyz_scaled = xyz × scale`（voxel_size = 1/scale，默认 2cm）
2. 体素化: `voxelization_idx()` 将点云转为稀疏体素
3. U-Net 推理: 输出 `semantic_scores (N, 2)`, `pt_offsets (N, 3)`, `proposal_scores`
4. 语义阈值过滤 (`pg_sem_thresh`): softmax 概率 > threshold 判定为甘蓝

---

## Stage 2: 实例分割

### 2.1 Hybrid 模式（默认，`segmentation.method = hybrid`）

四阶段渐进式实例分割：

```
PointGroup proposal
       │
       ▼
┌──────────────────┐
│ Step 1: PG 提案   │  NMS/Greedy/IACh 筛选
│ → inst_init      │  已分配点保留, 未分配点进入下一步
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Step 2: 邻近归入  │  未分配点 → cKDTree 最近邻
│ r < expand_radius│  距离 < 阈值 → 归入最近提案
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Step 3: 补充聚类  │  剩余未分配点 → 密度过滤
│ · 半径离群过滤    │  → 配置聚类算法 (watershed_3d 等)
│ · 主聚类 + DBSCAN │  → DBSCAN 兜底
│ · 密度校验        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Step 4: 合并      │  PG提案 + 新聚类簇 → 最终实例
└──────────────────┘
```

**Proposal 筛选策略** (通过 `proposal_selection` 配置):

| 策略 | 说明 | 来源 |
|------|------|------|
| `nms` | IoU > 0.5 的非极大值抑制 | PointGroup |
| `greedy` | 高分优先独占点, IoU > 0.5 保留 | SoftGroup |
| `iach` | offset 移位 + DBSCAN 聚类 | IACH 风格 |

### 2.2 PointGroup 模式 (`segmentation.method = pointgroup`)

仅使用 PointGroup proposal（NMS），不做二次聚类和后处理。

### 2.3 Clustering 模式 (`segmentation.method = clustering`)

PointGroup 语义筛选甘蓝点 → 传统聚类算法（由 `instance.method` 配置）。

### 2.4 支持的聚类算法

| 算法 | 配置键 | 核心参数 |
|------|--------|----------|
| **Watershed 3D** (默认) | `watershed_3d` | `voxel_resolution=0.01m`, `min_seed_distance=0.3m`, `smoothing_sigma=1` |
| **MeanShift** | `meanshift` | `bandwidth=0.25m` (甘蓝半径), `bin_seeding=true` |
| **DBSCAN** | `dbscan` | `eps=0.05m`, `min_samples=10` |
| **HDBSCAN** | `hdbscan` | `min_cluster_size=500`, `min_samples=10` |
| **Graph-based** | `graph_based` | `n_neighbors=30`, `distance_threshold=0.05m`, `linkage=average` |
| **Euclidean** | `euclidean` | `tolerance=0.05m`, `min_cluster_size=50` |

---

## Stage 3: 后处理

### 3.1 骨架拆分 (`pca_split`)

对可能包含多株的簇进行 PCA 分析，沿主方向拆分粘连植株。

### 3.2 碎片合并 (`fragment_voting`)

- **丢弃阈值** (`discard_threshold`): < 300 点的小碎片直接丢弃
- **合并阈值** (`merge_threshold`): < 1500 点的碎片尝试合并到邻近大簇

### 3.3 噪声点处理

未分配点（label = -1）通过 cKDTree 最近邻搜索归入最近的有效实例。

### 3.4 最终过滤

丢弃点数 < `min_cluster_points`（默认 1000）的碎片簇，重新编号输出。

---

## Stage 4: 表型参数提取

**入口**: `cabbage_pheno/traits/ → TraitCalculator.calculate_traits()`

对每个最终实例提取以下表型参数：

| 参数 | 单位 | 说明 |
|------|------|------|
| `height_cm` | cm | 植株高度（Z 轴范围） |
| `crown_diameter_cm` | cm | 冠幅直径（XY 平面投影拟合圆） |
| `volume_voxel_cm3` | cm³ | 体素法体积估计 |
| `compactness` | — | 紧凑度（体积/外包球体积比） |
| `leaf_area_estimate` | cm² | 叶面积估计（凸包表面积） |

---

## Stage 5: 输出与可视化

### 输出文件

| 文件 | 内容 |
|------|------|
| `cabbage_points.ply` | 语义分割后的甘蓝点云 |
| `stage1_hybrid_{sel}.ply` | Stage 1 实例着色 |
| `stage2_clustering.ply` | Stage 2 补充聚类后着色 |
| `stage3_skeleton.ply` | Stage 3 骨架拆分后着色 |
| `stage4_final.ply` | 最终实例着色结果 |
| `plants.json` | 各植株表型参数 (JSON) |
| `plants.csv` | 各植株表型参数 (CSV) |

### 运行命令

```bash
# 单文件推理
python main.py --input data/sample.ply --config configs/default.yaml

# 批量评估
python evaluate.py --input evalaute_test --config configs/default.yaml --split test

# 多配置对比
python tools/compare_experiments.py \
  --input e_data/train \
  --configs configs/default.yaml configs/ablation_no_pca.yaml \
  --labels default ablation \
  --output output/compare.csv
```

---

## 关键配置路径

| 用途 | 配置文件 |
|------|----------|
| 主流程默认配置 | `configs/default.yaml` |
| PointGroup 模型配置 | `PointGroup_Ours/config/pointgroup_cabbage.yaml` |
| PointNet2 模型配置 | `PointNet2_Ours/config/pointnet2_cabbage.yaml` |
| SoftGroup 模型配置 | `SoftGroup_Ours/config/softgroup_cabbage.yaml` |
| 评估 Val/Test 划分 | `configs/split.yaml` |
| 消融实验 (无PCA) | `configs/ablation_no_pca.yaml` |
| 聚类对比实验 | `configs/compare_clustering.yaml` |
