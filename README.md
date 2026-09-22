# 甘蓝表型参数自动提取系统开发说明（基于点云：先语义分割，后单株聚类）

## 快速开始 (Quick Start)

### 1. 环境安装
确保已安装 Python 3.8+，然后安装依赖：
```bash
pip install -r requirements.txt
```

### 2. 准备数据
准备一个甘蓝的点云文件（支持 `.ply`, `.pcd` 格式）。
如果没有训练好的模型，系统将使用**高度阈值**进行简单的启发式分割。

### 3. 运行提取
```bash
# 使用默认配置运行
python main.py --input data/sample.ply

# 指定配置文件运行
python main.py --input data/sample.ply --config configs/default.yaml
```

### 4. 结果查看
运行完成后，结果将保存在 `output/` 目录下：
- `plants.csv`: 包含每株甘蓝的表型参数（株高、冠幅、体积等）。
- `instance_colored.ply`: 可视化结果，不同植株用不同颜色标记。
- `ground.ply`: 分割出的地面点云。

### 5. 高级配置
修改 `configs/default.yaml` 文件以调整参数：
- **体素大小 (`voxel_size`)**: 控制点云稀疏程度。
- **分割方法 (`segmentation.method`)**: 设为 `pointnet` 使用深度学习模型，或 `heuristic` 使用高度阈值。
- **聚类参数 (`instance.dbscan`)**: 调整 `eps` 以控制单株分离的灵敏度。

### 6. 统一评估与对比
单次评估复用统一预处理入口与共享指标脚本：
```bash
python evaluate.py --input data/raw --config configs/default.yaml
```

批量对比多个配置时，使用统一对比脚本：
```bash
python tools/compare_experiments.py \
  --input data/raw \
  --configs configs/default.yaml configs/ablation_dbscan.yaml \
  --labels default dbscan_ablation \
  --output output/compare_experiments.csv
```

脚本会同时导出：
- `output/compare_experiments.csv`：每个配置的总指标表
- `output/compare_experiments_per_file.csv`：逐文件结果明细
- `output/compare_experiments.json`：完整结构化结果

### 7. 接入外部模型对比 (BGPSeg 等)
两个方法共用同一预处理管线 (SOR + 地面去除) 和同一数据集 (`e_data/train/`)。

先通过桥接脚本跑 BGPSeg 推理并得到统一指标 JSON：
```bash
python tools/bgpseg_bridge.py \
  --data e_data/train \
  --config configs/default.yaml \
  --output output/bgpseg_result.json
```

然后和 Cabbage 配置一起对比：
```bash
python tools/compare_experiments.py \
  --input e_data/train \
  --configs configs/default.yaml \
  --labels Cabbage-Hybrid \
  --precomputed output/bgpseg_result.json \
  --precomputed-labels BGPSeg \
  --output output/compare_paper.csv
```

控制变量说明：两方法均使用 `configs/default.yaml` 中的预处理参数（SOR + RANSAC 地面去除），
评估数据均来自 `e_data/train/`，指标计算共用同一套函数。

---

## 1. 目标与范围
系统输入田间/室内采集的甘蓝三维点云数据，自动完成：
1) 点云预处理与标准化；  
2) **语义分割**：将点分类为“植株/地面/杂草/支架或背景”等语义；  
3) **单株聚类**：在“植株”语义点上做实例级分离，得到每一株的点云；  
4) 表型参数计算：株高、冠幅、投影面积、体积/体素体积、叶片/植株点数、紧实度等；  
5) 导出结果（CSV/JSON）与可视化（PLY/PCD/渲染截图）。

不做的事（可扩展）：叶片级实例分割、病斑识别、多时相生长曲线自动配准（可在二期加入）。

---

## 2. 总体架构
**分层结构（推荐）**
- `io`：数据读写（PCD/PLY/LAS/ROS bag/自定义）  
- `preprocess`：去噪、下采样、地面估计、坐标系对齐、法向量/高度特征  
- `segmentation`：语义分割推理（深度模型 or 传统分类器）  
- `instance`：单株聚类（DBSCAN/欧式聚类/图聚类/分水岭体素）  
- `traits`：表型参数计算模块（每株输出统一 schema）  
- `qa`：质量控制（置信度、异常检测、缺失/遮挡提示）  
- `service`：CLI/REST API/批处理任务  
- `viz`：渲染与中间结果导出（语义上色、实例上色、包围盒/凸包）

**数据流**
1. 读取点云 → 2. 预处理 → 3. 语义分割 → 4. 过滤“植株”点 → 5. 单株聚类 → 6. 每株表型计算 → 7. 结果导出/可视化/日志

---

## 3. 输入输出约定

### 3.1 输入
- 点云：`{x,y,z}`，可选 `rgb/intensity/timestamp/ring`  
- 传感器元数据（可选）：外参、单位（m/mm）、采样频率、采集高度  
- 场景配置（必须）：地面法向方向（默认 +Z）、行距株距先验（可选但强烈建议）

### 3.2 输出
- `scene_result.json`：全场景统计、处理耗时、参数、版本号、失败原因
- `plants.csv/jsonl`：每株一行/一条记录
- `artifacts/`：  
  - `semantic_colored.ply`（按语义上色）  
  - `instance_colored.ply`（按单株 ID 上色）  
  - 每株点云 `plant_{id}.ply`（可选）

**单株结果 schema（示例）**
```json
{
  "plant_id": 12,
  "point_count": 158234,
  "centroid": [1.23, -0.45, 0.31],
  "height_m": 0.42,
  "crown_diameter_m": 0.58,
  "projected_area_m2": 0.21,
  "volume_voxel_m3": 0.034,
  "compactness": 1.67,
  "bbox": {"min":[...],"max":[...]},
  "quality": {"confidence":0.91, "occlusion_flag":false, "notes":[]}
}
```

---

## 4. 点云预处理（Preprocess）
### 4.1 单位与坐标系
- 统一单位为 **米（m）**
- 若传感器坐标非“地面为 XY、竖直为 Z”，需做外参旋转/重力对齐
- 输出点云必须满足：地面附近 `z≈0`，植株 `z>0`

### 4.2 去噪与下采样（建议默认）
- 统计滤波（SOR）：去除离群点
- 半径滤波：剔除孤立点簇
- 体素下采样（VoxelGrid）：降低密度，提升推理与聚类速度

### 4.3 地面估计（强烈建议做）
- RANSAC 平面拟合 / CSF / Progressive Morphological Filter  
- 得到地面模型后：  
  - 用于高度特征计算（`height = z - z_ground(x,y)`）  
  - 辅助语义分割（地面类更稳定）  
  - 限制聚类（避免地面残留连通）

---

## 5. 语义分割模块（先做语义，再做实例）
### 5.1 类别设计（可按场景裁剪）
- `0: background/unknown`
- `1: ground`
- `2: cabbage`（植株主体）
- `3: weed`（可选）
- `4: structure`（滴灌管/支架/墙体等，可选）

### 5.2 模型选择建议
- 室外稀疏大场景：RandLA-Net、KPConv、SparseConv（Minkowski）  
- 室内高密近景：PointNet++ / DGCNN / KPConv  
- 若算力有限：先体素化 + 稀疏卷积推理

### 5.3 输入特征（推荐）
- `x,y,z`（归一化）
- `height_above_ground`
- `normal`（法向量）
- 可选：`rgb/intensity`
- 可选：局部密度/曲率

### 5.4 推理输出
- `semantic_label[i]`（每点类别）
- `semantic_score[i]`（每点置信度/softmax 最大值）
- 输出保留原始点索引，便于回写颜色与追踪

### 5.5 语义后处理（关键）
- 只保留 `cabbage` 类点：`P_cab = {p | label(p)=cabbage && score>t}`
- 对 `ground` 残留做二次剔除：若 `height_above_ground < h_min` 则剔除
- 形态学/连通域清理：去除很小的“植株”碎片（阈值按点数或体素数）

---

## 6. 单株聚类（Instance Clustering）
> 原则：**在“植株语义点”子集上聚类**，避免地面/背景把不同植株粘连。

### 6.1 聚类策略（推荐组合）
**A. 欧式聚类 / DBSCAN（默认首选）**
- 距离阈值 `eps` 与点云密度强相关（建议与体素大小绑定）
- DBSCAN 优点：自动忽略噪声点；对不规则形状友好  
- 输出：每点 `instance_id`

**B. 行列先验 + 分段聚类（行栽场景强烈推荐）**
1) 对 `P_cab` 投影到 XY，做栅格密度图/聚类获取行方向  
2) 沿行方向切分（带宽/窗口）→ 每段内再做 DBSCAN  
这样能显著减少相邻株叶片接触导致的“合株”。

**C. 体素分水岭（密集冠层/粘连严重时）**
- 体素化后对距离变换或密度做分水岭
- 再将体素标签映射回点

### 6.2 聚类后处理（必须）
- 合并/拆分规则：
  - **过小簇**：点数 < `min_points` → 视为噪声丢弃或并入最近簇  
  - **过大簇（疑似合株）**：  
    - 检测簇内 XY 分布多峰（KDE/GMM）  
    - 或沿行方向做二次切分  
- ID 稳定性（可选）：若多帧/多时相，使用最近邻匹配或匈牙利匹配保持 plant_id 连续

---

## 7. 表型参数计算（Traits）
对每个 `instance_id` 的点集 `S` 计算：

### 7.1 基础几何量
- 质心：`centroid = mean(S)`
- 轴对齐包围盒 AABB：`min/max(x,y,z)`
- 主方向（PCA）：主轴方向与长短轴长度（冠幅方向有用）

### 7.2 株高（height）
- `height = percentile(z_above_ground, 99%) - percentile(z_above_ground, 1%)`
- 或 `max(z_above_ground)`（对离群点敏感，不推荐）
- 建议输出同时包含：`height_p99`、`height_max` 方便 QA

### 7.3 冠幅（crown diameter）
- 将点投影到 XY：
  - 方案1：PCA 长轴/短轴直径（更稳健）
  - 方案2：凸包直径（最远点对，易受离群影响）
- 输出：`crown_major`, `crown_minor`, `crown_diameter = (major+minor)/2`

### 7.4 投影面积（projected area）
- XY 投影后计算：
  - 方案1：α-shape/凸包面积（快速）
  - 方案2：栅格占据面积（体素/像素分辨率 r；面积 = occupied_cells * r^2，稳健可控）

### 7.5 体积（volume）
- **体素体积（推荐）**：3D voxelize（分辨率 r），体积 = occupied_voxels * r^3
- 或凸包体积（对凹形不准）

### 7.6 紧实度/蓬松度（compactness）
给一个可解释、可复现的定义（必须在文档里写死）：
- `compactness = volume_voxel / (bbox_volume + eps)`
或
- `compactness = point_count / projected_area`（反映密度）

### 7.7 质量指标（QA）
- `confidence`：语义置信度均值/中位数
- `occlusion_flag`：若株体缺顶（高度分布异常）或点数过少
- `merge_suspect`：簇内多峰、冠幅异常大、bbox 过长

---

## 8. 模块接口设计（示例）
### 8.1 Python 包结构（建议）
```
cabbage_pheno/
  io/
  preprocess/
  segmentation/
  instance/
  traits/
  viz/
  service/
  configs/
  tests/
```

### 8.2 核心接口（伪代码）
```python
class PipelineConfig:
    voxel_size: float
    seg_model_path: str
    seg_score_thresh: float
    dbscan_eps: float
    dbscan_min_samples: int
    min_cluster_points: int

def run_pipeline(pointcloud, cfg) -> SceneResult:
    pc = preprocess(pointcloud, cfg)
    sem = semantic_segment(pc, cfg)           # per-point label + score
    cab_points = filter_cabbage(pc, sem, cfg) # keep cabbage only
    inst = cluster_instances(cab_points, cfg) # per-point instance_id
    plants = extract_traits(cab_points, inst, cfg)
    return assemble_scene_result(pc, sem, inst, plants, cfg)
```

---

## 9. 配置、日志与可复现性
### 9.1 配置管理
- 用 `YAML/JSON` 管理：体素大小、模型路径、阈值、聚类参数、输出开关
- 每次运行将配置与 git commit hash 写入 `scene_result.json`

### 9.2 日志与监控
- 记录每阶段耗时：`preprocess/seg/cluster/traits/io`
- 记录关键统计：点数变化、植株数、噪声点比例、失败原因
- 异常时输出中间产物（语义上色/实例上色）便于定位

---

## 10. 性能与工程化建议
- 大场景：先体素下采样，再语义分割；导出时可映射回原密度（可选）
- GPU 推理：batch + 分块（tile）处理，避免显存爆
- 聚类加速：KDTree/voxel adjacency
- 并行：按地块/行/分块并行处理（multiprocessing）

---

## 11. 测试与验收标准
### 11.1 单元测试
- 地面估计正确性（平面 RMSE）
- 语义过滤阈值边界
- 聚类对已知合成数据（两株间距变化）稳定性
- 表型计算对简单几何体（圆柱/椭球）近似正确

### 11.2 指标验收（建议）
- 语义分割：mIoU / per-class F1（至少关注 cabbage vs ground）
- 单株聚类：实例数误差、合株率、裂株率
- 表型误差：与人工测量对比（MAE/RMSE），并给出在不同遮挡程度下的分层统计

---

## 12. 常见问题与处理策略
- **叶片接触导致合株**：引入行方向先验切分；或在过大簇上做二次多峰分裂  
- **地面残留连通**：加强 ground 类训练；语义后加 `height_above_ground` 门控  
- **不同采集密度导致 eps 不通用**：让 `eps = k * voxel_size`，并记录点密度自动估计  
- **风动/噪声点**：提高 SOR 强度 + DBSCAN min_samples + 小簇剔除
