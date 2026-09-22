# 计算效率与资源开销分析（ADPV）

## 实验设置

本节评估本文提出的 **ADPV**（几何感知实例解耦后处理，GIDM）模块的计算效率与资源开销。

- **硬件**：单张 NVIDIA GeForce RTX 5070（12 GB 显存）。
- **数据集**：测试集，共 14 个甘蓝田地点云场景（去除地面后单场景点数约 67K～398K）。
- **上游骨干**：RandLA-Net（0.16M 参数）、PointGroup（7.71M 参数）、PointNeXt-L（7.12M 参数）。
- **基础聚类**：Watershed 3D、MeanShift、HDBSCAN 三种方法。
- **统计口径**：所有耗时均为逐文件测量后的算术平均；GPU 显存为语义推理阶段的峰值分配；CPU 内存为 ADPV 后处理阶段的 Python 峰值内存（`tracemalloc` 测量）。

> 说明：ADPV 完全由纯 CPU 几何计算构成（PCA 主轴投影、密度峰谷检测、骨架切割与碎片回并），不依赖任何可学习参数，因此不引入 GPU 显存开销。

---

## 图 a — 端到端耗时构成

**Fig. X** End-to-end inference latency breakdown per scene for three semantic backbones. Each bar stacks the semantic inference (blue), baseline clustering (orange), and the proposed ADPV post-processing (red). The ADPV module contributes a nearly constant overhead of ~91–111 ms regardless of backbone, corresponding to only 0.5%–8.6% of the total latency.

### 正文解释

图 a 展示了三种语义骨干网络在单场景上的端到端耗时构成。RandLA-Net、PointGroup 与 PointNeXt-L 的完整推理耗时分别约为 1453 ms、1292 ms 与 16542 ms。其中，本文提出的 ADPV 后处理模块耗时分别为 92 ms、111 ms 与 91 ms，几乎不随骨干网络变化，占比仅为 6.3%、8.6% 与 0.5%。这一结果表明 ADPV 是与上游骨干网络解耦的轻量后处理步骤，其时间开销相对完整管线可忽略不计。

| 骨干网络 | 语义推理 | 基础聚类 | ADPV 后处理 | 总计 |
|:---|:---:|:---:|:---:|:---:|
| RandLA-Net | 1076 ms | 285 ms | **92 ms** | 1453 ms |
| PointGroup | 854 ms | 327 ms | **111 ms** | 1292 ms |
| PointNeXt-L | 16185 ms | 266 ms | **91 ms** | 16542 ms |

---

## 图 b — 内存占用

**Fig. X** Memory footprint comparison across the three backbones. Left axis (colored bars) reports the peak GPU memory of the semantic backbone; right axis (gray bars) reports the peak CPU memory consumed by the ADPV post-processing. ADPV introduces **zero additional GPU memory** and only 5.2–5.7 MB of CPU memory.

### 正文解释

图 b 从内存维度进一步验证 ADPV 的轻量性。三个骨干网络的 GPU 峰值显存分别为 5726 MB（RandLA-Net）、773 MB（PointGroup）与 598 MB（PointNeXt-L）。由于 ADPV 完全在 CPU 上以纯几何计算实现，它不引入任何额外的 GPU 显存开销，仅消耗 5.2–5.7 MB 的 CPU 内存。这说明 ADPV 可以在不增加 GPU 资源负担的前提下，作为插件式后处理模块嵌入任意实例分割管线。

| 骨干网络 | GPU 峰值显存 | ADPV CPU 峰值内存 |
|:---|:---:|:---:|
| RandLA-Net | 5726 MB | 5.2 MB |
| PointGroup | 773 MB | 5.7 MB |
| PointNeXt-L | 598 MB | 5.2 MB |

---

## 图 c — 场景级可扩展性

**Fig. X** Scene-level scalability: full pipeline runtime versus the number of non-ground points per scene (logarithmic y-axis). Solid, dashed, and dash-dotted curves denote RandLA-Net, PointGroup, and PointNeXt-L, respectively. The runtime of RandLA-Net and PointGroup grows modestly with scene size, whereas PointNeXt-L exhibits a substantially higher cost due to its multi-round sampling-and-propagation inference.

### 正文解释

图 c 考察了各方法随场景规模增长的时间可扩展性。横轴为去除地面后的单场景点数（约 67K 至 398K 点），纵轴为完整管线耗时（对数坐标）。RandLA-Net 与 PointGroup 的耗时随场景规模平缓增长，二者曲线相近；而 PointNeXt-L 由于采用"多轮随机采样 + 最近邻传播"的推理策略，耗时整体高出一个数量级。这一对比表明，在追求实时或近实时的田间表型分析场景中，RandLA-Net 与 PointGroup 是更合适的上游骨干，而 ADPV 的后处理开销在所有规模下均保持稳定。

---

## 图 d — 后处理成本 vs 精度增益

**Fig. X** Accuracy gain versus post-processing cost of ADPV under three baseline clustering methods. The x-axis is the additional latency introduced by ADPV; the y-axis is the absolute improvement in instance F1 (percentage points). ADPV consistently yields substantial gains (up to +66.6 pp for HDBSCAN) with a latency increment of only 79–161 ms and zero GPU memory.

### 正文解释

图 d 量化了 ADPV 后处理的"精度–成本"权衡关系。在 Watershed 3D、MeanShift 与 HDBSCAN 三种基础聚类方法上，ADPV 引入的额外延迟仅为 103 ms、79 ms 与 161 ms，却分别带来 +29.5、+3.2 与 +66.6 个百分点的实例 F1 提升。特别地，在 HDBSCAN 这一基线极弱（F1 = 0.114）的场景下，ADPV 以约 161 ms 的代价将其 F1 提升至 0.780，相对提升超过 5.8 倍。

| 基础聚类方法 | ADPV 延迟增量 | 基线 F1 | +ADPV F1 | F1 增益 |
|:---|:---:|:---:|:---:|:---:|
| Watershed 3D | 103 ms | 0.530 | 0.825 | +29.5 pp |
| MeanShift | 79 ms | 0.856 | 0.888 | +3.2 pp |
| HDBSCAN | 161 ms | 0.114 | 0.780 | +66.6 pp |

结合图 a、b 的结果可以得出结论：**ADPV 以极低的计算与内存开销，实现了跨聚类方法、跨骨干网络的稳定精度增益，验证了其作为通用轻量后处理模块的有效性。**
