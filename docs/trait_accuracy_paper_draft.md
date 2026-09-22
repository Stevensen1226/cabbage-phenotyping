# 表型参数精度验证 — 论文方法与结果草稿

> 本文档为「甘蓝点云表型参数自动提取」的论文写作草稿，覆盖方法（Methods）与结果（Results）两部分，中英双语。
> 关键约定：本研究的"参考值"为**人工标注点云计算值**，非田间尺量值（详见 1.3 节）。

---

## 1. 方法（Methods）

### 1.1 表型参数定义

对每株分割得到的甘蓝点云，采用以下几何定义计算 4 项核心表型参数：

| 表型参数 | 符号 | 定义 | 公式 / 方法 |
|----------|------|------|------------|
| 株高 | $H$ | 点云垂直方向 1%–99% 分位差 | $H = P_{99}(z) - P_1(z)$ |
| 冠幅 | $D$ | 水平投影等面积圆直径 | $D = 2\sqrt{A/\pi}$，$A$ 为 2D 凸包面积 |
| 植株体积 | $V$ | alpha-shape 体积 | 外接球半径 $\le 3\,\mathrm{cm}$ 的 Delaunay 四面体体积之和 |
| 紧实度 | $C$ | 体积与包围盒体积之比 | $C = V / (l_x \cdot l_y \cdot l_z)$（无量纲） |

### 1.2 参考值（Reference）的生成方式

为评估自动提取的精度，我们为每个正确匹配的实例对（IoU ≥ 0.5）计算两组值：

- **计算值（Calculated）**：实例分割算法输出的点云 → 表型计算。
- **参考值（Reference）**：人工逐点标注的实例真值（GT）点云 → **相同的**表型计算。

两组值采用**完全一致**的计算函数、地面参考平面与参数定义（等面积圆冠幅、alpha-shape 体积）。因此二者差异**仅反映实例分割质量对表型测量的影响**，而非"点云算法 vs 人工尺量"的绝对误差。

### 1.3 评估指标

采用线性拟合与误差指标量化精度：

1. **拟合方程** $y = a x + b$：理想情况下 $a=1$、$b=0$（即 $y=x$）。
   - $a$ 越接近 1，比例偏差越小；$b$ 越接近 0，整体偏移越小。
2. **决定系数 $R^2$**：衡量拟合优度，越接近 1 相关性越强。
3. **均方根误差 RMSE**：$\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i}(x_i - y_i)^2}$，数值越小误差越小，单位与对应表型一致。

---

## 2. 结果（Results）

### 2.1 实例分割精度

test 集（14 个田间场景，224 个真值植株）实例分割指标：

| Instance Precision | Instance Recall | Instance F1-score | Instance mIoU | Semantic mIoU | MAE(Count) |
|-------------------:|----------------:|------------------:|--------------:|--------------:|-----------:|
| 0.859 | 0.856 | 0.855 | 0.777 | 0.760 | 1.429 |

### 2.2 表型参数精度（PointGroup + ADPV，186 对匹配）

| 表型参数 | $a$ | $b$ | $R^2$ | RMSE |
|----------|----:|----:|------:|-----:|
| 株高 | 0.731 | 7.039 | 0.527 | 3.46 cm |
| 冠幅 | 0.360 | 29.789 | 0.262 | 7.86 cm |
| 植株体积 | 0.843 | 488.25 | **0.776** | 599.7 cm³ |
| 紧实度 | 0.972 | 0.008 | 0.613 | 0.012 |

### 2.3 ADPV 模块的消融对比（test 集）

#### PointGroup 骨干

| 指标 | 无 ADPV | +ADPV | 变化 |
|------|--------:|------:|------|
| 匹配株数 | 86 | 186 | +100 |
| 体积 $R^2$ | 0.692 | 0.776 | +0.084 |
| 体积 RMSE (cm³) | 784.2 | 599.7 | −23% |
| 冠幅 RMSE (cm) | 10.94 | 7.86 | −28% |

#### RandLA-Net 骨干

| 指标 | 无 ADPV | +ADPV | 变化 |
|------|--------:|------:|------|
| 匹配株数 | 102 | 184 | +82 |
| 体积 $R^2$ | 0.659 | 0.766 | +0.107 |
| 体积 RMSE (cm³) | 857.9 | 591.8 | −31% |
| 冠幅 RMSE (cm) | 12.11 | 8.36 | −31% |

### 2.4 冠幅系统性偏差分析

冠幅的参考值范围仅 36–63 cm（种内变异小），且预测值存在系统性正偏差：

- 预测点云的水平投影面积比 GT 点云大 **19.6%**；
- 冠幅（等面积圆）平均偏大 **+4.38 cm**（78.5% 样本偏大）；
- 根因：语义分割在植株边缘将展开叶片误判为甘蓝，导致边界外扩；
- 经系统偏差校正（−4.38 cm）后，冠幅 RMSE 由 7.86 cm 降至 **4.50 cm**。

---

## 3. 结论（可写进论文的段落）

植株体积的自动测量精度最高（$R^2=0.776$，RMSE $=599.7\,\mathrm{cm^3}$），株高次之（$R^2=0.527$，RMSE $=3.46\,\mathrm{cm}$）。冠幅的 $R^2$ 较低（0.262），主要源于甘蓝冠幅种内变异较小以及语义分割在植株边缘的系统性外扩（投影面积偏大 19.6%）。加入 ADPV 模块后，可正确匹配的植株数提升约一倍，体积 RMSE 降低 23%–31%，冠幅 RMSE 降低 28%–31%，验证了 ADPV 对表型测量精度的提升作用。

---

## 4. 英文版本（English Draft）

### 4.1 Methods

For each correctly matched instance pair (IoU ≥ 0.5), we compute two sets of trait values:

- **Calculated value**: derived from the point cloud produced by instance segmentation.
- **Reference value**: derived from the manually annotated ground-truth (GT) point cloud using the **identical** geometric pipeline.

Both values share the same computational functions, ground reference plane, and trait definitions (equivalent-circle canopy diameter, alpha-shape volume). Thus, their discrepancy reflects **only the influence of instance segmentation quality on trait measurement**.

### 4.2 Results

| Trait | $a$ | $b$ | $R^2$ | RMSE |
|-------|----:|----:|------:|-----:|
| Plant Height | 0.731 | 7.039 | 0.527 | 3.46 cm |
| Canopy Diameter | 0.360 | 29.789 | 0.262 | 7.86 cm |
| Plant Volume | 0.843 | 488.25 | 0.776 | 599.7 cm³ |
| Compactness | 0.972 | 0.008 | 0.613 | 0.012 |

### 4.3 Conclusion

Plant volume achieves the highest estimation accuracy ($R^2=0.776$, RMSE $=599.7\,\mathrm{cm^3}$), followed by plant height ($R^2=0.527$). The relatively low $R^2$ for canopy diameter is attributed to the small intra-species variation in cabbage canopy size and a systematic over-segmentation at plant edges (projected area overestimated by 19.6%). With the ADPV module, the number of correctly matched plants approximately doubles, and the RMSE of volume and canopy diameter decrease by 23%–31% and 28%–31%, respectively.

---

## 5. 表述规范（避免审稿人质疑）

| ❌ 避免 | ✅ 推荐 |
|--------|--------|
| 与人工实测值高度一致 | 与标注点云参考值高度一致 |
| Measured value（无尺量数据时） | Reference (GT-derived) |
| 点云测量可替代人工尺量 | 点云测量可稳定复现标注点云的表型 |
| 冠幅精度 7.86 cm（不提偏差） | 冠幅 RMSE 7.86 cm，其中约 4.4 cm 为系统性偏差 |

> 若后续补充田间人工尺量数据（株高/冠幅实测记录），可将 Y 轴升级为真实尺量值，此时方可使用 "Measured value" 并声称绝对测量精度。
