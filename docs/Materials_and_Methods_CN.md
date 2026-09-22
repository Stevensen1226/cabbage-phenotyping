# 材料与方法

## 2.1. 研究区域与数据获取

田间试验在位于江苏省南通市如东县（121.183°E, 32.304°N）的半亩方塘生态农场进行。试验场地面积约 0.26 公顷（3.95 亩），属于典型的沿海平原农业景观。目标作物为露天种植甘蓝（*Brassica oleracea* var. *capitata* L.）。数据采集在移栽后约 90 天进行，此时甘蓝冠层达到最大横向扩展，株间叶片严重交织，形成连续封闭冠层。

三维点云数据采用大疆 Matrice 350 RTK 无人机搭载禅思 L1 LiDAR 传感器采集。数据采集在晴朗、无风、光照充足的条件下进行。飞行任务规划离地高度（AGL）10 m，飞行速度 3 m/s，旁向重叠率 70%。L1 传感器工作在重复扫描模式，利用其多回波能力部分穿透上层冠层。

原始 LiDAR 数据使用大疆智图（DJI Terra）软件进行处理和地理配准，生成整块田地平均点密度超过 20,000 点/m$^2$ 的高精度三维点云。为便于计算处理和地面真值标注，将该连续田间尺度点云划分为多个空间连续的子区域，每个子区域包含若干株按行排列的甘蓝植株。点云以 PLY 格式存储，逐点属性为三维笛卡尔坐标 $(x, y, z)$（m）及颜色 $(R, G, B)$（8 位整数）。

地面真值标注为每个点赋予语义标签 $l^{\text{sem}} \in \{0, 1\}$（0 = 背景；1 = 甘蓝）及实例标签 $l^{\text{inst}} \in \{0, 1, \dots, K\}$（正整数唯一标识单株；0 = 非植株）。标注数据集专用于算法评估，未参与模型训练。

## 2.2. 点云预处理

### 2.2.1. 统计离群点移除

对每个点云应用 SOR 滤波器（Rusu and Cousins, 2011）。对于点云 $\mathcal{P} = \{\mathbf{p}_1, \dots, \mathbf{p}_N\}$（$\mathbf{p}_i \in \mathbb{R}^3$），计算每个点到其 $k$ 个最近邻的平均欧氏距离：

$$\bar{d}_i = \frac{1}{k} \sum_{j \in \mathcal{N}_k(i)} \|\mathbf{p}_i - \mathbf{p}_j\|_2 \tag{1}$$

假设 $\{\bar{d}_i\}$ 近似服从 $\mathcal{N}(\mu_d, \sigma_d^2)$，移除满足以下条件的离群点：

$$\bar{d}_i > \mu_d + \alpha \cdot \sigma_d \tag{2}$$

参数设置为 $k = 10$，$\alpha = 3.5$。点数少于 $2 \times 10^5$ 的点云豁免体素下采样。

### 2.2.2. 地面去除

采用 RANSAC 平面拟合算法（Fischler and Bolles, 1981）。取 $z$ 坐标最低 $15\%$ 的点作为平面拟合候选：

$$\mathcal{P}_{\text{fit}} = \{\mathbf{p}_i \in \mathcal{P} : z_i \leq Q_z(15\%)\} \tag{3}$$

估计平面模型 $ax + by + cz + d = 0$，距离阈值 $\delta = 0.015$ m，迭代 200 次，约束法向量向上（$c > 0$）。全点云按正交距离分类：

$$d_i = \frac{|a x_i + b y_i + c z_i + d|}{\sqrt{a^2 + b^2 + c^2}} \tag{4}$$

$d_i < 0.02$ m 的点移除为地面，得到非地面点云 $\mathcal{P}_{\text{ng}}$。

## 2.3. 语义分割

采用基于深度学习的语义分割模型，将 $\mathcal{P}_{\text{ng}}$ 中各点分类为甘蓝（类别 1）或背景（类别 0），输出二值甘蓝点集 $\mathcal{P}_{\text{cab}} = \{\mathbf{p}_i \in \mathcal{P}_{\text{ng}} : \hat{y}_i = 1\}$。

## 2.4. 基线实例聚类

$\mathcal{P}_{\text{cab}}$ 使用三种聚类算法划分为单株实例：MeanShift（MS; Comaniciu and Meer, 2002）、HDBSCAN（HDB; Campello et al., 2013）和三维分水岭（WS3D）。所有方法共享预处理和后处理：半径离群过滤（$\varepsilon = 0.03$ m，最少 10 个邻居）及密度阈值过滤（线密度 $< 500$ pts/m 或绝对点数 $< 100$）。

MeanShift 配置为带宽 $h = 0.25$ m（基于成熟甘蓝典型冠层半径 ~0.20–0.30 m），分箱种子（单元大小 $h$，最小频率 10）。HDBSCAN 使用互达距离 $d_{\text{mreach},k}(\mathbf{p}, \mathbf{q}) = \max\{\text{core}_k(\mathbf{p}), \text{core}_k(\mathbf{q}), \|\mathbf{p} - \mathbf{q}\|_2\}$，$k = 10$，EOM 准则选择聚类，$\text{min\_cluster\_size} = 500$，$\text{min\_samples} = 10$。三维分水岭通过六阶段执行：体素化（$v = 0.01$ m，自适应分辨率上限 $5 \times 10^7$ 体素）、单次二值膨胀（$3 \times 3 \times 3$ 结构元素）、三维欧氏距离变换、高斯平滑（$\sigma = 1.0$ 体素）、局部峰值检测（最小峰间距 $0.30$ m，绝对阈值 $0.006$ m，相对阈值 $0.05$）和标记控制的分水岭分割。

## 2.5. PCA 引导的骨架拆分

### 2.5.1. 基于 PCA 的异常检测

对基线聚类产生的每个聚类 $C_k$ 计算 PCA。协方差矩阵 $\boldsymbol{\Sigma}_k$ 的特征值为 $\lambda_1 \geq \lambda_2 \geq \lambda_3 \geq 0$，对应特征向量为 $\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3$：

$$\boldsymbol{\Sigma}_k = \frac{1}{|C_k|} \sum_{i=1}^{|C_k|} (\mathbf{p}_i - \bar{\mathbf{p}}_k)(\mathbf{p}_i - \bar{\mathbf{p}}_k)^\top \tag{5}$$

提取两个几何描述子：

$$L_1 = \max_i (\mathbf{p}_i \cdot \mathbf{v}_1) - \min_i (\mathbf{p}_i \cdot \mathbf{v}_1), \qquad r = \frac{L_1}{\max_i (\mathbf{p}_i \cdot \mathbf{v}_2) - \min_i (\mathbf{p}_i \cdot \mathbf{v}_2)} \tag{6}$$

其中 $L_1$ 为沿第一主轴的投影跨度，$r$ 为纵横比。若聚类满足以下条件则标记为拆分候选：

$$L_1 > 0.60\ \text{m} \quad \text{或} \quad (r > 1.60 \ \text{且}\ L_1 > 0.25\ \text{m}) \quad \text{或} \quad (\max\text{-axis}(C_k) > 0.60\ \text{m}) \tag{7}$$

### 2.5.2. 骨架密度分析

对每个候选聚类，将点投影到第一主轴：

$$s_i = (\mathbf{p}_i - \bar{\mathbf{p}}_k) \cdot \mathbf{v}_1 \tag{8}$$

投影范围 $[s_{\text{min}}, s_{\text{max}}]$（长度 $\ell = s_{\text{max}} - s_{\text{min}}$）划分为 $N = 15$ 个等宽分箱（最小箱宽 $0.02$ m）。各箱点数构成一维密度曲线 $\boldsymbol{\rho} \in \mathbb{N}^{N}$，经高斯滤波平滑（$\sigma = 0.5$ 个分箱）：

$$\tilde{\boldsymbol{\rho}} = \boldsymbol{\rho} * G_{\sigma=0.5} \tag{9}$$

### 2.5.3. 波谷检测与拆分

在 $\tilde{\boldsymbol{\rho}}$ 上使用 `find_peaks` 检测峰值（最小峰间距 $0.30$ m；最小相对峰高 $0.1 \cdot \max(\tilde{\boldsymbol{\rho}})$）。若检测到两个及以上峰值 $\{p_1, \dots, p_m\}$，定位首峰与末峰间最深波谷：

$$v^* = \arg\min_{b \in [p_1, p_m]} \tilde{\rho}_b, \quad \eta = \frac{\tilde{\rho}_{v^*}}{\min(\tilde{\rho}_{p_1}, \tilde{\rho}_{p_m})} \tag{10}$$

当 $\eta < \tau_{\text{valley}}$ 时触发切割，自适应阈值：

$$\tau_{\text{valley}} = \begin{cases} 0.90 & \ell \leq 0.60\ \text{m} \\ 0.98 & \ell > 0.60\ \text{m} \end{cases} \tag{11}$$

点按到切割平面的有符号距离划分，切割平面穿过波谷分箱质心，法向量沿局部骨架切线方向（由相邻非空分箱的质心差向量归一化得到；若任一侧缺失则以 $\mathbf{v}_1$ 回退）：

$$\text{assign}(\mathbf{p}_i) = \begin{cases} \text{左侧} & (\mathbf{p}_i - \mathbf{c}_{v^*}) \cdot \mathbf{n}_{\text{cut}} < 0 \\ \text{右侧} & \text{否则} \end{cases} \tag{12}$$

递归应用（最大深度 3，最小子聚类 50 点），外层迭代最多 5 轮。

### 2.5.4. 回并

拆分后，满足以下三条件的相邻子聚类 $(C_a, C_b)$ 予以合并：(1) 较小碎片直径 $< 0.15$ m；(2) 质心距离 $< 0.20$ m；(3) 第一主轴余弦相似度 $> \cos(30^\circ) \approx 0.866$：

$$\frac{|\mathbf{v}_{1,a} \cdot \mathbf{v}_{1,b}|}{\|\mathbf{v}_{1,a}\| \cdot \|\mathbf{v}_{1,b}\|} > \cos(30^\circ) \tag{13}$$

### 2.5.5. 碎片投票

残留聚类经三级处理：(1) $|C_k| < 500$ → 丢弃；(2) $500 \leq |C_k| < 1500$ → 并入最近有效聚类（$\geq 1500$ 点）；(3) 同化后丢弃 $|C_k| < 2000$，截断 $|C_k| > 10^5$。未分配点通过 k-d 树重分配至最近有效聚类。标签重新索引为 $\{1, 2, \dots, K_{\text{final}}\}$。

## 参考文献

Campello, R.J.G.B., Moulavi, D., Sander, J., 2013. Density-based clustering based on hierarchical density estimates. In: PAKDD, pp. 160–172.

Comaniciu, D., Meer, P., 2002. Mean shift: A robust approach toward feature space analysis. IEEE Trans. Pattern Anal. Mach. Intell. 24(5), 603–619.

Ester, M., Kriegel, H.P., Sander, J., Xu, X., 1996. A density-based algorithm for discovering clusters in large spatial databases with noise. In: KDD, pp. 226–231.

Fischler, M.A., Bolles, R.C., 1981. Random sample consensus: A paradigm for model fitting. Commun. ACM 24(6), 381–395.

Rusu, R.B., Cousins, S., 2011. 3D is here: Point Cloud Library (PCL). In: ICRA, pp. 1–4.
