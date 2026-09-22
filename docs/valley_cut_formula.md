# 密度波谷切割点判定

对于待解耦的粘连簇，将其内所有点沿第一主成分方向 $\mathbf{v}_1$ 投影至一维流形空间，得到各点的投影标量 $s_i = (\mathbf{p}_i - \bar{\mathbf{p}}) \cdot \mathbf{v}_1$，其取值范围构成有效投影区间 $[s_{\min}, s_{\max}]$，其中 $\bar{\mathbf{p}}$ 为簇质心。

在上述一维流形空间内，将有效投影区间均匀离散划分为 $m$ 个连续区间（bin）：

$$S = \left\{ [s_0, s_1), [s_1, s_2), \dots, [s_{m-1}, s_m] \right\} \tag{4}$$

区间宽度 $\Delta s$ 定义为：

$$\Delta s = \frac{s_{\max} - s_{\min}}{m} \tag{5}$$

其中分箱数 $m$ 默认取 15，并约束区间宽度下限为 0.02 m（当 $\Delta s$ 小于该下限时自动减少分箱数），以保证统计有效性，避免在过短的簇上产生噪声主导的过细分箱。随后统计落入每个区间内的点数，得到一维离散密度序列：

$$\boldsymbol{\rho} = [\rho_1, \rho_2, \dots, \rho_m],\quad
\rho_k = \left|\left\{ \mathbf{p}_i \mid s_i \in [s_{k-1}, s_k) \right\}\right| \tag{6}$$

由于区间宽度 $\Delta s$ 恒定，点数 $\rho_k$ 与线密度 $\rho_k / \Delta s$ 仅相差一个常数因子，不影响峰谷位置与相对深度比，故可直接以点数作为密度度量。本质上，密度序列 $\boldsymbol{\rho}$ 是植株沿延展轴质量分布轮廓的数字化采样，能够灵敏捕捉相邻植株中心（密度峰）与粘连交界（密度谷）之间的结构差异。为抑制噪声，对密度序列做高斯平滑（$\sigma = 0.5$ bin），得到平滑密度曲线 $\hat{\boldsymbol{\rho}}$。

对平滑后的密度曲线执行局部峰值检测，检测须满足最小峰间距 0.30 m 与最小峰高（大于全局最大密度的 10%）两个约束。当检测到的峰数量少于 2 时，认为该簇仅含单株，不执行切割。设两端的显著峰为 $s_{\text{peak}_1}$ 与 $s_{\text{peak}_2}$（取首个与末个峰），将二者之间平滑密度最小的位置作为候选分割点：

$$\begin{aligned}
s_{\text{cut}} &= \arg\min_{s_k \in \left(s_{\text{peak}_1},\ s_{\text{peak}_2}\right)} \hat{\rho}(s_k) \\
\text{s.t.}\quad &\frac{\hat{\rho}(s_{\text{cut}})}{\min\big(\hat{\rho}(s_{\text{peak}_1}),\ \hat{\rho}(s_{\text{peak}_2})\big)} < \theta
\end{aligned} \tag{7}$$

式中：$\theta$ 为相对深度阈值，采用自适应策略，由簇沿主轴的物理长度 $L$ 决定：

$$\theta = \begin{cases}
\theta_0, & L \leq 0.60\ \text{m} \\[2pt]
\max(\theta_0,\ 0.98), & L > 0.60\ \text{m}
\end{cases} \tag{8}$$

其中 $\theta_0 = 0.90$ 为经验相对深度阈值。其物理含义为：当主轴长度超过 0.60 m 时，该簇几乎必然包含多株粘连植株，故放宽切割条件（允许更浅的波谷触发切割），以避免漏切。此外，切割须满足有效性约束，即沿局部切割平面二分后两个子簇的点数均须大于 10：

$$\min\left( \left|\mathcal{C}_1\right|,\ \left|\mathcal{C}_2\right| \right) > 10 \tag{9}$$

其中 $\mathcal{C}_1$、$\mathcal{C}_2$ 为切割平面两侧的子簇点集，切割平面由波谷两侧相邻切面的质心连线确定其局部切线方向，切割中心为波谷切面的质心。

满足上述条件后，在 $s_{\text{cut}}$ 处执行切割，并对两个子簇分别递归重复上述过程，直至满足任一终止条件：子簇点数小于 50、递归深度达到上限 3、子簇主轴跨距小于 0.20 m、子簇投影长度小于 0.30 m（无法满足最小峰间距）、或检测到的峰数量少于 2。
