# Materials and Methods

## 2.1. Study Area and Data Acquisition

Field experiments were conducted at the Banmu Fangtang ecological farm located in Rudong County, Nantong, Jiangsu Province, China (121.183°E, 32.304°N). The experimental site covers approximately 0.26 ha (3.95 mu) and represents a typical coastal plain agricultural landscape. The target crop was open-field cabbage (*Brassica oleracea* var. *capitata* L.). Data acquisition was carried out approximately 90 days after transplanting, when cabbage canopies reach their maximum lateral expansion, resulting in severe inter-plant leaf interweaving and continuous closed-canopy conditions.

Three-dimensional point cloud data were acquired using a DJI Matrice 350 RTK unmanned aerial vehicle equipped with a Zenmuse L1 LiDAR sensor. Data collection was performed under clear, windless conditions with sufficient illumination to ensure flight stability and data quality. Low-altitude flight missions were planned at an above-ground level (AGL) of 10 m. The flight speed was set to 3 m/s, with a lateral overlap of 70%. The L1 sensor operated in repeated scanning mode, leveraging its multi-return capability to partially penetrate upper canopy layers.

The raw LiDAR data were processed and georeferenced using DJI Terra software, resulting in a high-precision 3D point cloud of the entire field with an average point density exceeding 20,000 points/m$^2$. For computational tractability and to facilitate ground truth annotation, this continuous field-scale point cloud was partitioned into multiple spatially contiguous sub-regions, each containing several individual cabbage plants arranged in rows. Point clouds were stored in PLY format with per-point attributes: three-dimensional Cartesian coordinates $(x, y, z)$ (m) and color $(R, G, B)$ (8-bit integer).

Ground truth annotations assigned each point a semantic label $l^{\text{sem}} \in \{0, 1\}$ (0 = background; 1 = cabbage) and an instance label $l^{\text{inst}} \in \{0, 1, \dots, K\}$ (positive integers uniquely identify individual plants; 0 = non-plant). The annotated dataset was used exclusively for algorithm evaluation and was not involved in model training.

## 2.2. Point Cloud Preprocessing

### 2.2.1. Statistical Outlier Removal

A Statistical Outlier Removal (SOR) filter (Rusu and Cousins, 2011) was applied to each point cloud. For a point cloud $\mathcal{P} = \{\mathbf{p}_1, \dots, \mathbf{p}_N\}$ ($\mathbf{p}_i \in \mathbb{R}^3$), the mean Euclidean distance from each point to its $k$ nearest neighbors was computed:

$$\bar{d}_i = \frac{1}{k} \sum_{j \in \mathcal{N}_k(i)} \|\mathbf{p}_i - \mathbf{p}_j\|_2 \tag{1}$$

Assuming $\{\bar{d}_i\}$ approximately follows $\mathcal{N}(\mu_d, \sigma_d^2)$, points satisfying $\bar{d}_i > \mu_d + \alpha \cdot \sigma_d$ were classified as outliers and removed:

$$\bar{d}_i > \mu_d + \alpha \cdot \sigma_d \tag{2}$$

Parameters were set to $k = 10$ and $\alpha = 3.5$. Point clouds with fewer than $2 \times 10^5$ points were exempted from voxel down-sampling.

### 2.2.2. Ground Removal

A RANSAC plane fitting algorithm (Fischler and Bolles, 1981) was employed. The lowest $15\%$ of points ranked by $z$-coordinate were used as plane-fitting candidates:

$$\mathcal{P}_{\text{fit}} = \{\mathbf{p}_i \in \mathcal{P} : z_i \leq Q_z(15\%)\} \tag{3}$$

A plane model $ax + by + cz + d = 0$ was estimated with distance threshold $\delta = 0.015$ m and 200 iterations, constraining the normal upward ($c > 0$). All points were then classified by orthogonal distance:

$$d_i = \frac{|a x_i + b y_i + c z_i + d|}{\sqrt{a^2 + b^2 + c^2}} \tag{4}$$

Points with $d_i < 0.02$ m were removed as ground, yielding the non-ground point cloud $\mathcal{P}_{\text{ng}}$.

## 2.3. Semantic Segmentation

A deep learning-based semantic segmentation model was trained to classify each point in $\mathcal{P}_{\text{ng}}$ as cabbage (class 1) or background (class 0). The output was a binary cabbage point set $\mathcal{P}_{\text{cab}} = \{\mathbf{p}_i \in \mathcal{P}_{\text{ng}} : \hat{y}_i = 1\}$.

## 2.4. Baseline Instance Clustering

$\mathcal{P}_{\text{cab}}$ was partitioned into individual plant instances using three clustering algorithms: MeanShift (MS; Comaniciu and Meer, 2002), HDBSCAN (HDB; Campello et al., 2013), and 3D Watershed (WS3D). All methods operated on 3D coordinates and shared common pre- and post-processing: radius outlier filtering ($\varepsilon = 0.03$ m, minimum 10 neighbors) and density thresholding (linear density $< 500$ pts/m or absolute count $< 100$).

MeanShift was configured with bandwidth $h = 0.25$ m, set based on the typical crown radius of mature cabbage (~0.20–0.30 m), and bin seeding (cell size $h$, minimum bin frequency 10). HDBSCAN used the mutual reachability distance $d_{\text{mreach},k}(\mathbf{p}, \mathbf{q}) = \max\{\text{core}_k(\mathbf{p}), \text{core}_k(\mathbf{q}), \|\mathbf{p} - \mathbf{q}\|_2\}$ with $k = 10$, EOM cluster selection, $\text{min\_cluster\_size} = 500$, and $\text{min\_samples} = 10$. The 3D Watershed algorithm proceeded through six stages: voxelization ($v = 0.01$ m, adaptive resolution cap $5 \times 10^7$ voxels), single-iteration binary dilation ($3 \times 3 \times 3$ structuring element), 3D Euclidean Distance Transform, Gaussian smoothing ($\sigma = 1.0$ voxels), local peak detection (minimum inter-peak distance $0.30$ m, absolute threshold $0.006$ m, relative threshold $0.05$), and marker-controlled watershed on the inverted distance field.

## 2.5. PCA-Guided Skeleton Splitting

### 2.5.1. Abnormality Detection via PCA

For each cluster $C_k$ produced by baseline clustering, PCA was computed on its point coordinates. The covariance matrix $\boldsymbol{\Sigma}_k$ yields eigenvalues $\lambda_1 \geq \lambda_2 \geq \lambda_3 \geq 0$ and eigenvectors $\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3$:

$$\boldsymbol{\Sigma}_k = \frac{1}{|C_k|} \sum_{i=1}^{|C_k|} (\mathbf{p}_i - \bar{\mathbf{p}}_k)(\mathbf{p}_i - \bar{\mathbf{p}}_k)^\top \tag{5}$$

Two geometric descriptors were derived:

$$L_1 = \max_i (\mathbf{p}_i \cdot \mathbf{v}_1) - \min_i (\mathbf{p}_i \cdot \mathbf{v}_1), \qquad r = \frac{L_1}{\max_i (\mathbf{p}_i \cdot \mathbf{v}_2) - \min_i (\mathbf{p}_i \cdot \mathbf{v}_2)} \tag{6}$$

where $L_1$ is the projected span along the first principal axis and $r$ is the aspect ratio. A cluster was flagged as a splitting candidate if:

$$L_1 > 0.60\ \text{m} \quad \text{or} \quad (r > 1.60 \ \text{and}\ L_1 > 0.25\ \text{m}) \quad \text{or} \quad (\max\text{-axis}(C_k) > 0.60\ \text{m}) \tag{7}$$

### 2.5.2. Skeleton Density Analysis

For each flagged cluster, points were projected onto the first principal axis:

$$s_i = (\mathbf{p}_i - \bar{\mathbf{p}}_k) \cdot \mathbf{v}_1 \tag{8}$$

The projection range $[s_{\text{min}}, s_{\text{max}}]$ of length $\ell = s_{\text{max}} - s_{\text{min}}$ was partitioned into $N = 15$ equal bins (minimum bin width $0.02$ m). Per-bin point counts formed a 1D density profile $\boldsymbol{\rho} \in \mathbb{N}^{N}$, smoothed via Gaussian filtering ($\sigma = 0.5$ bins):

$$\tilde{\boldsymbol{\rho}} = \boldsymbol{\rho} * G_{\sigma=0.5} \tag{9}$$

### 2.5.3. Valley Detection and Splitting

Peaks in $\tilde{\boldsymbol{\rho}}$ were detected using `find_peaks` (minimum inter-peak distance $0.30$ m; minimum relative height $0.1 \cdot \max(\tilde{\boldsymbol{\rho}})$). If two or more peaks $\{p_1, \dots, p_m\}$ were found, the deepest valley between $p_1$ and $p_m$ was located:

$$v^* = \arg\min_{b \in [p_1, p_m]} \tilde{\rho}_b, \quad \eta = \frac{\tilde{\rho}_{v^*}}{\min(\tilde{\rho}_{p_1}, \tilde{\rho}_{p_m})} \tag{10}$$

A cut was triggered when $\eta < \tau_{\text{valley}}$, with adaptive threshold:

$$\tau_{\text{valley}} = \begin{cases} 0.90 & \ell \leq 0.60\ \text{m} \\ 0.98 & \ell > 0.60\ \text{m} \end{cases} \tag{11}$$

Points were partitioned by their signed distance to the cutting plane, defined as passing through the valley bin centroid and oriented along the local skeleton tangent (computed as the normalized vector between the centroids of the adjacent non-empty bins; if either neighbor is missing, $\mathbf{v}_1$ is used as fallback):

$$\text{assign}(\mathbf{p}_i) = \begin{cases} \text{left} & (\mathbf{p}_i - \mathbf{c}_{v^*}) \cdot \mathbf{n}_{\text{cut}} < 0 \\ \text{right} & \text{otherwise} \end{cases} \tag{12}$$

Splitting was applied recursively (max depth 3, min sub-cluster size 50 points), with the full cycle embedded in an outer iterative loop (max 5 passes).

### 2.5.4. Merge-Back

After splitting, adjacent sub-clusters $(C_a, C_b)$ were merged if all three criteria held: (i) diameter of the smaller fragment $< 0.15$ m; (ii) centroid distance $< 0.20$ m; (iii) cosine similarity of first principal axes $> \cos(30^\circ) \approx 0.866$:

$$\frac{|\mathbf{v}_{1,a} \cdot \mathbf{v}_{1,b}|}{\|\mathbf{v}_{1,a}\| \cdot \|\mathbf{v}_{1,b}\|} > \cos(30^\circ) \tag{13}$$

### 2.5.5. Fragment Voting

Residual clusters were processed through three tiers: (1) $|C_k| < 500$ → discard; (2) $500 \leq |C_k| < 1500$ → merge to nearest valid cluster ($\geq 1500$ pts); (3) after assimilation, discard $|C_k| < 2000$ and cap $|C_k| > 10^5$. Unassigned points were re-assigned to the nearest valid cluster via k-d tree. Labels were re-indexed to $\{1, 2, \dots, K_{\text{final}}\}$.

## References

Campello, R.J.G.B., Moulavi, D., Sander, J., 2013. Density-based clustering based on hierarchical density estimates. In: PAKDD, pp. 160–172.

Comaniciu, D., Meer, P., 2002. Mean shift: A robust approach toward feature space analysis. IEEE Trans. Pattern Anal. Mach. Intell. 24(5), 603–619.

Ester, M., Kriegel, H.P., Sander, J., Xu, X., 1996. A density-based algorithm for discovering clusters in large spatial databases with noise. In: KDD, pp. 226–231.

Fischler, M.A., Bolles, R.C., 1981. Random sample consensus: A paradigm for model fitting. Commun. ACM 24(6), 381–395.

Rusu, R.B., Cousins, S., 2011. 3D is here: Point Cloud Library (PCL). In: ICRA, pp. 1–4.
