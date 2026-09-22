"""
Pheno4D 叶片投影密度可视化试点脚本
=====================================

目的: 在决定 Pheno4D 泛化实验方向之前, 先看看番茄叶片的"PCA 主轴投影密度曲线"
长什么样, 判断 DPVIS (密度峰谷拆分) 是否可能泛化到叶片粘连场景。

方法: 复用 cabbage_pheno/instance/clustering.py 中 _compute_geometric_features
的 PCA 投影逻辑 (cov -> eigh -> 最大特征值方向 v1 -> dot(pts-center, v1)),
对叶点云做投影 -> 分 bin -> 高斯平滑 -> find_peaks 找峰, 并可视化。

用法:
  python tools/pheno4d_probe_density.py
  python tools/pheno4d_probe_density.py --sample Tomato01/T01_0305_a.txt
  python tools/pheno4d_probe_density.py --sample Tomato01/T01_0305_a.txt --top 6
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHENO4D_ROOT = os.path.join(ROOT, "Pheno4D", "Pheno4D")
OUT_DIR = os.path.join(ROOT, "output", "pheno4d_probe")
os.makedirs(OUT_DIR, exist_ok=True)


def compute_features(points):
    """与 clustering.py 的 _compute_geometric_features 保持一致的 PCA 投影。
    返回 {'v1': 主轴方向, 'center': 质心, 'extents': [L1,L2,L3]}。
    """
    center = points.mean(axis=0)
    centered = points - center
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    proj = np.dot(centered, evecs)
    extents = proj.max(axis=0) - proj.min(axis=0)
    return {"v1": evecs[:, 0], "center": center, "extents": extents}


def density_profile(points, nbins=30):
    """沿主轴投影 -> 密度曲线 + 峰检测。返回 (bins_center, density, peaks)。"""
    feat = compute_features(points)
    v1, center = feat["v1"], feat["center"]
    scalars = np.dot(points - center, v1)
    s_min, s_max = scalars.min(), scalars.max()
    if s_max - s_min < 1e-9:
        return None, None, None, feat

    bins = np.linspace(s_min, s_max, nbins + 1)
    hist, _ = np.histogram(scalars, bins=bins)
    hist = hist.astype(float)
    # 单位: mm (与输入坐标单位一致)
    centers_bin = (bins[:-1] + bins[1:]) / 2.0

    density = gaussian_filter(hist, sigma=0.8)
    peaks, _ = find_peaks(density, distance=max(1, nbins // 10),
                          height=np.max(density) * 0.05)
    return centers_bin, density, peaks, feat


def load_leaf_points(txt_path):
    """读取 Pheno4D txt (x y z label), 提取叶点 (label==2)。"""
    arr = np.loadtxt(txt_path)
    xyz = arr[:, :3]
    label = arr[:, 3].astype(int)
    mask = label == 2
    if mask.sum() < 50:
        # 部分样本叶标签编号可能不同, 打印 label 分布便于排查
        from collections import Counter
        print(f"    [警告] label==2 的点只有 {mask.sum()} 个, label 分布: {Counter(label.tolist())}")
    return xyz[mask]


def cluster_leaf_instances(leaf_xyz, eps_mm=5.0, min_samples=40):
    """欧式聚类 (DBSCAN) 得到粘连叶簇候选。坐标单位 mm。"""
    db = DBSCAN(eps=eps_mm, min_samples=min_samples, n_jobs=-1).fit(leaf_xyz)
    labels = db.labels_
    n = labels.max() + 1
    print(f"    DBSCAN(eps={eps_mm}mm, min_samples={min_samples}) 得到 {n} 个叶簇 (噪声 {sum(labels == -1)} 点)")
    clusters = []
    for c in range(n):
        pts = leaf_xyz[labels == c]
        if len(pts) >= 200:
            clusters.append((c, pts))
    clusters.sort(key=lambda x: -len(x[1]))
    return clusters


def plot_panel(ax, points, title):
    """画单叶簇的投影密度曲线 + 峰标记。"""
    centers_bin, density, peaks, feat = density_profile(points, nbins=30)
    if centers_bin is None:
        ax.set_title(title + "\n(无法投影)", fontsize=8)
        return

    L1, L2, L3 = feat["extents"]
    ax.plot(centers_bin / 10.0, density, color="#3b6fb6", lw=1.6)
    if peaks is not None and len(peaks) > 0:
        ax.plot(centers_bin[peaks] / 10.0, density[peaks], "o",
                color="#c0392b", ms=4, label=f"{len(peaks)} peaks")
    ax.set_title(f"{title}\nL1={L1/10:.1f}cm  peaks={len(peaks) if peaks is not None else 0}",
                 fontsize=7)
    ax.set_xlabel("projection (cm)", fontsize=7)
    ax.set_ylabel("density", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(ls=":", alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="Tomato01/T01_0305_a.txt")
    ap.add_argument("--top", type=int, default=6, help="展示点数最多的前 N 个叶簇")
    ap.add_argument("--eps_mm", type=float, default=5.0)
    args = ap.parse_args()

    txt_path = os.path.join(PHENO4D_ROOT, args.sample)
    print(f"读取: {txt_path}")
    leaf_xyz = load_leaf_points(txt_path)
    print(f"叶点云点数: {len(leaf_xyz)}")

    # 1) 整株叶点云整体投影
    clusters = cluster_leaf_instances(leaf_xyz, eps_mm=args.eps_mm)
    top = clusters[:args.top]
    n_panels = 1 + len(top)

    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows), dpi=300)
    axes = np.atleast_1d(axes).ravel()

    # 整株投影
    plot_panel(axes[0], leaf_xyz, "whole plant leaf cloud")
    # 各粘连叶簇
    for i, (cid, pts) in enumerate(top, start=1):
        plot_panel(axes[i], pts, f"leaf cluster {cid} (n={len(pts)})")

    # 隐藏多余子图
    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{args.sample}: leaf projection density (DPVIS 泛化可行性试点)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = os.path.join(OUT_DIR, "leaf_projection_probe.png")
    fig.savefig(out_png, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"已保存: {out_png}")


if __name__ == "__main__":
    main()
