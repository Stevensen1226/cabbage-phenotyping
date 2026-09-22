#!/usr/bin/env python3
"""过分割 (over-segmentation) 案例检测与可视化。

背景:
  GIDM 骨架密度波谷切割是实例分割的核心后处理: 对粘连的甘蓝簇, 沿第一主成分
  方向投影到一维密度曲线, 在"峰-峰"之间的波谷处落刀, 递归拆开多株粘连。

  当过分割发生时, 一株甘蓝被错误地切成 >=2 块。根因通常有三类:
    1. 假谷 (false valley): 单株甘蓝叶片间隙 / 稀疏噪声在密度曲线上制造了
       看似"株间缺口"的波谷, 且谷值比满足 valley_depth_rel 阈值 → 被误切。
    2. 假峰 (false peak): min_peak_dist_m 过小, 把叶片局部起伏当成第二个峰。
    3. 递归累积: 第一刀切错后, 子簇继续递归切割, 误差逐级放大。

本脚本:
  1. 从 evalaute_test/cloudR*.ply (自带 GT instance 标签) 提取粘连簇。
  2. 对每个簇跑 GIDM 递归切割, 得到预测分块。
  3. 检测"一株真值被切成多块"的过分割案例。
  4. 对每个案例输出 3 面板图: 真值着色 / 预测着色 / 密度剖面(标出假谷落刀点)。

用法:
  python tools/oversegmentation_analysis.py                    # 扫描全部粘连簇
  python tools/oversegmentation_analysis.py --cluster cloudR10 --instances 2 3 4 5
  python tools/oversegmentation_analysis.py --min-peak-dist 0.15  # 人为加剧过分割
  python tools/oversegmentation_analysis.py --config configs/default.yaml --top 6
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from cabbage_pheno.instance.clustering import InstanceClusterer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "evalaute_test")
OUT_DIR = os.path.join(ROOT, "output", "oversegmentation")


# ═══════════════════════════════════════════════
# 数据加载 (复用 density_sensitivity 的解析逻辑)
# ═══════════════════════════════════════════════
def load_cloud_ply(path):
    """读取 cloudR*.ply (ASCII), 返回 (xyz, semantic, instance)。"""
    with open(path, "r") as f:
        header = []
        n_vertices = 0
        for line in f:
            header.append(line.rstrip("\n"))
            if line.startswith("element vertex "):
                n_vertices = int(line.split()[-1])
            if line.startswith("end_header"):
                break
    data = np.loadtxt(path, skiprows=len(header))
    xyz = data[:, 0:3].astype(np.float64)
    semantic = np.nan_to_num(data[:, 6], nan=0.0).astype(int)
    instance = np.nan_to_num(data[:, 7], nan=-1.0).astype(int)
    return xyz, semantic, instance


def find_sticky_clusters(xyz, instance, eps=0.03, min_points=50):
    """找粘连簇: 空间连通(eps 邻域) 且含 >=2 个不同实例标签的点集。"""
    veg = instance > 0
    xyz_v = xyz[veg]
    inst_v = instance[veg]
    idx_v = np.where(veg)[0]

    if len(xyz_v) < 10:
        return []

    tree = cKDTree(xyz_v)
    n = len(xyz_v)
    visited = np.zeros(n, dtype=bool)
    clusters = []

    for start in range(n):
        if visited[start]:
            continue
        comp = [start]
        visited[start] = True
        head = 0
        while head < len(comp):
            cur = comp[head]
            head += 1
            for nb in tree.query_ball_point(xyz_v[cur], r=eps):
                if not visited[nb]:
                    visited[nb] = True
                    comp.append(nb)
        comp = np.array(comp)
        if len(comp) < min_points:
            continue
        ids = np.unique(inst_v[comp])
        ids = ids[ids > 0]
        if len(ids) < 2:
            continue
        clusters.append({
            "file": None,  # 由调用方填充
            "idx": idx_v[comp],
            "xyz": xyz_v[comp],
            "inst": inst_v[comp],
            "n_points": int(len(comp)),
            "n_instances": int(len(ids)),
            "instances": sorted(ids.tolist()),
        })
    clusters.sort(key=lambda c: -c["n_points"])
    return clusters


# ═══════════════════════════════════════════════
# 密度剖面复现 (与 _recursive_skeleton_split 一致)
# ═══════════════════════════════════════════════
def replicate_density_profile(points, skel_bins=15, min_peak_dist_m=0.35):
    center = points.mean(axis=0)
    centered = points - center
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]
    v1 = evecs[:, 0]

    scalars = np.dot(centered, v1)
    s_min, s_max = scalars.min(), scalars.max()
    length = s_max - s_min

    nbins = skel_bins
    if length / nbins < 0.02:
        nbins = max(5, int(length / 0.02))

    bins = np.linspace(s_min, s_max, nbins + 1)
    bin_indices = np.digitize(scalars, bins) - 1
    bin_indices = np.clip(bin_indices, 0, nbins - 1)

    counts = np.bincount(bin_indices, minlength=nbins).astype(float)
    density_smooth = gaussian_filter(counts, sigma=0.5)

    bin_width = length / nbins
    min_dist_bins = max(1, int(min_peak_dist_m / bin_width))
    peaks, _ = find_peaks(density_smooth, distance=min_dist_bins,
                          height=np.max(density_smooth) * 0.1)

    res = {
        "scalars": scalars, "s_min": float(s_min), "s_max": float(s_max),
        "length": float(length), "bins": bins, "counts": counts,
        "density_smooth": density_smooth, "peaks": peaks,
        "nbins": nbins, "bin_width": float(bin_width), "v1": v1, "center": center,
        "valleys": [], "cut_positions": [],
    }

    if len(peaks) >= 2:
        first_peak, last_peak = peaks[0], peaks[-1]
        if first_peak < last_peak:
            # 收集所有相邻峰之间的谷 (GIDM 递归切割依次在这些谷落刀)
            for a, b in zip(peaks[:-1], peaks[1:]):
                seg = density_smooth[a:b + 1]
                valley_idx = int(a + int(np.argmin(seg)))
                res["valleys"].append(valley_idx)
    return res


def run_gidm(clusterer, points):
    """跑真实 GIDM 递归切割, 返回分块索引列表 (每个分块是 point 下标数组)。"""
    parts = clusterer._recursive_skeleton_split(points, np.arange(len(points)), depth=0)
    return parts


# ═══════════════════════════════════════════════
# 过分割检测
# ═══════════════════════════════════════════════
def detect_oversegmentation(gt_inst, fragments, overlap_frac=0.5, cover_frac=0.2):
    """检测被切碎的真值植株。

    返回 [(gt_id, [fragment_idx...]), ...]:
      一株真值 gt_id 的点被 >=2 个预测分块覆盖, 且每个分块多数点来自该真值
      (overlap_frac) 并覆盖该真值足够比例 (cover_frac) → 判为过分割。
    """
    gt_ids = np.unique(gt_inst)
    gt_ids = gt_ids[gt_ids > 0]

    frag_gt = []  # 每个分块对应的多数真值
    for fi, frag in enumerate(fragments):
        if len(frag) == 0:
            frag_gt.append(-1)
            continue
        cnt = Counter(gt_inst[frag].tolist())
        cnt.pop(-1, None)
        cnt.pop(0, None)
        if not cnt:
            frag_gt.append(-1)
            continue
        gid, gcnt = cnt.most_common(1)[0]
        frag_gt.append(gid if gcnt / len(frag) >= overlap_frac else -1)

    over = []
    for gid in gt_ids:
        g_mask = gt_inst == gid
        g_total = int(g_mask.sum())
        covering = []
        for fi, fg in enumerate(frag_gt):
            if fg != gid:
                continue
            frag = fragments[fi]
            n_overlap = int((gt_inst[frag] == gid).sum())
            if n_overlap / max(g_total, 1) >= cover_frac:
                covering.append(fi)
        if len(covering) >= 2:
            over.append((int(gid), covering))

    return over, frag_gt


# ═══════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════
_CJK_FONT_PATHS = [
    "/mnt/i/WeGameApps/rail_apps/无畏契约(2001715)/ACLOS/DiagnoseTool/fonts/NotoSansSC-Regular.otf",
    "/mnt/i/WeGameApps/rail_apps/无畏契约(2001715)/ACLOS/DiagnoseTool/fonts/NotoSansSC-Bold.otf",
]


def _setup_font():
    """注册 Noto Sans CJK 字体, 使图中中文正常显示 (找不到则退回默认字体)。"""
    import matplotlib
    from matplotlib import font_manager
    for p in _CJK_FONT_PATHS:
        if os.path.exists(p):
            try:
                font_manager.fontManager.addfont(p)
                name = font_manager.FontProperties(fname=p).get_name()
                matplotlib.rcParams["font.family"] = name
                matplotlib.rcParams["axes.unicode_minus"] = False
                return
            except Exception:
                continue


def _scatter_colored(ax, pts, labels, title, cmap_name="tab20"):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(cmap_name)
    uniq = np.unique(labels)
    uniq = uniq[uniq >= 0]
    for i, lab in enumerate(uniq):
        m = labels == lab
        color = cmap((i % 20) / 20.0)
        ax.scatter(pts[m, 0], pts[m, 1], s=0.4, c=[color], alpha=0.8,
                   rasterized=True)
    # 背景点 (标签 <0)
    bg = labels < 0
    if bg.any():
        ax.scatter(pts[bg, 0], pts[bg, 1], s=0.2, c="0.85", alpha=0.5,
                   rasterized=True)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.2)


def make_case_figure(pts, gt_inst, fragments, frag_gt, over_gt_ids, profile, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _setup_font()

    # 预测分块标签 (对每个点)
    pred_labels = np.full(len(pts), -1, dtype=int)
    for fi, frag in enumerate(fragments):
        pred_labels[frag] = fi

    # 被切碎株的集合 (用于高亮)
    over_gt_set = set(g for g, _ in over_gt_ids)
    over_frag_set = set()
    for gid, frags in over_gt_ids:
        over_frag_set.update(frags)

    # 密度剖面诊断: 计算每个被切碎株在 v1 上的投影区间, 判定"假谷/真谷"
    scalars = np.asarray(profile["scalars"])
    bins = np.asarray(profile["bins"])
    xs = (bins[:-1] + bins[1:]) / 2
    valleys = profile.get("valleys", [])
    margin = profile["bin_width"] * 0.75

    over_intervals = {}   # gid -> (s_lo, s_hi)
    for gid in over_gt_set:
        m = gt_inst == gid
        if m.any():
            over_intervals[gid] = (float(scalars[m].min()), float(scalars[m].max()))

    # 对每个落刀点(谷)判定: 落在被切碎株内部 -> 假谷(错误刀); 否则 -> 真谷
    false_valleys = []   # 落刀点 x 位置
    true_valleys = []
    for vi in valleys:
        xv = float(xs[vi])
        is_false = any(lo - margin <= xv <= hi + margin
                       for lo, hi in over_intervals.values())
        (false_valleys if is_false else true_valleys).append((vi, xv))

    fig = plt.figure(figsize=(15, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.3])

    # (a) 真值 (被切碎株红色高亮, 其余灰色)
    ax_gt = fig.add_subplot(gs[0])
    gt_colors = ["#E74C3C", "#D35400"]
    ax_gt.scatter(pts[~np.isin(gt_inst, list(over_gt_set)) | (gt_inst < 0), 0],
                  pts[~np.isin(gt_inst, list(over_gt_set)) | (gt_inst < 0), 1],
                  s=0.3, c="0.82", alpha=0.6, rasterized=True)
    for i, gid in enumerate(sorted(over_gt_set)):
        m = gt_inst == gid
        ax_gt.scatter(pts[m, 0], pts[m, 1], s=0.8,
                      c=[gt_colors[i % 2]], alpha=0.95, rasterized=True)
    ax_gt.set_aspect("equal")
    ax_gt.set_title(f"GT 真值 (红=被切碎株: {', '.join(f'#{g}' for g in sorted(over_gt_set))})",
                    fontsize=11)
    ax_gt.set_xlabel("x (m)")
    ax_gt.set_ylabel("y (m)")
    ax_gt.grid(alpha=0.2)

    # (b) 预测 (被切碎株的多个分块用红系高亮)
    ax_pred = fig.add_subplot(gs[1])
    cmap = plt.get_cmap("tab20")
    reds = ["#E74C3C", "#F39C12", "#C0392B", "#D35400", "#E67E22"]
    used = 0
    for fi in range(len(fragments)):
        m = pred_labels == fi
        if fi in over_frag_set:
            color = reds[used % len(reds)]
            used += 1
        else:
            color = "0.82"
        ax_pred.scatter(pts[m, 0], pts[m, 1], s=0.4 if fi not in over_frag_set else 0.8,
                        c=[color], alpha=0.9, rasterized=True)
    ax_pred.set_aspect("equal")
    n_over = len(over_gt_ids)
    ax_pred.set_title("GIDM 预测 (红/橙=同株被切成多块)\n"
                      f"{n_over} 株真值 → {len(fragments)} 个分块",
                      fontsize=11)
    ax_pred.set_xlabel("x (m)")
    ax_pred.set_ylabel("y (m)")
    ax_pred.grid(alpha=0.2)

    # (c) 密度剖面 + 真/假谷诊断
    ax_d = fig.add_subplot(gs[2])
    ax_d.bar(xs, profile["counts"], width=profile["bin_width"] * 0.9,
             color="#AED6F1", alpha=0.5, label="raw counts")
    ax_d.plot(xs, profile["density_smooth"], color="#1F618D", lw=2,
              label="smoothed density $\\rho$")
    pk = profile["peaks"]
    ax_d.plot(xs[pk], profile["density_smooth"][pk], "v", ms=9,
              color="#1E8449", label="peaks")

    # 被切碎株的投影区间 (半透明红带)
    for gid, (lo, hi) in over_intervals.items():
        ax_d.axvspan(lo, hi, color="#F5B7B1", alpha=0.35, zorder=0)

    # 真谷(株间正确刀) / 假谷(株内错误刀)
    for vi, xv in false_valleys:
        ax_d.axvline(xv, color="#E74C3C", ls="--", lw=1.6)
        ax_d.plot(xv, profile["density_smooth"][vi], "X", ms=13,
                  color="#E74C3C", mew=2.5, label="假谷(误切)" if vi == false_valleys[0][0] else None)
    for vi, xv in true_valleys:
        ax_d.axvline(xv, color="#27AE60", ls=":", lw=1.2, alpha=0.7)
        ax_d.plot(xv, profile["density_smooth"][vi], "o", ms=8,
                  color="#27AE60", mfc="none", mew=1.6,
                  label="真谷(株间)" if vi == true_valleys[0][0] else None)

    ax_d.axhline(0, color="0.5", lw=0.5)
    ax_d.set_xlabel("投影到第一主成分 $v_1$ (m)")
    ax_d.set_ylabel("点数密度")
    ax_d.set_title("1D 密度剖面诊断\n"
                   f"红带=被切碎株投影区间, 红X=落在株内的假谷(误切)",
                   fontsize=11)
    ax_d.grid(alpha=0.3)
    # 图例放到面板下方外侧, 避免与密度曲线/峰值重合
    ax_d.legend(fontsize=7.5, loc="upper center",
                bbox_to_anchor=(0.5, -0.18), ncol=5, frameon=False)

    fig.suptitle(
        f"过分割失败案例: 单株真值被 GIDM 切成 {max(len(f) for _, f in over_gt_ids)} 块",
        fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--cluster", default=None, help="指定文件 cloudR10")
    ap.add_argument("--instances", nargs="+", type=int, default=None,
                    help="指定实例 ID (配合 --cluster)")
    ap.add_argument("--min-peak-dist", type=float, default=None,
                    help="覆盖 min_peak_dist_m (人为加剧过分割)")
    ap.add_argument("--valley-depth-rel", type=float, default=None,
                    help="覆盖 valley_depth_rel")
    ap.add_argument("--top", type=int, default=6, help="最多展示的案例数")
    ap.add_argument("--min-points", type=int, default=2000, help="簇最少点数")
    ap.add_argument("--min-plants", type=int, default=2, help="簇最少株数")
    ap.add_argument("--max-plants", type=int, default=6, help="簇最多株数")
    ap.add_argument("--tag", default="", help="输出目录后缀 (区分不同参数实验)")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.min_peak_dist is not None:
        cfg.setdefault("instance", {}).setdefault("pca_split", {})["min_peak_dist_m"] = args.min_peak_dist
    if args.valley_depth_rel is not None:
        cfg.setdefault("instance", {}).setdefault("pca_split", {})["valley_depth_rel"] = args.valley_depth_rel

    clusterer = InstanceClusterer(cfg)
    min_peak_dist = float(clusterer.min_peak_dist_m)
    print(f"[config] min_peak_dist_m={min_peak_dist:.2f}, "
          f"valley_depth_rel={clusterer.valley_depth_rel:.2f}, "
          f"abnormal_diameter={clusterer.abn_d:.2f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_dir = OUT_DIR if not args.tag else os.path.join(OUT_DIR, args.tag)
    os.makedirs(out_dir, exist_ok=True)

    # 1. 发现粘连簇
    clusters = []
    ply_files = sorted(
        f for f in os.listdir(DATA_DIR)
        if f.startswith("cloudR") and f.endswith(".ply")
        and "_step" not in f and "_gt" not in f
    )
    for fn in ply_files:
        xyz, sem, inst = load_cloud_ply(os.path.join(DATA_DIR, fn))
        cs = find_sticky_clusters(xyz, inst, min_points=args.min_points)
        for c in cs:
            c["file"] = fn
        clusters.extend(cs)

    print(f"[discover] 共 {len(clusters)} 个粘连簇")

    # 过滤: 株数范围 (排除超大粘连簇, 聚焦 GIDM 真正处理的 2~6 株粘连场景)
    clusters = [c for c in clusters
                if args.min_plants <= c["n_instances"] <= args.max_plants]

    # 过滤: 指定 cluster/instances
    if args.cluster:
        clusters = [c for c in clusters if c["file"].startswith(args.cluster)]
        if args.instances:
            clusters = [c for c in clusters
                        if set(args.instances).issubset(set(c["instances"]))]
    print(f"[filter] 剩余 {len(clusters)} 个簇 "
          f"({args.min_plants}~{args.max_plants} 株, >= {args.min_points} 点)")

    # 2. 逐簇跑 GIDM + 检测过分割
    over_cases = []   # 每个元素: dict(簇信息, gt_id, frags, profile)
    summary_all = []
    for ci, c in enumerate(clusters):
        pts = c["xyz"]
        gt = c["inst"]
        fragments = run_gidm(clusterer, pts)
        over, frag_gt = detect_oversegmentation(gt, fragments)

        n_gt = len(c["instances"])
        n_pred = len(fragments)
        summary_all.append({
            "file": c["file"], "instances": c["instances"],
            "n_points": c["n_points"], "n_gt": n_gt, "n_pred": n_pred,
            "over_gt": [g for g, _ in over],
        })

        if over:
            profile = replicate_density_profile(pts, skel_bins=clusterer.skel_bins,
                                                min_peak_dist_m=min_peak_dist)
            over_cases.append({
                "cluster": c, "fragments": fragments, "frag_gt": frag_gt,
                "over": over, "profile": profile,
            })

    # 3. 统计汇总
    n_over_clusters = sum(1 for s in summary_all if s["over_gt"])
    n_over_plants = sum(len(s["over_gt"]) for s in summary_all)
    n_gt_total = sum(s["n_gt"] for s in summary_all)
    print(f"[result] {n_over_clusters}/{len(summary_all)} 个粘连簇存在过分割, "
          f"{n_over_plants}/{n_gt_total} 株真值被切碎")

    # 4. 排序: 优先展示"单株被切得最多"的案例
    over_cases.sort(key=lambda x: -max(len(f) for _, f in x["over"]))

    # 5. 可视化
    shown = 0
    manifest = []
    for case in over_cases[:args.top]:
        c = case["cluster"]
        for gid, frags in case["over"]:
            if shown >= args.top:
                break
            safe = f"{c['file'].replace('.ply','')}_gt{gid}_x{len(frags)}"
            out_png = os.path.join(out_dir, f"over_{safe}.png")
            make_case_figure(c["xyz"], c["inst"], case["fragments"],
                             case["frag_gt"], [(gid, frags)],
                             case["profile"], out_png)
            manifest.append({
                "png": out_png,
                "file": c["file"], "gt_instance": gid,
                "n_fragments": len(frags),
                "cluster_instances": c["instances"],
                "cluster_n_points": c["n_points"],
                "n_pred_parts": len(case["fragments"]),
                "valleys": case["profile"]["valleys"],
                "n_peaks": int(len(case["profile"]["peaks"])),
                "length_m": case["profile"]["length"],
            })
            shown += 1
        if shown >= args.top:
            break

    out_json = os.path.join(out_dir, "oversegmentation_summary.json")
    with open(out_json, "w") as f:
        json.dump({
            "config": args.config,
            "min_peak_dist_m": min_peak_dist,
            "valley_depth_rel": clusterer.valley_depth_rel,
            "n_clusters": len(summary_all),
            "n_over_clusters": n_over_clusters,
            "n_over_plants": n_over_plants,
            "n_gt_total": n_gt_total,
            "per_cluster": summary_all,
            "cases": manifest,
        }, f, indent=2, ensure_ascii=False)
    print(f"[output] 汇总 -> {out_json}")
    for m in manifest:
        print(f"  {os.path.basename(m['png'])}  "
              f"({m['file']} GT#{m['gt_instance']} → {m['n_fragments']} 块, "
              f"谷数={len(m['valleys'])}, 峰数={m['n_peaks']}, L={m['length_m']:.2f}m)")


if __name__ == "__main__":
    main()
