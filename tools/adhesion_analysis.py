#!/usr/bin/env python3
"""粘连植株簇(3/4/5颗) 的 基线 vs ADPV 实例分割对比。

数据来源:
  - GT: 测试/{场景}.ply 中的 scalar_Instance_ID (真实植株标签)
  - 基线预测: 测试/output_*/{场景}_step4_cluster_coarse.ply (基础聚类, 无 GIDM)
  - ADPV 预测: 测试/output_*/{场景}_step5_final.ply (含 GIDM + 后处理)

场景与粘连数:
  cloudR - Cloud2.segmented : 3 颗粘连
  cloudR5 - Cloud           : 4 颗粘连
  cloudR6 - Cloud           : 5 颗粘连

输出: 每个场景的 真实数/基线预测数/ADPV预测数/F1/欠分割修正/过分割修正。
"""

import os
import sys
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB20 = plt.get_cmap('tab20')


def read_cc_ply(fp):
    """读 CloudCompare PLY, 返回 (xyz, sem, inst)。兼容属性名差异 + ascii/binary。"""
    props = []
    fmt = None
    vertex = 0
    with open(fp, 'rb') as f:
        while True:
            line = f.readline()
            t = line.decode('latin1').strip()
            if t == 'end_header':
                break
            if t.startswith('format'):
                fmt = t.split()[1]
            if t.startswith('element vertex'):
                vertex = int(t.split()[-1])
            if t.startswith('property'):
                props.append(t.split()[1:])

    # 找语义和实例属性 (兼容 scalar_Semantic_Label_ 尾下划线)
    def find(name_key):
        for i, p in enumerate(props):
            if p[-1].startswith(name_key):
                return i, p[-1]
        return None, None

    si, sem_name = find('scalar_Semantic')
    ii, inst_name = find('scalar_Instance')
    ti = [i for i, p in enumerate(props) if p[-1] == 'x'][0]

    if fmt == 'ascii':
        # ascii PLY: 统计 header 行数 (到 end_header), 跳过
        header_lines = 0
        with open(fp, 'r') as f:
            for line in f:
                header_lines += 1
                if line.strip() == 'end_header':
                    break
        data = np.loadtxt(fp, skiprows=header_lines)
    else:
        types = [p[0] for p in props]
        names = [p[-1] for p in props]
        dt = np.dtype([(n, '<f4' if t == 'float' else ('u1' if t == 'uchar' else '<f4')) for n, t in zip(names, types)])
        with open(fp, 'rb') as f:
            while f.readline().strip() != b'end_header':
                pass
            raw = np.fromfile(f, dtype=dt, count=vertex)
        data = np.column_stack([raw[n] for n in names])

    xyz = data[:, ti:ti + 3].astype(np.float64)
    sem = data[:, si].astype(int)
    inst = data[:, ii].astype(int)
    return xyz, sem, inst


def read_step_ply(fp):
    """读中间步骤 PLY (只有 xyz+rgb, 颜色编码实例), 返回 (xyz, labels)。

    颜色编码规则 (get_colors_from_labels): label%20 映射到 tab20, -1=噪声[0.1,0.1,0.1]。
    反推: 每个唯一颜色 -> 一个实例标签。
    """
    pcd = o3d.io.read_point_cloud(fp)
    xyz = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    # 噪声点颜色 [0.1,0.1,0.1]
    noise = (np.abs(colors - 0.1).sum(1) < 0.03)
    # 用 tab20 反查: 对每个非噪声点, 找最近 tab20 颜色
    tab = TAB20(np.linspace(0, 1, 20))[:, :3]
    labels = np.full(len(xyz), -1, dtype=int)
    valid = ~noise
    if valid.sum() > 0:
        d = np.linalg.norm(colors[valid][:, None, :] - tab[None, :, :], axis=2)
        labels[valid] = d.argmin(1)  # tab20 索引 = label % 20
    return xyz, labels


def align_gt_to_pred(gt_xyz, gt_inst, pred_xyz):
    """把 GT 实例标签通过最近邻对齐到预测点。"""
    from scipy.spatial import cKDTree
    tree = cKDTree(gt_xyz)
    _, idx = tree.query(pred_xyz, k=1)
    return gt_inst[idx]


def instance_f1(pred, gt):
    """实例级 F1 (IoU=0.5 匹配, 一对一)。返回 (prec, rec, f1, tp, n_pred, n_gt)。"""
    pred_ids = np.unique(pred[pred > 0])
    gt_ids = np.unique(gt[gt > 0])
    matches = []
    matched_gt = set()
    for pid in pred_ids:
        pm = pred == pid
        p_area = pm.sum()
        best_iou, best_gid = 0.0, -1
        under = gt[pm]
        for gid in np.unique(under[under > 0]):
            gm = gt == gid
            g_area = gm.sum()
            inter = (under == gid).sum()
            union = p_area + g_area - inter
            iou = inter / union
            if iou > best_iou:
                best_iou, best_gid = iou, gid
        if best_iou >= 0.5 and best_gid not in matched_gt:
            matches.append((pid, best_gid, best_iou))
            matched_gt.add(best_gid)
    tp = len(matches)
    n_pred = len(pred_ids)
    n_gt = len(gt_ids)
    prec = tp / n_pred if n_pred else 0.0
    rec = tp / n_gt if n_gt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1, tp, n_pred, n_gt


def analyze_scene(gt_ply, base_dir, adv_dir, scene, base_step='step4_cluster_coarse'):
    """单个场景的基线 vs ADPV 对比。

    base_step: 基线步骤文件名 (不含 .ply)。
      - RandLA-Net: step4_cluster_coarse (基础聚类, 无 GIDM)
      - PointGroup: step3.5_pointgroup_proposal (offset 提案 + NMS, 未补全)
    """
    xyz_gt, sem_gt, inst_gt = read_cc_ply(gt_ply)
    # 只保留甘蓝点 (sem>0)
    cabbage = sem_gt > 0
    xyz_g = xyz_gt[cabbage]
    inst_g = inst_gt[cabbage]
    K = len(np.unique(inst_g[inst_g > 0]))

    results = {'scene': scene, 'K': K}

    for tag, d, step in [('base', base_dir, base_step), ('adv', adv_dir, 'step5_final')]:
        xyz_p, lab_p = read_step_ply(os.path.join(d, f'{scene}_{step}.ply'))
        # 对齐 GT 到预测点
        gt_aligned = align_gt_to_pred(xyz_g, inst_g, xyz_p)
        prec, rec, f1, tp, n_pred, n_gt = instance_f1(lab_p, gt_aligned)
        results[f'{tag}_n_pred'] = n_pred
        results[f'{tag}_n_gt'] = n_gt
        results[f'{tag}_prec'] = prec
        results[f'{tag}_rec'] = rec
        results[f'{tag}_f1'] = f1
        results[f'{tag}_tp'] = tp
    return results


def main():
    base_randla = os.path.join(ROOT, '测试', 'output_RandLA-Net')
    base_pg = os.path.join(ROOT, '测试', 'output_pointgroup')
    test_dir = os.path.join(ROOT, '测试')

    scenes = [
        ('cloudR - Cloud2.segmented', 'cloudR - Cloud2.segmented.ply', 3),
        ('cloudR5 - Cloud', 'cloudR5 - Cloud.ply', 4),
        ('cloudR6 - Cloud', 'cloudR6 - Cloud.ply', 5),
    ]

    print('=' * 90)
    print(f'{"粘连":<6}{"backbone":<14}{"真实":>5}{"基线N":>6}{"ADPV_N":>7}{"基线F1":>8}{"ADPV_F1":>9}{"ΔF1":>7}')
    print('-' * 90)

    all_rows = []
    for scene, ply, k in scenes:
        gt_ply = os.path.join(test_dir, ply)
        for tag, d, base_step in [('RandLA', base_randla, 'step4_cluster_coarse'),
                                  ('PointGroup', base_pg, 'step3.5_pointgroup_proposal')]:
            r = analyze_scene(gt_ply, d, d, scene, base_step=base_step)
            r['k'] = k
            r['backbone'] = tag
            all_rows.append(r)
            df1 = r['adv_f1'] - r['base_f1']
            print(f'{k}颗     {tag:<14}{r["K"]:>5}{r["base_n_pred"]:>6}{r["adv_n_pred"]:>7}'
                  f'{r["base_f1"]:>8.3f}{r["adv_f1"]:>9.3f}{df1:>+7.3f}')

    print('=' * 90)
    print()

    # 汇总: 每个粘连等级的欠分割/过分割修正
    print('=== 欠分割/过分割修正分析 ===')
    print(f'{"粘连":<6}{"backbone":<14}{"真实":>5}{"基线N":>6}{"ADPV_N":>7}  判断')
    print('-' * 70)
    for r in all_rows:
        k = r['k']
        K = r['K']
        nb = r['base_n_pred']
        na = r['adv_n_pred']
        if nb < K:
            status = f'基线欠分割(漏{ K-nb }株) → ADPV {na}'
        elif nb > K:
            status = f'基线过分割(多{nb-K}块) → ADPV {na}'
        else:
            status = f'基线正确({nb}株) → ADPV {na}'
        print(f'{k}颗     {r["backbone"]:<14}{K:>5}{nb:>6}{na:>7}  {status}')

    # 保存 json
    import json
    out = os.path.join(ROOT, 'output', 'adhesion_analysis.json')
    json.dump(all_rows, open(out, 'w'), indent=2, default=float)
    print(f'\n已保存: {out}')


if __name__ == '__main__':
    main()
