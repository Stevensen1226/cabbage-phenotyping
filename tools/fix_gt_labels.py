#!/usr/bin/env python3
"""修复 GT 标注中的漏标点。

两类漏标点（语义=甘蓝(1) 但无实例编号）：
1. 实例标签为 NaN 的点（308 个，分散在 10 个文件）
2. 实例标签为 -1 的点（cloudR5/6/7/8，共约 2.3 万个）

修复方式：将每个漏标点按空间最近邻，归入最近的已标注实例（inst>0 的甘蓝点）。
"""
import glob
import os
import numpy as np
from scipy.spatial import cKDTree

FMT = ['%.8f', '%.8f', '%.8f', '%d', '%d', '%d', '%.6f', '%.6f']


def fix_file(path):
    d = np.loadtxt(path)
    if d.ndim != 2 or d.shape[1] != 8:
        print(f'  [SKIP] {os.path.basename(path)}: shape={d.shape}')
        return 0

    sem = d[:, -2].copy()
    inst = d[:, -1].copy()

    sem1 = (sem == 1)
    good = sem1 & (inst > 0) & (~np.isnan(inst))
    bad = sem1 & (np.isnan(inst) | (inst <= 0))

    n_bad = int(bad.sum())
    if n_bad == 0:
        return 0

    good_pts = d[good, :3]
    good_inst = inst[good].astype(np.int64)
    bad_pts = d[bad, :3]

    tree = cKDTree(good_pts)
    _, idx = tree.query(bad_pts, k=1)
    inst[bad] = good_inst[idx]

    out = np.column_stack([
        d[:, :3],
        d[:, 3:6].astype(np.int64),
        sem.reshape(-1, 1),
        inst.reshape(-1, 1),
    ])
    np.savetxt(path, out, fmt=FMT)
    return n_bad


def main():
    files = sorted(glob.glob('evalaute_test/cloudR*_gt.txt'))
    total = 0
    for f in files:
        n = fix_file(f)
        if n > 0:
            print(f'{os.path.basename(f)}: 修复 {n} 个点')
        total += n
    print(f'\n总计修复: {total} 个点')


if __name__ == '__main__':
    main()
