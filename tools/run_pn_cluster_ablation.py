#!/usr/bin/env python3
"""PointNeXt-L 聚类方法对比 — 批量运行 & 汇总
用法:
  /home/stevensen/miniconda3/envs/pointgroup_blackwell/bin/python tools/run_pn_cluster_ablation.py
"""
import os, sys, json, time, subprocess, glob

os.chdir('/home/stevensen/Cabbage')
PYTHON = '/home/stevensen/miniconda3/envs/pointgroup_blackwell/bin/python'
SCRIPT = 'tools/pointnext_eval.py'

# ── 需要对比的配置 ──
methods = ['watershed_3d', 'meanshift', 'dbscan', 'hdbscan', 'euclidean', 'graph_based']
configs = []
for m in methods:
    configs.append((f'configs/pn_{m}.yaml',        f'{m}+PCA',   f'output/pn_{m}_pca.json'))
    configs.append((f'configs/pn_{m}_noPCA.yaml',   f'{m}',       f'output/pn_{m}_nopca.json'))

results = []

for config_path, label, out_json in configs:
    # 跳过已完成的
    if os.path.exists(out_json):
        print(f'\n✓ {label}: 已存在 {out_json}, 跳过')
        with open(out_json) as f:
            res = json.load(f)
        s = res['summary']
        results.append((label, s))
        continue

    print(f'\n{"="*60}')
    print(f'▶ 运行: {label} ({config_path})')
    print(f'{"="*60}')

    t0 = time.time()
    rc = subprocess.run(
        [PYTHON, SCRIPT, '--config', config_path, '--output', out_json],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=1800,
    )
    elapsed = time.time() - t0

    if rc.returncode != 0:
        print(f'✗ {label} 失败 (exit={rc.returncode})')
        print(rc.stdout[-500:])
        continue

    # 提取最后几行
    lines = rc.stdout.strip().split('\n')
    for line in lines[-30:]:
        print(line)

    # 加载结果
    if os.path.exists(out_json):
        with open(out_json) as f:
            res = json.load(f)
        s = res['summary']
        results.append((label, s))
        print(f'✓ {label}: F1={s["avg_inst_f1"]:.4f}, {elapsed:.0f}s')
    else:
        print(f'✗ {label}: 输出文件未生成')

# ── 汇总 ──
print('\n\n' + '=' * 90)
print('PointNeXt-L + 不同聚类方法 — 完整对比')
print('=' * 90)

# 分组: 有 PCA vs 无 PCA
pca_results = [(l, s) for l, s in results if '+PCA' in l]
nopca_results = [(l, s) for l, s in results if '+PCA' not in l]

print(f'\n{\"方法\":<24} {\"Sem mIoU\":>8} {\"Inst Prec\":>8} {\"Inst Rec\":>8} {\"Inst F1\":>8} {\"Inst mIoU\":>8} {\"MAE\":>5}')
print('-' * 75)

for label, s in sorted(results, key=lambda x: -x[1]['avg_inst_f1']):
    marker = '◀ PCA' if '+PCA' in label else ''
    print(f'{label:<24} {s[\"avg_sem_miou\"]:>8.4f} {s[\"avg_inst_prec\"]:>8.4f} {s[\"avg_inst_rec\"]:>8.4f} {s[\"avg_inst_f1\"]:>8.4f} {s[\"avg_inst_miou\"]:>8.4f} {s[\"mae_count\"]:>4.1f}  {marker}')

print('-' * 75)

# PCA 增益
print('\nPCA 增益 (每方法):')
print(f'{\"方法\":<22} {\"F1 w/PCA\":>10} {\"F1 w/o PCA\":>10} {\"Δ\":>8} {\"%\":>8}')
print('-' * 62)
for (lp, sp), (ln, sn) in zip(
    sorted([(l, s) for l, s in pca_results], key=lambda x: x[0]),
    sorted([(l, s) for l, s in nopca_results], key=lambda x: x[0]),
):
    d = sp['avg_inst_f1'] - sn['avg_inst_f1']
    print(f'{lp.replace(\"+PCA\",\"\"):<22} {sp[\"avg_inst_f1\"]:>10.4f} {sn[\"avg_inst_f1\"]:>10.4f} {d:>+8.4f} {d/sn[\"avg_inst_f1\"]*100:>+7.0f}%')
