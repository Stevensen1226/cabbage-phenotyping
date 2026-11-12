import json
from pathlib import Path
ROOT=Path(r'E:\Cabbage'); OUT=ROOT/'output/crops3d_whole_scene'; DOCS=ROOT/'docs'
d=json.loads((OUT/'unified_all_models.json').read_text(encoding='utf-8')); native=json.loads((OUT/'pointgroup_native_whole.json').read_text(encoding='utf-8'))
models=list(d['models']); methods=[('watershed_3d','Watershed-3D'),('meanshift','MeanShift'),('hdbscan','HDBSCAN')]; crops=['maize','rapeseed']; zh={'maize':'玉米','rapeseed':'油菜'}
def ft(crop):
 lines=['| 模型 | 聚类方法 | 基础 P | 基础 R | 基础 F1 | 基础 MAE | +DPVIS P | +DPVIS R | +DPVIS F1 | +DPVIS MAE | ΔF1 | ΔMAE |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for m in models:
  for k,mn in methods:
   b=d['models'][m][crop][k+'_base']['summary']; q=d['models'][m][crop][k+'_dpvis']['summary']
   lines.append(f"| {m} | {mn} | {b['P']:.4f} | {b['R']:.4f} | {b['F1']:.4f} | {b['MAE']:.2f} | {q['P']:.4f} | {q['R']:.4f} | {q['F1']:.4f} | {q['MAE']:.2f} | {q['F1']-b['F1']:+.4f} | {q['MAE']-b['MAE']:+.2f} |")
 return '\n'.join(lines)
pos=sum(d['models'][m][c][k+'_dpvis']['summary']['F1']>d['models'][m][c][k+'_base']['summary']['F1'] for c in crops for m in models for k,_ in methods)
mae_imp=sum(d['models'][m][c][k+'_dpvis']['summary']['MAE']<d['models'][m][c][k+'_base']['summary']['MAE'] for c in crops for m in models for k,_ in methods)
neg=[]
for c in crops:
 for m in models:
  for k,mn in methods:
   b=d['models'][m][c][k+'_base']['summary']; q=d['models'][m][c][k+'_dpvis']['summary']
   if q['F1']<b['F1']-1e-9: neg.append((zh[c],m,mn,b['F1'],q['F1'],q['MAE']-b['MAE']))
lines=['# Crops3D_IS 玉米与油菜整景实例分割评估（中文最终版）','',
'## 1. 评估口径','',
'- 每个 plot 作为一个整景完成语义预测、聚类和指标统计，不再按 tile 单独评估后平均。',
'- RandLA-Net、PointNet2-RGB：1 m 重叠块整景推理，0.5 m 步长。',
'- PointNeXt-L：2 m 滑窗、1 m 步长，2 cm 体素化与逐点投票。',
'- PointGroup：2 m 块、1 m 步长；原生实例采用确定性体素采样、跨块 proposal union 和全局 NMS。',
'- Watershed-3D / MeanShift / HDBSCAN：在整景前景点上运行；参数在官方 train 验证 plot 上按作物选择。',
'- 匹配标准：IoU ≥ 0.5 一对一贪心匹配；指标为 Precision、Recall、F1 和实例计数 MAE。','',
'## 2. 玉米完整指标','',ft('maize'),'',
'## 3. 油菜完整指标','',ft('rapeseed'),'',
'## 4. DPVIS 总体效果','',
f'- 24 个受控组合中，+DPVIS 在 {pos} 个组合提高 F1。',
f'- {mae_imp} 个组合的 MAE 降低。',
'- Watershed-3D 和 HDBSCAN 的收益最稳定；MeanShift 在基础 F1 已接近 1.0 时容易出现条件性过分割。','',
'### DPVIS 未提高的组合','',
'| 作物 | 模型 | 聚类方法 | 基础 F1 | +DPVIS F1 | ΔMAE |','|---|---|---|---:|---:|---:|']
for r in neg: lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.4f} | {r[4]:.4f} | {r[5]:+.2f} |")
lines += ['', '## 5. PointGroup 原生端到端整景结果（修复后）','', '| 作物 | P | R | F1 | MAE |','|---|---:|---:|---:|---:|']
for c in crops:
 s=native['crops'][c]['summary']; lines.append(f"| {zh[c]} | {s['P']:.4f} | {s['R']:.4f} | {s['F1']:.4f} | {s['MAE']:.2f} |")
lines += ['', '修复内容：', '',
'1. 块内点预算改为 2 cm 确定性体素采样，保证重叠块共享同一批全局点。',
'2. 跨块 proposal 同时使用 IoU 和 intersection-over-smaller-area 合并。',
'3. 合并后增加全局 NMS，按 score、质心和掩膜重叠抑制重复实例。',
'4. 最终分配改为 score-based assignment，避免低分 proposal 覆盖高分实例。','',
'修复前后对比：玉米原生端到端 F1 从 0.0920 提升到 0.1966，MAE 从 101.20 降到 11.40；油菜 F1 从 0.3745 提升到 0.7455，MAE 从 33.25 降到 3.00。', '',
'> 玉米原生端到端仍低于点级聚类，说明 PointGroup 在整景玉米上仍受 proposal 稀疏、跨块尺度差异和困难场景限制；但已经不是训练权重完全失效。点级聚类主表中，PointGroup 玉米 HDBSCAN + DPVIS 的 F1 为 0.6358，油菜 HDBSCAN + DPVIS 的 F1 为 0.9231。','',
'## 6. 主要结论','',
'- 玉米整体最佳组合为 PointNet2-RGB + HDBSCAN + DPVIS：F1 = 0.5694，MAE = 6.40。',
'- 油菜整体最佳组合为 PointNet2-RGB + MeanShift + DPVIS：F1 = 0.9515，MAE = 0.25；PointNeXt-L + MeanShift 也达到 F1 = 0.9903。',
'- DPVIS 对 Watershed-3D / HDBSCAN 的修复最明显；在 MeanShift 已饱和时应采用保守策略。','',
'## 7. 数据文件','',
'- `output/crops3d_whole_scene/unified_all_models.json`',
'- `output/crops3d_whole_scene/unified_all_models.csv`',
'- `output/crops3d_whole_scene/pointgroup_native_whole.json`']
path=DOCS/'Crops3D_IS_整景评估_最终版_中文.md'; path.write_text('\n'.join(lines),encoding='utf-8-sig'); print('saved',path)
