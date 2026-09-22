#!/usr/bin/env python3
"""Build a clean paper release without duplicated or leaking data."""
from __future__ import annotations
import hashlib, json
from datetime import date
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import numpy as np

ROOT=Path(r'E:\Cabbage')
STAMP=date.today().strftime('%Y%m%d')
DIST=ROOT/'dist'/f'cabbage_github_release_clean_{STAMP}'
DIST.mkdir(parents=True,exist_ok=True)
SPLIT=json.loads((ROOT/'evalaute_test/split.json').read_text(encoding='utf-8'))
EVAL=ROOT/'evalaute_test'
PTH=ROOT/'e_data/pointgroup_format/train'
SOURCE_FILES=['.editorconfig','.gitignore','README.md','requirements.txt','setup.py','train.py','main.py','evaluate.py','commands.txt']
SOURCE_DIRS=['cabbage_pheno','configs','docs','tools','PointNet2_Ours','RandLANet_Ours','PointGroup_Ours','SoftGroup-main','SoftGroup_Ours']
MODEL_DIRS={'PointNet2_Ours','RandLANet_Ours','PointGroup_Ours','SoftGroup-main','SoftGroup_Ours'}
CODE_EXT={'.py','.yaml','.yml','.json','.md','.txt','.sh','.bat','.cpp','.cu','.c','.h','.hpp','.toml','.cfg','.ini'}
EXCLUDE_PARTS={'__pycache__','.git','.idea','.vscode','build','dist','exp','checkpoints','logs','log','output','visualizations'}
EXCLUDE_EXT={'.pyc','.pyo','.so','.pyd','.dll','.lib','.exp','.obj','.o','.pth','.pt','.ckpt','.onnx','.npy','.npz','.log','.tfevents','.png','.jpg','.jpeg','.gif','.pdf','.docx','.zip','.7z','.rar','.svg'}

def excluded(p:Path,model:bool=False):
 rel=p.relative_to(ROOT)
 if any(x in rel.parts for x in EXCLUDE_PARTS): return True
 if p.name.startswith('.tmp_') or p.suffix.lower() in EXCLUDE_EXT: return True
 if model and p.suffix and p.suffix.lower() not in CODE_EXT: return True
 return False

def source_entries():
 for n in SOURCE_FILES:
  p=ROOT/n
  if p.is_file(): yield p,p.relative_to(ROOT).as_posix()
 for d in SOURCE_DIRS:
  base=ROOT/d
  if not base.exists(): continue
  for p in base.rglob('*'):
   if p.is_file() and not excluded(p,d in MODEL_DIRS): yield p,p.relative_to(ROOT).as_posix()

def raw_entries():
 for split in ('train','val','test'):
  for name in SPLIT[split]:
   for suffix in ('.ply','_gt.txt'):
    p=EVAL/f'{name}{suffix}'
    if p.exists(): yield p,f'dataset/{split}/{p.name}'

def pth_entries():
 for split in ('train','val','test'):
  for name in SPLIT[split]:
   p=PTH/f'{name}.pth'
   if p.exists(): yield p,f'pointgroup_format/{split}/{p.name}'

def dataset_manifest():
 out={'subregions':23,'plants':0,'points':0,'splits':{}}
 for split in ('train','val','test'):
  info={'subregions':len(SPLIT[split]),'plants':0,'points':0,'files':[]}
  for name in SPLIT[split]:
   gt=EVAL/f'{name}_gt.txt'; data=np.loadtxt(gt); plants=int(len(np.unique(data[:,-1][data[:,-1]>0])))
   info['files'].append({'id':name,'points':len(data),'plants':plants})
   info['plants']+=plants; info['points']+=len(data)
  out['splits'][split]=info; out['plants']+=info['plants']; out['points']+=info['points']
 return out

def write_zip(path,entries,texts):
 seen=set(); n=0
 with ZipFile(path,'w',compression=ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
  for src,arc in entries:
   if arc in seen: continue
   z.write(src,arc); seen.add(arc); n+=1
  for arc,txt in texts.items():
   if arc not in seen: z.writestr(arc,txt); seen.add(arc)
 print('built',path.name,'files',n,'sizeMB',round(path.stat().st_size/1024/1024,2))

def sha256(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

manifest=dataset_manifest()
manifest_text=json.dumps(manifest,indent=2)
source_zip=DIST/f'cabbage_source_{STAMP}_clean.zip'
data_zip=DIST/f'cabbage_paper_dataset_raw_{STAMP}.zip'
pth_zip=DIST/f'cabbage_pointgroup_format_{STAMP}.zip'

source_readme='''# 源码包说明\n\n只包含论文复现所需源码、配置和工具，不包含数据、模型权重、实验结果及临时文件。\n'''
data_readme='''# 论文数据集说明\n\n目录按训练、验证、测试严格隔离，避免数据泄漏：\n\n- dataset/train: 13个子区域，203株\n- dataset/val: 4个子区域，60株\n- dataset/test: 6个子区域，102株\n\nPLY为点云，`_gt.txt`为逐点标签：`x y z r g b semantic instance`。\n\n统一数据合计：23个子区域、365株、4,581,227点。论文中如仍写约422万点，应改为4,581,227点。\n'''
pth_readme='''# PointGroup格式数据\n\n按train、val、test隔离存放，避免训练集与验证/测试集混用。\n'''

write_zip(source_zip,source_entries(),{'README_RELEASE.md':source_readme})
write_zip(data_zip,raw_entries(),{'README_DATASET.md':data_readme,'dataset_manifest.json':manifest_text})
write_zip(pth_zip,pth_entries(),{'README_POINTGROUP.md':pth_readme,'dataset_manifest.json':manifest_text})

lines=[]
for p in (source_zip,data_zip,pth_zip): lines.append(f'{sha256(p)}  {p.name}')
(DIST/'SHA256SUMS.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(DIST/'release_manifest.json').write_text(json.dumps({'date':STAMP,'dataset':manifest},indent=2),encoding='utf-8')
print('release_dir',DIST)
print('\n'.join(lines))
