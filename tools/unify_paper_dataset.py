#!/usr/bin/env python3
"""Unify cloudR22/R23 and write the paper's 13/4/6 dataset split."""
from __future__ import annotations
import json, shutil
from pathlib import Path
import numpy as np, open3d as o3d, torch

ROOT=Path(r'E:\Cabbage')
NEW_R22_PLY=Path(r'E:\cloudR22 - Cloud..ply')
NEW_R22_TXT=Path(r'E:\cloudR22.txt')
OLD_R22=ROOT/'e_data/pointgroup_format/train/cloudR22.pth'
R23_PTH=ROOT/'e_data/pointgroup_format/train/cloudR23.pth'
BACKUP=ROOT/'dist/backup_before_R22_replace'
BACKUP.mkdir(parents=True,exist_ok=True)

def backup(path:Path):
    if path.exists():
        dst=BACKUP/path.name
        if not dst.exists(): shutil.copy2(path,dst)
        print('backup',dst)

def load_new_r22():
    pcd=o3d.io.read_point_cloud(str(NEW_R22_PLY)); xyz=np.asarray(pcd.points,dtype=np.float32); rgb=np.asarray(pcd.colors,dtype=np.float32)
    gt=np.loadtxt(NEW_R22_TXT,dtype=np.float64)
    if len(xyz)!=len(gt): raise RuntimeError('R22 PLY/TXT point count mismatch')
    if np.max(np.abs(gt[:,:3]-xyz))>1e-5: raise RuntimeError('R22 PLY/TXT order mismatch')
    sem=gt[:,6].astype(np.int64); inst=gt[:,7].astype(np.int64)
    if not set(np.unique(sem)).issubset({0,1}): raise RuntimeError('R22 semantic labels must be 0/1')
    if not np.all(inst[sem==0]==-1): raise RuntimeError('R22 background instance IDs must be -1')
    return xyz,rgb,sem,inst

def write_ply_gt(prefix:Path,xyz,rgb,sem,inst):
    pcd=o3d.geometry.PointCloud(); pcd.points=o3d.utility.Vector3dVector(xyz); pcd.colors=o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud(str(prefix.with_suffix('.ply')),pcd)
    gt=np.column_stack([xyz,np.round(rgb*255).astype(np.uint8),sem,inst])
    np.savetxt(str(prefix.parent/(prefix.name+'_gt.txt')),gt,fmt='%.6f %.6f %.6f %d %d %d %d %d')
    return prefix.with_suffix('.ply'),prefix.parent/(prefix.name+'_gt.txt')

def save_pth(path:Path,xyz,rgb,sem,inst):
    inst=inst.copy(); inst[sem==0]=-100
    torch.save((xyz.astype(np.float32),rgb.astype(np.float32),sem.astype(np.int64),inst.astype(np.int64)),path)

backup(OLD_R22); backup(ROOT/'evalaute_test/split.json')
xyz22,rgb22,sem22,inst22=load_new_r22()
for base in [ROOT/'evalaute_test/cloudR22',ROOT/'e_data/train/cloudR22']:
    write_ply_gt(base,xyz22,rgb22,sem22,inst22)
save_pth(OLD_R22,xyz22,rgb22,sem22,inst22)
print('R22 replaced',len(xyz22),'points',len(np.unique(inst22[inst22>0])),'plants')

d=torch.load(R23_PTH,map_location='cpu',weights_only=False)
xyz23,rgb23,sem23,inst23=map(np.asarray,d)
inst23_txt=inst23.copy(); inst23_txt[sem23==0]=-1
for base in [ROOT/'evalaute_test/cloudR23',ROOT/'e_data/train/cloudR23']:
    write_ply_gt(base,xyz23,rgb23,sem23,inst23_txt)
print('R23 exported',len(xyz23),'points',len(np.unique(inst23_txt[inst23_txt>0])),'plants')

train=[f'cloudR{i}' for i in [1,2,3,4,6,7,9,10,12,18,19,22,23]]
val=[f'cloudR{i}' for i in [15,17,20,21]]
test=[f'cloudR{i}' for i in [5,8,11,13,14,16]]
split={'train':train,'val':val,'test':test,'seed':42,'protocol':'paper_13_4_6'}
(ROOT/'evalaute_test/split.json').write_text(json.dumps(split,indent=2),encoding='utf-8')
print('split',json.dumps(split,indent=2))
