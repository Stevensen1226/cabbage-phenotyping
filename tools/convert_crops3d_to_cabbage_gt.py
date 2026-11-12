#!/usr/bin/env python3
"""Convert Crops3D PLY files to the cabbage project's semantic+instance format."""
from __future__ import annotations
import argparse,json,re
from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement

def find_prop(names, candidates):
    lower={n.lower():n for n in names}
    for c in candidates:
        if c.lower() in lower: return lower[c.lower()]
    raise KeyError(f'none of {candidates} found; properties={names}')

def read_ply(path):
    ply=PlyData.read(str(path)); vertex=ply['vertex']; names=list(vertex.data.dtype.names or [])
    xyz=np.column_stack([np.asarray(vertex[find_prop(names,['x','X'])]),np.asarray(vertex[find_prop(names,['y','Y'])]),np.asarray(vertex[find_prop(names,['z','Z'])])]).astype(np.float32)
    try:
        rgb=np.column_stack([np.asarray(vertex[find_prop(names,['red','r'])]),np.asarray(vertex[find_prop(names,['green','g'])]),np.asarray(vertex[find_prop(names,['blue','b'])])]).astype(np.float32)
        if np.nanmax(rgb)<=1.01: rgb*=255.0
    except KeyError:
        rgb=np.full_like(xyz,128.0)
    return xyz,rgb,np.clip(np.rint(rgb),0,255).astype(np.uint8),names

def label_property(names, preferred=None):
    if preferred:
        return find_prop(names,[preferred])
    for cand in ['scalar_sf','scalar_Constant','sf','label','instance','instance_id']:
        try:
            return find_prop(names,[cand])
        except KeyError:
            pass
    for n in names:
        if n.lower().startswith('scalar'):
            return n
    raise KeyError(f'no label property found; properties={names}')

def split_for(path, root, split_dir):
    text=(str(path).replace('\\','/'))
    for split in ('train','test'):
        f=split_dir/f'{split}.txt'
        if not f.exists(): continue
        for line in f.read_text(encoding='utf-8',errors='ignore').splitlines():
            item=line.strip().replace('\\','/')
            if item and (text.endswith(item) or Path(item).stem==path.stem): return split
    return 'external'

def write_ply(path,xyz,rgb):
    arr=np.empty(len(xyz),dtype=[('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1')])
    arr['x'],arr['y'],arr['z']=xyz[:,0],xyz[:,1],xyz[:,2];arr['red'],arr['green'],arr['blue']=rgb[:,0],rgb[:,1],rgb[:,2]
    PlyData([PlyElement.describe(arr,'vertex')],text=False).write(str(path))

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--mode',choices=['instance','organ'],default='instance');p.add_argument('--field',default=None);p.add_argument('--splits',default=None);p.add_argument('--organ-binary',action='store_true');a=p.parse_args()
    src=Path(a.input);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);split_dir=Path(a.splits) if a.splits else None;rows=[]
    for f in sorted(src.rglob('*.ply')):
        xyz,rgb,rgb8,names=read_ply(f); raw=np.asarray(PlyData.read(str(f))['vertex'][label_property(names,a.field)]).reshape(-1); raw_i=np.rint(raw).astype(np.int64)
        if a.mode=='instance':
            sem=(raw_i>0).astype(np.int64); inst=np.where(raw_i>0,raw_i,-1); mapped=raw_i
        else:
            sem=(raw_i>0).astype(np.int64) if a.organ_binary else raw_i.copy(); inst=np.full(len(raw_i),-1,dtype=np.int64); mapped=raw_i
        split=split_for(f,src,split_dir) if split_dir else 'external'; od=out/split; od.mkdir(parents=True,exist_ok=True); stem=re.sub(r'[^0-9A-Za-z._-]+','_',f.stem)
        write_ply(od/f'{stem}.ply',xyz,rgb8); gt=np.column_stack([xyz,rgb8,sem,inst]);np.savetxt(od/f'{stem}_gt.txt',gt,fmt='%.6f %.6f %.6f %d %d %d %d %d')
        rows.append({'source':str(f),'output_ply':str(od/f'{stem}.ply'),'output_gt':str(od/f'{stem}_gt.txt'),'split':split,'points':int(len(xyz)),'plants':int(len(np.unique(inst[inst>0]))) if a.mode=='instance' else 0,'raw_sf_unique':sorted(np.unique(raw_i).tolist())[:100]})
    (out/'conversion_manifest.json').write_text(json.dumps({'mode':a.mode,'files':rows},indent=2),encoding='utf-8');print(f'converted {len(rows)} files -> {out}')
if __name__=='__main__':main()
