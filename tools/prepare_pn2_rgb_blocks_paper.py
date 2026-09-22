#!/usr/bin/env python3
from pathlib import Path
import numpy as np, torch
root=Path('/root/autodl-tmp/cabbage-phenotyping'); src=root/'pointgroup_format'; out=root/'pn2_rgb_blocks'
for split in ['train','val','test']:
 od=out/split; od.mkdir(parents=True,exist_ok=True); total=0
 for path in sorted((src/split).glob('*.pth')):
  d=torch.load(path,map_location='cpu',weights_only=False); xyz=np.asarray(d[0],np.float32); rgb=np.asarray(d[1],np.float32); sem=np.asarray(d[2],np.float32)
  lo=xyz[:,:2].min(0); hi=xyz[:,:2].max(0); k=0
  for x0 in np.arange(lo[0],hi[0],.5):
   for y0 in np.arange(lo[1],hi[1],.5):
    m=(xyz[:,0]>=x0)&(xyz[:,0]<x0+1)&(xyz[:,1]>=y0)&(xyz[:,1]<y0+1)
    if int(m.sum())<100: continue
    np.save(od/f'{path.stem}_block_{k:05d}.npy',np.column_stack([xyz[m],rgb[m],sem[m]])); k+=1
  total+=k; print(split,path.name,k,flush=True)
 print(split,'TOTAL',total,flush=True)
