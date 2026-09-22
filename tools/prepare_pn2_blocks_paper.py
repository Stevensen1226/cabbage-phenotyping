#!/usr/bin/env python3
"""Create deterministic PointNet2 blocks for the paper split."""
from pathlib import Path
import numpy as np, torch
root=Path('/root/autodl-tmp/cabbage-phenotyping')
src=root/'pointgroup_format'; out=root/'pn2_blocks'
block_size=1.0; stride=0.5; min_points=100
for split in ['train','val','test']:
    od=out/split; od.mkdir(parents=True,exist_ok=True)
    total=0
    for path in sorted((src/split).glob('*.pth')):
        d=torch.load(path,map_location='cpu',weights_only=False)
        xyz=np.asarray(d[0],dtype=np.float32); sem=np.asarray(d[2],dtype=np.float32)
        lo=xyz[:,:2].min(0); hi=xyz[:,:2].max(0); k=0; xs=np.arange(lo[0],hi[0],stride); ys=np.arange(lo[1],hi[1],stride)
        for x0 in xs:
            for y0 in ys:
                mask=(xyz[:,0]>=x0)&(xyz[:,0]<x0+block_size)&(xyz[:,1]>=y0)&(xyz[:,1]<y0+block_size)
                if int(mask.sum())<min_points: continue
                np.save(od/f'{path.stem}_block_{k:05d}.npy',np.column_stack([xyz[mask],sem[mask]])); k+=1
        total+=k; print(split,path.name,len(xyz),'blocks',k,flush=True)
    print(split,'TOTAL',total,flush=True)
