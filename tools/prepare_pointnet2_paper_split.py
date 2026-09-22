#!/usr/bin/env python3
from pathlib import Path
import numpy as np, torch
root=Path('/root/autodl-tmp/cabbage-phenotyping')
splits=['train','val','test']
out=root/'pn2_split'
for split in splits:
    od=out/split; od.mkdir(parents=True,exist_ok=True)
    for src in sorted((root/'pointgroup_format'/split).glob('*.pth')):
        d=torch.load(src,map_location='cpu',weights_only=False)
        xyz=np.asarray(d[0],dtype=np.float32); sem=np.asarray(d[2],dtype=np.float32).reshape(-1,1)
        np.save(od/(src.stem+'.npy'),np.concatenate([xyz,sem],axis=1))
        print(split,src.name,len(xyz))
