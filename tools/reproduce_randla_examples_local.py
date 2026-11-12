from pathlib import Path
import sys
ROOT=Path(r'E:\Cabbage'); sys.path.insert(0,str(ROOT))
import numpy as np
from tools.rerender_randla_rapeseed_examples import render
z=np.load(ROOT/'server_backup/autodl-tmp/logs/randla_examples_data.npz',allow_pickle=True)
outdir=ROOT/'output/crops3d_generalization/figures/reproduced'; outdir.mkdir(parents=True,exist_ok=True)
for prefix,outname in [('r','rapeseed_randlanet_reproduced.png'),('m','maize_randlanet_reproduced.png')]:
    rows=[]
    for i in (0,1):
        rows.append(tuple(z[f'{prefix}{i}_{k}'] for k in ['label','xyz','gt','rgb','sem','base','dp']))
    render(rows,outdir/outname,seed=0)
