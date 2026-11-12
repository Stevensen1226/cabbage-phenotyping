#!/usr/bin/env python3
"""Re-render the RandLA-Net rapeseed example figure with a different sampling seed."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]

def render(rows, out: Path, seed: int):
    def gt_palette(gt):
        vals=np.unique(gt[gt>=0]); cmap=plt.get_cmap('turbo')
        cols=cmap(np.linspace(.04,.96,max(1,len(vals))))
        return {int(v):cols[i] for i,v in enumerate(vals)}
    def gt_colors(vals,pal):
        c=np.zeros((len(vals),4)); c[:,:3]=.82; c[:,3]=1
        for g,col in pal.items(): c[vals==g]=col
        return c
    def match(pred,gt):
        m=gt>=0; p=pred[m]; g=gt[m]
        pairs,ct=np.unique(np.stack([p,g],1),axis=0,return_counts=True); inter={}
        for (a,b),k in zip(pairs,ct):
            a=int(a); b=int(b); k=int(k)
            if a>=0: inter[(a,b)]=k
        best={}
        for (a,b),k in inter.items():
            if k>best.get(a,(-1,None))[0]: best[a]=(k,b)
        return {a:b for a,(k,b) in best.items()}
    def pred_colors(pred,gt,pal):
        c=np.zeros((len(pred),4)); c[:,:3]=.82; c[:,3]=1; mp=match(pred,gt)
        for p in np.unique(pred):
            p=int(p)
            if p>=0: c[pred==p]=pal.get(mp.get(p,-1),np.array([.82,.82,.82,1.]))
        return c
    def semcolors(v):
        c=np.zeros((len(v),4)); c[:,:3]=.82; c[:,3]=1; c[v>0]=[.08,.72,.25,1]
        return c
    def style(ax,p,title):
        lo=np.percentile(p,1,axis=0); hi=np.percentile(p,99,axis=0)
        ax.set_xlim(lo[0],hi[0]); ax.set_ylim(lo[1],hi[1]); ax.set_zlim(lo[2],hi[2])
        ax.set_box_aspect(hi-lo); ax.view_init(24,-62); ax.set_proj_type('ortho')
        ax.set_axis_off(); ax.set_title(title,fontsize=24,pad=6)

    fig=plt.figure(figsize=(25,5.4*len(rows)),dpi=300)
    for r,(label,xyz,gt,rgb,sem,base,dp) in enumerate(rows):
        pal=gt_palette(gt)
        rng=np.random.default_rng(seed)
        sel=rng.choice(len(xyz),size=min(70000,len(xyz)),replace=False)
        panels=[
            ('Original RGB',rgb),
            ('GT instances',gt_colors(gt,pal)),
            ('RandLA-Net semantic',semcolors(sem)),
            ('RandLA-Net +\nwatershed',pred_colors(base,gt,pal)),
            ('RandLA-Net +\nwatershed + DPVIS',pred_colors(dp,gt,pal)),
        ]
        for c,(title,col) in enumerate(panels):
            ax=fig.add_subplot(len(rows),5,r*5+c+1,projection='3d')
            ax.scatter(xyz[sel,0],xyz[sel,1],xyz[sel,2],c=col[sel],s=.35,
                       alpha=1.0,depthshade=False,rasterized=True)
            style(ax,xyz[sel],title)
            if c==0:
                ax.text2D(-.04,.5,str(label),transform=ax.transAxes,rotation=90,
                          va='center',ha='right',fontsize=20)
    fig.tight_layout()
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    print('saved',out,flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--npz',type=Path,default=ROOT/'server_backup/autodl-tmp/logs/randla_examples_data.npz')
    ap.add_argument('--seed',type=int,default=123)
    ap.add_argument('--out',type=Path,default=None)
    a=ap.parse_args()
    if a.out is None:
        a.out=ROOT/f'output/crops3d_generalization/figures/randla_rapeseed_examples_seed{a.seed}.png'
    z=np.load(a.npz,allow_pickle=True)
    rows=[]
    for i in (0,1):
        rows.append(tuple(z[f'r{i}_{k}'] for k in ['label','xyz','gt','rgb','sem','base','dp']))
    render(rows,a.out,a.seed)

if __name__=='__main__':
    main()
