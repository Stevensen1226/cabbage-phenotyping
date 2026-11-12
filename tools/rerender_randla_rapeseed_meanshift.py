#!/usr/bin/env python3
"""Re-render rapeseed examples with watershed and larger seed distances."""
from __future__ import annotations
import argparse, copy, sys
from pathlib import Path
import numpy as np, yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
ROOT=Path(r'E:\Cabbage'); sys.path.insert(0,str(ROOT))
from cabbage_pheno.instance.clustering import InstanceClusterer
class Pcd:
    def __init__(self,p): self.points=np.asarray(p,dtype=np.float32)
def fill_foreground(labels,xyz,sem,radius=.05,k=5):
    out=labels.copy(); fg=sem==1
    for _ in range(50):
        noise=np.where(fg&(out<0))[0]; valid=np.where(fg&(out>=0))[0]
        if len(noise)==0 or len(valid)==0: break
        tree=cKDTree(xyz[valid]); kk=min(k,len(valid)); d,nn=tree.query(xyz[noise],k=kk,distance_upper_bound=radius,workers=-1)
        if kk==1: d=d[:,None]; nn=nn[:,None]
        idx=[]; lab=[]
        for j in range(len(noise)):
            ok=np.isfinite(d[j])
            if not np.any(ok): continue
            vals,cnt=np.unique(out[valid[nn[j][ok]]],return_counts=True); idx.append(noise[j]); lab.append(vals[np.argmax(cnt)])
        if not idx: break
        out[np.asarray(idx)]=np.asarray(lab)
    noise=np.where(fg&(out<0))[0]; valid=np.where(fg&(out>=0))[0]
    if len(noise) and len(valid):
        tree=cKDTree(xyz[valid]); _,nn=tree.query(xyz[noise],k=1,workers=-1); out[noise]=out[valid][nn]
    return out
def ms_cfg(dpvis):
    c=yaml.safe_load(open(ROOT/'configs/randla_watershed_dpvis_tuned.yaml',encoding='utf-8'))
    i=c['instance']; i['method']='meanshift'
    i['meanshift']={'bandwidth':0.32,'bin_seeding':True,'min_bin_freq':10}
    i['min_cluster_points']=20; i['max_cluster_points']=10_000_000; i['noise_reassign_radius']=0.03
    i['pca_split']['enable']=bool(dpvis); i['fragment_voting']['enable']=bool(dpvis)
    i['pca_split'].update({'horizontal_split_only':True,'min_peak_dist_m':0.30,'valley_depth_rel':0.85,
                          'abnormal_diameter':0.59,'abnormal_aspect_ratio':1.7,
                          'dynamic_min_peak_dist':{'enable':False},'vertical_merge':{'enable':False}})
    i['fragment_voting'].update({'discard_threshold':50,'merge_threshold':400,'fragment_merge_max_dist':0.25})
    return c
def run_ms(xyz,sem):
    fg=sem==1; out=[]
    for dpvis in (False,True):
        lab,_=InstanceClusterer(ms_cfg(dpvis)).cluster(Pcd(xyz[fg]))
        full=np.full(len(xyz),-1,dtype=np.int64); full[np.flatnonzero(fg)]=lab
        out.append(fill_foreground(full,xyz,sem))
    return out[0],out[1]
def render(rows,out,seed=123,scale=1.0,dpi=300):
    def gt_palette(gt):
        vals=np.unique(gt[gt>=0]); cols=plt.get_cmap('turbo')(np.linspace(.04,.96,max(1,len(vals)))); return {int(v):cols[i] for i,v in enumerate(vals)}
    def gt_colors(vals,pal):
        c=np.zeros((len(vals),4)); c[:,:3]=.82; c[:,3]=1
        for g,col in pal.items(): c[vals==g]=col
        return c
    def semcolors(v):
        c=np.zeros((len(v),4)); c[:,:3]=.82; c[:,3]=1; c[v>0]=[.08,.72,.25,1]; return c
    def label_colors(vals,max_label):
        c=np.zeros((len(vals),4)); c[:,:3]=.82; c[:,3]=1; cmap=plt.get_cmap('turbo')
        for v in np.unique(vals):
            if v>=0: c[vals==v]=cmap((float(v)+1.0)/(max_label+1.0))
        return c
    def style(ax,p,title):
        lo=np.percentile(p,1,axis=0); hi=np.percentile(p,99,axis=0)
        ax.set_xlim(lo[0],hi[0]); ax.set_ylim(lo[1],hi[1]); ax.set_zlim(lo[2],hi[2]); ax.set_box_aspect(hi-lo)
        ax.view_init(24,-62); ax.set_proj_type('ortho'); ax.set_axis_off(); ax.set_title(title,fontsize=24*scale,pad=6)
    fig=plt.figure(figsize=(25*scale,5.4*len(rows)*scale),dpi=dpi)
    for r,(label,xyz,gt,rgb,sem,base,dp) in enumerate(rows):
        pal=gt_palette(gt); max_label=max(int(base.max()),int(dp.max()),1); rng=np.random.default_rng(seed); sel=rng.choice(len(xyz),size=min(70000,len(xyz)),replace=False)
        panels=[('Original RGB',rgb),('GT instances',gt_colors(gt,pal)),('RandLA-Net semantic',semcolors(sem)),('RandLA-Net +\nMeanShift',label_colors(base,max_label)),('RandLA-Net +\nMeanShift + DPVIS',label_colors(dp,max_label))]
        for c,(title,col) in enumerate(panels):
            ax=fig.add_subplot(len(rows),5,r*5+c+1,projection='3d'); ax.scatter(xyz[sel,0],xyz[sel,1],xyz[sel,2],c=col[sel],s=max(.18,.35*scale),alpha=1.0,depthshade=False,rasterized=True); style(ax,xyz[sel],title)
            if c==0: ax.text2D(-.04,.5,str(label),transform=ax.transAxes,rotation=90,va='center',ha='right',fontsize=20*scale)
    fig.tight_layout(); out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=dpi,bbox_inches='tight',facecolor='white'); plt.close(fig); print('saved',out,flush=True)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,default=123); ap.add_argument('--scale',type=float,default=1.0); ap.add_argument('--dpi',type=int,default=300); ap.add_argument('--out',type=Path,default=None); a=ap.parse_args()
    if a.out is None: a.out=ROOT/f'output/crops3d_generalization/figures/randla_rapeseed_examples_meanshift{a.seed}.png'
    z=np.load(ROOT/'server_backup/autodl-tmp/logs/randla_examples_data.npz',allow_pickle=True); rows=[]
    for i in (0,1):
        xyz=z[f'r{i}_xyz']; gt=z[f'r{i}_gt']; rgb=z[f'r{i}_rgb']; sem=z[f'r{i}_sem']; base,dp=run_ms(xyz,sem); rows.append((z[f'r{i}_label'],xyz,gt,rgb,sem,base,dp)); print('row',i,'base_n',int(base.max()+1),'dp_n',int(dp.max()+1),flush=True)
    render(rows,a.out,a.seed,a.scale,a.dpi)
if __name__=='__main__': main()
