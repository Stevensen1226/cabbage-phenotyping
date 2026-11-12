#!/usr/bin/env python3
"""Full-resolution title-free subplots reconstructed from raw Crops3D PLY/GT tiles."""
from __future__ import annotations
import csv,re,sys,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
ROOT=Path(r'E:\Cabbage');sys.path.insert(0,str(ROOT))
from tools.crops3d_whole_common import load_gt
LOG=ROOT/'server_backup/autodl-tmp/logs';OUT=ROOT/'output/crops3d_generalization/figures/split_subplots_fullres';OUT.mkdir(parents=True,exist_ok=True)
def san(s):
 s=str(s).strip().replace('\n','_');s=re.sub(r'[^0-9A-Za-z._-]+','_',s);return re.sub(r'_+','_',s).strip('_')
def tile_indices(xyzc,k):
 lo=xyzc[:,:2].min(0);hi=xyzc[:,:2].max(0);kk=0
 for x0 in np.arange(lo[0],hi[0],1.0):
  for y0 in np.arange(lo[1],hi[1],1.0):
   m=(xyzc[:,0]>=x0)&(xyzc[:,0]<x0+2)&(xyzc[:,1]>=y0)&(xyzc[:,1]<y0+2)
   if int(m.sum())<2000:continue
   if kk==k:return np.flatnonzero(m)
   kk+=1
 raise ValueError(k)
def load_tile(plot,k):
 xyz,rgb,_,inst=load_gt(plot);xyzc=xyz-xyz.mean(0);idx=tile_indices(xyzc,k);q=xyzc[idx].copy();q-=q.mean(0,keepdims=True);return q,rgb[idx],inst[idx]
def transfer(q,X,vals):
 _,nn=cKDTree(X).query(q,k=1,workers=-1);return vals[nn]
def gt_pal(g):
 vals=np.unique(g[g>=0]);cols=plt.get_cmap('turbo')(np.linspace(.04,.96,max(1,len(vals))));return {int(v):cols[i] for i,v in enumerate(vals)}
def gt_col(g,pal):
 c=np.zeros((len(g),4));c[:,:3]=.82;c[:,3]=1
 for v,col in pal.items():c[g==v]=col
 return c
def match(p,g):
 m=g>=0;pp=p[m];gg=g[m];pairs,ct=np.unique(np.stack([pp,gg],1),axis=0,return_counts=True);inter={}
 for (a,b),k in zip(pairs,ct):
  a=int(a);b=int(b);k=int(k)
  if a>=0:inter[(a,b)]=k
 best={}
 for (a,b),k in inter.items():
  if k>best.get(a,(-1,None))[0]:best[a]=(k,b)
 return {a:b for a,(k,b) in best.items()}
def pred_col(p,g,pal):
 c=np.zeros((len(p),4));c[:,:3]=.82;c[:,3]=1;mp=match(p,g)
 for q in np.unique(p):
  q=int(q)
  if q>=0:c[p==q]=pal.get(mp.get(q,-1),np.array([.25,.25,.25,1.]))
 return c
def sem_col(v):
 c=np.zeros((len(v),4));c[:,:3]=.82;c[:,3]=1;c[v>0]=[.08,.72,.25,1];return c
def render(path,xyz,col,size=.04):
 fig=plt.figure(figsize=(5.2,5.2),dpi=400);ax=fig.add_subplot(111,projection='3d');ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c=col,s=size,alpha=1.0,depthshade=False,rasterized=True);lo=np.percentile(xyz,1,axis=0);hi=np.percentile(xyz,99,axis=0);ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_zlim(lo[2],hi[2]);ax.set_box_aspect(hi-lo);ax.view_init(24,-62);ax.set_proj_type('ortho');ax.set_axis_off();fig.tight_layout(pad=0);path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=400,bbox_inches='tight',pad_inches=.02,facecolor='white');plt.close(fig)
def write_panel(outcrop,outmethod,scene,panel,xyz,col,man,size=.04):
 path=OUT/outcrop/outmethod/f'{san(scene)}__{panel}.png';render(path,xyz,col,size);man.append({'crop':outcrop,'method':outmethod,'scene':scene,'panel':panel,'path':str(path.relative_to(ROOT))})
def randla(man):
    z=np.load(LOG/'randla_examples_data.npz',allow_pickle=True)
    pgm=np.load(LOG/'pg_maize_tall_screen.npz',allow_pickle=True)
    specs=[('r','rapeseed',[('rapeseed-4_t001',1,'r0',None),('rapeseed-4_t000',0,'r1',None)]),('m','maize',[('maize-12_t001',1,'m0',4),('maize-14_t001',1,'m1',2)])]
    for prefix,crop,rows in specs:
        for scene,k,key,lbl in rows:
            q,rgb,gt=load_tile(scene.rsplit('_',1)[0],k)
            scene_out=scene
            if lbl is not None:
                nat=transfer(q,pgm[f'{scene}_xyz'],pgm[f'{scene}_native']);mask=nat==lbl;q=q[mask];rgb=rgb[mask];gt=gt[mask];scene_out=f'{scene}_native_cluster_{lbl}'
            X=z[f'{key}_xyz'];sem=transfer(q,X,z[f'{key}_sem']);base=transfer(q,X,z[f'{key}_base']);dp=transfer(q,X,z[f'{key}_dp']);pal=gt_pal(gt)
            write_panel(crop,'randlanet',scene_out,'original_rgb',q,rgb,man,size=.038)
            write_panel(crop,'randlanet',scene_out,'gt_instances',q,gt_col(gt,pal),man,size=.045)
            write_panel(crop,'randlanet',scene_out,'randlanet_semantic',q,sem_col(sem),man,size=.045)
            write_panel(crop,'randlanet',scene_out,'randlanet_watershed',q,pred_col(base,gt,pal),man,size=.05)
            write_panel(crop,'randlanet',scene_out,'randlanet_watershed_dpvis',q,pred_col(dp,gt,pal),man,size=.05)
def pointgroup_maize(man):
 z=np.load(LOG/'pg_maize_tall_screen.npz',allow_pickle=True)
 for scene,lbl,k in [('maize-12_t001',4,1),('maize-14_t001',2,1)]:
  q,rgb,gt=load_tile(scene.rsplit('_',1)[0],k);X=z[f'{scene}_xyz'];nat=transfer(q,X,z[f'{scene}_native']);hyb=transfer(q,X,z[f'{scene}_hybrid']);sem=transfer(q,X,z[f'{scene}_sem']);m=nat==lbl;q=q[m];rgb=rgb[m];gt=gt[m];nat=nat[m];hyb=hyb[m];sem=sem[m];pal=gt_pal(gt);scene2=f'{scene}_native_cluster_{lbl}'
  write_panel('maize','pointgroup',scene2,'original_rgb',q,rgb,man,size=.04)
  write_panel('maize','pointgroup',scene2,'gt_instances',q,gt_col(gt,pal),man,size=.05)
  write_panel('maize','pointgroup',scene2,'pointgroup_semantic',q,sem_col(sem),man,size=.05)
  write_panel('maize','pointgroup',scene2,'pointgroup_native',q,pred_col(nat,gt,pal),man,size=.05)
  write_panel('maize','pointgroup',scene2,'pointgroup_hybrid_dpvis',q,pred_col(hyb,gt,pal),man,size=.05)
def pointgroup_rapeseed(man):
 z=np.load(LOG/'pg_rapeseed_hybrid_vis_rapsplit_058.npz',allow_pickle=True)
 for scene,k in [('rapeseed-4_t001',1),('rapeseed-4_t000',0)]:
  q,rgb,gt=load_tile('rapeseed-4',k);X=z[f'{scene}_xyz'];nat=transfer(q,X,z[f'{scene}_native']);hyb=transfer(q,X,z[f'{scene}_hybrid']);sem=transfer(q,X,z[f'{scene}_sem']);pal=gt_pal(gt)
  write_panel('rapeseed','pointgroup',scene,'original_rgb',q,rgb,man,size=.025)
  write_panel('rapeseed','pointgroup',scene,'gt_instances',q,gt_col(gt,pal),man,size=.035)
  write_panel('rapeseed','pointgroup',scene,'pointgroup_semantic',q,sem_col(sem),man,size=.035)
  write_panel('rapeseed','pointgroup',scene,'pointgroup_native',q,pred_col(nat,gt,pal),man,size=.04)
  write_panel('rapeseed','pointgroup',scene,'pointgroup_hybrid_dpvis',q,pred_col(hyb,gt,pal),man,size=.04)
def main():
 global vman
 vman=[];man=[];randla(man);pointgroup_maize(man);pointgroup_rapeseed(man)
 with (OUT/'manifest.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=['crop','method','scene','panel','path']);w.writeheader();w.writerows(man)
 print('saved',len(man),'full-resolution subplots to',OUT)
if __name__=='__main__':main()
