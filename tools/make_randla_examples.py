#!/usr/bin/env python3
import sys, copy
from pathlib import Path
import numpy as np, torch, yaml, matplotlib.pyplot as plt
from scipy.spatial import cKDTree
ROOT=Path('/root/autodl-tmp/cabbage-phenotyping'); sys.path.insert(0,str(ROOT))
from cabbage_pheno.instance.clustering_maize import InstanceClusterer
from cabbage_pheno.instance.adaptive import cluster_with_adaptive_seed
class Pcd:
    def __init__(self,p): self.points=p
def tile_indices(xyzc,k):
    lo=xyzc[:,:2].min(0); hi=xyzc[:,:2].max(0); kk=0
    for x0 in np.arange(lo[0],hi[0],1.0):
        for y0 in np.arange(lo[1],hi[1],1.0):
            m=(xyzc[:,0]>=x0)&(xyzc[:,0]<x0+2.0)&(xyzc[:,1]>=y0)&(xyzc[:,1]<y0+2.0)
            if int(m.sum())<2000: continue
            if kk==k: return np.flatnonzero(m)
            kk+=1
    raise ValueError(f'tile index {k} not found')
def load_plot(plot):
    pg=torch.load(f'/root/autodl-tmp/data/external/crops3d_pointgroup/test/{plot}.pth',map_location='cpu',weights_only=False)
    rl=torch.load(f'/root/autodl-tmp/data/external/crops3d_randla/test/{plot}.pth',map_location='cpu',weights_only=False)
    xyzc=np.asarray(pg[0],np.float32); inst=np.asarray(pg[3],np.int64)
    xyzr=np.asarray(rl[0],np.float32); rgb=np.asarray(rl[1],np.float32); pred=np.load(f'/root/autodl-tmp/preds/crops3d_randla/{plot}_pred.npy').astype(np.int64)
    if len(xyzc)!=len(xyzr) or len(xyzc)!=len(pred): raise RuntimeError('length mismatch')
    if np.abs((xyzr-xyzr.mean(0))-xyzc).max()>2e-4: raise RuntimeError('point order mismatch')
    return xyzc,inst,rgb,pred
def fill_foreground(labels,xyz,sem,radius=.05,k=5,max_passes=50):
    out=labels.copy(); fg=sem==1
    for _ in range(max_passes):
        noise=np.where(fg&(out<0))[0]; valid=np.where(fg&(out>=0))[0]
        if len(noise)==0 or len(valid)==0: break
        tree=cKDTree(xyz[valid]); kk=min(k,len(valid)); d,nn=tree.query(xyz[noise],k=kk,distance_upper_bound=radius,workers=-1)
        if kk==1: d=d[:,None]; nn=nn[:,None]
        idx=[]; lab=[]
        for j in range(len(noise)):
            ok=np.isfinite(d[j])
            if not np.any(ok): continue
            labs=out[valid[nn[j][ok]]]; vals,cnt=np.unique(labs,return_counts=True); idx.append(noise[j]); lab.append(vals[np.argmax(cnt)])
        if not idx: break
        out[np.asarray(idx)]=np.asarray(lab)
    noise=np.where(fg&(out<0))[0]; valid=np.where(fg&(out>=0))[0]
    if len(noise) and len(valid):
        tree=cKDTree(xyz[valid]); _,nn=tree.query(xyz[noise],k=1,workers=-1); out[noise]=out[valid][nn]
    return out
def ground_filter(xyz,pred,low_h=.03,high_h=.10,high_r=.08,k=50):
    xy=xyz[:,:2]; tree=cKDTree(xy); kk=min(k,len(xyz)); _,nn=tree.query(xy,k=kk,workers=-1)
    if kk==1: nn=nn[:,None]
    gz=np.percentile(xyz[nn,2],5,axis=1); h=xyz[:,2]-gz
    fg=pred==1; high=fg&(h>high_h); out=pred.copy()
    if high.sum()>0:
        dh,_=cKDTree(xyz[high]).query(xyz,k=1,workers=-1)
        keep=fg&((h>low_h)|(dh<high_r))
    else:
        keep=fg&(h>low_h)
    out[~keep]=0
    return out,int((fg&~keep).sum()),h
def run_randla(xyz,gt,rgb,sem_pred,seed_mode,seed_value=None):
    fg=sem_pred==1
    base=yaml.safe_load((ROOT/'configs/randla_watershed_dpvis_tuned.yaml').read_text())
    base['instance']['pca_split']['enable']=False; base['instance']['fragment_voting']['enable']=False
    if seed_mode=='adaptive':
        lab,_,rec=cluster_with_adaptive_seed(InstanceClusterer,base,Pcd(xyz[fg]),candidates=(.20,.22,.25,.28,.30),max_unassigned_frac=.10); seed=rec['seed']
    else:
        seed=float(seed_value); base['instance']['watershed_3d']['min_seed_distance']=seed; lab,_=InstanceClusterer(base).cluster(Pcd(xyz[fg]))
    base_lab=np.full(len(xyz),-1,dtype=np.int64); base_lab[np.flatnonzero(fg)]=np.asarray(lab)
    dcfg=copy.deepcopy(base); dcfg['instance']['pca_split']['enable']=True; dcfg['instance']['fragment_voting']['enable']=True; dcfg['instance']['watershed_3d']['min_seed_distance']=seed
    dlab,_=InstanceClusterer(dcfg).cluster(Pcd(xyz[fg])); dp_lab=np.full(len(xyz),-1,dtype=np.int64); dp_lab[np.flatnonzero(fg)]=np.asarray(dlab)
    return fill_foreground(base_lab,xyz,sem_pred), fill_foreground(dp_lab,xyz,sem_pred), seed
def map_npz_to_plot(npz,name,plot,mask=None):
    xyzc,inst,rgb,pred=load_plot(plot); k=int(name.rsplit('_t',1)[1]); idx=tile_indices(xyzc,k); q=xyzc[idx].copy(); q-=q.mean(0,keepdims=True)
    xyz=npz[f'{name}_xyz']; gt=npz[f'{name}_gt']; d,nn=cKDTree(q).query(xyz,k=1,workers=-1); print(name,'mapdist',float(d.max()),flush=True)
    use=np.arange(len(xyz)) if mask is None else np.flatnonzero(mask); j=idx[nn]; return xyz[use],gt[use],np.clip((rgb[j[use]]+1.0)*0.5,0,1),pred[j[use]]
def render(rows,out):
    def gt_palette(gt):
        ls=np.unique(gt[gt>=0]); p=plt.get_cmap('turbo')(np.linspace(.04,.96,max(1,len(ls)))); return {int(g):p[i] for i,g in enumerate(ls)}
    def gt_colors(vals,pal):
        c=np.zeros((len(vals),4)); c[:,:3]=.82; c[:,3]=1
        for g,col in pal.items(): c[vals==g]=col
        return c
    def match(pred,gt):
        m=gt>=0; p=pred[m]; g=gt[m]; pairs,ct=np.unique(np.stack([p,g],1),axis=0,return_counts=True); inter={}
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
        c=np.zeros((len(v),4)); c[:,:3]=.82; c[:,3]=1; c[v>0]=[.08,.72,.25,1]; return c
    def style(ax,p,t):
        lo=np.percentile(p,1,axis=0); hi=np.percentile(p,99,axis=0); ax.set_xlim(lo[0],hi[0]); ax.set_ylim(lo[1],hi[1]); ax.set_zlim(lo[2],hi[2]); ax.set_box_aspect(hi-lo); ax.view_init(24,-62); ax.set_proj_type('ortho'); ax.set_axis_off(); ax.set_title(t,fontsize=24,pad=6)
    fig=plt.figure(figsize=(25,5.4*len(rows)),dpi=300)
    for r,(label,xyz,gt,rgb,sem,base,dp) in enumerate(rows):
        pal=gt_palette(gt); rng=np.random.default_rng(0); sel=rng.choice(len(xyz),size=min(70000,len(xyz)),replace=False)
        panels=[('Original RGB',rgb),('GT instances',gt_colors(gt,pal)),('RandLA-Net semantic',semcolors(sem)),('RandLA-Net +\nwatershed',pred_colors(base,gt,pal)),('RandLA-Net +\nwatershed + DPVIS',pred_colors(dp,gt,pal))]
        for c,(title,col) in enumerate(panels):
            ax=fig.add_subplot(len(rows),5,r*5+c+1,projection='3d'); ax.scatter(xyz[sel,0],xyz[sel,1],xyz[sel,2],c=col[sel],s=.35,alpha=1.0,depthshade=False,rasterized=True); style(ax,xyz[sel],title)
            if c==0: ax.text2D(-.04,.5,label,transform=ax.transAxes,rotation=90,va='center',ha='right',fontsize=20)
    fig.tight_layout(); fig.savefig(out,dpi=300,bbox_inches='tight',facecolor='white'); plt.close(fig); print('saved',out,flush=True)
rz=np.load('/root/autodl-tmp/logs/pg_rapeseed_hybrid_vis_rapsplit_058.npz'); rrows=[]
for name,plot in [('rapeseed-4_t001','rapeseed-4'),('rapeseed-4_t000','rapeseed-4')]:
    xyz,gt,rgb,pred0=map_npz_to_plot(rz,name,plot,None); pred,nrem,_=ground_filter(xyz,pred0); print(name,'ground_removed',nrem,flush=True);
    if name=='rapeseed-4_t000': base,dp,seed=run_randla(xyz,gt,rgb,pred,'fixed',0.22)
    else: base,dp,seed=run_randla(xyz,gt,rgb,pred,'adaptive')
    rrows.append((name,xyz,gt,rgb,pred,base,dp)); print(name,'seed',seed,flush=True)
render(rrows,'/root/autodl-tmp/logs/randla_rapeseed_examples.png')
mz=np.load('/root/autodl-tmp/logs/pg_maize_tall_screen.npz'); mrows=[]
for name,plot,lab,seed in [('maize-12_t001','maize-12',4,.28),('maize-14_t001','maize-14',2,.22)]:
    mask=mz[f'{name}_native']==lab; xyz,gt,rgb,pred0=map_npz_to_plot(mz,name,plot,mask); pred,nrem,_=ground_filter(xyz,pred0); print(name,'ground_removed',nrem,flush=True); base,dp,used=run_randla(xyz,gt,rgb,pred,'fixed',seed); mrows.append((f'{name}\nnative cluster {lab}',xyz,gt,rgb,pred,base,dp)); print(name,'seed',used,flush=True)
render(mrows,'/root/autodl-tmp/logs/randla_maize_examples.png')




np.savez_compressed('/root/autodl-tmp/logs/randla_examples_data.npz', **{f'r{i}_{k}':v for i,row in enumerate(rrows) for k,v in zip(['label','xyz','gt','rgb','sem','base','dp'],row)}, **{f'm{i}_{k}':v for i,row in enumerate(mrows) for k,v in zip(['label','xyz','gt','rgb','sem','base','dp'],row)})



