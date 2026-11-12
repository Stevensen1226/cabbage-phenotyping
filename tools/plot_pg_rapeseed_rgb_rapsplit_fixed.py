import numpy as np, torch, matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial import cKDTree
from scipy.spatial import cKDTree

PG_DIR='/root/autodl-tmp/data/external/crops3d_pg_tiles/test'
OLD='/root/autodl-tmp/logs/pg_rapeseed_hybrid_vis.npz'
NEW='/root/autodl-tmp/logs/pg_rapeseed_hybrid_vis_rapsplit_058.npz'
zold=np.load(OLD); znew=np.load(NEW)

def load_rgb_for_npz(name, xyz_npz):
    d=torch.load(f'{PG_DIR}/{name}.pth', map_location='cpu', weights_only=False)
    xyz=np.asarray(d[0], np.float32); rgb=(np.asarray(d[1], np.float32)+1.0)*0.5
    tree=cKDTree(xyz)
    dist,idx=tree.query(xyz_npz,k=1,workers=-1)
    print(name,'rgb match maxdist',float(dist.max()),'mean',float(dist.mean()),flush=True)
    return np.clip(rgb[idx],0,1)

def fill_foreground(labels,xyz,sem,radius=0.05,k=5,max_passes=50):
    out=labels.copy(); fg=sem==1
    for _ in range(max_passes):
        noise=np.where(fg & (out<0))[0]; valid=np.where(fg & (out>=0))[0]
        if len(noise)==0 or len(valid)==0: break
        tree=cKDTree(xyz[valid]); kk=min(k,len(valid))
        d,nn=tree.query(xyz[noise],k=kk,distance_upper_bound=radius,workers=-1)
        if kk==1: d=d[:,None]; nn=nn[:,None]
        idx=[]; lab=[]
        for j in range(len(noise)):
            ok=np.isfinite(d[j])
            if not np.any(ok): continue
            labs=out[valid[nn[j][ok]]]; vals,cnt=np.unique(labs,return_counts=True)
            idx.append(noise[j]); lab.append(vals[np.argmax(cnt)])
        if not idx: break
        out[np.asarray(idx)]=np.asarray(lab)
    noise=np.where(fg & (out<0))[0]; valid=np.where(fg & (out>=0))[0]
    if len(noise) and len(valid):
        tree=cKDTree(xyz[valid]); _,nn=tree.query(xyz[noise],k=1,workers=-1); out[noise]=out[valid][nn]
    return out
def gt_palette(gt):
    labels=np.unique(gt[gt>=0]); pal=cm.turbo(np.linspace(.04,.96,max(1,len(labels)))); return {int(g):pal[i] for i,g in enumerate(labels)}
def gt_colors(vals,palette):
    c=np.zeros((len(vals),4)); c[:,:3]=.82; c[:,3]=1
    for g,col in palette.items(): c[vals==g]=col
    return c
def match_pred_to_gt(pred,gt):
    m=gt>=0; p=pred[m]; g=gt[m]; pairs,c=np.unique(np.stack([p,g],1),axis=0,return_counts=True); inter={}
    for (a,b),k in zip(pairs,c):
        a=int(a);b=int(b);k=int(k)
        if a<0:continue
        inter[(a,b)]=k
    best={}
    for (a,b),k in inter.items():
        if k>best.get(a,(-1,None))[0]:best[a]=(k,b)
    return {a:b for a,(k,b) in best.items()}
def pred_colors(pred,gt,palette):
    c=np.zeros((len(pred),4)); c[:,:3]=.82; c[:,3]=1; mp=match_pred_to_gt(pred,gt)
    for p in np.unique(pred):
        p=int(p)
        if p<0:continue
        c[pred==p]=palette.get(mp.get(p,-1),np.array([.25,.25,.25,1.]))
    return c
def semcolors(v):
    c=np.zeros((len(v),4)); c[:,:3]=.82; c[:,3]=1; c[v>0]=[.08,.72,.25,1]; return c
def met(pred,gt):
    m=gt>=0;p=pred[m];g=gt[m];pairs,c=np.unique(np.stack([p,g],1),axis=0,return_counts=True);pt={};gtt={};inter={}
    for (a,b),k in zip(pairs,c):
        a=int(a);b=int(b);k=int(k);gtt[b]=gtt.get(b,0)+k
        if a<0:continue
        pt[a]=pt.get(a,0)+k;inter[(a,b)]=k
    cands=[]
    for (a,b),k in inter.items():
        iou=k/(pt[a]+gtt[b]-k)
        if iou>=.5:cands.append((iou,a,b))
    cands.sort(reverse=True);ma=set();mb=set()
    for _,a,b in cands:
        if a in ma or b in mb:continue
        ma.add(a);mb.add(b)
    tp=len(ma);npr=len(pt);ngt=len(gtt);P=tp/npr if npr else 0;R=tp/ngt if ngt else 0;F=2*P*R/(P+R) if P+R else 0
    return F,abs(npr-ngt)
def style(ax,xyz,t):
    lo=np.percentile(xyz,1,axis=0);hi=np.percentile(xyz,99,axis=0);ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_zlim(lo[2],hi[2]);ax.set_box_aspect(hi-lo);ax.view_init(24,-62);ax.set_proj_type('ortho');ax.set_axis_off();ax.set_title(t,fontsize=24,pad=2)
def sc(ax,xyz,sel,col,t,size=.22):
    ax.scatter(xyz[sel,0],xyz[sel,1],xyz[sel,2],c=col[sel],s=size,alpha=.82,depthshade=False,rasterized=True);style(ax,xyz[sel],t)

names=['rapeseed-4_t001','rapeseed-4_t000']
fig=plt.figure(figsize=(25,5.2*len(names)),dpi=300)
for r,n in enumerate(names):
    xyz=znew[f'{n}_xyz'];gt=znew[f'{n}_gt'];sem=znew[f'{n}_sem'];nat=znew[f'{n}_native'];hyb=znew[f'{n}_hybrid'];nat=fill_foreground(nat,xyz,sem);hyb=fill_foreground(hyb,xyz,sem);rgb=load_rgb_for_npz(n,xyz)
    mn=met(nat,gt);mh=met(hyb,gt);pal=gt_palette(gt);rng=np.random.default_rng(0);sel=rng.choice(len(xyz),size=min(70000,len(xyz)),replace=False)
    panels=[('Original RGB',rgb),('GT instances',gt_colors(gt,pal)),('PointGroup semantic',semcolors(sem)),('PointGroup native',pred_colors(nat,gt,pal)),('PointGroup hybrid + DPVIS',pred_colors(hyb,gt,pal))]
    for c,(t,col) in enumerate(panels):
        ax=fig.add_subplot(len(names),5,r*5+c+1,projection='3d');sc(ax,xyz,sel,col,t)
        if c==0:ax.text2D(-.04,.5,n,transform=ax.transAxes,rotation=90,va='center',ha='right',fontsize=20)
fig.tight_layout();out='/root/autodl-tmp/logs/pg_rapeseed_panoptic_rgb_rapsplit_fixed_fullscene.png';fig.savefig(out,dpi=300,bbox_inches='tight',facecolor='white');plt.close(fig);print('saved',out)

# zoom version for the first-row GT3+GT4 case
n='rapeseed-4_t001';xyz=znew[f'{n}_xyz'];gt=znew[f'{n}_gt'];rgb=load_rgb_for_npz(n,xyz);nat=znew[f'{n}_native'];oh=zold[f'{n}_hybrid'];nh=znew[f'{n}_hybrid'];nat=fill_foreground(nat,xyz,znew[f'{n}_sem']);oh=fill_foreground(oh,xyz,znew[f'{n}_sem']);nh=fill_foreground(nh,xyz,znew[f'{n}_sem']);pal=gt_palette(gt);mask=(gt==3)|(gt==4);p=xyz[mask]
oldcol=pred_colors(oh,gt,pal)[mask];newcol=pred_colors(nh,gt,pal)[mask];gtcol=gt_colors(gt,pal)[mask];rgbcol=rgb[mask]
fig=plt.figure(figsize=(18,5.2),dpi=300)
for i,(title,col) in enumerate([('Original RGB',rgbcol),('GT3 / GT4',gtcol),('before: one merged prediction',oldcol),('after',newcol)]):
    ax=fig.add_subplot(1,4,i+1,projection='3d');ax.scatter(p[:,0],p[:,1],p[:,2],c=col,s=.8,alpha=.95,depthshade=False,rasterized=True)
    lo=np.percentile(p,1,axis=0);hi=np.percentile(p,99,axis=0);ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_zlim(lo[2],hi[2]);ax.set_box_aspect(hi-lo);ax.view_init(20,-65);ax.set_proj_type('ortho');ax.set_axis_off();ax.set_title(title,fontsize=24)
fig.tight_layout();out='/root/autodl-tmp/logs/pg_rapeseed_t001_pair_zoom_rgb_fixed.png';fig.savefig(out,dpi=300,bbox_inches='tight',facecolor='white');plt.close(fig);print('saved',out)






