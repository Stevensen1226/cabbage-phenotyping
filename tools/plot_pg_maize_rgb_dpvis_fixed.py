import numpy as np, matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial import cKDTree
za=np.load('/root/autodl-tmp/logs/pg_maize_rgb_vis_dpvis.npz')
zb=np.load('/root/autodl-tmp/logs/pg_maize_rgb_vis_dpvis_candidates.npz')
def arr(n,k):
    return za[f'{n}_{k}'] if f'{n}_{k}' in za.files else zb[f'{n}_{k}']

def fill_foreground(labels,xyz,sem,radius=0.05,k=5,max_passes=50):
    out=labels.copy(); fg=sem==1
    before=int((fg & (out<0)).sum())
    for _ in range(max_passes):
        noise=np.where(fg & (out<0))[0]
        valid=np.where(fg & (out>=0))[0]
        if len(noise)==0 or len(valid)==0: break
        tree=cKDTree(xyz[valid]); kk=min(k,len(valid))
        d,nn=tree.query(xyz[noise],k=kk,distance_upper_bound=radius,workers=-1)
        if kk==1: d=d[:,None]; nn=nn[:,None]
        idx=[]; lab=[]
        for j in range(len(noise)):
            ok=np.isfinite(d[j])
            if not np.any(ok): continue
            labs=out[valid[nn[j][ok]]]
            vals,cnt=np.unique(labs,return_counts=True)
            idx.append(noise[j]); lab.append(vals[np.argmax(cnt)])
        if not idx: break
        out[np.asarray(idx)]=np.asarray(lab)
    # Any remaining semantic foreground is still plant; assign it to the nearest valid instance.
    noise=np.where(fg & (out<0))[0]; valid=np.where(fg & (out>=0))[0]
    if len(noise) and len(valid):
        tree=cKDTree(xyz[valid]); _,nn=tree.query(xyz[noise],k=1,workers=-1)
        out[noise]=out[valid][nn]
    after=int((fg & (out<0)).sum())
    print('foreground gray',before,'->',after,flush=True)
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
        a=int(a); b=int(b); k=int(k)
        if a<0: continue
        inter[(a,b)]=k
    best={}
    for (a,b),k in inter.items():
        if k>best.get(a,(-1,None))[0]: best[a]=(k,b)
    return {a:b for a,(k,b) in best.items()}
def pred_colors(pred,gt,palette):
    c=np.zeros((len(pred),4)); c[:,:3]=.82; c[:,3]=1; mp=match_pred_to_gt(pred,gt)
    for p in np.unique(pred):
        p=int(p)
        if p<0: continue
        c[pred==p]=palette.get(mp.get(p,-1),np.array([.25,.25,.25,1.]))
    return c
def semcolors(v):
    c=np.zeros((len(v),4)); c[:,:3]=.82; c[:,3]=1; c[v>0]=[.08,.72,.25,1]; return c
def style(ax,xyz,t):
    lo=np.percentile(xyz,1,axis=0); hi=np.percentile(xyz,99,axis=0); ax.set_xlim(lo[0],hi[0]); ax.set_ylim(lo[1],hi[1]); ax.set_zlim(lo[2],hi[2]); ax.set_box_aspect(hi-lo); ax.view_init(24,-62); ax.set_proj_type('ortho'); ax.set_axis_off(); ax.set_title(t,fontsize=10,pad=2)
def scatter(ax,xyz,sel,col,t):
    ax.scatter(xyz[sel,0],xyz[sel,1],xyz[sel,2],c=col[sel],s=.22,alpha=1.0,depthshade=False,rasterized=True); style(ax,xyz[sel],t)

names=['maize-9_t000','maize-9_t001','maize-9_t002']
fig=plt.figure(figsize=(25,5.2*len(names)),dpi=300)
fixed={}
for r,n in enumerate(names):
    xyz=arr(n,'xyz'); rgb=np.clip((arr(n,'rgb')+1.0)*0.5,0,1); gt=arr(n,'gt'); sem=arr(n,'sem')
    nat=fill_foreground(arr(n,'native'),xyz,sem); hyb=fill_foreground(arr(n,'hybrid'),xyz,sem)
    fixed[n]={'xyz':xyz,'rgb':rgb,'gt':gt,'sem':sem,'native':nat,'hybrid':hyb}
    pal=gt_palette(gt); rng=np.random.default_rng(0); sel=rng.choice(len(xyz),size=min(70000,len(xyz)),replace=False)
    panels=[('Original RGB',rgb),('GT instances',gt_colors(gt,pal)),('PointGroup semantic',semcolors(sem)),('PointGroup native',pred_colors(nat,gt,pal)),('PointGroup hybrid + DPVIS',pred_colors(hyb,gt,pal))]
    for c,(t,col) in enumerate(panels):
        ax=fig.add_subplot(len(names),5,r*5+c+1,projection='3d'); scatter(ax,xyz,sel,col,t)
        if c==0: ax.text2D(-.04,.5,n,transform=ax.transAxes,rotation=90,va='center',ha='right',fontsize=9)
fig.tight_layout(); out='/root/autodl-tmp/logs/pg_maize_rgb_dpvis_fixed_fullscene.png'; fig.savefig(out,dpi=300,bbox_inches='tight',facecolor='white'); print('saved',out)
np.savez_compressed('/root/autodl-tmp/logs/pg_maize_rgb_vis_fixed.npz',**{f'{n}_{k}':v for n,d in fixed.items() for k,v in d.items()})
