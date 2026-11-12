import numpy as np, matplotlib.pyplot as plt
from matplotlib import cm
z=np.load('/root/autodl-tmp/logs/pg_rapeseed_hybrid_vis.npz')
def gt_palette(gt):
    labels=np.unique(gt[gt>=0]); pal=cm.turbo(np.linspace(.04,.96,max(1,len(labels)))); return {int(g):pal[i] for i,g in enumerate(labels)}
def gt_colors(vals,palette):
    c=np.zeros((len(vals),4)); c[:,:3]=.82; c[:,3]=1
    for g,col in palette.items(): c[vals==g]=col
    return c
def match_pred_to_gt(pred,gt):
    # Visualization-only mapping: assign every predicted fragment to the GT
    # instance with which it has the largest point overlap. This keeps all
    # over-segmented fragments of the same GT plant in the same color.
    m=gt>=0; p=pred[m]; g=gt[m]; pairs,c=np.unique(np.stack([p,g],1),axis=0,return_counts=True); inter={}
    for (a,b),k in zip(pairs,c):
        a=int(a); b=int(b); k=int(k)
        if a<0: continue
        inter[(a,b)]=k
    mapping={}; best={}
    for (a,b),k in inter.items():
        if k>best.get(a,(-1,None))[0]: best[a]=(k,b)
    for a,(k,b) in best.items(): mapping[a]=b
    return mapping
def pred_colors(pred,gt,palette):
    c=np.zeros((len(pred),4)); c[:,:3]=.82; c[:,3]=1; mapping=match_pred_to_gt(pred,gt)
    for p in np.unique(pred):
        p=int(p)
        if p<0: continue
        c[pred==p]=palette.get(mapping.get(p,-1),np.array([.25,.25,.25,1.0]))
    return c
def semcolors(v):
    c=np.zeros((len(v),4)); c[:,:3]=.82; c[:,3]=1; c[v>0]=[.08,.72,.25,1]; return c
def met(pred,gt):
    m=gt>=0; p=pred[m]; g=gt[m]; pairs,c=np.unique(np.stack([p,g],1),axis=0,return_counts=True); pt={}; gtt={}; inter={}
    for (a,b),k in zip(pairs,c):
        a=int(a); b=int(b); k=int(k); gtt[b]=gtt.get(b,0)+k
        if a<0: continue
        pt[a]=pt.get(a,0)+k; inter[(a,b)]=k
    cands=[]
    for (a,b),k in inter.items():
        iou=k/(pt[a]+gtt[b]-k)
        if iou>=.5: cands.append((iou,a,b))
    cands.sort(reverse=True); ma=set(); mb=set()
    for _,a,b in cands:
        if a in ma or b in mb: continue
        ma.add(a); mb.add(b)
    tp=len(ma); npr=len(pt); ngt=len(gtt); P=tp/npr if npr else 0; R=tp/ngt if ngt else 0; F=2*P*R/(P+R) if P+R else 0
    return F,abs(npr-ngt)
def style(ax,xyz,t):
    lo=np.percentile(xyz,1,axis=0); hi=np.percentile(xyz,99,axis=0); ax.set_xlim(lo[0],hi[0]); ax.set_ylim(lo[1],hi[1]); ax.set_zlim(lo[2],hi[2]); ax.set_box_aspect(hi-lo); ax.view_init(24,-62); ax.set_proj_type('ortho'); ax.set_axis_off(); ax.set_title(t,fontsize=10,pad=2)
names=['rapeseed-4_t001','rapeseed-4_t000']; fig=plt.figure(figsize=(20,5.2*len(names)),dpi=300)
for r,n in enumerate(names):
    xyz=z[f'{n}_xyz']; gt=z[f'{n}_gt']; sem=z[f'{n}_sem']; nat=z[f'{n}_native']; hyb=z[f'{n}_hybrid']; mn=met(nat,gt); mh=met(hyb,gt); pal=gt_palette(gt); rng=np.random.default_rng(0); sel=rng.choice(len(xyz),size=min(70000,len(xyz)),replace=False)
    panels=[('GT instances',lambda:gt_colors(gt,pal)),('PointGroup semantic',lambda:semcolors(sem)),(f'PointGroup native\nF1={mn[0]:.3f} MAE={mn[1]}',lambda:pred_colors(nat,gt,pal)),(f'PointGroup hybrid + DPVIS\nF1={mh[0]:.3f} MAE={mh[1]}',lambda:pred_colors(hyb,gt,pal))]
    for c,(t,color_fn) in enumerate(panels):
        ax=fig.add_subplot(len(names),4,r*4+c+1,projection='3d'); col=color_fn(); ax.scatter(xyz[sel,0],xyz[sel,1],xyz[sel,2],c=col[sel],s=.22,alpha=.82,depthshade=False,rasterized=True); style(ax,xyz[sel],t)
        if c==0: ax.text2D(-.04,.5,n,transform=ax.transAxes,rotation=90,va='center',ha='right',fontsize=9)
fig.suptitle('PointGroup native vs hybrid + improved DPVIS on rapeseed',fontsize=13,y=1.01); fig.tight_layout(); out='/root/autodl-tmp/logs/pg_rapeseed_panoptic_fullscene.png'; fig.savefig(out,dpi=300,bbox_inches='tight',facecolor='white'); print('saved',out)
