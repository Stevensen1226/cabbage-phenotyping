import numpy as np, matplotlib.pyplot as plt
from matplotlib import cm
z=np.load('/root/autodl-tmp/logs/pg_maize_rgb_vis.npz')

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

names=['maize-11_t005','maize-12_t002','maize-14_t003']
fig=plt.figure(figsize=(25,5.2*len(names)),dpi=300)
for r,n in enumerate(names):
    xyz=z[f'{n}_xyz']; rgb=np.clip((z[f'{n}_rgb']+1.0)*0.5,0,1); gt=z[f'{n}_gt']; sem=z[f'{n}_sem']; nat=z[f'{n}_native']; hyb=z[f'{n}_hybrid']; pal=gt_palette(gt)
    rng=np.random.default_rng(0); sel=rng.choice(len(xyz),size=min(70000,len(xyz)),replace=False)
    panels=[('Original RGB',rgb),('GT instances',gt_colors(gt,pal)),('PointGroup semantic',semcolors(sem)),('PointGroup native',pred_colors(nat,gt,pal)),('PointGroup hybrid + DPVIS',pred_colors(hyb,gt,pal))]
    for c,(t,col) in enumerate(panels):
        ax=fig.add_subplot(len(names),5,r*5+c+1,projection='3d'); scatter(ax,xyz,sel,col,t)
        if c==0: ax.text2D(-.04,.5,n,transform=ax.transAxes,rotation=90,va='center',ha='right',fontsize=9)
fig.tight_layout(); out='/root/autodl-tmp/logs/pg_maize_rgb_fullscene.png'; fig.savefig(out,dpi=300,bbox_inches='tight',facecolor='white'); print('saved',out)

