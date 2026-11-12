import numpy as np, matplotlib.pyplot as plt
from matplotlib import cm
z=np.load('/root/autodl-tmp/logs/pg_rapeseed_hybrid_vis.npz')
cases=[('rapeseed-4_t001',4),('rapeseed-4_t000',8)]
def palette_for(gt):
    labs=np.unique(gt[gt>=0]); pal=cm.turbo(np.linspace(.04,.96,max(1,len(labs)))); return {int(g):pal[i] for i,g in enumerate(labs)}
def colors_for_gt(vals,pal):
    c=np.zeros((len(vals),4)); c[:,:3]=.82; c[:,3]=1
    for g,col in pal.items(): c[vals==g]=col
    return c
def colors_for_hybrid(hyb,gt,pal):
    c=np.zeros((len(hyb),4)); c[:,:3]=.25; c[:,3]=1
    for q in np.unique(hyb):
        if q<0:continue
        m=hyb==q; u,cc=np.unique(gt[m],return_counts=True); valid=[(int(a),int(b)) for a,b in zip(u,cc) if a>=0]
        if valid:
            g=max(valid,key=lambda x:x[1])[0]
            if g in pal:c[m]=pal[g]
    return c
def style(ax,xyz,t):
    lo=np.percentile(xyz,1,axis=0);hi=np.percentile(xyz,99,axis=0);ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_zlim(lo[2],hi[2]);ax.set_box_aspect(hi-lo);ax.view_init(22,-58);ax.set_proj_type('ortho');ax.set_axis_off();ax.set_title(t,fontsize=10,pad=2)
fig=plt.figure(figsize=(15,5.2*len(cases)),dpi=300)
for r,(name,nat_id) in enumerate(cases):
    xyz=z[f'{name}_xyz'];gt=z[f'{name}_gt'];hyb=z[f'{name}_hybrid'];nat=z[f'{name}_native'];m=nat==nat_id;pts=xyz[m];g=gt[m];h=hyb[m];pal=palette_for(g)
    # recompute local hybrid components for labels
    panels=[('GT instances\n2 plants',colors_for_gt(g,pal)),('PointGroup native\n1 merged instance',np.tile([.12,.28,.72,1.0],(len(pts),1))),('PointGroup + hybrid DPVIS\nsplit components',colors_for_hybrid(h,g,pal))]
    for c,(title,col) in enumerate(panels):
        ax=fig.add_subplot(len(cases),3,r*3+c+1,projection='3d');ax.scatter(pts[:,0],pts[:,1],pts[:,2],c=col,s=.35,alpha=.88,depthshade=False,rasterized=True);style(ax,pts,title)
        if c==0:ax.text2D(-.04,.5,f'{name}\nnative label {nat_id}',transform=ax.transAxes,rotation=90,va='center',ha='right',fontsize=9)
fig.suptitle('DPVIS adhesion splitting: merged PointGroup instances split into GT-consistent components',fontsize=12,y=1.0);fig.tight_layout();out='/root/autodl-tmp/logs/pg_rapeseed_adhesion_splits.png';fig.savefig(out,dpi=300,bbox_inches='tight',facecolor='white');print('saved',out)
