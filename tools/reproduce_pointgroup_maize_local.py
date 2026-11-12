import numpy as np, matplotlib.pyplot as plt
z=np.load('E:/Cabbage/server_backup/autodl-tmp/logs/pg_maize_tall_screen.npz')
rows=[('maize-12_t001',4),('maize-14_t001',2)]
def gt_palette(gt):
    labels=np.unique(gt[gt>=0]); pal=plt.get_cmap('turbo')(np.linspace(.04,.96,max(1,len(labels)))); return {int(g):pal[i] for i,g in enumerate(labels)}
def gt_colors(vals,palette):
    out=np.zeros((len(vals),4)); out[:,:3]=.82; out[:,3]=1
    for g,col in palette.items(): out[vals==g]=col
    return out
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
    out=np.zeros((len(pred),4)); out[:,:3]=.82; out[:,3]=1; mp=match_pred_to_gt(pred,gt)
    for p in np.unique(pred):
        p=int(p)
        if p<0: continue
        out[pred==p]=palette.get(mp.get(p,-1),np.array([.25,.25,.25,1.0]))
    return out
def inst_colors(vals,cmap_name='tab20'):
    out=np.zeros((len(vals),4)); out[:,:3]=.82; out[:,3]=1
    labs=[int(v) for v in np.unique(vals) if v>=0]; colors=plt.get_cmap(cmap_name)(np.linspace(0,1,max(1,len(labs))))
    for i,l in enumerate(labs): out[vals==l]=colors[i]
    return out
def semcolors(v):
    out=np.zeros((len(v),4)); out[:,:3]=.82; out[:,3]=1; out[v>0]=[.08,.72,.25,1]; return out
def style(ax,p,t):
    lo=np.percentile(p,1,axis=0); hi=np.percentile(p,99,axis=0); ax.set_xlim(lo[0],hi[0]); ax.set_ylim(lo[1],hi[1]); ax.set_zlim(lo[2],hi[2]); ax.set_box_aspect(hi-lo); ax.view_init(24,-62); ax.set_proj_type('ortho'); ax.set_axis_off(); ax.set_title(t,fontsize=24,pad=2)
fig=plt.figure(figsize=(25,5.2*len(rows)),dpi=300)
for r,(name,lbl) in enumerate(rows):
    xyz=z[f'{name}_xyz']; rgb=np.clip((z[f'{name}_rgb']+1.0)*0.5,0,1); gt=z[f'{name}_gt']; sem=z[f'{name}_sem']; nat=z[f'{name}_native']; hyb=z[f'{name}_hybrid']
    m=nat==lbl; p=xyz[m]; g=gt[m]; s=sem[m]; na=nat[m]; hy=hyb[m]
    pal=gt_palette(g); panels=[('Original RGB',rgb[m]),('GT instances',gt_colors(g,pal)),('PointGroup semantic',semcolors(s)),('PointGroup native',pred_colors(na,g,pal)),('PointGroup hybrid + DPVIS',pred_colors(hy,g,pal))]
    for c,(title,col) in enumerate(panels):
        ax=fig.add_subplot(len(rows),5,r*5+c+1,projection='3d'); ax.scatter(p[:,0],p[:,1],p[:,2],c=col,s=.7,alpha=1.0,depthshade=False,rasterized=True); style(ax,p,title)
        if c==0: ax.text2D(-.04,.5,f'{name}\nnative cluster {lbl}',transform=ax.transAxes,rotation=90,va='center',ha='right',fontsize=20)
fig.tight_layout(); out='E:/Cabbage/output/crops3d_generalization/figures/reproduced/maize_pointgroup_reproduced.png'; fig.savefig(out,dpi=300,bbox_inches='tight',facecolor='white'); print('saved',out)





