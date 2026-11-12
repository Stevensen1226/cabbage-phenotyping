import numpy as np, matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial import cKDTree
z=np.load('/root/autodl-tmp/logs/pg_maize_rgb_vis_dpvis.npz'); n='maize-8_t000'
xyz=z[n+'_xyz']; rgb=np.clip((z[n+'_rgb']+1.0)*0.5,0,1); gt=z[n+'_gt']; sem=z[n+'_sem']; nat0=z[n+'_native']; hyb0=z[n+'_hybrid']
def fill_foreground(labels,xyz,sem,radius=0.05,k=5,max_passes=50):
    out=labels.copy(); fg=sem==1
    for _ in range(max_passes):
        noise=np.where(fg & (out<0))[0]; valid=np.where(fg & (out>=0))[0]
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
    noise=np.where(fg & (out<0))[0]; valid=np.where(fg & (out>=0))[0]
    if len(noise) and len(valid):
        tree=cKDTree(xyz[valid]); _,nn=tree.query(xyz[noise],k=1,workers=-1); out[noise]=out[valid][nn]
    return out
nat=fill_foreground(nat0,xyz,sem); hyb=fill_foreground(hyb0,xyz,sem)
def inst_colors(vals,cmap_name='tab20'):
    out=np.zeros((len(vals),4)); out[:,:3]=.82; out[:,3]=1
    labs=[int(v) for v in np.unique(vals) if v>=0]; colors=plt.get_cmap(cmap_name)(np.linspace(0,1,max(1,len(labs))))
    for i,l in enumerate(labs): out[vals==l]=colors[i]
    return out
def semcolors(v):
    out=np.zeros((len(v),4)); out[:,:3]=.82; out[:,3]=1; out[v>0]=[0.08,0.72,0.25,1]; return out
def style(ax,xyz,t):
    lo=np.percentile(xyz,1,axis=0); hi=np.percentile(xyz,99,axis=0); ax.set_xlim(lo[0],hi[0]); ax.set_ylim(lo[1],hi[1]); ax.set_zlim(lo[2],hi[2]); ax.set_box_aspect(hi-lo); ax.view_init(24,-62); ax.set_proj_type('ortho'); ax.set_axis_off(); ax.set_title(t,fontsize=10,pad=2)
rows=[10,11]; fig=plt.figure(figsize=(25,5.2*len(rows)),dpi=300)
for r,lbl in enumerate(rows):
    m=nat0==lbl; p=xyz[m]; g=gt[m]; s=sem[m]; na=nat[m]; hy=hyb[m]
    panels=[('Original RGB',rgb[m]),('GT instances',inst_colors(g)),('PointGroup semantic',semcolors(s)),('PointGroup native',inst_colors(na)),('PointGroup hybrid + DPVIS',inst_colors(hy))]
    for c,(title,col) in enumerate(panels):
        ax=fig.add_subplot(len(rows),5,r*5+c+1,projection='3d'); ax.scatter(p[:,0],p[:,1],p[:,2],c=col,s=1.0,alpha=1.0,depthshade=False,rasterized=True); style(ax,p,title)
        if c==0: ax.text2D(-.04,.5,f'maize-8_t000\nnative cluster {lbl}',transform=ax.transAxes,rotation=90,va='center',ha='right',fontsize=9)
fig.tight_layout(); out='/root/autodl-tmp/logs/pg_maize_dpvis_adhesion_zoom_unique.png'; fig.savefig(out,dpi=300,bbox_inches='tight',facecolor='white'); print('saved',out)

