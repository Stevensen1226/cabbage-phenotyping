import numpy as np, matplotlib.pyplot as plt
z=np.load('/root/autodl-tmp/logs/pg_maize_tall_screen.npz')
rows=[('maize-12_t001',4),('maize-14_t002',1),('maize-14_t001',2)]
def inst_colors(vals,cmap_name='tab20'):
    out=np.zeros((len(vals),4)); out[:,:3]=.82; out[:,3]=1
    labs=[int(v) for v in np.unique(vals) if v>=0]; colors=plt.get_cmap(cmap_name)(np.linspace(0,1,max(1,len(labs))))
    for i,l in enumerate(labs): out[vals==l]=colors[i]
    return out
def semcolors(v):
    out=np.zeros((len(v),4)); out[:,:3]=.82; out[:,3]=1; out[v>0]=[.08,.72,.25,1]; return out
def style(ax,p,t):
    lo=np.percentile(p,1,axis=0); hi=np.percentile(p,99,axis=0); ax.set_xlim(lo[0],hi[0]); ax.set_ylim(lo[1],hi[1]); ax.set_zlim(lo[2],hi[2]); ax.set_box_aspect(hi-lo); ax.view_init(24,-62); ax.set_proj_type('ortho'); ax.set_axis_off(); ax.set_title(t,fontsize=10,pad=2)
fig=plt.figure(figsize=(25,5.2*len(rows)),dpi=300)
for r,(name,lbl) in enumerate(rows):
    xyz=z[f'{name}_xyz']; rgb=np.clip((z[f'{name}_rgb']+1.0)*0.5,0,1); gt=z[f'{name}_gt']; sem=z[f'{name}_sem']; nat=z[f'{name}_native']; hyb=z[f'{name}_hybrid']
    m=nat==lbl; p=xyz[m]; g=gt[m]; s=sem[m]; na=nat[m]; hy=hyb[m]
    panels=[('Original RGB',rgb[m]),('GT instances',inst_colors(g)),('PointGroup semantic',semcolors(s)),('PointGroup native',inst_colors(na)),('PointGroup hybrid + DPVIS',inst_colors(hy))]
    for c,(title,col) in enumerate(panels):
        ax=fig.add_subplot(len(rows),5,r*5+c+1,projection='3d'); ax.scatter(p[:,0],p[:,1],p[:,2],c=col,s=.7,alpha=1.0,depthshade=False,rasterized=True); style(ax,p,title)
        if c==0: ax.text2D(-.04,.5,f'{name}\nnative cluster {lbl}',transform=ax.transAxes,rotation=90,va='center',ha='right',fontsize=9)
fig.tight_layout(); out='/root/autodl-tmp/logs/pg_maize_tall_screened_zoom.png'; fig.savefig(out,dpi=300,bbox_inches='tight',facecolor='white'); print('saved',out)
