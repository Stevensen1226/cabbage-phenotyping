from pathlib import Path
from collections import Counter
import numpy as np, open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
root=Path(r'E:\Cabbage')
scenes=['cloudR5','cloudR9','cloudR12','cloudR14']
cases=[]
for scene in scenes:
    d=root/'output'/'step_preview'/f'{scene}_randla_watershed_ransac025'
    b=o3d.io.read_point_cloud(str(d/f'{scene}_step4b_skeleton_split.ply'))
    c=o3d.io.read_point_cloud(str(d/f'{scene}_step4c_fragment_merged.ply'))
    pts=np.asarray(b.points); cb=np.round(np.asarray(b.colors)*255).astype(np.uint8); cc=np.round(np.asarray(c.colors)*255).astype(np.uint8)
    ub=np.unique(cb,axis=0)
    for color in ub:
        sb=np.all(cb==color,axis=1)
        if sb.sum()<80: continue
        modes=Counter(map(tuple,cc[sb])).most_common()
        target,nt=modes[0]
        if target==tuple(color): continue
        if nt/sb.sum()<0.60: continue
        st=np.all(cb==np.asarray(target,dtype=np.uint8),axis=1)
        if st.sum()<80: continue
        sel=sb|st
        p=pts[sel]
        # area not degenerate, preserve cases
        cases.append(dict(scene=scene,bcolor=tuple(map(int,color)),tcolor=tuple(map(int,target)),n_b=int(sb.sum()),n_t=int(st.sum()),pts=p,mask=sb[sel]))
cases.sort(key=lambda x:(x['scene'],x['n_b']))
print('cases',len(cases))
for i,x in enumerate(cases): print(i,x['scene'],x['n_b'],x['n_t'],x['bcolor'],x['tcolor'])
ncol=4; nrow=int(np.ceil(len(cases)/ncol))
fig=plt.figure(figsize=(3.2*ncol,2.5*nrow),facecolor='white')
for i,x in enumerate(cases):
    ax=fig.add_subplot(nrow,ncol,i+1)
    p=x['pts']; m=x['mask']
    ax.scatter(p[~m,0],p[~m,1],s=.35,c='#94c4e8',edgecolors='none',alpha=.48,rasterized=True)
    ax.scatter(p[m,0],p[m,1],s=.35,c='#a64b3b',edgecolors='none',alpha=.70,rasterized=True)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f"{i}: {x['scene']}  {x['n_b']}->{x['n_t']}",fontsize=9)
    lo=p.min(0); hi=p.max(0); pad=.03*np.maximum(hi-lo,1e-6)
    ax.set_xlim(lo[0]-pad[0],hi[0]+pad[0]); ax.set_ylim(lo[1]-pad[1],hi[1]+pad[1])
fig.tight_layout(pad=.4)
out=root/'tools'/'figures'/'_randla_merge_pair_contact.png'
fig.savefig(out,dpi=200,bbox_inches='tight'); print(out)
