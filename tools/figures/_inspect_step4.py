from pathlib import Path
import numpy as np, open3d as o3d
root=Path(r'E:\Cabbage')
for scene in ['cloudR5','cloudR9','cloudR12','cloudR14']:
    d=root/'output'/'step_preview'/f'{scene}_randla_watershed_ransac025'
    b=o3d.io.read_point_cloud(str(d/f'{scene}_step4b_skeleton_split.ply'))
    c=o3d.io.read_point_cloud(str(d/f'{scene}_step4c_fragment_merged.ply'))
    pb=np.asarray(b.points); pc=np.asarray(c.points); cb=np.round(np.asarray(b.colors)*255).astype(np.uint8); cc=np.round(np.asarray(c.colors)*255).astype(np.uint8)
    ub, cntb=np.unique(cb,axis=0,return_counts=True); uc, cntc=np.unique(cc,axis=0,return_counts=True)
    print(scene, 'points', len(pb), len(pc), 'same_xy', np.allclose(pb,pc), 'colors',len(ub),len(uc),'counts_top',sorted(cntb,reverse=True)[:8])
