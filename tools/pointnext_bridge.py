#!/usr/bin/env python3
"""PointNeXt Bridge: 训好的 PointNeXt-L 接 Cabbage 统一评估"""
import sys,os,numpy as np,torch,json,glob,time,gc
sys.path.insert(0,'/home/stevensen/Cabbage')
sys.path.insert(0,'/home/stevensen/PointNeXt-master')

from openpoints.models import build_model_from_cfg
from openpoints.utils import EasyConfig
from cabbage_pheno.io import read_point_cloud
from cabbage_pheno.service.pipeline import collect_input_files,preprocess_point_cloud
from cabbage_pheno.service.evaluation import load_config,load_ground_truth,align_gt_to_pred,evaluate_from_predictions
from cabbage_pheno.instance import InstanceClusterer
from scipy.spatial import cKDTree
import open3d as o3d

cfg=load_config('configs/default.yaml')
pn_cfg=EasyConfig();pn_cfg.load('/home/stevensen/PointNeXt-master/cfgs/cabbage.yaml',recursive=True)
model=build_model_from_cfg(pn_cfg.model).cuda().eval()
sd=torch.load('/home/stevensen/PointNeXt-master/best_model.pth',map_location='cpu')
model.load_state_dict(sd,strict=False)
print(f'PointNeXt-L loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params')

# Verify: pos=(1,N,3), x=(1,N,6), offset=(1,)
x=torch.randn(1,500,3).cuda();x-=x.min(1)[0].values.unsqueeze(1)
f=torch.randn(1,500,6).cuda();o=torch.tensor([500],dtype=torch.int32).unsqueeze(0).cuda()
out=model({'pos':x,'x':f,'offset':o})
print(f'Forward OK: {list(out.shape)}')

files=collect_input_files('e_data/train')
print(f'Total files: {len(files)}')

recs=[]
for idx,fp in enumerate(files):
    nm=os.path.splitext(os.path.basename(fp))[0];print(f'[{idx+1}/{len(files)}] {nm}...',flush=True)
    t0=time.time()
    pcd=read_point_cloud(fp)
    if pcd is None:continue
    gt_pts,gt_sem,gt_inst=load_ground_truth(fp,-2,-1,True)
    if gt_inst is None:continue
    pr=preprocess_point_cloud(pcd,cfg)
    pc=pr.points_clean;ng=pr.non_ground_pcd
    xyz_ng=np.asarray(ng.points,np.float32)
    rgb_ng=np.asarray(ng.colors,np.float32) if len(ng.colors)>0 else np.ones((len(xyz_ng),3),np.float32)*0.5
    
    # Key: add batch dim -> (1,N,3) for pos, (1,N,6) for x, (1,) for offset
    N=len(xyz_ng)
    b_xyz=torch.FloatTensor(xyz_ng).unsqueeze(0).cuda()
    b_xyz-=b_xyz.min(1)[0].values.unsqueeze(1)
    b_feat=torch.FloatTensor(np.concatenate([xyz_ng,rgb_ng],1)).unsqueeze(0).cuda()
    b_off=torch.tensor([N],dtype=torch.int32).unsqueeze(0).cuda()
    with torch.no_grad():sem_ng=model({'pos':b_xyz,'x':b_feat,'offset':b_off}).squeeze(0).argmax(1).cpu().numpy()
    del b_xyz,b_feat,b_off;torch.cuda.empty_cache()
    
    cabbage_mask=sem_ng==1
    if cabbage_mask.sum()<10:
        rt=time.time()-t0;fs=np.zeros(len(pc),int);fi=np.full(len(pc),-1,int)
        if len(pc)>0:t=cKDTree(pc);ngi=t.query(xyz_ng,k=1)[1];fs[np.array(ngi)[cabbage_mask]]=1
        gs,gi=align_gt_to_pred(gt_pts,gt_sem,gt_inst,pc)
        if gi is not None:gi=gi.copy();gi[gi<=0]=0
        recs.append({'fname':nm,'pred_sem':fs,'pred_inst':fi,'gt_sem':gs,'gt_inst':gi,'runtime':rt})
        print(f'  no cabbage, {rt:.1f}s',flush=True);continue
    
    cab_pcd=o3d.geometry.PointCloud();cab_pcd.points=o3d.utility.Vector3dVector(xyz_ng[cabbage_mask])
    if len(rgb_ng)>0:cab_pcd.colors=o3d.utility.Vector3dVector(rgb_ng[cabbage_mask])
    clusterer=InstanceClusterer(cfg)
    try:inst_cab,_=clusterer.cluster(cab_pcd)
    except:inst_cab=np.full(cabbage_mask.sum(),-1,dtype=np.int64)
    fs=np.zeros(len(pc),int);fi=np.full(len(pc),-1,int)
    if len(pc)>0:t=cKDTree(pc);ngi=t.query(xyz_ng,k=1)[1];cab2c=np.array(ngi)[cabbage_mask];fs[cab2c]=1
    v=inst_cab>=0;fi[cab2c[v]]=inst_cab[v]
    gs,gi=align_gt_to_pred(gt_pts,gt_sem,gt_inst,pc)
    if gi is not None:gi=gi.copy();gi[gi<=0]=0
    rt=time.time()-t0;nf=len(np.unique(fi[fi>0]))
    print(f'  {nf} inst, {rt:.1f}s',flush=True)
    recs.append({'fname':nm,'pred_sem':fs,'pred_inst':fi,'gt_sem':gs,'gt_inst':gi,'runtime':rt})

res=evaluate_from_predictions(recs,iou_thresh=0.5,label='PointNeXt-L')
os.makedirs('output',exist_ok=True)
with open('output/pointnext_result.json','w') as f:json.dump(res,f,ensure_ascii=False,indent=2)
s=res['summary'];print();print('=== PointNeXt-L RESULT ===')
for k,v in sorted(s.items()):print(f'  {k}: {v}')
