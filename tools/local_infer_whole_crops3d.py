#!/usr/bin/env python3
"""Run whole-scene RandLA-Net / PointNet2-RGB inference locally on Crops3D."""
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/external/crops3d_is_converted/test'
OUT = ROOT / 'output/local_crops3d_preds'
NAMES = {'maize': ['maize-11','maize-12','maize-14','maize-2','maize-3'],
         'rapeseed': ['rapeseed-11','rapeseed-3','rapeseed-4','rapeseed-6']}
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'RandLANet_Ours'))
from randlanet import RandLANet
from cabbage_pheno.segmentation.pointnet2_model import PointNet2SemSeg

def load_gt(path):
    a=np.fromfile(path,sep=' ',dtype=np.float64); a=a[:a.size-(a.size%8)].reshape(-1,8)
    return np.ascontiguousarray(a[:,:3],dtype=np.float32), np.ascontiguousarray(a[:,3:6]/255.0*2.0-1.0,dtype=np.float32)

def votes_for(model_name, model, xyz, rgb, block, stride, npts, min_pts):
    lo=xyz[:,:2].min(0); hi=xyz[:,:2].max(0)
    votes=np.zeros((len(xyz),2),dtype=np.int32); seen=np.zeros(len(xyz),dtype=np.int32)
    for x0 in np.arange(lo[0],hi[0],stride):
        for y0 in np.arange(lo[1],hi[1],stride):
            m=((xyz[:,0]>=x0)&(xyz[:,0]<x0+block)&(xyz[:,1]>=y0)&(xyz[:,1]<y0+block)); idx=np.flatnonzero(m)
            if len(idx)<min_pts: continue
            take=np.linspace(0,len(idx)-1,npts).astype(np.int64) if len(idx)>=npts else np.arange(len(idx)); sel=idx[take]
            q=xyz[sel].copy(); q-=q.mean(0,keepdims=True); c=rgb[sel]
            if model_name=='randla': out=model(torch.from_numpy(q).unsqueeze(0).cuda(),torch.from_numpy(c).unsqueeze(0).cuda())
            else: out=model(torch.from_numpy(q.T).unsqueeze(0).cuda(),torch.from_numpy(c.T).unsqueeze(0).cuda())
            if isinstance(out,(tuple,list)): out=out[0]
            lab=out.argmax(-1)[0].cpu().numpy() if model_name=='randla' else out.argmax(1)[0].cpu().numpy()
            votes[sel,0]+=(lab==0); votes[sel,1]+=(lab==1); seen[sel]+=1
    pred=votes.argmax(1).astype(np.int8); cov=seen>0
    if cov.any() and not cov.all():
        tree=cKDTree(xyz[cov][:,:2]); _,nn=tree.query(xyz[~cov][:,:2],k=1); pred[~cov]=pred[cov][nn]
    return pred,int(cov.sum())

def run_model(name, model, block, stride, npts, min_pts):
    outdir=OUT/name; outdir.mkdir(parents=True,exist_ok=True)
    for crop,names in NAMES.items():
        for plot in names:
            t0=time.time(); xyz,rgb=load_gt(DATA/f'{plot}_gt.txt'); pred,covered=votes_for(name,model,xyz,rgb,block,stride,npts,min_pts)
            np.save(outdir/f'{plot}_pred.npy',pred); print(f'{name} {plot} n={len(xyz)} covered={covered} fg={int((pred==1).sum())} sec={time.time()-t0:.1f}',flush=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',choices=['randla','pointnet2','all'],default='all'); a=ap.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    if a.model in ('randla','all'):
        ck=torch.load(ROOT/'models/paper_training_20260923/randlanet_final/best.pth',map_location='cpu',weights_only=False); m=RandLANet(d_in=6,num_classes=2).cuda().eval(); m.load_state_dict(ck['model']); run_model('randla',m,1.0,0.5,65536,100); del m; torch.cuda.empty_cache()
    if a.model in ('pointnet2','all'):
        ck=torch.load(ROOT/'models/paper_training_20260923/pointnet2_rgb/best_model.pth',map_location='cpu',weights_only=False); m=PointNet2SemSeg(2,additional_channel=3).cuda().eval(); m.load_state_dict(ck['model']); run_model('pointnet2_rgb',m,1.0,0.5,4096,100); del m; torch.cuda.empty_cache()
if __name__=='__main__': main()

