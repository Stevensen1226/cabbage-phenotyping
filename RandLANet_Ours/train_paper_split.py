#!/usr/bin/env python3
"""Train RandLA-Net with an explicit paper train/val/test split."""
from __future__ import annotations
import argparse, csv, json, logging, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from randlanet import RandLANet

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
logger=logging.getLogger('RandLANetPaper')

class CabbageDataset(Dataset):
    def __init__(self,root:str,num_points:int,train:bool):
        self.files=sorted(Path(root).glob('*.pth'))
        if not self.files: raise FileNotFoundError(f'No .pth files in {root}')
        self.num_points=int(num_points); self.train=bool(train)
        logger.info(f'Dataset {root}: {len(self.files)} scans, train={self.train}')
    def __len__(self): return len(self.files)
    def __getitem__(self,idx):
        d=torch.load(self.files[idx],map_location='cpu',weights_only=False)
        xyz,rgb,sem=d[0],d[1],d[2]
        xyz=np.asarray(xyz,dtype=np.float32); rgb=np.asarray(rgb,dtype=np.float32); sem=np.asarray(sem,dtype=np.int64)
        n=len(xyz)
        if n>=self.num_points:
            if self.train: sel=np.random.choice(n,self.num_points,replace=False)
            else: sel=np.linspace(0,n-1,self.num_points,dtype=np.int64)
        else:
            sel=np.random.choice(n,self.num_points,replace=True) if self.train else np.arange(self.num_points)%n
        xyz=xyz[sel].copy(); rgb=rgb[sel].copy(); sem=sem[sel].copy()
        xyz-=xyz.mean(0,keepdims=True)
        if self.train:
            if np.random.rand()>0.5:
                t=np.random.rand()*2*np.pi; c,s=np.cos(t),np.sin(t); R=np.array([[c,-s,0],[s,c,0],[0,0,1]],dtype=np.float32); xyz=xyz@R.T
            xyz=xyz*np.random.uniform(.9,1.1)+np.random.randn(*xyz.shape).astype(np.float32)*.002
        sem[sem<0]=-100
        return torch.from_numpy(xyz),torch.from_numpy(rgb),torch.from_numpy(sem)

def collate(batch):
    xyz,rgb,sem=zip(*batch); return torch.stack(xyz),torch.stack(rgb),torch.stack(sem)

def evaluate(model,loader,device,criterion):
    model.eval(); loss_sum=0.; inter=np.zeros(2,dtype=np.float64); union=np.zeros(2,dtype=np.float64); correct=0; total=0
    with torch.no_grad():
        for xyz,rgb,sem in loader:
            xyz=xyz.to(device,non_blocking=True); rgb=rgb.to(device,non_blocking=True); sem=sem.to(device,non_blocking=True)
            logits=model(xyz,rgb); loss=criterion(logits.reshape(-1,2),sem.reshape(-1)); loss_sum+=float(loss)
            pred=logits.argmax(-1)
            for c in range(2):
                pi=pred==c; ti=sem==c; inter[c]+=float((pi&ti).sum()); union[c]+=float((pi|ti).sum())
            correct+=int((pred==sem).sum()); total+=int(sem.numel())
    iou=np.divide(inter,union,out=np.full(2,np.nan),where=union>0)
    return {'loss':loss_sum/max(len(loader),1),'iou_background':float(iou[0]),'iou_cabbage':float(iou[1]),'miou':float(np.nanmean(iou)),'accuracy':correct/max(total,1)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train_data',required=True); p.add_argument('--val_data',required=True); p.add_argument('--test_data',default=None)
    p.add_argument('--epochs',type=int,default=200); p.add_argument('--patience',type=int,default=30); p.add_argument('--batch',type=int,default=2)
    p.add_argument('--lr',type=float,default=1e-3); p.add_argument('--num_points',type=int,default=65536); p.add_argument('--workers',type=int,default=2)
    p.add_argument('--save_dir',required=True); p.add_argument('--seed',type=int,default=42); p.add_argument('--evaluate_test',action='store_true')
    a=p.parse_args(); random_state=np.random.RandomState(a.seed); torch.manual_seed(a.seed); np.random.seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    out=Path(a.save_dir); out.mkdir(parents=True,exist_ok=True)
    train_ds=CabbageDataset(a.train_data,a.num_points,True); val_ds=CabbageDataset(a.val_data,a.num_points,False)
    train_dl=DataLoader(train_ds,a.batch,shuffle=True,num_workers=a.workers,collate_fn=collate,pin_memory=True,drop_last=False)
    val_dl=DataLoader(val_ds,1,shuffle=False,num_workers=a.workers,collate_fn=collate,pin_memory=True)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=RandLANet(d_in=6,num_classes=2).to(dev)
    crit=nn.CrossEntropyLoss(ignore_index=-100,weight=torch.tensor([0.4,1.0],device=dev)); opt=torch.optim.Adam(model.parameters(),lr=a.lr)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs,eta_min=a.lr*.05)
    best=-1.; stale=0; history=[]
    for ep in range(1,a.epochs+1):
        model.train(); t0=time.time(); loss_sum=0.
        for xyz,rgb,sem in train_dl:
            xyz=xyz.to(dev,non_blocking=True); rgb=rgb.to(dev,non_blocking=True); sem=sem.to(dev,non_blocking=True)
            logits=model(xyz,rgb); loss=crit(logits.reshape(-1,2),sem.reshape(-1))
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); loss_sum+=float(loss)
        sched.step(); vm=evaluate(model,val_dl,dev,crit); vm['epoch']=ep; vm['train_loss']=loss_sum/max(len(train_dl),1); vm['seconds']=time.time()-t0; history.append(vm)
        logger.info('epoch=%d train_loss=%.4f val_loss=%.4f IoU_bg=%.4f IoU_cab=%.4f mIoU=%.4f acc=%.4f time=%.1fs',ep,vm['train_loss'],vm['loss'],vm['iou_background'],vm['iou_cabbage'],vm['miou'],vm['accuracy'],vm['seconds'])
        score=vm['iou_cabbage']+0.25*vm['miou']
        state={'epoch':ep,'model':model.state_dict(),'optimizer':opt.state_dict(),'val_metrics':vm,'args':vars(a)}
        torch.save(state,out/'last.pth')
        if score>best:
            best=score; stale=0; torch.save(state,out/'best.pth'); logger.info('saved new best epoch %d',ep)
        else:
            stale+=1
        if stale>=a.patience: logger.info('early stopping at epoch %d',ep); break
    with (out/'history.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(history[0].keys())); w.writeheader(); w.writerows(history)
    result={'best_selection_score':best,'epochs_completed':len(history),'best_val':max(history,key=lambda x:x['iou_cabbage']+0.25*x['miou'])}
    if a.evaluate_test and a.test_data:
        ckpt=torch.load(out/'best.pth',map_location='cpu',weights_only=False); model.load_state_dict(ckpt['model']); test_ds=CabbageDataset(a.test_data,a.num_points,False); test_dl=DataLoader(test_ds,1,shuffle=False,num_workers=a.workers,collate_fn=collate); result['test']=evaluate(model,test_dl,dev,crit)
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); logger.info('training complete: %s',result)

if __name__=='__main__': main()
