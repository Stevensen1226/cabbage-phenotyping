#!/usr/bin/env python3
"""PointNet2 semantic training with RGB features on the paper split."""
import argparse,csv,json,logging,time
from pathlib import Path
import numpy as np,torch,torch.nn as nn
from torch.utils.data import Dataset,DataLoader
root=Path('/root/autodl-tmp/cabbage-phenotyping')
from cabbage_pheno.segmentation.pointnet2_model import PointNet2SemSeg
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s'); logger=logging.getLogger('PN2RGB')
class DS(Dataset):
 def __init__(self,d,n=4096,train=False): self.files=sorted(Path(d).glob('*.npy')); self.n=n; self.train=train
 def __len__(self): return len(self.files)
 def __getitem__(self,i):
  a=np.load(self.files[i]).astype(np.float32); xyz=a[:,:3];rgb=a[:,3:6];lab=a[:,6].astype(np.int64);n=len(xyz)
  idx=np.random.choice(n,self.n,replace=True) if self.train else np.linspace(0,n-1,self.n,dtype=np.int64)
  xyz=xyz[idx].copy();rgb=rgb[idx].copy();lab=lab[idx].copy();xyz-=xyz.mean(0,keepdims=True)
  if self.train:
   t=np.random.rand()*2*np.pi;c,s=np.cos(t),np.sin(t);R=np.array([[c,-s,0],[s,c,0],[0,0,1]],np.float32);xyz=xyz@R.T;xyz=xyz*np.random.uniform(.9,1.1)+np.random.randn(*xyz.shape).astype(np.float32)*.002
  return torch.from_numpy(xyz),torch.from_numpy(rgb),torch.from_numpy(lab)
def collate(b): return tuple(torch.stack(x) for x in zip(*b))
def ev(model,dl,dev,crit):
 model.eval(); inter=np.zeros(2);union=np.zeros(2);loss_sum=0.;correct=total=0
 with torch.no_grad():
  for xyz,rgb,lab in dl:
   xyz=xyz.to(dev);rgb=rgb.to(dev);lab=lab.to(dev);out=model(xyz.permute(0,2,1),rgb.permute(0,2,1))[0];loss=crit(out,lab);loss_sum+=float(loss);pred=out.argmax(1)
   for c in range(2):
    pi=pred==c;ti=lab==c;inter[c]+=float((pi&ti).sum());union[c]+=float((pi|ti).sum())
   correct+=int((pred==lab).sum());total+=int(lab.numel())
 iou=np.divide(inter,union,out=np.full(2,np.nan),where=union>0);return {'loss':loss_sum/max(len(dl),1),'iou_background':float(iou[0]),'iou_cabbage':float(iou[1]),'miou':float(np.nanmean(iou)),'accuracy':correct/max(total,1)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',default=str(root/'pn2_rgb_blocks'));p.add_argument('--save_dir',default='/root/autodl-tmp/runs/pointnet2_rgb_paper');p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch',type=int,default=8);p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--npoints',type=int,default=4096);p.add_argument('--patience',type=int,default=25);p.add_argument('--workers',type=int,default=4);a=p.parse_args();torch.manual_seed(42);np.random.seed(42);dev='cuda';out=Path(a.save_dir);out.mkdir(parents=True,exist_ok=True)
 tr=DS(Path(a.data)/'train',a.npoints,True);va=DS(Path(a.data)/'val',a.npoints,False);tl=DataLoader(tr,a.batch,True,num_workers=a.workers,collate_fn=collate,drop_last=True);vl=DataLoader(va,a.batch,False,num_workers=a.workers,collate_fn=collate)
 model=PointNet2SemSeg(2,additional_channel=3).to(dev);crit=nn.NLLLoss(weight=torch.tensor([1.1,.9],device=dev));opt=torch.optim.Adam(model.parameters(),lr=a.lr,weight_decay=1e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs,eta_min=1e-5);best=-1;stale=0;hist=[]
 for ep in range(1,a.epochs+1):
  model.train();ls=0.;t0=time.time()
  for xyz,rgb,lab in tl:
   xyz=xyz.to(dev);rgb=rgb.to(dev);lab=lab.to(dev);logit=model(xyz.permute(0,2,1),rgb.permute(0,2,1))[0];loss=crit(logit,lab);opt.zero_grad(set_to_none=True);loss.backward();opt.step();ls+=float(loss)
  sched.step();m=ev(model,vl,dev,crit);m['epoch']=ep;m['train_loss']=ls/max(len(tl),1);m['seconds']=time.time()-t0;hist.append(m);logger.info('epoch=%d train_loss=%.4f val_loss=%.4f IoU_bg=%.4f IoU_cab=%.4f mIoU=%.4f acc=%.4f',ep,m['train_loss'],m['loss'],m['iou_background'],m['iou_cabbage'],m['miou'],m['accuracy'])
  state={'epoch':ep,'model':model.state_dict(),'val_metrics':m};torch.save(state,out/'last.pth')
  if m['miou']>best:best=m['miou'];stale=0;torch.save(state,out/'best_model.pth');logger.info('new best mIoU %.4f',best)
  else:stale+=1
  if stale>=a.patience:logger.info('early stop epoch %d',ep);break
 with (out/'history.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=hist[0].keys());w.writeheader();w.writerows(hist)
 (out/'summary.json').write_text(json.dumps({'best_miou':best,'best':max(hist,key=lambda x:x['miou'])},indent=2))
if __name__=='__main__':main()
