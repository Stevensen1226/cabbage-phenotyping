#!/usr/bin/env python3
import os, sys, json, torch, numpy as np, argparse
from pathlib import Path
from torch.utils.data import DataLoader
root = Path('/root/autodl-tmp/cabbage-phenotyping/PointGroup_Ours')
custom_args = sys.argv[1:]
os.chdir(root); sys.path.insert(0, str(root)); sys.argv = ['eval', '--config', 'config/pointgroup_cabbage_paper.yaml']
from util.config import cfg
from dataset.cabbage_dataset import CabbageDataset, cabbage_collate_fn
from model.pointgroup.pointgroup import PointGroup, model_fn_decorator
p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--output', required=True)
p.add_argument('--split', default='test')
a, _ = p.parse_known_args(custom_args); cfg.task = 'test'; cfg.batch_size = 1
ds = CabbageDataset(cfg, a.split)
dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=cabbage_collate_fn)
model = PointGroup(cfg).cuda(); ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False); model.load_state_dict(ck.get('model', ck)); model.eval(); mfn = model_fn_decorator(test=True)
inter = np.zeros(2); union = np.zeros(2); correct = total = 0
with torch.no_grad():
    for batch in dl:
        pred = mfn(batch, model, 200)['semantic'].argmax(1); t = batch['labels'].cuda()
        valid = t >= 0; pred = pred[valid]; t = t[valid]
        for c in range(2):
            pi = pred == c; ti = t == c
            inter[c] += float((pi & ti).sum()); union[c] += float((pi | ti).sum())
        correct += int((pred == t).sum()); total += int(t.numel())
iou = np.divide(inter, union, out=np.full(2, np.nan), where=union > 0)
out = {'checkpoint': a.checkpoint, 'split': a.split, 'points': total, 'iou_background': float(iou[0]), 'iou_cabbage': float(iou[1]), 'miou': float(np.nanmean(iou)), 'accuracy': correct / total}
Path(a.output).write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2))

