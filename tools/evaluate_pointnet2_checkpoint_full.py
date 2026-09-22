#!/usr/bin/env python3
import sys, json, torch, numpy as np, argparse
from pathlib import Path
root = Path('/root/autodl-tmp/cabbage-phenotyping')
sys.path.insert(0, str(root))
from cabbage_pheno.segmentation.pointnet2_model import PointNet2SemSeg
p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--output', required=True)
p.add_argument('--split', default='test')
a = p.parse_args()
model = PointNet2SemSeg(2).cuda()
ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
state = ck.get('model') or ck.get('model_state_dict') or ck
model.load_state_dict(state)
model.eval()
files = sorted((root / 'pn2_split' / a.split).glob('*.npy'))
inter = np.zeros(2); union = np.zeros(2); correct = total = 0
with torch.no_grad():
    for f in files:
        d = np.load(f).astype(np.float32); xyz = d[:, :3]; sem = d[:, 3].astype(np.int64); n = len(xyz)
        for start in range(0, n, 4096):
            idx = np.arange(start, min(start + 4096, n)); valid = len(idx)
            if valid < 4096: idx = np.concatenate([idx, np.full(4096 - valid, idx[-1], dtype=np.int64)])
            q = xyz[idx].copy(); q -= q.mean(0, keepdims=True); t = sem[idx]
            pred = model(torch.from_numpy(q).T.unsqueeze(0).cuda())[0].argmax(1)[0].cpu().numpy()[:valid]; t = t[:valid]
            for c in range(2):
                pi = pred == c; ti = t == c
                inter[c] += float((pi & ti).sum()); union[c] += float((pi | ti).sum())
            correct += int((pred == t).sum()); total += valid
iou = np.divide(inter, union, out=np.full(2, np.nan), where=union > 0)
out = {'checkpoint': a.checkpoint, 'split': a.split, 'points': total, 'iou_background': float(iou[0]), 'iou_cabbage': float(iou[1]), 'miou': float(np.nanmean(iou)), 'accuracy': correct / total}
Path(a.output).write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
