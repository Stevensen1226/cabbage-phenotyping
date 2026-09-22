#!/usr/bin/env bash
set -u
ROOT=/root/autodl-tmp/cabbage-phenotyping
RUNROOT=/root/autodl-tmp/runs
LOGROOT=/root/autodl-tmp/logs
STATUS=$RUNROOT/queue_status.txt
mkdir -p "$RUNROOT" "$LOGROOT"
echo "[$(date '+%F %T')] queue started, waiting for randlanet_paper" >> "$STATUS"
while screen -ls | grep -q 'randlanet_paper'; do sleep 20; done
echo "[$(date '+%F %T')] randlanet_paper finished" >> "$STATUS"
cd "$ROOT" || exit 1
for seed in 123 2026; do
  run="$RUNROOT/randlanet_seed${seed}"
  log="$LOGROOT/randlanet_seed${seed}.log"
  mkdir -p "$run"
  echo "[$(date '+%F %T')] START seed=$seed" >> "$STATUS"
  python RandLANet_Ours/train_paper_split.py --train_data pointgroup_format/train --val_data pointgroup_format/val --test_data pointgroup_format/test --epochs 200 --patience 30 --batch 2 --workers 4 --seed "$seed" --save_dir "$run" --evaluate_test > "$log" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] END seed=$seed rc=$rc" >> "$STATUS"
  if [ "$rc" -ne 0 ]; then
    echo "[$(date '+%F %T')] queue stopped on failure seed=$seed" >> "$STATUS"
    exit "$rc"
  fi
done
python - <<'PY'
import json, shutil
from pathlib import Path
root=Path('/root/autodl-tmp/runs')
runs=[root/'randlanet_paper',root/'randlanet_seed123',root/'randlanet_seed2026']
rows=[]
for d in runs:
    s=json.loads((d/'summary.json').read_text())
    v=s['best_val']
    rows.append({'run':d.name,'score':float(v['iou_cabbage'])+0.25*float(v['miou']),'epoch':v['epoch'],'iou_cabbage':v['iou_cabbage'],'miou':v['miou'],'val_loss':v['loss'],'test':s.get('test')})
best=max(rows,key=lambda x:x['score'])
out=root/'randlanet_final'; out.mkdir(exist_ok=True)
shutil.copy2(root/best['run']/'best.pth',out/'best.pth')
shutil.copy2(root/best['run']/'history.csv',out/'history.csv')
(out/'selection.json').write_text(json.dumps({'selected':best,'candidates':rows},indent=2))
print(json.dumps({'selected':best,'candidates':rows},indent=2))
PY
echo "[$(date '+%F %T')] ALL DONE best copied to $RUNROOT/randlanet_final/best.pth" >> "$STATUS"
