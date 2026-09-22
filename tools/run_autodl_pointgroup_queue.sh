#!/usr/bin/env bash
set -u
ROOT=/root/autodl-tmp/cabbage-phenotyping
RUNROOT=/root/autodl-tmp/runs
LOGROOT=/root/autodl-tmp/logs
STATUS=$RUNROOT/queue_status.txt
mkdir -p "$RUNROOT/pointgroup_final" "$LOGROOT"
while screen -ls | grep -q 'pointnet2_queue'; do sleep 30; done
echo "[$(date '+%F %T')] PointNet2 queue finished, starting PointGroup" >> "$STATUS"
cd "$ROOT/PointGroup_Ours" || exit 1
export PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin:/root/miniconda3/bin
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
/root/miniconda3/bin/python train.py --config config/pointgroup_cabbage_paper.yaml > "$LOGROOT/pointgroup_train.log" 2>&1
rc=$?
echo "[$(date '+%F %T')] PointGroup finished rc=$rc" >> "$STATUS"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
cp "$ROOT/PointGroup_Ours/exp/cabbage_dataset/pointgroup/pointgroup_cabbage_paper/best.pth" "$RUNROOT/pointgroup_final/best.pth"
echo "[$(date '+%F %T')] PointGroup best copied to $RUNROOT/pointgroup_final/best.pth" >> "$STATUS"
