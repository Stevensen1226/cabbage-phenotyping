#!/usr/bin/env bash
set -u
ROOT=/root/autodl-tmp/cabbage-phenotyping
RUNROOT=/root/autodl-tmp/runs
LOGROOT=/root/autodl-tmp/logs
STATUS=$RUNROOT/queue_status.txt
cd "$ROOT" || exit 1
while screen -ls | grep -q 'randlanet_queue'; do sleep 30; done
echo "[$(date '+%F %T')] RandLA queue finished, preparing PointNet2 split" >> "$STATUS"
python tools/prepare_pointnet2_paper_split.py > "$LOGROOT/pointnet2_prepare.log" 2>&1
python train.py --data_path pn2_split --save_path "$RUNROOT/pointnet2_paper/best_model.pth" --num_classes 2 --batch_size 8 --epochs 100 --lr 0.001 --npoints 4096 > "$LOGROOT/pointnet2_train.log" 2>&1
rc=$?
echo "[$(date '+%F %T')] PointNet2 finished rc=$rc" >> "$STATUS"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
mkdir -p "$RUNROOT/pointnet2_final"
cp "$RUNROOT/pointnet2_paper/best_model.pth" "$RUNROOT/pointnet2_final/best_model.pth"
cp "$RUNROOT/pointnet2_paper/checkpoint_latest.pth" "$RUNROOT/pointnet2_final/checkpoint_latest.pth" 2>/dev/null || true
echo "[$(date '+%F %T')] PointNet2 best copied to $RUNROOT/pointnet2_final/best_model.pth" >> "$STATUS"
