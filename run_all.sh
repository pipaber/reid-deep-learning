#!/bin/bash
# Tres runs en secuencia: metodo del paper, baseline DINO v1, ablacion DINOv2 (Sinkhorn-Knopp + KoLeo)
cd ~/reid-utec
UV=~/.local/bin/uv
run() { name=$1; shift; mkdir -p runs/$name; $UV run python train.py --name $name --epochs 100 --batch 128 --workers 8 "$@" > runs/$name/train.log 2>&1; }
run paper
run dino --no-prior --no-patch-loss --lambda-cls 1
run paper_dinov2 --centering sinkhorn --koleo 0.1
for n in paper dino paper_dinov2; do $UV run python evaluate.py runs/$n/last.pt --split testing.h5 > runs/$n/eval_test.log 2>&1; done
echo ALL_DONE > runs/ALL_DONE
