#!/bin/bash
# Espera a que termine paper_dinov2 y lanza el run con prior suave; luego evalua ambos en test.
cd ~/reid-utec
UV=~/.local/bin/uv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
while pgrep -f "train.py --name paper_dinov2" > /dev/null; do sleep 60; done
mkdir -p runs/paper_soft
$UV run python train.py --name paper_soft --epochs 100 --batch 128 --workers 8 --mask-ratio 0.3 --downsample-p 0 > runs/paper_soft/train.log 2>&1
for n in paper_dinov2 paper_soft; do $UV run python evaluate.py runs/$n/last.pt --split testing.h5 > runs/$n/eval_test.log 2>&1; done
echo DONE > runs/NEXT_DONE
