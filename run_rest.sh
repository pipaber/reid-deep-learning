#!/bin/bash
# 1) espera la evaluacion en test de paper_dinov2, 2) cierre con los 3 runs, 3) paper_soft + test, 4) cierre final.
cd ~/reid-utec
export PATH=$HOME/.local/bin:/usr/local/bin:$PATH PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q "guardado en" runs/paper_dinov2/eval_test.log 2>/dev/null; do sleep 20; done
uv run python finalize.py > finalize1.log 2>&1
echo "FINALIZE1_EXIT=$?" > runs/FINALIZE1_DONE
mkdir -p runs/paper_soft
uv run python train.py --name paper_soft --epochs 100 --batch 128 --workers 8 --mask-ratio 0.3 --downsample-p 0 > runs/paper_soft/train.log 2>&1
uv run python evaluate.py runs/paper_soft/last.pt --split testing.h5 > runs/paper_soft/eval_test.log 2>&1
git fetch -q origin && git reset -q --hard origin/main
uv run python finalize.py > finalize2.log 2>&1
echo "FINALIZE2_EXIT=$?" > runs/FINALIZE2_DONE
