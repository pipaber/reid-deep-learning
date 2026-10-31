#!/bin/bash
# Espera a que terminen los runs (run_next.sh escribe runs/NEXT_DONE) y cierra el proyecto: tablas, figuras, presentacion, guion, push.
cd ~/reid-utec
export PATH=$HOME/.local/bin:/usr/local/bin:$PATH PYTHONUNBUFFERED=1
while [ ! -f runs/NEXT_DONE ]; do sleep 120; done
(git fetch -q origin && git reset -q --hard origin/main) || echo "sin acceso a GitHub: uso la copia local"
uv run python finalize.py
echo FINALIZE_EXIT=$? > runs/FINALIZE_DONE
