# Person Re-ID auto-supervisado (Laboratorio 3)

Implementación en PyTorch de *Learn from restoration: exploiting task-oriented knowledge distillation in
self-supervised person re-identification* (Yang et al., The Visual Computer 2025) adaptada a Market-1501
(`training.h5`, `validation.h5`, `testing.h5`), más una ablación con las mejoras de DINOv2.

## Método

El paper es DINO v1 con tres cambios, todos en `reid/`:

| Componente | Paper | Archivo |
|---|---|---|
| ReID-oriented Prior: LQ = blur gaussiano (σ=1) + motion blur (L=3); RM = 60 % de tokens enmascarados | §3.1 | `reid/prior.py` |
| ViT con token de máscara, proyector de 3 capas + L2 + prototipos, **compartido** entre [CLS] y patches | §4.1.2 | `reid/model.py` |
| `L = L_patch (misma vista) + λ·L_cls (vista cruzada)`, λ = 0.55, maestro EMA con centering + sharpening | eq. (3) | `reid/loss.py`, `train.py` |

Adaptaciones al lab (documentadas en el código):

- Entrada a resolución nativa 128×64 con patch 8 (128 tokens); ViT-Small (384-d) por defecto, `--model tiny` para pruebas.
- LQ incluye además una reducción ×4 bicúbica y vuelta a 128×64 con p = 0.5, porque la evaluación usa consultas a 1/4 de resolución.
- Sin LUPerson ni fine-tuning supervisado: el embedding de evaluación es el token [CLS] del backbone (y también [CLS] ⊕ media de patches, eq. 4) normalizado L2, similitud coseno.
- Loss de patches sobre todos los tokens (eq. 1), no solo los enmascarados como en iBOT, porque los visibles también están degradados por LQ.
- Solo dos vistas globales (m, n), sin multi-crop.

## Instalación y uso

```bash
uv sync
uv run pytest
```

Entrenar (guarda `runs/<name>/last.pt`, `log.jsonl` con la validación cada `--eval-every` épocas):

```bash
# Método del paper
uv run python train.py --name paper --epochs 100 --batch 128

# Baseline: DINO v1 vanilla (sin prior, solo loss de [CLS])
uv run python train.py --name dino --epochs 100 --batch 128 --no-prior --no-patch-loss --lambda-cls 1

# Ablación DINOv2: Sinkhorn-Knopp en vez de centering + regularizador KoLeo
uv run python train.py --name paper_dinov2 --epochs 100 --batch 128 --centering sinkhorn --koleo 0.1

# Ablaciones opcionales del propio paper
uv run python train.py --name paper_noprior --epochs 100 --batch 128 --no-prior
```

Evaluar en ambas condiciones (consultas originales y de baja resolución), maestro y estudiante:

```bash
uv run python evaluate.py runs/paper/last.pt --split validation.h5
uv run python evaluate.py runs/paper/last.pt --split testing.h5
```

Figura de aciertos y fallos (top-5 por consulta):

```bash
uv run python visualize.py runs/paper/last.pt --split testing.h5 --out retrieval.png
```

Smoke test local (GPU pequeña):

```bash
uv run python train.py --name smoke --model tiny --epochs 2 --batch 32 --limit 320 --eval-every 1 --workers 0
```

## Estructura

```
reid/data.py     carga de H5, normalización, dataset de dos vistas (augs de DINO)
reid/prior.py    ReID-oriented Prior: gaussian_blur, motion_blur, downsample, random_mask
reid/model.py    ViT (token de máscara) y Head (proyector DINO)
reid/loss.py     DistillLoss (centering EMA | Sinkhorn-Knopp), koleo
reid/metrics.py  mAP y Rank-k con el protocolo Market-1501
train.py         bucle de pre-entrenamiento maestro-estudiante
evaluate.py      extracción de embeddings y métricas en las dos condiciones
visualize.py     figura de aciertos y fallos
```
