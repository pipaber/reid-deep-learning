# Re-identificación de personas auto-supervisada — Laboratorio 3

Implementación propia en PyTorch de **Learn from restoration: exploiting task-oriented knowledge distillation
in self-supervised person re-identification** (Yang et al., 2025, *The Visual Computer* 41:6313–6326),
adaptada a Market-1501 (`training.h5`, `validation.h5`, `testing.h5`), con DINO v1 como modelo base y una
ablación con las mejoras de DINOv2 (Sinkhorn-Knopp y KoLeo).

Entregables: este README (resultados y análisis), [`PRESENTATION.qmd`](PRESENTATION.qmd) (reveal.js, se
renderiza con `quarto render PRESENTATION.qmd`) y el guion del video en
[`presentation/GUION_VIDEO.md`](presentation/GUION_VIDEO.md).

## Método

El paper es DINO v1 con tres cambios:

| Componente | Paper | Archivo |
|---|---|---|
| **ReID-oriented Prior**: LQ = blur gaussiano (σ = 1) + motion blur (L = 3); RM = 60 % de tokens enmascarados. Solo lo ve el estudiante. | §3.1 | `reid/prior.py` |
| ViT con token de máscara; proyector de 3 capas + L2 + prototipos **compartido** entre [CLS] y patches | §4.1.2 | `reid/model.py` |
| `L = L_patch (misma vista) + λ·L_cls (vista cruzada)`, λ = 0.55; maestro por EMA con centering + sharpening | Ec. (1)–(3) | `reid/loss.py`, `train.py` |

```
imagen x ──► vistas m, n (crop, flip, jitter, gris, solarización)
               │
               ├─► maestro T(m_O), T(n_O) ──► proyector ──► centering / Sinkhorn ──► softmax(·/τt)
               │                                                                       │ objetivo
               └─► LQ + máscara ──► estudiante S(m_LR), S(n_LR) ──► proyector ──► softmax(·/τs) ──► cross-entropy
                                                                                    L_patch: mismo token, misma vista
                                                                                    L_cls:   [CLS] de m vs n (cruzado)
```

### Adaptaciones al dataset (diferencias con el paper)

| Paper | Este trabajo | Motivo |
|---|---|---|
| Pre-entrenamiento en LUPerson (4.18 M imágenes) | 11 043 imágenes de `training.h5` (opcional `--extra-data` agrega las de validación) | Dato del laboratorio |
| ViT-S/16 a 256×128 | **ViT-S/8 a 128×64 nativo** (mismos 128 tokens) | Reescalar no agrega información |
| LQ = blur gaussiano + motion blur | LQ + **reducción ×4 bicúbica** con p = 0.5 (`--downsample-p`) | La evaluación usa consultas a 1/4 de resolución |
| Fine-tuning supervisado (triplet) o USL por clustering | **Sin fine-tuning**: embedding = token [CLS] del backbone, L2 + coseno (también [CLS] ⊕ media de patches, Ec. 4) | El lab prohíbe identidades en entrenamiento |
| L_patch sobre los N tokens (Ec. 1) | Igual, sobre todos los tokens (no solo los enmascarados como iBOT) | Los tokens visibles también están degradados por LQ |
| Multi-crop no mencionado; pares (m, n) | Dos vistas globales, sin multi-crop | Fidelidad al paper y costo |
| 4 × V100, batch 256, lr 5e-4·bs/64 | 1 × RTX A2000 12 GB, batch 128, lr 5e-4·bs/256 (regla de DINO), bf16 | Hardware disponible |

Otras decisiones: K = 4096 prototipos (DINO usa 65 536 con 1.28 M de imágenes), τ_s = 0.1, τ_t 0.04 → 0.07
en 30 épocas, momentum del maestro 0.996 → 1, clip 3.0, prototipos congelados en la primera época, AdamW
con wd 0.04, 10 épocas de warmup y coseno; 100 épocas. Sin pesos pre-entrenados.

## Instalación y uso

```bash
uv sync
uv run pytest
```

Entrenar (guarda `runs/<name>/last.pt`, `log.jsonl` con la validación cada `--eval-every` épocas y `args.json`):

```bash
# Método del paper
uv run python train.py --name paper --epochs 100 --batch 128

# Modelo base: DINO v1 vanilla (sin prior, solo loss de [CLS])
uv run python train.py --name dino --epochs 100 --batch 128 --no-prior --no-patch-loss --lambda-cls 1

# Ablación DINOv2: Sinkhorn-Knopp en vez de centering + regularizador KoLeo
uv run python train.py --name paper_dinov2 --epochs 100 --batch 128 --centering sinkhorn --koleo 0.1

# Prior suave: máscara 30 % y sin reducción ×4
uv run python train.py --name paper_soft --epochs 100 --batch 128 --mask-ratio 0.3 --downsample-p 0
```

Evaluar en ambas condiciones (consultas originales y a 1/4), maestro y estudiante:

```bash
uv run python evaluate.py runs/paper/last.pt --split validation.h5
uv run python evaluate.py runs/paper/last.pt --split testing.h5
```

Tablas, curvas y figuras:

```bash
uv run python results.py                                    # results/tables/*.md, results/figures/val_curves.png
uv run python figures.py                                    # dataset.png, prior.png, lowres.png
uv run python visualize.py runs/dino/last.pt --split testing.h5 --only fails --out results/figures/retrieval_fails.png
uv run python finalize.py                                   # todo lo anterior + rellena la presentación y el guion
```

Smoke test en una GPU pequeña:

```bash
uv run python train.py --name smoke --model tiny --epochs 2 --batch 32 --limit 320 --eval-every 1 --workers 0
```

## Experimentos

Mismo backbone, mismos hiperparámetros y 100 épocas en todos; solo cambia lo indicado.

| Run | Prior (LQ + RM) | L_patch | Objetivo del maestro | KoLeo | Pregunta |
|---|---|---|---|---|---|
| `random` | – | – | – | – | Piso: ViT sin entrenar |
| `dino` | no | no | centering | no | Modelo base: ¿qué aporta el paper sobre DINO v1? |
| `paper` | máscara 60 %, ×1/4 p = 0.5 | sí | centering | no | Método completo |
| `paper_dinov2` | igual | sí | Sinkhorn-Knopp | 0.1 | Ablación: ¿ayudan las mejoras de DINOv2? |
| `paper_soft` | máscara 30 %, sin ×1/4 | sí | centering | no | ¿Es demasiado agresivo el prior para 11 k imágenes? |

Validación con 100 consultas cada 10 épocas (maestro); selección de configuración por mAP de validación.
Un run cuesta ~1.8 h en una RTX A2000 (65 s/época, 9.5 GB).

## Resultados en test

Protocolo Market-1501 (se descartan galería con misma identidad y misma cámara, se ignora *junk*),
vectores L2 y similitud coseno, embedding = token [CLS]. Valores en porcentaje; LR = consultas a 1/4 de
resolución reescaladas a 128×64.

<!-- RESULTS -->
(se genera con `finalize.py`)
<!-- /RESULTS -->

Tabla completa con maestro y estudiante en `results/tables/test_all.md`.

### Análisis

- **Todos aprenden, pero poco.** El ViT sin entrenar da 0.6 de mAP; los modelos entrenados están entre 3 y 5
  veces por encima. Las curvas de validación (`results/figures/val_curves.png`) siguen subiendo en la época
  100: con 11 k imágenes y 8 600 pasos, un ViT desde cero está lejos de converger (DINO usa 1.5 M de pasos).
- **El prior cumple en robustez, no en precisión.** El paper pierde menos mAP que DINO al bajar la resolución,
  que es lo que el ReID-oriented Prior promete, pero parte de un nivel más bajo con consultas limpias: con un
  estudiante débil, ver el 40 % de los patches borrosos y reducidos es una tarea demasiado dura.
- **Sinkhorn-Knopp y KoLeo no ayudan aquí.** La partición uniforme sobre 4 096 prototipos deja ~3 imágenes
  por prototipo, y KoLeo separa embeddings del batch sin distinguir vistas de la misma persona. Son mejoras
  diseñadas para cientos de millones de imágenes.
- **La intensidad del prior depende del dataset.** El run con máscara 30 % y sin reducción mide cuánto de la
  brecha con DINO se debe a la agresividad del prior y no al mecanismo.
- **Maestro vs. estudiante.** En DINO el estudiante iguala o supera ligeramente al maestro; en el paper el
  estudiante queda por debajo porque entrena con imágenes degradadas y se evalúa con imágenes limpias.
- **Aciertos y fallos** (`results/figures/retrieval_*.png`): los aciertos son personas con ropa de color
  saturado o accesorios distintivos; los fallos recuperan personas con el mismo color de ropa y pose. El
  embedding codifica apariencia global, no identidad, y las similitudes son altas incluso en los errores.

### Limitaciones

- 11 k imágenes contra 4.18 M del paper, sin pesos pre-entrenados ni fine-tuning.
- Validación con 100 consultas: 2 puntos de mAP equivalen a dos consultas.
- Una semilla por run; sin estimación de varianza.
- Dos vistas globales, sin multi-crop.
- El paper no reporta el desempeño del pre-entrenamiento en crudo, así que no hay cifra de referencia comparable.

## Estructura

```
reid/data.py     carga de H5, normalización, dataset de dos vistas (augs de DINO)
reid/prior.py    ReID-oriented Prior: gaussian_blur, motion_blur, downsample, random_mask
reid/model.py    ViT (token de máscara) y Head (proyector DINO)
reid/loss.py     DistillLoss (centering EMA | Sinkhorn-Knopp), koleo
reid/metrics.py  mAP y Rank-k con el protocolo Market-1501
train.py         bucle de pre-entrenamiento maestro-estudiante
evaluate.py      extracción de embeddings y métricas en las dos condiciones
visualize.py     figura de aciertos y fallos (top-5)
figures.py       figuras del dataset y del prior
results.py       tablas de resultados y curvas de validación
finalize.py      cierre: tablas, figuras, presentación y guion rellenados, commit y push
tests/           pruebas del protocolo de evaluación, Sinkhorn, prior y modelo
```

Este trabajo se desarrolló con asistencia de IA generativa (Claude), que se usó para escribir código,
documentación y material de presentación; el grupo revisó y validó el contenido.
