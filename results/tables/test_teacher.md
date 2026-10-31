| run | red | mAP | R1 | R5 | R10 | mAP (LR) | R1 (LR) | R5 (LR) | R10 (LR) | caida mAP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ViT sin entrenar | maestro | 0.6 | 0.7 | 3.7 | 6.5 | 0.6 | 0.5 | 3.2 | 5.8 | 0.0 |
| DINO v1 (baseline) | maestro | 3.0 | 8.2 | 19.2 | 25.9 | 1.9 | 3.7 | 10.9 | 16.0 | 1.1 |
| Paper (prior + L_patch) | maestro | 2.1 | 5.9 | 13.2 | 18.6 | 1.7 | 3.1 | 8.8 | 13.4 | 0.4 |
| Paper + Sinkhorn-Knopp + KoLeo | maestro | 1.6 | 4.0 | 12.0 | 17.1 | 1.5 | 2.7 | 8.3 | 12.1 | 0.1 |
| Paper, prior suave (mascara 30 %, sin x1/4) | maestro | 1.6 | 3.5 | 10.1 | 15.0 | 0.7 | 0.7 | 2.8 | 5.5 | 0.9 |
