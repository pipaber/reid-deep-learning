| run | red | mAP | R1 | R5 | R10 | mAP (LR) | R1 (LR) | R5 (LR) | R10 (LR) | caida mAP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ViT sin entrenar | maestro | 0.6 | 0.7 | 3.7 | 6.5 | 0.6 | 0.5 | 3.2 | 5.8 | 0.0 |
| ViT sin entrenar | estudiante | 0.6 | 0.7 | 3.7 | 6.5 | 0.6 | 0.5 | 3.2 | 5.8 | 0.0 |
| DINO v1 (baseline) | maestro | 3.0 | 8.2 | 19.2 | 25.9 | 1.9 | 3.7 | 10.9 | 16.0 | 1.1 |
| DINO v1 (baseline) | estudiante | 3.4 | 9.2 | 20.5 | 27.8 | 2.1 | 4.2 | 11.9 | 16.4 | 1.3 |
| Paper (prior + L_patch) | maestro | 2.1 | 5.9 | 13.2 | 18.6 | 1.7 | 3.1 | 8.8 | 13.4 | 0.4 |
| Paper (prior + L_patch) | estudiante | 1.7 | 4.7 | 11.2 | 15.9 | 1.5 | 2.6 | 7.8 | 11.6 | 0.2 |
