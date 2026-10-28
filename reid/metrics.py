"""Protocolo de evaluacion Market-1501: mAP y CMC Rank-k con similitud coseno."""

import numpy as np
import torch


def evaluate(q_feat: torch.Tensor, g_feat: torch.Tensor, q_pid, q_cam, g_pid, g_cam,
             ranks=(1, 5, 10)) -> dict[str, float]:
    """Las galerias con misma persona y misma camara que la consulta se ignoran; pid -1 es junk."""
    sim = (q_feat @ g_feat.T).cpu().numpy()
    order = np.argsort(-sim, axis=1)
    aps, cmc = [], np.zeros(max(ranks))
    for i in range(len(q_pid)):
        pid, cam = g_pid[order[i]], g_cam[order[i]]
        keep = ~((pid == q_pid[i]) & (cam == q_cam[i])) & (pid != -1)
        hits = np.flatnonzero(pid[keep] == q_pid[i])
        if len(hits) == 0:
            continue
        aps.append(np.mean(np.arange(1, len(hits) + 1) / (hits + 1)))
        cmc[hits[0]:] += 1
    n = len(aps)
    return {"mAP": float(np.mean(aps)), **{f"R{k}": float(cmc[k - 1] / n) for k in ranks}}
