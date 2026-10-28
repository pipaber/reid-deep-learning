"""Perdidas de auto-destilacion: DINO (centering) o DINOv2 (Sinkhorn-Knopp) y regularizador KoLeo."""

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def sinkhorn_knopp(scores: torch.Tensor, n_iters: int = 3) -> torch.Tensor:
    """Asignacion doblemente estocastica (DINOv2/SwAV): filas = muestras, columnas = prototipos."""
    q = (scores - scores.max()).exp_().T  # (K, B)
    q /= q.sum()
    k, b = q.shape
    for _ in range(n_iters):
        q /= q.sum(dim=1, keepdim=True)
        q /= k
        q /= q.sum(dim=0, keepdim=True)
        q /= b
    return (q * b).T


class DistillLoss(nn.Module):
    """Cross-entropy entre la distribucion del maestro (afilada) y la del estudiante."""

    def __init__(self, out_dim: int, centering: str = "ema", student_temp: float = 0.1, momentum: float = 0.9):
        super().__init__()
        self.centering, self.student_temp, self.momentum = centering, student_temp, momentum
        self.register_buffer("center_cls", torch.zeros(1, out_dim))
        self.register_buffer("center_patch", torch.zeros(1, 1, out_dim))

    @torch.no_grad()
    def teacher_probs(self, logits: torch.Tensor, temp: float, center: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        if self.centering == "sinkhorn":
            return sinkhorn_knopp(logits.reshape(-1, logits.shape[-1]) / temp).view_as(logits)
        probs = F.softmax((logits - center) / temp, dim=-1)
        batch_mean = logits.mean(dim=tuple(range(logits.dim() - 1)), keepdim=True)
        center.mul_(self.momentum).add_(batch_mean, alpha=1 - self.momentum)
        return probs

    def forward(self, student_logits: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(student_logits.float() / self.student_temp, dim=-1)
        return -(teacher_probs * log_p).sum(dim=-1).mean()


def koleo(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Kozachenko-Leonenko: maximiza la entropia diferencial alejando cada vector de su vecino mas cercano."""
    x = F.normalize(x.float(), dim=-1)
    dist = torch.cdist(x, x).masked_fill(torch.eye(len(x), dtype=torch.bool, device=x.device), float("inf"))
    return -torch.log(dist.min(dim=1).values + eps).mean()
