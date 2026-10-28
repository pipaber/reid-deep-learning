"""ReID-oriented Prior: transformacion de baja calidad (LQ) y mascara aleatoria (RM) del paper."""

import math

import torch
import torch.nn.functional as F

from reid.data import SIZE


def gaussian_blur(x: torch.Tensor, sigma: float = 1.0, k: int = 5) -> torch.Tensor:
    """Q_GB: convolucion con kernel gaussiano separable (sigma=1 segun el paper)."""
    g = torch.exp(-(torch.arange(k, device=x.device) - k // 2) ** 2 / (2 * sigma**2))
    g = (g / g.sum()).to(x.dtype)
    x = F.conv2d(x, g.view(1, 1, k, 1).repeat(3, 1, 1, 1), padding=(k // 2, 0), groups=3)
    return F.conv2d(x, g.view(1, 1, 1, k).repeat(3, 1, 1, 1), padding=(0, k // 2), groups=3)


def motion_blur(x: torch.Tensor, length: int = 3) -> torch.Tensor:
    """Q_MB: kernel de linea de longitud L con angulo aleatorio por imagen (L=3 segun el paper)."""
    b = x.shape[0]
    theta = torch.rand(b, 1, device=x.device) * 2 * math.pi
    steps = torch.linspace(-(length - 1) / 2, (length - 1) / 2, length, device=x.device)
    c = length // 2
    px = (c + steps * torch.cos(theta)).round().long().clamp(0, length - 1)
    py = (c + steps * torch.sin(theta)).round().long().clamp(0, length - 1)
    kernel = torch.zeros(b, length * length, device=x.device, dtype=x.dtype)
    kernel.scatter_(1, py * length + px, 1.0)
    kernel = kernel / kernel.sum(1, keepdim=True)
    kernel = kernel.view(b, 1, 1, length, length).expand(b, 3, 1, length, length).reshape(b * 3, 1, length, length)
    out = F.conv2d(x.reshape(1, b * 3, *x.shape[-2:]), kernel, padding=c, groups=b * 3)
    return out.view_as(x)


def downsample(x: torch.Tensor, factor: int = 4, p: float = 0.5) -> torch.Tensor:
    """Adaptacion al lab: simula las consultas de baja resolucion (x1/4 bicubico y vuelta a SIZE)."""
    low = F.interpolate(x, scale_factor=1 / factor, mode="bicubic", align_corners=False)
    low = F.interpolate(low, SIZE, mode="bicubic", align_corners=False)
    keep = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < p
    return torch.where(keep, low, x)


def low_quality(x: torch.Tensor, p_down: float = 0.5) -> torch.Tensor:
    """Q(): LQ = Gaussian blur + motion blur (mejor combinacion en la Tabla 4) + baja resolucion."""
    return downsample(motion_blur(gaussian_blur(x)), p=p_down)


def random_mask(b: int, n: int, ratio: float, device: torch.device) -> torch.Tensor:
    """M(): mascara booleana (B,N) con exactamente ratio*N tokens enmascarados por imagen."""
    n_mask = int(ratio * n)
    order = torch.rand(b, n, device=device).argsort(1)
    mask = torch.zeros(b, n, dtype=torch.bool, device=device)
    return mask.scatter_(1, order[:, :n_mask], True)
