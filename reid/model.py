"""ViT con token de mascara (estilo iBOT) y cabezal de proyeccion DINO."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from reid.data import SIZE

VIT = {"tiny": dict(dim=192, heads=3), "small": dict(dim=384, heads=6)}


class ViT(nn.Module):
    def __init__(self, dim: int, heads: int, depth: int = 12, patch: int = 8):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)
        self.num_patches = (SIZE[0] // patch) * (SIZE[1] // patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, dim))
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, activation="gelu",
                                       batch_first=True, norm_first=True)
            for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        for p in (self.cls_token, self.mask_token, self.pos_embed):
            nn.init.trunc_normal_(p, std=0.02)

    def embed(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Patches + token de mascara (si hay) + [CLS] + posicion: (B, 1+N, D)."""
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        if mask is not None:
            x = torch.where(mask[..., None], self.mask_token.to(x.dtype), x)
        return torch.cat([self.cls_token.expand(len(x), -1, -1), x], dim=1) + self.pos_embed

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B,3,H,W), mask: (B,N) bool. Devuelve tokens (B, 1+N, D): [CLS] seguido de los patches."""
        x = self.embed(x, mask)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def cls_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Atencion del [CLS] a los patches en el ultimo bloque, por cabeza (B, heads, N): softmax(q_cls K^T / sqrt(d))."""
        x = self.embed(x)
        for block in self.blocks[:-1]:
            x = block(x)
        attn = self.blocks[-1].self_attn
        q, k, _ = F.linear(self.blocks[-1].norm1(x), attn.in_proj_weight, attn.in_proj_bias).chunk(3, dim=-1)
        q = q.view(len(x), -1, attn.num_heads, attn.head_dim).transpose(1, 2)  # (B, heads, 1+N, d)
        k = k.view(len(x), -1, attn.num_heads, attn.head_dim).transpose(1, 2)
        scores = q[:, :, :1] @ k.transpose(-1, -2) / attn.head_dim**0.5  # fila del [CLS]
        return scores.softmax(dim=-1)[:, :, 0, 1:]


class Head(nn.Module):
    """Proyector f: MLP de 3 capas -> normalizacion L2 -> prototipos (capa lineal con pesos normalizados).

    El mismo modulo se aplica al token [CLS] y a los patches (pesos compartidos, como en el paper).
    """

    def __init__(self, dim: int, out_dim: int, hidden: int = 2048, bottleneck: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, bottleneck))
        self.prototypes = nn.Parameter(torch.randn(out_dim, bottleneck) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(self.mlp(x), dim=-1)
        return F.linear(x, F.normalize(self.prototypes, dim=-1))
