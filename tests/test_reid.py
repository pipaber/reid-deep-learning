import numpy as np
import torch

from reid.loss import koleo, sinkhorn_knopp
from reid.metrics import evaluate
from reid.model import VIT, Head, ViT
from reid.prior import low_quality, random_mask


def test_metrics_market_protocol():
    # consulta 0 (pid 1, cam 1): galeria ordenada por similitud -> [junk, mismo pid misma cam, pid 2, pid 1, pid 1]
    q = torch.eye(5)[:1]
    g = torch.tensor([[0.9, 0, 0, 0, 0], [0.8, 0, 0, 0, 0], [0.7, 0, 0, 0, 0], [0.6, 0, 0, 0, 0], [0.5, 0, 0, 0, 0]])
    g_pid, g_cam = np.array([-1, 1, 2, 1, 1]), np.array([1, 1, 2, 2, 3])
    m = evaluate(q, g, np.array([1]), np.array([1]), g_pid, g_cam)
    # tras filtrar quedan [pid2, pid1, pid1]: AP = mean(1/2, 2/3), R1 = 0, R5 = 1
    assert abs(m["mAP"] - (0.5 + 2 / 3) / 2) < 1e-6 and m["R1"] == 0 and m["R5"] == 1


def test_sinkhorn_is_doubly_stochastic():
    q = sinkhorn_knopp(torch.randn(64, 16))
    assert torch.allclose(q.sum(1), torch.ones(64), atol=1e-4)
    assert torch.allclose(q.sum(0), torch.full((16,), 4.0), atol=1e-2)


def test_prior_shapes():
    x = torch.randn(4, 3, 128, 64)
    assert low_quality(x).shape == x.shape
    mask = random_mask(4, 128, 0.6, x.device)
    assert mask.shape == (4, 128) and (mask.sum(1) == 76).all()


def test_model_and_head():
    vit, head = ViT(**VIT["tiny"], depth=2), Head(192, 32)
    tokens = vit(torch.randn(2, 3, 128, 64), random_mask(2, 128, 0.6, "cpu"))
    assert tokens.shape == (2, 129, 192) and head(tokens).shape == (2, 129, 32)
    assert koleo(tokens[:, 0]).ndim == 0
