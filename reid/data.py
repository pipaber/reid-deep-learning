"""Carga de los H5 de Market-1501 y transformaciones de entrada."""

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import v2

SIZE = (128, 64)  # alto x ancho nativo de Market-1501
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def normalize(x: torch.Tensor) -> torch.Tensor:
    """uint8 (B,3,H,W) -> float normalizado en SIZE (las consultas de baja resolucion se reescalan)."""
    x = x.float() / 255
    if x.shape[-2:] != SIZE:
        x = F.interpolate(x, SIZE, mode="bicubic", align_corners=False).clamp(0, 1)
    return v2.functional.normalize(x, MEAN, STD)


def load_train_images(path: str, extra: str | None = None) -> np.ndarray:
    """Imagenes de entrenamiento sin etiquetas; `extra` agrega query+gallery de validacion."""
    with h5py.File(path) as f:
        images = [f["images"][:]]
    if extra:
        with h5py.File(extra) as f:
            images += [f["query/images"][:], f["gallery/images"][:]]
    return np.concatenate(images)


def load_eval_split(path: str) -> dict[str, np.ndarray]:
    keys = ["query/images", "query/images_low_resolution", "query/person_ids", "query/camera_ids",
            "gallery/images", "gallery/person_ids", "gallery/camera_ids"]
    with h5py.File(path) as f:
        return {k.replace("/", "_"): f[k][:] for k in keys}


class TwoViews(Dataset):
    """Devuelve dos vistas aumentadas (m, n) de la misma imagen, como en DINO."""

    def __init__(self, images: np.ndarray):
        self.images = torch.from_numpy(images)
        self.augment = v2.Compose([
            v2.RandomResizedCrop(SIZE, scale=(0.5, 1.0), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.RandomSolarize(threshold=128, p=0.2),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(MEAN, STD),
        ])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.images[i]
        return self.augment(x), self.augment(x)
