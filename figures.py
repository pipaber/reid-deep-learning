"""Figuras del dataset y del ReID-oriented Prior para el reporte y la presentacion (results/figures/)."""

from pathlib import Path

import matplotlib.pyplot as plt
import h5py
import numpy as np
import torch

from reid.data import MEAN, STD, TwoViews, load_eval_split, load_train_images
from reid.prior import low_quality, random_mask

OUT = Path("results/figures")


def to_image(x: torch.Tensor) -> np.ndarray:
    """Tensor normalizado (3,H,W) -> imagen uint8 HWC."""
    x = x * torch.tensor(STD).view(3, 1, 1) + torch.tensor(MEAN).view(3, 1, 1)
    return (x.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()


def dataset_grid(images: np.ndarray, cams: np.ndarray, rng: np.random.Generator) -> None:
    idx = rng.choice(len(images), 16, replace=False)
    fig, axes = plt.subplots(2, 8, figsize=(10, 5.2))
    for ax, i in zip(axes.flat, idx):
        ax.imshow(images[i].transpose(1, 2, 0))
        ax.set_title(f"cam {cams[i]}", fontsize=9)
        ax.axis("off")
    fig.suptitle("training.h5: 11 043 imagenes de 128x64, 6 camaras, sin identidades", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "dataset.png", dpi=130)


def prior_pipeline(images: np.ndarray, rng: np.random.Generator) -> None:
    torch.manual_seed(0)
    ds = TwoViews(images)
    idx = rng.choice(len(images), 4, replace=False)
    m, n = zip(*(ds[i] for i in idx))
    m, n = torch.stack(m), torch.stack(n)
    lq = low_quality(m)
    mask = random_mask(4, 128, 0.6, torch.device("cpu"))
    masked = lq.clone()
    for b in range(4):
        for t in mask[b].nonzero().flatten():
            r, c = divmod(int(t), 8)
            masked[b, :, r * 8:(r + 1) * 8, c * 8:(c + 1) * 8] = 0.0  # gris medio tras desnormalizar
    cols = [("original", images[idx].transpose(0, 2, 3, 1)), ("vista m (maestro)", [to_image(x) for x in m]),
            ("vista n (maestro)", [to_image(x) for x in n]), ("m_L = LQ(m)", [to_image(x) for x in lq]),
            ("m_L + mascara 60 %", [to_image(x) for x in masked])]
    fig, axes = plt.subplots(4, len(cols), figsize=(9, 9.5))
    for j, (title, imgs) in enumerate(cols):
        axes[0, j].set_title(title, fontsize=9)
        for i in range(4):
            axes[i, j].imshow(imgs[i])
            axes[i, j].axis("off")
    fig.suptitle("ReID-oriented Prior: el maestro ve m y n; el estudiante ve m_L enmascarada", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "prior.png", dpi=130)


def low_res_queries(split: dict, rng: np.random.Generator) -> None:
    idx = rng.choice(len(split["query_images"]), 6, replace=False)
    fig, axes = plt.subplots(2, 6, figsize=(8, 5))
    for j, i in enumerate(idx):
        axes[0, j].imshow(split["query_images"][i].transpose(1, 2, 0))
        axes[1, j].imshow(split["query_images_low_resolution"][i].transpose(1, 2, 0), interpolation="nearest")
        for ax in axes[:, j]:
            ax.axis("off")
    axes[0, 0].set_title("consulta original 128x64", fontsize=9, loc="left")
    axes[1, 0].set_title("baja resolucion 32x16 (x1/4)", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "lowres.png", dpi=130)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    with h5py.File("training.h5") as f:
        cams = f["camera_ids"][:]
    images = load_train_images("training.h5")
    dataset_grid(images, cams, rng)
    prior_pipeline(images, rng)
    low_res_queries(load_eval_split("validation.h5"), rng)
    print("figuras en", OUT)


if __name__ == "__main__":
    main()
