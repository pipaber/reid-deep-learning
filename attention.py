"""Mapas de atencion del token [CLS] en el ultimo bloque (como la Fig. 6 del paper y la Fig. 3 de DINO)."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from reid.data import SIZE, load_eval_split, normalize
from reid.model import VIT, ViT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt")
    parser.add_argument("--split", default="testing.h5")
    parser.add_argument("--net", default="teacher")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="results/figures/attention.png")
    args = parser.parse_args()
    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model = ViT(**VIT[ckpt["args"]["model"]]).to(device).eval()
    model.load_state_dict(ckpt[args.net])
    images = load_eval_split(args.split)["query_images"]
    idx = np.random.default_rng(0).choice(len(images), args.n, replace=False)
    x = torch.from_numpy(images[idx]).to(device)
    with torch.no_grad():
        attn = model.cls_attention(normalize(x))  # (n, heads, N)
    grid = SIZE[0] // 8, SIZE[1] // 8
    maps = F.interpolate(attn.view(args.n, -1, *grid), SIZE, mode="bilinear", align_corners=False).cpu().numpy()

    heads = maps.shape[1]
    fig, axes = plt.subplots(args.n, heads + 2, figsize=(1.3 * (heads + 2), 2.4 * args.n))
    titles = ["imagen", "media"] + [f"cabeza {h + 1}" for h in range(heads)]
    for i in range(args.n):
        img = images[idx[i]].transpose(1, 2, 0)
        panels = [None, maps[i].mean(0)] + list(maps[i])
        for ax, title, m in zip(axes[i], titles, panels):
            ax.imshow(img)
            if m is not None:
                ax.imshow(m, cmap="inferno", alpha=0.6)
            ax.axis("off")
            if i == 0:
                ax.set_title(title, fontsize=8)
    fig.suptitle(f"Atencion del [CLS] en el ultimo bloque ({args.net})", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print("guardado en", args.out)


if __name__ == "__main__":
    main()
