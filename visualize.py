"""Figura con los top-5 de galeria para consultas acertadas y falladas (verde = misma persona, rojo = distinta)."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluate import extract
from reid.data import load_eval_split
from reid.model import VIT, ViT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt")
    parser.add_argument("--split", default="testing.h5")
    parser.add_argument("--net", default="teacher")
    parser.add_argument("--low-res", action="store_true", help="usa las consultas de baja resolucion")
    parser.add_argument("--n", type=int, default=5, help="consultas por grupo (aciertos / fallos)")
    parser.add_argument("--out", default="retrieval.png")
    args = parser.parse_args()
    device = torch.device("cuda")
    ckpt = torch.load(args.ckpt, map_location=device)
    model = ViT(**VIT[ckpt["args"]["model"]]).to(device)
    model.load_state_dict(ckpt[args.net])
    split = load_eval_split(args.split)
    queries = split["query_images_low_resolution" if args.low_res else "query_images"]
    sim = (extract(model, queries, device)["cls"] @ extract(model, split["gallery_images"], device)["cls"].T).cpu().numpy()
    q_pid, q_cam, g_pid, g_cam = (split[k] for k in ("query_person_ids", "query_camera_ids", "gallery_person_ids", "gallery_camera_ids"))

    rng = np.random.default_rng(0)
    rows = []  # (indice de consulta, top-5 indices de galeria, acierto en rank-1)
    for i in rng.permutation(len(q_pid)):
        order = np.argsort(-sim[i])
        order = order[~((g_pid[order] == q_pid[i]) & (g_cam[order] == q_cam[i])) & (g_pid[order] != -1)][:5]
        hit = g_pid[order[0]] == q_pid[i]
        if sum(r[2] == hit for r in rows) < args.n:
            rows.append((i, order, hit))
        if len(rows) == 2 * args.n:
            break
    rows.sort(key=lambda r: not r[2])

    fig, axes = plt.subplots(len(rows), 6, figsize=(7, 2.3 * len(rows)))
    for (i, order, hit), row in zip(rows, axes):
        row[0].imshow(queries[i].transpose(1, 2, 0))
        row[0].set_title(f"query pid {q_pid[i]} cam {q_cam[i]}\n{'ACIERTO' if hit else 'FALLO'}", fontsize=8)
        for j, ax in zip(order, row[1:]):
            ax.imshow(split["gallery_images"][j].transpose(1, 2, 0))
            ax.set_title(f"pid {g_pid[j]} cam {g_cam[j]}\nsim {sim[i, j]:.2f}", fontsize=8,
                         color="green" if g_pid[j] == q_pid[i] else "red")
        for ax in row:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print("guardado en", args.out)


if __name__ == "__main__":
    main()
