"""Extraccion de embeddings (L2) y evaluacion con coseno en las dos condiciones: original y baja resolucion."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from reid.data import load_eval_split, normalize
from reid.metrics import evaluate
from reid.model import VIT, ViT

CONDITIONS = {"original": "query_images", "low_res": "query_images_low_resolution"}


@torch.no_grad()
def extract(model: ViT, images, device: torch.device, batch: int = 256) -> dict[str, torch.Tensor]:
    """Embeddings normalizados: token [CLS] y [CLS] concatenado con el promedio de patches (eq. 4 del paper)."""
    model.eval()
    cls, patch = [], []
    for i in range(0, len(images), batch):
        tokens = model(normalize(torch.from_numpy(images[i:i + batch]).to(device)))
        cls.append(tokens[:, 0].float())
        patch.append(tokens[:, 1:].mean(dim=1).float())
    cls, patch = F.normalize(torch.cat(cls), dim=1), F.normalize(torch.cat(patch), dim=1)
    return {"cls": cls, "cls+patch": F.normalize(torch.cat([cls, patch], dim=1), dim=1)}


def run(model: ViT, split: dict, device: torch.device) -> dict[str, dict[str, float]]:
    gallery = extract(model, split["gallery_images"], device)
    results = {}
    for condition, key in CONDITIONS.items():
        query = extract(model, split[key], device)
        for feat in query:
            results[f"{condition}/{feat}"] = evaluate(
                query[feat], gallery[feat], split["query_person_ids"], split["query_camera_ids"],
                split["gallery_person_ids"], split["gallery_camera_ids"])
    return results


def report(results: dict[str, dict[str, float]], prefix: str = "") -> None:
    for key, metrics in results.items():
        print(f"{prefix}{key:22s}" + "".join(f"  {k}={100 * v:5.1f}" for k, v in metrics.items()))
    drop = results["original/cls"]["mAP"] - results["low_res/cls"]["mAP"]
    print(f"{prefix}caida de mAP (cls) por baja resolucion: {100 * drop:.1f} puntos")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt")
    parser.add_argument("--split", default="validation.h5")
    args = parser.parse_args()
    device = torch.device("cuda")
    ckpt = torch.load(args.ckpt, map_location=device)
    split = load_eval_split(args.split)
    results = {}
    for net in ("teacher", "student"):
        model = ViT(**VIT[ckpt["args"]["model"]]).to(device)
        model.load_state_dict(ckpt[net])
        results[net] = run(model, split, device)
        report(results[net], prefix=f"{net:8s} ")
    out = Path(args.ckpt).with_name(f"eval_{Path(args.split).stem}.json")
    out.write_text(json.dumps(results, indent=2))
    print("guardado en", out)


if __name__ == "__main__":
    main()
