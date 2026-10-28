"""Pre-entrenamiento auto-supervisado: ReID-oriented Prior + destilacion maestro-estudiante de [CLS] y patches.

Flags de ablacion: --no-prior (sin LQ/RM), --no-patch-loss (baseline DINO v1), --centering sinkhorn y --koleo (DINOv2).
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from evaluate import report, run as run_eval
from reid.data import TwoViews, load_eval_split, load_train_images
from reid.loss import DistillLoss, koleo
from reid.model import VIT, Head, ViT
from reid.prior import low_quality, random_mask


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="carpeta de salida en runs/")
    p.add_argument("--model", choices=VIT, default="small")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4, help="se escala por batch/256 como en DINO")
    p.add_argument("--warmup", type=int, default=10, help="epocas de calentamiento del lr")
    p.add_argument("--out-dim", type=int, default=4096, help="numero de prototipos K")
    p.add_argument("--mask-ratio", type=float, default=0.6)
    p.add_argument("--downsample-p", type=float, default=0.5, help="prob. de reducir x4 la resolucion en LQ")
    p.add_argument("--lambda-cls", type=float, default=0.55, help="peso lambda del loss de [CLS] (paper: 0.55)")
    p.add_argument("--no-prior", action="store_true", help="ablacion: estudiante sin LQ ni mascara")
    p.add_argument("--no-patch-loss", action="store_true", help="baseline DINO v1: solo loss de [CLS]")
    p.add_argument("--centering", choices=["ema", "sinkhorn"], default="ema", help="DINO v1 vs DINOv2")
    p.add_argument("--koleo", type=float, default=0.0, help="peso del regularizador KoLeo (DINOv2 usa 0.1)")
    p.add_argument("--extra-data", action="store_true", help="agrega las imagenes de validacion al entrenamiento")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="usa solo N imagenes (smoke test)")
    return p.parse_args()


def cosine(start: float, end: float, step: int, total: int) -> float:
    return end + (start - end) * (1 + math.cos(math.pi * step / total)) / 2


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    out = Path("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    images = load_train_images("training.h5", "validation.h5" if args.extra_data else None)
    images = images[: args.limit] if args.limit else images
    loader = DataLoader(TwoViews(images), args.batch, shuffle=True, drop_last=True, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0)
    val = load_eval_split("validation.h5")

    student = ViT(**VIT[args.model]).to(device)
    head_s = Head(VIT[args.model]["dim"], args.out_dim).to(device)
    teacher, head_t = copy.deepcopy(student), copy.deepcopy(head_s)  # el maestro se actualiza por EMA
    params = [*student.parameters(), *head_s.parameters()]
    teacher_params = [*teacher.parameters(), *head_t.parameters()]
    for p in teacher_params:
        p.requires_grad = False
    criterion = DistillLoss(args.out_dim, args.centering).to(device)
    base_lr = args.lr * args.batch / 256
    opt = torch.optim.AdamW(params, lr=base_lr, weight_decay=0.04)
    amp = torch.cuda.get_device_capability()[0] >= 8  # bf16 solo en Ampere o superior; fp32 en el resto
    total, warmup, step = args.epochs * len(loader), args.warmup * len(loader), 0
    print(f"{len(images)} imagenes, {len(loader)} pasos/epoca, {sum(p.numel() for p in student.parameters()) / 1e6:.1f}M params")

    for epoch in range(args.epochs):
        student.train()
        t0, losses = time.time(), []
        teacher_temp = 0.04 + 0.03 * min(1.0, epoch / max(1, 0.3 * args.epochs))  # DINO: 0.04 -> 0.07
        for m, n in loader:
            lr = base_lr * step / warmup if step < warmup else cosine(base_lr, 1e-6, step - warmup, total - warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            views = torch.cat([m, n]).to(device, non_blocking=True)  # (2B,3,H,W): vistas originales m_O, n_O
            b = len(m)
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=amp):
                t_tokens = teacher(views)
                t_cls = criterion.teacher_probs(head_t(t_tokens[:, 0]), teacher_temp, criterion.center_cls)
                t_patch = criterion.teacher_probs(head_t(t_tokens[:, 1:]), teacher_temp, criterion.center_patch)
            with torch.autocast("cuda", torch.bfloat16, enabled=amp):
                if args.no_prior:
                    s_in, mask = views, None
                else:  # ReID-oriented Prior: (m_L, m_R), (n_L, n_R)
                    s_in = low_quality(views, args.downsample_p)
                    mask = random_mask(2 * b, student.num_patches, args.mask_ratio, device)
                s_tokens = student(s_in, mask)
                s_cls, s_patch = head_s(s_tokens[:, 0]), head_s(s_tokens[:, 1:])
            # L_cls cruza las vistas (m_O con n_LR y n_O con m_LR); L_patch alinea patches de la misma vista
            loss = args.lambda_cls * criterion(s_cls, t_cls.roll(b, dims=0))
            if not args.no_patch_loss:
                loss = loss + criterion(s_patch, t_patch)
            if args.koleo:
                loss = loss + args.koleo * sum(koleo(c) for c in s_tokens[:, 0].chunk(2)) / 2

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 3.0)
            if epoch == 0:  # DINO: congela los prototipos en la primera epoca para estabilizar
                head_s.prototypes.grad = None
            opt.step()
            with torch.no_grad():  # EMA del maestro: theta_t <- m * theta_t + (1 - m) * theta_s, m de 0.996 a 1
                momentum = cosine(0.996, 1.0, step, total)
                for p_t, p_s in zip(teacher_params, params):
                    p_t.lerp_(p_s, 1 - momentum)
            step += 1
            losses.append(loss.item())
        print(f"epoca {epoch + 1}/{args.epochs}  loss {sum(losses) / len(losses):.4f}  lr {lr:.2e}  {time.time() - t0:.0f}s")

        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            results = run_eval(teacher, val, device)
            report(results, prefix="  val/teacher ")
            with (out / "log.jsonl").open("a") as f:
                f.write(json.dumps({"epoch": epoch + 1, "loss": sum(losses) / len(losses), **results}) + "\n")
            torch.save({"args": vars(args), "student": student.state_dict(), "teacher": teacher.state_dict(),
                        "head_student": head_s.state_dict(), "head_teacher": head_t.state_dict()}, out / "last.pt")


if __name__ == "__main__":
    main()
