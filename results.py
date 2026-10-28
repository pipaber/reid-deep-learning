"""Resume runs/*/ en tablas markdown (results/tables/) y grafica las curvas de validacion (results/figures/)."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

RUNS, TABLES, FIGURES = Path("runs"), Path("results/tables"), Path("results/figures")
COLS = ["mAP", "R1", "R5", "R10"]
LABELS = {"random": "ViT sin entrenar", "dino": "DINO v1 (baseline)", "paper": "Paper (prior + L_patch)",
          "paper_dinov2": "Paper + Sinkhorn-Knopp + KoLeo", "paper_soft": "Paper, prior suave (mascara 30 %, sin x1/4)"}


def row(name: str, net: str, r: dict) -> str:
    o, l = r["original/cls"], r["low_res/cls"]
    cells = [f"{100 * o[c]:.1f}" for c in COLS] + [f"{100 * l[c]:.1f}" for c in COLS] + [f"{100 * (o['mAP'] - l['mAP']):.1f}"]
    return f"| {LABELS.get(name, name)} | {net} | " + " | ".join(cells) + " |"


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    header = ["| run | red | mAP | R1 | R5 | R10 | mAP (LR) | R1 (LR) | R5 (LR) | R10 (LR) | caida mAP |",
              "|---|---|" + "---:|" * 9]
    teacher, both = list(header), list(header)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for run in sorted(RUNS.iterdir(), key=lambda p: list(LABELS).index(p.name) if p.name in LABELS else 99):
        test = run / "eval_testing.json"
        if test.exists():
            results = json.loads(test.read_text())
            teacher.append(row(run.name, "maestro", results["teacher"]))
            for net, label in (("teacher", "maestro"), ("student", "estudiante")):
                both.append(row(run.name, label, results[net]))
        log = run / "log.jsonl"
        if log.exists():
            rows = [json.loads(line) for line in log.read_text().splitlines()]
            ax.plot([r["epoch"] for r in rows], [100 * r["original/cls"]["mAP"] for r in rows], marker="o",
                    label=LABELS.get(run.name, run.name))
    ax.set(xlabel="epoca", ylabel="mAP validacion (%)", title="Validacion: maestro, [CLS], consultas originales")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "val_curves.png", dpi=130)
    (TABLES / "test_teacher.md").write_text("\n".join(teacher) + "\n")
    (TABLES / "test_all.md").write_text("\n".join(both) + "\n")
    print("\n".join(both))


if __name__ == "__main__":
    main()
