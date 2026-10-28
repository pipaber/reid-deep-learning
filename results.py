"""Resume runs/*/ en una tabla markdown (test, maestro y estudiante) y grafica las curvas de validacion."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

RUNS = Path("runs")
COLS = ["mAP", "R1", "R5", "R10"]


def main() -> None:
    lines = ["| run | red | condicion | " + " | ".join(COLS) + " | caida mAP |", "|" + "---|" * (len(COLS) + 4)]
    fig, ax = plt.subplots(figsize=(6, 4))
    for run in sorted(RUNS.iterdir()):
        test = run / "eval_testing.json"
        if test.exists():
            results = json.loads(test.read_text())
            for net in ("teacher", "student"):
                drop = results[net]["original/cls"]["mAP"] - results[net]["low_res/cls"]["mAP"]
                for cond in ("original", "low_res"):
                    m = results[net][f"{cond}/cls"]
                    lines.append(f"| {run.name} | {net} | {cond} | " + " | ".join(f"{100 * m[c]:.1f}" for c in COLS)
                                 + f" | {100 * drop:.1f} |")
        log = run / "log.jsonl"
        if log.exists():
            rows = [json.loads(l) for l in log.read_text().splitlines()]
            ax.plot([r["epoch"] for r in rows], [100 * r["original/cls"]["mAP"] for r in rows], marker="o", label=run.name)
    ax.set(xlabel="epoca", ylabel="mAP validacion (%)", title="Validacion (maestro, [CLS], consultas originales)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RUNS / "val_curves.png", dpi=120)
    (RUNS / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
