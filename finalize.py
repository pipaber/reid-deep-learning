"""Cierre del proyecto: tablas, figuras de recuperacion, rellena la presentacion y el guion, y hace commit + push.

Placeholders en presentation/*.template.*: @@<run>.<teacher|student>.<original|low_res>.<mAP|R1|R5|R10>@@,
@@<run>.<red>.drop@@, @@<run>.<red>.drop_rel@@ y @@<run>.val.mAP@@. Se reemplazan por porcentajes con un decimal.
"""

import json
import re
import subprocess
from pathlib import Path

RUNS = Path("runs")
TEMPLATES = {"presentation/PRESENTATION.template.qmd": "PRESENTATION.qmd",
             "presentation/GUION.template.md": "presentation/GUION_VIDEO.md"}


def sh(*cmd: str) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def values() -> dict[str, str]:
    out = {}
    for run in RUNS.iterdir():
        test, log = run / "eval_testing.json", run / "log.jsonl"
        if test.exists():
            results = json.loads(test.read_text())
            for net, r in results.items():
                for cond in ("original", "low_res"):
                    for metric, v in r[f"{cond}/cls"].items():
                        out[f"{run.name}.{net}.{cond}.{metric}"] = f"{100 * v:.1f}"
                drop = r["original/cls"]["mAP"] - r["low_res/cls"]["mAP"]
                out[f"{run.name}.{net}.drop"] = f"{100 * drop:.1f}"
                out[f"{run.name}.{net}.drop_rel"] = f"{100 * drop / r['original/cls']['mAP']:.0f}"
        if log.exists():
            last = json.loads(log.read_text().splitlines()[-1])
            out[f"{run.name}.val.mAP"] = f"{100 * last['original/cls']['mAP']:.1f}"
    return out


def fill(vals: dict[str, str]) -> None:
    for src, dst in TEMPLATES.items():
        text = Path(src).read_text(encoding="utf-8")
        missing = sorted(set(re.findall(r"@@([\w.]+)@@", text)) - set(vals))
        if missing:
            print(f"{src}: sin valor para {missing}")
        Path(dst).write_text(re.sub(r"@@([\w.]+)@@", lambda m: vals.get(m[1], m[0]), text), encoding="utf-8")


def best_run(vals: dict[str, str]) -> str:
    candidates = [r.name for r in RUNS.iterdir() if (r / "last.pt").exists() and f"{r.name}.teacher.original.mAP" in vals]
    return max(candidates, key=lambda n: float(vals[f"{n}.teacher.original.mAP"]))


def main() -> None:
    vals = values()
    best = best_run(vals)
    print("mejor run en test:", best)
    sh("uv", "run", "python", "results.py")
    sh("uv", "run", "python", "figures.py")
    ckpt = str(RUNS / best / "last.pt")
    sh("uv", "run", "python", "visualize.py", ckpt, "--split", "testing.h5", "--only", "hits", "--out", "results/figures/retrieval_hits.png")
    sh("uv", "run", "python", "visualize.py", ckpt, "--split", "testing.h5", "--only", "fails", "--out", "results/figures/retrieval_fails.png")
    sh("uv", "run", "python", "visualize.py", ckpt, "--split", "testing.h5", "--low-res", "--out", "results/figures/retrieval_lowres.png")
    sh("uv", "run", "python", "attention.py", ckpt, "--split", "testing.h5", "--out", "results/figures/attention.png")
    fill(vals)
    readme = Path("README.md").read_text(encoding="utf-8")
    table = Path("results/tables/test_teacher.md").read_text(encoding="utf-8")
    block = (f"<!-- RESULTS -->\n{table}\nFiguras de aciertos y fallos del mejor run (`{best}`) en `results/figures/`; "
             "curvas de validación en `results/figures/val_curves.png`.\n<!-- /RESULTS -->")
    Path("README.md").write_text(re.sub(r"<!-- RESULTS -->.*<!-- /RESULTS -->", block, readme, flags=re.S), encoding="utf-8")
    sh("quarto", "render", "PRESENTATION.qmd")
    sh("git", "add", "-A")
    sh("git", "commit", "-m", "Resultados finales, presentacion y guion\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>")
    try:
        sh("git", "push", "origin", "main")
    except subprocess.CalledProcessError:
        print("push fallido (sin acceso a GitHub); el commit queda local")


if __name__ == "__main__":
    main()
