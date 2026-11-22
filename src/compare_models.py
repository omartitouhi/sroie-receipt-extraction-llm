"""Compare les modèles de base entre eux (Qwen2.5-1.5B vs Qwen3.5-2B, ...).

Lit results/<modèle>/metrics/<condition>_<split>.json (produits par src.evaluate pour chaque
modèle) et écrit results/COMPARAISON_MODELES_<split>.md + une figure.

Question : le gain du fine-tuning se confirme-t-il sur un modèle plus récent ?

Usage : python -m src.compare_models [--split test]
"""
from __future__ import annotations

import argparse
import json
import os

from .config import RESULTS_ROOT

CONDS = ["zero_shot", "few_shot", "rag", "ft", "ft_rag"]
LABELS = {"zero_shot": "zero-shot", "few_shot": "few-shot", "rag": "mini-RAG", "ft": "QLoRA",
          "ft_rag": "QLoRA + RAG"}


def pct(x):
    return f"{100 * x:.1f}" if x is not None else "–"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    a = ap.parse_args()

    models = {}
    for d in sorted(p for p in RESULTS_ROOT.iterdir() if (p / "metrics").is_dir()):
        m = {}
        for c in CONDS + ["zero_shot__bruit_ocr", "zero_shot__injection", "ft__bruit_ocr",
                          "ft__injection"]:
            f = d / "metrics" / f"{c}_{a.split}.json"
            if f.exists():
                m[c] = json.loads(f.read_text(encoding="utf-8"))
        tests = d / "metrics" / f"tests_{a.split}.json"
        m["_tests"] = json.loads(tests.read_text(encoding="utf-8")) if tests.exists() else {}
        if any(c in m for c in CONDS):
            models[d.name] = m
    if not models:
        raise SystemExit("Aucune métrique : lancez d'abord src.evaluate pour chaque modèle.")

    md = [f"# Comparaison des modèles de base — split `{a.split}`\n",
          "Même données, même prompt, même protocole ; seul le modèle de base change.\n",
          "## EM moyen (%) par condition\n",
          "| Modèle | " + " | ".join(LABELS[c] for c in CONDS) + " | gain QLoRA vs zero-shot (IC 95 %) |",
          "|---|" + "---|" * (len(CONDS) + 1)]
    for name, m in models.items():
        gain = "–"
        bs = m["_tests"].get("zero_shot_vs_ft", {}).get("bootstrap")
        if bs:
            gain = f"{100*bs['diff']:+.1f} [{100*bs['ci_low']:+.1f} ; {100*bs['ci_high']:+.1f}]"
        md.append(f"| {name} | " + " | ".join(pct(m[c]["mean_em"]) if c in m else "–" for c in CONDS)
                  + f" | {gain} |")

    md += ["\n## Généralisation et hallucinations\n",
           "| Modèle | condition | EM enseigne vue | EM enseigne jamais vue | JSON valide | "
           "valeurs non ancrées | fabrications | mémorisations |",
           "|---|---|---|---|---|---|---|---|"]
    for name, m in models.items():
        for c in ("zero_shot", "ft"):
            if c in m:
                s = m[c]
                md.append(f"| {name} | {LABELS[c]} | {pct(s.get('mean_em_seen_vendor'))} | "
                          f"{pct(s.get('mean_em_unseen_vendor'))} | {pct(s['json_valid_rate'])} | "
                          f"{pct(s['unsupported_rate'])} | {s['categories']['fabrication']} | "
                          f"{s['categories']['memorisation']} |")

    md += ["\n## Robustesse\n",
           "| Modèle | condition | EM propre | EM bruit OCR | EM injection | attaques réussies |",
           "|---|---|---|---|---|---|"]
    for name, m in models.items():
        rob = m["_tests"].get("robustesse", {})
        for c in ("zero_shot", "ft"):
            if c in m:
                asr = rob.get(c, {}).get("attack_success_rate")
                md.append(f"| {name} | {LABELS[c]} | {pct(m[c]['mean_em'])} | "
                          f"{pct(m.get(c + '__bruit_ocr', {}).get('mean_em'))} | "
                          f"{pct(m.get(c + '__injection', {}).get('mean_em'))} | "
                          f"{pct(asr) + ' %' if asr is not None else '–'} |")

    md += ["\n## Coût\n", "| Modèle | condition | s / ticket | pic VRAM inférence (Go) |",
           "|---|---|---|---|"]
    for name, m in models.items():
        for c in CONDS:
            meta = m.get(c, {}).get("meta")
            if meta:
                md.append(f"| {name} | {LABELS[c]} | {meta.get('sec_per_doc', '–')} | "
                          f"{(meta.get('gpu') or {}).get('max_allocated_gb', '–')} |")

    os.environ["MPLBACKEND"] = "Agg"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        w = 0.8 / len(models)
        for i, (name, m) in enumerate(models.items()):
            ax.bar([x + i * w for x in range(len(CONDS))],
                   [100 * m[c]["mean_em"] if c in m else 0 for c in CONDS], w, label=name)
        ax.set_xticks([x + w * (len(models) - 1) / 2 for x in range(len(CONDS))])
        ax.set_xticklabels([LABELS[c] for c in CONDS])
        ax.set_ylabel("EM moyen (%)")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.set_title(f"Comparaison des modèles de base ({a.split})")
        fig.tight_layout()
        p = RESULTS_ROOT / f"comparaison_modeles_{a.split}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        md.append(f"\n![comparaison]({p.name})")
    except ImportError:
        pass

    out = RESULTS_ROOT / f"COMPARAISON_MODELES_{a.split}.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:5 + len(models)]))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
