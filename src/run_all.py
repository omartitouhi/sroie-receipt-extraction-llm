"""Pipeline complet : données -> baselines -> RAG -> QLoRA -> robustesse -> évaluation.

Usage :
  python -m src.run_all              # expérience complète
  python -m src.run_all --quick      # test de fumée (~5 min) : 20 tickets, 10 pas d'entraînement
  python -m src.run_all --skip_train # réutilise l'adaptateur déjà entraîné
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from .config import ADAPTER_DIR, PROC_DIR


def sh(*args):
    cmd = [sys.executable, "-m", *args]
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()

    if not (PROC_DIR / "test.jsonl").exists():
        sh("src.data")

    lim = ["--limit", "20"] if a.quick else []
    bs = ["--batch_size", str(a.batch_size), "--split", a.split]
    for cond in ("zero_shot", "few_shot", "rag"):
        sh("src.infer", "--condition", cond, *bs, *lim)
    sh("src.infer", "--condition", "zero_shot", "--perturb", "bruit_ocr", "injection", *bs, *lim)

    if not a.skip_train or not (ADAPTER_DIR / "qlora-sroie" / "final").exists():
        sh("src.train", *(["--max_steps", "10"] if a.quick else []))

    for cond in ("ft", "ft_rag"):
        sh("src.infer", "--condition", cond, *bs, *lim)
    sh("src.infer", "--condition", "ft", "--perturb", "bruit_ocr", "injection", *bs, *lim)

    sh("src.evaluate", "--split", a.split)


if __name__ == "__main__":
    main()
