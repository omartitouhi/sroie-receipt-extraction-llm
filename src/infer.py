"""Génère les prédictions d'une condition expérimentale sur un split.

Conditions :
  zero_shot : modèle de base + instructions seules           (baseline prompt)
  few_shot  : modèle de base + 2 exemples fixes du train     (baseline prompt renforcée)
  rag       : modèle de base + 2 exemples récupérés (mini-RAG)
  ft        : modèle fine-tuné QLoRA + instructions seules
  ft_rag    : modèle fine-tuné + 2 exemples récupérés        (exploratoire)

Robustesse (--perturb, test uniquement) : bruit_ocr, injection -> fichier <condition>__<perturbation>

Usage : python -m src.infer --condition zero_shot --split test
        python -m src.infer --condition ft --perturb bruit_ocr injection
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from .config import BASE_MODEL, MAX_NEW_TOKENS, PRED_DIR, PROC_DIR, ROOT, SEED, ensure_dirs, ADAPTER_DIR
from .data import load_jsonl
from .metrics import norm_text
from .modeling import gpu_mem_report, load_model, load_tokenizer
from .perturb import PERTURBATIONS
from .prompts import build_messages

CONDITIONS = ["zero_shot", "few_shot", "rag", "ft", "ft_rag"]


def fixed_demos(train, k=2, seed=SEED):
    """k exemples fixes, d'enseignes différentes, de longueur médiane."""
    rng = random.Random(seed)
    pool = sorted(train, key=lambda r: len(r["text"]))
    mid = pool[len(pool) // 4: 3 * len(pool) // 4]
    rng.shuffle(mid)
    demos, vendors = [], set()
    for r in mid:
        v = norm_text(r["target"]["company"])
        if v not in vendors and all(r["target"].values()):
            demos.append(r)
            vendors.add(v)
        if len(demos) == k:
            break
    return demos


def render(tok, messages):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


# Budget d'attention par batch : batch x longueur² <= ATTN_BUDGET.
# Sur T4, SDPA retombe sur l'attention « math » dont la mémoire croît en B x têtes x L² :
# 16 prompts RAG de ~2900 tokens demandent 6,9 Go pour UNE couche -> OOM.
# 16 x 1000² : batch 16 pour les prompts zero-shot (~900 tokens), ~6 pour le RAG moyen, 1-2 au pire.
ATTN_BUDGET = 16 * 1000 ** 2


def make_batches(lengths, max_batch):
    """Regroupe les indices (triés par longueur décroissante) sous le budget d'attention."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    batches, cur = [], []
    for i in order:
        L = lengths[cur[0]] if cur else lengths[i]  # le 1er élément est le plus long (padding)
        if cur and (len(cur) + 1 > max_batch or (len(cur) + 1) * L * L > ATTN_BUDGET):
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


@torch.no_grad()
def _gen(model, tok, texts, max_new_tokens):
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         temperature=None, top_p=None, top_k=None,
                         pad_token_id=tok.pad_token_id)
    return tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def generate(model, tok, prompts, batch_size=16, max_new_tokens=MAX_NEW_TOKENS):
    tok.padding_side = "left"
    lengths = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    outputs = [None] * len(prompts)
    queue = make_batches(lengths, batch_size)
    done = 0
    while queue:
        idx = queue.pop(0)
        try:
            seqs = _gen(model, tok, [prompts[i] for i in idx], max_new_tokens)
        except torch.OutOfMemoryError:
            if len(idx) == 1:
                raise
            torch.cuda.empty_cache()
            h = len(idx) // 2
            print(f"\n  OOM sur un batch de {len(idx)} -> on le coupe en deux")
            queue[:0] = [idx[:h], idx[h:]]
            continue
        for i, seq in zip(idx, seqs):
            outputs[i] = seq.strip()
        done += len(idx)
        print(f"  {done}/{len(prompts)}", end="\r", flush=True)
    print()
    return outputs


def _relative_to_root(path):
    """Chemin relatif à la racine du projet si possible (pas de chemin perso dans les résultats)."""
    if path is None:
        return None
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def run(condition, split="test", model_name=BASE_MODEL, adapter=None, k=2, limit=None,
        batch_size=16, load_in_4bit=True, tag=None, force=False, perturb=None):
    ensure_dirs()
    train = load_jsonl(PROC_DIR / "train.jsonl")
    data = load_jsonl(PROC_DIR / f"{split}.jsonl")
    if limit:
        data = data[:limit]
    if perturb:
        # Le texte perturbé remplace l'original : c'est ce que voit le modèle, et c'est sur lui
        # que l'on vérifie l'ancrage des valeurs prédites. La vérité terrain ne change pas.
        fn = PERTURBATIONS[perturb]
        data = [{**r, "text": fn(r["text"], key=r["id"])} for r in data]

    # Reprise après coupure Colab : on ne recalcule pas une condition déjà complète
    name = (tag or condition) + (f"__{perturb}" if perturb else "")
    path = PRED_DIR / f"{name}_{split}.jsonl"
    meta_path = PRED_DIR / f"{name}_{split}.meta.json"
    same_model = (not meta_path.exists()
                  or json.loads(meta_path.read_text(encoding="utf-8")).get("model") == model_name)
    if not force and same_model and path.exists() and len(load_jsonl(path)) == len(data):
        print(f"[{name}/{split}] déjà calculé ({len(data)} tickets) -> on passe (--force pour refaire)")
        return path

    demos_per_doc = [[] for _ in data]
    extra = {}
    if condition == "few_shot":
        demos = fixed_demos(train, k)
        demos_per_doc = [demos for _ in data]
    elif condition in ("rag", "ft_rag"):
        from .rag import ExampleRetriever, retrieval_vendor_hit_rate
        retriever = ExampleRetriever(train, device="cuda" if torch.cuda.is_available() else "cpu")
        demos_per_doc = retriever.retrieve([r["text"] for r in data], k=k)
        extra["retrieval_vendor_hit_rate@1"] = retrieval_vendor_hit_rate(data, demos_per_doc)
        del retriever
        torch.cuda.empty_cache()

    if condition.startswith("ft"):
        adapter = adapter or str(ADAPTER_DIR / "qlora-sroie" / "final")

    tok = load_tokenizer(model_name)
    model = load_model(model_name, load_in_4bit=load_in_4bit,
                       adapter=adapter if condition.startswith("ft") else None)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    prompts = [render(tok, build_messages(r["text"], d)) for r, d in zip(data, demos_per_doc)]
    n_tok = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    print(f"[{name}/{split}] {len(data)} docs, prompt moyen {sum(n_tok)/len(n_tok):.0f} tokens")

    t0 = time.time()
    outs = generate(model, tok, prompts, batch_size=batch_size)
    elapsed = time.time() - t0

    with open(path, "w", encoding="utf-8") as f:
        for r, d, o, n in zip(data, demos_per_doc, outs, n_tok):
            f.write(json.dumps({
                "id": r["id"], "text": r["text"], "target": r["target"],
                "gold_in_text": r["gold_in_text"], "vendor_seen": r.get("vendor_seen"),
                "demo_ids": [x["id"] for x in d], "prompt_tokens": n, "raw_output": o,
                "perturb": perturb,
            }, ensure_ascii=False) + "\n")
    meta = {"condition": condition, "split": split, "model": model_name,
            "adapter": _relative_to_root(adapter) if condition.startswith("ft") else None,
            "n": len(data), "seconds": round(elapsed, 1),
            "sec_per_doc": round(elapsed / len(data), 3),
            "avg_prompt_tokens": round(sum(n_tok) / len(n_tok), 1),
            "load_in_4bit": load_in_4bit, "perturb": perturb, **extra, "gpu": gpu_mem_report(name)}
    with open(PRED_DIR / f"{name}_{split}.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"-> {path} ({elapsed:.0f}s)")
    del model
    torch.cuda.empty_cache()
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=CONDITIONS, required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--no_4bit", action="store_true")
    ap.add_argument("--tag", default=None, help="nom du fichier de sortie (défaut : condition)")
    ap.add_argument("--force", action="store_true", help="recalculer même si déjà fait")
    ap.add_argument("--perturb", nargs="*", default=None, choices=list(PERTURBATIONS),
                    help="tests de robustesse : une ou plusieurs perturbations du texte")
    a = ap.parse_args()
    for pert in (a.perturb or [None]):
        run(a.condition, a.split, a.model, a.adapter, a.k, a.limit, a.batch_size,
            not a.no_4bit, a.tag, a.force, pert)


if __name__ == "__main__":
    main()
