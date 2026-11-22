"""Fine-tuning QLoRA (base 4 bits NF4 gelée + adaptateurs LoRA entraînables).

- La perte n'est calculée que sur la réponse JSON (tokens du prompt masqués à -100).
- Le prompt d'entraînement est exactement le prompt zero-shot de la baseline :
  la seule différence entre « ft » et « zero_shot » est donc l'adaptateur.
- Sélection du meilleur checkpoint sur la perte de VALIDATION (enseignes disjointes),
  jamais sur le test.

Usage : python -m src.train [--epochs 3] [--train_frac 1.0]
"""
from __future__ import annotations

import argparse
import glob
import inspect
import json
import math
import os
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, BitsAndBytesConfig, Trainer,
                          TrainingArguments)

from pathlib import Path

from .config import ADAPTER_DIR, BASE_MODEL, FIG_DIR, MAX_SEQ_LEN, PROC_DIR, SEED, ensure_dirs
from .data import load_jsonl
from .modeling import compute_dtype, gpu_mem_report, limit_vram, load_tokenizer
from .prompts import build_messages, target_json


def encode(tok, rec, max_len=MAX_SEQ_LEN):
    msgs = build_messages(rec["text"])
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    answer = target_json(rec["target"]) + tok.eos_token
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(answer, add_special_tokens=False)["input_ids"]
    p_ids = p_ids[-(max_len - len(a_ids)):]  # tronque le début du prompt si trop long
    ids = p_ids + a_ids
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(p_ids) + a_ids}


# Projections linéaires adaptées par LoRA, selon l'architecture :
#  - attention classique (Qwen2.5, couches « full attention » de Qwen3.5) : q, k, v, o
#  - MLP : gate, up, down
#  - attention linéaire Gated DeltaNet de Qwen3.5 : in_proj_qkv, in_proj_z, out_proj
LORA_CANDIDATES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                   "in_proj_qkv", "in_proj_z", "out_proj"]


def lora_targets(model):
    """Garde les noms candidats réellement présents dans le modèle chargé."""
    present = {n.rsplit(".", 1)[-1] for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) or m.__class__.__name__ == "Linear4bit"}
    return [t for t in LORA_CANDIDATES if t in present]


def prepare_for_qlora(model):
    """Équivalent allégé de peft.prepare_model_for_kbit_training.

    La version de peft remet en fp32 TOUS les poids non quantifiés, y compris la table
    d'embeddings. Avec le vocabulaire de 248 k tokens de Qwen3.5 (508 M paramètres partagés
    avec la tête de sortie), cela ajoute ~2 Go de VRAM. Ici, seuls les petits paramètres
    (normalisations, paramètres de l'attention linéaire) passent en fp32 pour la stabilité ;
    les embeddings restent en bf16/fp16 (ils sont gelés de toute façon).
    """
    for p in model.parameters():
        p.requires_grad_(False)
        if p.dtype in (torch.float16, torch.bfloat16) and p.numel() < 10_000_000:
            p.data = p.data.float()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()  # nécessaire au gradient checkpointing avec base gelée
    model.config.use_cache = False
    return model


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        L = max(len(b["input_ids"]) for b in batch)
        out = {k: [] for k in ("input_ids", "attention_mask", "labels")}
        for b in batch:
            n = L - len(b["input_ids"])
            out["input_ids"].append(b["input_ids"] + [self.pad_id] * n)
            out["attention_mask"].append(b["attention_mask"] + [0] * n)
            out["labels"].append(b["labels"] + [-100] * n)
        return {k: torch.tensor(v) for k, v in out.items()}


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--out", default=str(ADAPTER_DIR / "qlora-sroie"))
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    # Batch effectif 16 = 1 x 16. Avec un vocabulaire de 152 k (Qwen2.5) à 248 k tokens (Qwen3.5),
    # les logits d'un batch de 4 x 1280 tokens pèsent 3 à 5 Go en fp32 : on reste en batch 1.
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--train_frac", type=float, default=1.0, help="courbe d'apprentissage")
    ap.add_argument("--max_steps", type=int, default=-1, help="test rapide")
    ap.add_argument("--no_resume", action="store_true", help="ignorer les checkpoints existants")
    a = ap.parse_args()

    ensure_dirs()
    if (Path(a.out) / "final" / "adapter_model.safetensors").exists() and not a.no_resume:
        print(f"Adaptateur déjà entraîné : {a.out}/final -> on passe (--no_resume pour ré-entraîner)")
        return
    torch.manual_seed(SEED)
    random.seed(SEED)
    tok = load_tokenizer(a.model)
    tok.padding_side = "right"

    train = load_jsonl(PROC_DIR / "train.jsonl")
    val = load_jsonl(PROC_DIR / "val.jsonl")
    if a.train_frac < 1.0:
        random.Random(SEED).shuffle(train)
        train = train[: max(8, int(a.train_frac * len(train)))]
    tr = [encode(tok, r) for r in train]
    va = [encode(tok, r) for r in val]
    lens = [len(x["input_ids"]) for x in tr]
    print(f"train={len(tr)} val={len(va)} | tokens/exemple : moy {sum(lens)/len(lens):.0f}, max {max(lens)}")

    limit_vram()
    dtype = compute_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=dtype, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
    model = prepare_for_qlora(model)
    targets = lora_targets(model)
    print("Modules adaptés par LoRA :", targets)
    model = get_peft_model(model, LoraConfig(
        r=a.r, lora_alpha=a.alpha, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=targets))
    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(tr) / (a.batch_size * a.grad_accum))
    total_steps = a.max_steps if a.max_steps > 0 else int(steps_per_epoch * a.epochs)
    targs = dict(
        output_dir=a.out,
        warmup_steps=max(1, int(0.05 * total_steps)),
        num_train_epochs=a.epochs,
        max_steps=a.max_steps,
        per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        logging_steps=5,
        eval_strategy="epoch" if a.max_steps < 0 else "steps",
        save_strategy="epoch" if a.max_steps < 0 else "steps",
        eval_steps=max(1, a.max_steps) if a.max_steps > 0 else None,
        save_steps=max(1, a.max_steps) if a.max_steps > 0 else None,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        dataloader_num_workers=0,
        report_to="none",
        seed=SEED,
        torch_empty_cache_steps=4,  # limite la fragmentation / le pic de VRAM réservée
    )
    # Compatibilité transformers 4.x / 5.x : on ignore les arguments inconnus de la version installée
    known = inspect.signature(TrainingArguments.__init__).parameters
    dropped = [k for k in targs if k not in known]
    if dropped:
        print("Arguments ignorés (version de transformers) :", dropped)
    args = TrainingArguments(**{k: v for k, v in targs.items() if k in known})
    trainer = Trainer(model=model, args=args, train_dataset=ListDataset(tr),
                      eval_dataset=ListDataset(va), data_collator=Collator(tok.pad_token_id))
    torch.cuda.reset_peak_memory_stats()
    # Reprise automatique : si la session Colab a coupé, on repart du dernier checkpoint (Drive)
    ckpts = sorted(glob.glob(f"{a.out}/checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[1]))
    resume = ckpts[-1] if ckpts and not a.no_resume else None
    if resume:
        print(f"Reprise depuis {resume}")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    elapsed = time.time() - t0

    final = f"{a.out}/final"
    trainer.model.save_pretrained(final)
    tok.save_pretrained(final)
    log = {"args": vars(a), "train_size": len(tr), "val_size": len(va),
           "train_minutes": round(elapsed / 60, 1), "gpu": gpu_mem_report("train"),
           "log_history": trainer.state.log_history,
           "best_checkpoint": trainer.state.best_model_checkpoint}
    with open(f"{a.out}/train_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    try:  # un échec du graphique ne doit jamais faire perdre un entraînement terminé
        plot_loss(trainer.state.log_history, FIG_DIR / f"courbe_perte_{Path(a.out).name}.png",
                  trainer.state.best_model_checkpoint)
    except Exception as e:
        print(f"Courbe de perte non tracée ({e}) ; l'adaptateur et train_log.json sont sauvegardés.")
    print(f"Adaptateur sauvegardé dans {final} ({elapsed/60:.1f} min)")


def plot_loss(history, path, best=None):
    """Sauvegarde les courbes de perte train / validation (exigence de reproductibilité)."""
    # Lancé depuis un notebook, le sous-processus hérite de MPLBACKEND (backend « inline »
    # de Jupyter), invalide hors du noyau : on force un backend fichier.
    os.environ["MPLBACKEND"] = "Agg"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    tr = [(h["step"], float(h["loss"])) for h in history if "loss" in h]
    va = [(h["step"], float(h["eval_loss"])) for h in history if "eval_loss" in h]
    if not tr:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(*zip(*tr), label="entraînement")
    if va:
        ax.plot(*zip(*va), "o-", label="validation (enseignes disjointes)")
    if best:
        step = int(str(best).rsplit("-", 1)[-1])
        ax.axvline(step, ls="--", color="gray", lw=1, label=f"checkpoint retenu (pas {step})")
    ax.set_xlabel("pas d'optimisation")
    ax.set_ylabel("perte (entropie croisée sur le JSON)")
    ax.set_title("Courbes de perte QLoRA")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Courbe de perte -> {path}")


if __name__ == "__main__":
    main()
