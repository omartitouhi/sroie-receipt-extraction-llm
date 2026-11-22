"""Chargement du modèle (4 bits NF4 par défaut) — commun à l'entraînement et à l'inférence."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def limit_vram():
    """Plafonne la VRAM utilisable par PyTorch (défaut 75 %, variable PROJET9_VRAM_FRACTION).

    Sur un portable dont le GPU affiche aussi l'écran (Windows/WDDM), saturer la VRAM fait
    déborder le pilote en RAM partagée : le bureau peut geler au lieu d'un simple OOM.
    Avec ce plafond, PyTorch lève « CUDA out of memory » proprement.
    """
    import os
    if torch.cuda.is_available():
        frac = float(os.environ.get("PROJET9_VRAM_FRACTION", "0.75"))
        torch.cuda.set_per_process_memory_fraction(frac, 0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] plafond VRAM PyTorch : {frac:.0%} de {total:.1f} Go")


def compute_dtype():
    """bf16 si le GPU le supporte (Ampere+ : RTX 30xx/40xx/50xx, A100), sinon fp16 (T4)."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model_name: str, load_in_4bit: bool = True, adapter: str | None = None):
    limit_vram()
    dtype = compute_dtype()
    kwargs = {"dtype": dtype, "device_map": {"": 0} if torch.cuda.is_available() else None}
    if load_in_4bit and torch.cuda.is_available():
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model


def gpu_mem_report(tag=""):
    if not torch.cuda.is_available():
        return {}
    r = {"tag": tag, "device": torch.cuda.get_device_name(0),
         "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
         "max_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
         "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)}
    print(f"[GPU {tag}] pic alloué {r['max_allocated_gb']} Go / {r['total_gb']} Go")
    return r
