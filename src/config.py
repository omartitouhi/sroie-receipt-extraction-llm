"""Configuration centrale du projet (chemins + hyperparamètres par défaut)."""
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"

# Jeu de données : ICDAR 2019 SROIE (tickets de caisse scannés), version dédoublonnée
# (SROIE-datasetv2). NB : la copie "jsdnrs/ICDAR2019-SROIE" tronque chaque ligne OCR à la
# première virgule (adresse lisible dans seulement 10 % des tickets) -> à éviter.
HF_DATASET = "rth/sroie-2019-v2"
FIELDS = ["company", "date", "address", "total"]

# Modèles
# Modèle principal : Qwen3.5-2B (mars 2026, Apache 2.0, 2 milliards de paramètres -> haut de la
# fourchette 1-2 B visée). Modèle de comparaison : Qwen2.5-1.5B-Instruct (fin 2024).
# Changer de modèle sans toucher au code : variable d'environnement SROIE_LLM_MODEL.
BASE_MODEL = os.environ.get("SROIE_LLM_MODEL", "Qwen/Qwen3.5-2B")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def model_tag(name: str) -> str:
    """'Qwen/Qwen3.5-2B' -> 'qwen3.5-2b' (nom de dossier)."""
    return re.sub(r"[^a-z0-9.\-]+", "-", name.split("/")[-1].lower()).strip("-")


MODEL_TAG = model_tag(BASE_MODEL)

# Résultats et adaptateurs rangés PAR MODÈLE : changer de modèle n'écrase jamais
# les prédictions, métriques ou checkpoints d'un autre modèle.
RESULTS_ROOT = ROOT / "results"
RESULTS_DIR = RESULTS_ROOT / MODEL_TAG
PRED_DIR = RESULTS_DIR / "predictions"
METRICS_DIR = RESULTS_DIR / "metrics"
FIG_DIR = RESULTS_DIR / "figures"
ADAPTER_DIR = ROOT / "outputs" / MODEL_TAG

SEED = 42
VAL_FRACTION = 0.12  # part du train officiel réservée à la validation (split par enseigne)

# Plus long exemple d'entraînement mesuré : 1250 tokens (Qwen2.5), 1268 (Qwen3.5) -> aucune troncature
MAX_SEQ_LEN = 1280
MAX_NEW_TOKENS = 160


def ensure_dirs():
    for d in (RAW_DIR, PROC_DIR, PRED_DIR, METRICS_DIR, FIG_DIR, ADAPTER_DIR):
        d.mkdir(parents=True, exist_ok=True)
