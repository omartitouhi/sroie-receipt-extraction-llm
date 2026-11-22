"""Normalisation, parsing JSON, métriques par champ et taxonomie des hallucinations.

Aucune dépendance GPU : ce module est testé par tests/test_metrics.py.
"""
from __future__ import annotations

import json
import math
import random
import re
from collections import Counter

from .config import FIELDS

# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^A-Z0-9 ]+")


def norm_text(s) -> str:
    """Majuscules, ponctuation -> espace, espaces fusionnés."""
    if s is None:
        return ""
    s = str(s).upper()
    s = _NON_ALNUM.sub(" ", s)
    return _WS.sub(" ", s).strip()


def norm_total(s):
    """'RM 1,234.50' -> 1234.5 ; None si non numérique."""
    if s is None:
        return None
    txt = str(s).upper().replace("RM", "").replace(",", "").replace(" ", "")
    m = re.search(r"-?\d+(?:\.\d+)?", txt)
    if not m:
        return None
    try:
        return round(float(m.group()), 2)
    except ValueError:
        return None


_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def norm_date(s):
    """Essaie d'extraire (jour, mois, année) ; renvoie None si échec."""
    if s is None:
        return None
    t = str(s).upper()
    m = re.search(r"(\d{1,4})[./\-\s](\d{1,2}|[A-Z]{3})[A-Z]*[./\-\s](\d{2,4})", t)
    if not m:
        return None
    a, b, c = m.groups()
    if b.isalpha():
        if b[:3] not in _MONTHS:
            return None
        b = _MONTHS[b[:3]]
    a, b, c = int(a), int(b), int(c)
    if a > 31:  # format ISO aaaa-mm-jj
        a, c = c, a
    if c < 100:
        c += 2000
    return (a, b, c)


def field_equal(field: str, pred, gold) -> bool:
    """Égalité 'officielle' (souple sur la forme, stricte sur le fond)."""
    if field == "total":
        p, g = norm_total(pred), norm_total(gold)
        return p is not None and p == g
    if field == "date":
        p, g = norm_date(pred), norm_date(gold)
        if p is not None and g is not None:
            return p == g
    return norm_text(pred) == norm_text(gold) and norm_text(gold) != ""


def token_f1(pred, gold) -> float:
    p, g = norm_text(pred).split(), norm_text(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


# --------------------------------------------------------------------------- #
# Parsing de la sortie du modèle
# --------------------------------------------------------------------------- #
def parse_json_output(text: str):
    """Renvoie (dict|None, json_valide: bool). Tolère du texte autour du JSON."""
    if text is None:
        return None, False
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    m = re.search(r"\{.*\}", t, flags=re.S)
    if not m:
        return None, False
    cand = m.group()
    for attempt in (cand, re.sub(r",\s*}", "}", cand)):
        try:
            obj = json.loads(attempt)
            if isinstance(obj, dict):
                return obj, True
        except json.JSONDecodeError:
            continue
    return None, False


# --------------------------------------------------------------------------- #
# Ancrage dans la source (pour détecter les hallucinations)
# --------------------------------------------------------------------------- #
def is_supported(value, source: str, field: str | None = None) -> bool:
    """La valeur est-elle présente dans le texte source ?

    - sous-chaîne après normalisation, ou
    - >= 90 % des tokens de la valeur présents dans la source (adresses multi-lignes), ou
    - pour 'total', le même montant numérique apparaît dans la source.
    """
    v = norm_text(value)
    if not v:
        return False
    src = norm_text(source)
    if f" {v} " in f" {src} ":
        return True
    if field == "total":
        amt = norm_total(value)
        if amt is not None:
            nums = {norm_total(x) for x in re.findall(r"\d[\d,]*\.\d{2}", source)}
            return amt in nums
    toks = v.split()
    if len(toks) >= 3:
        src_toks = set(src.split())
        cov = sum(t in src_toks for t in toks) / len(toks)
        return cov >= 0.9
    return False


# --------------------------------------------------------------------------- #
# Évaluation d'un fichier de prédictions
# --------------------------------------------------------------------------- #
ERROR_TYPES = ["correct", "json_invalide", "omission", "mauvais_span", "partiel",
               "fabrication", "memorisation"]


def classify(field, pred, gold, source, json_ok, train_values: set | None = None) -> str:
    """Catégorise chaque (document, champ).

    correct       : égal à la vérité terrain
    json_invalide : sortie non parsable -> aucun champ récupérable
    omission      : null / vide alors que la vérité existe
    mauvais_span  : valeur bien présente dans le ticket mais ce n'est pas la bonne
                    (ex. sous-total au lieu du total, autre date)
    partiel       : recouvrement partiel (F1 >= 0.5) avec la vérité (adresse tronquée...)
    fabrication   : valeur absente du ticket ET sans lien avec la vérité -> hallucination
    memorisation  : valeur absente du ticket mais recopiée d'une cible du train
                    -> hallucination « par cœur » (typique après fine-tuning)
    """
    if not json_ok:
        return "json_invalide"
    if field_equal(field, pred, gold):
        return "correct"
    if pred is None or norm_text(pred) == "":
        return "omission"
    supported = is_supported(pred, source, field)
    if token_f1(pred, gold) >= 0.5 and field in ("company", "address"):
        return "partiel"
    if supported:
        return "mauvais_span"
    if train_values and norm_text(pred) in train_values:
        return "memorisation"
    return "fabrication"


def evaluate_records(records, train_values_by_field=None):
    """records : liste de dicts {id, text, target, raw_output, vendor_seen?}.

    Renvoie (résumé, lignes détaillées)."""
    rows = []
    for r in records:
        obj, ok = parse_json_output(r["raw_output"])
        obj = obj or {}
        for f in FIELDS:
            pred = obj.get(f)
            if isinstance(pred, (int, float)):
                pred = str(pred)
            elif pred is not None and not isinstance(pred, str):
                pred = json.dumps(pred, ensure_ascii=False)
            gold = r["target"][f]
            tv = (train_values_by_field or {}).get(f)
            cat = classify(f, pred, gold, r["text"], ok, tv)
            rows.append({
                "id": r["id"], "field": f, "pred": pred, "gold": gold,
                "json_ok": ok, "correct": cat == "correct", "f1": token_f1(pred, gold),
                "category": cat,
                "pred_supported": is_supported(pred, r["text"], f) if pred else False,
                "gold_in_text": r.get("gold_in_text", {}).get(f, True),
                "vendor_seen": r.get("vendor_seen"),
            })
    return summarize(rows), rows


def summarize(rows):
    out = {"n_docs": len({r["id"] for r in rows})}
    docs = {}
    for r in rows:
        docs.setdefault(r["id"], []).append(r)
    out["json_valid_rate"] = sum(v[0]["json_ok"] for v in docs.values()) / max(1, len(docs))
    out["doc_all_correct"] = sum(all(x["correct"] for x in v) for v in docs.values()) / max(1, len(docs))
    for f in FIELDS:
        fr = [r for r in rows if r["field"] == f]
        out[f"{f}_em"] = sum(r["correct"] for r in fr) / max(1, len(fr))
        out[f"{f}_f1"] = sum(r["f1"] for r in fr) / max(1, len(fr))
    out["mean_em"] = sum(out[f"{f}_em"] for f in FIELDS) / len(FIELDS)
    out["mean_f1"] = sum(out[f"{f}_f1"] for f in FIELDS) / len(FIELDS)
    pred_rows = [r for r in rows if r["pred"]]
    out["unsupported_rate"] = (sum(not r["pred_supported"] for r in pred_rows) / len(pred_rows)
                               if pred_rows else 0.0)
    cats = Counter(r["category"] for r in rows)
    out["categories"] = {c: cats.get(c, 0) for c in ERROR_TYPES}
    # Sous-groupes enseigne vue / jamais vue au train
    for flag, name in ((True, "seen"), (False, "unseen")):
        sub = [r for r in rows if r["vendor_seen"] is flag]
        if sub:
            out[f"mean_em_{name}_vendor"] = sum(r["correct"] for r in sub) / len(sub)
            out[f"n_docs_{name}_vendor"] = len({r["id"] for r in sub})
    return out


# --------------------------------------------------------------------------- #
# Tests statistiques appariés
# --------------------------------------------------------------------------- #
def mcnemar_exact(a_correct, b_correct):
    """Test de McNemar exact (bilatéral) sur des décisions appariées."""
    b = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    c = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    n = b + c
    if n == 0:
        return {"a_only": b, "b_only": c, "p_value": 1.0}
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"a_only": b, "b_only": c, "p_value": min(1.0, 2 * p)}


def paired_bootstrap(a_scores, b_scores, n_boot=2000, seed=0):
    """IC 95 % de la différence moyenne (b - a), rééchantillonnage par document."""
    rng = random.Random(seed)
    n = len(a_scores)
    diffs = [b - a for a, b in zip(a_scores, b_scores)]
    obs = sum(diffs) / n
    boots = []
    for _ in range(n_boot):
        s = [diffs[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(s) / n)
    boots.sort()
    return {"diff": obs, "ci_low": boots[int(0.025 * n_boot)],
            "ci_high": boots[int(0.975 * n_boot) - 1],
            "p_le_0": sum(x <= 0 for x in boots) / n_boot}
