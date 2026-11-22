"""Préparation des données SROIE.

1. Télécharge les fichiers parquet du hub HF (on ne lit pas la colonne image).
2. Reconstruit le texte OCR ligne par ligne à partir des boîtes englobantes.
3. Découpe train officiel -> train / val **par enseigne** (pas de fuite val).
4. Garde le test officiel (361 tickets) intact comme jeu jamais vu.
5. Contrôles anti-fuite : doublons exacts, quasi-doublons, enseignes communes.

Usage : python -m src.data
"""
from __future__ import annotations

import hashlib
import json
import random
import statistics

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from .config import (FIELDS, HF_DATASET, PROC_DIR, SEED, VAL_FRACTION,
                     ensure_dirs)
from .metrics import is_supported, norm_text

def download_split(split: str):
    """Renvoie une liste de {key, words, bboxes, entities} (on ne lit pas les images)."""
    # Cache HF par défaut (~/.cache/huggingface) : les ~500 Mo d'images embarquées ne vont
    # ni dans Google Drive ni dans OneDrive.
    path = hf_hub_download(HF_DATASET, f"data/{split}-00000-of-00001.parquet",
                           repo_type="dataset")
    rows = pq.read_table(path, columns=["objects"]).to_pylist()
    out = []
    for i, r in enumerate(rows):
        o = r["objects"]
        # bbox = [[x1, x2, x3, x4], [y1, y2, y3, y4]] (quadrilatère) -> [xmin, ymin, xmax, ymax]
        boxes = [[min(b[0]), min(b[1]), max(b[0]), max(b[1])] for b in o["bbox"]]
        out.append({"key": f"{split}_{i:03d}", "words": o["text"], "bboxes": boxes,
                    "entities": o["entities"]})
    return out


def rebuild_text(words, bboxes) -> str:
    """Regroupe les segments OCR en lignes (même hauteur) puis les trie de gauche à droite."""
    segs = []
    for w, b in zip(words, bboxes):
        x1, y1, x2, y2 = b
        segs.append({"w": (w or "").strip(), "x": x1, "yc": (y1 + y2) / 2, "h": max(1, y2 - y1)})
    segs = [s for s in segs if s["w"]]
    if not segs:
        return ""
    med_h = statistics.median(s["h"] for s in segs)
    segs.sort(key=lambda s: s["yc"])
    lines, cur = [], [segs[0]]
    for s in segs[1:]:
        line_yc = sum(t["yc"] for t in cur) / len(cur)
        if abs(s["yc"] - line_yc) < 0.5 * med_h:
            cur.append(s)
        else:
            lines.append(cur)
            cur = [s]
    lines.append(cur)
    return "\n".join("  ".join(t["w"] for t in sorted(line, key=lambda t: t["x"]))
                     for line in lines)


def to_record(row):
    text = rebuild_text(row["words"], row["bboxes"])
    target = {f: (row["entities"].get(f) or "").strip() for f in FIELDS}
    return {
        "id": row["key"],
        "text": text,
        "target": target,
        # La vérité terrain est-elle lisible dans l'OCR ? (bruit d'annotation / OCR)
        "gold_in_text": {f: is_supported(target[f], text, f) for f in FIELDS},
    }


def shingles(text, k=5):
    t = norm_text(text)
    return {t[i:i + k] for i in range(max(1, len(t) - k + 1))}


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def group_split(records, frac, seed):
    """Split par enseigne : toutes les factures d'une même enseigne vont du même côté."""
    vendors = sorted({norm_text(r["target"]["company"]) for r in records})
    rng = random.Random(seed)
    rng.shuffle(vendors)
    val_vendors, n_val, target_n = set(), 0, int(frac * len(records))
    counts = {}
    for r in records:
        v = norm_text(r["target"]["company"])
        counts[v] = counts.get(v, 0) + 1
    for v in vendors:
        if n_val >= target_n:
            break
        val_vendors.add(v)
        n_val += counts[v]
    train = [r for r in records if norm_text(r["target"]["company"]) not in val_vendors]
    val = [r for r in records if norm_text(r["target"]["company"]) in val_vendors]
    return train, val


def leakage_report(train, test):
    h = lambda r: hashlib.md5(norm_text(r["text"]).encode()).hexdigest()
    train_hashes = {h(r) for r in train}
    exact = [r["id"] for r in test if h(r) in train_hashes]

    train_sh = [shingles(r["text"]) for r in train]
    near = []
    for r in test:
        s = shingles(r["text"])
        best = max(jaccard(s, t) for t in train_sh)
        r["max_sim_train"] = round(best, 3)
        if best >= 0.8:
            near.append(r["id"])

    train_vendors = {norm_text(r["target"]["company"]) for r in train}
    for r in test:
        r["vendor_seen"] = norm_text(r["target"]["company"]) in train_vendors
    return {
        "test_exact_duplicates_in_train": len(exact),
        "test_near_duplicates_in_train(jaccard>=0.8)": len(near),
        "test_vendor_seen_in_train": sum(r["vendor_seen"] for r in test),
        "test_vendor_unseen": sum(not r["vendor_seen"] for r in test),
        "near_duplicate_ids": near[:50],
    }


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ensure_dirs()
    print("Téléchargement de SROIE (colonnes texte uniquement)...")
    train_full = [to_record(r) for r in download_split("train")]
    test = [to_record(r) for r in download_split("test")]

    train, val = group_split(train_full, VAL_FRACTION, SEED)
    for r in val:
        r["vendor_seen"] = False  # par construction

    rep = leakage_report(train, test)
    # Doublons exacts test/train : on les retire du test (sinon on évalue sur du vu)
    train_hashes = {hashlib.md5(norm_text(r["text"]).encode()).hexdigest() for r in train}
    n_before = len(test)
    test = [r for r in test
            if hashlib.md5(norm_text(r["text"]).encode()).hexdigest() not in train_hashes]
    rep["test_removed_exact_duplicates"] = n_before - len(test)
    rep["test_vendor_seen_in_train"] = sum(r["vendor_seen"] for r in test)
    rep["test_vendor_unseen"] = sum(not r["vendor_seen"] for r in test)

    for name, rows in (("train", train), ("val", val), ("test", test)):
        write_jsonl(PROC_DIR / f"{name}.jsonl", rows)

    stats = {
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "avg_chars": {k: round(sum(len(r["text"]) for r in v) / len(v))
                      for k, v in (("train", train), ("val", val), ("test", test))},
        "gold_in_text_rate_test": {f: round(sum(r["gold_in_text"][f] for r in test) / len(test), 3)
                                   for f in FIELDS},
        "leakage": rep,
    }
    with open(PROC_DIR / "data_report.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in stats.items() if k != "leakage"}, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "near_duplicate_ids"}, indent=2))
    print("Exemple de texte reconstruit :\n" + test[0]["text"][:600])


if __name__ == "__main__":
    main()
