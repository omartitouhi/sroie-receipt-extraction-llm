"""Exporte le jeu de données utilisé dans un dossier lisible, pour la présentation.

dataset_sroie/
  LISEZMOI.md            fiche du jeu de données (source, splits, champs, contrôles anti-fuite)
  apercu.html            galerie : image du ticket | texte OCR | JSON attendu
  train.csv val.csv test.csv   une ligne par ticket (ouvrable dans Excel)
  train.jsonl ...        exactement ce que lit le modèle
  images/train|val|test/ les scans des tickets (JPEG d'origine, sans ré-encodage)
  images/test_exclus/    les 2 tickets test retirés (doublons exacts du train)

Prérequis : python -m src.data (crée data/processed/*.jsonl).
Usage : python -m src.export_dataset [--no_images]
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import shutil

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from .config import FIELDS, HF_DATASET, PROC_DIR, ROOT
from .data import load_jsonl
from .prompts import target_json

OUT = ROOT / "dataset_sroie"


def image_rows(split):
    """[(nom de fichier d'origine, octets JPEG)] dans l'ordre des lignes du parquet."""
    path = hf_hub_download(HF_DATASET, f"data/{split}-00000-of-00001.parquet",
                           repo_type="dataset")
    col = pq.read_table(path, columns=["image"]).column("image").to_pylist()
    return [(c["path"], c["bytes"]) for c in col]


def write_csv(path, rows, split):
    cols = (["id", "split", "fichier_original", "image", *FIELDS, "enseigne_vue_au_train",
             "similarite_max_train"] + [f"lisible_{f}" for f in FIELDS] + ["texte_ocr"])
    with open(path, "w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig : accents OK dans Excel
        w = csv.writer(f, delimiter=";")
        w.writerow(cols)
        for r in rows:
            w.writerow([r["id"], split, r.get("orig", ""), r.get("img", ""),
                        *[r["target"][k] for k in FIELDS],
                        {True: "oui", False: "non"}.get(r.get("vendor_seen"), ""),
                        r.get("max_sim_train", ""),
                        *[("oui" if r["gold_in_text"][k] else "non") for k in FIELDS],
                        r["text"]])


def gallery(splits, n_per_split=4):
    cards = []
    for split, rows in splits.items():
        step = max(1, len(rows) // n_per_split)
        for r in rows[::step][:n_per_split]:
            badge = ""
            if r.get("vendor_seen") is not None and split == "test":
                badge = (" · enseigne vue au train" if r["vendor_seen"]
                         else " · enseigne jamais vue")
            img = (f'<img src="{html.escape(r["img"])}" loading="lazy">' if r.get("img")
                   else "<p>(image non exportée)</p>")
            cards.append(f"""
<section class="card"><h2>{r['id']} <small>{split}{badge}</small></h2>
<div class="row">{img}
<div><h3>Texte OCR (entrée du modèle)</h3><pre>{html.escape(r['text'])}</pre>
<h3>Sortie attendue</h3><pre class="json">{html.escape(json.dumps(json.loads(target_json(r['target'])), indent=2, ensure_ascii=False))}</pre></div>
</div></section>""")
    return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Aperçu SROIE</title><style>
body{{font-family:system-ui,sans-serif;margin:0 auto;max-width:1200px;padding:16px;background:#fafafa;color:#222}}
.card{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:16px;margin:16px 0}}
.row{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.2fr);gap:16px}}
img{{width:100%;border:1px solid #ccc}} pre{{white-space:pre-wrap;background:#f4f4f4;padding:8px;font-size:12px;max-height:520px;overflow:auto}}
.json{{background:#eef6ee}} small{{color:#666;font-weight:normal}}
@media(max-width:700px){{.row{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>SROIE 2019 — aperçu du jeu de données</h1>
<p>Tâche : à partir du texte OCR d'un ticket de caisse, produire le JSON
<code>{{company, date, address, total}}</code>. Quelques tickets par split ; la totalité est dans
<code>images/</code> et les fichiers CSV.</p>
{''.join(cards)}</body></html>"""


def readme(splits, report, excluded):
    s = report["sizes"]
    lk = report["leakage"]
    g = report["gold_in_text_rate_test"]
    return f"""# Jeu de données : ICDAR 2019 SROIE (tickets de caisse)

## Source
- Compétition ICDAR 2019 *Scanned Receipts OCR and Information Extraction* (SROIE), tâche 3 :
  extraction d'informations clés. Site officiel : https://rrc.cvc.uab.es/?ch=13
- Copie utilisée : Hugging Face `{HF_DATASET}` (version dédoublonnée : 626 tickets train, 347 test).
- Tickets réels de commerces malaisiens (anglais), scannés ; texte OCR et boîtes fournies par les organisateurs.
- Usage : recherche / enseignement.

## Tâche
Entrée : le texte OCR du ticket, reconstruit ligne par ligne à partir des boîtes.
Sortie : un JSON à 4 champs.

| Champ | Sens |
|---|---|
| `company` | nom de l'enseigne / société émettrice |
| `date` | date du ticket |
| `address` | adresse du commerce |
| `total` | montant total payé |

## Splits utilisés dans le projet
| Split | Tickets | Origine |
|---|---|---|
| train | {s['train']} | train officiel, moins la validation |
| val | {s['val']} | 12 % du train officiel, **enseignes disjointes** du train (sélection du checkpoint) |
| test | {s['test']} | test officiel (347) moins {lk['test_removed_exact_duplicates']} doublons exacts du train — **jamais vu** à l'entraînement |

## Contrôles anti-fuite (test vs train)
- Doublons exacts retirés : {lk['test_removed_exact_duplicates']} (images dans `images/test_exclus/` : {', '.join(excluded) or '—'}).
- Quasi-doublons (similarité de Jaccard ≥ 0,8 sur le texte) : {lk['test_near_duplicates_in_train(jaccard>=0.8)']} (gardés, signalés par `similarite_max_train`).
- Enseigne déjà vue au train : {lk['test_vendor_seen_in_train']} tickets ; jamais vue : {lk['test_vendor_unseen']} (colonne `enseigne_vue_au_train`).

## Qualité : la réponse est-elle lisible dans l'OCR ? (test)
company {g['company']:.1%} · date {g['date']:.1%} · address {g['address']:.1%} · total {g['total']:.1%}
(colonnes `lisible_*`). C'est le plafond atteignable sans mémoriser.

Une autre copie publique (`jsdnrs/ICDAR2019-SROIE`) coupe chaque ligne OCR à la première virgule :
l'adresse n'y est lisible que dans ~10 % des tickets. Elle a été écartée grâce à ce contrôle.

## Contenu du dossier
- `apercu.html` : quelques tickets avec image, texte OCR et JSON attendu (à ouvrir dans un navigateur).
- `train.csv`, `val.csv`, `test.csv` : séparateur `;`, ouvrables dans Excel.
- `*.jsonl` : fichiers exacts lus par le code (`data/processed/`).
- `images/` : scans d'origine, nommés par identifiant (`fichier_original` donne le nom SROIE).

Regénérer : `python -m src.data` puis `python -m src.export_dataset`.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_images", action="store_true")
    a = ap.parse_args()

    splits = {s: load_jsonl(PROC_DIR / f"{s}.jsonl") for s in ("train", "val", "test")}
    report = json.load(open(PROC_DIR / "data_report.json", encoding="utf-8"))
    if OUT.exists():
        shutil.rmtree(OUT / "images", ignore_errors=True)
    OUT.mkdir(exist_ok=True)

    excluded = []
    if not a.no_images:
        print("Lecture des images (parquet HF)...")
        imgs = {"train": image_rows("train"), "test": image_rows("test")}
        used = set()
        for split, rows in splits.items():
            d = OUT / "images" / split
            d.mkdir(parents=True, exist_ok=True)
            for r in rows:
                src_split, i = r["id"].rsplit("_", 1)
                orig, data = imgs[src_split][int(i)]
                ext = orig.rsplit(".", 1)[-1].lower() if "." in orig else "jpg"
                (d / f"{r['id']}.{ext}").write_bytes(data)
                r["orig"], r["img"] = orig, f"images/{split}/{r['id']}.{ext}"
                used.add(r["id"])
        d = OUT / "images" / "test_exclus"
        for i, (orig, data) in enumerate(imgs["test"]):
            if f"test_{i:03d}" not in used:
                d.mkdir(parents=True, exist_ok=True)
                ext = orig.rsplit(".", 1)[-1].lower() if "." in orig else "jpg"
                (d / f"test_{i:03d}.{ext}").write_bytes(data)
                excluded.append(f"test_{i:03d}")

    for split, rows in splits.items():
        write_csv(OUT / f"{split}.csv", rows, split)
        shutil.copy(PROC_DIR / f"{split}.jsonl", OUT / f"{split}.jsonl")
    (OUT / "apercu.html").write_text(gallery(splits), encoding="utf-8")
    (OUT / "LISEZMOI.md").write_text(readme(splits, report, excluded), encoding="utf-8")
    n_img = sum(1 for _ in (OUT / "images").rglob("*.*")) if not a.no_images else 0
    print(f"-> {OUT} : {sum(map(len, splits.values()))} tickets, {n_img} images")


if __name__ == "__main__":
    main()
