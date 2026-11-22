"""Mini-RAG : récupère les k tickets annotés du TRAIN les plus proches (few-shot dynamique).

L'index ne contient que le split train -> aucune fuite vers val/test.
"""
from __future__ import annotations

import numpy as np

from .config import EMBED_MODEL
from .metrics import norm_text


class ExampleRetriever:
    def __init__(self, train_records, model_name: str = EMBED_MODEL, device=None):
        from sentence_transformers import SentenceTransformer
        self.records = train_records
        self.model = SentenceTransformer(model_name, device=device)
        self.emb = self._encode([r["text"] for r in train_records])

    def _encode(self, texts):
        return self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                                 show_progress_bar=False, convert_to_numpy=True)

    def retrieve(self, texts, k=2, exclude_ids=None):
        q = self._encode(texts)
        sims = q @ self.emb.T
        out = []
        for i, row in enumerate(sims):
            order = np.argsort(-row)
            picked = []
            for j in order:
                r = self.records[j]
                if exclude_ids and r["id"] == exclude_ids[i]:
                    continue
                picked.append({**r, "score": float(row[j])})
                if len(picked) == k:
                    break
            out.append(picked)
        return out


def retrieval_vendor_hit_rate(queries, retrieved):
    """Part des requêtes dont le 1er exemple récupéré est de la même enseigne
    (uniquement mesurable quand l'enseigne existe au train)."""
    hits, n = 0, 0
    for q, ex in zip(queries, retrieved):
        if q.get("vendor_seen"):
            n += 1
            hits += norm_text(ex[0]["target"]["company"]) == norm_text(q["target"]["company"])
    return hits / n if n else None
