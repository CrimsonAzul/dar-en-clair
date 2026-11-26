"""Prepare les donnees d'entrainement : separation par article et negatifs difficiles.

- Separation par ARTICLE (pas par question) : toutes les questions d'un article vont du meme
  cote. Sinon le modele verrait a l'entrainement des questions quasi identiques a celles de
  validation, et le gain mesure serait faux.
- Negatif difficile : un chunk qui ressemble a la question pour BM25 mais d'un AUTRE article,
  tire entre les rangs 3 et 20. On saute les 2 premiers : ce sont souvent de faux negatifs
  (articles voisins qui repondent aussi).
- Filtre de coherence (Promptagator) : une question d'entrainement n'est gardee que si bge-m3
  de base retrouve son article dans le top 20. Ecarte les questions inventees par le LLM
  ("droit de mutation" pour un article sur les terrains non batis). Le top 20 reste large :
  les questions difficiles mais justes passent. JAMAIS applique a la validation : sinon elle ne
  contiendrait que des questions que le modele de base reussit deja, et la comparaison serait
  truquee en sa faveur.

Usage : python -m darenclair.finetune.donnees [--sans-filtre]
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path

from darenclair.ingest.parse import Chunk
from darenclair.retrieval.lexical import BM25

RACINE = Path(__file__).resolve().parents[3]
DOSSIER = RACINE / "data" / "finetune"
PART_VALIDATION = 0.10


def cote(article_id: str) -> str:
    """Repartition deterministe et stable : le meme article tombe toujours du meme cote."""
    h = int(hashlib.sha1(article_id.encode()).hexdigest(), 16) % 1000
    return "val" if h < PART_VALIDATION * 1000 else "train"


def charger() -> tuple[dict[str, Chunk], list[dict]]:
    chunks: dict[str, Chunk] = {}
    for f in sorted((RACINE / "data" / "processed").glob("*.jsonl")):
        for l in f.read_text(encoding="utf-8").splitlines():
            if l.strip():
                c = Chunk(**json.loads(l))
                chunks[c.id] = c
    lignes = [json.loads(l) for l in (DOSSIER / "questions.jsonl").read_text(encoding="utf-8").splitlines() if l]
    questions = [q for q in lignes if "question" in q and q["chunk_id"] in chunks]
    return chunks, questions


def negatif_difficile(bm: BM25, q: dict, chunks: dict[str, Chunk], rng: random.Random) -> str | None:
    """Jamais un chunk d'un article de validation : son texte ne doit pas du tout servir a l'entrainement."""
    candidats = [
        i for i in bm.search(q["question"], k=20)[2:]
        if chunks[i].article_id != q["article_id"] and cote(chunks[i].article_id) == "train"
    ]
    return rng.choice(candidats) if candidats else None


def coherentes(questions: list[dict], chunks: dict[str, Chunk], k: int = 20) -> list[bool]:
    """True si bge-m3 de base retrouve l'article de la question dans ses k premiers articles."""
    from darenclair.evaluation.metrics import articles_uniques
    from darenclair.finetune.evaluer import classements, modele

    liste = list(chunks.values())
    rangs = classements(modele(None), liste, questions, k=60)
    return [q["article_id"] in articles_uniques(r)[:k] for q, r in zip(questions, rangs, strict=True)]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sans-filtre", action="store_true", help="pas de filtre de coherence (pas de GPU)")
    a = ap.parse_args()

    chunks, questions = charger()
    rng = random.Random(0)
    bm = BM25({i: c.texte_index() for i, c in chunks.items()})

    a_entrainer = [q for q in questions if cote(q["article_id"]) == "train"]
    garder = [True] * len(a_entrainer) if a.sans_filtre else coherentes(a_entrainer, chunks)
    print(f"filtre de coherence : {sum(garder)}/{len(garder)} questions d'entrainement gardees")
    gardees = {id(q) for q, g in zip(a_entrainer, garder, strict=True) if g}

    train, val = [], []
    for q in questions:
        if cote(q["article_id"]) == "val":
            val.append({"question": q["question"], "langue": q["langue"], "article_id": q["article_id"]})
            continue
        if id(q) not in gardees:
            continue
        neg = negatif_difficile(bm, q, chunks, rng)
        if neg is None:
            continue
        train.append({
            "anchor": q["question"],
            "positive": chunks[q["chunk_id"]].texte_index(),
            "negative": chunks[neg].texte_index(),
        })
    rng.shuffle(train)

    for nom, lignes in (("train", train), ("val", val)):
        with (DOSSIER / f"{nom}.jsonl").open("w", encoding="utf-8") as f:
            for l in lignes:
                f.write(json.dumps(l, ensure_ascii=False) + "\n")
    articles_val = {v["article_id"] for v in val}
    print(f"train : {len(train)} triplets | val : {len(val)} questions sur {len(articles_val)} articles")


if __name__ == "__main__":
    main()
    # Sous Windows, le processus reste bloque a la fermeture apres usage de CUDA : on sort net.
    sys.stdout.flush()
    os._exit(0)
