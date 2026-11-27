"""Compare BM25, dense, hybride (RRF) et trois facons de reclasser avec le reranker
(candidats denses, candidats RRF, union des deux) sur eval/questions.jsonl.

Mesure au niveau ARTICLE (recall@5, recall@10, MRR) avec IC a 95 % par bootstrap, et ecarts
apparies entre configurations. Sur dev, choisit aussi le seuil de refus du reranker et l'ecrit
dans config/recherche.yaml ; sur test, relit ce seuil sans rien regler. Le refus et les
contextes enregistres viennent de Recherche.chercher, le chemin de l'API.

Utilise l'index pgvector (python -m darenclair.ingest.indexer) et le meme objet Recherche que
l'API. Enregistre le detail par question dans eval/results/recherche_<split>.json (contextes
reutilises par l'evaluation de la generation).

Usage : python -m darenclair.evaluation.recherche --split dev
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

from darenclair import db
from darenclair.evaluation.metrics import (
    bootstrap_difference,
    bootstrap_ic,
    recall_at_k,
    reciprocal_rank,
)
from darenclair.ingest.parse import Chunk
from darenclair.retrieval.dense import adaptateur_configure, charger_encodeur, encoder, nom_encodeur
from darenclair.retrieval.pipeline import Recherche, charger_parametres, chemin_parametres
from darenclair.retrieval.rerank import choisir_seuil

RACINE = Path(__file__).resolve().parents[3]
CONFIG_RECHERCHE = chemin_parametres()
CONFIGS = ("bm25", "dense", "hybride", "dense+reranker", "hybride+reranker", "union+reranker")
STRATEGIE = "dense+reranker"  # celle de l'API (Recherche.chercher), choisie sur dev
charger_config = charger_parametres


def ecrire_seuil(seuil: float, detail: str) -> None:
    contenu = (
        "# Parametres de la recherche. Ecrit par python -m darenclair.evaluation.recherche --split dev :\n"
        "# le seuil de refus est choisi sur le jeu dev uniquement, jamais sur le test.\n"
        f"seuil_refus: {seuil:.4f}\n"
        f"# {detail}, le {datetime.date.today().isoformat()}\n"
        "k_candidats: 30   # candidats denses reclasses par le reranker\n"
        "n_passages: 5     # passages donnes au LLM : [S1]..[S5]\n"
    )
    CONFIG_RECHERCHE.write_text(contenu, encoding="utf-8")


def ligne_ic(valeurs: list[float | None]) -> str:
    m, bas, haut = bootstrap_ic(valeurs)
    return f"{m:.3f} [{bas:.3f}, {haut:.3f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "externe"], required=True)
    ap.add_argument("--questions", type=Path, help="par defaut eval/questions.jsonl, ou eval/questions_externes.jsonl pour --split externe")
    a = ap.parse_args()

    fichier = a.questions or RACINE / "eval" / (
        "questions_externes.jsonl" if a.split == "externe" else "questions.jsonl")
    questions = [json.loads(l) for l in fichier.read_text(encoding="utf-8").splitlines() if l.strip()]
    questions = [q for q in questions if q["split"] == a.split]
    adaptateur = adaptateur_configure()

    with db.connect() as conn:
        indexe = db.encodeur_indexe(conn)
        if indexe != nom_encodeur(adaptateur):
            sys.exit(f"index construit avec {indexe}, questions encodees avec {nom_encodeur(adaptateur)}")
        chunks = [Chunk(**c) for c in db.charger_chunks(conn)]
        m = charger_encodeur(adaptateur)
        vecteurs = dict(zip((q["question"] for q in questions),
                            encoder(m, [q["question"] for q in questions], batch_size=32), strict=True))
        recherche = Recherche(
            chunks, lambda q, k, dom: db.recherche_dense(conn, vecteurs[q], k, dom),
            seuil_refus=charger_config().get("seuil_refus", 0.0),
        )
        print(f"{len(questions)} questions ({a.split}), {len(chunks)} chunks, encodeur {indexe}")

        details = []
        for q in questions:
            texte = q["question"]
            dense, lexicale = recherche.dense(texte), recherche.lexicale(texte)
            hybride = recherche.hybride(texte)
            # Le reranker note une seule fois l'union des deux listes ; les variantes reclassent
            # chacune son propre ensemble de candidats avec ces memes scores.
            union = list(dict.fromkeys(hybride + dense + lexicale))
            note = dict(recherche.reclasser(texte, union))
            variantes = {
                "dense+reranker": dense, "hybride+reranker": hybride, "union+reranker": union,
            }
            classes = {
                nom: sorted(((i, note[i]) for i in ids), key=lambda x: -x[1])
                for nom, ids in variantes.items()
            }
            finale = recherche.chercher(texte).classes  # chemin de l'API = dense+reranker
            classes[STRATEGIE] = finale
            details.append({
                "id": q["id"], "type": q["type"], "langue": q["langue"], "sources": q["sources"],
                "rangs": {"bm25": lexicale, "dense": dense, "hybride": hybride,
                          **{nom: [i for i, _ in c] for nom, c in classes.items()}},
                "candidats": [i for i, _ in finale],
                "scores_reranker": [round(s, 4) for _, s in finale],
                "score_max": round(finale[0][1], 4) if finale else 0.0,
            })

    avec_sources = [d for d in details if d["sources"]]
    pieges = [d for d in details if not d["sources"]]
    print(f"\n== recherche : {len(avec_sources)} questions avec sources (IC 95 % par bootstrap)")
    scores = {}
    for cfg in CONFIGS:
        r5 = [recall_at_k(d["rangs"][cfg], d["sources"], 5) for d in avec_sources]
        r10 = [recall_at_k(d["rangs"][cfg], d["sources"], 10) for d in avec_sources]
        mrr = [reciprocal_rank(d["rangs"][cfg], d["sources"]) for d in avec_sources]
        scores[cfg] = {"recall@5": r5, "recall@10": r10, "MRR": mrr}
        print(f"  {cfg:<17} recall@5 {ligne_ic(r5)}  recall@10 {ligne_ic(r10)}  MRR {ligne_ic(mrr)}")

    print("\n== ecarts apparies (b - a)")
    for x, y in (("bm25", "dense"), ("dense", "hybride"), ("hybride", "hybride+reranker"),
                 ("dense", "dense+reranker"), ("dense+reranker", "union+reranker"),
                 ("hybride+reranker", "union+reranker")):
        for metrique in ("recall@5", "MRR"):
            d, bas, haut = bootstrap_difference(scores[x][metrique], scores[y][metrique])
            signe = "significatif" if bas > 0 or haut < 0 else "non significatif"
            print(f"  {y} - {x:<10} {metrique:<9} {d:+.3f} [{bas:+.3f}, {haut:+.3f}] {signe}")

    normaux = [d["score_max"] for d in avec_sources]
    scores_pieges = [d["score_max"] for d in pieges]
    if a.split == "dev":
        seuil = choisir_seuil(normaux, scores_pieges)
        ecrire_seuil(seuil, f"choisi sur dev : {len(normaux)} questions normales, {len(scores_pieges)} pieges")
        print(f"\n== refus : seuil choisi sur dev = {seuil:.4f} (ecrit dans {CONFIG_RECHERCHE.name})")
    else:
        seuil = charger_config()["seuil_refus"]
        print(f"\n== refus : seuil relu dans {CONFIG_RECHERCHE.name} = {seuil:.4f} (pas de reglage sur test)")
    repondues = sum(s >= seuil for s in normaux)
    refusees = sum(s < seuil for s in scores_pieges)
    print(f"  questions normales repondues : {repondues}/{len(normaux)}")
    print(f"  pieges refuses               : {refusees}/{len(scores_pieges)}")
    print(f"  scores max des pieges        : {sorted(scores_pieges)}")
    print(f"  scores max normaux < seuil   : {sorted(s for s in normaux if s < seuil)}")

    sortie = RACINE / "eval" / "results" / f"recherche_{a.split}.json"
    sortie.parent.mkdir(parents=True, exist_ok=True)
    sortie.write_text(json.dumps({"seuil_refus": seuil, "encodeur": indexe, "questions": details},
                                 ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\ndetail par question : {sortie.relative_to(RACINE)}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)  # Windows : le processus reste bloque a la fermeture apres usage de CUDA
