"""Compare la recherche dense de bge-m3 de base et avec l'adaptateur LoRA.

Mesure au niveau ARTICLE (chunks dedupliques) : recall@1/5/10 et MRR, par langue de la
question et entre langues (question francaise -> article arabe, et inversement), avec
intervalles de confiance par bootstrap apparie et par grappes (on tire des articles, pas des
questions : les questions d'un meme article ne sont pas independantes). Par defaut sur data/finetune/val.jsonl
(questions synthetiques d'articles jamais vus a l'entrainement) ; --questions pour un autre
fichier (le jeu d'evaluation, au format eval/questions.example.jsonl).

Usage : python -m darenclair.finetune.evaluer [--adaptateur models/bge-m3-lora] [--questions ...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from darenclair.evaluation.metrics import (
    bootstrap_difference,
    bootstrap_ic,
    recall_at_k,
    reciprocal_rank,
)
from darenclair.ingest.corpus import charger_chunks
from darenclair.ingest.parse import Chunk

RACINE = Path(__file__).resolve().parents[3]


def charger_questions(path: Path, split: str | None = None) -> list[dict]:
    """Accepte val.jsonl (article_id) et le format eval/ (sources : liste d'articles).
    Les pieges (sans source) sont ignores : ils se mesurent par le taux de refus."""
    res = []
    for l in path.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        q = json.loads(l)
        if split and q.get("split") != split:
            continue
        sources = q.get("sources") or ([q["article_id"]] if "article_id" in q else [])
        if sources:
            res.append({"question": q["question"], "langue": q.get("langue", "fr"), "sources": set(sources)})
    return res


def modele(adaptateur: Path | None):
    import torch
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer("BAAI/bge-m3", device="cuda", model_kwargs={"torch_dtype": torch.float16})
    m.max_seq_length = 512
    if adaptateur:
        m[0].auto_model.load_adapter(str(adaptateur))
    return m


def classements(m, chunks: list[Chunk], questions: list[dict], k: int = 50) -> list[list[str]]:
    import torch

    corpus = m.encode([c.texte_index() for c in chunks], batch_size=16, normalize_embeddings=True,
                      convert_to_tensor=True, show_progress_bar=True)
    requetes = m.encode([q["question"] for q in questions], batch_size=32, normalize_embeddings=True,
                        convert_to_tensor=True)
    top = torch.topk(requetes @ corpus.T, k=k, dim=1).indices.cpu().tolist()
    return [[chunks[i].id for i in ligne] for ligne in top]


def scores(rangs: list[list[str]], questions: list[dict]) -> dict[str, list[float]]:
    return {
        "recall@1": [recall_at_k(r, q["sources"], 1) for r, q in zip(rangs, questions, strict=True)],
        "recall@5": [recall_at_k(r, q["sources"], 5) for r, q in zip(rangs, questions, strict=True)],
        "recall@10": [recall_at_k(r, q["sources"], 10) for r, q in zip(rangs, questions, strict=True)],
        "MRR": [reciprocal_rank(r, q["sources"]) for r, q in zip(rangs, questions, strict=True)],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adaptateur", type=Path, default=RACINE / "models" / "bge-m3-lora")
    ap.add_argument("--questions", type=Path, default=RACINE / "data" / "finetune" / "val.jsonl")
    ap.add_argument("--split", choices=["dev", "test", "externe"], help="pour eval/questions.jsonl ou questions_externes.jsonl")
    a = ap.parse_args()

    chunks = charger_chunks()
    langue_article = {c.article_id: c.langue for c in chunks}
    questions = charger_questions(a.questions, a.split)
    print(f"{len(questions)} questions, {len(chunks)} chunks")

    resultats = {}
    for nom, adapt in (("base", None), ("lora", a.adaptateur if a.adaptateur.exists() else None)):
        if nom == "lora" and adapt is None:
            print("pas d'adaptateur : evaluation de la base seulement")
            break
        resultats[nom] = scores(classements(modele(adapt), chunks, questions), questions)

    # Une question dont les sources melangent les deux langues ne compte que dans "toutes".
    groupes = {"toutes": lambda q: True}
    for lq in ("fr", "ar"):
        for la in ("fr", "ar"):
            groupes[f"question {lq} -> article {la}"] = (
                lambda q, lq=lq, la=la: q["langue"] == lq
                and {langue_article.get(s) for s in q["sources"]} == {la}
            )
    for groupe, garder in groupes.items():
        idx = [i for i, q in enumerate(questions) if garder(q)]
        if not idx:
            continue
        articles = [min(questions[i]["sources"]) for i in idx]
        print(f"\n== {groupe} ({len(idx)} questions, {len(set(articles))} articles)")
        for metrique in ("recall@1", "recall@5", "recall@10", "MRR"):
            ligne = f"  {metrique:<10}"
            for nom, s in resultats.items():
                m, bas, haut = bootstrap_ic([s[metrique][i] for i in idx], groupes=articles)
                ligne += f"  {nom} {m:.3f} [{bas:.3f}, {haut:.3f}]"
            if len(resultats) == 2:
                d, bas, haut = bootstrap_difference(
                    [resultats["base"][metrique][i] for i in idx], [resultats["lora"][metrique][i] for i in idx],
                    groupes=articles,
                )
                signe = "significatif" if bas > 0 or haut < 0 else "non significatif"
                ligne += f"  | ecart {d:+.3f} [{bas:+.3f}, {haut:+.3f}] {signe}"
            print(ligne)


if __name__ == "__main__":
    main()
    # Sous Windows, le processus reste bloque a la fermeture apres usage de CUDA : on sort net.
    sys.stdout.flush()
    os._exit(0)
