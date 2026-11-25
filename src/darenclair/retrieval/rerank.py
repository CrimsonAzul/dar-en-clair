"""Reranking (cross-encoder bge-reranker-v2-m3) et decision de refus.

Le refus se decide sur le score du meilleur passage apres reranking, avec un seuil choisi sur
le jeu dev (questions pieges vs questions normales). C'est plus fiable que de demander au LLM
s'il "sait" : il repond presque toujours quelque chose.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import lru_cache


@lru_cache(maxsize=1)
def _modele():
    from FlagEmbedding import FlagReranker

    return FlagReranker(os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"), use_fp16=False)


def reranker(question: str, passages: Sequence[tuple[str, str]], top_n: int = 5) -> list[tuple[str, float]]:
    """passages : (id, texte_index). Renvoie les top_n (id, score dans [0, 1]) tries."""
    if not passages:
        return []
    scores = _modele().compute_score(
        [[question, texte] for _, texte in passages], normalize=True, max_length=1024
    )
    if isinstance(scores, float):
        scores = [scores]
    classes = sorted(zip((i for i, _ in passages), scores, strict=True), key=lambda x: -x[1])
    return classes[:top_n]


def doit_refuser(classes: Sequence[tuple[str, float]], seuil: float) -> bool:
    return not classes or classes[0][1] < seuil


def choisir_seuil(scores_normaux: Sequence[float], scores_pieges: Sequence[float]) -> float:
    """Seuil qui maximise la precision equilibree (bonnes reponses + bons refus) sur le dev."""
    candidats = sorted(set(scores_normaux) | set(scores_pieges))
    meilleur, seuil = -1.0, 0.0
    for s in candidats:
        rep = sum(x >= s for x in scores_normaux) / max(len(scores_normaux), 1)
        ref = sum(x < s for x in scores_pieges) / max(len(scores_pieges), 1)
        if (rep + ref) / 2 > meilleur:
            meilleur, seuil = (rep + ref) / 2, s
    return seuil
