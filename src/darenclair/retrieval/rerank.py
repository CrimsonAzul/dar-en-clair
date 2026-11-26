"""Reranking (cross-encoder bge-reranker-v2-m3) et decision de refus.

Le refus se decide sur le score du meilleur passage apres reranking, avec un seuil choisi sur
le jeu dev (questions pieges vs questions normales). C'est plus fiable que de demander au LLM
s'il "sait" : il repond presque toujours quelque chose.

Le modele est charge par sentence-transformers (CrossEncoder) : meme score que FlagReranker
avec normalize=True (sigmoide du logit), sans la dependance FlagEmbedding.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import lru_cache


@lru_cache(maxsize=1)
def _modele():
    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return CrossEncoder(
        os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
        device=device,
        max_length=1024,
        activation_fn=torch.nn.Sigmoid(),
    )


def reranker(question: str, passages: Sequence[tuple[str, str]], top_n: int | None = 5) -> list[tuple[str, float]]:
    """passages : (id, texte_index). Renvoie les top_n (id, score dans [0, 1]) tries."""
    if not passages:
        return []
    scores = _modele().predict([(question, texte) for _, texte in passages], batch_size=8)
    classes = sorted(
        zip((i for i, _ in passages), (float(s) for s in scores), strict=True), key=lambda x: -x[1]
    )
    return classes[:top_n] if top_n else classes


def doit_refuser(classes: Sequence[tuple[str, float]], seuil: float) -> bool:
    return not classes or classes[0][1] < seuil


def choisir_seuil(scores_normaux: Sequence[float], scores_pieges: Sequence[float]) -> float:
    """Seuil qui maximise la precision equilibree (bonnes reponses + bons refus) sur le dev.

    Les seuils candidats sont les milieux entre deux scores du dev consecutifs (un seuil colle
    a un score du dev serait fragile). A precision egale, on garde celui qui a le plus de marge,
    c'est-a-dire l'ecart le plus large entre les deux scores qui l'entourent."""
    scores = sorted(set(scores_normaux) | set(scores_pieges))
    if not scores:
        return 0.0
    bords = [0.0, *scores, 1.0]
    candidats = [((a + b) / 2, b - a) for a, b in zip(bords, bords[1:], strict=False)]

    def precision(s: float) -> float:
        rep = sum(x >= s for x in scores_normaux) / max(len(scores_normaux), 1)
        ref = sum(x < s for x in scores_pieges) / max(len(scores_pieges), 1)
        return (rep + ref) / 2

    return max(candidats, key=lambda c: (round(precision(c[0]), 9), c[1]))[0]
