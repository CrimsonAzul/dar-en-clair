"""Fusion de classements par Reciprocal Rank Fusion (RRF, Cormack et al. 2009, k = 60).

RRF n'utilise que les rangs : pas besoin de normaliser des scores BM25 et cosinus qui ne
sont pas comparables. Prendre ~30 candidats par liste, fusionner, puis reranker les 30 premiers.
"""

from __future__ import annotations

from collections.abc import Sequence


def rrf(classements: Sequence[Sequence[str]], k: int = 60, top_n: int | None = None) -> list[tuple[str, float]]:
    """Fusionne plusieurs listes d'ids triees (la meilleure en premier).

    score(d) = somme sur les listes de 1 / (k + rang), rang commencant a 1.
    """
    scores: dict[str, float] = {}
    for liste in classements:
        for rang, doc_id in enumerate(liste, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rang)
    fusion = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return fusion[:top_n] if top_n else fusion
