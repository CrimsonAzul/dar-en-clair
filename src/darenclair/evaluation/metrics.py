"""Metriques de recherche et de generation."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence


def recall_at_k(recuperes: Sequence[str], pertinents: Iterable[str], k: int) -> float:
    """Part des articles pertinents presents dans les k premiers resultats."""
    pertinents = set(pertinents)
    if not pertinents:
        return 0.0
    return len(pertinents & set(recuperes[:k])) / len(pertinents)


def reciprocal_rank(recuperes: Sequence[str], pertinents: Iterable[str]) -> float:
    pertinents = set(pertinents)
    for rang, doc_id in enumerate(recuperes, start=1):
        if doc_id in pertinents:
            return 1.0 / rang
    return 0.0


def moyenne(valeurs: Iterable[float]) -> float:
    valeurs = list(valeurs)
    return sum(valeurs) / len(valeurs) if valeurs else 0.0


CITATION_RE = re.compile(r"\[([^\[\]]+?),\s*art\.?\s*([\w\- ]+?)\]", re.IGNORECASE)


def extraire_citations(reponse: str) -> set[tuple[str, str]]:
    """Recupere les citations au format [Loi 67-12, art. 34] -> {("loi 67-12", "34")}."""
    return {(loi.strip().lower(), art.strip().lower()) for loi, art in CITATION_RE.findall(reponse)}


def citation_precision(reponse: str, attendues: Iterable[tuple[str, str]]) -> float:
    """Part des citations de la reponse qui sont correctes. 0 si la reponse ne cite rien."""
    citees = extraire_citations(reponse)
    if not citees:
        return 0.0
    attendues = {(loi.lower(), art.lower()) for loi, art in attendues}
    return len(citees & attendues) / len(citees)
