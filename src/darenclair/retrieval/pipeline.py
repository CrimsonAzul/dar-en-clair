"""Pipeline de recherche : dense (bge-m3 + LoRA) -> reranker -> decision de refus.

Le meme objet sert a l'API et a l'evaluation : les chiffres publies mesurent le code livre.
La recherche dense est passee en fonction (pgvector en vrai, une fausse en test).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from darenclair.ingest.parse import Chunk
from darenclair.retrieval.hybrid import rrf
from darenclair.retrieval.lexical import BM25
from darenclair.retrieval.rerank import doit_refuser, reranker

CONFIG_DEFAUT = Path(__file__).resolve().parents[3] / "config" / "recherche.yaml"


def chemin_parametres() -> Path:
    # En Docker le paquet est installe dans site-packages : le chemin vient de RECHERCHE_CONFIG.
    return Path(os.getenv("RECHERCHE_CONFIG", CONFIG_DEFAUT))


def charger_parametres() -> dict:
    """config/recherche.yaml : seuil de refus (choisi sur dev), k candidats, n passages."""
    import yaml

    chemin = chemin_parametres()
    if not chemin.exists():
        return {}
    return yaml.safe_load(chemin.read_text(encoding="utf-8")) or {}

# (question, k, domaine ou None) -> ids de chunks tries du plus proche au moins proche
RechercheDense = Callable[[str, int, str | None], list[str]]
Reranker = Callable[[str, Sequence[tuple[str, str]], int | None], list[tuple[str, float]]]


@dataclass
class Resultat:
    candidats: list[str]                 # candidats denses, avant reranking
    classes: list[tuple[str, float]]     # tous les candidats apres reranking, tries
    refus: bool

    @property
    def score_max(self) -> float:
        return self.classes[0][1] if self.classes else 0.0


class Recherche:
    """Chemin livre : 30 candidats denses, reclasses par le reranker. BM25 et la fusion RRF
    restent disponibles comme references mesurees : sur le jeu dev (questions en langage
    courant), BM25 seul est faible et la fusion degrade l'ordre des candidats denses."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        dense: RechercheDense,
        seuil_refus: float,
        k: int = 30,
        rerank: Reranker = reranker,
    ):
        self.chunks = {c.id: c for c in chunks}
        self.dense_fn, self.seuil, self.k, self.rerank_fn = dense, seuil_refus, k, rerank
        self.par_domaine: dict[str, set[str]] = {}
        for c in chunks:
            self.par_domaine.setdefault(c.domaine, set()).add(c.id)

    @cached_property
    def bm25(self) -> BM25:
        return BM25({c.id: c.texte_index() for c in self.chunks.values()})

    def lexicale(self, question: str, domaine: str | None = None) -> list[str]:
        autorises = self.par_domaine.get(domaine, set()) if domaine else None
        return self.bm25.search(question, k=self.k, ids_autorises=autorises)

    def dense(self, question: str, domaine: str | None = None) -> list[str]:
        return self.dense_fn(question, self.k, domaine)

    def hybride(self, question: str, domaine: str | None = None) -> list[str]:
        fusion = rrf([self.dense(question, domaine), self.lexicale(question, domaine)], top_n=self.k)
        return [i for i, _ in fusion]

    def reclasser(self, question: str, candidats: list[str]) -> list[tuple[str, float]]:
        return self.rerank_fn(question, [(i, self.chunks[i].texte_index()) for i in candidats], None)

    def chercher(self, question: str, domaine: str | None = None) -> Resultat:
        candidats = self.dense(question, domaine)
        classes = self.reclasser(question, candidats)
        return Resultat(candidats, classes, doit_refuser(classes, self.seuil))

    def passages(self, resultat: Resultat, n: int = 5) -> list[Chunk]:
        """Les n meilleurs chunks apres reranking, dans l'ordre : ce seront [S1]..[Sn]."""
        return [self.chunks[i] for i, _ in resultat.classes[:n]]
