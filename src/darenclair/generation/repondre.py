"""Question -> recherche -> refus ou reponse du LLM avec citations [S#] -> sources verifiees."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from darenclair.evaluation.metrics import SOURCE_RE, citations_inventees
from darenclair.generation.outils import OUTILS, executer
from darenclair.generation.prompt import (
    HORS_CORPUS,
    MESSAGES_REFUS,
    SYSTEME,
    langue_question,
    message,
    reference,
)
from darenclair.ingest.parse import Chunk


@dataclass
class Reponse:
    texte: str
    refus: bool
    motif_refus: str | None = None          # "seuil" (score du reranker) ou "llm" (HORS_CORPUS)
    passages: list[dict] = field(default_factory=list)
    appels_outils: list[dict] = field(default_factory=list)
    citations_inventees: int = 0


def numeros_cites(texte: str) -> set[int]:
    return {int(n) for n in SOURCE_RE.findall(texte)}


def generer(question: str, passages: Sequence[Chunk], scores: Sequence[float], llm) -> Reponse:
    """Passages deja choisis (et refus par seuil deja ecarte) : appel du LLM et mise en forme."""
    sortie = llm.repondre(SYSTEME, message(question, passages), OUTILS, executer)
    # HORS_CORPUS n'importe ou : le modele ecrit parfois une reponse puis conclut qu'il ne sait pas.
    if not sortie.texte or HORS_CORPUS in sortie.texte:
        return Reponse(MESSAGES_REFUS[langue_question(question)], True, "llm", appels_outils=sortie.appels)
    cites = numeros_cites(sortie.texte)
    ids = [c.id for c in passages]
    return Reponse(
        texte=sortie.texte,
        refus=False,
        passages=[
            {"numero": i, "id": c.id, "article_id": c.article_id, "reference": reference(c),
             "contexte": c.contexte, "langue": c.langue, "texte": c.texte, "score": round(s, 4),
             "cite": i in cites}
            for i, (c, s) in enumerate(zip(passages, scores, strict=True), start=1)
        ],
        appels_outils=sortie.appels,
        citations_inventees=citations_inventees(sortie.texte, ids),
    )


def refus_seuil(question: str) -> Reponse:
    return Reponse(MESSAGES_REFUS[langue_question(question)], True, "seuil")


def repondre(question: str, recherche, llm, domaine: str | None = None, n: int = 5) -> Reponse:
    resultat = recherche.chercher(question, domaine)
    if resultat.refus:
        return refus_seuil(question)
    classes = resultat.classes[:n]
    return generer(question, recherche.passages(resultat, n), [s for _, s in classes], llm)
