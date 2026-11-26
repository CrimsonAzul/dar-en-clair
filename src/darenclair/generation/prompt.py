"""Consignes du LLM et mise en forme des passages numerotes."""

from __future__ import annotations

import re
from collections.abc import Sequence

from darenclair.ingest.parse import Chunk

HORS_CORPUS = "HORS_CORPUS"

SYSTEME = f"""Tu es Dar en Clair, un assistant d'information sur le droit immobilier marocain \
(location, vente, fiscalite immobiliere, copropriete).

Regles :
1. Reponds uniquement a partir des passages numerotes [S1], [S2]... fournis avec la question. \
N'ajoute rien qui vienne de tes connaissances generales.
2. Apres chaque affirmation, cite le ou les passages qui la justifient, au format [S2]. \
Ne cite jamais un numero absent de la liste.
3. Verifie que les passages traitent la meme situation que la question (locataire ou bailleur, \
achat ou vente, bail d'habitation ou commercial). Si les passages ne permettent pas de repondre \
a la question posee, ecris seulement : {HORS_CORPUS}
4. Si la question demande un montant (impot sur une vente, impot sur des loyers, droits \
d'enregistrement) et qu'un outil correspond, appelle cet outil, meme si un passage donne le taux, \
puis reprends son resultat. Ne fais jamais le calcul toi-meme et n'ecris jamais l'appel d'outil \
dans ta reponse. Si l'outil renvoie une erreur, dis que le calcul n'est pas disponible.
5. Reponds dans la langue de la question (francais, arabe ou anglais), en phrases simples, \
150 mots au plus. Les passages en arabe ou en francais peuvent servir quelle que soit la langue de la question.
6. Pas de conseil personnalise ni de formule de politesse : l'application ajoute elle-meme un avertissement."""

MESSAGES_REFUS = {
    "fr": "Je ne trouve pas de réponse à cette question dans les textes couverts (lois 67-12, 49-16, "
          "39-08, 32-09, 18-00, 47-06 et Code général des impôts 2026).",
    "en": "I could not find an answer to this question in the texts covered (laws 67-12, 49-16, "
          "39-08, 32-09, 18-00, 47-06 and the 2026 General Tax Code).",
    "ar": "لم أجد جوابا لهذا السؤال في النصوص المشمولة (القوانين 67.12 و49.16 و39.08 و32.09 "
          "و18.00 و47.06 والمدونة العامة للضرائب 2026).",
}

_ARABE = re.compile("[؀-ۿ]")
_ANGLAIS = re.compile(r"\b(the|my|is|are|can|do|does|what|how|who|i|i'm|landlord|tenant|buy|pay)\b", re.I)


def langue_question(question: str) -> str:
    if _ARABE.search(question):
        return "ar"
    return "en" if len(_ANGLAIS.findall(question)) >= 2 else "fr"


def reference(c: Chunk) -> str:
    """Reference lisible d'un passage : "Loi 67-12, art. 34"."""
    return f"{c.loi}, art. {c.article}"


def message(question: str, passages: Sequence[Chunk]) -> str:
    blocs = []
    for i, c in enumerate(passages, start=1):
        chemin = f" ({c.contexte})" if c.contexte else ""
        blocs.append(f"[S{i}] {reference(c)}{chemin}\n{c.texte}")
    return "Passages :\n\n" + "\n\n".join(blocs) + f"\n\nQuestion : {question}"
