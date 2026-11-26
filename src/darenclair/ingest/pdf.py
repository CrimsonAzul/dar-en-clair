"""Extraction du texte d'un PDF juridique en tenant compte de la mise en page.

On lit chaque ligne avec sa position et sa taille de police (pymupdf, mode "dict"), puis :
1. en-tetes et pieds de page : lignes des marges haute et basse repetees sur plusieurs pages ;
2. notes de bas de page : bloc de lignes en police plus petite en bas de page ;
3. appels de note : chiffres isoles plus petits que le reste de la ligne ("Article 4 2") ;
4. tables des matieres : pages entieres ecartees (sinon chaque article apparait deux fois).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

MARGE = 0.12  # part de la hauteur de page consideree comme marge haute ou basse

NUM_PAGE_RE = re.compile(r"^[\s|\-\u2013]*\d{1,4}[\s|\-\u2013]*$")
POINTILLES_RE = re.compile(r"(\.{5,}|-{5,}|…{2,}|_{5,})\s*\d*\s*$")
TITRE_SOMMAIRE_RE = re.compile(
    r"^\s*(table\s+des?\s+mati[eè]res?|sommaire|الفهرس|فهرس)\s*$", re.IGNORECASE
)
INTITULE_RE = re.compile(
    r"^\s*(?:(?:livre|titre|chapitre|section|sous-section|article)\b|المادة|القسم|الجزء|الباب|الفصل|الفرع)", re.IGNORECASE
)  # en arabe pas de \b : "المادة5" colle le numero au mot


@dataclass
class Ligne:
    texte: str
    y0: float  # haut de la ligne, en part de la hauteur de page (0 = haut)
    y1: float  # bas de la ligne
    taille: float  # taille de police de la ligne, appels de note exclus


ALEFS = "اأإآ"
LAM = "ل"


def texte_span(span: dict) -> str:
    """Texte d'un span "rawdict", avec reparation des ligatures lam-alef arabes.

    Certains PDF arabes (Word) extraient la ligature "la" a l'envers : "ال" au lieu de
    "لا" ("الأولى" devient "األولى"). On ne peut pas
    inverser tous les "ال" (c'est aussi l'article defini) ; mais dans la ligature, l'alef a une
    largeur nulle et le lam porte toute la largeur du glyphe. C'est ce signe qu'on utilise.
    """
    if "chars" not in span:
        return span.get("text", "")
    cs = span["chars"]
    sortie, i = [], 0
    while i < len(cs):
        c = cs[i]["c"]
        largeur = cs[i]["bbox"][2] - cs[i]["bbox"][0]
        if c in ALEFS and i + 1 < len(cs) and cs[i + 1]["c"] == LAM and largeur < 0.05:
            sortie.append(LAM + c)
            i += 2
            continue
        sortie.append(c)
        i += 1
    return "".join(sortie)


def lire_pdf(path: Path) -> list[list[Ligne]]:
    import pymupdf

    pages: list[list[Ligne]] = []
    with pymupdf.open(path) as doc:
        for page in doc:
            h = page.rect.height or 1.0
            lignes = []
            for bloc in page.get_text("rawdict")["blocks"]:
                for l in bloc.get("lines", []):
                    for s in l["spans"]:
                        s["text"] = texte_span(s)
                    # Les espaces sont parfois des spans a part ("Article", " ", "17") : on les garde
                    # pour assembler, mais seuls les spans non vides comptent pour les tailles.
                    pleins = [s for s in l["spans"] if s["text"].strip()]
                    if not pleins:
                        continue
                    tmax = max(s["size"] for s in pleins)

                    def appel(s: dict, tmax: float = tmax) -> bool:
                        return s["text"].strip().isdigit() and s["size"] < 0.85 * tmax

                    garde = [s for s in l["spans"] if not appel(s)]
                    if not any(s["text"].strip() for s in garde):
                        garde = l["spans"]
                    texte = "".join(s["text"] for s in garde).rstrip()
                    taille = round(max(s["size"] for s in garde if s["text"].strip()), 1)
                    lignes.append(Ligne(texte, l["bbox"][1] / h, l["bbox"][3] / h, taille))
            pages.append(lignes)
    return pages


def _cle(texte: str) -> str:
    """Forme normalisee pour reconnaitre un en-tete d'une page a l'autre (numeros ignores)."""
    return re.sub(r"\d+", "#", " ".join(texte.split())).lower()


def _en_marge(l: Ligne) -> bool:
    return l.y0 < MARGE or l.y1 > 1 - MARGE


def retirer_entetes(pages: list[list[Ligne]]) -> list[list[Ligne]]:
    """Les intitules (Article, Titre, Chapitre...) ne sont jamais retires : une fois les numeros
    normalises, "Article 12" et "Article 13" ont la meme cle et passeraient pour un en-tete."""
    n = len(pages)
    if n < 3:
        return pages

    def candidate(l: Ligne) -> bool:
        return not INTITULE_RE.match(l.texte)

    en_marge: Counter[str] = Counter()
    partout: Counter[str] = Counter()
    for p in pages:
        en_marge.update({_cle(l.texte) for l in p if _en_marge(l) and candidate(l)})
        partout.update({_cle(l.texte) for l in p if len(l.texte.strip()) >= 30 and candidate(l)})
    repetees_marge = {k for k, c in en_marge.items() if c >= max(3, 0.2 * n)}
    repetees = {k for k, c in partout.items() if c >= 0.6 * n}

    def garder(l: Ligne) -> bool:
        if not candidate(l):
            return True
        cle = _cle(l.texte)
        return not ((_en_marge(l) and cle in repetees_marge) or cle in repetees)

    return [[l for l in p if garder(l)] for p in pages]


def taille_corps(pages: list[list[Ligne]]) -> float:
    """Taille de police la plus frequente, ponderee par le nombre de caracteres."""
    compte: Counter[float] = Counter()
    for p in pages:
        for l in p:
            compte[round(l.taille * 2) / 2] += len(l.texte)
    return compte.most_common(1)[0][0] if compte else 0.0


def retirer_notes(lignes: list[Ligne], corps: float) -> list[Ligne]:
    """Retire le bloc de lignes en petite police au bas de la page (notes de bas de page)."""
    notes = set()
    for i in sorted(range(len(lignes)), key=lambda i: lignes[i].y0, reverse=True):
        l = lignes[i]
        if NUM_PAGE_RE.match(l.texte):
            continue
        if l.taille < corps - 0.6 and l.y0 > 0.4:
            notes.add(i)
            continue
        break
    return [l for i, l in enumerate(lignes) if i not in notes]


def est_sommaire(lignes: list[Ligne]) -> bool:
    textes = [l.texte.strip() for l in lignes if l.texte.strip()]
    if not textes:
        return False
    if any(TITRE_SOMMAIRE_RE.match(t) for t in textes[:8]):
        return True
    pointilles = sum(bool(POINTILLES_RE.search(t)) for t in textes)
    intitules = sum(bool(INTITULE_RE.match(t)) for t in textes)
    if pointilles >= 5 and (intitules >= 3 or pointilles >= 0.5 * len(textes)):
        return True
    numeros = sum(bool(NUM_PAGE_RE.match(t)) for t in textes)
    return numeros >= 0.3 * len(textes) and intitules >= 5


def assembler(pages: list[list[Ligne]]) -> str:
    pages = retirer_entetes(pages)
    corps = taille_corps(pages)
    pages = [retirer_notes(p, corps) for p in pages]
    pages = [p for p in pages if not est_sommaire(p)]
    return "\n".join("\n".join(l.texte for l in p) for p in pages)


def pdf_to_text(path: Path) -> str:
    return assembler(lire_pdf(path))
