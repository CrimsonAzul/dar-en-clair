"""Decoupage d'un texte de loi en chunks (un article, ou plusieurs morceaux si l'article est long).

Chaque chunk garde son contexte (Titre > Chapitre > Section) : on l'ajoute au texte indexe,
ce qui aide beaucoup la recherche quand l'article seul est ambigu ("le preneur", "ce delai").

Usage :
    python -m darenclair.ingest.parse data/raw/loi_67-12.pdf --loi "Loi 67-12" --slug loi_67-12 \
        --domaine location --edition 2013 --out data/processed/loi_67-12.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

from darenclair.ingest import arabe

ORDINAUX_LATINS = (
    "bis|ter|quater|quinquies|sexies|septies|octies|nonies|novies|decies|undecies|duodecies"
    "|terdecies|quaterdecies|quindecies|sexdecies|septdecies|octodecies|novodecies|vicies"
)
MOTS_PREMIER = "premier|premi\u00e8re|premiere|unique"
NUMERO = (
    rf"((?i:{MOTS_PREMIER})|\d+(?:[-.]\d+)*(?:[ \t]+(?i:{ORDINAUX_LATINS}))?)"
)
SEPARATEUR = r"[.:\-\u2013\u2014]"

# En-tete d'article en debut de ligne : "Article premier", "Article 2.- Le bail...", "ARTICLE 12-1",
# "Article 59 duodecies", "Article 4 2" (le 2 final est un appel de note, ignore).
# "Article" doit avoir une majuscule : "article 12 de la presente loi" (renvoi coupe en fin
# de ligne) ne doit pas couper, et un en-tete est toujours suivi d'un separateur ou d'une fin de ligne.
ARTICLE_RE = re.compile(
    rf"^[ \t]*(?:(?:Article|ARTICLE)(?:\.[ \t]*|[ \t]+)|Art\.[ \t]*){NUMERO}"
    rf"(?:[ \t]+\d{{1,2}}(?=[ \t]*$))?"
    rf"(?=[ \t]*(?:$|{SEPARATEUR}))"
    rf"[ \t]*\.?[ \t]*(?:[\-\u2013\u2014:][ \t]*)?",
    re.MULTILINE,
)

NIVEAUX = {"livre": 0, "titre": 1, "chapitre": 2, "section": 3, "sous-section": 4}
TITRE_RE = re.compile(
    r"^[ \t]*(LIVRE|Livre|TITRE|Titre|CHAPITRE|Chapitre|SECTION|Section|SOUS-SECTION|Sous-section)"
    rf"[ \t]+([IVXLC]+|\d+|(?i:{MOTS_PREMIER}|pr\u00e9liminaire|preliminaire))\b[^\n]{{0,120}}$",
    re.MULTILINE,
)

# Lignes parasites : numeros de page ("12", "| 12", "- 12 -"), en-tetes du Bulletin officiel.
PARASITES = [
    re.compile(r"^[ \t]*[|\-\u2013]?[ \t]*\d{1,4}[ \t]*[|\-\u2013]?[ \t]*$", re.MULTILINE),
    re.compile(r"^[^\n]*BULLETIN OFFICIEL[^\n]*$", re.MULTILINE | re.IGNORECASE),
]

# Appels de note colles au texte : "batis1", "copropriete18 :", "immatricules.3" en fin de ligne.
APPEL_MOT_RE = re.compile(r"\b([a-z\u00e0-\u00ff\u0153]+)(\d{1,3})(?!\w)")
APPEL_PONCTUATION_RE = re.compile(r"(?<=[a-z\u00e0-\u00ff\u0153)\u00bb][.;,])\d{1,3}(?=[ \t]*$)", re.MULTILINE)
UNITES = {"m", "cm", "mm", "dm", "km", "ha"}  # "m2", "km2" ne sont pas des appels de note

PHRASE_RE = re.compile(r"(?<=[.;:!?؛؟])\s+(?=[A-ZÀ-Ý0-9«\"(ء-ي])")
PUCES_RE = re.compile("[" + chr(0xE000) + "-" + chr(0xF8FF) + "]")  # zone a usage prive
MINUSCULE = r"[a-zà-ÿœ]"


@dataclass
class Chunk:
    id: str
    article_id: str
    loi: str
    article: str
    domaine: str
    edition: str
    contexte: str
    texte: str
    langue: str = "fr"  # langue du texte source : "fr" ou "ar"

    def texte_index(self) -> str:
        """Texte reellement embarque et indexe : en-tete de contexte + contenu."""
        etiquette = "المادة" if self.langue == "ar" else "Article"
        chemin = " > ".join(p for p in (self.loi, self.contexte, f"{etiquette} {self.article}") if p)
        return f"{chemin}\n{self.texte}"


def nettoyer(texte: str) -> str:
    # NFKC : ligatures des PDF ("ﬁ" -> "fi"), espaces insecables -> espaces
    texte = unicodedata.normalize("NFKC", texte).replace("­", "")  # + cesure invisible
    texte = PUCES_RE.sub("•", texte)  # puces en police symbole -> point
    texte = arabe.normaliser(texte)  # chiffres arabes, tatweel, mots coupes
    for motif in PARASITES:
        texte = motif.sub("", texte)
    texte = APPEL_MOT_RE.sub(lambda m: m.group(0) if m.group(1) in UNITES else m.group(1), texte)
    texte = APPEL_PONCTUATION_RE.sub("", texte)
    # Trait d'union en fin de ligne : sur ce corpus ce sont toujours de vrais mots composes
    # ("ci-dessus", "plus-values", "nue-propriete"), jamais des cesures : on garde le trait.
    return re.sub(rf"({MINUSCULE})-[ \t]*\n[ \t]*({MINUSCULE})", r"\1-\2", texte)


def normalise_numero(raw: str) -> str:
    raw = re.sub(r"\s+", " ", raw.strip().lower())
    return "1" if raw in {"premier", "première", "premiere"} else raw


def decouper(texte: str, max_mots: int = 350, chevauchement: int = 1) -> list[str]:
    """Coupe un texte long en morceaux de phrases entieres, avec une phrase de recouvrement.

    350 mots ~ 500 tokens : le reranker (512 a 1024 tokens) voit tout le morceau.
    """
    if len(texte.split()) <= max_mots:
        return [texte]
    unites: list[str] = []
    for phrase in PHRASE_RE.split(texte):
        mots = phrase.split()
        if len(mots) > max_mots:  # phrase interminable (listes d'alineas) : fenetres de mots
            unites += [" ".join(mots[i:i + max_mots]) for i in range(0, len(mots), max_mots)]
        elif mots:
            unites.append(phrase)

    morceaux: list[str] = []
    courant: list[str] = []
    n = 0
    for u in unites:
        lu = len(u.split())
        if courant and n + lu > max_mots:
            morceaux.append(" ".join(courant))
            courant = courant[-chevauchement:] if chevauchement else []
            n = sum(len(x.split()) for x in courant)
            if n + lu > max_mots:
                courant, n = [], 0
        courant.append(u)
        n += lu
    if courant:
        morceaux.append(" ".join(courant))
    return morceaux


ABROGE_RE = re.compile(r"^\(?\s*abrogée?s?\s*\)?\.?\s*(.*)$", re.IGNORECASE)


def _entier(numero: str) -> int | None:
    m = re.match(r"\d+", numero)
    return int(m.group()) if m else (1 if numero == "1" else None)


@dataclass
class Marqueur:
    sorte: str  # "article" ou "titre"
    debut: int
    fin: int
    entete: str  # texte brut de l'en-tete
    numero: str = ""  # articles
    niveau: int = 0  # titres


def trouver_marqueurs(texte: str) -> list[Marqueur]:
    """En-tetes d'articles et intitules, en francais et en arabe, dans l'ordre du texte."""
    ms = [Marqueur("article", m.start(), m.end(), m.group(0), numero=normalise_numero(m.group(1)))
          for m in ARTICLE_RE.finditer(texte)]
    ms += [Marqueur("article", m.start(), m.end(), m.group(0), numero=arabe.numero(m))
           for m in arabe.ARTICLE_AR_RE.finditer(texte) if not arabe.est_renvoi(texte, m.end())]
    ms += [Marqueur("titre", m.start(), m.end(), m.group(0), niveau=NIVEAUX[m.group(1).lower()])
           for m in TITRE_RE.finditer(texte)]
    ms += [Marqueur("titre", m.start(), m.end(), m.group(0), niveau=arabe.NIVEAUX_AR[m.group(1)])
           for m in arabe.TITRE_AR_RE.finditer(texte)]
    return sorted(ms, key=lambda x: x.debut)


def est_abroge(corps: str) -> bool:
    m = ABROGE_RE.match(corps) or arabe.ABROGE_AR_RE.match(corps)
    return bool(m) and len(m.groups()[-1].split()) < 15


def split_articles(
    texte: str,
    loi: str,
    domaine: str,
    edition: str,
    slug: str | None = None,
    max_mots: int = 350,
    depuis_article_1: bool = True,
    saut_renumerotation: int = 30,
    langue: str = "fr",
) -> list[Chunk]:
    """depuis_article_1 : ignore les "Article N" cites dans une preface avant l'article 1.
    saut_renumerotation : arrete le decoupage quand la numerotation repart en arriere de plus
    de N (annexes : le CGI reproduit ensuite les articles des lois de finances, numerotes de 1).
    Les articles abroges ("(abroge)", "نسخت") sont ignores."""
    texte = nettoyer(texte)
    slug = slug or re.sub(r"\s+", "_", loi.strip().lower())
    marqueurs = trouver_marqueurs(texte)

    if depuis_article_1:
        premier = next((i for i, x in enumerate(marqueurs) if x.sorte == "article" and x.numero == "1"), None)
        if premier is not None:
            marqueurs = [x for i, x in enumerate(marqueurs) if i >= premier or x.sorte == "titre"]

    fin_texte = len(texte)
    # Annexe apres le premier article (listes, baremes) : hors du texte de loi, on coupe.
    premier_article = next((x.debut for x in marqueurs if x.sorte == "article"), None)
    if premier_article is not None:
        annexe = arabe.ANNEXE_RE.search(texte, premier_article)
        if annexe:
            fin_texte = annexe.start()
            marqueurs = [x for x in marqueurs if x.debut < fin_texte]
    max_vu = 0
    for i, x in enumerate(marqueurs):
        n = _entier(x.numero) if x.sorte == "article" else None
        if n is None:
            continue
        if saut_renumerotation and max_vu >= saut_renumerotation and n < max_vu - saut_renumerotation:
            marqueurs, fin_texte = marqueurs[:i], min(fin_texte, x.debut)
            break
        max_vu = max(max_vu, n)

    contexte: dict[int, str] = {}
    chunks: list[Chunk] = []
    vus: dict[str, int] = {}
    for i, x in enumerate(marqueurs):
        fin = marqueurs[i + 1].debut if i + 1 < len(marqueurs) else fin_texte

        if x.sorte == "titre":
            lignes = [l.strip() for l in texte[x.fin:fin].splitlines() if l.strip()]
            libelle = " ".join(x.entete.split()).rstrip(" :")
            if lignes and len(lignes[0]) <= 150:  # intitule sur la ligne suivante
                libelle += f" : {lignes[0].lstrip(' :')}"
            contexte = {k: v for k, v in contexte.items() if k < x.niveau}
            contexte[x.niveau] = libelle
            continue

        corps = " ".join(texte[x.fin:fin].split())
        if not corps or est_abroge(corps):
            continue
        numero = x.numero
        # Un meme numero peut revenir (texte modificatif reproduit) : on suffixe.
        vus[numero] = vus.get(numero, 0) + 1
        article_id = f"{slug}:art_{numero.replace(' ', '_')}"
        if vus[numero] > 1:
            article_id += f"#dup{vus[numero]}"

        chemin = " > ".join(contexte[k] for k in sorted(contexte))
        morceaux = decouper(corps, max_mots)
        for j, morceau in enumerate(morceaux, start=1):
            cid = article_id if len(morceaux) == 1 else f"{article_id}#p{j}"
            chunks.append(Chunk(cid, article_id, loi, numero, domaine, edition, chemin, morceau, langue))
    return chunks


def main() -> None:
    from darenclair.ingest.pdf import pdf_to_text

    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--loi", required=True, help='nom affiche, ex. "Loi 67-12"')
    ap.add_argument("--slug", help="prefixe des ids, ex. loi_67-12 (doit correspondre a eval/)")
    ap.add_argument("--domaine", required=True)
    ap.add_argument("--edition", required=True)
    ap.add_argument("--max-mots", type=int, default=350)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    chunks = split_articles(pdf_to_text(a.pdf), a.loi, a.domaine, a.edition, a.slug, a.max_mots)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    n_articles = len({c.article_id for c in chunks})
    print(f"{n_articles} articles, {len(chunks)} chunks ecrits dans {a.out}")
    print("Verifier a la main quelques articles : un PDF mal extrait fausse toute l'evaluation.")


if __name__ == "__main__":
    main()
