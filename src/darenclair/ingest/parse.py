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
from dataclasses import asdict, dataclass
from pathlib import Path

NUMERO = r"(premier|unique|\d+(?:[-.]\d+)*(?:[ \t]+(?:bis|ter|quater|quinquies))?)"
SEPARATEUR = r"[.:\-\u2013\u2014]"

# En-tete d'article en debut de ligne : "Article premier", "Article 2.- Le bail...", "ARTICLE 12-1".
# Pas d'IGNORECASE : "article 12 de la presente loi" (renvoi coupe en fin de ligne) ne doit pas couper.
ARTICLE_RE = re.compile(
    rf"^[ \t]*(?:Article|ARTICLE|Art\.)[ \t]+{NUMERO}(?=[ \t]*(?:$|{SEPARATEUR}))"
    rf"[ \t]*\.?[ \t]*(?:[\-\u2013\u2014:][ \t]*)?",
    re.MULTILINE,
)

NIVEAUX = {"titre": 0, "chapitre": 1, "section": 2, "sous-section": 3}
TITRE_RE = re.compile(
    r"^[ \t]*(TITRE|Titre|CHAPITRE|Chapitre|SECTION|Section|SOUS-SECTION|Sous-section)"
    r"[ \t]+([IVXLC]+|premier|PREMIER|unique|UNIQUE|\d+)\b[^\n]{0,120}$",
    re.MULTILINE,
)

# Lignes parasites des PDF du Bulletin officiel : numeros de page, en-tetes de page.
PARASITES = [
    re.compile(r"^[ \t]*\d{1,4}[ \t]*$", re.MULTILINE),
    re.compile(r"^[^\n]*BULLETIN OFFICIEL[^\n]*$", re.MULTILINE | re.IGNORECASE),
]

PHRASE_RE = re.compile(r"(?<=[.;:!?])\s+(?=[A-ZÀ-Ý0-9«\"(])")
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

    def texte_index(self) -> str:
        """Texte reellement embarque et indexe : en-tete de contexte + contenu."""
        chemin = " > ".join(p for p in (self.loi, self.contexte, f"Article {self.article}") if p)
        return f"{chemin}\n{self.texte}"


def nettoyer(texte: str) -> str:
    texte = texte.replace("­", "")  # cesure invisible
    for motif in PARASITES:
        texte = motif.sub("", texte)
    # Mot coupe en fin de ligne : "loca-\ntaire" -> "locataire"
    return re.sub(rf"({MINUSCULE})-\n({MINUSCULE})", r"\1\2", texte)


def normalise_numero(raw: str) -> str:
    raw = re.sub(r"\s+", " ", raw.strip().lower())
    return "1" if raw == "premier" else raw


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


def split_articles(
    texte: str,
    loi: str,
    domaine: str,
    edition: str,
    slug: str | None = None,
    max_mots: int = 350,
) -> list[Chunk]:
    texte = nettoyer(texte)
    slug = slug or re.sub(r"\s+", "_", loi.strip().lower())

    marqueurs = [("article", m) for m in ARTICLE_RE.finditer(texte)]
    marqueurs += [("titre", m) for m in TITRE_RE.finditer(texte)]
    marqueurs.sort(key=lambda x: x[1].start())

    contexte: dict[int, str] = {}
    chunks: list[Chunk] = []
    vus: dict[str, int] = {}
    for i, (sorte, m) in enumerate(marqueurs):
        fin = marqueurs[i + 1][1].start() if i + 1 < len(marqueurs) else len(texte)

        if sorte == "titre":
            niveau = NIVEAUX[m.group(1).lower()]
            lignes = [l.strip() for l in texte[m.end():fin].splitlines() if l.strip()]
            libelle = " ".join(m.group(0).split())
            if lignes and len(lignes[0]) <= 150:  # intitule sur la ligne suivante
                libelle += f" : {lignes[0]}"
            contexte = {k: v for k, v in contexte.items() if k < niveau}
            contexte[niveau] = libelle
            continue

        corps = " ".join(texte[m.end():fin].split())
        if not corps:
            continue
        numero = normalise_numero(m.group(1))
        # Un meme numero peut revenir (texte modificatif reproduit) : on suffixe.
        vus[numero] = vus.get(numero, 0) + 1
        article_id = f"{slug}:art_{numero.replace(' ', '_')}"
        if vus[numero] > 1:
            article_id += f"#dup{vus[numero]}"

        chemin = " > ".join(contexte[k] for k in sorted(contexte))
        morceaux = decouper(corps, max_mots)
        for j, morceau in enumerate(morceaux, start=1):
            cid = article_id if len(morceaux) == 1 else f"{article_id}#p{j}"
            chunks.append(Chunk(cid, article_id, loi, numero, domaine, edition, chemin, morceau))
    return chunks


def pdf_to_text(path: Path) -> str:
    import fitz  # pymupdf, importe ici pour garder les tests legers

    with fitz.open(path) as doc:
        return "\n".join(page.get_text() for page in doc)


def main() -> None:
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
