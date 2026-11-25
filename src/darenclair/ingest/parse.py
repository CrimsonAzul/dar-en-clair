"""Decoupage d'un texte de loi en un chunk par article.

Usage :
    python -m darenclair.ingest.parse data/raw/loi_67-12.pdf --loi "Loi 67-12" \
        --domaine location --edition 2013 --out data/processed/loi_67-12.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

# "Article premier", "Article 12", "Article 12-1", "Article 12 bis", "Art. 63"
ARTICLE_RE = re.compile(
    r"^\s*(?:Article|Art\.)\s+(premier|\d+(?:[-.]\d+)*(?:\s+(?:bis|ter|quater))?)\b\s*[.:]?",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class Chunk:
    id: str
    loi: str
    article: str
    domaine: str
    edition: str
    texte: str


def normalise_numero(raw: str) -> str:
    raw = raw.strip().lower()
    return "1" if raw == "premier" else re.sub(r"\s+", " ", raw)


def split_articles(texte: str, loi: str, domaine: str, edition: str) -> list[Chunk]:
    """Coupe le texte a chaque en-tete d'article. Le preambule avant l'article 1 est ignore."""
    matches = list(ARTICLE_RE.finditer(texte))
    chunks: list[Chunk] = []
    vus: dict[str, int] = {}
    for i, m in enumerate(matches):
        fin = matches[i + 1].start() if i + 1 < len(matches) else len(texte)
        corps = texte[m.end():fin].strip()
        if not corps:
            continue
        numero = normalise_numero(m.group(1))
        # Un meme numero peut revenir (renvois, textes modificatifs) : on suffixe.
        vus[numero] = vus.get(numero, 0) + 1
        suffixe = "" if vus[numero] == 1 else f"#{vus[numero]}"
        cid = f"{loi.lower().replace(' ', '_')}:art_{numero.replace(' ', '_')}{suffixe}"
        chunks.append(Chunk(cid, loi, numero, domaine, edition, re.sub(r"[ \t]+", " ", corps)))
    return chunks


def pdf_to_text(path: Path) -> str:
    import fitz  # pymupdf, importe ici pour garder les tests legers

    with fitz.open(path) as doc:
        return "\n".join(page.get_text() for page in doc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--loi", required=True)
    ap.add_argument("--domaine", required=True)
    ap.add_argument("--edition", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    chunks = split_articles(pdf_to_text(a.pdf), a.loi, a.domaine, a.edition)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    print(f"{len(chunks)} articles ecrits dans {a.out}")


if __name__ == "__main__":
    main()
