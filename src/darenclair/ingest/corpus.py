"""Decoupe tout le corpus decrit dans config/corpus.yaml et affiche un controle par texte.

Usage : python -m darenclair.ingest.corpus
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from darenclair.ingest.parse import Chunk, split_articles
from darenclair.ingest.pdf import pdf_to_text

RACINE = Path(__file__).resolve().parents[3]


def charger_chunks() -> list[Chunk]:
    """Tous les chunks decoupes (data/processed/*.jsonl), dans un ordre stable."""
    chunks = []
    for f in sorted((RACINE / "data" / "processed").glob("*.jsonl")):
        lignes = f.read_text(encoding="utf-8").splitlines()
        chunks += [Chunk(**json.loads(ligne)) for ligne in lignes if ligne.strip()]
    return chunks


def numeros_manquants(articles: list[str]) -> list[int]:
    entiers = {int(m.group()) for a in articles if (m := re.match(r"\d+", a))}
    return [i for i in range(1, max(entiers, default=0) + 1) if i not in entiers]


def main() -> None:
    import yaml

    textes = yaml.safe_load((RACINE / "config" / "corpus.yaml").read_text(encoding="utf-8"))
    sortie = RACINE / "data" / "processed"
    sortie.mkdir(parents=True, exist_ok=True)
    for t in textes:
        pdf = RACINE / "data" / "raw" / t["fichier"]
        if not pdf.exists():
            print(f"{t['loi']:<10} absent ({pdf.name})")
            continue
        texte = pdf_to_text(pdf)
        ocr = len(texte.strip()) < 500
        if ocr:  # PDF scanne : pas de couche texte
            from darenclair.ingest.ocr import ocr_pdf_en_cache
            from darenclair.ingest.pdf import assembler

            print(f"{t['loi']:<10} PDF scanne, OCR en cours (environ 5 s par page)...")
            texte = assembler(ocr_pdf_en_cache(pdf, sortie))
        chunks = split_articles(
            texte, t["loi"], t["domaine"], t["edition"], slug=t["slug"], langue=t.get("langue", "fr")
        )
        with (sortie / f"{t['slug']}.jsonl").open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        articles = list(dict.fromkeys(c.article for c in chunks))
        doublons = sum("#dup" in c.article_id for c in chunks)
        manquants = numeros_manquants(articles)
        print(
            f"{t['loi']:<10} {len(articles):>4} articles {len(chunks):>4} chunks"
            f"  doublons {doublons}  numeros absents {manquants[:10]}{'...' if len(manquants) > 10 else ''}"
            + ("  (OCR : relire quelques articles)" if ocr else "")
        )
    print("Numeros absents : souvent des articles abroges. Verifier les autres dans le PDF.")


if __name__ == "__main__":
    main()
