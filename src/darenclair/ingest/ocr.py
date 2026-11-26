"""OCR des PDF scannes avec Tesseract (modele francais tessdata_best).

Chaque page est rendue a 300 dpi puis lue par Tesseract en mode "analyse de page" (psm 3), qui
detecte les colonnes du Bulletin officiel. La sortie TSV donne chaque mot avec sa position : on
reconstruit des `Ligne` comme pour un PDF texte, puis le meme nettoyage s'applique.

Le resultat est mis en cache (JSON) : l'OCR prend environ 5 s par page.
"""

from __future__ import annotations

import csv
import difflib
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from darenclair.ingest.pdf import Ligne

RACINE = Path(__file__).resolve().parents[3]
DPI = 300
CONFIANCE_MIN = 20  # mots en dessous : taches, tampons, bruit


def tesseract() -> str:
    for c in (os.getenv("TESSERACT_CMD"), shutil.which("tesseract"),
              r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
        if c and Path(c).exists():
            return c
    raise FileNotFoundError("Tesseract introuvable : installer Tesseract ou definir TESSERACT_CMD")


def tessdata() -> str:
    return os.getenv("TESSDATA_PREFIX", str(RACINE / ".tessdata"))


# --- corrections OCR des intitules (un "Articie 8" rate fait perdre tout l'article) ---------

INTITULE_OCR_RE = re.compile(r"^(\s*)([A-Za-z\u00c0-\u00ff1|]{5,9})(\s+)(?=\d|premier|PREMIER|Premier)")
ROMAIN_OCR_RE = re.compile(r"^(\s*(?:Section|SECTION|Chapitre|CHAPITRE|Titre|TITRE)\s+)([IVXHl1|]+)(?=[\s.\-:\u2014]|$)")


BRUIT_DEBUT_RE = re.compile(
    r"^\s*[.,;:!'\"*|‘’°]+\s*(?=(?:Arti|ARTI|Chapitre|CHAPITRE|Section|SECTION|Titre|TITRE))"
)


def corriger_intitule(texte: str) -> str:
    """ "Articie 8" -> "Article 8", "Section HI." -> "Section III.", ".Article 71" -> "Article 71" """
    texte = BRUIT_DEBUT_RE.sub("", texte)
    m = INTITULE_OCR_RE.match(texte)
    if m and m.group(2) not in ("Article", "ARTICLE"):
        mot = m.group(2)
        proche = difflib.SequenceMatcher(None, mot.lower(), "article").ratio() >= 0.75
        if mot[0] in "Aa" and 6 <= len(mot) <= 8 and proche:  # "Partie 2" n'est pas un article
            majuscules = sum(c.isupper() for c in mot) >= len(mot) - 1
            remplace = "ARTICLE" if majuscules else "Article"
            texte = m.group(1) + remplace + m.group(3) + texte[m.end():]
    m = ROMAIN_OCR_RE.match(texte)
    if m:
        romain = m.group(2).replace("H", "II").replace("l", "I").replace("1", "I").replace("|", "I")
        texte = m.group(1) + romain + texte[m.end():]
    return texte


# --- lecture -----------------------------------------------------------------------------

def lignes_depuis_tsv(tsv: str, hauteur_px: float) -> list[Ligne]:
    """Regroupe les mots du TSV Tesseract en lignes, dans l'ordre de lecture de Tesseract."""
    groupes: dict[tuple, list[dict]] = {}
    for r in csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
        if r.get("level") != "5" or not (r.get("text") or "").strip():
            continue
        if float(r["conf"]) < CONFIANCE_MIN:
            continue
        groupes.setdefault((int(r["block_num"]), int(r["par_num"]), int(r["line_num"])), []).append(r)
    lignes = []
    for cle in sorted(groupes):  # ordre bloc > paragraphe > ligne = ordre des colonnes
        mots = groupes[cle]
        haut = min(int(m["top"]) for m in mots)
        bas = max(int(m["top"]) + int(m["height"]) for m in mots)
        hauteurs = sorted(int(m["height"]) for m in mots)
        taille = round(hauteurs[len(hauteurs) // 2] * 72 / DPI, 1)  # en points, approximatif
        texte = corriger_intitule(" ".join(m["text"] for m in mots))
        lignes.append(Ligne(texte, haut / hauteur_px, bas / hauteur_px, taille))
    return lignes


def ocr_pdf(path: Path, langue: str = "fra") -> list[list[Ligne]]:
    import pymupdf

    exe, env = tesseract(), {**os.environ, "TESSDATA_PREFIX": tessdata()}
    pages: list[list[Ligne]] = []
    with pymupdf.open(path) as doc, tempfile.TemporaryDirectory() as tmp:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=DPI, colorspace=pymupdf.csGRAY)
            img = Path(tmp) / f"p{i}.png"
            pix.save(img)
            sortie = subprocess.run(
                # -c plutot que le nom de config "tsv" : ce fichier vit dans le tessdata
                # d'installation, absent quand TESSDATA_PREFIX pointe vers .tessdata/
                [exe, str(img), "stdout", "-l", langue, "--psm", "3", "-c", "tessedit_create_tsv=1"],
                capture_output=True, env=env, check=True,
            ).stdout.decode("utf-8")
            pages.append(lignes_depuis_tsv(sortie, pix.height))
    return pages


def ocr_pdf_en_cache(path: Path, cache_dir: Path, langue: str = "fra") -> list[list[Ligne]]:
    cache = cache_dir / f"ocr_{path.stem}.json"
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        donnees = json.loads(cache.read_text(encoding="utf-8"))
        return [[Ligne(**l) for l in p] for p in donnees]
    pages = ocr_pdf(path, langue)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps([[l.__dict__ for l in p] for p in pages], ensure_ascii=False), encoding="utf-8"
    )
    return pages
