"""Genere des questions en langage courant pour chaque chunk, avec un LLM local (Ollama).

Pour chaque chunk : 2 questions dans la langue de l'article + 1 dans l'autre langue
(francais <-> arabe), pour que le modele apprenne aussi la recherche entre langues.
Reprend la ou il s'est arrete : on peut l'interrompre et le relancer.

Usage : python -m darenclair.finetune.questions [--modele qwen2.5:7b] [--limite 20]
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

RACINE = Path(__file__).resolve().parents[3]
SORTIE = RACINE / "data" / "finetune" / "questions.jsonl"
OLLAMA = "http://localhost:11434/api/chat"

CONSIGNE = """Voici un extrait de loi marocaine ({loi}{contexte}, article {article}) :
\"\"\"
{texte}
\"\"\"

Ecris 3 questions differentes qu'un particulier (locataire, proprietaire, acheteur, vendeur,
coproprietaire, contribuable) pourrait poser, et auxquelles cet extrait repond.
Regles :
- langage courant, comme un message a un ami ou sur un forum ; un mot simple plutot que le terme juridique ;
- ne cite ni le numero de l'article ni le nom de la loi ;
- ne recopie pas les phrases de l'extrait ;
- "fr" : questions en francais ; "ar" : questions en arabe standard (pas en darija).
Reponds uniquement en JSON : {format}"""

# La langue est imposee par les cles du JSON : un modele 7B ignore souvent "la 3e en arabe",
# mais remplit bien une cle "ar".
FORMATS = {
    "fr": '{"fr": ["question 1", "question 2"], "ar": ["question 3"]}',
    "ar": '{"ar": ["question 1", "question 2"], "fr": ["question 3"]}',
}
FUITE_RE = re.compile(
    r"\b(article|art\.)\s*\d|المادة\s*\d|\bloi\s+(n\s*[°o.]?\s*)?\d",
    re.IGNORECASE,
)
ARABE_RE = re.compile("[ء-ي]")
# Une vraie personne ne parle pas de "ce texte" : la question trahit qu'elle a ete ecrite en lisant l'extrait.
REFERENCE_EXTRAIT_RE = re.compile(
    r"\b(ce texte|cet extrait|cet article|ce document|ce passage|cette loi|ce code|selon la loi|d'après la loi)\b"
    "|هذا القانون|لهذا القانون|لهذه القانون|هذه القانون"
    "|هذا النص|هذه المادة",  # هذا النص, هذه المادة
    re.IGNORECASE,
)


def langue_de(texte: str) -> str:
    return "ar" if len(ARABE_RE.findall(texte)) > len(re.findall(r"[A-Za-z]", texte)) else "fr"


def demander(modele: str, chunk: dict) -> list[str]:
    contexte = f", {chunk['contexte']}" if chunk["contexte"] else ""
    prompt = CONSIGNE.format(
        loi=chunk["loi"], contexte=contexte, article=chunk["article"],
        texte=chunk["texte"][:2500], format=FORMATS[chunk.get("langue", "fr")],
    )
    corps = json.dumps({
        "model": modele, "stream": False, "format": "json",
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.7, "num_ctx": 2048, "num_predict": 400},
    }).encode()
    req = urllib.request.Request(OLLAMA, corps, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        contenu = json.loads(r.read())["message"]["content"]
    try:
        donnees = json.loads(contenu)
    except json.JSONDecodeError:
        return []
    if not isinstance(donnees, dict):
        return []
    questions = []
    for cle in ("fr", "ar", "questions"):
        valeur = donnees.get(cle, [])
        questions += [valeur] if isinstance(valeur, str) else [q for q in valeur if isinstance(q, str)]
    return questions


def ecriture_valide(q: str) -> bool:
    """Seulement latin (avec accents), arabe, chiffres et ponctuation : un 7B derape parfois en
    chinois ou en espagnol au milieu d'une question."""
    for c in q:
        o = ord(c)
        if c.isalpha() and not (o < 0x250 or 0x600 <= o <= 0x6FF or 0xFB50 <= o <= 0xFEFF):
            return False
    return not re.search(r"\b(artículo|podrá|según|del|los|las)\b", q, re.IGNORECASE)


def filtrer(questions: list[str]) -> list[str]:
    """Ecarte les questions qui trahissent la source, qui sont trop courtes ou trop longues,
    ou qui ne sont ni en francais ni en arabe."""
    gardees, vues = [], set()
    for q in questions:
        q = " ".join(q.split())
        trahit = FUITE_RE.search(q) or REFERENCE_EXTRAIT_RE.search(q)
        if not 4 <= len(q.split()) <= 40 or trahit or not ecriture_valide(q) or q.lower() in vues:
            continue
        vues.add(q.lower())
        gardees.append(q)
    return gardees


def charger_chunks() -> list[dict]:
    chunks = []
    for f in sorted((RACINE / "data" / "processed").glob("*.jsonl")):
        chunks += [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modele", default="qwen2.5:7b")
    ap.add_argument("--limite", type=int, default=0, help="nombre max de chunks (0 = tous)")
    ap.add_argument("--sortie", type=Path, default=SORTIE)
    a = ap.parse_args()
    sortie = a.sortie

    sortie.parent.mkdir(parents=True, exist_ok=True)
    faits = set()
    if sortie.exists():
        faits = {json.loads(l)["chunk_id"] for l in sortie.read_text(encoding="utf-8").splitlines() if l}
    a_faire = [c for c in charger_chunks() if c["id"] not in faits]
    if a.limite:
        a_faire = a_faire[: a.limite]
    print(f"{len(faits)} chunks deja faits, {len(a_faire)} a faire")

    debut = time.time()
    with sortie.open("a", encoding="utf-8") as f:
        for i, c in enumerate(a_faire, 1):
            try:
                questions = filtrer(demander(a.modele, c))
            except Exception as e:  # serveur occupe, timeout : on passe, la reprise le refera
                print(f"  echec {c['id']} : {e}")
                continue
            for q in questions:
                f.write(json.dumps({"chunk_id": c["id"], "article_id": c["article_id"],
                                    "langue": langue_de(q), "question": q}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"chunk_id": c["id"], "article_id": c["article_id"], "fait": True}) + "\n")
            f.flush()
            if i % 25 == 0:
                vitesse = (time.time() - debut) / i
                print(f"  {i}/{len(a_faire)}  {vitesse:.1f} s/chunk  reste ~{vitesse * (len(a_faire) - i) / 60:.0f} min")


if __name__ == "__main__":
    main()
