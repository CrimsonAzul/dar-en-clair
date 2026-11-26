"""Evalue la generation sur eval/questions.jsonl a partir des contextes deja calcules par
python -m darenclair.evaluation.recherche (pas de GPU ni de base ici : le LLM a la carte seul).

Mesures :
- pieges : part refusee (par le seuil du reranker ou par le LLM, qui repond HORS_CORPUS) ;
- questions normales : part repondue, precision et rappel des citations au niveau article,
  citations inventees ([S9] alors qu'il n'y a que 5 passages), reponses sans aucune citation ;
- calculs : le montant attendu (calcule a la main) apparait-il dans la reponse ?
Pas de juge LLM pour la fidelite : il faudrait d'abord le valider sur 20 reponses notees a la main.

Usage : python -m darenclair.evaluation.generation --split dev
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict
from pathlib import Path

from darenclair.evaluation.metrics import bootstrap_ic, citation_precision, citation_rappel
from darenclair.generation.llm import choisir_llm
from darenclair.generation.repondre import generer, refus_seuil
from darenclair.ingest.corpus import charger_chunks

RACINE = Path(__file__).resolve().parents[3]


def montant_present(texte: str, montant: float) -> bool:
    """36 000, 36.000, 36,000 ou 36000 : on compare les nombres sans separateurs de milliers."""
    nombres = {re.sub(r"[\s .,]", "", m) for m in re.findall(r"\d[\d\s .,]*\d|\d", texte)}
    return str(int(montant)) in nombres


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--questions", type=Path, default=RACINE / "eval" / "questions.jsonl")
    ap.add_argument("--n", type=int, default=5, help="passages donnes au LLM")
    a = ap.parse_args()

    recherche = json.loads((RACINE / "eval" / "results" / f"recherche_{a.split}.json").read_text(encoding="utf-8"))
    seuil = recherche["seuil_refus"]
    contextes = {d["id"]: d for d in recherche["questions"]}
    questions = [json.loads(l) for l in a.questions.read_text(encoding="utf-8").splitlines() if l.strip()]
    questions = [q for q in questions if q["split"] == a.split]
    chunks = {c.id: c for c in charger_chunks()}
    llm = choisir_llm()
    print(f"{len(questions)} questions ({a.split}), LLM {llm.nom}, seuil de refus {seuil:.4f}")

    resultats = []
    debut = time.time()
    for q in questions:
        ctx = contextes[q["id"]]
        if ctx["score_max"] < seuil:
            rep = refus_seuil(q["question"])
        else:
            ids = ctx["candidats"][: a.n]
            rep = generer(q["question"], [chunks[i] for i in ids], ctx["scores_reranker"][: a.n], llm)
        cites = {p["article_id"] for p in rep.passages if p["cite"]}
        resultats.append({
            "id": q["id"], "type": q["type"], "langue": q["langue"], "question": q["question"],
            "attendu": q["reponse_attendue"], "sources": q["sources"], "calcul_attendu": q.get("calcul_attendu"),
            "reponse": asdict(rep), "articles_cites": sorted(cites),
        })
        print(f"  {q['id']} {'REFUS (' + rep.motif_refus + ')' if rep.refus else 'repondu'}", flush=True)
    duree = time.time() - debut

    pieges = [r for r in resultats if not r["sources"]]
    normaux = [r for r in resultats if r["sources"]]
    repondus = [r for r in normaux if not r["reponse"]["refus"]]
    print(f"\n== {llm.nom}, {a.split} ({duree / len(resultats):.1f} s par question)")
    print(f"  pieges refuses              : {sum(r['reponse']['refus'] for r in pieges)}/{len(pieges)}"
          f"  (dont par le LLM : {sum(r['reponse']['motif_refus'] == 'llm' for r in pieges)})")
    print(f"  questions normales repondues : {len(repondus)}/{len(normaux)}"
          f"  (refus par le LLM : {sum(r['reponse']['motif_refus'] == 'llm' for r in normaux)})")
    if repondus:
        prec = [citation_precision(set(r["articles_cites"]), r["sources"]) for r in repondus]
        rapp = [citation_rappel(set(r["articles_cites"]), r["sources"]) for r in repondus]
        m, b, h = bootstrap_ic(prec)
        print(f"  precision des citations      : {m:.3f} [{b:.3f}, {h:.3f}]  (sur les reponses donnees)")
        m, b, h = bootstrap_ic(rapp)
        print(f"  rappel des citations         : {m:.3f} [{b:.3f}, {h:.3f}]")
        print(f"  reponses sans citation       : {sum(not r['articles_cites'] for r in repondus)}/{len(repondus)}")
        print(f"  citations inventees          : {sum(r['reponse']['citations_inventees'] for r in repondus)}")
    calculs = [r for r in resultats if r["calcul_attendu"]]
    for r in calculs:
        ok = montant_present(r["reponse"]["texte"], r["calcul_attendu"])
        outils = [c["outil"] for c in r["reponse"]["appels_outils"]]
        print(f"  calcul {r['id']} : attendu {r['calcul_attendu']:,.0f} DH, {'trouve' if ok else 'ABSENT'}"
              f" dans la reponse, outils appeles {outils}")

    sortie = RACINE / "eval" / "results" / f"generation_{a.split}_{llm.nom.replace(':', '_')}.json"
    sortie.write_text(json.dumps(resultats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nreponses completes : {sortie.relative_to(RACINE)}")


if __name__ == "__main__":
    main()
