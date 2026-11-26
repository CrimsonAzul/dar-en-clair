"""API FastAPI : POST /ask (recherche hybride, refus, reponse citee), GET /health.

Lancer en local (base demarree et indexee) : uvicorn darenclair.api.main:app --port 8000
Les modeles sont charges au premier appel de /ask, pas a l'import.
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from typing import Annotated, Literal

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from darenclair import DOMAINES

AVERTISSEMENT = "Réponse informative, pas un conseil juridique ou fiscal."

app = FastAPI(title="Dar en Clair")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    domaine: Literal["location", "vente", "fiscalite", "copropriete"] | None = None


class Passage(BaseModel):
    numero: int
    reference: str
    article_id: str
    contexte: str
    langue: str
    texte: str
    score: float
    cite: bool


class Reponse(BaseModel):
    reponse: str
    refus: bool
    motif_refus: str | None = None
    passages: list[Passage] = []
    calculs: list[dict] = []
    avertissement: str = AVERTISSEMENT


class Moteur:
    """Base, index BM25, encodeur, reranker et LLM, charges une fois. Un seul appel a la fois :
    le GPU et la connexion ne sont pas partages entre requetes."""

    def __init__(self):
        from darenclair import db
        from darenclair.generation.llm import choisir_llm
        from darenclair.ingest.parse import Chunk
        from darenclair.retrieval.dense import (
            adaptateur_configure,
            charger_encodeur,
            encoder,
            nom_encodeur,
        )
        from darenclair.retrieval.pipeline import Recherche, charger_parametres

        self.conn = db.connect()
        adaptateur = adaptateur_configure()
        indexe = db.encodeur_indexe(self.conn)
        if indexe != nom_encodeur(adaptateur):
            raise RuntimeError(f"index construit avec {indexe}, l'API encoderait avec {nom_encodeur(adaptateur)}")
        modele = charger_encodeur(adaptateur)
        params = charger_parametres()
        if "seuil_refus" not in params:
            raise RuntimeError("seuil de refus absent : lancer python -m darenclair.evaluation.recherche --split dev")

        def dense(question: str, k: int, domaine: str | None) -> list[str]:
            return db.recherche_dense(self.conn, encoder(modele, [question], batch_size=1)[0], k, domaine)

        chunks = [Chunk(**c) for c in db.charger_chunks(self.conn)]
        self.recherche = Recherche(chunks, dense, params["seuil_refus"], k=params.get("k_candidats", 30))
        self.n = params.get("n_passages", 5)
        self.llm = choisir_llm()
        self.verrou = threading.Lock()

    def repondre(self, question: str, domaine: str | None):
        from darenclair.generation.repondre import repondre

        with self.verrou:
            return repondre(question, self.recherche, self.llm, domaine, self.n)


@lru_cache(maxsize=1)
def moteur() -> Moteur:
    return Moteur()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "domaines": list(DOMAINES)}


@app.post("/ask", response_model=Reponse)
def ask(q: Question, m: Annotated[Moteur, Depends(moteur)]) -> Reponse:
    r = m.repondre(q.question, q.domaine)
    return Reponse(
        reponse=r.texte, refus=r.refus, motif_refus=r.motif_refus,
        passages=[Passage(**{k: v for k, v in p.items() if k != "id"}) for p in r.passages],
        calculs=r.appels_outils,
    )
