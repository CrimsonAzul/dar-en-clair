"""API FastAPI. /ask sera branche sur la recherche hybride + generation (jours 7 a 9)."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from darenclair import DOMAINES

app = FastAPI(title="Dar en Clair")

AVERTISSEMENT = "Reponse informative, pas un conseil juridique ou fiscal."


class Question(BaseModel):
    question: str
    domaine: str | None = None


class Reponse(BaseModel):
    reponse: str
    sources: list[str]
    avertissement: str = AVERTISSEMENT


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "domaines": list(DOMAINES)}


@app.post("/ask", response_model=Reponse)
def ask(q: Question) -> Reponse:
    # TODO jour 7 : recherche hybride -> reranker -> LLM avec citations et outils fiscaux
    return Reponse(reponse="Pas encore implemente.", sources=[])
