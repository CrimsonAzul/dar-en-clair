"""Encodeur bge-m3, avec ou sans l'adaptateur LoRA, partage par l'indexation, l'evaluation et l'API.

Les vecteurs des chunks (en base) et ceux des questions doivent venir du MEME encodeur :
l'indexation enregistre son nom dans la table meta, et l'API refuse de demarrer s'il differe.
"""

from __future__ import annotations

import os
from pathlib import Path

RACINE = Path(__file__).resolve().parents[3]
ADAPTATEUR_DEFAUT = RACINE / "models" / "bge-m3-lora"


def adaptateur_configure() -> Path | None:
    """ADAPTATEUR="" desactive le LoRA ; sinon chemin donne, ou models/bge-m3-lora s'il existe."""
    valeur = os.getenv("ADAPTATEUR")
    if valeur is not None:
        return Path(valeur) if valeur else None
    return ADAPTATEUR_DEFAUT if ADAPTATEUR_DEFAUT.exists() else None


def nom_encodeur(adaptateur: Path | None) -> str:
    base = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    return f"{base}+lora:{adaptateur.name}" if adaptateur else base


def charger_encodeur(adaptateur: Path | None):
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    m = SentenceTransformer(
        os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"), device=device, model_kwargs={"torch_dtype": dtype}
    )
    m.max_seq_length = 512
    if adaptateur:
        m[0].auto_model.load_adapter(str(adaptateur))
    return m


def encoder(m, textes: list[str], batch_size: int = 16, progression: bool = False):
    """Vecteurs normalises (numpy float32) : le produit scalaire est le cosinus."""
    return m.encode(
        textes, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=progression,
    ).astype("float32")
