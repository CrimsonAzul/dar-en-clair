"""Calcule les vecteurs des chunks et remplit pgvector (remplace l'index existant).

Encodeur : bge-m3 avec l'adaptateur LoRA de models/bge-m3-lora s'il existe (choisi sur la
validation synthetique et le jeu dev, jamais sur le test). ADAPTATEUR="" pour bge-m3 de base.

Usage : docker compose up -d db && python -m darenclair.ingest.indexer
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict

from darenclair import db
from darenclair.ingest.corpus import charger_chunks
from darenclair.retrieval.dense import adaptateur_configure, charger_encodeur, encoder, nom_encodeur


def main() -> None:
    chunks = charger_chunks()
    adaptateur = adaptateur_configure()
    nom = nom_encodeur(adaptateur)
    print(f"{len(chunks)} chunks, encodeur {nom}")
    debut = time.time()
    vecteurs = encoder(charger_encodeur(adaptateur), [c.texte_index() for c in chunks], progression=True)
    print(f"vecteurs calcules en {time.time() - debut:.0f} s")
    with db.connect() as conn:
        db.remplacer(conn, [asdict(c) for c in chunks], vecteurs, nom)
    print(f"index remplace : {len(chunks)} chunks dans pgvector")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)  # Windows : le processus reste bloque a la fermeture apres usage de CUDA
