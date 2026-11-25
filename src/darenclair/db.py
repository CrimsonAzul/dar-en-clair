"""Acces PostgreSQL / pgvector : stockage des chunks et recherche dense exacte."""

from __future__ import annotations

import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://darenclair:darenclair@localhost:5432/darenclair")


def connect():
    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    return conn


def inserer(conn, chunks: list[dict], embeddings) -> None:
    """chunks : dicts issus du JSONL ; embeddings : vecteurs bge-m3 de chunk.texte_index()."""
    lignes = [
        (c["id"], c["article_id"], c["loi"], c["article"], c["domaine"], c["edition"],
         c["contexte"], c["texte"], e)
        for c, e in zip(chunks, embeddings, strict=True)
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO chunks (id, article_id, loi, article, domaine, edition, contexte, texte, embedding)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (id) DO UPDATE SET
                   texte = EXCLUDED.texte, contexte = EXCLUDED.contexte, embedding = EXCLUDED.embedding""",
            lignes,
        )
    conn.commit()


def charger_chunks(conn) -> list[dict]:
    """Tous les chunks, pour construire l'index BM25 en memoire au demarrage."""
    with conn.cursor() as cur:
        cur.execute("SELECT id, article_id, loi, article, domaine, edition, contexte, texte FROM chunks")
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]


def recherche_dense(conn, vecteur, k: int = 30, domaine: str | None = None) -> list[str]:
    """Plus proches voisins par distance cosinus (recherche exacte, pas d'index approche)."""
    where, params = ("WHERE domaine = %s", [domaine]) if domaine else ("", [])
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT id FROM chunks {where} ORDER BY embedding <=> %s LIMIT %s",
            [*params, vecteur, k],
        )
        return [r[0] for r in cur.fetchall()]
