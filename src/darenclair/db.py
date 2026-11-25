"""Acces PostgreSQL / pgvector : insertion des chunks et deux recherches (dense, BM25-like)."""

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
    with conn.cursor() as cur:
        for c, e in zip(chunks, embeddings, strict=True):
            cur.execute(
                """INSERT INTO chunks (id, loi, article, domaine, edition, texte, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET texte = EXCLUDED.texte, embedding = EXCLUDED.embedding""",
                (c["id"], c["loi"], c["article"], c["domaine"], c["edition"], c["texte"], e),
            )
    conn.commit()


def recherche_dense(conn, vecteur, k: int = 20, domaine: str | None = None) -> list[str]:
    sql = "SELECT id FROM chunks {where} ORDER BY embedding <=> %s LIMIT %s"
    where, params = ("WHERE domaine = %s", [domaine]) if domaine else ("", [])
    with conn.cursor() as cur:
        cur.execute(sql.format(where=where), [*params, vecteur, k])
        return [r[0] for r in cur.fetchall()]


def recherche_lexicale(conn, question: str, k: int = 20, domaine: str | None = None) -> list[str]:
    """Plein texte francais de Postgres (ts_rank_cd), utilise comme equivalent BM25."""
    sql = """SELECT id FROM chunks, websearch_to_tsquery('french', %s) q
             WHERE tsv @@ q {filtre}
             ORDER BY ts_rank_cd(tsv, q) DESC LIMIT %s"""
    filtre, params = ("AND domaine = %s", [domaine]) if domaine else ("", [])
    with conn.cursor() as cur:
        cur.execute(sql.format(filtre=filtre), [question, *params, k])
        return [r[0] for r in cur.fetchall()]
