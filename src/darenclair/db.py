"""Acces PostgreSQL / pgvector : stockage des chunks et recherche dense exacte."""

from __future__ import annotations

import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://darenclair:darenclair@localhost:5432/darenclair")

META_SQL = "CREATE TABLE IF NOT EXISTS meta (cle TEXT PRIMARY KEY, valeur TEXT NOT NULL)"


def connect():
    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    return conn


def remplacer(conn, chunks: list[dict], embeddings, encodeur: str) -> None:
    """Remplace tout l'index en une transaction et note l'encodeur utilise.
    chunks : dicts au format du JSONL ; embeddings : vecteurs de chunk.texte_index()."""
    lignes = [
        (c["id"], c["article_id"], c["loi"], c["article"], c["domaine"], c["edition"],
         c["contexte"], c.get("langue", "fr"), c["texte"], e)
        for c, e in zip(chunks, embeddings, strict=True)
    ]
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(META_SQL)
        cur.execute("TRUNCATE chunks")
        cur.executemany(
            """INSERT INTO chunks
                   (id, article_id, loi, article, domaine, edition, contexte, langue, texte, embedding)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            lignes,
        )
        cur.executemany(
            "INSERT INTO meta (cle, valeur) VALUES (%s, %s) "
            "ON CONFLICT (cle) DO UPDATE SET valeur = EXCLUDED.valeur",
            [("encodeur", encodeur), ("nb_chunks", str(len(lignes)))],
        )


def encodeur_indexe(conn) -> str | None:
    with conn.cursor() as cur:
        cur.execute(META_SQL)
        cur.execute("SELECT valeur FROM meta WHERE cle = 'encodeur'")
        ligne = cur.fetchone()
    conn.commit()
    return ligne[0] if ligne else None


def charger_chunks(conn) -> list[dict]:
    """Tous les chunks, pour construire l'index BM25 en memoire au demarrage."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, article_id, loi, article, domaine, edition, contexte, langue, texte FROM chunks"
        )
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]


def recherche_dense(conn, vecteur, k: int = 30, domaine: str | None = None) -> list[str]:
    """Plus proches voisins par distance cosinus (recherche exacte, pas d'index approche)."""
    where, params = ("WHERE domaine = %s", [domaine]) if domaine else ("", [])
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT id FROM chunks {where} ORDER BY embedding <=> %s, id LIMIT %s",
            [*params, vecteur, k],
        )
        return [r[0] for r in cur.fetchall()]
