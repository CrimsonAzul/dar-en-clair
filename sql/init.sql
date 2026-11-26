CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id         TEXT PRIMARY KEY,          -- ex. "cgi_2026:art_63#p2"
    article_id TEXT NOT NULL,             -- ex. "cgi_2026:art_63" (unite d'evaluation)
    loi        TEXT NOT NULL,
    article    TEXT NOT NULL,
    domaine    TEXT NOT NULL CHECK (domaine IN ('location', 'vente', 'fiscalite', 'copropriete')),
    edition    TEXT NOT NULL,
    contexte   TEXT NOT NULL DEFAULT '',
    langue     TEXT NOT NULL DEFAULT 'fr' CHECK (langue IN ('fr', 'ar')),
    texte      TEXT NOT NULL,
    embedding  vector(1024) NOT NULL      -- bge-m3, normalise
);

CREATE INDEX IF NOT EXISTS chunks_domaine_idx ON chunks (domaine);

-- Pas d'index HNSW volontairement : avec quelques milliers de chunks, la recherche exacte
-- prend quelques millisecondes, et un index approche ajouterait des erreurs de recherche
-- (surtout combine au filtre par domaine) qui fausseraient l'evaluation.
-- La recherche lexicale (BM25) est faite en Python : voir retrieval/lexical.py.
