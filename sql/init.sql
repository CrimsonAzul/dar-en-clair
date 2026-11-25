CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id        TEXT PRIMARY KEY,          -- ex. "loi_67-12:art_34"
    loi       TEXT NOT NULL,
    article   TEXT NOT NULL,
    domaine   TEXT NOT NULL CHECK (domaine IN ('location', 'vente', 'fiscalite', 'copropriete')),
    edition   TEXT NOT NULL,
    texte     TEXT NOT NULL,
    embedding vector(1024),              -- bge-m3 (dense)
    tsv       tsvector GENERATED ALWAYS AS (to_tsvector('french', texte)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_domaine_idx ON chunks (domaine);
-- Index HNSW pour la recherche dense (cosinus)
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
