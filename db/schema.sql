-- Idempotent: safe to run on every load, also against a hosted database (Neon/Supabase).
CREATE EXTENSION IF NOT EXISTS vector;

-- One row per PubMed article.
CREATE TABLE IF NOT EXISTS documents (
    pmid     TEXT PRIMARY KEY,
    title    TEXT NOT NULL,
    abstract TEXT NOT NULL,
    year     INTEGER,
    journal  TEXT
);

-- One row per searchable piece of text. Most abstracts fit in one chunk; long ones are split.
CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    pmid        TEXT NOT NULL REFERENCES documents(pmid) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    embedding   vector(384) NOT NULL,          -- bge-small-en-v1.5 output size
    UNIQUE (pmid, chunk_index)
);

-- Approximate nearest-neighbour index. vector_cosine_ops -> queries must use the <=> operator.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_pmid ON chunks (pmid);

-- Keyword search (for hybrid retrieval). A generated column is recomputed by PostgreSQL
-- whenever `text` changes, so it can never be out of sync with the chunk.
ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;

CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);

-- How many chunks contain each word (lexeme): the "df" in IDF = log(N / df).
-- PostgreSQL's ts_rank has no IDF, so keyword search computes it from this table.
-- Refreshed by ingest.load after every load.
CREATE TABLE IF NOT EXISTS lexeme_df (
    lexeme TEXT PRIMARY KEY,
    ndoc   INTEGER NOT NULL
);
