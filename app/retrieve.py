"""Find the abstracts most relevant to a question.

Three modes (see eval/run.py for how they compare):
  vector   meaning-based: cosine distance between embeddings, HNSW index
  keyword  word-based: PostgreSQL full-text search (stemmed words), IDF-weighted, GIN index
  hybrid   both lists fused with Reciprocal Rank Fusion (RRF)

Every mode returns at most one chunk per article, so one long article split into several
chunks cannot take several of the k slots.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import psycopg

from app.config import KEYWORD_MAX_DF, RETRIEVAL_MODE, TOP_K
from app.embeddings import embed_query

RRF_K = 60  # standard constant from Cormack et al. (2009); damps the weight of top ranks
MODES = ("vector", "keyword", "hybrid")


@dataclass
class Passage:
    pmid: str
    title: str
    year: int | None
    journal: str
    text: str
    score: float  # vector: cosine similarity · keyword: sum of IDF · hybrid: RRF score

    @property
    def url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"


# Each query returns (pmid, text, score), best first, one row per article.
# Inner ORDER BY ... LIMIT is what lets PostgreSQL use the index; DISTINCT ON then keeps
# the best chunk of each article.
_VECTOR_SQL = """
SELECT pmid, text, 1 - distance AS score FROM (
    SELECT DISTINCT ON (pmid) pmid, text, distance FROM (
        SELECT pmid, text, embedding <=> %(q)s AS distance
        FROM chunks ORDER BY embedding <=> %(q)s LIMIT %(pool)s
    ) nearest
    ORDER BY pmid, distance
) best
ORDER BY distance LIMIT %(k)s
"""

# Keyword search with IDF weighting (the core idea of BM25):
#   score(chunk) = sum of log(N / df) over the question's words that the chunk contains
# A rare word like "gadolinium" (10 chunks) counts far more than "mice" (226 chunks).
# PostgreSQL's built-in ts_rank has no IDF, which is why it is not used here.
# Words found in more than max_df of all chunks ("diseas", "alzheim" in this single-topic
# corpus) are dropped entirely. Words are stemmed by the English config ("labelled" -> "label").
_KEYWORD_SQL = """
WITH n AS (SELECT count(*)::float AS total FROM chunks),
terms AS (
    SELECT DISTINCT t.lexeme, ln(n.total / d.ndoc) AS idf
    FROM unnest(tsvector_to_array(to_tsvector('english', %(text)s))) AS t(lexeme)
    JOIN lexeme_df d USING (lexeme), n
    WHERE d.ndoc <= %(max_df)s * n.total
), scored AS (
    SELECT c.pmid, c.text, sum(t.idf) AS score
    FROM chunks c
    JOIN terms t ON c.tsv @@ to_tsquery('simple', quote_literal(t.lexeme))
    GROUP BY c.id, c.pmid, c.text
)
SELECT pmid, text, score FROM (
    SELECT DISTINCT ON (pmid) pmid, text, score FROM scored ORDER BY pmid, score DESC
) best
ORDER BY score DESC LIMIT %(k)s
"""

_DOCS_SQL = "SELECT pmid, title, year, journal FROM documents WHERE pmid = ANY(%s)"


def _vector(conn, q_vec, k):
    return conn.execute(_VECTOR_SQL, {"q": q_vec, "pool": k * 4, "k": k}).fetchall()


def _keyword(conn, question, k, max_df=KEYWORD_MAX_DF):
    params = {"text": question, "k": k, "max_df": max_df}
    return conn.execute(_KEYWORD_SQL, params).fetchall()


def rrf(*ranked_lists: list[tuple], k: int = RRF_K) -> list[tuple[str, str, float]]:
    """Reciprocal Rank Fusion: score(doc) = sum over lists of 1 / (k + rank).

    Uses only ranks, not raw scores, so a cosine similarity and an IDF sum (different
    scales) can be combined without tuning weights.
    """
    scores: dict[str, float] = {}
    texts: dict[str, str] = {}
    for ranked in ranked_lists:
        for rank, (pmid, text, _) in enumerate(ranked, start=1):
            scores[pmid] = scores.get(pmid, 0.0) + 1.0 / (k + rank)
            texts.setdefault(pmid, text)  # first list (vector) wins if both have the article
    order = sorted(scores, key=scores.get, reverse=True)
    return [(pmid, texts[pmid], scores[pmid]) for pmid in order]


def retrieve(
    conn: psycopg.Connection,
    question: str,
    k: int = TOP_K,
    mode: str = RETRIEVAL_MODE,
    query_vector: np.ndarray | None = None,
) -> list[Passage]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode != "keyword" and query_vector is None:
        query_vector = embed_query(question)

    if mode == "vector":
        rows = _vector(conn, query_vector, k)
    elif mode == "keyword":
        rows = _keyword(conn, question, k)
    else:
        # take deeper lists than k from each side, so an article ranked 8th by both can still win
        rows = rrf(_vector(conn, query_vector, k * 4), _keyword(conn, question, k * 4))[:k]

    meta = {r[0]: r[1:] for r in conn.execute(_DOCS_SQL, ([r[0] for r in rows],))}
    return [Passage(pmid, *meta[pmid], text, float(score)) for pmid, text, score in rows]
