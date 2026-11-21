"""Find the abstracts most similar in meaning to a question."""

from __future__ import annotations

from dataclasses import dataclass

import psycopg

from app.config import TOP_K
from app.embeddings import embed_query


@dataclass
class Passage:
    pmid: str
    title: str
    year: int | None
    journal: str
    text: str
    score: float  # cosine similarity, 1.0 = same direction

    @property
    def url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"


# 1. `nearest`: the k*4 closest chunks (this ORDER BY ... LIMIT is what uses the HNSW index)
# 2. `best`:    keep only the closest chunk of each article (DISTINCT ON), so one long
#               article split into several chunks cannot take several of the k slots
# 3. outer:     order by similarity again and keep k
_SQL = """
WITH nearest AS (
    SELECT pmid, text, embedding <=> %(q)s AS distance
    FROM chunks
    ORDER BY embedding <=> %(q)s
    LIMIT %(pool)s
), best AS (
    SELECT DISTINCT ON (pmid) pmid, text, distance
    FROM nearest
    ORDER BY pmid, distance
)
SELECT b.pmid, d.title, d.year, d.journal, b.text, 1 - b.distance AS score
FROM best b JOIN documents d USING (pmid)
ORDER BY b.distance
LIMIT %(k)s
"""


def retrieve(conn: psycopg.Connection, question: str, k: int = TOP_K) -> list[Passage]:
    q = embed_query(question)
    rows = conn.execute(_SQL, {"q": q, "pool": k * 4, "k": k}).fetchall()
    return [Passage(*row) for row in rows]
