"""Load downloaded abstracts into PostgreSQL: documents + embedded chunks.

Re-running is safe: documents are upserted and their chunks are replaced.

Usage:
  docker compose up -d db
  python -m ingest.load data/raw/pubmed_20261119.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from app.config import DATABASE_URL
from app.embeddings import embed_passages
from ingest.chunk import chunk_abstract

log = logging.getLogger("ingest.load")
SCHEMA = Path(__file__).resolve().parent.parent / "db" / "schema.sql"
BATCH = 256  # documents per transaction


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load(conn: psycopg.Connection, docs: list[dict]) -> int:
    total_chunks = 0
    for i in range(0, len(docs), BATCH):
        batch = docs[i : i + BATCH]

        # 1. chunk every document, remember which document each chunk belongs to
        rows = []
        for doc in batch:
            for idx, text in enumerate(chunk_abstract(doc["title"], doc["abstract"])):
                rows.append((doc["pmid"], idx, text))

        # 2. embed all chunks of the batch in one call (much faster than one by one)
        vectors = embed_passages([text for _, _, text in rows])

        # 3. write in one transaction: either the whole batch is stored or nothing
        with conn.transaction(), conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO documents (pmid, title, abstract, year, journal)
                VALUES (%(pmid)s, %(title)s, %(abstract)s, %(year)s, %(journal)s)
                ON CONFLICT (pmid) DO UPDATE SET
                    title = EXCLUDED.title, abstract = EXCLUDED.abstract,
                    year = EXCLUDED.year, journal = EXCLUDED.journal
                """,
                batch,
            )
            cur.execute("DELETE FROM chunks WHERE pmid = ANY(%s)", ([d["pmid"] for d in batch],))
            cur.executemany(
                "INSERT INTO chunks (pmid, chunk_index, text, embedding) VALUES (%s, %s, %s, %s)",
                [(pmid, idx, text, vec) for (pmid, idx, text), vec in zip(rows, vectors, strict=True)],
            )
        total_chunks += len(rows)
        log.info("%d/%d documents, %d chunks", min(i + BATCH, len(docs)), len(docs), total_chunks)
    return total_chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    docs = read_jsonl(args.path)
    start = time.perf_counter()
    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute(SCHEMA.read_text())  # also creates the vector extension
        conn.commit()
        register_vector(conn)  # lets psycopg send numpy arrays as pgvector values
        n_chunks = load(conn, docs)
    log.info("done: %d documents, %d chunks in %.0f s", len(docs), n_chunks, time.perf_counter() - start)


if __name__ == "__main__":
    main()
