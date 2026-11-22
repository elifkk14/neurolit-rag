"""Integration tests against a real PostgreSQL + pgvector (a separate `neurolit_test` database).

They check what unit tests with fakes cannot: the SQL itself (HNSW query, DISTINCT ON,
upsert, cascade). Embeddings are replaced with hand-made vectors so results are exact.

Run with the compose database up:  docker compose up -d db && pytest -m integration
Skipped automatically when no database is reachable.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import psycopg
import pytest
from pgvector.psycopg import register_vector

import app.retrieve as retrieve_mod
import ingest.load as load_mod
from app.config import EMBED_DIM

pytestmark = pytest.mark.integration

ADMIN_URL = os.getenv("TEST_ADMIN_URL", "postgresql://neurolit:neurolit@localhost:5433/neurolit")
TEST_DB = "neurolit_test"
TEST_URL = ADMIN_URL.rsplit("/", 1)[0] + f"/{TEST_DB}"
SCHEMA = Path(__file__).resolve().parent.parent / "db" / "schema.sql"


def unit(*idx_weight: tuple[int, float]) -> np.ndarray:
    """A normalized 384-d vector with weight on chosen dimensions: easy to reason about cosine."""
    v = np.zeros(EMBED_DIM, dtype=np.float32)
    for i, w in idx_weight:
        v[i] = w
    return v / np.linalg.norm(v)


@pytest.fixture(scope="module")
def conn():
    try:
        admin = psycopg.connect(ADMIN_URL, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError:
        pytest.skip("no PostgreSQL on " + ADMIN_URL)
    with admin:
        admin.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
        admin.execute(f"CREATE DATABASE {TEST_DB}")
    c = psycopg.connect(TEST_URL)
    c.execute(SCHEMA.read_text())
    c.commit()
    register_vector(c)
    yield c
    c.close()


@pytest.fixture(autouse=True)
def clean(conn):
    conn.execute("TRUNCATE documents CASCADE")
    conn.commit()


def fake_embed(mapping: dict[str, np.ndarray]):
    """embed_passages replacement: the vector is chosen by a keyword in the chunk text."""

    def embed(texts):
        return np.stack([next(v for k, v in mapping.items() if k in t) for t in texts])

    return embed


def doc(pmid, title, abstract, year=2020):
    return {"pmid": pmid, "title": title, "abstract": abstract, "year": year, "journal": "J"}


def counts(conn):
    return conn.execute("SELECT (SELECT count(*) FROM documents), (SELECT count(*) FROM chunks)").fetchone()


def test_load_is_idempotent(conn, monkeypatch):
    monkeypatch.setattr(
        load_mod, "embed_passages", fake_embed({"tau": unit((0, 1)), "amyloid": unit((1, 1))})
    )
    docs = [doc("1", "tau", "Tau short."), doc("2", "amyloid", "Amyloid short.")]

    load_mod.load(conn, docs)
    load_mod.load(conn, docs)  # second run must not duplicate anything

    assert counts(conn) == (2, 2)


def test_reload_replaces_chunks_of_changed_document(conn, monkeypatch):
    monkeypatch.setattr(load_mod, "embed_passages", fake_embed({"tau": unit((0, 1))}))
    load_mod.load(conn, [doc("1", "tau", "Old abstract.")])
    load_mod.load(conn, [doc("1", "tau", "New abstract.")])

    texts = [r[0] for r in conn.execute("SELECT text FROM chunks WHERE pmid = '1'")]
    assert texts == ["tau\nNew abstract."]


def test_deleting_a_document_cascades_to_its_chunks(conn, monkeypatch):
    monkeypatch.setattr(load_mod, "embed_passages", fake_embed({"tau": unit((0, 1))}))
    load_mod.load(conn, [doc("1", "tau", "A.")])
    conn.execute("DELETE FROM documents WHERE pmid = '1'")
    assert counts(conn) == (0, 0)


def test_retrieve_orders_by_similarity_and_keeps_one_chunk_per_article(conn, monkeypatch):
    # Article "1" is long -> 2 chunks, both very close to the query. It must appear only once.
    long_abstract = " ".join(f"Tau sentence number {i} is here." for i in range(80))
    vectors = {
        "number 0 ": unit((0, 1.0)),  # chunk 1 of article 1: identical to the query
        "number 5": unit((0, 1.0), (2, 0.05)),  # chunk 2 of article 1: almost identical
        "amyloid": unit((0, 1.0), (1, 0.5)),  # article 2: close
        "hippocampus": unit((0, 0.2), (1, 1.0)),  # article 3: far
    }

    def embed(texts):
        out = []
        for t in texts:
            key = "number 0 " if "number 0 " in t else next(k for k in vectors if k in t and k != "number 0 ")
            out.append(vectors[key])
        return np.stack(out)

    monkeypatch.setattr(load_mod, "embed_passages", embed)
    load_mod.load(
        conn,
        [
            doc("1", "Tau", long_abstract),
            doc("2", "Amyloid", "amyloid only."),
            doc("3", "Hippo", "hippocampus only."),
        ],
    )
    assert conn.execute("SELECT count(*) FROM chunks WHERE pmid = '1'").fetchone()[0] >= 2

    monkeypatch.setattr(retrieve_mod, "embed_query", lambda q: unit((0, 1.0)))
    results = retrieve_mod.retrieve(conn, "tau?", k=3)

    assert [p.pmid for p in results] == ["1", "2", "3"]  # each article once, best first
    assert results[0].score == pytest.approx(1.0, abs=1e-4)
    assert results[0].score > results[1].score > results[2].score
    assert retrieve_mod.retrieve(conn, "tau?", k=2)[-1].pmid == "2"  # k is respected


def test_vector_query_uses_hnsw_index(conn, monkeypatch):
    monkeypatch.setattr(load_mod, "embed_passages", fake_embed({"x": unit((0, 1))}))
    load_mod.load(conn, [doc(str(i), "x", "x.") for i in range(50)])
    conn.execute("ANALYZE chunks")
    conn.execute("SET enable_seqscan = off")  # tiny table: force the planner to show its index choice
    plan = "\n".join(
        r[0]
        for r in conn.execute(
            "EXPLAIN SELECT pmid FROM chunks ORDER BY embedding <=> %s LIMIT 5", (unit((0, 1)),)
        )
    )
    conn.execute("RESET enable_seqscan")
    assert "chunks_embedding_hnsw" in plan
