"""API tests. No database, embedding model or LLM is used: they are replaced with fakes.

`TestClient(app)` without a `with` block does not run the startup code, so we set
`app.state` ourselves.
"""

from contextlib import contextmanager

import pytest
import requests
from fastapi.testclient import TestClient

import app.main as main
from app.retrieve import Passage
from tests.test_answer import FakeLLM

PASSAGES = [Passage("111", "Tau PET study", 2021, "J1", "Tau PET predicts decline.", 0.9123)]


class FakePool:
    def __init__(self, fail=False):
        self.fail = fail

    @contextmanager
    def connection(self, timeout=None):
        if self.fail:
            raise ConnectionError("db down")

        class Conn:
            def execute(self, *_):
                class Cur:
                    def fetchone(self):
                        return (6007,)

                return Cur()

        yield Conn()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "retrieve", lambda conn, q: PASSAGES)
    main.app.state.pool = FakePool()
    return TestClient(main.app)


def test_health_reports_chunk_count(client):
    main.app.state.llm = FakeLLM()
    assert client.get("/health").json() == {"status": "ok", "chunks": 6007, "llm": "fake"}


def test_health_is_503_when_database_is_down(client):
    main.app.state.pool = FakePool(fail=True)
    assert client.get("/health").status_code == 503


def test_ask_returns_answer_sources_and_timings(client):
    main.app.state.llm = FakeLLM(
        {"found": True, "answer": "Tau PET predicts decline [111].", "citations": []}
    )
    body = client.post("/ask", json={"question": "  tau vs amyloid?  "}).json()
    assert body["question"] == "tau vs amyloid?"
    assert body["found"] is True
    assert body["sources"][0] == {
        "pmid": "111",
        "title": "Tau PET study",
        "journal": "J1",
        "year": 2021,
        "url": "https://pubmed.ncbi.nlm.nih.gov/111/",
        "score": 0.912,
    }
    assert set(body["timings"]) == {"retrieval_ms", "llm_ms"}


@pytest.mark.parametrize("question", ["", "ab", "x" * 501])
def test_ask_rejects_too_short_or_long_questions(client, question):
    assert client.post("/ask", json={"question": question}).status_code == 422


def test_ask_is_502_when_model_keeps_inventing_sources(client):
    bad = {"found": True, "answer": "Fact [999].", "citations": ["999"]}
    main.app.state.llm = FakeLLM(bad, bad)
    assert client.post("/ask", json={"question": "tau?"}).status_code == 502


def test_ask_is_503_when_llm_provider_is_down(client):
    class DownLLM:
        name = "down"

        def complete_json(self, system, user):
            raise requests.ConnectionError("no route")

    main.app.state.llm = DownLLM()
    assert client.post("/ask", json={"question": "tau?"}).status_code == 503


def test_index_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "NeuroLit" in r.text
