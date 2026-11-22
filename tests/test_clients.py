"""HTTP clients tested with fake responses: no network, no API key."""

import pytest
import requests

import app.llm as llm
import ingest.pubmed as pubmed


class FakeResponse:
    def __init__(self, status=200, json_data=None, text=""):
        self.status_code = status
        self._json = json_data
        self.text = text

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


# ---------------------------------------------------------------- LLM clients


def test_groq_sends_json_mode_and_returns_content(monkeypatch):
    sent = {}

    def fake_post(url, headers, json, timeout):
        sent.update(url=url, headers=headers, body=json)
        return FakeResponse(json_data={"choices": [{"message": {"content": '{"found": false}'}}]})

    monkeypatch.setattr(llm.requests, "post", fake_post)
    out = llm.GroqClient("key-123").complete_json("sys", "user")

    assert out == '{"found": false}'
    assert sent["headers"]["Authorization"] == "Bearer key-123"
    assert sent["body"]["response_format"] == {"type": "json_object"}
    assert sent["body"]["temperature"] == 0
    assert [m["role"] for m in sent["body"]["messages"]] == ["system", "user"]


def test_groq_requires_api_key():
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        llm.GroqClient("")


def test_groq_http_error_propagates(monkeypatch):
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: FakeResponse(status=429))
    with pytest.raises(requests.HTTPError):
        llm.GroqClient("k").complete_json("s", "u")


def test_ollama_uses_json_format(monkeypatch):
    sent = {}

    def fake_post(url, json, timeout):
        sent.update(url=url, body=json)
        return FakeResponse(json_data={"message": {"content": "{}"}})

    monkeypatch.setattr(llm.requests, "post", fake_post)
    assert llm.OllamaClient("http://h:11434/").complete_json("s", "u") == "{}"
    assert sent["url"] == "http://h:11434/api/chat"
    assert sent["body"]["format"] == "json" and sent["body"]["stream"] is False


@pytest.mark.parametrize(("provider", "expected"), [("groq", llm.GroqClient), ("ollama", llm.OllamaClient)])
def test_get_llm_picks_provider_from_config(monkeypatch, provider, expected):
    monkeypatch.setattr(llm.config, "LLM_PROVIDER", provider)
    monkeypatch.setattr(llm.config, "GROQ_API_KEY", "k")
    assert isinstance(llm.get_llm(), expected)


def test_get_llm_rejects_unknown_provider(monkeypatch):
    monkeypatch.setattr(llm.config, "LLM_PROVIDER", "nope")
    with pytest.raises(ValueError, match="nope"):
        llm.get_llm()


# ---------------------------------------------------------------- PubMed client


@pytest.fixture
def no_sleep(monkeypatch):
    waits = []
    monkeypatch.setattr(pubmed.time, "sleep", waits.append)
    return waits


class ScriptedSession:
    """Returns the prepared responses (or raises the prepared exceptions) in order."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append((url, params))
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def test_eutils_adds_ncbi_parameters(no_sleep):
    session = ScriptedSession(FakeResponse(text="ok"))
    pubmed.EutilsClient("me@example.org", "abc", session).get("efetch.fcgi", {"id": "1"})
    params = session.calls[0][1]
    assert params["tool"] == "neurolit-rag" and params["email"] == "me@example.org"
    assert params["api_key"] == "abc" and params["db"] == "pubmed"


def test_eutils_retries_on_429_and_network_errors_then_succeeds(no_sleep):
    session = ScriptedSession(
        FakeResponse(status=429),
        requests.ConnectionError("dns"),
        FakeResponse(status=503),
        FakeResponse(text="ok"),
    )
    resp = pubmed.EutilsClient(None, None, session).get("esearch.fcgi", {})
    assert resp.text == "ok" and len(session.calls) == 4
    assert [w for w in no_sleep if w >= 2] == [2, 4, 8]  # exponential backoff


def test_eutils_does_not_retry_client_errors(no_sleep):
    session = ScriptedSession(FakeResponse(status=400))
    with pytest.raises(requests.HTTPError):
        pubmed.EutilsClient(None, None, session).get("esearch.fcgi", {})
    assert len(session.calls) == 1


def test_eutils_gives_up_after_max_retries(no_sleep):
    session = ScriptedSession(*[FakeResponse(status=500)] * 4)
    with pytest.raises(RuntimeError, match="after 4 attempts"):
        pubmed.EutilsClient(None, None, session).get("efetch.fcgi", {})


def test_download_takes_equal_share_per_year_and_skips_duplicates(tmp_path, monkeypatch):
    class FakeClient:
        def search(self, query):
            year = int(query.rsplit("AND ", 1)[1][:4])
            return 100, f"env{year}", "1"

        def fetch_batch(self, webenv, query_key, start, size):
            return webenv  # passed through to the fake parser below

    def fake_parse(webenv):
        year = int(webenv[3:])
        # the same article "dup" appears in both years
        return [pubmed.Article(f"{year}-a", "t", "a", year, "j"), pubmed.Article("dup", "t", "a", year, "j")]

    monkeypatch.setattr(pubmed, "parse_articles", fake_parse)
    out = tmp_path / "x.jsonl"
    n = pubmed.download(FakeClient(), "q", [2020, 2021], max_records=4, out_path=out)

    assert n == 3  # 2020-a, dup, 2021-a
    assert len(out.read_text().splitlines()) == 3
