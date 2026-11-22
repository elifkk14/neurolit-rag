"""LLM providers behind one small interface.

The rest of the code only calls `complete_json(system, user)`. Which provider answers is
decided by LLM_PROVIDER, so moving from a local model to a hosted one is a config change.
"""

from __future__ import annotations

import logging
import time
from typing import Protocol

import requests

from app import config

log = logging.getLogger("app.llm")


class LLMClient(Protocol):
    name: str

    def complete_json(self, system: str, user: str) -> str:
        """Return the model's reply, which must be a JSON object as text."""
        ...


class GroqClient:
    """Groq exposes an OpenAI-compatible chat completions API."""

    URL = "https://api.groq.com/openai/v1/chat/completions"
    DEFAULT_MODEL = "openai/gpt-oss-120b"

    def __init__(self, api_key: str, model: str = "", timeout: float = 60):
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set (see .env.example)")
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout
        self.name = f"groq:{self.model}"

    MAX_RETRIES = 3
    MAX_WAIT_S = 30  # never keep a web request waiting longer than this per retry

    def complete_json(self, system: str, user: str) -> str:
        # Free tier: 8,000 tokens/minute, i.e. ~3 questions/minute with 5 abstracts each.
        # On 429 Groq says how long to wait (Retry-After); we wait and retry a few times.
        for attempt in range(self.MAX_RETRIES + 1):
            resp = self._post(system, user)
            if resp.status_code != 429 or attempt == self.MAX_RETRIES:
                break
            wait = min(float(resp.headers.get("retry-after", 2**attempt)), self.MAX_WAIT_S)
            log.warning("Groq rate limit, waiting %.0f s (retry %d/%d)", wait, attempt + 1, self.MAX_RETRIES)
            time.sleep(wait)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _post(self, system: str, user: str) -> requests.Response:
        return requests.post(
            self.URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": 0,  # same question -> (nearly) same answer; we want facts, not creativity
                "response_format": {"type": "json_object"},  # the API guarantees syntactically valid JSON
            },
            timeout=self.timeout,
        )


class OllamaClient:
    """Local models through Ollama (https://ollama.com). Free and offline, but small models on a laptop."""

    DEFAULT_MODEL = "llama3.2:3b"

    def __init__(self, base_url: str, model: str = "", timeout: float = 180):
        self.base_url = base_url.rstrip("/")
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout
        self.name = f"ollama:{self.model}"

    def complete_json(self, system: str, user: str) -> str:
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "format": "json",
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


def get_llm() -> LLMClient:
    if config.LLM_PROVIDER == "groq":
        return GroqClient(config.GROQ_API_KEY, config.LLM_MODEL)
    if config.LLM_PROVIDER == "ollama":
        return OllamaClient(config.OLLAMA_URL, config.LLM_MODEL)
    raise ValueError(f"unknown LLM_PROVIDER: {config.LLM_PROVIDER!r} (use 'groq' or 'ollama')")
