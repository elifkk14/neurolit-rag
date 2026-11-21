"""LLM providers behind one small interface.

The rest of the code only calls `complete_json(system, user)`. Which provider answers is
decided by LLM_PROVIDER, so moving from a local model to a hosted one is a config change.
"""

from __future__ import annotations

from typing import Protocol

import requests

from app import config


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

    def complete_json(self, system: str, user: str) -> str:
        resp = requests.post(
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
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


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
