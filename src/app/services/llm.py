"""LLM provider resolver — returns an OpenAI-compatible client + model name
for the requested provider (mimo). Clients are cached per provider."""
from __future__ import annotations

from openai import OpenAI

from app.core.config import settings

_clients: dict[str, OpenAI] = {}


def get_llm(provider: str | None = None) -> tuple[OpenAI, str]:
    """Return (client, model_name) for the given LLM provider."""
    provider = provider or settings.DEFAULT_LLM
    registry = settings.llm_providers()
    if provider not in registry:
        raise ValueError(f"LLM provider '{provider}' tidak dikenal. Pilihan: {list(registry)}")
    cfg = registry[provider]
    if not cfg["api_key"]:
        raise ValueError(f"API key untuk LLM '{provider}' belum diset (cek .env).")
    if provider not in _clients:
        _clients[provider] = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    return _clients[provider], cfg["model"]
