"""Runtime settings (API key + model names) and chat-model construction."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# Models are served through OpenRouter's OpenAI-compatible API.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_AGENT_MODEL = "openai/gpt-4o"
DEFAULT_GUARD_MODEL = "openai/gpt-4o-mini"


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    agent_model: str = DEFAULT_AGENT_MODEL
    guard_model: str = DEFAULT_GUARD_MODEL

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)

    def signature(self) -> tuple:
        """Identity used to decide when cached agents must be rebuilt."""
        return (self.api_key, self.agent_model, self.guard_model)


def env_api_key() -> str | None:
    return os.getenv("OPENROUTER_KEY") or os.getenv("OPENROUTER_API_KEY") or None


# OpenRouter reserves credit for `max_tokens` up front; without a cap it assumes the
# model's maximum (16k for gpt-4o) and low-balance keys get a 402. Demo replies are short.
AGENT_MAX_TOKENS = 1024
GUARD_MAX_TOKENS = 300


def _llm(model: str, api_key: str | None, max_tokens: int) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=0,
        max_tokens=max_tokens,
        default_headers={"X-Title": "Guardrails Studio"},
    )


def agent_llm(settings: Settings) -> ChatOpenAI:
    """The main model that drives the agents."""
    return _llm(settings.agent_model, settings.api_key, AGENT_MAX_TOKENS)


def guard_llm(settings: Settings) -> ChatOpenAI:
    """A smaller, cheaper model used as a safety judge / classifier."""
    return _llm(settings.guard_model, settings.api_key, GUARD_MAX_TOKENS)
