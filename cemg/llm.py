"""
cemg/llm.py
───────────
Thin provider-agnostic LLM wrapper.

Supports:
  - Anthropic Claude  (set ANTHROPIC_API_KEY)
  - OpenAI / any OpenAI-compatible endpoint (set OPENAI_API_KEY + OPENAI_BASE_URL)
    This includes: Mistral, Together AI, Ollama (local), Groq, Anyscale …

Usage:
    from cemg.llm import get_llm, chat
    llm = get_llm()                 # auto-detects from env
    reply = chat(llm, messages=[...], system="...")
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


# ── Abstract base ─────────────────────────────────────────────────────────────
@dataclass
class LLMProvider(ABC):
    model: str

    @abstractmethod
    def complete(
        self,
        messages:   list[dict],
        system:     str  = "",
        max_tokens: int  = 1024,
        temperature:float = 0.3,
    ) -> str:
        """Return the assistant reply as a plain string."""


# ── Claude ────────────────────────────────────────────────────────────────────
@dataclass
class ClaudeProvider(LLMProvider):
    model: str = "claude-haiku-4-5-20251001"

    def complete(self, messages, system="", max_tokens=1024, temperature=0.3) -> str:
        import anthropic  # lazy import — not needed if using OpenAI
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        kwargs: dict = dict(
            model      = self.model,
            max_tokens = max_tokens,
            messages   = messages,
        )
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        return resp.content[0].text


# ── OpenAI-compatible ─────────────────────────────────────────────────────────
@dataclass
class OpenAIProvider(LLMProvider):
    model:    str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"

    def complete(self, messages, system="", max_tokens=1024, temperature=0.3) -> str:
        from openai import OpenAI  # lazy import
        client = OpenAI(
            api_key  = os.getenv("OPENAI_API_KEY", "not-needed"),
            base_url = self.base_url,
        )
        full_msgs = []
        if system:
            full_msgs.append({"role": "system", "content": system})
        full_msgs.extend(messages)
        
        api_model = self.model
        if api_model == "GPT 5.4 mini":
            api_model = "gpt-4o-mini"

        resp = client.chat.completions.create(
            model       = api_model,
            messages    = full_msgs,
            max_tokens  = max_tokens,
            temperature = temperature,
        )
        return resp.choices[0].message.content


# ── Auto-detect from environment ──────────────────────────────────────────────
def get_llm(override_model: Optional[str] = None) -> LLMProvider:
    """
    Returns the right provider based on available env variables.

    Priority:  ANTHROPIC_API_KEY  →  Claude Haiku
               OPENAI_API_KEY     →  OpenAI (or compatible) with OPENAI_BASE_URL
    """
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    openai_key    = os.getenv("OPENAI_API_KEY",    "")
    base_url      = os.getenv("OPENAI_BASE_URL",   "https://api.openai.com/v1")

    if anthropic_key and anthropic_key.startswith("sk-ant"):
        model = override_model or "claude-haiku-4-5-20251001"
        print(f"[CEMG] Using Claude — model: {model}")
        return ClaudeProvider(model=model)

    if openai_key or base_url != "https://api.openai.com/v1":
        model = override_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        print(f"[CEMG] Using OpenAI-compatible — model: {model}  base: {base_url}")
        return OpenAIProvider(model=model, base_url=base_url)

    raise RuntimeError(
        "No LLM provider configured. "
        "Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your .env file."
    )


# ── Convenience wrapper ───────────────────────────────────────────────────────
def chat(
    llm:         LLMProvider,
    messages:    list[dict],
    system:      str  = "",
    max_tokens:  int  = 1024,
    temperature: float = 0.3,
) -> str:
    """Thin wrapper so call-sites don't need to know the provider class."""
    return llm.complete(
        messages    = messages,
        system      = system,
        max_tokens  = max_tokens,
        temperature = temperature,
    )
