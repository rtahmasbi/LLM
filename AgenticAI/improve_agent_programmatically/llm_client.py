"""
llm_client.py — Unified LLM client supporting OpenAI and Anthropic Claude.

Both generate_agents_md.py and improve_agents_md.py import this module.

Usage:
    from llm_client import build_client, LLMClient

    client = build_client()          # auto-detects from env vars
    client = build_client("openai")  # force OpenAI
    client = build_client("claude")  # force Claude

    # Chat completion
    text = client.chat(
        system="You are ...",
        user="Write me ...",
        max_tokens=4096,
        json_mode=True,   # OpenAI: response_format=json_object
                          # Claude: appends JSON instruction to system prompt
    )

    # Embeddings (OpenAI only — Claude has no embedding API)
    vectors = client.embed(["text one", "text two"])

Environment variables:
    OPENAI_API_KEY    → enables OpenAI backend
    ANTHROPIC_API_KEY → enables Claude backend

If both are set, OpenAI is preferred unless --provider claude is passed.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def _detect_provider() -> Optional[str]:
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "claude"
    return None


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Thin wrapper over OpenAI and Anthropic SDKs exposing a consistent interface.
    Only two methods: .chat() and .embed().
    """

    # Model defaults — override via build_client(model=...)
    OPENAI_CHAT_MODEL  = "gpt-4o"
    CLAUDE_CHAT_MODEL  = "claude-sonnet-4-5"
    OPENAI_EMBED_MODEL = "text-embedding-3-small"

    def __init__(self, provider: str, model: Optional[str] = None):
        self.provider = provider

        if provider == "openai":
            from openai import OpenAI
            self._openai = OpenAI()
            self.chat_model = model or self.OPENAI_CHAT_MODEL
            self.embed_model = self.OPENAI_EMBED_MODEL

        elif provider == "claude":
            import anthropic
            self._anthropic = anthropic.Anthropic()
            self.chat_model = model or self.CLAUDE_CHAT_MODEL

        else:
            raise ValueError(f"Unknown provider: {provider!r}. Choose 'openai' or 'claude'.")

    # ------------------------------------------------------------------
    # chat()
    # ------------------------------------------------------------------

    def chat(
        self,
        user: str,
        system: str = "",
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> str:
        """
        Send a chat request and return the text response.

        json_mode=True:
          OpenAI → sets response_format={"type": "json_object"}
          Claude → appends "Return valid JSON only." to the system prompt
                   and wraps the response in a JSON-extraction step.
        """
        if self.provider == "openai":
            return self._openai_chat(user, system, max_tokens, json_mode)
        else:
            return self._claude_chat(user, system, max_tokens, json_mode)

    def _openai_chat(
        self, user: str, system: str, max_tokens: int, json_mode: bool
    ) -> str:
        kwargs: dict = dict(
            model=self.chat_model,
            max_tokens=max_tokens,
            messages=[
                *([ {"role": "system", "content": system} ] if system else []),
                {"role": "user", "content": user},
            ],
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._openai.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()

    def _claude_chat(
        self, user: str, system: str, max_tokens: int, json_mode: bool
    ) -> str:
        effective_system = system
        if json_mode:
            json_instruction = (
                "\n\nReturn valid JSON only — no prose, no markdown fences, "
                "no explanation outside the JSON object."
            )
            effective_system = (system + json_instruction).strip()

        kwargs: dict = dict(
            model=self.chat_model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": user}],
        )
        if effective_system:
            kwargs["system"] = effective_system

        response = self._anthropic.messages.create(**kwargs)
        text = response.content[0].text.strip()

        # Claude sometimes wraps JSON in ```json ... ``` even when asked not to
        if json_mode:
            text = _strip_json_fences(text)

        return text

    # ------------------------------------------------------------------
    # embed()
    # ------------------------------------------------------------------

    def embed(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        """
        Embed a list of texts. Returns a list of float vectors.
        Only supported for OpenAI — raises NotImplementedError for Claude.
        """
        if self.provider == "claude":
            raise NotImplementedError(
                "Claude does not have an embeddings API. "
                "Use --provider openai for semantic clustering (Stage 3), "
                "or run with --no-cluster to skip it."
            )

        vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._openai.embeddings.create(
                model=self.embed_model,
                input=batch,
            )
            vectors.extend([r.embedding for r in response.data])
        return vectors

    @property
    def supports_embeddings(self) -> bool:
        return self.provider == "openai"

    def __repr__(self) -> str:
        return f"LLMClient(provider={self.provider!r}, model={self.chat_model!r})"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[LLMClient]:
    """
    Build an LLMClient from environment variables.

    provider: 'openai', 'claude', or None (auto-detect).
    Returns None if no API key is available and no provider forced.
    """
    resolved = provider or _detect_provider()

    if resolved is None:
        print(
            "[warn] No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY. "
            "Running without LLM — stages 3 and 4 disabled.",
            file=sys.stderr,
        )
        return None

    if resolved == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("[warn] --provider openai requested but OPENAI_API_KEY not set.", file=sys.stderr)
        return None

    if resolved == "claude" and not os.getenv("ANTHROPIC_API_KEY"):
        print("[warn] --provider claude requested but ANTHROPIC_API_KEY not set.", file=sys.stderr)
        return None

    try:
        client = LLMClient(resolved, model=model)
        print(f"[llm] Using {client}", file=sys.stderr)
        return client
    except ImportError as e:
        print(f"[warn] Could not load {resolved} SDK: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_json_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers Claude sometimes adds."""
    import re
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n", "", text)
    text = re.sub(r"\n```\s*$", "", text)
    return text.strip()
