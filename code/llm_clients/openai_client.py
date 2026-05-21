"""GPT-5.5 client (OpenAI) — Responses API.

GPT-5.5 returns "not a chat model" on /v1/chat/completions; this client uses
the /v1/responses endpoint directly. Supports safety-augmented and default
system prompt variants and an optional retrieval-augmented (RAG) variant
that mirrors `ClaudeRAGClient` (k-NN over the 1,280-example labeled training
corpus via the shared `RAGIndex`).
"""
from __future__ import annotations

import os
from typing import Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .base import LLMClient


class GPT55Client(LLMClient):
    """GPT-5.5 with safety-augmented or default system prompt (Responses API)."""

    model_version = "gpt-5.5"

    def __init__(self, prompt_variant: str = "safety", api_key: Optional[str] = None):
        super().__init__(prompt_variant=prompt_variant)
        from openai import OpenAI
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self._client = OpenAI(api_key=api_key)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call(self, system_prompt: str, message: str) -> str:
        # Responses API: system role becomes "developer"; no temperature.
        resp = self._client.responses.create(
            model=self.model_version,
            input=[
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": message},
            ],
        )
        return resp.output_text or ""
