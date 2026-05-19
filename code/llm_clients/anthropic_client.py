"""Claude Opus 4.7 client (Anthropic).

Uses Anthropic credits; preferred LLM comparator per plan.
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


class ClaudeOpusClient(LLMClient):
    """Claude Opus 4.7 with safety-augmented or default system prompt."""

    model_version = "claude-opus-4-7"

    def __init__(self, prompt_variant: str = "safety", api_key: Optional[str] = None):
        super().__init__(prompt_variant=prompt_variant)
        import anthropic
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.Anthropic(api_key=api_key)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call(self, system_prompt: str, message: str) -> str:
        response = self._client.messages.create(
            model=self.model_version,
            max_tokens=512,
            temperature=0.0,
            system=system_prompt,
            messages=[{"role": "user", "content": message}],
        )
        # Concatenate all text blocks in the response
        return "".join(
            block.text for block in response.content if hasattr(block, "text")
        )
