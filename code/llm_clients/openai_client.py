"""GPT-5.5 client (OpenAI) — OPTIONAL row.

Paid API. Default plan omits this row. Include only if Phase B literature
review reveals BMC MIDM reviewers commonly expect OpenAI parity in addition to
Anthropic + Google frontier coverage.
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
    """GPT-5.5 with safety-augmented or default system prompt."""

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
        # Use Flex tier where supported to minimize cost
        response = self._client.chat.completions.create(
            model=self.model_version,
            temperature=0.0,
            max_tokens=512,
            seed=42,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ],
        )
        return response.choices[0].message.content or ""
