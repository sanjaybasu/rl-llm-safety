"""DeepSeek-R1 client (local Ollama on Modal GPU container).

Optional row. Local inference; no API cost. Lower-priority comparator unless
Phase B literature review reveals reviewers expect open-source LLM coverage.
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


class DeepSeekR1Client(LLMClient):
    """DeepSeek-R1 via local Ollama (deepseek-r1:8b)."""

    model_version = "deepseek-r1:8b"

    def __init__(
        self,
        prompt_variant: str = "safety",
        ollama_base_url: Optional[str] = None,
    ):
        super().__init__(prompt_variant=prompt_variant)
        import httpx
        self._http = httpx.Client(
            base_url=ollama_base_url or os.environ.get(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            ),
            timeout=60.0,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call(self, system_prompt: str, message: str) -> str:
        response = self._http.post(
            "/api/chat",
            json={
                "model": self.model_version,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message},
                ],
                "options": {"temperature": 0.0, "seed": 42},
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")
