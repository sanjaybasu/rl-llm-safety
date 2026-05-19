"""Gemini 3.1 Pro Preview client (Google).

Uses Google credits; secondary frontier LLM comparator per plan.
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


class GeminiProClient(LLMClient):
    """Gemini 3.1 Pro Preview with safety-augmented or default system prompt."""

    model_version = "gemini-3.1-pro-preview"

    def __init__(self, prompt_variant: str = "safety", api_key: Optional[str] = None):
        super().__init__(prompt_variant=prompt_variant)
        import google.generativeai as genai
        api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get(
            "GOOGLE_API_KEY"
        )
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) not set")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            model_name=self.model_version,
            system_instruction=self.system_prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": 512},
        )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call(self, system_prompt: str, message: str) -> str:
        # system_prompt was already set via system_instruction at model init
        response = self._model.generate_content(message)
        return response.text or ""
