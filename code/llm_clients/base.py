"""Shared LLMClient interface for all providers."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMPrediction:
    """Single LLM prediction on a single message."""

    message_id: str
    hazard: bool                       # True if model flagged as hazard
    action_label: str                  # raw model output, e.g. "Action 4"
    action_num: int                    # parsed 1-8 (or 0 if unparseable; treat as Action 1)
    rationale: str                     # brief model reasoning
    raw_response: str                  # raw API response (for forensics)
    inference_time_s: float
    model_version: str
    prompt_variant: str                # "safety" or "default"
    error: Optional[str] = None        # set if API call failed and was retried

    def to_csv_row(self) -> dict:
        """Convert to the canonical per-message CSV schema."""
        return {
            "message_id": self.message_id,
            "pred_hazard": 1 if self.hazard else 0,
            "pred_action": self.action_num,
            "pred_rationale": self.rationale,
            "model_version": self.model_version,
            "prompt_variant": self.prompt_variant,
            "inference_time_s": round(self.inference_time_s, 3),
            "error": self.error or "",
        }


_ACTION_RE = re.compile(r"action\s*([0-8])", re.IGNORECASE)


def parse_action_label(label: str) -> int:
    """Parse 'Action N' from a free-text label. Returns 0 if unparseable.

    Index 0 is the implementation sentinel; downstream code treats it as
    Action 1 (self-care/no action) for evaluation.
    """
    if not label:
        return 0
    m = _ACTION_RE.search(label)
    if m:
        return int(m.group(1))
    # Fallback: look for a bare digit
    for ch in label:
        if ch.isdigit() and ch in "12345678":
            return int(ch)
    return 0


def parse_llm_json(raw: str) -> dict:
    """Extract JSON from an LLM response, tolerant of code fences and prose.

    Returns dict with keys {hazard, action, rationale}. Missing keys default to safe values.
    """
    # Strip markdown code fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"\s*```\s*$", "", raw)

    # Try direct parse
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Find the first {...} block
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

    return {
        "hazard": bool(data.get("hazard", False)),
        "action": str(data.get("action", "Action 1")),
        "rationale": str(data.get("rationale", ""))[:500],
    }


class LLMClient:
    """Base class for provider clients. Subclasses implement `_call(prompt, message)`."""

    model_version: str = "unset"

    def __init__(self, prompt_variant: str = "safety"):
        from .prompts import get_system_prompt
        self.prompt_variant = prompt_variant
        self.system_prompt = get_system_prompt(prompt_variant)

    def _call(self, system_prompt: str, message: str) -> str:
        raise NotImplementedError("Subclass must implement _call(system_prompt, message)")

    def predict(self, message_id: str, message: str) -> LLMPrediction:
        """Run inference on a single message; return LLMPrediction."""
        start = time.time()
        error = None
        raw = ""
        try:
            raw = self._call(self.system_prompt, message)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            raw = ""
        elapsed = time.time() - start

        parsed = parse_llm_json(raw)
        action_num = parse_action_label(parsed["action"])

        return LLMPrediction(
            message_id=message_id,
            hazard=parsed["hazard"],
            action_label=parsed["action"],
            action_num=action_num,
            rationale=parsed["rationale"],
            raw_response=raw[:2000],  # truncate for storage
            inference_time_s=elapsed,
            model_version=self.model_version,
            prompt_variant=self.prompt_variant,
            error=error,
        )
