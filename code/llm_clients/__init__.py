"""LLM clients for clinical hazard detection.

Each client implements the same interface:
    def predict(message: str, prompt_variant: str = "safety") -> LLMPrediction

Where prompt_variant ∈ {"safety", "default"} and LLMPrediction has:
    hazard: bool
    action: str (recommended action; mapped to 1-8 scale downstream)
    rationale: str (model's free-text reasoning, for audit)
    raw_response: str (raw API response, for forensics)
    inference_time_s: float
    model_version: str

Safety-augmented prompt: explicitly instructs the model to prioritize patient
safety and err toward escalation. Default prompt: same task description without
safety-prioritization language.

Determinism: all clients call with temperature=0 and a fixed seed where the API
supports it. Retries use exponential backoff with deterministic random seed
(seed=42 for jitter).
"""
from .base import LLMPrediction, LLMClient
from .prompts import SAFETY_AUGMENTED_PROMPT, DEFAULT_PROMPT
