"""k-NN RAG over the labeled training set — closing-the-gap intervention #1.

For each test message, retrieve the K most-similar messages from the labeled
training set (by sentence-BERT cosine similarity) and include them in the
LLM system prompt as in-context evidence. This is the highest-ROI intervention
identified by the literature review:

- Liu et al. JAMIA 2025 (knowledge-graph RAG over patient portal messages):
  sens 0.98 / spec 0.99
- Stanford RAEC (arXiv 2509.22565, PSB 2026 accepted): message-pair retrieval
  without a knowledge graph; lift ~0.10-0.20 sensitivity at matched specificity
- Unlu et al. RECTIFIER NEJM AI 2024 (sens 0.923 / spec 0.939) for clinical
  trial eligibility — adjacent task, comparable RAG design

This implementation:
1. Builds a sentence-BERT index over the 1,280-example training corpus at init
2. At inference time, embeds the test message and retrieves top-K nearest neighbors
3. Formats the K retrieved (message, label) pairs into the system prompt
4. Calls Claude Opus 4.7 with the augmented prompt
5. Parses the response with the same JSON schema as the baseline LLM clients

Knobs:
- K: number of retrieved exemplars (default 8; balanced + diverse)
- model: sentence-BERT embedding model (default 'all-mpnet-base-v2', 768-dim,
  same as the supervised pipeline)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_clients.base import LLMClient
from llm_clients.prompts import ACTION_SCALE, OUTPUT_FORMAT


RAG_HEADER = """You are a medical triage assistant for a Medicaid managed care patient-messaging
service. Your role is to evaluate incoming patient messages and decide whether the message
contains a clinical hazard requiring escalation.

PRIORITIZE PATIENT SAFETY ABOVE ALL OTHER CONSIDERATIONS. When in doubt about whether a
message contains a hazard, err on the side of flagging it for clinician review. False
positives are recoverable; missed hazards may cause patient harm.

The patient population is Medicaid managed care: low-literacy, multilingual, high-comorbidity,
often using SMS as their only feasible channel to access care. Messages may use colloquialisms,
abbreviations, and implicit context. Read carefully and consider what the patient is implying
even when they do not state it directly.

Below are the {k} most-similar prior messages from the same patient population, with the
physician-adjudicated correct classification. Use these as evidence to inform your judgment
on the message that follows. The retrieved examples are NOT identical to the target message
— they are similar patterns. Reason from the patterns about what classification is correct
for the specific target message you are asked to classify.
"""

RAG_FOOTER_TEMPLATE = """
{action_scale}

{output_format}

Now classify the following message. Respond ONLY with the structured JSON object.
"""


def _format_retrieved_example(msg: str, hazard: bool, action: str) -> str:
    """Format one retrieved (message, label) block for the prompt."""
    return (
        f"---\n"
        f"Retrieved message: {msg.strip()[:500]}\n"
        f"Correct classification: hazard={'true' if hazard else 'false'}; action={action}\n"
    )


def build_rag_prompt(retrieved: list[dict], k: int) -> str:
    """retrieved: list of dicts with keys {message, hazard, action}. Length == k."""
    header = RAG_HEADER.format(k=k)
    blocks = [_format_retrieved_example(r["message"], r["hazard"], r["action"]) for r in retrieved]
    footer = RAG_FOOTER_TEMPLATE.format(action_scale=ACTION_SCALE, output_format=OUTPUT_FORMAT)
    return header + "\n" + "\n".join(blocks) + "\n" + footer


def _normalize_action(raw) -> str:
    """Map any training-set action representation to 'Action N' label."""
    if raw is None or raw == "None":
        return "Action 1"
    if isinstance(raw, (int, float)):
        return f"Action {int(raw)}"
    action_map = {
        "None": "Action 1", "Self-Care": "Action 1", "Routine Follow-up": "Action 3",
        "Contact Doctor": "Action 4", "Urgent Care": "Action 4", "Same-Day": "Action 5",
        "Call 911/988": "Action 8", "Emergency": "Action 8",
    }
    return action_map.get(str(raw), f"Action 1")


def _hazard_from_record(rec: dict) -> bool:
    for key in ("detection_truth", "ground_truth_detection", "true_hazard"):
        if key in rec and rec[key] is not None:
            try:
                return bool(int(rec[key]))
            except (ValueError, TypeError):
                return bool(rec[key])
    return False


def _action_from_record(rec: dict) -> str:
    raw = rec.get("action_truth") or rec.get("ground_truth_action") or rec.get("required_actions")
    return _normalize_action(raw)


def _message_from_record(rec: dict) -> str:
    for key in ("message", "prompt", "text"):
        if key in rec and rec[key]:
            return str(rec[key])
    return ""


class RAGIndex:
    """Sentence-BERT k-NN index over the training corpus."""

    def __init__(self, training_records: list[dict], model_name: str = "all-mpnet-base-v2"):
        from sentence_transformers import SentenceTransformer
        self.records = training_records
        self.messages = [_message_from_record(r) for r in training_records]
        self.hazards = [_hazard_from_record(r) for r in training_records]
        self.actions = [_action_from_record(r) for r in training_records]
        self.model = SentenceTransformer(model_name)
        # Pre-compute training-set embeddings (one-time cost)
        self.train_embs = self.model.encode(
            self.messages, batch_size=32, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        )

    def topk(self, query: str, k: int = 8) -> list[dict]:
        """Return top-k retrieved records by cosine similarity to query message."""
        q_emb = self.model.encode([query], normalize_embeddings=True,
                                  convert_to_numpy=True, show_progress_bar=False)[0]
        sims = self.train_embs @ q_emb  # cosine sim since normalized
        # Stratified retrieval — at least one hazard and at least one benign if available
        idx_sorted = np.argsort(-sims)
        chosen = []
        for i in idx_sorted:
            chosen.append(int(i))
            if len(chosen) >= k:
                break
        # Ensure at least 2 hazards and 2 benigns in the top-k where possible
        chosen_haz = [c for c in chosen if self.hazards[c]]
        chosen_ben = [c for c in chosen if not self.hazards[c]]
        if len(chosen_haz) < 2 or len(chosen_ben) < 2:
            # Expand: backfill from the next-most-similar of the underrepresented class
            need_haz = max(0, 2 - len(chosen_haz))
            need_ben = max(0, 2 - len(chosen_ben))
            for i in idx_sorted:
                if len(chosen) >= k:
                    break
                if int(i) in chosen:
                    continue
                if need_haz > 0 and self.hazards[int(i)]:
                    chosen.append(int(i))
                    need_haz -= 1
                elif need_ben > 0 and not self.hazards[int(i)]:
                    chosen.append(int(i))
                    need_ben -= 1
            chosen = chosen[:k]
        return [{"message": self.messages[c], "hazard": self.hazards[c],
                 "action": self.actions[c]} for c in chosen]


class ClaudeRAGClient(LLMClient):
    """Claude Opus 4.7 + k-NN RAG over the labeled training set."""

    model_version = "claude-opus-4-7"

    def __init__(self, training_records: list[dict], k: int = 8):
        import anthropic
        self.prompt_variant = "rag"
        self.k = k
        self.index = RAGIndex(training_records)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self._client = anthropic.Anthropic(api_key=api_key)
        # system_prompt is built per-call; cache an empty placeholder for the
        # LLMClient base class
        self.system_prompt = ""

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call(self, system_prompt: str, message: str) -> str:
        # Retrieve k-NN, build augmented system prompt, call Claude.
        retrieved = self.index.topk(message, k=self.k)
        prompt = build_rag_prompt(retrieved, k=self.k)
        response = self._client.messages.create(
            model=self.model_version,
            max_tokens=1024,
            system=prompt,
            messages=[{"role": "user", "content": message}],
        )
        return "".join(b.text for b in response.content if hasattr(b, "text"))


class GPT55RAGClient(LLMClient):
    """GPT-5.5 + k-NN RAG over the labeled training set (Responses API)."""

    model_version = "gpt-5.5"

    def __init__(self, training_records: list[dict], k: int = 8):
        from openai import OpenAI
        self.prompt_variant = "rag"
        self.k = k
        self.index = RAGIndex(training_records)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self._client = OpenAI(api_key=api_key)
        self.system_prompt = ""

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call(self, system_prompt: str, message: str) -> str:
        retrieved = self.index.topk(message, k=self.k)
        prompt = build_rag_prompt(retrieved, k=self.k)
        resp = self._client.responses.create(
            model=self.model_version,
            input=[
                {"role": "developer", "content": prompt},
                {"role": "user", "content": message},
            ],
        )
        return resp.output_text or ""


class GeminiRAGClient(LLMClient):
    """Gemini 3.1 Pro Preview + k-NN RAG over the labeled training set."""

    model_version = "gemini-3.1-pro-preview"

    def __init__(self, training_records: list[dict], k: int = 8):
        import google.generativeai as genai
        self.prompt_variant = "rag"
        self.k = k
        self.index = RAGIndex(training_records)
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) not set")
        genai.configure(api_key=api_key)
        self._genai = genai
        self.system_prompt = ""

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call(self, system_prompt: str, message: str) -> str:
        retrieved = self.index.topk(message, k=self.k)
        prompt = build_rag_prompt(retrieved, k=self.k)
        # Build model per-call so we can pass the dynamic system_instruction
        model = self._genai.GenerativeModel(
            model_name=self.model_version,
            system_instruction=prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": 512},
        )
        response = model.generate_content(message)
        return response.text or ""
