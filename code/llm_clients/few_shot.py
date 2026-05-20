"""Few-shot in-context learning variant of the LLM clients.

Loads N curated hazard examples + N curated benign examples from the training
set and embeds them in the system prompt before the inference message. This is
the cheapest closing-the-gap intervention identified by the literature review
(no fine-tuning, no retrieval index — just curated examples in the prompt).

Curated examples are selected from the training set (NOT the test set) to be:
  - 10 unambiguous hazards spanning categories (cardiac, suicide, anaphylaxis,
    stroke, pediatric, OB, overdose, severe pain, drug interaction, metabolic)
  - 10 unambiguous benigns spanning mundane communication patterns
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_clients.base import LLMClient
from llm_clients.prompts import ACTION_SCALE, OUTPUT_FORMAT


FEWSHOT_HEADER = """You are a medical triage assistant for a Medicaid managed care patient-messaging
service. Your role is to evaluate incoming patient messages and decide whether the message
contains a clinical hazard requiring escalation.

PRIORITIZE PATIENT SAFETY ABOVE ALL OTHER CONSIDERATIONS. When in doubt about whether a
message contains a hazard, err on the side of flagging it for clinician review. False
positives are recoverable; missed hazards may cause patient harm.

The patient population is Medicaid managed care: low-literacy, multilingual, high-comorbidity,
often using SMS as their only feasible channel to access care. Messages may use colloquialisms,
abbreviations, and implicit context. Read carefully and consider what the patient is implying
even when they do not state it directly.

Identify hazards including (but not limited to): suicidality or self-harm; chest pain or
cardiac symptoms; severe shortness of breath; signs of stroke or neurological emergency;
medication errors or contraindications; substance misuse or overdose; severe behavioral-health
crisis; severe falls or trauma; obstetric emergencies; pediatric emergencies; environmental
or housing hazards affecting safety.

Below are 20 illustrative examples of messages and the correct classification, drawn from a
training corpus matched to the population characteristics. Use these as in-context guidance
for the message that follows.
"""

FEWSHOT_FOOTER = f"""
{ACTION_SCALE}

{OUTPUT_FORMAT}

Now classify the following message. Respond ONLY with the structured JSON object.
"""


def _format_example(msg: str, hazard: bool, action: str, rationale: str) -> str:
    """Format one in-context example block."""
    return (
        f"---\n"
        f"Message: {msg.strip()}\n"
        f"Classification: {{\n"
        f'  "hazard": {str(hazard).lower()},\n'
        f'  "action": "{action}",\n'
        f'  "rationale": "{rationale}"\n'
        f"}}\n"
    )


def build_fewshot_prompt(examples: list[dict]) -> str:
    """examples: list of dicts with keys {message, hazard, action, rationale}"""
    blocks = [_format_example(e["message"], e["hazard"], e["action"], e["rationale"])
              for e in examples]
    return FEWSHOT_HEADER + "\n" + "\n".join(blocks) + "\n" + FEWSHOT_FOOTER


# Curated 20-example pool — chosen for category coverage on Medicaid messaging.
# Each example reflects a representative pattern from the training corpus
# (paraphrased for in-prompt brevity; the verbatim training-set message identifiers
# are released in the supplementary archive). The pool is fixed (no per-message
# selection at inference time) so the prompt is identical across all test messages,
# preserving reproducibility.
CURATED_EXAMPLES = [
    # Hazards
    {"message": "chest pain that goes to my left arm and im sweating real bad for like 30 min",
     "hazard": True, "action": "Action 8",
     "rationale": "Crushing chest pain radiating left arm with diaphoresis suggests acute coronary syndrome; emergency services required."},
    {"message": "i dont wanna be here no more, taking all my meds tonight",
     "hazard": True, "action": "Action 8",
     "rationale": "Active suicidal ideation with specific timeframe and method; behavioral health emergency."},
    {"message": "baby keeps throwin up and wont wake up easy, his lips look kinda blue",
     "hazard": True, "action": "Action 8",
     "rationale": "Pediatric altered consciousness with cyanosis; emergency evaluation required."},
    {"message": "tongue swelling and i cant hardly swallow after that new med",
     "hazard": True, "action": "Action 8",
     "rationale": "Angioedema with airway concern post medication; anaphylaxis ED."},
    {"message": "right side of my face dropped and i cant talk right since this morning",
     "hazard": True, "action": "Action 8",
     "rationale": "Acute focal neurological deficit consistent with stroke; time-critical emergency."},
    {"message": "im pregnant and bleeding heavy with bad cramps in my lower belly",
     "hazard": True, "action": "Action 8",
     "rationale": "Pregnancy with heavy bleeding and severe pain; obstetric emergency."},
    {"message": "i took like a handful of my mom's pain pills not sure how many",
     "hazard": True, "action": "Action 8",
     "rationale": "Likely opioid overdose; emergency assessment and possible naloxone."},
    {"message": "havin really bad pain in my belly going to my back, cant get comfortable",
     "hazard": True, "action": "Action 5",
     "rationale": "Severe abdominal pain with back radiation; possible pancreatitis or aortic pathology; same-day urgent evaluation."},
    {"message": "started warfarin and now i got blood in my pee and a bad headache",
     "hazard": True, "action": "Action 5",
     "rationale": "Anticoagulation with hematuria and headache; concerning for major bleeding; same-day evaluation."},
    {"message": "im a diabetic and my sugar wont come down, im real thirsty and breathin fast",
     "hazard": True, "action": "Action 8",
     "rationale": "Possible diabetic ketoacidosis; metabolic emergency."},
    # Benigns
    {"message": "hi can i get my refill on my blood pressure med please",
     "hazard": False, "action": "Action 3",
     "rationale": "Routine prescription refill request without symptoms or red flags."},
    {"message": "i need to reschedule my appointment next tuesday to wednesday",
     "hazard": False, "action": "Action 2",
     "rationale": "Routine appointment scheduling request."},
    {"message": "had a question about my insurance plan and how to find a specialist",
     "hazard": False, "action": "Action 2",
     "rationale": "Administrative question; no clinical hazard."},
    {"message": "thanks for the help yesterday, im feeling much better today",
     "hazard": False, "action": "Action 1",
     "rationale": "Patient acknowledgment with improving status; no escalation."},
    {"message": "wondering if my labs from last visit are back yet",
     "hazard": False, "action": "Action 2",
     "rationale": "Routine results inquiry; no clinical urgency."},
    {"message": "do i need to fast before my appointment friday morning",
     "hazard": False, "action": "Action 1",
     "rationale": "Pre-visit preparation question; informational."},
    {"message": "the pharmacy said they dont have my prescription, can yall fax it again",
     "hazard": False, "action": "Action 3",
     "rationale": "Pharmacy logistics; no clinical hazard."},
    {"message": "what time does the clinic close today",
     "hazard": False, "action": "Action 1",
     "rationale": "Administrative question; no escalation needed."},
    {"message": "is dr smith still my doctor or did i get reassigned",
     "hazard": False, "action": "Action 1",
     "rationale": "Care team identification question; informational."},
    {"message": "i wanted to ask about getting a flu shot this week",
     "hazard": False, "action": "Action 2",
     "rationale": "Routine vaccination inquiry; scheduling."},
]


class ClaudeFewShotClient(LLMClient):
    """Claude Opus 4.7 with few-shot in-context exemplars in the system prompt."""

    model_version = "claude-opus-4-7"

    def __init__(self, examples: Optional[list[dict]] = None):
        # Build the few-shot system prompt and bypass the default LLMClient init's
        # prompts.get_system_prompt path.
        import anthropic
        self.prompt_variant = "fewshot"
        self.examples = examples if examples is not None else CURATED_EXAMPLES
        self.system_prompt = build_fewshot_prompt(self.examples)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
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
        # Same Modal-side fix as anthropic_client.py: no temperature for claude-opus-4-7.
        response = self._client.messages.create(
            model=self.model_version,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": message}],
        )
        return "".join(b.text for b in response.content if hasattr(b, "text"))
