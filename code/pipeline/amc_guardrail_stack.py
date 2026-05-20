"""Academic Medical Center standard guardrail stack — operationally-deployed configuration.

Major US AMCs (Duke Institute for Health Innovation Health AI Community of Practice;
Nationwide Network of Health AI Assurance Laboratories per Shah et al. JAMA 2024;
CHAI Blueprint for trustworthy AI implementation guidance 2023) deploy a four-
component guardrail stack for clinical AI:

  1. Clinical-safety system prompt (already in our LLM clients as safety-augmented)
  2. Content-moderation filter pre-screen (filters out spam, profanity, off-topic
     messages BEFORE the clinical hazard pipeline runs)
  3. Retrieval-augmented generation with citation grounding (RAG; we have this
     as claude_opus_4_7_rag; citation grounding is implicit in the retrieved
     exemplar references)
  4. Learned hazard-classifier pre-screen (a supervised model filters messages
     that warrant LLM review)

This module composes the stack from existing per-message predictions:
  - Component 1 + 3 (RAG with safety prompt): claude_opus_4_7_rag
  - Component 4 (learned hazard classifier): XGBoost+SBERT or LogReg+TF-IDF
  - Component 2 (content moderation): implicit pass-through for clinically-themed
    messages in our test set (no purely toxic or off-topic content); the deployed
    component would gate at the message-ingestion layer prior to entering the
    clinical hazard pipeline. We model this as a no-op on the present test set.

Compositions evaluated (AND-rule cascade; both stages must flag for hazard alert):
  - AMC stack (XGBoost gate + Claude RAG)
  - AMC stack (LogReg gate + Claude RAG)
  - AMC stack (Constellation gate + Claude RAG)
  - AMC stack (CQL_reward gate + Claude RAG)  — RL-controller gate per Compass-style deployment
  - AMC stack (Guardrails rule-based gate + Claude RAG)

Output: results/amc_guardrail_stack.csv

References (added to manuscript):
  - Shah NH, Halamka JD, Saria S, et al. A nationwide network of health AI
    assurance laboratories. JAMA 2024;331:245-249.
  - Coalition for Health AI. Blueprint for trustworthy AI implementation
    guidance and assurance for healthcare. CHAI 2023.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CLINICAL_GRADE = 0.80
LLM_NAME = "claude_opus_4_7_rag"   # component 1+3 of the stack (safety-prompt + RAG)
# Component 4 (learned hazard-classifier pre-screen) candidates
GATE_CANDIDATES = [
    "xgboost_sbert",
    "logreg_tfidf",
    "constellation",
    "cql_reward_opt",
    "guardrails",
    "actionhead",
]


def metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    pred = pred.astype(bool); true = true.astype(bool)
    tp = int((pred & true).sum()); fp = int((pred & ~true).sum())
    tn = int((~pred & ~true).sum()); fn = int((~pred & true).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else float("nan")
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": round(sens, 4), "specificity": round(spec, 4),
        "ppv": round(ppv, 4), "f1": round(f1, 4), "mcc": round(mcc, 4),
        "clinical_grade": bool(sens >= CLINICAL_GRADE and spec >= CLINICAL_GRADE),
    }


def load_arch_pred(predictions_dir: Path, arch_prefix: str, dataset: str,
                    base_msg_order: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load pred_hazard + true_hazard for the given architecture on the given dataset."""
    files = list(Path(predictions_dir).glob(f"{arch_prefix}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV matches prefix {arch_prefix} in {predictions_dir}")
    df = pd.read_csv(files[0])
    sub = df[df["dataset"] == dataset].sort_values("message_id").reset_index(drop=True)
    if base_msg_order is not None:
        sub = sub.set_index("message_id").loc[base_msg_order].reset_index()
    return sub["pred_hazard"].astype(int).values, sub["true_hazard"].astype(int).values, sub["message_id"].astype(str).tolist()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="realworld_n2000")
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    # Anchor on the LLM (Component 1+3); enforce message_id alignment for the gate
    llm_pred, true, msg_order = load_arch_pred(args.predictions_dir, LLM_NAME, args.dataset)
    print(f"Anchor: {LLM_NAME} loaded ({len(true)} messages, {int(true.sum())} hazards)")

    rows = []
    # First add the standalone Claude-RAG (Component 1+3 alone) as baseline reference
    m = metrics(llm_pred, true)
    m.update({"stack_label": f"Component 1+3 alone ({LLM_NAME})",
               "gate_component_4": "(none)", "llm_component_1_3": LLM_NAME})
    rows.append(m)

    # AMC stack: each gate × LLM as AND-rule cascade
    for gate in GATE_CANDIDATES:
        try:
            gate_pred, gate_true, _ = load_arch_pred(args.predictions_dir, gate,
                                                     args.dataset, base_msg_order=msg_order)
            assert (gate_true == true).all(), f"true_hazard mismatch between {gate} and {LLM_NAME}"
        except FileNotFoundError as e:
            print(f"  skipping {gate}: {e}")
            continue
        cascade = (gate_pred & llm_pred).astype(int)
        m = metrics(cascade, true)
        m.update({"stack_label": f"AMC stack ({gate} gate + {LLM_NAME})",
                   "gate_component_4": gate, "llm_component_1_3": LLM_NAME})
        rows.append(m)

    df = pd.DataFrame(rows)
    out_path = args.results_dir / "amc_guardrail_stack.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  → wrote {out_path}")
    print()
    print(df[["stack_label", "sensitivity", "specificity", "ppv", "f1", "mcc", "clinical_grade"]].to_string(index=False))
    n_cg = int(df["clinical_grade"].sum())
    print(f"\n{n_cg} / {len(df)} AMC stack configurations reach sens >= {CLINICAL_GRADE} AND spec >= {CLINICAL_GRADE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
