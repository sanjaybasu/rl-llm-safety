"""Local inference module — apply trained local models to test sets.

Reads trained artifacts from `models_dir` and writes ONE per-message CSV per
architecture to `predictions_dir/canonical/<architecture>_<run_id>.csv`.

Architectures inferred here (no API calls):
  - logreg_tfidf
  - xgboost_sbert
  - constellation
  - guardrails
  - cql_sens_opt   (calibrated_detector + sens-opt threshold)
  - cql_reward_opt (calibrated_detector + reward-opt threshold)
  - actionhead     (action-recommendation endpoint only)

Schema (matches the canonical per-message CSV defined in protocol.md §11):
    message_id, dataset, true_hazard, true_action, hazard_category,
    pred_proba, pred_hazard, pred_action, threshold_used,
    architecture, model_version, run_id, inference_time_s
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"


def load_test_records(path: Path) -> tuple[list[dict], list[str]]:
    """Return (records, message_ids) loaded from canonical test set JSON."""
    with open(path) as f:
        data = json.load(f)
    message_ids = [r.get("message_id", f"unknown_{i}") for i, r in enumerate(data)]
    return data, message_ids


def get_test_text(record: dict) -> str:
    """Extract the message text field (handles both schemas in v2 and v3)."""
    for key in ("message", "prompt", "text"):
        if key in record and record[key]:
            return str(record[key])
    return ""


def get_true_hazard(record: dict) -> int:
    """Extract binary hazard label (handles both schemas)."""
    for key in ("detection_truth", "ground_truth_detection", "true_hazard"):
        if key in record:
            val = record[key]
            return int(val) if val is not None else 0
    return 0


def get_true_action(record: dict) -> int:
    """Extract action label as 1-8 integer."""
    raw = record.get("action_truth") or record.get("ground_truth_action") or None
    if raw is None or raw == "None":
        return 1
    if isinstance(raw, (int, float)):
        return int(raw)
    # String label → int
    action_map = {
        "None": 1, "Self-Care": 1, "Routine Follow-up": 3,
        "Contact Doctor": 4, "Urgent Care": 4, "Same-Day": 5,
        "Call 911/988": 8, "Emergency": 8,
    }
    return action_map.get(str(raw), 1)


def get_hazard_category(record: dict) -> str:
    """Extract hazard category string."""
    for key in ("hazard_category", "ground_truth_hazard_category"):
        if key in record:
            return str(record[key] or "benign")
    return "benign"


def compute_embeddings(texts: list[str]) -> np.ndarray:
    """Compute SBERT embeddings for test messages."""
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(SBERT_MODEL)
    return encoder.encode(texts, batch_size=32, show_progress_bar=False,
                          convert_to_numpy=True)


def build_canonical_row(
    message_id: str, dataset: str, record: dict,
    pred_proba: Optional[float], pred_hazard: int, pred_action: int,
    threshold_used: float, architecture: str, model_version: str,
    run_id: str, inference_time_s: float,
) -> dict:
    """Build a single row in the canonical per-message CSV schema."""
    return {
        "message_id": message_id,
        "dataset": dataset,
        "true_hazard": get_true_hazard(record),
        "true_action": get_true_action(record),
        "hazard_category": get_hazard_category(record),
        "pred_proba": pred_proba if pred_proba is not None else "",
        "pred_hazard": int(pred_hazard),
        "pred_action": int(pred_action),
        "threshold_used": float(threshold_used),
        "architecture": architecture,
        "model_version": model_version,
        "run_id": run_id,
        "inference_time_s": round(inference_time_s, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-architecture inference
# ─────────────────────────────────────────────────────────────────────────────


def infer_logreg_tfidf(records: list[dict], message_ids: list[str], dataset: str,
                       artifact_path: Path, run_id: str) -> pd.DataFrame:
    with open(artifact_path, "rb") as f:
        art = pickle.load(f)
    texts = [get_test_text(r) for r in records]
    t0 = time.time()
    X = art["vectorizer"].transform(texts)
    proba = art["classifier"].predict_proba(X)[:, 1]
    elapsed = (time.time() - t0) / max(len(texts), 1)

    threshold = art["threshold"]
    rows = []
    for i, r in enumerate(records):
        p = float(proba[i])
        pred_hazard = int(p >= threshold)
        pred_action = 4 if pred_hazard else 1  # binary mapping: hazard → urgent care, else self-care
        rows.append(build_canonical_row(
            message_ids[i], dataset, r,
            pred_proba=p, pred_hazard=pred_hazard, pred_action=pred_action,
            threshold_used=threshold, architecture="logreg_tfidf",
            model_version="LogisticRegression+TF-IDF",
            run_id=run_id, inference_time_s=elapsed,
        ))
    return pd.DataFrame(rows)


def infer_xgboost_sbert(records: list[dict], message_ids: list[str], dataset: str,
                       embeddings: np.ndarray, artifact_path: Path, run_id: str) -> pd.DataFrame:
    with open(artifact_path, "rb") as f:
        art = pickle.load(f)
    t0 = time.time()
    proba = art["classifier"].predict_proba(embeddings)[:, 1]
    elapsed = (time.time() - t0) / max(len(records), 1)

    threshold = art["threshold"]
    rows = []
    for i, r in enumerate(records):
        p = float(proba[i])
        pred_hazard = int(p >= threshold)
        pred_action = 4 if pred_hazard else 1
        rows.append(build_canonical_row(
            message_ids[i], dataset, r,
            pred_proba=p, pred_hazard=pred_hazard, pred_action=pred_action,
            threshold_used=threshold, architecture="xgboost_sbert",
            model_version="XGBoost+SBERT",
            run_id=run_id, inference_time_s=elapsed,
        ))
    return pd.DataFrame(rows)


def infer_constellation(records: list[dict], message_ids: list[str], dataset: str,
                        embeddings: np.ndarray, artifact_path: Path, run_id: str) -> pd.DataFrame:
    with open(artifact_path, "rb") as f:
        art = pickle.load(f)
    t0 = time.time()
    proba_all = art["classifier"].predict_proba(embeddings)
    benign_idx = art["benign_index"]
    proba_hazard = (1 - proba_all[:, benign_idx]) if benign_idx >= 0 else proba_all.max(axis=1)
    elapsed = (time.time() - t0) / max(len(records), 1)

    threshold = art["threshold"]
    rows = []
    for i, r in enumerate(records):
        p = float(proba_hazard[i])
        pred_hazard = int(p >= threshold)
        # Category-specific action selection: pick the highest-prob non-benign category
        # then map to an action level (simplification: hazard → 4, else 1)
        pred_action = 4 if pred_hazard else 1
        rows.append(build_canonical_row(
            message_ids[i], dataset, r,
            pred_proba=p, pred_hazard=pred_hazard, pred_action=pred_action,
            threshold_used=threshold, architecture="constellation",
            model_version="Constellation-MultinomialLR+SBERT",
            run_id=run_id, inference_time_s=elapsed,
        ))
    return pd.DataFrame(rows)


def infer_guardrails(records: list[dict], message_ids: list[str], dataset: str,
                    artifact_path: Path, run_id: str) -> pd.DataFrame:
    with open(artifact_path, "rb") as f:
        art = pickle.load(f)
    rule_triggers = art["rule_triggers"]
    clf = art["calibrator"]
    threshold = art["threshold"]

    def rule_features(text: str) -> np.ndarray:
        tl = text.lower()
        return np.array([
            int(any(t in tl for t in triggers)) for triggers in rule_triggers.values()
        ])

    texts = [get_test_text(r) for r in records]
    t0 = time.time()
    X = np.array([rule_features(t) for t in texts])
    proba = clf.predict_proba(X)[:, 1]
    elapsed = (time.time() - t0) / max(len(texts), 1)

    rows = []
    for i, r in enumerate(records):
        p = float(proba[i])
        pred_hazard = int(p >= threshold)
        pred_action = 4 if pred_hazard else 1
        rows.append(build_canonical_row(
            message_ids[i], dataset, r,
            pred_proba=p, pred_hazard=pred_hazard, pred_action=pred_action,
            threshold_used=threshold, architecture="guardrails",
            model_version="RuleBasedGuardrails-v1",
            run_id=run_id, inference_time_s=elapsed,
        ))
    return pd.DataFrame(rows)


def infer_cql_sens_opt(records: list[dict], message_ids: list[str], dataset: str,
                      embeddings: np.ndarray, detector_path: Path,
                      cql_thresholds_path: Path, run_id: str) -> pd.DataFrame:
    """CQL sens-optimized: calibrated detector + threshold tuned for sens at min spec."""
    with open(detector_path, "rb") as f:
        det = pickle.load(f)
    with open(cql_thresholds_path) as f:
        thresholds = json.load(f)

    clf = det["classifier"]
    temperature = det["temperature"]
    benign_idx = det["benign_index"]

    t0 = time.time()
    logits = clf.decision_function(embeddings)
    if logits.ndim == 1:
        logits = np.stack([-logits, logits], axis=1)
    scaled = logits / temperature
    exp = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    proba_hazard = (1 - probs[:, benign_idx]) if benign_idx >= 0 else probs.max(axis=1)
    elapsed = (time.time() - t0) / max(len(records), 1)

    threshold = thresholds["sens_opt"]
    rows = []
    for i, r in enumerate(records):
        p = float(proba_hazard[i])
        pred_hazard = int(p >= threshold)
        pred_action = 4 if pred_hazard else 1
        rows.append(build_canonical_row(
            message_ids[i], dataset, r,
            pred_proba=p, pred_hazard=pred_hazard, pred_action=pred_action,
            threshold_used=threshold, architecture="cql_sens_opt",
            model_version="CQL-SensOptimized",
            run_id=run_id, inference_time_s=elapsed,
        ))
    return pd.DataFrame(rows)


def infer_cql_reward_opt(records: list[dict], message_ids: list[str], dataset: str,
                       embeddings: np.ndarray, detector_path: Path,
                       cql_thresholds_path: Path, run_id: str) -> pd.DataFrame:
    """CQL reward-optimized: same detector + reward-opt threshold."""
    with open(detector_path, "rb") as f:
        det = pickle.load(f)
    with open(cql_thresholds_path) as f:
        thresholds = json.load(f)

    clf = det["classifier"]
    temperature = det["temperature"]
    benign_idx = det["benign_index"]

    t0 = time.time()
    logits = clf.decision_function(embeddings)
    if logits.ndim == 1:
        logits = np.stack([-logits, logits], axis=1)
    scaled = logits / temperature
    exp = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    proba_hazard = (1 - probs[:, benign_idx]) if benign_idx >= 0 else probs.max(axis=1)
    elapsed = (time.time() - t0) / max(len(records), 1)

    threshold = thresholds["reward_opt"]
    rows = []
    for i, r in enumerate(records):
        p = float(proba_hazard[i])
        pred_hazard = int(p >= threshold)
        pred_action = 4 if pred_hazard else 1
        rows.append(build_canonical_row(
            message_ids[i], dataset, r,
            pred_proba=p, pred_hazard=pred_hazard, pred_action=pred_action,
            threshold_used=threshold, architecture="cql_reward_opt",
            model_version="CQL-RewardOptimized",
            run_id=run_id, inference_time_s=elapsed,
        ))
    return pd.DataFrame(rows)


def infer_actionhead(records: list[dict], message_ids: list[str], dataset: str,
                    embeddings: np.ndarray, artifact_path: Path, run_id: str) -> pd.DataFrame:
    """ActionHead: 8-class action classifier; hazard binary derived as action ≥ 4."""
    with open(artifact_path, "rb") as f:
        art = pickle.load(f)
    t0 = time.time()
    pred_actions = art["classifier"].predict(embeddings)
    elapsed = (time.time() - t0) / max(len(records), 1)

    rows = []
    for i, r in enumerate(records):
        action = int(pred_actions[i])
        # Action ≥ 4 → hazard (urgent/emergency)
        pred_hazard = int(action >= 4)
        rows.append(build_canonical_row(
            message_ids[i], dataset, r,
            pred_proba=None, pred_hazard=pred_hazard, pred_action=action,
            threshold_used=4,  # the action-class threshold for binary derivation
            architecture="actionhead",
            model_version="ActionHead-MultinomialLR+SBERT",
            run_id=run_id, inference_time_s=elapsed,
        ))
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator: run every local architecture on every dataset
# ─────────────────────────────────────────────────────────────────────────────


def run_all_local_inference(
    models_dir: Path,
    test_sets: dict[str, Path],
    predictions_dir: Path,
    run_id: str,
) -> dict:
    """Apply all local architectures to all test sets; save per-architecture CSVs.

    test_sets: dict mapping dataset name (e.g. "realworld_n2000") to JSON path.
    """
    predictions_dir = Path(predictions_dir)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    output_csv_paths: dict[str, Path] = {}

    # Architectures and their dispatch functions
    LOCAL_ARCHS = [
        "logreg_tfidf", "xgboost_sbert", "constellation", "guardrails",
        "cql_sens_opt", "cql_reward_opt", "actionhead",
    ]

    for arch in LOCAL_ARCHS:
        print(f"\n[local inference] {arch}")
        all_rows: list[pd.DataFrame] = []
        for dataset_name, test_path in test_sets.items():
            records, mids = load_test_records(test_path)
            texts = [get_test_text(r) for r in records]
            print(f"  {dataset_name}: n={len(records)}")
            # Compute SBERT embeddings once per dataset (SBERT-based archs reuse them)
            if arch in {"xgboost_sbert", "constellation", "cql_sens_opt",
                        "cql_reward_opt", "actionhead"}:
                embeds = compute_embeddings(texts)
            else:
                embeds = None
            if arch == "logreg_tfidf":
                df = infer_logreg_tfidf(records, mids, dataset_name,
                                       models_dir / "logreg_tfidf.pkl", run_id)
            elif arch == "xgboost_sbert":
                df = infer_xgboost_sbert(records, mids, dataset_name, embeds,
                                        models_dir / "xgboost_sbert.pkl", run_id)
            elif arch == "constellation":
                df = infer_constellation(records, mids, dataset_name, embeds,
                                         models_dir / "constellation.pkl", run_id)
            elif arch == "guardrails":
                df = infer_guardrails(records, mids, dataset_name,
                                     models_dir / "guardrails.pkl", run_id)
            elif arch == "cql_sens_opt":
                df = infer_cql_sens_opt(records, mids, dataset_name, embeds,
                                       models_dir / "calibrated_detector.pkl",
                                       models_dir / "cql_thresholds.json", run_id)
            elif arch == "cql_reward_opt":
                df = infer_cql_reward_opt(records, mids, dataset_name, embeds,
                                         models_dir / "calibrated_detector.pkl",
                                         models_dir / "cql_thresholds.json", run_id)
            elif arch == "actionhead":
                df = infer_actionhead(records, mids, dataset_name, embeds,
                                     models_dir / "actionhead.pkl", run_id)
            else:
                raise ValueError(f"Unknown architecture {arch}")
            all_rows.append(df)

        combined = pd.concat(all_rows, ignore_index=True)
        out_path = predictions_dir / f"{arch}_{run_id}.csv"
        combined.to_csv(out_path, index=False)
        print(f"  → wrote {out_path} ({len(combined)} rows)")
        output_csv_paths[arch] = out_path

    return output_csv_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--realworld", type=Path, required=True)
    parser.add_argument("--physician", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()
    run_id = args.run_id or str(uuid.uuid4())
    test_sets = {"realworld_n2000": args.realworld, "physician_n41": args.physician}
    run_all_local_inference(args.models_dir, test_sets, args.predictions_dir, run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
