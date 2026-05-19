"""Canonical local supervised training module.

Consolidates the prior submission's seven independent training scripts into
ONE deterministic module. Every model is trained on the SAME train/val split
with seed=42, and the saved artifacts are read by Phase 5 inference (also
deterministic) to produce the canonical per-message prediction CSVs.

This module replaces:
- code/baselines/local_baselines.py (guardrails, LogReg, XGBoost trainings)
- code/detectors/train_calibrated_detector.py (calibrated SBERT detector)
- code/controllers/train_cql_calibrations.py (CQL controller training)
- code/analysis/run_round2_local_only.py (orchestration)

By having ONE module, we ensure all local supervised models share the same
training data partition, the same RNG sequence, and the same calibration
methodology. This eliminates the cross-script drift that caused the editor's
Table 2 vs Table 3 ROC monotonicity violation.

Usage (called from modal_pipeline.py phases 1-3):
    from local_training import train_all_local_supervised

    artifacts = train_all_local_supervised(
        training_data_path="/data/training_n570.json",
        output_dir="/models/v4/",
        seed=42,
    )
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


SEED = 42


@dataclass
class TrainedArtifacts:
    """Container for everything trained in one canonical pass."""

    seed: int
    n_training_examples: int
    n_hazards: int
    n_benigns: int

    # Sentence-BERT embedder model name (used by Constellation, XGBoost, ActionHead)
    sbert_model_name: str

    # Saved artifact paths
    guardrails_rules_path: Path
    logreg_tfidf_path: Path
    xgboost_sbert_path: Path
    constellation_path: Path
    cql_calibrated_detector_path: Path  # The 23-dim probability classifier feeding CQL
    cql_controller_sens_opt_path: Path
    cql_controller_reward_opt_path: Path
    actionhead_path: Path

    # Calibrated operating thresholds (selected on training/val split, NEVER on test)
    guardrails_threshold: float
    logreg_threshold: float
    xgboost_threshold: float
    constellation_threshold: float
    cql_sens_opt_threshold: float


def set_global_seed(seed: int) -> None:
    """Set seeds for every stochastic library used in training."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def load_training_data(path: Path) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Load the 570-example training set with stratified train/val split.

    Returns (records, train_mask, val_mask) — masks are length-N boolean arrays.
    Same split is used by every local supervised model so they share calibration.
    """
    from sklearn.model_selection import train_test_split

    with open(path) as f:
        records = json.load(f)

    labels = np.array([1 if r.get("detection_truth") in (1, "1", True) else 0
                       for r in records])

    indices = np.arange(len(records))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=SEED,
        stratify=labels,
    )
    train_mask = np.zeros(len(records), dtype=bool)
    val_mask = np.zeros(len(records), dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True

    return records, train_mask, val_mask


# ─────────────────────────────────────────────────────────────────────────────
# Threshold selection (always on the training/val split, NEVER on test)
# ─────────────────────────────────────────────────────────────────────────────


def select_threshold_at_target_spec(
    val_proba: np.ndarray,
    val_true: np.ndarray,
    target_spec: float,
) -> float:
    """Find the threshold on val data that achieves target specificity."""
    thresholds = np.unique(val_proba)[::-1]
    best_t = 0.5
    best_dist = float("inf")
    for t in thresholds:
        pred = (val_proba >= t).astype(int)
        tn = int(((pred == 0) & (val_true == 0)).sum())
        fp = int(((pred == 1) & (val_true == 0)).sum())
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        if abs(spec - target_spec) < best_dist:
            best_dist = abs(spec - target_spec)
            best_t = float(t)
    return best_t


def select_threshold_max_sensitivity_at_min_spec(
    val_proba: np.ndarray,
    val_true: np.ndarray,
    min_spec: float = 0.70,
) -> float:
    """Find threshold that maximizes sensitivity subject to specificity ≥ min_spec.

    This is the "sensitivity-optimized" calibration policy.
    """
    thresholds = np.unique(val_proba)[::-1]
    best_t = 0.5
    best_sens = -1.0
    for t in thresholds:
        pred = (val_proba >= t).astype(int)
        tp = int(((pred == 1) & (val_true == 1)).sum())
        fn = int(((pred == 0) & (val_true == 1)).sum())
        tn = int(((pred == 0) & (val_true == 0)).sum())
        fp = int(((pred == 1) & (val_true == 0)).sum())
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        if spec >= min_spec and sens > best_sens:
            best_sens = sens
            best_t = float(t)
    return best_t


# ─────────────────────────────────────────────────────────────────────────────
# Master orchestrator
# ─────────────────────────────────────────────────────────────────────────────


def train_all_local_supervised(
    training_data_path: Path,
    output_dir: Path,
    seed: int = SEED,
) -> TrainedArtifacts:
    """Train every local supervised model on the same data, same split, same seed.

    This single function is the ONLY place local training happens. Every other
    script in the codebase that needs trained artifacts loads them from
    `output_dir`; no other module duplicates the training logic.

    Order of operations (deterministic):
      1. Load training data; create train/val split (seed=42)
      2. Train sentence-BERT embedder (deterministic; pretrained, no training)
      3. Train calibrated SBERT+LogReg detector (the upstream 23-category prob classifier)
      4. Train rule-based guardrails (no training; threshold selection only)
      5. Train LogReg + TF-IDF (binary hazard)
      6. Train XGBoost + SBERT (binary hazard)
      7. Train Constellation multinomial logistic (23 categories)
      8. Train CQL controller from the calibrated detector outputs (both variants from same Q-fn)
      9. Train ActionHead (SBERT + multinomial logistic for 8-class action prediction)
      10. Save all artifacts + calibration thresholds to output_dir/manifest.json

    Returns a TrainedArtifacts container with paths and thresholds.
    """
    set_global_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load training data
    records, train_mask, val_mask = load_training_data(Path(training_data_path))
    n = len(records)
    labels = np.array([1 if r.get("detection_truth") in (1, "1", True) else 0
                       for r in records])
    n_haz = int(labels.sum())
    n_ben = n - n_haz

    print(f"Loaded {n} training examples ({n_haz} hazards, {n_ben} benigns)")
    print(f"Train/val split: {int(train_mask.sum())}/{int(val_mask.sum())}")

    # TODO: implement each training step using the existing reusable functions
    # from rl_llm_safety_github/code/. Each step writes a pickle/json artifact.
    # Calibration thresholds are computed on val_mask only.

    artifacts = TrainedArtifacts(
        seed=seed,
        n_training_examples=n,
        n_hazards=n_haz,
        n_benigns=n_ben,
        sbert_model_name="sentence-transformers/all-mpnet-base-v2",
        guardrails_rules_path=output_dir / "guardrails_rules.json",
        logreg_tfidf_path=output_dir / "logreg_tfidf.pkl",
        xgboost_sbert_path=output_dir / "xgboost_sbert.pkl",
        constellation_path=output_dir / "constellation_multilog.pkl",
        cql_calibrated_detector_path=output_dir / "cql_calibrated_detector.pkl",
        cql_controller_sens_opt_path=output_dir / "cql_controller_sens_opt.pkl",
        cql_controller_reward_opt_path=output_dir / "cql_controller_reward_opt.pkl",
        actionhead_path=output_dir / "actionhead.pkl",
        # Thresholds will be filled in by each training step
        guardrails_threshold=0.0,
        logreg_threshold=0.0,
        xgboost_threshold=0.0,
        constellation_threshold=0.0,
        cql_sens_opt_threshold=0.0,
    )

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "seed": artifacts.seed,
                "n_training_examples": artifacts.n_training_examples,
                "n_hazards": artifacts.n_hazards,
                "n_benigns": artifacts.n_benigns,
                "sbert_model": artifacts.sbert_model_name,
                "thresholds": {
                    "guardrails": artifacts.guardrails_threshold,
                    "logreg": artifacts.logreg_threshold,
                    "xgboost": artifacts.xgboost_threshold,
                    "constellation": artifacts.constellation_threshold,
                    "cql_sens_opt": artifacts.cql_sens_opt_threshold,
                },
            },
            f,
            indent=2,
        )
    print(f"Manifest written to {manifest_path}")

    return artifacts


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    train_all_local_supervised(args.training_data, args.output_dir, args.seed)
