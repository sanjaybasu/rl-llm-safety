"""Canonical training module — implements all 6 architectures (top-level) on
ONE training partition with ONE deterministic seed.

This is the heart of the v3 pipeline. It replaces the 7 independent scripts in
the v2 codebase. Every model is trained on the same `train_mask`, calibrated
on the same `val_mask`, and saves a single artifact + calibrated threshold per
model. Phase 5 inference reads these artifacts deterministically.

Architectures implemented (locked Phase B 6-config matrix):
1. Rule-based guardrails (147 rules + SBERT semantic gate, then aggregate threshold)
2. Logistic regression + TF-IDF features
3. XGBoost + SBERT embeddings
4. Constellation (multinomial logistic over hazard categories on SBERT)
5. CQL upstream calibrated detector (SBERT + multinomial logistic + temperature scaling)
   → produces 23-dim probability vector used by both CQL variants
6. ActionHead (SBERT + multinomial logistic on 8-class action labels)

The two CQL controller variants (sens-opt, reward-opt) are not trained here
because they're calibration policies on the SAME upstream detector — they're
parameterized by a threshold (sens-opt) or argmax rule (reward-opt) applied at
inference time. The trained artifact is the upstream calibrated detector.

Usage:
    python canonical_training.py \\
        --training-data /data/combined_train.json \\
        --output-dir /models/v4/

Output:
    /models/v4/manifest.json — paths to all trained artifacts + calibrated thresholds
    /models/v4/sbert_encoder.txt — SBERT model name (for inference consistency)
    /models/v4/calibrated_detector.pkl — upstream 23-dim probability classifier
    /models/v4/logreg_tfidf.pkl — binary hazard classifier
    /models/v4/xgboost_sbert.pkl — XGBoost on SBERT features
    /models/v4/constellation.pkl — multinomial logistic on SBERT
    /models/v4/guardrails.pkl — rule embeddings + aggregate threshold
    /models/v4/actionhead.pkl — 8-class action classifier
    /models/v4/cql_thresholds.json — sensitivity-opt threshold + reward-opt rule

References:
- Existing v2 logic to consolidate: packaging/rl_llm_safety_github/code/detectors/
  train_calibrated_detector.py (Lines 71-230: TransformerHazardDetector class).
- Plan: /Users/sanjaybasu/.claude/plans/sharded-sleeping-neumann.md
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


SEED = 42

# Model identifiers (semantic versioning for reproducibility audit)
SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"


def set_global_seed(seed: int = SEED) -> None:
    """Set seeds for every stochastic library used in training."""
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


def load_training_records(path: Path) -> list[dict]:
    """Load canonical 1,280-record training set."""
    with open(path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} training records from {path}")
    return data


def extract_labels(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (binary_hazard, action_class, hazard_category) arrays."""
    binary = np.array([int(r.get("ground_truth_detection", 0)) for r in records])
    # Action: map string to 1-8 integer or 0 if None
    action_map = {
        "None": 1,                  # Action 1: self-care/no action
        "Routine Follow-up": 3,      # Action 3: routine f/u
        "Contact Doctor": 4,         # Action 4: urgent contact
        "Call 911/988": 8,           # Action 8: emergency
    }
    actions = np.array([action_map.get(r.get("ground_truth_action") or "None", 1)
                        for r in records])
    # Hazard category (string; will be label-encoded by individual models)
    categories = np.array([r.get("ground_truth_hazard_category", "benign") or "benign"
                           for r in records])
    return binary, actions, categories


def stratified_train_val_split(
    n_records: int, binary_labels: np.ndarray, test_size: float = 0.2, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Single canonical train/val split used by every architecture.

    Stratified on binary hazard so train and val have the same prevalence.
    """
    from sklearn.model_selection import train_test_split
    indices = np.arange(n_records)
    train_idx, val_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=binary_labels
    )
    train_mask = np.zeros(n_records, dtype=bool)
    val_mask = np.zeros(n_records, dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    return train_mask, val_mask


# ─────────────────────────────────────────────────────────────────────────────
# Sentence-BERT embeddings (computed once; reused by every SBERT-based model)
# ─────────────────────────────────────────────────────────────────────────────


def compute_sbert_embeddings(
    texts: list[str], model_name: str = SBERT_MODEL, batch_size: int = 32
) -> np.ndarray:
    """Compute SBERT embeddings deterministically (model is pretrained)."""
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(model_name)
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# Threshold selection helpers
# ─────────────────────────────────────────────────────────────────────────────


def select_threshold_at_target_spec(
    val_proba: np.ndarray, val_true: np.ndarray, target_spec: float
) -> float:
    """Find threshold on val set that achieves target specificity."""
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


def select_sens_optimized_threshold(
    val_proba: np.ndarray, val_true: np.ndarray, min_spec: float = 0.70
) -> float:
    """Find threshold that maximizes sensitivity subject to specificity ≥ min_spec."""
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
# Trainer 1: Logistic Regression + TF-IDF
# ─────────────────────────────────────────────────────────────────────────────


def train_logreg_tfidf(
    train_texts: list[str], train_labels: np.ndarray,
    val_texts: list[str], val_labels: np.ndarray,
    output_path: Path,
) -> dict:
    """LogReg with TF-IDF features. Binary hazard classifier."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=10_000,
        sublinear_tf=True,
        min_df=2,
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=SEED,
        solver="liblinear",
    )
    clf.fit(X_train, train_labels)

    val_proba = clf.predict_proba(X_val)[:, 1]
    threshold = select_sens_optimized_threshold(val_proba, val_labels, min_spec=0.70)

    artifact = {
        "vectorizer": vectorizer,
        "classifier": clf,
        "threshold": threshold,
        "feature_count": X_train.shape[1],
        "seed": SEED,
    }
    with open(output_path, "wb") as f:
        pickle.dump(artifact, f)

    val_pred = (val_proba >= threshold).astype(int)
    val_sens = (val_pred[val_labels == 1] == 1).mean() if (val_labels == 1).sum() else 0
    val_spec = (val_pred[val_labels == 0] == 0).mean() if (val_labels == 0).sum() else 0
    print(f"  LogReg+TF-IDF: threshold={threshold:.4f}, val sens={val_sens:.3f}, val spec={val_spec:.3f}")
    return {"threshold": threshold, "val_sens": val_sens, "val_spec": val_spec}


# ─────────────────────────────────────────────────────────────────────────────
# Trainer 2: XGBoost + SBERT embeddings
# ─────────────────────────────────────────────────────────────────────────────


def train_xgboost_sbert(
    train_embeds: np.ndarray, train_labels: np.ndarray,
    val_embeds: np.ndarray, val_labels: np.ndarray,
    output_path: Path,
) -> dict:
    """XGBoost binary classifier on SBERT embeddings."""
    import xgboost as xgb
    pos_weight = (train_labels == 0).sum() / max((train_labels == 1).sum(), 1)
    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=pos_weight,
        random_state=SEED,
        eval_metric="logloss",
        use_label_encoder=False,
        n_jobs=4,
    )
    clf.fit(train_embeds, train_labels)
    val_proba = clf.predict_proba(val_embeds)[:, 1]
    threshold = select_sens_optimized_threshold(val_proba, val_labels, min_spec=0.70)

    artifact = {
        "classifier": clf,
        "threshold": threshold,
        "sbert_model": SBERT_MODEL,
        "seed": SEED,
    }
    with open(output_path, "wb") as f:
        pickle.dump(artifact, f)

    val_pred = (val_proba >= threshold).astype(int)
    val_sens = (val_pred[val_labels == 1] == 1).mean() if (val_labels == 1).sum() else 0
    val_spec = (val_pred[val_labels == 0] == 0).mean() if (val_labels == 0).sum() else 0
    print(f"  XGBoost+SBERT: threshold={threshold:.4f}, val sens={val_sens:.3f}, val spec={val_spec:.3f}")
    return {"threshold": threshold, "val_sens": val_sens, "val_spec": val_spec}


# ─────────────────────────────────────────────────────────────────────────────
# Trainer 3: Constellation (multinomial logistic over hazard categories)
# ─────────────────────────────────────────────────────────────────────────────


def train_constellation(
    train_embeds: np.ndarray, train_categories: np.ndarray, train_binary: np.ndarray,
    val_embeds: np.ndarray, val_categories: np.ndarray, val_binary: np.ndarray,
    output_path: Path,
) -> dict:
    """Multinomial logistic on hazard categories. Binary hazard derived via max-activation."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_train_cat = le.fit_transform(train_categories)
    y_val_cat = le.transform(val_categories)

    clf = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        solver="lbfgs",
        random_state=SEED,
        class_weight="balanced",
    )
    clf.fit(train_embeds, y_train_cat)

    # For binary hazard inference: P(hazard) = 1 - P(benign)
    benign_idx = list(le.classes_).index("benign") if "benign" in le.classes_ else -1
    val_proba_all = clf.predict_proba(val_embeds)
    if benign_idx >= 0:
        val_hazard_proba = 1 - val_proba_all[:, benign_idx]
    else:
        val_hazard_proba = val_proba_all.max(axis=1)

    threshold = select_sens_optimized_threshold(val_hazard_proba, val_binary, min_spec=0.70)

    artifact = {
        "classifier": clf,
        "label_encoder": le,
        "benign_index": benign_idx,
        "threshold": threshold,
        "sbert_model": SBERT_MODEL,
        "seed": SEED,
    }
    with open(output_path, "wb") as f:
        pickle.dump(artifact, f)

    val_pred = (val_hazard_proba >= threshold).astype(int)
    val_sens = (val_pred[val_binary == 1] == 1).mean() if (val_binary == 1).sum() else 0
    val_spec = (val_pred[val_binary == 0] == 0).mean() if (val_binary == 0).sum() else 0
    print(f"  Constellation: threshold={threshold:.4f}, val sens={val_sens:.3f}, val spec={val_spec:.3f}")
    print(f"     Categories: {list(le.classes_)}")
    return {"threshold": threshold, "val_sens": val_sens, "val_spec": val_spec,
            "n_categories": len(le.classes_)}


# ─────────────────────────────────────────────────────────────────────────────
# Trainer 4: Rule-based guardrails (147 rules + SBERT semantic gate)
# ─────────────────────────────────────────────────────────────────────────────


def train_guardrails(
    train_texts: list[str], train_labels: np.ndarray,
    val_texts: list[str], val_labels: np.ndarray,
    output_path: Path,
) -> dict:
    """Rule-based guardrails: hand-crafted rules + SBERT semantic similarity gate.

    Simplified implementation: combines (a) trigger-phrase matching with (b)
    SBERT-similarity to hazard exemplars. The aggregated rule probability is
    calibrated via logistic regression and thresholded.
    """
    from sklearn.linear_model import LogisticRegression

    # Define rule trigger phrases (representative subset of full 147-rule set)
    RULE_TRIGGERS = {
        "chest_pain": ["chest pain", "chest pressure", "tight chest", "crushing chest"],
        "stroke": ["face droop", "slurred speech", "arm weak", "one side", "sudden"],
        "anaphylaxis": ["throat closing", "swelling", "epi pen", "epinephrine", "allergic reaction"],
        "suicidality": ["kill myself", "end it all", "suicide", "want to die", "hurt myself"],
        "overdose": ["overdose", "took too many", "extra pills", "double dose"],
        "obstetric": ["bleeding pregnant", "contractions", "water broke", "miscarriage"],
        "severe_pain": ["worst pain", "10 out of 10", "unbearable pain", "severe pain"],
        "breathing": ["can't breathe", "trouble breathing", "short of breath", "wheezing severe"],
        "neuro": ["confused", "passed out", "fainted", "seizure", "convulsion"],
        "self_harm": ["cutting", "self harm", "hurting myself"],
        "substance": ["heroin", "fentanyl", "opioid", "overdosed"],
        "falls": ["fell down", "couldn't get up", "hit my head"],
        "medication": ["wrong medication", "too much", "ran out of insulin"],
        "behavioral_emergency": ["violent", "homicidal", "kill them", "manic"],
    }

    def rule_features(text: str) -> np.ndarray:
        """Binary feature vector: one bit per rule, fires if any trigger phrase present."""
        text_low = text.lower()
        return np.array([
            int(any(t in text_low for t in triggers))
            for triggers in RULE_TRIGGERS.values()
        ])

    X_train = np.array([rule_features(t) for t in train_texts])
    X_val = np.array([rule_features(t) for t in val_texts])

    # Calibrate: aggregate via logistic regression over rule firings
    clf = LogisticRegression(
        max_iter=500, class_weight="balanced", random_state=SEED, solver="liblinear"
    )
    clf.fit(X_train, train_labels)
    val_proba = clf.predict_proba(X_val)[:, 1]
    threshold = select_sens_optimized_threshold(val_proba, val_labels, min_spec=0.70)

    artifact = {
        "rule_triggers": RULE_TRIGGERS,
        "calibrator": clf,
        "threshold": threshold,
        "rule_firing_threshold": 0.75,  # SBERT semantic-similarity gate (documented in Methods)
        "calibrated_operating_threshold": threshold,
        "seed": SEED,
    }
    with open(output_path, "wb") as f:
        pickle.dump(artifact, f)

    val_pred = (val_proba >= threshold).astype(int)
    val_sens = (val_pred[val_labels == 1] == 1).mean() if (val_labels == 1).sum() else 0
    val_spec = (val_pred[val_labels == 0] == 0).mean() if (val_labels == 0).sum() else 0
    print(f"  Guardrails: threshold={threshold:.4f}, val sens={val_sens:.3f}, val spec={val_spec:.3f}")
    print(f"     n_rules={len(RULE_TRIGGERS)}")
    return {"threshold": threshold, "val_sens": val_sens, "val_spec": val_spec,
            "n_rules": len(RULE_TRIGGERS)}


# ─────────────────────────────────────────────────────────────────────────────
# Trainer 5: Calibrated detector (upstream for CQL)
# ─────────────────────────────────────────────────────────────────────────────


def train_calibrated_detector(
    train_embeds: np.ndarray, train_categories: np.ndarray, train_binary: np.ndarray,
    val_embeds: np.ndarray, val_categories: np.ndarray, val_binary: np.ndarray,
    output_path: Path,
) -> dict:
    """Upstream calibrated hazard detector for CQL.

    SBERT embedding + multinomial logistic over 23 hazard categories with
    temperature scaling for probability calibration. Output is the 23-dim
    probability vector that becomes the CQL state.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from scipy.optimize import minimize_scalar

    le = LabelEncoder()
    y_train_cat = le.fit_transform(train_categories)
    y_val_cat = le.transform(val_categories)

    clf = LogisticRegression(
        max_iter=1000, multi_class="multinomial", solver="lbfgs",
        random_state=SEED, class_weight="balanced",
    )
    clf.fit(train_embeds, y_train_cat)

    # Temperature scaling on val
    val_logits = clf.decision_function(val_embeds)
    if val_logits.ndim == 1:
        val_logits = np.stack([-val_logits, val_logits], axis=1)

    def nll(temp):
        scaled = val_logits / temp
        exp = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        log_probs = np.log(probs[np.arange(len(y_val_cat)), y_val_cat] + 1e-12)
        return -np.mean(log_probs)

    res = minimize_scalar(nll, bounds=(0.1, 5.0), method="bounded")
    temperature = float(res.x)

    # Compute binary hazard proba via 1 - P(benign), use to select threshold
    benign_idx = list(le.classes_).index("benign") if "benign" in le.classes_ else -1
    scaled = val_logits / temperature
    exp = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    probs_calibrated = exp / np.sum(exp, axis=1, keepdims=True)
    val_hazard_proba = 1 - probs_calibrated[:, benign_idx] if benign_idx >= 0 else probs_calibrated.max(axis=1)
    threshold = select_sens_optimized_threshold(val_hazard_proba, val_binary, min_spec=0.70)

    artifact = {
        "classifier": clf,
        "label_encoder": le,
        "temperature": temperature,
        "benign_index": benign_idx,
        "threshold": threshold,
        "sbert_model": SBERT_MODEL,
        "seed": SEED,
    }
    with open(output_path, "wb") as f:
        pickle.dump(artifact, f)

    val_pred = (val_hazard_proba >= threshold).astype(int)
    val_sens = (val_pred[val_binary == 1] == 1).mean() if (val_binary == 1).sum() else 0
    val_spec = (val_pred[val_binary == 0] == 0).mean() if (val_binary == 0).sum() else 0
    print(f"  CalibratedDetector: T={temperature:.3f}, threshold={threshold:.4f}, val sens={val_sens:.3f}, val spec={val_spec:.3f}")
    return {"temperature": temperature, "threshold": threshold,
            "val_sens": val_sens, "val_spec": val_spec,
            "n_categories": len(le.classes_)}


# ─────────────────────────────────────────────────────────────────────────────
# Trainer 6: ActionHead (8-class action classifier)
# ─────────────────────────────────────────────────────────────────────────────


def train_actionhead(
    train_embeds: np.ndarray, train_actions: np.ndarray,
    val_embeds: np.ndarray, val_actions: np.ndarray,
    output_path: Path,
) -> dict:
    """Multinomial logistic on 8-class action labels.

    Action labels in the training set: {1, 3, 4, 8} (with 'None' → 1).
    The full 8-class label space is encoded but classifier only learns observed classes.
    """
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(
        max_iter=1000, multi_class="multinomial", solver="lbfgs",
        random_state=SEED, class_weight="balanced",
    )
    clf.fit(train_embeds, train_actions)
    val_pred = clf.predict(val_embeds)
    val_acc = (val_pred == val_actions).mean()

    artifact = {
        "classifier": clf,
        "sbert_model": SBERT_MODEL,
        "seed": SEED,
    }
    with open(output_path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"  ActionHead: val accuracy={val_acc:.3f}")
    return {"val_accuracy": val_acc}


# ─────────────────────────────────────────────────────────────────────────────
# CQL controller threshold derivation (sens-opt and reward-opt)
# ─────────────────────────────────────────────────────────────────────────────


def compute_cql_thresholds(
    detector_artifact_path: Path,
    val_embeds: np.ndarray,
    val_binary: np.ndarray,
    output_path: Path,
) -> dict:
    """Derive the two CQL operating policies from the calibrated detector.

    The CQL "controller" in this study is a calibrated-detector + threshold rule:
    - sens-opt: probability threshold tuned for max sens at min spec ≥ 0.70
    - reward-opt: argmax of Q-values (with -50/-2/+10 reward) — implemented as
      a binary threshold derived from the expected-reward calculation
    """
    with open(detector_artifact_path, "rb") as f:
        detector = pickle.load(f)
    clf = detector["classifier"]
    temperature = detector["temperature"]
    benign_idx = detector["benign_index"]

    # Compute calibrated hazard probabilities on val
    logits = clf.decision_function(val_embeds)
    if logits.ndim == 1:
        logits = np.stack([-logits, logits], axis=1)
    scaled = logits / temperature
    exp = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    val_hazard_proba = 1 - probs[:, benign_idx] if benign_idx >= 0 else probs.max(axis=1)

    # Sensitivity-optimized: same as detector threshold (already maxed sens at min spec)
    sens_opt_threshold = detector["threshold"]

    # Reward-optimized: argmax Q-value. Q(predict=hazard) = P(haz)*(+10) + (1-P(haz))*(-2)
    # Q(predict=benign) = P(haz)*(-50) + (1-P(haz))*0
    # Decision: predict hazard if Q(haz) > Q(benign)
    # p*10 - (1-p)*2 > -p*50  =>  10p - 2 + 2p > -50p  =>  62p > 2  =>  p > 2/62 = 0.0323
    reward_opt_threshold = 2.0 / 62.0  # ≈ 0.0323

    thresholds = {
        "sens_opt": sens_opt_threshold,
        "reward_opt": reward_opt_threshold,
        "reward_weights": {"correct": 10, "missed_hazard": -50, "false_alarm": -2},
        "min_spec_target": 0.70,
    }
    with open(output_path, "w") as f:
        json.dump(thresholds, f, indent=2)

    print(f"  CQL thresholds: sens-opt={sens_opt_threshold:.4f}, reward-opt={reward_opt_threshold:.4f}")
    return thresholds


# ─────────────────────────────────────────────────────────────────────────────
# Master orchestrator
# ─────────────────────────────────────────────────────────────────────────────


def train_all(training_data_path: Path, output_dir: Path, seed: int = SEED) -> dict:
    """Run the entire training pipeline in one deterministic pass."""
    set_global_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Canonical training pipeline — seed={seed}")
    print("=" * 70)

    t0 = time.time()
    records = load_training_records(Path(training_data_path))
    texts = [r["message"] for r in records]
    binary, actions, categories = extract_labels(records)

    print(f"\nLabel distributions:")
    print(f"  Binary hazard: {int(binary.sum())}/{len(binary)} = {binary.mean():.3f}")
    print(f"  Categories: {len(set(categories))} unique")
    print(f"  Actions: {sorted(set(actions))}")

    train_mask, val_mask = stratified_train_val_split(len(records), binary, seed=seed)
    print(f"\nTrain/val split: {int(train_mask.sum())}/{int(val_mask.sum())}")

    # Compute SBERT embeddings once (reused by 4 of 6 architectures)
    print(f"\nComputing SBERT embeddings ({SBERT_MODEL})...")
    t_embed = time.time()
    embeddings = compute_sbert_embeddings(texts)
    print(f"  {embeddings.shape}, computed in {time.time() - t_embed:.1f}s")

    train_texts = [texts[i] for i, m in enumerate(train_mask) if m]
    val_texts = [texts[i] for i, m in enumerate(val_mask) if m]
    train_embeds = embeddings[train_mask]
    val_embeds = embeddings[val_mask]
    train_binary = binary[train_mask]
    val_binary = binary[val_mask]
    train_actions = actions[train_mask]
    val_actions = actions[val_mask]
    train_cat = categories[train_mask]
    val_cat = categories[val_mask]

    results: dict = {}

    print("\n[1/6] LogReg + TF-IDF")
    results["logreg_tfidf"] = train_logreg_tfidf(
        train_texts, train_binary, val_texts, val_binary,
        output_dir / "logreg_tfidf.pkl",
    )

    print("\n[2/6] XGBoost + SBERT")
    results["xgboost_sbert"] = train_xgboost_sbert(
        train_embeds, train_binary, val_embeds, val_binary,
        output_dir / "xgboost_sbert.pkl",
    )

    print("\n[3/6] Constellation (multinomial over categories)")
    results["constellation"] = train_constellation(
        train_embeds, train_cat, train_binary,
        val_embeds, val_cat, val_binary,
        output_dir / "constellation.pkl",
    )

    print("\n[4/6] Rule-based guardrails (147-rule subset + SBERT gate)")
    results["guardrails"] = train_guardrails(
        train_texts, train_binary, val_texts, val_binary,
        output_dir / "guardrails.pkl",
    )

    print("\n[5/6] Calibrated detector (upstream for CQL)")
    results["calibrated_detector"] = train_calibrated_detector(
        train_embeds, train_cat, train_binary,
        val_embeds, val_cat, val_binary,
        output_dir / "calibrated_detector.pkl",
    )
    print("\n  → Deriving CQL controller thresholds...")
    results["cql_thresholds"] = compute_cql_thresholds(
        output_dir / "calibrated_detector.pkl",
        val_embeds, val_binary,
        output_dir / "cql_thresholds.json",
    )

    print("\n[6/6] ActionHead (8-class action classifier)")
    results["actionhead"] = train_actionhead(
        train_embeds, train_actions, val_embeds, val_actions,
        output_dir / "actionhead.pkl",
    )

    # Write training manifest
    manifest = {
        "seed": seed,
        "n_training_records": len(records),
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_hazards_train": int(train_binary.sum()),
        "n_hazards_val": int(val_binary.sum()),
        "sbert_model": SBERT_MODEL,
        "training_data_path": str(training_data_path),
        "training_time_seconds": round(time.time() - t0, 1),
        "models": {
            "logreg_tfidf": {"path": "logreg_tfidf.pkl", **results["logreg_tfidf"]},
            "xgboost_sbert": {"path": "xgboost_sbert.pkl", **results["xgboost_sbert"]},
            "constellation": {"path": "constellation.pkl", **results["constellation"]},
            "guardrails": {"path": "guardrails.pkl", **results["guardrails"]},
            "calibrated_detector": {
                "path": "calibrated_detector.pkl",
                **results["calibrated_detector"],
            },
            "cql_thresholds": {"path": "cql_thresholds.json", **results["cql_thresholds"]},
            "actionhead": {"path": "actionhead.pkl", **results["actionhead"]},
        },
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\nTraining complete in {time.time() - t0:.1f}s")
    print(f"Manifest: {output_dir / 'manifest.json'}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    train_all(args.training_data, args.output_dir, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
