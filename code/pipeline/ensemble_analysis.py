"""Ensemble analysis — does combining the 9 existing architectures close the gap?

For each ensemble rule (hard-voting majority, soft-voting where probabilities
exist, AND-rule k-of-N, OR-rule k-of-N), compute the operating-point metrics
on the canonical real-world test set. The question: at any ensemble
configuration, can we reach sens ≥ 0.80 AND spec ≥ 0.80?

This is a post-hoc analysis on the existing per-message predictions; no
new inference. Output: results/ensemble_results.csv

Ensemble rules evaluated:
  - hard_majority: at least floor(N/2)+1 of 9 architectures must flag
  - hard_k_of_N: at least k flags for k in {1, 2, 3, ..., 9}
  - soft_voting_unweighted: mean of pred_proba where present, threshold sweep
  - soft_voting_f1weighted: F1-weighted mean of pred_proba, threshold sweep
  - top2_AND: both of the two highest-F1 architectures flag
  - top3_AND: all three of the top-3 architectures flag
  - top2_OR: either of the top-2 flags
  - top3_OR: any of the top-3 flags

The headline question: does the Pareto envelope of the ensemble configurations
contain a (sens ≥ 0.80, spec ≥ 0.80) operating point?
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "audit"))
import metrics as M


CLINICAL_GRADE = 0.80


def load_aligned(predictions_dir: Path, dataset: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load all per-architecture pred_hazard and pred_proba aligned by message_id."""
    files = sorted(Path(predictions_dir).glob("*.csv"))
    pred_hazard = {}
    pred_proba = {}
    true_hazard: np.ndarray | None = None
    msg_order: list[str] | None = None
    for csv in files:
        df = pd.read_csv(csv)
        if "dataset" not in df.columns:
            continue
        sub = df[df["dataset"] == dataset].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("message_id").reset_index(drop=True)
        arch = sub["architecture"].iloc[0]
        if msg_order is None:
            msg_order = sub["message_id"].astype(str).tolist()
            true_hazard = sub["true_hazard"].astype(int).values
        else:
            sub = sub.set_index("message_id").loc[msg_order].reset_index()
        pred_hazard[arch] = sub["pred_hazard"].astype(int).values
        # pred_proba may be empty / NaN for LLMs
        proba_col = pd.to_numeric(sub["pred_proba"], errors="coerce")
        pred_proba[arch] = proba_col.values
    return pred_hazard, pred_proba, true_hazard


def metrics_from_binary(pred: np.ndarray, true: np.ndarray) -> dict:
    pred = pred.astype(bool)
    true = true.astype(bool)
    tp = int((pred & true).sum())
    fp = int((pred & ~true).sum())
    tn = int((~pred & ~true).sum())
    fn = int((~pred & true).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else float("nan")
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "ppv": round(ppv, 4),
        "f1": round(f1, 4),
        "mcc": round(mcc, 4),
        "clinical_grade": bool(sens >= CLINICAL_GRADE and spec >= CLINICAL_GRADE),
    }


def metrics_from_proba(proba: np.ndarray, true: np.ndarray, thr: float) -> dict:
    pred = (proba >= thr).astype(int)
    m = metrics_from_binary(pred, true)
    m["threshold"] = round(float(thr), 4)
    return m


def analyze_hard_voting(pred_hazard: dict[str, np.ndarray], true: np.ndarray) -> pd.DataFrame:
    archs = sorted(pred_hazard.keys())
    matrix = np.stack([pred_hazard[a] for a in archs])  # shape (N_archs, N_msgs)
    vote_count = matrix.sum(axis=0)  # how many architectures flagged each message
    rows = []
    for k in range(1, len(archs) + 1):
        pred_k = (vote_count >= k).astype(int)
        m = metrics_from_binary(pred_k, true)
        m["rule"] = f"hard_{k}_of_{len(archs)}"
        m["k"] = k
        rows.append(m)
    return pd.DataFrame(rows)


def analyze_soft_voting(pred_proba: dict[str, np.ndarray], true: np.ndarray,
                        weights: dict[str, float] | None = None,
                        rule_label: str = "soft_unweighted") -> pd.DataFrame:
    """Soft voting over architectures with calibrated probabilities."""
    archs = sorted([a for a, p in pred_proba.items() if not np.all(np.isnan(p))])
    if not archs:
        return pd.DataFrame()
    P = np.stack([np.nan_to_num(pred_proba[a], nan=0.0) for a in archs])  # (N_archs, N_msgs)
    if weights is None:
        w = np.ones(len(archs))
    else:
        w = np.array([weights.get(a, 0.0) for a in archs])
    w = w / w.sum() if w.sum() else np.ones_like(w) / len(w)
    ensemble_proba = (w[:, None] * P).sum(axis=0)
    rows = []
    for thr in np.linspace(0.0, 1.0, 101):
        m = metrics_from_proba(ensemble_proba, true, thr)
        m["rule"] = rule_label
        rows.append(m)
    return pd.DataFrame(rows)


def best_pareto_point(df: pd.DataFrame) -> pd.DataFrame:
    """Return the row with highest min(sens, spec) — closest to balanced clinical-grade."""
    if df.empty:
        return df
    df = df.copy()
    df["min_sens_spec"] = df[["sensitivity", "specificity"]].min(axis=1)
    return df.sort_values("min_sens_spec", ascending=False).head(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="realworld_n2000")
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    pred_hazard, pred_proba, true = load_aligned(args.predictions_dir, args.dataset)
    print(f"Loaded {len(pred_hazard)} architectures on {args.dataset} (N={len(true)}, "
          f"hazards={int(true.sum())})")

    all_rows = []

    # Hard voting
    print("\nHard voting (k-of-N):")
    hv = analyze_hard_voting(pred_hazard, true)
    print(hv[["rule", "sensitivity", "specificity", "f1", "mcc", "clinical_grade"]].to_string(index=False))
    all_rows.append(hv)

    # Soft voting unweighted (only over architectures with probabilities)
    print("\nSoft voting (unweighted, threshold sweep — best point shown):")
    sv = analyze_soft_voting(pred_proba, true, rule_label="soft_unweighted")
    if not sv.empty:
        best = best_pareto_point(sv)
        print(best[["rule", "threshold", "sensitivity", "specificity", "f1", "mcc", "clinical_grade"]].to_string(index=False))
        # Keep ALL threshold sweeps for output
        all_rows.append(sv)

    # F1-weighted soft voting — weights from each architecture's solo F1
    # Compute solo F1 for each architecture
    solo_f1 = {}
    for arch, p in pred_hazard.items():
        m = metrics_from_binary(p, true)
        solo_f1[arch] = m["f1"] if m["f1"] == m["f1"] else 0  # NaN-safe
    print("\nSoft voting (F1-weighted, threshold sweep — best point shown):")
    sv_w = analyze_soft_voting(pred_proba, true, weights=solo_f1, rule_label="soft_f1weighted")
    if not sv_w.empty:
        best = best_pareto_point(sv_w)
        print(best[["rule", "threshold", "sensitivity", "specificity", "f1", "mcc", "clinical_grade"]].to_string(index=False))
        all_rows.append(sv_w)

    # All-pair AND and OR — for the two highest-F1 architectures
    by_f1 = sorted(solo_f1.items(), key=lambda x: -x[1])
    top3 = [a for a, _ in by_f1[:3]]
    print(f"\nTop 3 by solo F1: {top3}")
    top3_AND = np.ones(len(true), dtype=bool)
    top3_OR = np.zeros(len(true), dtype=bool)
    for a in top3:
        top3_AND &= pred_hazard[a].astype(bool)
        top3_OR |= pred_hazard[a].astype(bool)
    for label, vec in [("top3_AND", top3_AND.astype(int)), ("top3_OR", top3_OR.astype(int))]:
        m = metrics_from_binary(vec, true)
        m["rule"] = label
        print(f"  {label}: sens={m['sensitivity']}, spec={m['specificity']}, f1={m['f1']}, clinical_grade={m['clinical_grade']}")
        all_rows.append(pd.DataFrame([m]))

    full = pd.concat(all_rows, ignore_index=True, sort=False)
    out_path = args.results_dir / "ensemble_results.csv"
    full.to_csv(out_path, index=False)
    print(f"\n  → wrote {out_path} ({len(full)} ensemble configurations)")

    # Headline question
    n_clinical_grade = int(full["clinical_grade"].sum())
    print(f"\nHEADLINE: {n_clinical_grade} / {len(full)} ensemble configurations reach "
          f"sens ≥ {CLINICAL_GRADE} AND spec ≥ {CLINICAL_GRADE}")
    if n_clinical_grade > 0:
        cg = full[full["clinical_grade"]].sort_values("f1", ascending=False).head(5)
        print("Clinical-grade ensembles (top 5 by F1):")
        print(cg[["rule", "sensitivity", "specificity", "f1", "mcc"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
