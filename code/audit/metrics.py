"""Canonical metrics computation — the SINGLE source of truth for every metric
in the manuscript.

Every value that appears in any table, figure, abstract sentence, or discussion
sentence must derive from a function in this module applied to a single
per-message prediction CSV. Do not duplicate metric logic anywhere else in the
codebase.

Functions provided:
    compute_metrics_full(df) -> dict[arch, dict[metric, value]]
    bootstrap_ci(pred, true, fn, n_iter=10000, seed=42)
    wilson_ci(k, n, alpha=0.05)
    hanley_mcnelll_auroc_ci(proba, true, alpha=0.05)
    delong_auroc_compare(proba1, proba2, true, alpha=0.05)
    mcnemar_paired(pred1, pred2, true, hochberg=True)
    operating_curve(proba, true) -> DataFrame with (threshold, sens, spec)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy import stats


SEED = 42
N_BOOTSTRAP = 10_000


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix and derived counts
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ConfusionMatrix:
    tp: int
    fn: int
    tn: int
    fp: int

    @property
    def n(self) -> int:
        return self.tp + self.fn + self.tn + self.fp

    @property
    def n_pos(self) -> int:
        return self.tp + self.fn

    @property
    def n_neg(self) -> int:
        return self.tn + self.fp


def confusion_matrix(pred: np.ndarray, true: np.ndarray) -> ConfusionMatrix:
    pred = np.asarray(pred).astype(int)
    true = np.asarray(true).astype(int)
    tp = int(((pred == 1) & (true == 1)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    tn = int(((pred == 0) & (true == 0)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    return ConfusionMatrix(tp=tp, fn=fn, tn=tn, fp=fp)


# ─────────────────────────────────────────────────────────────────────────────
# Wilson score CIs (for proportions)
# ─────────────────────────────────────────────────────────────────────────────


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    Returns (lo, hi).
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, center - half), min(1.0, center + half))


# ─────────────────────────────────────────────────────────────────────────────
# Hanley-McNeil AUROC CI
# ─────────────────────────────────────────────────────────────────────────────


def auroc(proba: np.ndarray, true: np.ndarray) -> float:
    """Mann-Whitney U-based AUROC."""
    proba = np.asarray(proba, dtype=float)
    true = np.asarray(true, dtype=int)
    pos = proba[true == 1]
    neg = proba[true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    u_stat, _ = stats.mannwhitneyu(pos, neg, alternative="greater")
    return float(u_stat) / (len(pos) * len(neg))


def hanley_mcnelll_auroc_ci(
    proba: np.ndarray, true: np.ndarray, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Hanley-McNeil 95% CI for AUROC.

    Returns (auroc_point, ci_lo, ci_hi).
    """
    a = auroc(proba, true)
    if np.isnan(a):
        return (float("nan"), float("nan"), float("nan"))
    n1 = int(np.sum(np.asarray(true) == 1))
    n2 = int(np.sum(np.asarray(true) == 0))
    q1 = a / (2 - a)
    q2 = 2 * a**2 / (1 + a)
    se = np.sqrt(
        (a * (1 - a) + (n1 - 1) * (q1 - a**2) + (n2 - 1) * (q2 - a**2))
        / (n1 * n2)
    )
    z = stats.norm.ppf(1 - alpha / 2)
    return (a, max(0.0, a - z * se), min(1.0, a + z * se))


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap CIs for any statistic
# ─────────────────────────────────────────────────────────────────────────────


def bootstrap_ci(
    statistic: Callable[[np.ndarray, np.ndarray], float],
    pred: np.ndarray,
    true: np.ndarray,
    n_iter: int = N_BOOTSTRAP,
    seed: int = SEED,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for statistic(pred, true).

    Returns (point_estimate, ci_lo, ci_hi).
    """
    pred = np.asarray(pred)
    true = np.asarray(true)
    n = len(pred)
    rng = np.random.default_rng(seed)

    point = statistic(pred, true)
    boot_stats = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        boot_stats[i] = statistic(pred[idx], true[idx])

    lo = float(np.nanpercentile(boot_stats, 100 * alpha / 2))
    hi = float(np.nanpercentile(boot_stats, 100 * (1 - alpha / 2)))
    return (float(point), lo, hi)


# ─────────────────────────────────────────────────────────────────────────────
# Derived metrics: F1, MCC, PPV, NPV
# ─────────────────────────────────────────────────────────────────────────────


def f1_score(pred: np.ndarray, true: np.ndarray) -> float:
    cm = confusion_matrix(pred, true)
    if cm.tp == 0:
        return 0.0
    p = cm.tp / (cm.tp + cm.fp)
    r = cm.tp / (cm.tp + cm.fn)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def mcc_score(pred: np.ndarray, true: np.ndarray) -> float:
    cm = confusion_matrix(pred, true)
    denom = np.sqrt(
        float((cm.tp + cm.fp) * (cm.tp + cm.fn) * (cm.tn + cm.fp) * (cm.tn + cm.fn))
    )
    if denom == 0:
        return 0.0
    return (cm.tp * cm.tn - cm.fp * cm.fn) / denom


def ppv(pred: np.ndarray, true: np.ndarray) -> float:
    cm = confusion_matrix(pred, true)
    if cm.tp + cm.fp == 0:
        return float("nan")
    return cm.tp / (cm.tp + cm.fp)


def npv(pred: np.ndarray, true: np.ndarray) -> float:
    cm = confusion_matrix(pred, true)
    if cm.tn + cm.fn == 0:
        return float("nan")
    return cm.tn / (cm.tn + cm.fn)


def sensitivity(pred: np.ndarray, true: np.ndarray) -> float:
    cm = confusion_matrix(pred, true)
    if cm.n_pos == 0:
        return float("nan")
    return cm.tp / cm.n_pos


def specificity(pred: np.ndarray, true: np.ndarray) -> float:
    cm = confusion_matrix(pred, true)
    if cm.n_neg == 0:
        return float("nan")
    return cm.tn / cm.n_neg


# ─────────────────────────────────────────────────────────────────────────────
# McNemar paired comparison with Hochberg step-up correction
# ─────────────────────────────────────────────────────────────────────────────


def mcnemar_chi2(pred1: np.ndarray, pred2: np.ndarray, true: np.ndarray) -> tuple[float, float, int, int]:
    """McNemar test on paired binary predictions (continuity-corrected).

    Restricted to positive cases (where true=1), as is standard for sensitivity comparison.

    Returns (chi2, p_value, b, c) where b = pred1 correct & pred2 wrong, c = pred1 wrong & pred2 correct.
    """
    pos_mask = np.asarray(true) == 1
    p1 = np.asarray(pred1)[pos_mask]
    p2 = np.asarray(pred2)[pos_mask]
    b = int(((p1 == 1) & (p2 == 0)).sum())
    c = int(((p1 == 0) & (p2 == 1)).sum())
    if b + c == 0:
        return (0.0, 1.0, b, c)
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)  # continuity correction
    p_value = float(1 - stats.chi2.cdf(chi2, df=1))
    return (chi2, p_value, b, c)


def hochberg_step_up(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Hochberg step-up correction. Returns list of significance booleans."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1], reverse=True)
    sig = [False] * n
    for rank, (idx, p) in enumerate(indexed):
        i = rank + 1  # 1-indexed
        threshold = alpha / i
        if p <= threshold:
            # All remaining (smaller p) are also significant
            for _, (j, _) in enumerate(indexed[rank:]):
                sig[j] = True
            break
    return sig


def mcnemar_matrix(
    predictions: dict[str, np.ndarray],
    true: np.ndarray,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Build the full pairwise McNemar matrix.

    predictions: dict mapping architecture name to per-message binary predictions
    true: per-message true_hazard labels

    Returns a long-format DataFrame: arch_a, arch_b, chi2, p_raw, b, c, sig_hochberg
    """
    archs = list(predictions.keys())
    rows = []
    p_raw_list = []
    pair_list = []
    for i, a in enumerate(archs):
        for b in archs[i + 1 :]:
            chi2, p, b_count, c_count = mcnemar_chi2(predictions[a], predictions[b], true)
            rows.append({
                "arch_a": a, "arch_b": b,
                "chi2": chi2, "p_raw": p,
                "b": b_count, "c": c_count,
            })
            p_raw_list.append(p)
            pair_list.append((a, b))

    sig = hochberg_step_up(p_raw_list, alpha=alpha)
    for row, s in zip(rows, sig):
        row["sig_hochberg"] = bool(s)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Operating curve — ALL operating points from ONE prediction file
# ─────────────────────────────────────────────────────────────────────────────


def operating_curve(proba: np.ndarray, true: np.ndarray) -> pd.DataFrame:
    """Full operating curve: at every unique threshold, return (threshold, sens, spec, fpr).

    This is the foundation of ROC monotonicity. Tables 2 and 3 should both
    threshold this same curve at different points; they cannot disagree if
    they come from the same prediction file.
    """
    proba = np.asarray(proba, dtype=float)
    true = np.asarray(true, dtype=int)
    thresholds = np.unique(proba)[::-1]  # descending
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        sens_val = sensitivity(pred, true)
        spec_val = specificity(pred, true)
        rows.append({
            "threshold": float(t),
            "sensitivity": sens_val,
            "specificity": spec_val,
            "fpr": 1 - spec_val,
        })
    return pd.DataFrame(rows)


def sens_at_target_spec(
    proba: np.ndarray, true: np.ndarray, target_spec: float
) -> tuple[float, float, float]:
    """Find the threshold that achieves target specificity, return (sens_at_threshold,
    achieved_spec, threshold).

    Used to populate Table 3: the threshold IS selected from the same prediction
    file used in Table 2, so the achieved-spec in Table 3 must be reachable along
    the same ROC curve from which Table 2's default-threshold (sens, spec) is taken.
    """
    curve = operating_curve(proba, true)
    # Find the threshold whose specificity is closest to target_spec
    curve["spec_distance"] = (curve["specificity"] - target_spec).abs()
    best = curve.sort_values(["spec_distance", "threshold"]).iloc[0]
    return float(best["sensitivity"]), float(best["specificity"]), float(best["threshold"])


# ─────────────────────────────────────────────────────────────────────────────
# Master compute function: every metric for every architecture × dataset
# ─────────────────────────────────────────────────────────────────────────────


def compute_full_metrics(
    df: pd.DataFrame, prevalence: Optional[float] = None
) -> dict:
    """Compute every metric for ONE (architecture, dataset) per-message predictions DataFrame.

    df columns required: pred_hazard, true_hazard
    df columns optional: pred_proba (enables AUROC + ROC + operating-curve metrics)

    Returns a flat dict of metric_name → value (or (point, lo, hi) for CIs).
    """
    pred = df["pred_hazard"].values
    true = df["true_hazard"].values
    cm = confusion_matrix(pred, true)
    n = cm.n
    n_pos = cm.n_pos
    n_neg = cm.n_neg
    if prevalence is None:
        prevalence = n_pos / n if n else float("nan")

    out = {
        "n_total": n,
        "n_hazards": n_pos,
        "n_benigns": n_neg,
        "prevalence": prevalence,
        "tp": cm.tp, "fn": cm.fn, "tn": cm.tn, "fp": cm.fp,
        "sensitivity": sensitivity(pred, true),
        "sensitivity_ci_lo": wilson_ci(cm.tp, cm.n_pos)[0],
        "sensitivity_ci_hi": wilson_ci(cm.tp, cm.n_pos)[1],
        "specificity": specificity(pred, true),
        "specificity_ci_lo": wilson_ci(cm.tn, cm.n_neg)[0],
        "specificity_ci_hi": wilson_ci(cm.tn, cm.n_neg)[1],
        "ppv": ppv(pred, true),
        "npv": npv(pred, true),
    }
    # Bootstrap CIs for F1 and MCC
    f1_pt, f1_lo, f1_hi = bootstrap_ci(f1_score, pred, true)
    mcc_pt, mcc_lo, mcc_hi = bootstrap_ci(mcc_score, pred, true)
    out.update({
        "f1": f1_pt, "f1_ci_lo": f1_lo, "f1_ci_hi": f1_hi,
        "mcc": mcc_pt, "mcc_ci_lo": mcc_lo, "mcc_ci_hi": mcc_hi,
    })
    # FN per 1000 messages
    out["fn_per_1000"] = (1 - out["sensitivity"]) * 1000 * prevalence

    # AUROC if pred_proba is available
    if "pred_proba" in df.columns and df["pred_proba"].notna().any():
        proba = df["pred_proba"].values
        auroc_pt, auroc_lo, auroc_hi = hanley_mcnelll_auroc_ci(proba, true)
        out.update({
            "auroc": auroc_pt,
            "auroc_ci_lo": auroc_lo,
            "auroc_ci_hi": auroc_hi,
        })
        # Balanced accuracy at default threshold (for systems without probs)
        out["balanced_accuracy"] = (out["sensitivity"] + out["specificity"]) / 2
    else:
        out["auroc"] = float("nan")
        out["balanced_accuracy"] = (out["sensitivity"] + out["specificity"]) / 2

    return out
