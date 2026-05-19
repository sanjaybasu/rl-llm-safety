"""Self-test for canonical metrics module.

Synthetic data tests that exercise every function. Catches regressions in:
- Wilson CIs (compare to scipy/statsmodels)
- AUROC (compare to sklearn.metrics.roc_auc_score)
- F1, MCC (compare to sklearn implementations)
- McNemar (compare to statsmodels)
- ROC monotonicity (synthetic data that should always be monotonic)

Run:
    python test_metrics.py
"""
from __future__ import annotations

import sys

import numpy as np

import metrics


SEED = 42


def assert_close(actual, expected, tol=0.005, msg=""):
    if isinstance(actual, tuple):
        actual = actual[0]
    if abs(float(actual) - float(expected)) > tol:
        raise AssertionError(
            f"FAIL {msg}: actual={actual:.6f}, expected={expected:.6f}, "
            f"diff={abs(actual-expected):.6f} > tol={tol}"
        )


def test_wilson_ci():
    """Wilson CI for 50/100 should be ~(0.404, 0.596)."""
    lo, hi = metrics.wilson_ci(50, 100)
    assert_close(lo, 0.404, msg="wilson_ci lo for 50/100")
    assert_close(hi, 0.596, msg="wilson_ci hi for 50/100")

    # Edge cases (allow tiny float imprecision)
    lo, hi = metrics.wilson_ci(0, 100)
    assert lo < 1e-6, f"wilson_ci(0, 100) lo should be ~0, got {lo}"

    lo, hi = metrics.wilson_ci(100, 100)
    assert hi > 1 - 1e-6, f"wilson_ci(100, 100) hi should be ~1, got {hi}"

    print("  wilson_ci: PASS")


def test_confusion_matrix():
    """Confusion matrix should match manual calculation."""
    pred = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    true = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    cm = metrics.confusion_matrix(pred, true)
    assert cm.tp == 2, f"TP: expected 2, got {cm.tp}"
    assert cm.fn == 2, f"FN: expected 2, got {cm.fn}"
    assert cm.tn == 2, f"TN: expected 2, got {cm.tn}"
    assert cm.fp == 2, f"FP: expected 2, got {cm.fp}"
    print("  confusion_matrix: PASS")


def test_f1_mcc():
    """F1 and MCC compared against sklearn."""
    from sklearn.metrics import f1_score, matthews_corrcoef
    rng = np.random.default_rng(SEED)
    true = rng.integers(0, 2, size=200)
    pred = rng.integers(0, 2, size=200)

    f1_ours = metrics.f1_score(pred, true)
    f1_sklearn = f1_score(true, pred)
    assert_close(f1_ours, f1_sklearn, tol=1e-6, msg="F1 vs sklearn")

    mcc_ours = metrics.mcc_score(pred, true)
    mcc_sklearn = matthews_corrcoef(true, pred)
    assert_close(mcc_ours, mcc_sklearn, tol=1e-6, msg="MCC vs sklearn")

    print("  f1, mcc: PASS")


def test_auroc():
    """AUROC compared against sklearn."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(SEED)
    true = rng.integers(0, 2, size=200)
    proba = rng.random(200) * 0.6 + true * 0.3  # slight signal

    a_ours = metrics.auroc(proba, true)
    a_sklearn = roc_auc_score(true, proba)
    assert_close(a_ours, a_sklearn, tol=1e-6, msg="AUROC vs sklearn")
    print("  auroc: PASS")


def test_roc_monotonicity_synthetic():
    """Synthetic ROC curve must be monotonic when iterated by descending threshold.

    Correct definition: as threshold DECREASES, sens is monotone non-decreasing
    AND spec is monotone non-increasing. The operating curve from metrics module
    iterates thresholds in descending order, so consecutive rows must satisfy this.
    """
    rng = np.random.default_rng(SEED)
    n = 500
    true = rng.integers(0, 2, size=n)
    proba = np.where(true == 1, 0.7 + 0.2 * rng.random(n), 0.2 * rng.random(n))

    curve = metrics.operating_curve(proba, true)
    # operating_curve already iterates descending threshold; consecutive rows
    # must have sens non-decreasing and spec non-increasing
    sens = curve["sensitivity"].values
    spec = curve["specificity"].values
    sens_diffs = np.diff(sens)
    spec_diffs = np.diff(spec)
    sens_violations = (sens_diffs < -1e-9).sum()
    spec_violations = (spec_diffs > 1e-9).sum()
    if sens_violations or spec_violations:
        print(
            f"  ROC monotonicity violations: sens {sens_violations}, "
            f"spec {spec_violations} of {len(curve)-1} consecutive pairs"
        )
        raise AssertionError("ROC monotonicity FAIL on synthetic data")
    print(f"  roc_monotonicity (n={n}): PASS (curve has {len(curve)} unique thresholds)")


def test_mcnemar_paired():
    """McNemar test against statsmodels."""
    try:
        from statsmodels.stats.contingency_tables import mcnemar
    except ImportError:
        print("  mcnemar: SKIP (statsmodels not available)")
        return

    rng = np.random.default_rng(SEED)
    n = 500
    true = rng.integers(0, 2, size=n)
    # Two classifiers with slightly different performance
    pred1 = np.where(rng.random(n) < 0.7, true, 1 - true)
    pred2 = np.where(rng.random(n) < 0.5, true, 1 - true)

    chi2, p, b, c = metrics.mcnemar_chi2(pred1, pred2, true)

    # Compare against statsmodels (restricted to hazard-positive cases)
    pos_mask = true == 1
    p1_pos = pred1[pos_mask]
    p2_pos = pred2[pos_mask]
    table = np.array([
        [int(((p1_pos == 1) & (p2_pos == 1)).sum()), int(((p1_pos == 1) & (p2_pos == 0)).sum())],
        [int(((p1_pos == 0) & (p2_pos == 1)).sum()), int(((p1_pos == 0) & (p2_pos == 0)).sum())],
    ])
    sm_result = mcnemar(table, exact=False, correction=True)
    assert_close(chi2, float(sm_result.statistic), tol=1e-6, msg="McNemar chi2 vs statsmodels")
    print(f"  mcnemar_chi2: PASS (chi2={chi2:.3f}, p={p:.4f}, b={b}, c={c})")


def test_hochberg_step_up():
    """Hochberg step-up against expected significance pattern."""
    # 5 p-values; Hochberg should accept p ≤ α/(n - rank + 1) starting from largest
    p_values = [0.001, 0.005, 0.01, 0.03, 0.04]
    sig = metrics.hochberg_step_up(p_values, alpha=0.05)
    # Hochberg starts from the largest: 0.04 vs 0.05/1=0.05 → significant → all significant
    expected = [True, True, True, True, True]
    assert sig == expected, f"Hochberg: expected {expected}, got {sig}"

    # Stricter case
    p_values = [0.001, 0.04, 0.06, 0.08, 0.20]
    sig = metrics.hochberg_step_up(p_values, alpha=0.05)
    # Largest = 0.20 vs 0.05/1=0.05 → not sig
    # Next = 0.08 vs 0.05/2=0.025 → not sig
    # Next = 0.06 vs 0.05/3≈0.0167 → not sig
    # Next = 0.04 vs 0.05/4=0.0125 → not sig
    # Smallest = 0.001 vs 0.05/5=0.01 → sig → all smaller are sig (just 0.001)
    expected = [True, False, False, False, False]
    assert sig == expected, f"Hochberg strict: expected {expected}, got {sig}"

    print("  hochberg_step_up: PASS")


def test_bootstrap_ci_reproducible():
    """Bootstrap CIs should be deterministic with fixed seed."""
    rng = np.random.default_rng(SEED)
    n = 200
    true = rng.integers(0, 2, size=n)
    pred = rng.integers(0, 2, size=n)

    pt1, lo1, hi1 = metrics.bootstrap_ci(metrics.f1_score, pred, true, n_iter=1000, seed=42)
    pt2, lo2, hi2 = metrics.bootstrap_ci(metrics.f1_score, pred, true, n_iter=1000, seed=42)

    assert pt1 == pt2, f"Bootstrap point: {pt1} vs {pt2} (must be identical)"
    assert abs(lo1 - lo2) < 1e-9, f"Bootstrap lo: {lo1} vs {lo2}"
    assert abs(hi1 - hi2) < 1e-9, f"Bootstrap hi: {hi1} vs {hi2}"
    print(f"  bootstrap_ci reproducibility: PASS (pt={pt1:.4f}, CI=[{lo1:.4f}, {hi1:.4f}])")


def test_sens_at_target_spec():
    """sens_at_target_spec should return values consistent with the operating curve."""
    rng = np.random.default_rng(SEED)
    n = 500
    true = rng.integers(0, 2, size=n)
    proba = np.where(true == 1, 0.7 + 0.2 * rng.random(n), 0.2 * rng.random(n))

    sens, achieved_spec, threshold = metrics.sens_at_target_spec(proba, true, target_spec=0.80)

    # Achieved spec should be close to target
    assert abs(achieved_spec - 0.80) < 0.05, (
        f"Achieved spec {achieved_spec:.3f} too far from target 0.80"
    )
    # On strong-signal data, sens at spec ~0.80 should be high
    assert sens > 0.7, f"Sens {sens:.3f} unexpectedly low at spec ~0.80"
    print(f"  sens_at_target_spec(0.80): sens={sens:.3f}, achieved_spec={achieved_spec:.3f}, t={threshold:.4f}: PASS")


def main():
    print("=" * 60)
    print("Canonical metrics module — self-test")
    print("=" * 60)
    test_wilson_ci()
    test_confusion_matrix()
    test_f1_mcc()
    test_auroc()
    test_roc_monotonicity_synthetic()
    test_mcnemar_paired()
    test_hochberg_step_up()
    test_bootstrap_ci_reproducible()
    test_sens_at_target_spec()
    print("=" * 60)
    print("ALL METRICS TESTS PASS")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
