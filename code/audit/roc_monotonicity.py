"""ROC monotonicity assertion — the specific test that the prior submission's
Table 2 vs Table 3 violation would have failed.

For every architecture with calibrated probability outputs, compute the full
ROC curve from the canonical per-message prediction file and assert that
sensitivity is monotonically non-increasing in specificity. This is a
mathematical identity on a single static prediction set; any violation means
two tables were computed from different prediction files.

Usage:
    python roc_monotonicity.py /predictions/canonical/*.csv
    python roc_monotonicity.py /predictions/canonical/ --strict  # exit 1 on failure
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def roc_curve_from_predictions(
    pred_proba: np.ndarray, true_hazard: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (thresholds, sens_array, spec_array) from a single prediction file.

    Thresholds are unique pred_proba values sorted descending. At each threshold,
    sensitivity = TP/(TP+FN) and specificity = TN/(TN+FP).
    """
    pred_proba = np.asarray(pred_proba, dtype=float)
    true_hazard = np.asarray(true_hazard, dtype=int)

    # Unique thresholds, sorted descending so spec is monotonically increasing
    # as threshold increases (fewer positives → fewer false positives → higher spec)
    thresholds = np.unique(pred_proba)[::-1]

    sens_arr = np.empty(len(thresholds))
    spec_arr = np.empty(len(thresholds))
    for i, t in enumerate(thresholds):
        pred = (pred_proba >= t).astype(int)
        tp = int(((pred == 1) & (true_hazard == 1)).sum())
        fn = int(((pred == 0) & (true_hazard == 1)).sum())
        tn = int(((pred == 0) & (true_hazard == 0)).sum())
        fp = int(((pred == 1) & (true_hazard == 0)).sum())
        sens_arr[i] = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec_arr[i] = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return thresholds, sens_arr, spec_arr


def assert_roc_monotonic(
    sens_arr: np.ndarray, spec_arr: np.ndarray, architecture: str
) -> bool:
    """Assert ROC monotonicity on a single prediction file.

    Definition: as threshold DECREASES, sensitivity is monotone non-decreasing
    AND specificity is monotone non-increasing. The arrays are assumed to be in
    descending-threshold order (the natural output of roc_curve_from_predictions).

    Returns True if monotonic; False otherwise.
    """
    sens_diffs = np.diff(sens_arr)
    spec_diffs = np.diff(spec_arr)
    # sens should be non-decreasing: diff >= 0 → violations where diff < 0
    sens_viol = np.where(sens_diffs < -1e-9)[0]
    # spec should be non-increasing: diff <= 0 → violations where diff > 0
    spec_viol = np.where(spec_diffs > 1e-9)[0]

    if len(sens_viol) == 0 and len(spec_viol) == 0:
        return True

    print(
        f"  ROC MONOTONICITY VIOLATION for {architecture}: "
        f"{len(sens_viol)} sens-decreasing pairs, {len(spec_viol)} spec-increasing pairs."
    )
    for v in list(sens_viol[:5]) + list(spec_viol[:5]):
        print(
            f"    Pair {v}: sens {sens_arr[v]:.4f} → {sens_arr[v+1]:.4f} "
            f"(Δsens={sens_diffs[v]:+.4f}), "
            f"spec {spec_arr[v]:.4f} → {spec_arr[v+1]:.4f} (Δspec={spec_diffs[v]:+.4f})"
        )
    return False


def check_file(path: Path) -> bool:
    """Check a single per-message prediction file. Returns True if OK."""
    df = pd.read_csv(path)
    if "pred_proba" not in df.columns or df["pred_proba"].isna().all():
        print(f"  {path.name}: no calibrated probabilities (skipped)")
        return True

    if "true_hazard" not in df.columns:
        print(f"  {path.name}: missing true_hazard column (FAIL)")
        return False

    arch = df["architecture"].iloc[0] if "architecture" in df.columns else path.stem
    dataset = df["dataset"].iloc[0] if "dataset" in df.columns else "unknown"

    # Check per dataset
    ok = True
    for ds, sub in df.groupby("dataset" if "dataset" in df.columns else lambda _: dataset):
        proba = sub["pred_proba"].values
        true = sub["true_hazard"].values
        if np.isnan(proba).all():
            continue
        thresholds, sens, spec = roc_curve_from_predictions(proba, true)
        if not assert_roc_monotonic(sens, spec, f"{arch} ({ds})"):
            ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if any architecture fails the monotonicity check.",
    )
    args = parser.parse_args()

    files: list[Path] = []
    for p in args.paths:
        if p.is_dir():
            files.extend(sorted(p.glob("*.csv")))
        else:
            files.append(p)

    if not files:
        print("No prediction files found.")
        return 1

    print(f"Checking {len(files)} prediction file(s) for ROC monotonicity...")
    all_ok = True
    for f in files:
        if not check_file(f):
            all_ok = False

    if all_ok:
        print("\nALL FILES PASS ROC monotonicity check.")
        return 0
    else:
        print("\nROC MONOTONICITY VIOLATIONS DETECTED.")
        return 1 if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
