"""Threshold-optimization analysis — what is the best a single architecture can do?

For each architecture with calibrated probability output, sweep all thresholds
on the operating curve and report the F1-maximizing and MCC-maximizing
operating points alongside the clinical-grade target test (sens ≥ 0.80 AND
spec ≥ 0.80). This is a post-hoc upper-bound analysis: the thresholds are
selected on the test-set operating curve, which characterizes the BEST that
threshold tuning could achieve in principle for each model on this population.

The deployed thresholds reported in Table 2 are calibrated on the
training/validation split, not the test set. The threshold-optimization
analysis here is descriptive of the model's discriminative capacity, not a
prescription for deployment threshold selection.

Output:
    results/threshold_optimized.csv  — per-architecture F1-max, MCC-max,
        threshold values, sens/spec/PPV at each, and clinical-grade reachability

The headline question the analysis answers: can any single architecture
achieve sens ≥ 0.80 AND spec ≥ 0.80 (clinical-grade screening best practice)
at any operating point on its receiver operating characteristic curve?

Inputs: results/operating_curves.csv (real-world n=2000 test set ROC curves)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "audit"))
import metrics as M


CLINICAL_GRADE_SENS = 0.80
CLINICAL_GRADE_SPEC = 0.80


def threshold_optimize(
    curves_df: pd.DataFrame,
    n_hazards: int,
    n_benigns: int,
) -> pd.DataFrame:
    """Per-architecture F1-max and MCC-max operating points."""
    rows = []
    for arch in sorted(curves_df["architecture"].unique()):
        c = curves_df[curves_df["architecture"] == arch].copy()
        sens = c["sensitivity"].values
        spec = c["specificity"].values
        thr = c["threshold"].values if "threshold" in c.columns else np.full_like(sens, np.nan)

        tp = sens * n_hazards
        fn = n_hazards - tp
        tn = spec * n_benigns
        fp = n_benigns - tn

        f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-9)
        mcc = (tp * tn - fp * fn) / denom
        ppv = tp / (tp + fp + 1e-9)

        f1_idx = int(np.argmax(f1))
        mcc_idx = int(np.argmax(mcc))

        clinical_grade_mask = (sens >= CLINICAL_GRADE_SENS) & (spec >= CLINICAL_GRADE_SPEC)
        clinical_grade_reachable = bool(clinical_grade_mask.any())
        if clinical_grade_reachable:
            # Among clinical-grade-reachable points, find the highest-F1 one
            f1_among_cg = np.where(clinical_grade_mask, f1, -1)
            cg_idx = int(np.argmax(f1_among_cg))
            cg_sens = float(sens[cg_idx])
            cg_spec = float(spec[cg_idx])
        else:
            # Find the operating point that comes closest (max min(sens, spec))
            cg_idx = int(np.argmax(np.minimum(sens, spec)))
            cg_sens = float(sens[cg_idx])
            cg_spec = float(spec[cg_idx])

        rows.append({
            "architecture": arch,
            # F1-max operating point
            "f1_max": round(float(f1[f1_idx]), 4),
            "f1_max_sens": round(float(sens[f1_idx]), 4),
            "f1_max_spec": round(float(spec[f1_idx]), 4),
            "f1_max_ppv": round(float(ppv[f1_idx]), 4),
            "f1_max_threshold": float(thr[f1_idx]) if not np.isnan(thr[f1_idx]) else None,
            # MCC-max operating point
            "mcc_max": round(float(mcc[mcc_idx]), 4),
            "mcc_max_sens": round(float(sens[mcc_idx]), 4),
            "mcc_max_spec": round(float(spec[mcc_idx]), 4),
            "mcc_max_ppv": round(float(ppv[mcc_idx]), 4),
            "mcc_max_threshold": float(thr[mcc_idx]) if not np.isnan(thr[mcc_idx]) else None,
            # Clinical-grade reachability
            "clinical_grade_reachable": clinical_grade_reachable,
            "closest_to_clinical_grade_sens": round(cg_sens, 4),
            "closest_to_clinical_grade_spec": round(cg_spec, 4),
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", type=Path, required=True,
                        help="results/operating_curves.csv")
    parser.add_argument("--output", type=Path, required=True,
                        help="results/threshold_optimized.csv")
    parser.add_argument("--n-hazards", type=int, default=165,
                        help="Hazards in test set (default 165 for n=2000)")
    parser.add_argument("--n-benigns", type=int, default=1835,
                        help="Benigns in test set (default 1835 for n=2000)")
    args = parser.parse_args()

    curves = pd.read_csv(args.curves)
    df = threshold_optimize(curves, args.n_hazards, args.n_benigns)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"  → wrote {args.output} ({len(df)} architectures)")

    print("\nThreshold-optimization summary (post-hoc upper bound):")
    print(df[[
        "architecture", "f1_max", "f1_max_sens", "f1_max_spec",
        "mcc_max", "mcc_max_sens", "mcc_max_spec",
        "clinical_grade_reachable",
        "closest_to_clinical_grade_sens", "closest_to_clinical_grade_spec",
    ]].to_string(index=False))

    n_reachable = int(df["clinical_grade_reachable"].sum())
    print(f"\n{n_reachable} / {len(df)} architectures can reach sens ≥ {CLINICAL_GRADE_SENS} "
          f"AND spec ≥ {CLINICAL_GRADE_SPEC} at any operating point on their ROC curve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
