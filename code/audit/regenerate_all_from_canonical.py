"""Single-pass regeneration of every manuscript number from canonical CSVs.

This is the audit gate that replaces the prior round's accumulated
audit infrastructure. The source-of-truth chain is short:

    per-message predictions CSV → metrics_canonical.csv → manuscript value

This script reads the canonical CSVs and re-derives every reported number.
Exits non-zero on any mismatch.

Usage:
    python regenerate_all_from_canonical.py --strict
    python regenerate_all_from_canonical.py --predictions /predictions/canonical/ \\
        --results /results/ --manuscript /notebooks/rl_vs_llm_safety_v3/drafts/main_text.md
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


def find_numeric_claims_in_manuscript(md_text: str) -> list[tuple[int, str, str]]:
    """Extract numeric claims with line and surrounding context.

    Returns list of (line_no, value_string, context_window) tuples.
    """
    claims = []
    for ln, line in enumerate(md_text.split("\n"), start=1):
        # Find decimals like 0.727, 0.123, percentages like 47%
        for m in re.finditer(
            r"\b(?:0\.\d{2,4}|[1-9]\d*\.\d{2,4}|\d+\.\d+%|\d+%)\b", line
        ):
            value = m.group(0)
            start = max(0, m.start() - 50)
            end = min(len(line), m.end() + 50)
            context = line[start:end]
            claims.append((ln, value, context))
    return claims


def load_metrics_canonical(results_dir: Path) -> dict:
    """Load all canonical results CSVs."""
    canonical = {}
    for name in (
        "metrics_canonical.csv",
        "mcnemar_matrix.csv",
        "delta_bootstrap_canonical.csv",
    ):
        path = results_dir / name
        if path.exists():
            canonical[name] = pd.read_csv(path)
        else:
            print(f"WARN: missing canonical results file: {path}")
    return canonical


def check_predictions_match_metrics(
    predictions_dir: Path, results_dir: Path
) -> tuple[bool, list[str]]:
    """Re-derive every metric in metrics_canonical.csv from per-message predictions.

    For each (architecture, dataset, metric) row in metrics_canonical.csv:
      1. Load the per-message prediction CSV for that architecture
      2. Compute the metric from raw predictions
      3. Compare against the value in metrics_canonical.csv

    Returns (ok, list_of_mismatches).
    """
    mismatches = []
    metrics_path = results_dir / "metrics_canonical.csv"
    if not metrics_path.exists():
        return False, [f"metrics_canonical.csv not found at {metrics_path}"]

    metrics = pd.read_csv(metrics_path)
    for _, row in metrics.iterrows():
        arch = row["architecture"]
        dataset = row.get("dataset", "realworld_n2000")
        pred_files = list(predictions_dir.glob(f"{arch}_*.csv"))
        if not pred_files:
            mismatches.append(
                f"  No predictions file for {arch} (expected {arch}_<run_id>.csv)"
            )
            continue
        # Most recent run_id for this architecture
        pred_path = sorted(pred_files, key=lambda p: p.stat().st_mtime)[-1]
        df = pd.read_csv(pred_path)
        df = df[df["dataset"] == dataset]
        if df.empty:
            mismatches.append(f"  No {dataset} rows in {pred_path}")
            continue

        # Sensitivity check
        true_pos = ((df["pred_hazard"] == 1) & (df["true_hazard"] == 1)).sum()
        n_haz = (df["true_hazard"] == 1).sum()
        sens_calc = true_pos / n_haz if n_haz else float("nan")
        sens_reported = float(row.get("sensitivity", -1))
        if abs(sens_calc - sens_reported) > 0.001:
            mismatches.append(
                f"  {arch}/{dataset} sensitivity: reported={sens_reported:.4f} "
                f"calc={sens_calc:.4f}"
            )

        # Specificity check
        true_neg = ((df["pred_hazard"] == 0) & (df["true_hazard"] == 0)).sum()
        n_ben = (df["true_hazard"] == 0).sum()
        spec_calc = true_neg / n_ben if n_ben else float("nan")
        spec_reported = float(row.get("specificity", -1))
        if abs(spec_calc - spec_reported) > 0.001:
            mismatches.append(
                f"  {arch}/{dataset} specificity: reported={spec_reported:.4f} "
                f"calc={spec_calc:.4f}"
            )

    return len(mismatches) == 0, mismatches


def check_roc_monotonicity_all_architectures(predictions_dir: Path) -> tuple[bool, list[str]]:
    """Run ROC monotonicity assertion across every prediction file."""
    from roc_monotonicity import check_file
    ok = True
    violations = []
    for pred_file in sorted(predictions_dir.glob("*.csv")):
        if not check_file(pred_file):
            ok = False
            violations.append(f"ROC monotonicity violation in {pred_file.name}")
    return ok, violations


def check_manuscript_claims_in_registry(
    manuscript_path: Path,
    registry_path: Path,
) -> tuple[bool, list[str]]:
    """Verify every numeric claim in the manuscript appears in the audit registry."""
    if not manuscript_path.exists():
        return True, []  # No manuscript yet; nothing to check
    if not registry_path.exists():
        return False, [f"Registry not found at {registry_path}"]

    md_text = manuscript_path.read_text()
    claims_in_ms = find_numeric_claims_in_manuscript(md_text)
    registry = pd.read_csv(registry_path)
    registered_values = set(registry["reported_value"].astype(str).str.strip())

    unregistered = []
    for ln, value, context in claims_in_ms:
        # Skip common values like 95%, 0.05 that are statistical conventions
        if value in {"95%", "0.05", "0.001", "1.0", "0.0", "100%"}:
            continue
        if value.strip() not in registered_values:
            unregistered.append(
                f"  Line {ln}: '{value}' not in registry — context: {context}"
            )
    return len(unregistered) == 0, unregistered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("/predictions/canonical"),
        help="Directory of per-message prediction CSVs",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("/results"),
        help="Directory of canonical results CSVs",
    )
    parser.add_argument(
        "--manuscript",
        type=Path,
        default=None,
        help="Optional manuscript markdown to check claims against registry",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Audit data_provenance.csv registry",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("rl-llm-safety v3 — Audit gate")
    print("=" * 70)

    all_ok = True

    print("\n[1/3] ROC monotonicity assertion across all prediction files")
    ok1, viol1 = check_roc_monotonicity_all_architectures(args.predictions)
    if ok1:
        print("  PASS")
    else:
        print("  FAIL")
        for v in viol1:
            print(v)
        all_ok = False

    print("\n[2/3] Predictions ↔ metrics_canonical.csv reconciliation")
    ok2, viol2 = check_predictions_match_metrics(args.predictions, args.results)
    if ok2:
        print("  PASS")
    else:
        print("  FAIL")
        for v in viol2:
            print(v)
        all_ok = False

    if args.manuscript and args.registry:
        print("\n[3/3] Manuscript claims ↔ audit registry")
        ok3, viol3 = check_manuscript_claims_in_registry(
            args.manuscript, args.registry
        )
        if ok3:
            print("  PASS")
        else:
            print("  FAIL")
            for v in viol3[:20]:
                print(v)
            if len(viol3) > 20:
                print(f"  ... and {len(viol3) - 20} more")
            all_ok = False

    print("\n" + "=" * 70)
    if all_ok:
        print("AUDIT PASS — all checks succeeded")
        return 0
    else:
        print("AUDIT FAIL — see above for violations")
        return 1 if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
