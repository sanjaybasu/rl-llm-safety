"""Failure mode analysis — what is being missed and why?

For each hazard in the real-world test set, count how many of the N
architectures correctly flagged it. Messages that all/most architectures
miss are the "hardest" cases; messages everyone catches are the "easiest".
The distribution of hazard-catch counts per message answers two questions:
  1. Is there an irreducible kernel of messages that no architecture catches?
  2. Are there messages catchable by SOME architecture, suggesting an
     ensemble-of-specialists could help?

For each false-positive (benign flagged by ≥1 architecture), similarly count
how many architectures incorrectly flagged it. Frequent false-positive
patterns reveal what triggers spurious alerts.

Output:
    results/failure_mode.csv             — per-message catch count + true label
    results/failure_mode_summary.csv     — distributions + key examples

This analysis is post-hoc on existing per-message predictions; no new compute.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def load_aligned_predictions(predictions_dir: Path, dataset: str) -> tuple[pd.DataFrame, list[str]]:
    """Load all architecture predictions aligned by message_id; return wide DataFrame.

    Wide format: rows = message_id, columns = {true_hazard, true_action,
    hazard_category, <arch1>_pred_hazard, <arch2>_pred_hazard, ...}
    """
    files = sorted(Path(predictions_dir).glob("*.csv"))
    base: pd.DataFrame | None = None
    arch_cols = []
    for csv in files:
        df = pd.read_csv(csv)
        if "dataset" not in df.columns:
            continue
        sub = df[df["dataset"] == dataset].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("message_id").reset_index(drop=True)
        arch = sub["architecture"].iloc[0]
        if base is None:
            base = sub[["message_id", "true_hazard", "true_action", "hazard_category"]].copy()
            base[f"{arch}_pred"] = sub["pred_hazard"].astype(int).values
        else:
            base[f"{arch}_pred"] = sub.set_index("message_id").loc[base["message_id"], "pred_hazard"].astype(int).values
        arch_cols.append(f"{arch}_pred")
    return base, arch_cols


def analyze_failure_modes(wide: pd.DataFrame, arch_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_arch = len(arch_cols)
    wide = wide.copy()
    wide["n_archs_flagged"] = wide[arch_cols].sum(axis=1)
    wide["n_archs_missed"] = n_arch - wide["n_archs_flagged"]

    # Per-hazard catch count distribution
    hazards = wide[wide["true_hazard"] == 1].copy()
    benigns = wide[wide["true_hazard"] == 0].copy()

    # Distribution of catch counts among true hazards
    haz_catch_dist = hazards["n_archs_flagged"].value_counts().sort_index()
    # Distribution of false-positive flag counts among true benigns
    ben_flag_dist = benigns["n_archs_flagged"].value_counts().sort_index()

    # Summary stats
    n_caught_by_none = (hazards["n_archs_flagged"] == 0).sum()
    n_caught_by_all = (hazards["n_archs_flagged"] == n_arch).sum()
    n_caught_by_at_least_one = (hazards["n_archs_flagged"] >= 1).sum()
    n_caught_by_majority = (hazards["n_archs_flagged"] > n_arch / 2).sum()

    n_flagged_by_none = (benigns["n_archs_flagged"] == 0).sum()
    n_flagged_by_all = (benigns["n_archs_flagged"] == n_arch).sum()
    n_flagged_by_at_least_one = (benigns["n_archs_flagged"] >= 1).sum()
    n_flagged_by_majority = (benigns["n_archs_flagged"] > n_arch / 2).sum()

    summary = pd.DataFrame([
        # Hazards
        {"metric": "total hazards in test set", "value": len(hazards)},
        {"metric": "hazards caught by ZERO architectures (irreducibly missed)", "value": int(n_caught_by_none)},
        {"metric": "hazards caught by AT LEAST ONE architecture", "value": int(n_caught_by_at_least_one)},
        {"metric": "hazards caught by MAJORITY (>5/10) architectures", "value": int(n_caught_by_majority)},
        {"metric": "hazards caught by ALL architectures", "value": int(n_caught_by_all)},
        {"metric": "median architectures correctly flagging each hazard", "value": float(hazards["n_archs_flagged"].median())},
        # Benigns
        {"metric": "total benigns in test set", "value": len(benigns)},
        {"metric": "benigns flagged by ZERO architectures (clean)", "value": int(n_flagged_by_none)},
        {"metric": "benigns flagged by AT LEAST ONE architecture (any false positive)", "value": int(n_flagged_by_at_least_one)},
        {"metric": "benigns flagged by MAJORITY (>5/10) architectures (strong false positives)", "value": int(n_flagged_by_majority)},
        {"metric": "benigns flagged by ALL architectures (irreducible false positives)", "value": int(n_flagged_by_all)},
        # Per-category irreducibility
    ])

    # Per-category breakdown of irreducibly-missed hazards
    if "hazard_category" in hazards.columns:
        cat_breakdown = hazards.groupby("hazard_category").agg(
            n=("true_hazard", "size"),
            n_missed_by_all=("n_archs_flagged", lambda x: int((x == 0).sum())),
            n_caught_by_majority=("n_archs_flagged", lambda x: int((x > n_arch / 2).sum())),
            mean_archs_flagging=("n_archs_flagged", "mean"),
        ).reset_index()
        cat_breakdown["irreducible_miss_rate"] = (cat_breakdown["n_missed_by_all"] / cat_breakdown["n"]).round(3)
        cat_breakdown["majority_catch_rate"] = (cat_breakdown["n_caught_by_majority"] / cat_breakdown["n"]).round(3)
        return wide, summary, cat_breakdown
    return wide, summary, pd.DataFrame()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="realworld_n2000")
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    wide, arch_cols = load_aligned_predictions(args.predictions_dir, args.dataset)
    if wide is None or wide.empty:
        print(f"No predictions found in {args.predictions_dir} for dataset={args.dataset}")
        return 1

    wide, summary, cat_breakdown = analyze_failure_modes(wide, arch_cols)
    n_arch = len(arch_cols)

    wide_path = args.results_dir / "failure_mode_per_message.csv"
    summary_path = args.results_dir / "failure_mode_summary.csv"
    cat_path = args.results_dir / "failure_mode_by_category.csv"
    wide.to_csv(wide_path, index=False)
    summary.to_csv(summary_path, index=False)
    if not cat_breakdown.empty:
        cat_breakdown.to_csv(cat_path, index=False)

    print(f"\n=== Failure Mode Summary (N={n_arch} architectures on {args.dataset}) ===\n")
    print(summary.to_string(index=False))
    if not cat_breakdown.empty:
        print(f"\n=== Per-Category Hazard Catch (sorted by irreducible_miss_rate desc) ===\n")
        print(cat_breakdown.sort_values("irreducible_miss_rate", ascending=False).to_string(index=False))

    print(f"\n  → wrote {wide_path}, {summary_path}, {cat_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
