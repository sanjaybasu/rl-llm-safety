"""Hazard-category stratification analysis.

For each hazard category present in the real-world test set with at least 3
adjudicated hazards, computes per-architecture sensitivity restricted to that
category. Surfaces which categories are best and worst detected by each
architecture — a likely reviewer ask and an actionable input to the research
agenda for closing the gap.

Output: results/category_stratification.csv
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


MIN_CATEGORY_N = 3


def analyze(predictions_dir: Path, dataset: str = "realworld_n2000") -> pd.DataFrame:
    files = sorted(Path(predictions_dir).glob("*.csv"))
    # First pass: figure out category distribution
    cat_counts: dict[str, int] = defaultdict(int)
    for csv in files:
        df = pd.read_csv(csv)
        sub = df[(df["dataset"] == dataset) & (df["true_hazard"] == 1)]
        if "hazard_category" not in sub.columns:
            continue
        # Count categories once per architecture; they're identical across architectures
        for cat in sub["hazard_category"].dropna().unique():
            cat_counts[str(cat)] += (sub["hazard_category"] == cat).sum()
        break  # all architectures share the same test set; one pass suffices

    # cat_counts accumulated from ONE architecture file (the break exits after first);
    # values are raw per-test-set hazard counts, identical across architectures.
    qualifying = sorted([c for c, n in cat_counts.items() if n >= MIN_CATEGORY_N])

    rows = []
    for csv in files:
        df = pd.read_csv(csv)
        sub = df[df["dataset"] == dataset].copy()
        if sub.empty or "hazard_category" not in sub.columns:
            continue
        arch = sub["architecture"].iloc[0]
        for cat in qualifying:
            cat_sub = sub[sub["hazard_category"] == cat]
            haz = cat_sub[cat_sub["true_hazard"] == 1]
            n_haz = len(haz)
            if n_haz < MIN_CATEGORY_N:
                continue
            tp = int((haz["pred_hazard"] == 1).sum())
            sens = tp / n_haz
            rows.append({
                "architecture": arch,
                "hazard_category": cat,
                "n_hazards": n_haz,
                "tp": tp,
                "fn": n_haz - tp,
                "sensitivity": round(sens, 4),
            })

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    df = analyze(args.predictions_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.results_dir / "category_stratification.csv"
    df.to_csv(out_path, index=False)
    print(f"  → wrote {out_path} ({len(df)} rows)")
    # Pivot view
    if not df.empty:
        pivot = df.pivot_table(index="hazard_category", columns="architecture",
                                values="sensitivity", aggfunc="first")
        print("\nPer-category × architecture sensitivity (real-world test set):")
        print(pivot.round(3).to_string())
        cat_avg = df.groupby("hazard_category")["sensitivity"].mean().sort_values()
        print("\nMean sensitivity across architectures, by category (sorted):")
        print(cat_avg.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
