"""Cascade architecture analysis — deployment-relevant Pareto frontier.

A single architecture cannot deliver both high sensitivity and high specificity
on this real-world Medicaid messaging population. Two-stage cascades (a
high-recall screen forwarding positives to a high-precision confirmation)
are the standard deployment pattern in clinical CAD/CDS and the natural
solution when no single model gives a useful operating point.

Cascade math (AND-rule series cascade):
    pred_cascade = pred_stage1 AND pred_stage2
For each (stage1, stage2) architecture pair on a given dataset:
    TP_c = #(true=1 & stage1=1 & stage2=1)
    FP_c = #(true=0 & stage1=1 & stage2=1)
    TN_c = #(true=0 & (stage1=0 | stage2=0))
    FN_c = #(true=1 & (stage1=0 | stage2=0))
    sens_c = TP_c / (TP_c + FN_c)
    spec_c = TN_c / (TN_c + FP_c)

The cascade can only DECREASE sensitivity vs either stage alone (sens_c ≤
min(sens_1, sens_2)) and can only INCREASE specificity vs either stage alone
(spec_c ≥ max(spec_1, spec_2)). The asymmetric trade is the value: pair a
high-sens stage 1 with a high-spec stage 2 and the cascade's spec lift comes
mostly from the stage 1 false alarms that stage 2 correctly rejects.

This computes all (stage1, stage2) pairs from the per-message CSVs on the
realworld_n2000 test set and outputs:
    results/cascade_matrix.csv  — per-pair sens, spec, F1, MCC, FN/1000 + 95% Wilson CIs

Inputs: predictions/canonical_filtered/{architecture}_{run_id}.csv (9 architectures × 2 datasets)
"""
from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "audit"))
import metrics as M


def load_aligned_predictions(predictions_dir: Path, dataset: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load all per-architecture pred_hazard vectors aligned by message_id.

    Returns ({arch: pred_hazard_vec}, true_hazard_vec).
    """
    files = sorted(predictions_dir.glob("*.csv"))
    arch_preds: dict[str, np.ndarray] = {}
    true_hazard: np.ndarray | None = None
    message_id_order: list[str] | None = None
    for csv_path in files:
        df = pd.read_csv(csv_path)
        if "dataset" not in df.columns:
            continue
        sub = df[df["dataset"] == dataset].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("message_id").reset_index(drop=True)
        arch = sub["architecture"].iloc[0]
        if message_id_order is None:
            message_id_order = sub["message_id"].astype(str).tolist()
            true_hazard = sub["true_hazard"].astype(int).values
        else:
            sub = sub.set_index("message_id").loc[[m for m in message_id_order]].reset_index()
        arch_preds[arch] = sub["pred_hazard"].astype(int).values
    return arch_preds, true_hazard


def cascade_metrics(stage1_pred: np.ndarray, stage2_pred: np.ndarray, true: np.ndarray) -> dict:
    """Series-cascade AND-rule metrics with 95% Wilson CIs."""
    cascade_pred = (stage1_pred == 1) & (stage2_pred == 1)
    tp = int(((true == 1) & cascade_pred).sum())
    fp = int(((true == 0) & cascade_pred).sum())
    tn = int(((true == 0) & ~cascade_pred).sum())
    fn = int(((true == 1) & ~cascade_pred).sum())

    n_hazards = tp + fn
    n_benigns = tn + fp
    sens = tp / n_hazards if n_hazards else float("nan")
    spec = tn / n_benigns if n_benigns else float("nan")
    sens_lo, sens_hi = M.wilson_ci(tp, n_hazards)
    spec_lo, spec_hi = M.wilson_ci(tn, n_benigns)
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else float("nan")
    fn_per_1000 = 1000.0 * fn / (n_hazards + n_benigns) if (n_hazards + n_benigns) else float("nan")
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": round(sens, 4),
        "sensitivity_ci_lo": round(sens_lo, 4),
        "sensitivity_ci_hi": round(sens_hi, 4),
        "specificity": round(spec, 4),
        "specificity_ci_lo": round(spec_lo, 4),
        "specificity_ci_hi": round(spec_hi, 4),
        "ppv": round(ppv, 4) if ppv == ppv else None,
        "npv": round(npv, 4) if npv == npv else None,
        "f1": round(f1, 4) if f1 == f1 else None,
        "mcc": round(mcc, 4) if mcc == mcc else None,
        "fn_per_1000": round(fn_per_1000, 2),
    }


def compute_cascade_matrix(predictions_dir: Path, dataset: str) -> pd.DataFrame:
    """Full (stage1, stage2) cascade matrix on the given dataset.

    The AND-rule cascade is symmetric (commutative). We compute both orderings
    so the table reads naturally for either direction of deployment framing.
    """
    arch_preds, true = load_aligned_predictions(predictions_dir, dataset)
    rows = []
    for stage1, stage2 in product(sorted(arch_preds), sorted(arch_preds)):
        if stage1 == stage2:
            continue
        m = cascade_metrics(arch_preds[stage1], arch_preds[stage2], true)
        m["stage1"] = stage1
        m["stage2"] = stage2
        m["dataset"] = dataset
        rows.append(m)
    cols = ["stage1", "stage2", "dataset",
            "sensitivity", "sensitivity_ci_lo", "sensitivity_ci_hi",
            "specificity", "specificity_ci_lo", "specificity_ci_hi",
            "ppv", "npv", "f1", "mcc", "fn_per_1000",
            "tp", "fp", "tn", "fn"]
    df = pd.DataFrame(rows)[cols]
    return df


def pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the (stage1, stage2) rows on the sens/spec Pareto frontier.

    A pair is on the frontier if no other pair has both higher sens AND higher spec.
    """
    df = df.copy()
    keep = []
    for _, row in df.iterrows():
        dominated = (
            (df["sensitivity"] >= row["sensitivity"]) &
            (df["specificity"] >= row["specificity"]) &
            ((df["sensitivity"] > row["sensitivity"]) | (df["specificity"] > row["specificity"]))
        ).any()
        keep.append(not dominated)
    return df[pd.Series(keep, index=df.index)].sort_values("sensitivity", ascending=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["realworld_n2000", "physician_n41"])
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    full = []
    for ds in args.datasets:
        m = compute_cascade_matrix(args.predictions_dir, ds)
        full.append(m)
        print(f"  {ds}: {len(m)} cascade pairs computed")
    df = pd.concat(full, ignore_index=True)
    out_path = args.results_dir / "cascade_matrix.csv"
    df.to_csv(out_path, index=False)
    print(f"  → wrote {out_path} ({len(df)} rows)")

    # Pareto frontier on real-world
    rw = df[df["dataset"] == "realworld_n2000"]
    pf = pareto_frontier(rw)
    pf_path = args.results_dir / "cascade_pareto.csv"
    pf.to_csv(pf_path, index=False)
    print(f"  → wrote {pf_path} ({len(pf)} Pareto-frontier pairs)")

    print("\nReal-world cascade Pareto frontier (sens descending):")
    show_cols = ["stage1", "stage2", "sensitivity", "specificity", "ppv", "f1", "mcc"]
    print(pf[show_cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
