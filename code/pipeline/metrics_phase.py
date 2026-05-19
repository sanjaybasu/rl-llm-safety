"""Metrics phase — read canonical per-message predictions, write canonical
metrics CSVs that every table and figure derives from.

Outputs:
    results/metrics_canonical.csv     (per architecture × dataset: sens, spec, F1, MCC, AUROC, PPV, NPV, FN/1000, CIs)
    results/mcnemar_matrix.csv        (per (arch_a, arch_b) pair: chi2, p_raw, sig_hochberg)
    results/delta_bootstrap_canonical.csv  (per architecture: Δ sens/spec/F1/AUROC physician → real-world with bootstrap CIs)
    results/operating_curves.csv      (per architecture: full ROC curve for monotonicity assertion)
    results/thresholds.json           (per architecture: the calibrated threshold used at inference)

All inputs are per-message prediction CSVs in predictions/canonical/.
This module never reads raw test data — only the prediction files.
This guarantees Tables 2 and 3 derive from the same prediction set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Import canonical metrics module
sys.path.insert(0, str(Path(__file__).parent.parent / "audit"))
import metrics as M


def load_all_predictions(
    predictions_dir: Path, run_id: str | None = None
) -> dict[str, pd.DataFrame]:
    """Load every per-architecture CSV in predictions/canonical/.

    If `run_id` is given, only files containing that run_id are loaded. This
    ensures that when multiple Modal runs have written to the same volume,
    metrics are computed from a single canonical run.

    If `run_id` is None, the most recently-modified CSV per architecture is used.

    Returns dict mapping architecture name to DataFrame.
    """
    out: dict[str, pd.DataFrame] = {}
    # Sort by mtime ascending so later files (more recent) overwrite in the dict
    files = sorted(Path(predictions_dir).glob("*.csv"), key=lambda p: p.stat().st_mtime)
    for csv_path in files:
        if run_id is not None and run_id not in csv_path.name:
            continue
        df = pd.read_csv(csv_path)
        if df.empty or "architecture" not in df.columns:
            continue
        arch = df["architecture"].iloc[0]
        out[arch] = df
    return out


def compute_metrics_table(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per (architecture, dataset). All metrics with CIs."""
    rows = []
    for arch, df in predictions.items():
        for dataset, sub in df.groupby("dataset"):
            # Coerce pred_proba to float (may be missing for LLMs)
            sub = sub.copy()
            sub["pred_proba"] = pd.to_numeric(sub["pred_proba"], errors="coerce")
            metrics_dict = M.compute_full_metrics(sub)
            metrics_dict["architecture"] = arch
            metrics_dict["dataset"] = dataset
            rows.append(metrics_dict)
    cols = ["architecture", "dataset"] + [
        c for c in rows[0].keys() if c not in {"architecture", "dataset"}
    ] if rows else []
    return pd.DataFrame(rows)[cols] if rows else pd.DataFrame()


def compute_mcnemar_matrix(predictions: dict[str, pd.DataFrame], dataset: str = "realworld_n2000") -> pd.DataFrame:
    """Full pairwise McNemar matrix on the specified dataset."""
    # Index every architecture's predictions by message_id for alignment
    aligned: dict[str, np.ndarray] = {}
    true_hazard: Optional[np.ndarray] = None
    message_id_order: Optional[list[str]] = None

    for arch, df in predictions.items():
        sub = df[df["dataset"] == dataset].sort_values("message_id")
        if sub.empty:
            continue
        if message_id_order is None:
            message_id_order = sub["message_id"].tolist()
            true_hazard = sub["true_hazard"].values
        else:
            # Verify the same message_ids in the same order
            if sub["message_id"].tolist() != message_id_order:
                # Re-align via message_id index
                sub = sub.set_index("message_id").loc[message_id_order].reset_index()
        aligned[arch] = sub["pred_hazard"].values

    if not aligned or true_hazard is None:
        return pd.DataFrame()

    return M.mcnemar_matrix(aligned, true_hazard, alpha=0.05)


def compute_delta_bootstrap(
    predictions: dict[str, pd.DataFrame],
    seed: int = 42,
    n_iter: int = 10_000,
) -> pd.DataFrame:
    """Δ sensitivity (physician → real-world) per architecture with bootstrap CI."""
    rows = []
    for arch, df in predictions.items():
        if "physician_n41" not in df["dataset"].unique() or "realworld_n2000" not in df["dataset"].unique():
            continue
        phys = df[df["dataset"] == "physician_n41"]
        real = df[df["dataset"] == "realworld_n2000"]
        phys_pred = phys["pred_hazard"].values
        phys_true = phys["true_hazard"].values
        real_pred = real["pred_hazard"].values
        real_true = real["true_hazard"].values

        # Point estimates
        phys_sens = M.sensitivity(phys_pred, phys_true)
        phys_spec = M.specificity(phys_pred, phys_true)
        real_sens = M.sensitivity(real_pred, real_true)
        real_spec = M.specificity(real_pred, real_true)
        delta_sens = real_sens - phys_sens
        delta_spec = real_spec - phys_spec

        # Parametric bootstrap of the difference
        phys_n_pos = int((phys_true == 1).sum())
        real_n_pos = int((real_true == 1).sum())
        phys_n_neg = int((phys_true == 0).sum())
        real_n_neg = int((real_true == 0).sum())

        rng = np.random.default_rng(seed)
        delta_sens_boot = np.empty(n_iter)
        delta_spec_boot = np.empty(n_iter)
        for i in range(n_iter):
            ps = rng.binomial(phys_n_pos, phys_sens) / phys_n_pos if phys_n_pos else float("nan")
            rs = rng.binomial(real_n_pos, real_sens) / real_n_pos if real_n_pos else float("nan")
            psp = rng.binomial(phys_n_neg, phys_spec) / phys_n_neg if phys_n_neg else float("nan")
            rsp = rng.binomial(real_n_neg, real_spec) / real_n_neg if real_n_neg else float("nan")
            delta_sens_boot[i] = (rs - ps) * 100
            delta_spec_boot[i] = (rsp - psp) * 100

        rows.append({
            "architecture": arch,
            "phys_sens": round(phys_sens, 4),
            "real_sens": round(real_sens, 4),
            "delta_sens_pp": round(delta_sens * 100, 2),
            "delta_sens_ci_lo": round(np.percentile(delta_sens_boot, 2.5), 2),
            "delta_sens_ci_hi": round(np.percentile(delta_sens_boot, 97.5), 2),
            "phys_spec": round(phys_spec, 4),
            "real_spec": round(real_spec, 4),
            "delta_spec_pp": round(delta_spec * 100, 2),
            "delta_spec_ci_lo": round(np.percentile(delta_spec_boot, 2.5), 2),
            "delta_spec_ci_hi": round(np.percentile(delta_spec_boot, 97.5), 2),
        })

    return pd.DataFrame(rows)


def compute_operating_curves(
    predictions: dict[str, pd.DataFrame], dataset: str = "realworld_n2000"
) -> pd.DataFrame:
    """Full ROC curve per architecture (where calibrated probabilities exist).

    This is the foundation of Table 3 and the ROC monotonicity assertion.
    Tables 2 and 3 both threshold this same curve at different points.
    """
    rows = []
    for arch, df in predictions.items():
        sub = df[df["dataset"] == dataset].copy()
        sub["pred_proba"] = pd.to_numeric(sub["pred_proba"], errors="coerce")
        if sub["pred_proba"].isna().all():
            continue
        proba = sub["pred_proba"].values
        true = sub["true_hazard"].values
        curve = M.operating_curve(proba, true)
        curve["architecture"] = arch
        rows.append(curve)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def extract_thresholds(predictions: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Pull the calibrated threshold from each architecture's prediction file."""
    out: dict[str, float] = {}
    for arch, df in predictions.items():
        if "threshold_used" in df.columns:
            out[arch] = float(df["threshold_used"].iloc[0])
    return out


def run_metrics_phase(
    predictions_dir: Path,
    results_dir: Path,
    dataset_for_mcnemar: str = "realworld_n2000",
    seed: int = 42,
    run_id: str | None = None,
) -> dict[str, Path]:
    """Read all per-architecture predictions; write all canonical metrics CSVs."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[metrics phase] Loading predictions from {predictions_dir}")
    if run_id:
        print(f"  Filtering to run_id={run_id}")
    predictions = load_all_predictions(Path(predictions_dir), run_id=run_id)
    print(f"  Loaded {len(predictions)} architectures: {sorted(predictions.keys())}")
    if not predictions:
        raise RuntimeError(
            f"No prediction files found in {predictions_dir}. "
            f"Run Phase 5 (local_inference) and Phase 6 (llm_inference) first."
        )

    out_paths: dict[str, Path] = {}

    print("\n[metrics phase] Computing per-architecture metrics table...")
    metrics_df = compute_metrics_table(predictions)
    out_paths["metrics_canonical"] = results_dir / "metrics_canonical.csv"
    metrics_df.to_csv(out_paths["metrics_canonical"], index=False)
    print(f"  → {out_paths['metrics_canonical']} ({len(metrics_df)} rows)")

    print("\n[metrics phase] Computing pairwise McNemar matrix...")
    mcnemar_df = compute_mcnemar_matrix(predictions, dataset=dataset_for_mcnemar)
    out_paths["mcnemar_matrix"] = results_dir / "mcnemar_matrix.csv"
    mcnemar_df.to_csv(out_paths["mcnemar_matrix"], index=False)
    print(f"  → {out_paths['mcnemar_matrix']} ({len(mcnemar_df)} pairwise comparisons)")

    print("\n[metrics phase] Computing Δ bootstrap (physician → real-world)...")
    delta_df = compute_delta_bootstrap(predictions, seed=seed)
    out_paths["delta_bootstrap_canonical"] = results_dir / "delta_bootstrap_canonical.csv"
    delta_df.to_csv(out_paths["delta_bootstrap_canonical"], index=False)
    print(f"  → {out_paths['delta_bootstrap_canonical']} ({len(delta_df)} rows)")

    print("\n[metrics phase] Computing operating curves...")
    curves_df = compute_operating_curves(predictions, dataset=dataset_for_mcnemar)
    out_paths["operating_curves"] = results_dir / "operating_curves.csv"
    curves_df.to_csv(out_paths["operating_curves"], index=False)
    print(f"  → {out_paths['operating_curves']} ({len(curves_df)} curve points)")

    print("\n[metrics phase] Extracting calibrated thresholds...")
    thresholds = extract_thresholds(predictions)
    out_paths["thresholds"] = results_dir / "thresholds.json"
    with open(out_paths["thresholds"], "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"  → {out_paths['thresholds']}")

    return out_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default="realworld_n2000")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", type=str, default=None,
                        help="If set, only files containing this run_id are used.")
    args = parser.parse_args()
    run_metrics_phase(args.predictions_dir, args.results_dir, args.dataset, args.seed,
                      run_id=args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
