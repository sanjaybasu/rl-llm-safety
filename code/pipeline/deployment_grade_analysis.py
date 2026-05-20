"""Deployment-grade configuration analysis — what IS achievable?

Clinical-grade autonomous deployment (sens ≥ 0.80 AND spec ≥ 0.80) is not
reached by any evaluated configuration on the Medicaid messaging population.
This analysis instead characterizes what IS achievable under two more
realistic deployment-grade targets:

1. **Sensitivity-floor target (deployment-grade for safety-critical screening)**:
   maximize specificity subject to sensitivity ≥ 0.85. This is the standard
   compromise in clinical screening when miss-cost dominates false-alarm-cost.
   The remaining false positives go to clinician review; the analysis
   characterizes the resulting workload.

2. **Disagreement-stratified clinician triage**:
   for each message, count the number of architectures (out of 10) that
   flagged it. High-agreement decisions (e.g., ≥7 flag OR ≤2 flag) are
   high-confidence and can be acted on autonomously; mid-agreement cases
   (3-6 flag) go to clinician review. Characterize accuracy and review
   workload at each agreement-stratified policy.

3. **Per-hazard-category best architecture**:
   for each hazard category, identify the architecture with the highest
   sensitivity on that category. A category-routed ensemble (route each
   message to the predicted-category's best architecture) is an upper-bound
   on what category-specialization could achieve.

Outputs:
    results/deployment_grade_sens_floor.csv   — configurations meeting sens >= floor
    results/deployment_grade_agreement.csv     — disagreement-stratified accuracy
    results/deployment_grade_per_category.csv  — best architecture per hazard category
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SENS_FLOORS = [0.80, 0.85, 0.90, 0.95]


def metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    pred = pred.astype(bool); true = true.astype(bool)
    tp = int((pred & true).sum()); fp = int((pred & ~true).sum())
    tn = int((~pred & ~true).sum()); fn = int((~pred & true).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sensitivity": round(sens, 4), "specificity": round(spec, 4),
            "ppv": round(ppv, 4), "f1": round(f1, 4)}


def collect_configurations(predictions_dir: Path, results_dir: Path,
                            dataset: str = "realworld_n2000") -> pd.DataFrame:
    """Pool all evaluated configurations (single arch + ensemble + cascade + multi-LLM + RAG)."""
    rows = []

    # 1. Single-architecture default-threshold
    m_path = results_dir / "metrics_canonical.csv"
    if m_path.exists():
        m = pd.read_csv(m_path)
        rw = m[m["dataset"] == dataset]
        for _, r in rw.iterrows():
            rows.append({
                "strategy_type": "single_architecture_default",
                "label": str(r["architecture"]),
                "sensitivity": float(r["sensitivity"]),
                "specificity": float(r["specificity"]),
                "ppv": float(r["ppv"]) if pd.notna(r.get("ppv")) else None,
                "f1": float(r["f1"]) if pd.notna(r.get("f1")) else None,
            })

    # 2. Threshold-optimized single architectures (operating curve sweep)
    curves_path = results_dir / "operating_curves.csv"
    if curves_path.exists():
        cu = pd.read_csv(curves_path)
        # Need n_hazards/n_benigns to derive PPV/F1 — use 165/1835 (canonical)
        n_haz, n_ben = 165, 1835
        for arch in cu["architecture"].unique():
            sub = cu[cu["architecture"] == arch].copy()
            sens = sub["sensitivity"].values
            spec = sub["specificity"].values
            for s, sp in zip(sens, spec):
                tp = s * n_haz; fp = (1 - sp) * n_ben
                ppv = tp / (tp + fp + 1e-9)
                f1 = 2 * tp / (2 * tp + fp + (n_haz - tp) + 1e-9)
                rows.append({
                    "strategy_type": "single_architecture_threshold_sweep",
                    "label": f"{arch}@thresh",
                    "sensitivity": round(float(s), 4),
                    "specificity": round(float(sp), 4),
                    "ppv": round(ppv, 4),
                    "f1": round(f1, 4),
                })

    # 3. Ensembles
    ens_path = results_dir / "ensemble_results.csv"
    if ens_path.exists():
        ens = pd.read_csv(ens_path)
        for _, r in ens.iterrows():
            rows.append({
                "strategy_type": "ensemble",
                "label": str(r["rule"]),
                "sensitivity": float(r["sensitivity"]),
                "specificity": float(r["specificity"]),
                "ppv": float(r["ppv"]) if pd.notna(r.get("ppv")) else None,
                "f1": float(r["f1"]) if pd.notna(r.get("f1")) else None,
            })

    # 4. Cascades
    cas_path = results_dir / "cascade_matrix.csv"
    if cas_path.exists():
        cas = pd.read_csv(cas_path)
        cas = cas[cas["dataset"] == dataset]
        for _, r in cas.iterrows():
            rows.append({
                "strategy_type": "cascade_AND_rule",
                "label": f"{r['stage1']} × {r['stage2']}",
                "sensitivity": float(r["sensitivity"]),
                "specificity": float(r["specificity"]),
                "ppv": float(r["ppv"]) if pd.notna(r.get("ppv")) else None,
                "f1": float(r["f1"]) if pd.notna(r.get("f1")) else None,
            })

    # 5. Multi-LLM consensus
    ml_path = results_dir / "multi_llm_consensus.csv"
    if ml_path.exists():
        ml = pd.read_csv(ml_path)
        for _, r in ml.iterrows():
            rows.append({
                "strategy_type": "multi_LLM_consensus",
                "label": str(r["rule"]),
                "sensitivity": float(r["sensitivity"]),
                "specificity": float(r["specificity"]),
                "ppv": float(r["ppv"]) if pd.notna(r.get("ppv")) else None,
                "f1": float(r["f1"]) if pd.notna(r.get("f1")) else None,
            })

    return pd.DataFrame(rows).drop_duplicates(subset=["strategy_type", "label", "sensitivity", "specificity"])


def sens_floor_winners(df: pd.DataFrame) -> pd.DataFrame:
    """For each sens floor, find max-specificity configuration meeting that floor."""
    rows = []
    for floor in SENS_FLOORS:
        eligible = df[df["sensitivity"] >= floor].copy()
        if eligible.empty:
            rows.append({
                "sens_floor": floor,
                "n_configurations_meeting_floor": 0,
                "max_specificity": None,
                "winning_strategy_type": None,
                "winning_label": None,
                "winning_sensitivity": None,
                "winning_f1": None,
            })
        else:
            eligible = eligible.sort_values("specificity", ascending=False)
            top = eligible.iloc[0]
            rows.append({
                "sens_floor": floor,
                "n_configurations_meeting_floor": len(eligible),
                "max_specificity": float(top["specificity"]),
                "winning_strategy_type": top["strategy_type"],
                "winning_label": top["label"],
                "winning_sensitivity": float(top["sensitivity"]),
                "winning_f1": float(top["f1"]) if pd.notna(top.get("f1")) else None,
            })
    return pd.DataFrame(rows)


def disagreement_stratified(failure_mode_path: Path) -> pd.DataFrame:
    """For each agreement-strata, compute accuracy on the subset of messages in that stratum.

    Strata defined by 'n_archs_flagged' (how many of 10 architectures voted hazard).
    Policies:
      - 0 or 1 flagging: confidently benign (autonomous benign)
      - 2-3 flagging: low-confidence; route to clinician review
      - 4-6 flagging: medium-confidence; route to clinician review
      - 7-10 flagging: confidently hazard (autonomous escalation)
    """
    if not failure_mode_path.exists():
        return pd.DataFrame()
    wide = pd.read_csv(failure_mode_path)
    n_arch_cols = [c for c in wide.columns if c.endswith("_pred")]
    n_arch = len(n_arch_cols)

    strata = [
        ("0-1 flagging (autonomous benign)", lambda x: x <= 1, "no_action"),
        ("2-3 flagging (clinician review)", lambda x: (x >= 2) & (x <= 3), "clinician_review"),
        ("4-6 flagging (clinician review)", lambda x: (x >= 4) & (x <= 6), "clinician_review"),
        (f"7-{n_arch} flagging (autonomous escalation)", lambda x: x >= 7, "escalate"),
    ]
    rows = []
    for label, predicate, policy in strata:
        mask = predicate(wide["n_archs_flagged"])
        subset = wide[mask]
        n = len(subset)
        if n == 0:
            rows.append({"stratum": label, "policy": policy, "n_messages": 0,
                          "n_hazards": 0, "n_benigns": 0,
                          "hazard_prevalence_in_stratum": None,
                          "share_of_total_messages": 0.0})
            continue
        n_haz = int((subset["true_hazard"] == 1).sum())
        n_ben = n - n_haz
        rows.append({
            "stratum": label,
            "policy": policy,
            "n_messages": n,
            "n_hazards": n_haz,
            "n_benigns": n_ben,
            "hazard_prevalence_in_stratum": round(n_haz / n, 4),
            "share_of_total_messages": round(n / len(wide), 4),
        })
    return pd.DataFrame(rows)


def per_category_best(predictions_dir: Path, results_dir: Path) -> pd.DataFrame:
    """For each hazard category, find the architecture with highest sensitivity."""
    cs_path = results_dir / "category_stratification.csv"
    if not cs_path.exists():
        return pd.DataFrame()
    cs = pd.read_csv(cs_path)
    rows = []
    for cat in cs["hazard_category"].unique():
        sub = cs[cs["hazard_category"] == cat].copy().sort_values("sensitivity", ascending=False)
        top = sub.iloc[0]
        rows.append({
            "hazard_category": cat,
            "n_hazards": int(top["n_hazards"]),
            "best_architecture": top["architecture"],
            "best_sensitivity": float(top["sensitivity"]),
            "clinical_grade_reached_at_sens_0.80": bool(top["sensitivity"] >= 0.80),
        })
    df = pd.DataFrame(rows).sort_values("n_hazards", ascending=False)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sensitivity-floor configurations
    pool = collect_configurations(args.predictions_dir, args.results_dir)
    floor_df = sens_floor_winners(pool)
    floor_path = args.results_dir / "deployment_grade_sens_floor.csv"
    floor_df.to_csv(floor_path, index=False)
    print("\n=== Sensitivity-floor winners: max specificity subject to sens >= floor ===\n")
    print(floor_df.to_string(index=False))

    # 2. Disagreement-stratified
    fm_path = args.results_dir / "failure_mode_per_message.csv"
    agree_df = disagreement_stratified(fm_path)
    agree_path = args.results_dir / "deployment_grade_agreement.csv"
    agree_df.to_csv(agree_path, index=False)
    print("\n=== Disagreement-stratified workload + accuracy ===\n")
    print(agree_df.to_string(index=False))

    # 3. Per-category best
    pc_df = per_category_best(args.predictions_dir, args.results_dir)
    pc_path = args.results_dir / "deployment_grade_per_category.csv"
    pc_df.to_csv(pc_path, index=False)
    print("\n=== Per-hazard-category best architecture (upper bound on category specialization) ===\n")
    print(pc_df.to_string(index=False))

    print(f"\n  → wrote {floor_path}, {agree_path}, {pc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
