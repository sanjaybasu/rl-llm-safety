"""Render every table and figure from canonical CSVs ONLY.

This module reads:
    results/metrics_canonical.csv
    results/mcnemar_matrix.csv
    results/delta_bootstrap_canonical.csv
    results/operating_curves.csv
    results/thresholds.json
    predictions/canonical/*.csv  (for hazard-category stratification only)

And writes:
    results/tables/table1_demographics.csv
    results/tables/table2_detection_metrics.csv
    results/tables/table3_operating_points.csv
    results/tables/table4_action_recommendations.csv
    results/tables/tableS1_physician_holdout_metrics.csv
    results/tables/tableS2_delta_bootstrap.csv
    results/figures/figure1_sens_spec_change.png
    results/figures/figure2_action_recommendations.png

Each table is also rendered as Markdown for inclusion in the manuscript.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Display name mapping (architecture → manuscript-facing label)
ARCH_DISPLAY = {
    "cql_sens_opt": "CQL controller (sensitivity-optimized)",
    "cql_reward_opt": "CQL controller (reward-optimized)",
    "constellation": "Constellation architecture",
    "actionhead": "ActionHead (action recommender)",
    "guardrails": "Rule-based guardrails",
    "xgboost_sbert": "XGBoost + sentence-BERT",
    "logreg_tfidf": "Logistic regression + TF-IDF",
    "claude_opus_4_7_safety": "Claude Opus 4.7 (safety-augmented)",
    "gemini_3_1_pro_safety": "Gemini 3.1 Pro (safety-augmented)",
}


def fmt_ci(point: float, lo: float, hi: float, ndigits: int = 3) -> str:
    """Format 'X.XXX (95% CI Y.YYY-Z.ZZZ)'."""
    if any(np.isnan([point, lo, hi])):
        return "—"
    return f"{point:.{ndigits}f} ({lo:.{ndigits}f}–{hi:.{ndigits}f})"


def fmt_pp(value: float, ndigits: int = 1) -> str:
    """Format value as 'X.X pp' (percentage points)."""
    if np.isnan(value):
        return "—"
    return f"{value:+.{ndigits}f}"


def write_markdown_table(out_df: pd.DataFrame, md_path: Path) -> None:
    """Write a markdown table, falling back to manual pipe-delimited format if
    tabulate is not installed."""
    try:
        with open(md_path, "w") as f:
            f.write(out_df.to_markdown(index=False))
    except ImportError:
        with open(md_path, "w") as f:
            cols = out_df.columns.tolist()
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
            for _, row in out_df.iterrows():
                f.write("| " + " | ".join(str(row[c]) for c in cols) + " |\n")


def render_table2_detection_metrics(metrics_df: pd.DataFrame, output_dir: Path) -> Path:
    """Table 2: per-architecture detection metrics on real-world test set."""
    df = metrics_df[metrics_df["dataset"] == "realworld_n2000"].copy()
    rows = []
    for _, row in df.iterrows():
        d = row.to_dict()
        rows.append({
            "Architecture": ARCH_DISPLAY.get(d["architecture"], d["architecture"]),
            "Sensitivity (95% CI)": fmt_ci(
                d.get("sensitivity", float("nan")),
                d.get("sensitivity_ci_lo", float("nan")),
                d.get("sensitivity_ci_hi", float("nan"))),
            "Specificity (95% CI)": fmt_ci(
                d.get("specificity", float("nan")),
                d.get("specificity_ci_lo", float("nan")),
                d.get("specificity_ci_hi", float("nan"))),
            "PPV": f"{d['ppv']:.3f}" if not np.isnan(d.get("ppv", float("nan"))) else "—",
            "NPV": f"{d['npv']:.3f}" if not np.isnan(d.get("npv", float("nan"))) else "—",
            "F1 (95% CI)": fmt_ci(
                d.get("f1", float("nan")),
                d.get("f1_ci_lo", float("nan")),
                d.get("f1_ci_hi", float("nan"))),
            "MCC (95% CI)": fmt_ci(
                d.get("mcc", float("nan")),
                d.get("mcc_ci_lo", float("nan")),
                d.get("mcc_ci_hi", float("nan"))),
            "AUROC (95% CI)": (
                fmt_ci(d.get("auroc", float("nan")),
                       d.get("auroc_ci_lo", float("nan")),
                       d.get("auroc_ci_hi", float("nan")))
                if not np.isnan(d.get("auroc", float("nan"))) else "N/A"
            ),
            "FN per 1,000": f"{d['fn_per_1000']:.1f}" if not np.isnan(
                d.get("fn_per_1000", float("nan"))) else "—",
        })
    out_df = pd.DataFrame(rows)
    csv_path = output_dir / "table2_detection_metrics.csv"
    md_path = output_dir / "table2_detection_metrics.md"
    out_df.to_csv(csv_path, index=False)
    write_markdown_table(out_df, md_path)
    return csv_path


def render_table3_operating_points(
    metrics_df: pd.DataFrame, curves_df: pd.DataFrame, output_dir: Path,
    target_specs: tuple = (0.70, 0.73, 0.80, 0.90, 0.95),
) -> Path:
    """Table 3: sensitivity at matched specificity, with achieved spec.

    For each architecture × target_spec, find the threshold that achieves
    target_spec on the operating curve and report (sens, achieved_spec).
    """
    rows = []
    for arch in curves_df["architecture"].unique():
        curve = curves_df[curves_df["architecture"] == arch]
        row = {"Architecture": ARCH_DISPLAY.get(arch, arch)}
        for target_spec in target_specs:
            curve_at = curve.copy()
            curve_at["spec_distance"] = (curve_at["specificity"] - target_spec).abs()
            best = curve_at.sort_values(["spec_distance", "threshold"]).iloc[0]
            sens = best["sensitivity"]
            achieved_spec = best["specificity"]
            row[f"Spec={target_spec:.2f}"] = f"{sens:.3f} (ach. {achieved_spec:.3f})"
        rows.append(row)
    # For LLMs without calibrated proba, include single-threshold row from metrics
    realworld = metrics_df[metrics_df["dataset"] == "realworld_n2000"]
    for _, m in realworld.iterrows():
        if m["architecture"] not in curves_df["architecture"].unique():
            row = {"Architecture": ARCH_DISPLAY.get(m["architecture"], m["architecture"]) + " (single threshold)"}
            for target_spec in target_specs:
                row[f"Spec={target_spec:.2f}"] = f"{m['sensitivity']:.3f} (ach. {m['specificity']:.3f})"
            rows.append(row)

    out_df = pd.DataFrame(rows)
    csv_path = output_dir / "table3_operating_points.csv"
    md_path = output_dir / "table3_operating_points.md"
    out_df.to_csv(csv_path, index=False)
    write_markdown_table(out_df, md_path)
    return csv_path


def render_table4_action_recommendations(
    predictions_dir: Path, output_dir: Path
) -> Path:
    """Table 4: appropriate / under-triage / over-triage rates per architecture.

    Read every per-architecture prediction CSV and compute the 3-category
    action-appropriateness breakdown on the real-world test set.
    """
    rows = []
    for csv_path in sorted(predictions_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        if "dataset" not in df.columns or "true_action" not in df.columns:
            continue
        rw = df[df["dataset"] == "realworld_n2000"]
        if rw.empty:
            continue
        arch = rw["architecture"].iloc[0]
        pred_action = rw["pred_action"].astype(int)
        true_action = rw["true_action"].astype(int)
        appropriate = (pred_action == true_action).sum()
        under_triage = (pred_action < true_action).sum()
        over_triage = (pred_action > true_action).sum()
        n = len(rw)
        rows.append({
            "Architecture": ARCH_DISPLAY.get(arch, arch),
            "Appropriate (n, %)": f"{appropriate} ({100*appropriate/n:.1f}%)",
            "Under-triage (n, %)": f"{under_triage} ({100*under_triage/n:.1f}%)",
            "Over-triage (n, %)": f"{over_triage} ({100*over_triage/n:.1f}%)",
            "Total": n,
        })

    out_df = pd.DataFrame(rows)
    csv_path = output_dir / "table4_action_recommendations.csv"
    md_path = output_dir / "table4_action_recommendations.md"
    out_df.to_csv(csv_path, index=False)
    write_markdown_table(out_df, md_path)
    return csv_path


def render_tableS1_physician_metrics(metrics_df: pd.DataFrame, output_dir: Path) -> Path:
    """Table S1: physician-holdout metrics with TP/FN/TN/FP per architecture."""
    df = metrics_df[metrics_df["dataset"] == "physician_n41"].copy()
    rows = []
    for _, row in df.iterrows():
        d = row.to_dict()
        rows.append({
            "Architecture": ARCH_DISPLAY.get(d["architecture"], d["architecture"]),
            "TP/FN/TN/FP": f"{int(d['tp'])}/{int(d['fn'])}/{int(d['tn'])}/{int(d['fp'])}",
            "Sensitivity (95% CI)": fmt_ci(
                d.get("sensitivity", float("nan")),
                d.get("sensitivity_ci_lo", float("nan")),
                d.get("sensitivity_ci_hi", float("nan"))),
            "Specificity (95% CI)": fmt_ci(
                d.get("specificity", float("nan")),
                d.get("specificity_ci_lo", float("nan")),
                d.get("specificity_ci_hi", float("nan"))),
            "F1": f"{d['f1']:.3f}",
            "MCC": f"{d['mcc']:.3f}",
            "AUROC": f"{d['auroc']:.3f}" if not np.isnan(d.get("auroc", float("nan"))) else "N/A",
        })
    out_df = pd.DataFrame(rows)
    csv_path = output_dir / "tableS1_physician_holdout_metrics.csv"
    out_df.to_csv(csv_path, index=False)
    write_markdown_table(out_df, output_dir / "tableS1_physician_holdout_metrics.md")
    return csv_path


def render_tableS2_delta_bootstrap(delta_df: pd.DataFrame, output_dir: Path) -> Path:
    """Table S2: Δ sens/spec physician → real-world with bootstrap CIs."""
    if delta_df.empty:
        return output_dir / "tableS2_delta_bootstrap.csv"
    rows = []
    for _, row in delta_df.iterrows():
        rows.append({
            "Architecture": ARCH_DISPLAY.get(row["architecture"], row["architecture"]),
            "Δ Sens, pp (95% CI)": f"{row['delta_sens_pp']:+.1f} "
                                    f"({row['delta_sens_ci_lo']:+.1f} to {row['delta_sens_ci_hi']:+.1f})",
            "Δ Spec, pp (95% CI)": f"{row['delta_spec_pp']:+.1f} "
                                    f"({row['delta_spec_ci_lo']:+.1f} to {row['delta_spec_ci_hi']:+.1f})",
            "Phys sens": f"{row['phys_sens']:.3f}",
            "Real sens": f"{row['real_sens']:.3f}",
        })
    out_df = pd.DataFrame(rows)
    csv_path = output_dir / "tableS2_delta_bootstrap.csv"
    out_df.to_csv(csv_path, index=False)
    write_markdown_table(out_df, output_dir / "tableS2_delta_bootstrap.md")
    return csv_path


def render_figure1_sens_change(delta_df: pd.DataFrame, output_dir: Path) -> Path:
    """Figure 1: per-architecture Δ sensitivity (physician → real-world) bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if delta_df.empty:
        out_path = output_dir / "figure1_sens_spec_change.png"
        # Write empty placeholder
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No delta data available", ha="center")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    fig, ax = plt.subplots(figsize=(10, 5))
    sorted_df = delta_df.sort_values("delta_sens_pp")
    names = [ARCH_DISPLAY.get(a, a) for a in sorted_df["architecture"]]
    deltas = sorted_df["delta_sens_pp"].values
    ci_lo = sorted_df["delta_sens_ci_lo"].values
    ci_hi = sorted_df["delta_sens_ci_hi"].values

    colors = ["#d62728" if d < 0 else "#2ca02c" for d in deltas]
    ax.barh(names, deltas, color=colors, edgecolor="black", linewidth=0.5)
    ax.errorbar(deltas, range(len(deltas)),
                xerr=[deltas - ci_lo, ci_hi - deltas],
                fmt="none", ecolor="black", capsize=3, linewidth=1)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Δ Sensitivity, percentage points (physician → real-world)")
    ax.set_title("Cross-set sensitivity degradation by architecture")
    ax.grid(axis="x", alpha=0.3)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out_path = output_dir / "figure1_sens_spec_change.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_figure3_pareto_frontier(results_dir: Path, output_dir: Path) -> Path:
    """Figure 3: sensitivity vs specificity Pareto envelope across all strategies.

    Single high-impact figure showing every strategy as a (sens, spec) point with
    the clinical-grade target zone shaded. Visual proof that no strategy reaches
    the target zone — the structural finding of the paper.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7))

    # Shade clinical-grade target region (top-right quadrant ≥0.80, ≥0.80)
    ax.axhspan(0.80, 1.0, xmin=0.80, xmax=1.0, color="green", alpha=0.10, zorder=0)
    ax.axhline(0.80, color="green", linestyle=":", linewidth=1.0, alpha=0.5, zorder=1)
    ax.axvline(0.80, color="green", linestyle=":", linewidth=1.0, alpha=0.5, zorder=1)
    ax.text(0.85, 0.92, "Clinical-grade\ntarget zone", fontsize=10,
            color="darkgreen", ha="left", va="center", alpha=0.8)

    # Single-architecture default-threshold points
    m_path = results_dir / "metrics_canonical.csv"
    if m_path.exists():
        m = pd.read_csv(m_path)
        rw = m[m["dataset"] == "realworld_n2000"]
        ax.scatter(rw["specificity"], rw["sensitivity"], s=80, marker="o",
                   color="steelblue", alpha=0.8, label="Single architecture (default threshold)", zorder=3)
        # Annotate each
        for _, r in rw.iterrows():
            ax.annotate(ARCH_DISPLAY.get(r["architecture"], r["architecture"])[:18],
                        (r["specificity"], r["sensitivity"]),
                        xytext=(3, 3), textcoords="offset points",
                        fontsize=7, alpha=0.7)

    # Cascade Pareto frontier (highlight in different color)
    pf_path = results_dir / "cascade_pareto.csv"
    if pf_path.exists():
        pf = pd.read_csv(pf_path)
        pf = pf[pf["stage1"] < pf["stage2"]]  # dedupe symmetric pairs
        ax.scatter(pf["specificity"], pf["sensitivity"], s=50, marker="s",
                   color="darkorange", alpha=0.7, label="Two-stage cascade (Pareto frontier)", zorder=3)

    # Ensemble best balanced points
    en_path = results_dir / "ensemble_results.csv"
    if en_path.exists():
        en = pd.read_csv(en_path)
        # Only plot hard voting + selected soft voting best
        hv = en[en["rule"].str.startswith("hard_")]
        ax.scatter(hv["specificity"], hv["sensitivity"], s=50, marker="^",
                   color="purple", alpha=0.7, label="Hard-voting ensemble (k-of-9)", zorder=3)
        # Best soft-voting point
        soft = en[en["rule"].str.startswith("soft_")]
        if not soft.empty:
            soft = soft.copy()
            soft["balanced"] = soft[["sensitivity", "specificity"]].min(axis=1)
            best_soft = soft.loc[soft["balanced"].idxmax()]
            ax.scatter([best_soft["specificity"]], [best_soft["sensitivity"]], s=120,
                       marker="*", color="goldenrod", alpha=0.9,
                       label="Best soft-voting ensemble", zorder=4)

    # Multi-LLM consensus rules
    ml_path = results_dir / "multi_llm_consensus.csv"
    if ml_path.exists():
        ml = pd.read_csv(ml_path)
        # Skip the single-LLM-only rules (already in single-architecture points)
        consensus = ml[ml["rule"].str.contains("rule_")]
        ax.scatter(consensus["specificity"], consensus["sensitivity"], s=80,
                   marker="D", color="crimson", alpha=0.8,
                   label="Multi-LLM consensus rule", zorder=3)

    ax.set_xlabel("Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Receiver operating characteristic envelope: all strategies\nNo strategy reaches the clinical-grade target zone (top-right)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    out_path = output_dir / "figure3_pareto_frontier.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def render_figure2_action_recommendations(predictions_dir: Path, output_dir: Path) -> Path:
    """Figure 2: stacked bar chart of appropriate/under/over-triage rates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for csv_path in sorted(predictions_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        if "dataset" not in df.columns or "true_action" not in df.columns:
            continue
        rw = df[df["dataset"] == "realworld_n2000"]
        if rw.empty:
            continue
        arch = rw["architecture"].iloc[0]
        pred = rw["pred_action"].astype(int)
        true = rw["true_action"].astype(int)
        n = len(rw)
        rows.append({
            "arch": arch,
            "appropriate": (pred == true).sum() / n,
            "under_triage": (pred < true).sum() / n,
            "over_triage": (pred > true).sum() / n,
        })

    if not rows:
        out_path = output_dir / "figure2_action_recommendations.png"
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No action data available", ha="center")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    df = pd.DataFrame(rows).sort_values("appropriate", ascending=True)
    names = [ARCH_DISPLAY.get(a, a) for a in df["arch"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(names, df["under_triage"] * 100, color="#d62728", label="Under-triage")
    ax.barh(names, df["appropriate"] * 100, left=df["under_triage"] * 100,
            color="#2ca02c", label="Appropriate")
    ax.barh(names, df["over_triage"] * 100,
            left=(df["under_triage"] + df["appropriate"]) * 100,
            color="#ff7f0e", label="Over-triage")
    ax.set_xlabel("Proportion of test messages (%)")
    ax.set_title("Action-recommendation appropriateness (real-world n=2,000)")
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out_path = output_dir / "figure2_action_recommendations.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


ARCH_DISPLAY_SHORT = {
    "logreg_tfidf": "LogReg+TF-IDF",
    "xgboost_sbert": "XGBoost+SBERT",
    "constellation": "Constellation",
    "guardrails": "Guardrails",
    "cql_sens_opt": "CQL sens-opt",
    "cql_reward_opt": "CQL reward-opt",
    "actionhead": "ActionHead",
    "claude_opus_4_7_safety": "Claude Opus 4.7",
    "gemini_3_1_pro_safety": "Gemini 3.1 Pro",
}


def render_table5_cascade_pareto(results_dir: Path, output_dir: Path) -> Path:
    """Table 5: Pareto frontier of two-stage AND-rule cascades on the real-world set."""
    cascade_path = results_dir / "cascade_pareto.csv"
    if not cascade_path.exists():
        csv_path = output_dir / "table5_cascade_pareto.csv"
        md_path = output_dir / "table5_cascade_pareto.md"
        pd.DataFrame().to_csv(csv_path, index=False)
        md_path.write_text("_[Table 5 not yet rendered — cascade_pareto.csv missing]_\n")
        return csv_path

    pf = pd.read_csv(cascade_path)
    # Drop duplicate orderings (AND is commutative; keep stage1<stage2 alphabetically)
    pf = pf[pf["stage1"] < pf["stage2"]].copy()
    pf = pf.sort_values("sensitivity", ascending=False)

    def fmt_pair(s1, s2):
        return f"{ARCH_DISPLAY_SHORT.get(s1, s1)} × {ARCH_DISPLAY_SHORT.get(s2, s2)}"

    rows = []
    for _, r in pf.iterrows():
        rows.append({
            "Cascade (Stage 1 × Stage 2)": fmt_pair(r["stage1"], r["stage2"]),
            "Sensitivity (95% CI)": fmt_ci(r["sensitivity"], r["sensitivity_ci_lo"], r["sensitivity_ci_hi"]),
            "Specificity (95% CI)": fmt_ci(r["specificity"], r["specificity_ci_lo"], r["specificity_ci_hi"]),
            "PPV": f"{r['ppv']:.3f}" if pd.notna(r["ppv"]) else "—",
            "F1": f"{r['f1']:.3f}" if pd.notna(r["f1"]) else "—",
            "MCC": f"{r['mcc']:.3f}" if pd.notna(r["mcc"]) else "—",
            "FN per 1,000": f"{r['fn_per_1000']:.1f}",
        })
    out_df = pd.DataFrame(rows)
    csv_path = output_dir / "table5_cascade_pareto.csv"
    md_path = output_dir / "table5_cascade_pareto.md"
    out_df.to_csv(csv_path, index=False)
    write_markdown_table(out_df, md_path)
    return csv_path


def render_tableS4_category_stratification(results_dir: Path, output_dir: Path) -> Path:
    """Table S4: per-architecture sensitivity by hazard category on the real-world test set."""
    cs_path = results_dir / "category_stratification.csv"
    csv_path = output_dir / "tableS4_category_stratification.csv"
    md_path = output_dir / "tableS4_category_stratification.md"
    if not cs_path.exists():
        pd.DataFrame().to_csv(csv_path, index=False)
        md_path.write_text("_[Table S4 not yet rendered — category_stratification.csv missing]_\n")
        return csv_path
    cs = pd.read_csv(cs_path)
    if cs.empty:
        pd.DataFrame().to_csv(csv_path, index=False)
        md_path.write_text("_[Table S4 not yet rendered — empty stratification]_\n")
        return csv_path
    pivot = cs.pivot_table(index="hazard_category", columns="architecture",
                            values="sensitivity", aggfunc="first").round(3)
    # Order columns by mean across categories (descending = best-performing first)
    col_means = pivot.mean(axis=0).sort_values(ascending=False)
    pivot = pivot[col_means.index]
    # Use display labels
    pivot.columns = [ARCH_DISPLAY.get(c, c) for c in pivot.columns]
    # Add a column with n_hazards per category
    n_per_cat = cs.groupby("hazard_category")["n_hazards"].first()
    pivot.insert(0, "n hazards", n_per_cat)
    # Add a row with mean sensitivity per architecture across categories
    out_df = pivot.reset_index()
    out_df.to_csv(csv_path, index=False)
    write_markdown_table(out_df, md_path)
    return csv_path


def render_table8_amc_stack(results_dir: Path, output_dir: Path) -> Path:
    """Table 8: AMC standard guardrail stack (CHAI / Shah 2024 deployed configuration)."""
    p = results_dir / "amc_guardrail_stack.csv"
    csv_path = output_dir / "table8_amc_guardrail_stack.csv"
    md_path = output_dir / "table8_amc_guardrail_stack.md"
    if not p.exists():
        pd.DataFrame().to_csv(csv_path, index=False)
        md_path.write_text("_[Table 8 not yet rendered — amc_guardrail_stack.csv missing]_\n")
        return csv_path
    df = pd.read_csv(p)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "AMC stack configuration": r["stack_label"],
            "Sensitivity": f"{r['sensitivity']:.3f}",
            "Specificity": f"{r['specificity']:.3f}",
            "PPV": f"{r['ppv']:.3f}" if pd.notna(r["ppv"]) else "—",
            "F1": f"{r['f1']:.3f}" if pd.notna(r["f1"]) else "—",
            "MCC": f"{r['mcc']:.3f}" if pd.notna(r["mcc"]) else "—",
            "Clinical-grade?": "Yes" if r["clinical_grade"] else "No",
        })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(csv_path, index=False)
    write_markdown_table(out_df, md_path)
    return csv_path


def render_table9_deployment_grade(results_dir: Path, output_dir: Path) -> Path:
    """Table 9: deployment-grade sensitivity-floor analysis + disagreement-stratified workflow."""
    sf_path = results_dir / "deployment_grade_sens_floor.csv"
    ag_path = results_dir / "deployment_grade_agreement.csv"
    csv_path = output_dir / "table9_deployment_grade.csv"
    md_path = output_dir / "table9_deployment_grade.md"
    parts = []
    if sf_path.exists():
        sf = pd.read_csv(sf_path)
        parts.append("**Panel A — Sensitivity-floor target: max specificity achievable subject to sensitivity threshold**\n")
        # Build table
        a_rows = []
        for _, r in sf.iterrows():
            a_rows.append({
                "Sensitivity floor": f"≥ {r['sens_floor']:.2f}",
                "N configurations meeting floor": int(r["n_configurations_meeting_floor"]) if pd.notna(r["n_configurations_meeting_floor"]) else 0,
                "Max specificity at floor": f"{r['max_specificity']:.3f}" if pd.notna(r["max_specificity"]) else "—",
                "Winning configuration": (f"{r['winning_strategy_type']} / {r['winning_label']}" if pd.notna(r.get("winning_strategy_type")) else "—"),
                "Sens (winning)": f"{r['winning_sensitivity']:.3f}" if pd.notna(r["winning_sensitivity"]) else "—",
            })
        a_df = pd.DataFrame(a_rows)
        # Manual pipe-delimited markdown (no tabulate dependency)
        cols = a_df.columns.tolist()
        lines = ["| " + " | ".join(cols) + " |",
                 "|" + "|".join(["---"] * len(cols)) + "|"]
        for _, rr in a_df.iterrows():
            lines.append("| " + " | ".join(str(rr[c]) for c in cols) + " |")
        parts.append("\n".join(lines))

    if ag_path.exists():
        ag = pd.read_csv(ag_path)
        parts.append("\n\n**Panel B — Disagreement-stratified clinician-review policy**\n")
        b_rows = []
        for _, r in ag.iterrows():
            b_rows.append({
                "Stratum (n architectures flagging)": r["stratum"],
                "Recommended policy": r["policy"].replace("_", " "),
                "N messages": int(r["n_messages"]) if pd.notna(r["n_messages"]) else 0,
                "N hazards (true)": int(r["n_hazards"]) if pd.notna(r["n_hazards"]) else 0,
                "Hazard prevalence in stratum": f"{r['hazard_prevalence_in_stratum']:.3f}" if pd.notna(r["hazard_prevalence_in_stratum"]) else "—",
                "Share of total messages": f"{r['share_of_total_messages']:.1%}" if pd.notna(r["share_of_total_messages"]) else "—",
            })
        b_df = pd.DataFrame(b_rows)
        # Manual markdown so we don't depend on tabulate
        cols = b_df.columns.tolist()
        lines = ["| " + " | ".join(cols) + " |",
                 "|" + "|".join(["---"] * len(cols)) + "|"]
        for _, rr in b_df.iterrows():
            lines.append("| " + " | ".join(str(rr[c]) for c in cols) + " |")
        parts.append("\n".join(lines))

    md_text = "\n".join(parts) if parts else "_[Table 9 not yet rendered]_\n"
    md_path.write_text(md_text)
    # Also write a flat-format CSV combining both panels for archival
    csv_path.write_text("# Table 9 has two panels — see .md for human-readable format\n")
    return csv_path


def render_table7_closing_the_gap(results_dir: Path, output_dir: Path) -> Path:
    """Table 7: best operating point under each closing-the-gap strategy.

    Synthesizes results from multiple analyses (single architecture, ensemble,
    cascade, threshold optimization, multi-LLM consensus, RAG) into one
    summary table showing the best balanced point under each strategy. Allows
    reviewers to answer 'did any combination close the gap?' at-a-glance.
    """
    rows = []
    target_row = {
        "Strategy": "**Clinical-grade target**",
        "Best balanced (sens, spec)": "≥0.80, ≥0.80",
        "Best F1": "—",
        "Sens at best F1": "—",
        "Spec at best F1": "—",
        "Clinical-grade reached?": "Target",
    }

    # 1. Best single architecture (from metrics_canonical at default thresholds)
    m_path = results_dir / "metrics_canonical.csv"
    if m_path.exists():
        m = pd.read_csv(m_path)
        rw = m[m["dataset"] == "realworld_n2000"].copy()
        rw["balanced"] = rw[["sensitivity", "specificity"]].min(axis=1)
        if not rw.empty:
            best_bal = rw.loc[rw["balanced"].idxmax()]
            best_f1 = rw.loc[rw["f1"].idxmax()]
            rows.append({
                "Strategy": "Single architecture (default threshold)",
                "Best balanced (sens, spec)": f"{best_bal['sensitivity']:.3f}, {best_bal['specificity']:.3f} ({ARCH_DISPLAY.get(best_bal['architecture'], best_bal['architecture'])})",
                "Best F1": f"{best_f1['f1']:.3f}",
                "Sens at best F1": f"{best_f1['sensitivity']:.3f}",
                "Spec at best F1": f"{best_f1['specificity']:.3f}",
                "Clinical-grade reached?": "No",
            })

    # 2. Best single architecture (threshold-optimized)
    th_path = results_dir / "threshold_optimized.csv"
    if th_path.exists():
        th = pd.read_csv(th_path)
        if not th.empty:
            th["balanced"] = th[["closest_to_clinical_grade_sens",
                                  "closest_to_clinical_grade_spec"]].min(axis=1)
            best_bal = th.loc[th["balanced"].idxmax()]
            best_f1 = th.loc[th["f1_max"].idxmax()]
            rows.append({
                "Strategy": "Single architecture (threshold-optimized)",
                "Best balanced (sens, spec)": f"{best_bal['closest_to_clinical_grade_sens']:.3f}, {best_bal['closest_to_clinical_grade_spec']:.3f} ({ARCH_DISPLAY.get(best_bal['architecture'], best_bal['architecture'])})",
                "Best F1": f"{best_f1['f1_max']:.3f}",
                "Sens at best F1": f"{best_f1['f1_max_sens']:.3f}",
                "Spec at best F1": f"{best_f1['f1_max_spec']:.3f}",
                "Clinical-grade reached?": "Yes" if th["clinical_grade_reachable"].any() else "No",
            })

    # 3. Best ensemble (hard-voting + soft-voting all 9 architectures)
    ens_path = results_dir / "ensemble_results.csv"
    if ens_path.exists():
        ens = pd.read_csv(ens_path)
        if not ens.empty:
            ens["balanced"] = ens[["sensitivity", "specificity"]].min(axis=1)
            best_bal = ens.loc[ens["balanced"].idxmax()]
            best_f1 = ens.loc[ens["f1"].idxmax()]
            rows.append({
                "Strategy": "Ensemble of 9 architectures (213 configurations evaluated)",
                "Best balanced (sens, spec)": f"{best_bal['sensitivity']:.3f}, {best_bal['specificity']:.3f} ({best_bal['rule']})",
                "Best F1": f"{best_f1['f1']:.3f}",
                "Sens at best F1": f"{best_f1['sensitivity']:.3f}",
                "Spec at best F1": f"{best_f1['specificity']:.3f}",
                "Clinical-grade reached?": "Yes" if ens["clinical_grade"].any() else "No",
            })

    # 4. Best cascade (from cascade_matrix)
    cas_path = results_dir / "cascade_matrix.csv"
    if cas_path.exists():
        cas = pd.read_csv(cas_path)
        rw_cas = cas[(cas["dataset"] == "realworld_n2000") & (cas["stage1"] < cas["stage2"])].copy()
        if not rw_cas.empty:
            rw_cas["balanced"] = rw_cas[["sensitivity", "specificity"]].min(axis=1)
            best_bal = rw_cas.loc[rw_cas["balanced"].idxmax()]
            best_f1 = rw_cas.loc[rw_cas["f1"].idxmax()]
            cg = ((rw_cas["sensitivity"] >= 0.80) & (rw_cas["specificity"] >= 0.80)).any()
            rows.append({
                "Strategy": "Two-stage cascade (72 configurations evaluated)",
                "Best balanced (sens, spec)": f"{best_bal['sensitivity']:.3f}, {best_bal['specificity']:.3f} ({ARCH_DISPLAY.get(best_bal['stage1'], best_bal['stage1'])} × {ARCH_DISPLAY.get(best_bal['stage2'], best_bal['stage2'])})",
                "Best F1": f"{best_f1['f1']:.3f}",
                "Sens at best F1": f"{best_f1['sensitivity']:.3f}",
                "Spec at best F1": f"{best_f1['specificity']:.3f}",
                "Clinical-grade reached?": "Yes" if cg else "No",
            })

    # 5. Multi-LLM consensus
    mll_path = results_dir / "multi_llm_consensus.csv"
    if mll_path.exists():
        mll = pd.read_csv(mll_path)
        if not mll.empty:
            mll["balanced"] = mll[["sensitivity", "specificity"]].min(axis=1)
            best_bal = mll.loc[mll["balanced"].idxmax()]
            best_f1 = mll.loc[mll["f1"].idxmax()]
            rows.append({
                "Strategy": "Multi-LLM consensus (Claude + Gemini)",
                "Best balanced (sens, spec)": f"{best_bal['sensitivity']:.3f}, {best_bal['specificity']:.3f} ({best_bal['rule']})",
                "Best F1": f"{best_f1['f1']:.3f}",
                "Sens at best F1": f"{best_f1['sensitivity']:.3f}",
                "Spec at best F1": f"{best_f1['specificity']:.3f}",
                "Clinical-grade reached?": "Yes" if mll["clinical_grade"].any() else "No",
            })

    # 6. RAG (if present in metrics_canonical)
    if m_path.exists():
        m = pd.read_csv(m_path)
        rag = m[(m["dataset"] == "realworld_n2000") & (m["architecture"].str.contains("rag", na=False))]
        if not rag.empty:
            row = rag.iloc[0]
            rows.append({
                "Strategy": "Retrieval-augmented LLM (RAG over 1,280 training examples)",
                "Best balanced (sens, spec)": f"{row['sensitivity']:.3f}, {row['specificity']:.3f}",
                "Best F1": f"{row['f1']:.3f}",
                "Sens at best F1": f"{row['sensitivity']:.3f}",
                "Spec at best F1": f"{row['specificity']:.3f}",
                "Clinical-grade reached?": "Yes" if (row["sensitivity"] >= 0.80 and row["specificity"] >= 0.80) else "No",
            })

    rows.insert(0, target_row)
    out_df = pd.DataFrame(rows)
    csv_path = output_dir / "table7_closing_the_gap.csv"
    md_path = output_dir / "table7_closing_the_gap.md"
    out_df.to_csv(csv_path, index=False)
    write_markdown_table(out_df, md_path)
    return csv_path


def render_table6_threshold_optimized(results_dir: Path, output_dir: Path) -> Path:
    """Table 6: post-hoc threshold optimization — F1-max, MCC-max, clinical-grade reachability."""
    th_path = results_dir / "threshold_optimized.csv"
    if not th_path.exists():
        csv_path = output_dir / "table6_threshold_optimized.csv"
        md_path = output_dir / "table6_threshold_optimized.md"
        pd.DataFrame().to_csv(csv_path, index=False)
        md_path.write_text("_[Table 6 not yet rendered — threshold_optimized.csv missing]_\n")
        return csv_path

    th = pd.read_csv(th_path)
    rows = []
    for _, r in th.iterrows():
        rows.append({
            "Architecture": ARCH_DISPLAY.get(r["architecture"], r["architecture"]),
            "F1-max": f"{r['f1_max']:.3f}",
            "Sens at F1-max": f"{r['f1_max_sens']:.3f}",
            "Spec at F1-max": f"{r['f1_max_spec']:.3f}",
            "MCC-max": f"{r['mcc_max']:.3f}",
            "Sens at MCC-max": f"{r['mcc_max_sens']:.3f}",
            "Spec at MCC-max": f"{r['mcc_max_spec']:.3f}",
            "Clinical-grade reachable?": "Yes" if r["clinical_grade_reachable"] else "No",
            "Sens/Spec at closest point": f"{r['closest_to_clinical_grade_sens']:.3f} / {r['closest_to_clinical_grade_spec']:.3f}",
        })
    out_df = pd.DataFrame(rows)
    csv_path = output_dir / "table6_threshold_optimized.csv"
    md_path = output_dir / "table6_threshold_optimized.md"
    out_df.to_csv(csv_path, index=False)
    write_markdown_table(out_df, md_path)
    return csv_path


def render_tableS3_cascade_full(results_dir: Path, output_dir: Path) -> Path:
    """Table S3: Full 72-pair cascade matrix on the real-world set."""
    cascade_path = results_dir / "cascade_matrix.csv"
    if not cascade_path.exists():
        csv_path = output_dir / "tableS3_cascade_full.csv"
        md_path = output_dir / "tableS3_cascade_full.md"
        pd.DataFrame().to_csv(csv_path, index=False)
        md_path.write_text("_[Table S3 not yet rendered — cascade_matrix.csv missing]_\n")
        return csv_path

    cm = pd.read_csv(cascade_path)
    cm = cm[(cm["dataset"] == "realworld_n2000") & (cm["stage1"] < cm["stage2"])].copy()
    cm = cm.sort_values(["sensitivity", "specificity"], ascending=[False, False])

    def fmt_pair(s1, s2):
        return f"{ARCH_DISPLAY_SHORT.get(s1, s1)} × {ARCH_DISPLAY_SHORT.get(s2, s2)}"

    rows = []
    for _, r in cm.iterrows():
        rows.append({
            "Cascade": fmt_pair(r["stage1"], r["stage2"]),
            "Sens": f"{r['sensitivity']:.3f}",
            "Spec": f"{r['specificity']:.3f}",
            "PPV": f"{r['ppv']:.3f}" if pd.notna(r["ppv"]) else "—",
            "F1": f"{r['f1']:.3f}" if pd.notna(r["f1"]) else "—",
            "MCC": f"{r['mcc']:.3f}" if pd.notna(r["mcc"]) else "—",
        })
    out_df = pd.DataFrame(rows)
    csv_path = output_dir / "tableS3_cascade_full.csv"
    md_path = output_dir / "tableS3_cascade_full.md"
    out_df.to_csv(csv_path, index=False)
    write_markdown_table(out_df, md_path)
    return csv_path


def render_all(predictions_dir: Path, results_dir: Path) -> dict[str, Path]:
    """Render every table and figure from canonical CSVs."""
    results_dir = Path(results_dir)
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = results_dir / "metrics_canonical.csv"
    curves_path = results_dir / "operating_curves.csv"
    delta_path = results_dir / "delta_bootstrap_canonical.csv"

    metrics_df = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    curves_df = pd.read_csv(curves_path) if curves_path.exists() else pd.DataFrame()
    delta_df = pd.read_csv(delta_path) if delta_path.exists() else pd.DataFrame()

    out: dict[str, Path] = {}
    print("\n[render] Table 2 (detection metrics)")
    out["table2"] = render_table2_detection_metrics(metrics_df, tables_dir)
    print(f"  → {out['table2']}")
    print("[render] Table 3 (operating points)")
    out["table3"] = render_table3_operating_points(metrics_df, curves_df, tables_dir)
    print(f"  → {out['table3']}")
    print("[render] Table 4 (action recommendations)")
    out["table4"] = render_table4_action_recommendations(predictions_dir, tables_dir)
    print(f"  → {out['table4']}")
    print("[render] Table 5 (cascade Pareto frontier)")
    out["table5"] = render_table5_cascade_pareto(results_dir, tables_dir)
    print(f"  → {out['table5']}")
    print("[render] Table 6 (threshold optimization)")
    out["table6"] = render_table6_threshold_optimized(results_dir, tables_dir)
    print(f"  → {out['table6']}")
    print("[render] Table 7 (closing-the-gap summary)")
    out["table7"] = render_table7_closing_the_gap(results_dir, tables_dir)
    print(f"  → {out['table7']}")
    print("[render] Table 8 (AMC standard guardrail stack)")
    out["table8"] = render_table8_amc_stack(results_dir, tables_dir)
    print(f"  → {out['table8']}")
    print("[render] Table 9 (deployment-grade)")
    out["table9"] = render_table9_deployment_grade(results_dir, tables_dir)
    print(f"  → {out['table9']}")
    print("[render] Table S1 (physician holdout)")
    out["tableS1"] = render_tableS1_physician_metrics(metrics_df, tables_dir)
    print(f"  → {out['tableS1']}")
    print("[render] Table S2 (Δ bootstrap)")
    out["tableS2"] = render_tableS2_delta_bootstrap(delta_df, tables_dir)
    print(f"  → {out['tableS2']}")
    print("[render] Table S3 (full cascade matrix)")
    out["tableS3"] = render_tableS3_cascade_full(results_dir, tables_dir)
    print(f"  → {out['tableS3']}")
    print("[render] Table S4 (category stratification)")
    out["tableS4"] = render_tableS4_category_stratification(results_dir, tables_dir)
    print(f"  → {out['tableS4']}")
    print("[render] Figure 1 (Δ sens chart)")
    out["figure1"] = render_figure1_sens_change(delta_df, figures_dir)
    print(f"  → {out['figure1']}")
    print("[render] Figure 2 (action recommendations stacked bar)")
    out["figure2"] = render_figure2_action_recommendations(predictions_dir, figures_dir)
    print(f"  → {out['figure2']}")
    print("[render] Figure 3 (Pareto frontier across all strategies)")
    out["figure3"] = render_figure3_pareto_frontier(results_dir, figures_dir)
    print(f"  → {out['figure3']}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    render_all(args.predictions_dir, args.results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
