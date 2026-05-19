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
    print("[render] Table S1 (physician holdout)")
    out["tableS1"] = render_tableS1_physician_metrics(metrics_df, tables_dir)
    print(f"  → {out['tableS1']}")
    print("[render] Table S2 (Δ bootstrap)")
    out["tableS2"] = render_tableS2_delta_bootstrap(delta_df, tables_dir)
    print(f"  → {out['tableS2']}")
    print("[render] Figure 1 (Δ sens chart)")
    out["figure1"] = render_figure1_sens_change(delta_df, figures_dir)
    print(f"  → {out['figure1']}")
    print("[render] Figure 2 (action recommendations stacked bar)")
    out["figure2"] = render_figure2_action_recommendations(predictions_dir, figures_dir)
    print(f"  → {out['figure2']}")
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
