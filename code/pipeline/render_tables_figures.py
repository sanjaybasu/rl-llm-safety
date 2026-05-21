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
    "claude_opus_4_7_rag": "Claude Opus 4.7 (RAG)",
    "gemini_3_1_pro_safety": "Gemini 3.1 Pro (safety-augmented)",
    "gemini_3_1_pro_rag": "Gemini 3.1 Pro (RAG)",
    "gpt_5_5_safety": "GPT-5.5 (safety-augmented)",
    "gpt_5_5_rag": "GPT-5.5 (RAG)",
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


def _wilson_ci(k: float, n: float, z: float = 1.96) -> tuple:
    """Wilson 95% CI for a proportion k/n. Returns (lo, hi)."""
    import math
    if n is None or n <= 0 or k is None or math.isnan(k) or math.isnan(n):
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _delong_auroc_ci(proba, true, alpha: float = 0.05) -> tuple:
    """DeLong 95% CI for a single AUROC via Mann-Whitney structural variance.

    Reference: DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas
    under two or more correlated receiver operating characteristic curves: a
    nonparametric approach. Biometrics 1988;44(3):837-845.

    Returns (auroc, ci_lo, ci_hi). Returns (nan, nan, nan) if there is no
    positive or no negative case, or fewer than 2 of either (variance is
    undefined with a single sample).
    """
    import math
    proba = np.asarray(proba, dtype=float)
    true = np.asarray(true, dtype=int)
    pos = proba[true == 1]
    neg = proba[true == 0]
    m = len(pos); n = len(neg)
    if m < 2 or n < 2:
        return (float("nan"), float("nan"), float("nan"))
    # Mann-Whitney U-based AUROC
    # AUROC = (sum over (i,j) of [pos_i > neg_j] + 0.5 * [pos_i == neg_j]) / (m*n)
    pos_sorted = np.sort(pos)
    neg_sorted = np.sort(neg)
    rank_sum = 0.0
    for p in pos:
        rank_sum += np.searchsorted(neg_sorted, p, side="right") - 0.5 * (
            np.searchsorted(neg_sorted, p, side="right") - np.searchsorted(neg_sorted, p, side="left")
        )
    auroc = rank_sum / (m * n)
    # DeLong structural components
    V10 = np.array([
        (np.searchsorted(neg_sorted, p, side="left")
         + 0.5 * (np.searchsorted(neg_sorted, p, side="right")
                  - np.searchsorted(neg_sorted, p, side="left")))
        / n
        for p in pos
    ])
    V01 = np.array([
        ((m - np.searchsorted(pos_sorted, q, side="right"))
         + 0.5 * (np.searchsorted(pos_sorted, q, side="right")
                  - np.searchsorted(pos_sorted, q, side="left")))
        / m
        for q in neg
    ])
    var_auroc = V10.var(ddof=1) / m + V01.var(ddof=1) / n
    se = math.sqrt(max(var_auroc, 0.0))
    # 1.96 ≈ scipy.stats.norm.ppf(0.975); hardcode to avoid scipy dependency
    z = 1.95996398454005423552
    return (auroc, max(0.0, auroc - z * se), min(1.0, auroc + z * se))


def render_table2_detection_metrics(metrics_df: pd.DataFrame, output_dir: Path) -> Path:
    """Table 2: per-architecture detection metrics on real-world test set."""
    df = metrics_df[metrics_df["dataset"] == "realworld_n2000"].copy()
    N_HAZ_RW = 165
    N_TOTAL_RW = 2000
    rows = []
    for _, row in df.iterrows():
        d = row.to_dict()
        tp = d.get("tp", float("nan"))
        fp = d.get("fp", float("nan"))
        tn = d.get("tn", float("nan"))
        fn = d.get("fn", float("nan"))
        ppv_lo, ppv_hi = _wilson_ci(tp, tp + fp)
        npv_lo, npv_hi = _wilson_ci(tn, tn + fn)
        # FN per 1,000 CI: derive from sensitivity Wilson CI bounds
        # FN/1k = (1 - sens) * N_HAZ * 1000 / N_TOTAL; upper sens -> lower FN/1k
        sens_lo = d.get("sensitivity_ci_lo", float("nan"))
        sens_hi = d.get("sensitivity_ci_hi", float("nan"))
        fn_per_1000 = d.get("fn_per_1000", float("nan"))
        if not (np.isnan(sens_lo) or np.isnan(sens_hi) or np.isnan(fn_per_1000)):
            fn1k_lo = (1.0 - sens_hi) * N_HAZ_RW * 1000.0 / N_TOTAL_RW
            fn1k_hi = (1.0 - sens_lo) * N_HAZ_RW * 1000.0 / N_TOTAL_RW
            fn_per_1k_str = f"{fn_per_1000:.1f} ({fn1k_lo:.1f}–{fn1k_hi:.1f})"
        elif not np.isnan(fn_per_1000):
            fn_per_1k_str = f"{fn_per_1000:.1f}"
        else:
            fn_per_1k_str = "—"
        def _i(x):
            return f"{int(x)}" if not (x is None or (isinstance(x, float) and np.isnan(x))) else "—"
        rows.append({
            "Architecture": ARCH_DISPLAY.get(d["architecture"], d["architecture"]),
            "TP / FN / TN / FP": f"{_i(tp)} / {_i(fn)} / {_i(tn)} / {_i(fp)}",
            "Sensitivity (95% CI)": fmt_ci(
                d.get("sensitivity", float("nan")),
                d.get("sensitivity_ci_lo", float("nan")),
                d.get("sensitivity_ci_hi", float("nan"))),
            "Specificity (95% CI)": fmt_ci(
                d.get("specificity", float("nan")),
                d.get("specificity_ci_lo", float("nan")),
                d.get("specificity_ci_hi", float("nan"))),
            "PPV (95% CI)": fmt_ci(d.get("ppv", float("nan")), ppv_lo, ppv_hi)
                if not np.isnan(d.get("ppv", float("nan"))) else "—",
            "NPV (95% CI)": fmt_ci(d.get("npv", float("nan")), npv_lo, npv_hi)
                if not np.isnan(d.get("npv", float("nan"))) else "—",
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
            "FN per 1,000 (95% CI)": fn_per_1k_str,
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
    # Each row reports a single architecture's sensitivity at five target
    # specificity points. The cell is the sensitivity (95% Wilson CI) at the
    # closest achievable operating point; achieved-specificity deviation is
    # captured in a footnote and the full sweep is in Table S7.
    N_HAZ_RW = 165
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
            # Wilson 95% CI on sensitivity at this operating point. TP at the
            # selected threshold = round(sens * N_HAZ); denominator is N_HAZ.
            tp_here = round(float(sens) * N_HAZ_RW)
            sens_lo, sens_hi = _wilson_ci(tp_here, N_HAZ_RW)
            # Flag the cell with † if achieved specificity deviates more than 0.02 from target
            flag = "†" if abs(achieved_spec - target_spec) > 0.02 else ""
            row[f"Sens at Spec ≥ {target_spec:.2f}"] = (
                f"{sens:.3f}{flag} ({sens_lo:.3f}–{sens_hi:.3f})"
            )
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


def render_tableS1_physician_metrics(
    metrics_df: pd.DataFrame, output_dir: Path,
    predictions_dir: Path | None = None,
) -> Path:
    """Table S1: physician-holdout metrics with TP/FN/TN/FP per architecture.

    AUROC 95% CIs are computed by the DeLong (1988) structural-variance method
    in-line from the per-message prediction files. Architectures that emit
    discrete hazard flags rather than continuous probability scalars (the
    frontier LLMs and ActionHead's hazard endpoint) display "N/A" for AUROC,
    consistent with the Table S1 caption.
    """
    df = metrics_df[metrics_df["dataset"] == "physician_n41"].copy()
    # Build a {architecture: (proba, true)} map from the per-message CSVs so
    # DeLong CIs can be computed locally without depending on whether the
    # metrics_phase wrote auroc_ci_lo/hi columns.
    pred_index: dict[str, tuple] = {}
    if predictions_dir is not None and predictions_dir.exists():
        for csv_path in sorted(Path(predictions_dir).glob("*.csv")):
            try:
                pdf = pd.read_csv(csv_path)
            except Exception:
                continue
            sub = pdf[pdf["dataset"] == "physician_n41"].copy()
            if sub.empty:
                continue
            arch = str(sub["architecture"].iloc[0])
            sub["pred_proba"] = pd.to_numeric(sub.get("pred_proba"), errors="coerce")
            if sub["pred_proba"].isna().all():
                continue
            proba = sub["pred_proba"].values
            true = sub["true_hazard"].astype(int).values
            pred_index[arch] = (proba, true)

    rows = []
    for _, row in df.iterrows():
        d = row.to_dict()
        arch = d["architecture"]
        # AUROC + DeLong CI cell
        if arch in pred_index:
            proba, true = pred_index[arch]
            auroc_pt, lo, hi = _delong_auroc_ci(proba, true)
            if not (np.isnan(auroc_pt) or np.isnan(lo) or np.isnan(hi)):
                auroc_cell = f"{auroc_pt:.3f} ({lo:.3f}–{hi:.3f})"
            elif not np.isnan(auroc_pt):
                auroc_cell = f"{auroc_pt:.3f}"
            else:
                auroc_cell = "N/A"
        else:
            auroc_cell = "N/A"
        rows.append({
            "Architecture": ARCH_DISPLAY.get(arch, arch),
            "TP/FN/TN/FP": f"{int(d['tp'])}/{int(d['fn'])}/{int(d['tn'])}/{int(d['fp'])}",
            "Sensitivity (95% CI)": fmt_ci(
                d.get("sensitivity", float("nan")),
                d.get("sensitivity_ci_lo", float("nan")),
                d.get("sensitivity_ci_hi", float("nan"))),
            "Specificity (95% CI)": fmt_ci(
                d.get("specificity", float("nan")),
                d.get("specificity_ci_lo", float("nan")),
                d.get("specificity_ci_hi", float("nan"))),
            "F1 (95% CI)": fmt_ci(
                d.get("f1", float("nan")),
                d.get("f1_ci_lo", float("nan")),
                d.get("f1_ci_hi", float("nan"))),
            "MCC (95% CI)": fmt_ci(
                d.get("mcc", float("nan")),
                d.get("mcc_ci_lo", float("nan")),
                d.get("mcc_ci_hi", float("nan"))),
            "AUROC (95% CI)": auroc_cell,
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
    # Sample sizes per test set (hazards-positive subgroup determines Wilson CI for sens)
    N_HAZ_PHYS = 27  # physician-holdout hazards
    N_HAZ_RW = 165   # real-world hazards
    rows = []
    for _, row in delta_df.iterrows():
        phys_lo, phys_hi = _wilson_ci(round(row["phys_sens"] * N_HAZ_PHYS), N_HAZ_PHYS)
        real_lo, real_hi = _wilson_ci(round(row["real_sens"] * N_HAZ_RW), N_HAZ_RW)
        rows.append({
            "Architecture": ARCH_DISPLAY.get(row["architecture"], row["architecture"]),
            "Physician-set sensitivity (95% CI)": fmt_ci(row["phys_sens"], phys_lo, phys_hi),
            "Real-world sensitivity (95% CI)": fmt_ci(row["real_sens"], real_lo, real_hi),
            "Δ Sens, pp (95% CI)": f"{row['delta_sens_pp']:+.1f} "
                                    f"({row['delta_sens_ci_lo']:+.1f} to {row['delta_sens_ci_hi']:+.1f})",
            "Δ Spec, pp (95% CI)": f"{row['delta_spec_pp']:+.1f} "
                                    f"({row['delta_spec_ci_lo']:+.1f} to {row['delta_spec_ci_hi']:+.1f})",
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
    fig, ax = plt.subplots(figsize=(9, 7))

    # Shade clinical-grade target region (top-right quadrant ≥0.80, ≥0.80)
    ax.axhspan(0.80, 1.0, xmin=0.80, xmax=1.0, color="green", alpha=0.10, zorder=0)
    ax.axhline(0.80, color="green", linestyle=":", linewidth=1.0, alpha=0.5, zorder=1)
    ax.axvline(0.80, color="green", linestyle=":", linewidth=1.0, alpha=0.5, zorder=1)
    ax.text(0.85, 0.92, "Clinical-grade\ntarget zone", fontsize=10,
            color="darkgreen", ha="left", va="center", alpha=0.8)

    n_single = n_cascade = n_hard = n_soft = n_consensus = 0

    # Single-architecture default-threshold points
    m_path = results_dir / "metrics_canonical.csv"
    if m_path.exists():
        m = pd.read_csv(m_path)
        rw = m[m["dataset"] == "realworld_n2000"]
        n_single = len(rw)
        ax.scatter(rw["specificity"], rw["sensitivity"], s=80, marker="o",
                   color="steelblue", alpha=0.8,
                   label=f"Single architecture, default threshold (n={n_single})", zorder=3)

    # Cascade Pareto frontier (highlight in different color)
    pf_path = results_dir / "cascade_pareto.csv"
    if pf_path.exists():
        pf = pd.read_csv(pf_path)
        pf = pf[pf["stage1"] < pf["stage2"]]  # dedupe symmetric pairs
        n_cascade = len(pf)
        ax.scatter(pf["specificity"], pf["sensitivity"], s=50, marker="s",
                   color="darkorange", alpha=0.7,
                   label=f"Two-stage cascade, Pareto frontier (n={n_cascade})", zorder=3)

    # Ensemble best balanced points
    en_path = results_dir / "ensemble_results.csv"
    if en_path.exists():
        en = pd.read_csv(en_path)
        hv = en[en["rule"].str.startswith("hard_")]
        n_hard = len(hv)
        ax.scatter(hv["specificity"], hv["sensitivity"], s=50, marker="^",
                   color="purple", alpha=0.7,
                   label=f"Hard-voting ensemble, k-of-9 for k=1..9 (n={n_hard})", zorder=3)
        soft = en[en["rule"].str.startswith("soft_")]
        if not soft.empty:
            soft = soft.copy()
            soft["balanced"] = soft[["sensitivity", "specificity"]].min(axis=1)
            best_soft = soft.loc[soft["balanced"].idxmax()]
            n_soft = 1
            ax.scatter([best_soft["specificity"]], [best_soft["sensitivity"]], s=160,
                       marker="*", color="goldenrod", alpha=0.95, edgecolor="black", linewidth=0.5,
                       label="Best soft-voting ensemble (n=1)", zorder=5)

    # Multi-LLM consensus rules
    ml_path = results_dir / "multi_llm_consensus.csv"
    if ml_path.exists():
        ml = pd.read_csv(ml_path)
        consensus = ml[ml["rule"].str.contains("rule_")]
        n_consensus = len(consensus)
        ax.scatter(consensus["specificity"], consensus["sensitivity"], s=80,
                   marker="D", color="crimson", alpha=0.8,
                   label=f"Multi-LLM consensus rule (n={n_consensus})", zorder=3)

    ax.set_xlabel("Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    total = n_single + n_cascade + n_hard + n_soft + n_consensus
    ax.set_title(
        f"Receiver operating characteristic envelope: {total} configurations across 5 strategy classes"
    )
    # Place legend BELOW the plot in two rows so that (a) no data points are
    # covered and (b) the green "Clinical-grade target zone" text in the
    # upper-right plot area is not clipped by a side legend.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3,
              fontsize=9, framealpha=1.0, borderaxespad=0)
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
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.barh(names, df["under_triage"] * 100, color="#d62728", label="Under-triage")
    ax.barh(names, df["appropriate"] * 100, left=df["under_triage"] * 100,
            color="#2ca02c", label="Appropriate")
    ax.barh(names, df["over_triage"] * 100,
            left=(df["under_triage"] + df["appropriate"]) * 100,
            color="#ff7f0e", label="Over-triage")
    ax.set_xlabel("Proportion of test messages (%)")
    ax.set_title("Action-recommendation appropriateness (real-world n=2,000)")
    ax.set_xlim(0, 100)
    # Place legend outside the plot area so it does not overlap the bars
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), framealpha=1.0, borderaxespad=0)
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
    "claude_opus_4_7_rag": "Claude 4.7 + RAG",
    "gemini_3_1_pro_safety": "Gemini 3.1 Pro",
    "gemini_3_1_pro_rag": "Gemini 3.1 + RAG",
    "gpt_5_5_safety": "GPT-5.5",
    "gpt_5_5_rag": "GPT-5.5 + RAG",
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

    def _ppv_ci(tp, fp):
        if pd.isna(tp) or pd.isna(fp) or (tp + fp) <= 0:
            return (float("nan"), float("nan"))
        return _wilson_ci(float(tp), float(tp) + float(fp))

    def _bootstrap_f1_mcc(tp, fp, tn, fn, n_iter=1000, seed=42):
        """Parametric-bootstrap 95% CI for F1 and MCC from a 2x2 confusion table."""
        import math
        if any(pd.isna(x) for x in [tp, fp, tn, fn]):
            return ((float("nan"), float("nan")), (float("nan"), float("nan")))
        rng = np.random.RandomState(seed)
        n_pos = int(tp + fn)
        n_neg = int(tn + fp)
        sens_p = tp / max(n_pos, 1)
        spec_p = tn / max(n_neg, 1)
        f1s, mccs = [], []
        for _ in range(n_iter):
            tp_b = rng.binomial(n_pos, sens_p)
            tn_b = rng.binomial(n_neg, spec_p)
            fn_b = n_pos - tp_b
            fp_b = n_neg - tn_b
            denom_f1 = (2 * tp_b + fp_b + fn_b)
            f1 = (2 * tp_b) / denom_f1 if denom_f1 > 0 else float("nan")
            denom_mcc = math.sqrt((tp_b + fp_b) * (tp_b + fn_b) * (tn_b + fp_b) * (tn_b + fn_b))
            mcc = (tp_b * tn_b - fp_b * fn_b) / denom_mcc if denom_mcc > 0 else float("nan")
            f1s.append(f1); mccs.append(mcc)
        f1_lo, f1_hi = np.nanpercentile(f1s, 2.5), np.nanpercentile(f1s, 97.5)
        mcc_lo, mcc_hi = np.nanpercentile(mccs, 2.5), np.nanpercentile(mccs, 97.5)
        return ((f1_lo, f1_hi), (mcc_lo, mcc_hi))

    N_HAZ_RW = 165
    N_TOTAL_RW = 2000
    rows = []
    for _, r in pf.iterrows():
        ppv_lo, ppv_hi = _ppv_ci(r.get("tp"), r.get("fp"))
        (f1_lo, f1_hi), (mcc_lo, mcc_hi) = _bootstrap_f1_mcc(
            r.get("tp"), r.get("fp"), r.get("tn"), r.get("fn"))
        # FN per 1,000 CI: derived from Wilson sensitivity CI bounds
        s_lo = r.get("sensitivity_ci_lo", float("nan"))
        s_hi = r.get("sensitivity_ci_hi", float("nan"))
        fn1k_lo = (1.0 - s_hi) * N_HAZ_RW * 1000.0 / N_TOTAL_RW if pd.notna(s_hi) else float("nan")
        fn1k_hi = (1.0 - s_lo) * N_HAZ_RW * 1000.0 / N_TOTAL_RW if pd.notna(s_lo) else float("nan")
        fn1k_pt = r["fn_per_1000"]
        if pd.notna(fn1k_pt) and pd.notna(fn1k_lo) and pd.notna(fn1k_hi):
            fn1k_str = f"{fn1k_pt:.1f} ({fn1k_lo:.1f}–{fn1k_hi:.1f})"
        elif pd.notna(fn1k_pt):
            fn1k_str = f"{fn1k_pt:.1f}"
        else:
            fn1k_str = "—"
        rows.append({
            "Cascade (Stage 1 × Stage 2)": fmt_pair(r["stage1"], r["stage2"]),
            "Sensitivity (95% CI)": fmt_ci(r["sensitivity"], r["sensitivity_ci_lo"], r["sensitivity_ci_hi"]),
            "Specificity (95% CI)": fmt_ci(r["specificity"], r["specificity_ci_lo"], r["specificity_ci_hi"]),
            "PPV (95% CI)": fmt_ci(r["ppv"], ppv_lo, ppv_hi) if pd.notna(r["ppv"]) else "—",
            "F1 (95% CI)": fmt_ci(r["f1"], f1_lo, f1_hi) if pd.notna(r["f1"]) else "—",
            "MCC (95% CI)": fmt_ci(r["mcc"], mcc_lo, mcc_hi) if pd.notna(r["mcc"]) else "—",
            "FN per 1,000 (95% CI)": fn1k_str,
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
    # Compute Wilson 95% CI per (architecture, category) cell from tp/fn
    def _cell_with_ci(tp, fn, sens):
        n = tp + fn
        if n <= 0:
            return "—"
        lo, hi = _wilson_ci(tp, n)
        return f"{sens:.3f} [{lo:.2f}–{hi:.2f}]"

    cs = cs.copy()
    cs["cell"] = cs.apply(lambda r: _cell_with_ci(r["tp"], r["fn"], r["sensitivity"]), axis=1)
    pivot = cs.pivot_table(index="hazard_category", columns="architecture",
                            values="cell", aggfunc="first")
    # Order columns by mean sensitivity across categories (descending = best-performing first)
    sens_pivot = cs.pivot_table(index="hazard_category", columns="architecture",
                                 values="sensitivity", aggfunc="first")
    col_means = sens_pivot.mean(axis=0).sort_values(ascending=False)
    pivot = pivot[col_means.index]
    pivot.columns = [ARCH_DISPLAY.get(c, c) for c in pivot.columns]
    n_per_cat = cs.groupby("hazard_category")["n_hazards"].first()
    pivot.insert(0, "N hazards", n_per_cat)
    out_df = pivot.reset_index().rename(columns={"hazard_category": "Hazard category"})
    # Humanize underscore-separated category names (e.g. "behavioral_other" → "Behavioral, other")
    out_df["Hazard category"] = out_df["Hazard category"].astype(str).str.replace("_", " ").str.capitalize()
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
    import math
    rows = []
    for _, r in df.iterrows():
        tp = r.get("tp", float("nan"))
        fp = r.get("fp", float("nan"))
        tn = r.get("tn", float("nan"))
        fn = r.get("fn", float("nan"))
        sens_lo, sens_hi = _wilson_ci(tp, tp + fn) if pd.notna(tp) and pd.notna(fn) else (float("nan"), float("nan"))
        spec_lo, spec_hi = _wilson_ci(tn, tn + fp) if pd.notna(tn) and pd.notna(fp) else (float("nan"), float("nan"))
        ppv_lo, ppv_hi = _wilson_ci(tp, tp + fp) if pd.notna(tp) and pd.notna(fp) else (float("nan"), float("nan"))
        # Parametric bootstrap CIs on F1 and MCC (1,000 iter, seed=42)
        f1_lo = f1_hi = mcc_lo = mcc_hi = float("nan")
        if all(pd.notna(x) for x in [tp, fp, tn, fn]):
            rng = np.random.RandomState(42)
            n_pos = int(tp + fn); n_neg = int(tn + fp)
            sens_p = tp / max(n_pos, 1); spec_p = tn / max(n_neg, 1)
            f1s, mccs = [], []
            for _b in range(1000):
                tp_b = rng.binomial(n_pos, sens_p); tn_b = rng.binomial(n_neg, spec_p)
                fn_b = n_pos - tp_b; fp_b = n_neg - tn_b
                df1 = 2*tp_b + fp_b + fn_b
                f1s.append((2*tp_b)/df1 if df1 > 0 else float("nan"))
                dm = math.sqrt((tp_b+fp_b)*(tp_b+fn_b)*(tn_b+fp_b)*(tn_b+fn_b))
                mccs.append((tp_b*tn_b - fp_b*fn_b)/dm if dm > 0 else float("nan"))
            f1_lo, f1_hi = np.nanpercentile(f1s, 2.5), np.nanpercentile(f1s, 97.5)
            mcc_lo, mcc_hi = np.nanpercentile(mccs, 2.5), np.nanpercentile(mccs, 97.5)
        rows.append({
            "AMC stack configuration": r["stack_label"],
            "Sensitivity (95% CI)": fmt_ci(r["sensitivity"], sens_lo, sens_hi),
            "Specificity (95% CI)": fmt_ci(r["specificity"], spec_lo, spec_hi),
            "PPV (95% CI)": fmt_ci(r["ppv"], ppv_lo, ppv_hi) if pd.notna(r["ppv"]) else "—",
            "F1 (95% CI)": fmt_ci(r["f1"], f1_lo, f1_hi) if pd.notna(r["f1"]) else "—",
            "MCC (95% CI)": fmt_ci(r["mcc"], mcc_lo, mcc_hi) if pd.notna(r["mcc"]) else "—",
            "Reached benchmark?": "Yes" if r["clinical_grade"] else "No",
        })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(csv_path, index=False)
    write_markdown_table(out_df, md_path)
    return csv_path


def render_table10_deployment_policies(results_dir: Path, output_dir: Path) -> Path:
    """Table 10: side-by-side operating point + clinician workload for the two
    actionable deployment policies vs the best single architecture, best ensemble,
    and best cascade. Reviewers ask 'what do the numbers actually look like under
    deployment?' — this table answers that, in alerts and missed hazards per 1000
    messages.
    """
    csv_path = output_dir / "table10_deployment_policies.csv"
    md_path = output_dir / "table10_deployment_policies.md"

    N_HAZ = 165
    N_BEN = 1835
    N_TOTAL = N_HAZ + N_BEN

    def from_sens_spec(label: str, sens: float, spec: float, note: str = "") -> dict:
        tp = sens * N_HAZ; fp = (1 - spec) * N_BEN
        fn = N_HAZ - tp; tn = N_BEN - fp
        alerts_per_1000 = 1000 * (tp + fp) / N_TOTAL
        misses_per_1000 = 1000 * fn / N_TOTAL
        caught_per_1000 = 1000 * tp / N_TOTAL
        ppv = tp / (tp + fp) if (tp + fp) else float("nan")
        sens_lo, sens_hi = _wilson_ci(tp, tp + fn)
        spec_lo, spec_hi = _wilson_ci(tn, tn + fp)
        ppv_lo, ppv_hi = _wilson_ci(tp, tp + fp)
        # Wilson CIs on per-1,000 rates: alerts and caught are proportions of N_TOTAL
        # (k/N) × 1000; missed is FN/N_TOTAL × 1000. Each uses Wilson on a binomial.
        n_alerts = int(round(tp + fp))
        n_caught = int(round(tp))
        n_missed = int(round(fn))
        a_lo, a_hi = _wilson_ci(n_alerts, N_TOTAL)
        c_lo, c_hi = _wilson_ci(n_caught, N_TOTAL)
        m_lo, m_hi = _wilson_ci(n_missed, N_TOTAL)
        alerts_str = f"{alerts_per_1000:.0f} ({a_lo*1000:.0f}–{a_hi*1000:.0f})"
        caught_str = f"{caught_per_1000:.1f} ({c_lo*1000:.1f}–{c_hi*1000:.1f})"
        missed_str = f"{misses_per_1000:.1f} ({m_lo*1000:.1f}–{m_hi*1000:.1f})"
        return {
            "Configuration": label,
            "Sensitivity (95% CI)": fmt_ci(sens, sens_lo, sens_hi),
            "Specificity (95% CI)": fmt_ci(spec, spec_lo, spec_hi),
            "PPV (95% CI)": fmt_ci(ppv, ppv_lo, ppv_hi),
            "Alerts per 1,000 messages (95% CI; clinician review load)": alerts_str,
            "Hazards caught per 1,000 messages (95% CI)": caught_str,
            "Hazards missed per 1,000 messages (95% CI)": missed_str,
            "Note": note,
        }

    rows = []

    # --- Single-architecture references (default thresholds) ---
    m_path = results_dir / "metrics_canonical.csv"
    if m_path.exists():
        m = pd.read_csv(m_path)
        rw = m[m["dataset"] == "realworld_n2000"]
        # Best balanced (highest min(sens, spec)) single architecture
        if not rw.empty:
            rw = rw.copy()
            rw["balanced"] = rw[["sensitivity", "specificity"]].min(axis=1)
            top_bal = rw.loc[rw["balanced"].idxmax()]
            rows.append(from_sens_spec(
                f"Best balanced single architecture ({ARCH_DISPLAY.get(top_bal['architecture'], top_bal['architecture'])})",
                float(top_bal["sensitivity"]),
                float(top_bal["specificity"]),
                "default threshold; baseline reference",
            ))
            # Best single-architecture sensitivity (the high-recall reference)
            top_sens = rw.loc[rw["sensitivity"].idxmax()]
            rows.append(from_sens_spec(
                f"Highest-sensitivity single architecture ({ARCH_DISPLAY.get(top_sens['architecture'], top_sens['architecture'])})",
                float(top_sens["sensitivity"]),
                float(top_sens["specificity"]),
                "default threshold; reference for sens-floor analysis",
            ))

    # --- Best ensemble (balanced) ---
    en_path = results_dir / "ensemble_results.csv"
    if en_path.exists():
        en = pd.read_csv(en_path)
        if not en.empty:
            en = en.copy()
            en["balanced"] = en[["sensitivity", "specificity"]].min(axis=1)
            top = en.loc[en["balanced"].idxmax()]
            rows.append(from_sens_spec(
                f"Best balanced ensemble ({top['rule']})",
                float(top["sensitivity"]),
                float(top["specificity"]),
                "hard- or soft-voting; reference for ensemble closing-the-gap finding",
            ))

    # --- Best cascade (balanced) ---
    cas_path = results_dir / "cascade_matrix.csv"
    if cas_path.exists():
        cas = pd.read_csv(cas_path)
        rw_cas = cas[(cas["dataset"] == "realworld_n2000") & (cas["stage1"] < cas["stage2"])].copy()
        if not rw_cas.empty:
            rw_cas["balanced"] = rw_cas[["sensitivity", "specificity"]].min(axis=1)
            top = rw_cas.loc[rw_cas["balanced"].idxmax()]
            rows.append(from_sens_spec(
                f"Best balanced cascade ({ARCH_DISPLAY.get(top['stage1'], top['stage1'])} × {ARCH_DISPLAY.get(top['stage2'], top['stage2'])})",
                float(top["sensitivity"]),
                float(top["specificity"]),
                "two-stage AND-rule; reference for cascade closing-the-gap finding",
            ))

    # --- Policy A: sens-floor 0.85 winning configuration ---
    sf_path = results_dir / "deployment_grade_sens_floor.csv"
    if sf_path.exists():
        sf = pd.read_csv(sf_path)
        floor_85 = sf[sf["sens_floor"] == 0.85]
        if not floor_85.empty:
            r = floor_85.iloc[0]
            if pd.notna(r["max_specificity"]):
                rows.append(from_sens_spec(
                    f"**Policy A**: high-recall screen ({r['winning_label']})",
                    float(r["winning_sensitivity"]),
                    float(r["max_specificity"]),
                    "sens >= 0.85 floor; all flagged messages sent to clinician confirmation queue",
                ))

    # --- Policy B: disagreement-stratified 2+ flagging ---
    ag_path = results_dir / "deployment_grade_agreement.csv"
    if ag_path.exists():
        ag = pd.read_csv(ag_path)
        # Sum hazards/benigns across strata where policy is clinician_review or escalate
        review_strata = ag[ag["policy"].isin(["clinician_review", "escalate"])]
        autonomous_benign = ag[ag["policy"] == "no_action"]
        if not review_strata.empty:
            tp_b = int(review_strata["n_hazards"].sum())
            fp_b = int(review_strata["n_benigns"].sum())
            fn_b = int(autonomous_benign["n_hazards"].sum()) if not autonomous_benign.empty else 0
            tn_b = int(autonomous_benign["n_benigns"].sum()) if not autonomous_benign.empty else 0
            sens_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) else float("nan")
            spec_b = tn_b / (tn_b + fp_b) if (tn_b + fp_b) else float("nan")
            rows.append(from_sens_spec(
                "**Policy B**: disagreement-stratified triage (2 or more of 10 architectures flag, route to clinician review)",
                sens_b, spec_b,
                "0-1 architectures flagging: autonomous benign; 2 or more: clinician review; 7 or more: autonomous escalation",
            ))

    # Reference: autonomous (physician-unassisted) triage benchmark
    rows.insert(0, {
        "Configuration": "**Autonomous (physician-unassisted) triage benchmark**",
        "Sensitivity (95% CI)": "≥ 0.800",
        "Specificity (95% CI)": "≥ 0.800",
        "PPV (95% CI)": "—",
        "Alerts per 1,000 messages (95% CI; clinician review load)": "—",
        "Hazards caught per 1,000 messages (95% CI)": "—",
        "Hazards missed per 1,000 messages (95% CI)": "≤ 16.5",
        "Note": "Anchored to IDx-DR pivotal trial (sens 87.2%, spec 90.7%)",
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
    N_HAZ_RW = 165
    N_BEN_RW = 1835
    parts = []
    if sf_path.exists():
        sf = pd.read_csv(sf_path)
        parts.append("**Panel A — Sensitivity-floor target: max specificity achievable subject to sensitivity threshold**\n")
        # Build table
        a_rows = []
        for _, r in sf.iterrows():
            # Wilson 95% CI on winning sens and max specificity
            ws = r.get("winning_sensitivity", float("nan"))
            ms = r.get("max_specificity", float("nan"))
            if pd.notna(ws):
                tp = round(float(ws) * N_HAZ_RW)
                ws_lo, ws_hi = _wilson_ci(tp, N_HAZ_RW)
                ws_str = f"{ws:.3f} ({ws_lo:.3f}–{ws_hi:.3f})"
            else:
                ws_str = "—"
            if pd.notna(ms):
                tn = round(float(ms) * N_BEN_RW)
                ms_lo, ms_hi = _wilson_ci(tn, N_BEN_RW)
                ms_str = f"{ms:.3f} ({ms_lo:.3f}–{ms_hi:.3f})"
            else:
                ms_str = "—"
            a_rows.append({
                "Sensitivity floor": f"≥ {r['sens_floor']:.2f}",
                "N configurations meeting floor": int(r["n_configurations_meeting_floor"]) if pd.notna(r["n_configurations_meeting_floor"]) else 0,
                "Max specificity at floor (95% CI)": ms_str,
                "Winning configuration": (f"{r['winning_strategy_type']} / {r['winning_label']}" if pd.notna(r.get("winning_strategy_type")) else "—"),
                "Sens (winning) (95% CI)": ws_str,
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
            # Wilson 95% CI on hazard prevalence within stratum
            n_msgs = int(r["n_messages"]) if pd.notna(r["n_messages"]) else 0
            n_haz = int(r["n_hazards"]) if pd.notna(r["n_hazards"]) else 0
            if n_msgs > 0:
                pr_lo, pr_hi = _wilson_ci(n_haz, n_msgs)
                pr_str = f"{r['hazard_prevalence_in_stratum']:.3f} ({pr_lo:.3f}–{pr_hi:.3f})"
            else:
                pr_str = "—"
            b_rows.append({
                "Stratum (n architectures flagging)": r["stratum"],
                "Recommended policy": r["policy"].replace("_", " "),
                "N messages": n_msgs,
                "N hazards (true)": n_haz,
                "Hazard prevalence in stratum (95% CI)": pr_str,
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


def render_tableS5_mcnemar_matrix(results_dir: Path, output_dir: Path) -> Path:
    """Table S5: Full pairwise McNemar matrix with chi-square, p-value, Hochberg significance."""
    p = results_dir / "mcnemar_matrix.csv"
    csv_path = output_dir / "tableS5_mcnemar_matrix.csv"
    md_path = output_dir / "tableS5_mcnemar_matrix.md"
    if not p.exists():
        pd.DataFrame().to_csv(csv_path, index=False)
        md_path.write_text("_[Table S5 not yet rendered]_\n")
        return csv_path
    df = pd.read_csv(p)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Architecture A": ARCH_DISPLAY.get(r.get("arch_a", ""), r.get("arch_a", "")),
            "Architecture B": ARCH_DISPLAY.get(r.get("arch_b", ""), r.get("arch_b", "")),
            "Chi-square": f"{r['chi2']:.3f}" if pd.notna(r.get("chi2")) else "—",
            "Raw p-value": f"{r['p_raw']:.4f}" if pd.notna(r.get("p_raw")) else "—",
            "Hochberg significant (α=0.05)": "Yes" if r.get("sig_hochberg", False) else "No",
        })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(csv_path, index=False)
    write_markdown_table(out_df, md_path)
    return csv_path


def render_tableS6_thresholds(results_dir: Path, output_dir: Path) -> Path:
    """Table S6: Calibrated decision threshold per architecture."""
    import json
    p = results_dir / "thresholds.json"
    csv_path = output_dir / "tableS6_thresholds.csv"
    md_path = output_dir / "tableS6_thresholds.md"
    if not p.exists():
        pd.DataFrame().to_csv(csv_path, index=False)
        md_path.write_text("_[Table S6 not yet rendered]_\n")
        return csv_path
    th = json.loads(p.read_text())
    rows = [{"Architecture": ARCH_DISPLAY.get(k, k), "Calibrated decision threshold": f"{v:.4f}"}
            for k, v in th.items()]
    out_df = pd.DataFrame(rows)
    out_df.to_csv(csv_path, index=False)
    write_markdown_table(out_df, md_path)
    return csv_path


def render_figureS1_operating_curves(results_dir: Path, output_dir: Path) -> Path:
    """Figure S1: ROC overlays for all calibrated-probability architectures."""
    import matplotlib.pyplot as plt
    curves_path = results_dir / "operating_curves.csv"
    out_path = output_dir / "figureS1_operating_curves.png"
    if not curves_path.exists():
        return out_path
    df = pd.read_csv(curves_path)
    fig, ax = plt.subplots(figsize=(8, 7))
    for arch in df["architecture"].unique():
        sub = df[df["architecture"] == arch]
        ax.plot(1 - sub["specificity"], sub["sensitivity"],
                label=ARCH_DISPLAY.get(arch, arch), linewidth=1.5)
    ax.plot([0, 1], [0, 1], "k:", alpha=0.4, label="Chance")
    ax.set_xlabel("False positive rate (1 − specificity)")
    ax.set_ylabel("Sensitivity")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Receiver operating characteristic curves: architectures with calibrated probability output")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def render_figureS2_calibration(results_dir: Path, predictions_dir: Path, output_dir: Path) -> Path:
    """Figure S2: Calibration plots per architecture (predicted vs observed in deciles)."""
    import matplotlib.pyplot as plt
    out_path = output_dir / "figureS2_calibration.png"
    if not predictions_dir.exists():
        return out_path
    archs_with_proba = []
    for csv_path in sorted(predictions_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        rw = df[df["dataset"] == "realworld_n2000"].copy()
        if rw.empty: continue
        rw["pred_proba"] = pd.to_numeric(rw["pred_proba"], errors="coerce")
        if rw["pred_proba"].isna().all(): continue
        arch = rw["architecture"].iloc[0]
        archs_with_proba.append((arch, rw["pred_proba"].values, rw["true_hazard"].astype(int).values))

    if not archs_with_proba:
        plt.figure(figsize=(6, 4)); plt.text(0.5, 0.5, "No calibrated probabilities available",
                                              ha="center", va="center"); plt.axis("off")
        plt.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close()
        return out_path

    n = len(archs_with_proba)
    ncols = min(3, n); nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 4*nrows), squeeze=False)
    for ax_i, (arch, proba, true) in enumerate(archs_with_proba):
        ax = axes[ax_i // ncols][ax_i % ncols]
        # Decile bins
        bins = np.quantile(proba, np.linspace(0, 1, 11))
        bins = np.unique(bins)  # collapse duplicates
        if len(bins) < 3:
            ax.text(0.5, 0.5, f"{ARCH_DISPLAY.get(arch, arch)}\nFew distinct\nprobability values",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_title(ARCH_DISPLAY.get(arch, arch), fontsize=10)
            continue
        idx = np.digitize(proba, bins[1:-1])
        x, y = [], []
        for b in range(len(bins) - 1):
            mask = idx == b
            if mask.sum() < 5: continue
            x.append(proba[mask].mean())
            y.append(true[mask].mean())
        ax.plot([0, 1], [0, 1], "k:", alpha=0.4)
        ax.scatter(x, y, s=40, color="steelblue")
        ax.plot(x, y, "-", color="steelblue", alpha=0.5)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(ARCH_DISPLAY.get(arch, arch), fontsize=10)
        ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed hazard fraction")
        ax.grid(alpha=0.3)
    for empty in range(len(archs_with_proba), nrows*ncols):
        axes[empty // ncols][empty % ncols].axis("off")
    plt.suptitle("Calibration: predicted probability versus observed hazard fraction (deciles)",
                 fontsize=12, y=1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close()
    return out_path


def render_figureS3_category_sensitivity(results_dir: Path, output_dir: Path) -> Path:
    """Figure S3: Per-category sensitivity bar chart."""
    import matplotlib.pyplot as plt
    cs_path = results_dir / "category_stratification.csv"
    out_path = output_dir / "figureS3_category_sensitivity.png"
    if not cs_path.exists():
        return out_path
    cs = pd.read_csv(cs_path)
    pivot = cs.pivot_table(index="hazard_category", columns="architecture",
                            values="sensitivity", aggfunc="first")
    pivot.columns = [ARCH_DISPLAY.get(c, c) for c in pivot.columns]
    fig, ax = plt.subplots(figsize=(13, 6))
    pivot.plot(kind="bar", ax=ax, width=0.85, edgecolor="black", linewidth=0.3, legend=False)
    ax.axhline(0.80, color="darkgreen", linestyle="--", linewidth=1.2, alpha=0.7,
               label="Clinical-grade sensitivity floor (0.80)")
    ax.set_xlabel("Hazard category")
    ax.set_ylabel("Sensitivity")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-architecture sensitivity by hazard category (real-world test set)")
    # Move legend outside the plot area to the right so it doesn't cover any bars
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=8, framealpha=1.0, borderaxespad=0)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close()
    return out_path


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
    N_HAZ = 165
    N_BEN = 1835

    def _ci_pair(sens, spec):
        """Return (sens_ci_str, spec_ci_str) at the given sens/spec on the n=2000 test set."""
        tp = round(sens * N_HAZ); fn = N_HAZ - tp
        tn = round(spec * N_BEN); fp = N_BEN - tn
        sens_lo, sens_hi = _wilson_ci(tp, tp + fn)
        spec_lo, spec_hi = _wilson_ci(tn, tn + fp)
        return fmt_ci(sens, sens_lo, sens_hi), fmt_ci(spec, spec_lo, spec_hi)

    def _f1_mcc_boot_at_op(sens, spec, n_iter=1000, seed=42):
        """Parametric bootstrap 95% CI for F1 and MCC at a given (sens, spec) operating point."""
        import math
        if any(pd.isna(x) for x in [sens, spec]):
            return ((float("nan"), float("nan")), (float("nan"), float("nan")))
        rng = np.random.RandomState(seed)
        f1s, mccs = [], []
        for _ in range(n_iter):
            tp_b = rng.binomial(N_HAZ, sens)
            tn_b = rng.binomial(N_BEN, spec)
            fn_b = N_HAZ - tp_b
            fp_b = N_BEN - tn_b
            denom_f1 = 2 * tp_b + fp_b + fn_b
            f1_b = (2 * tp_b) / denom_f1 if denom_f1 > 0 else float("nan")
            denom_mcc = math.sqrt((tp_b + fp_b) * (tp_b + fn_b) * (tn_b + fp_b) * (tn_b + fn_b))
            mcc_b = (tp_b * tn_b - fp_b * fn_b) / denom_mcc if denom_mcc > 0 else float("nan")
            f1s.append(f1_b); mccs.append(mcc_b)
        return ((np.nanpercentile(f1s, 2.5), np.nanpercentile(f1s, 97.5)),
                (np.nanpercentile(mccs, 2.5), np.nanpercentile(mccs, 97.5)))

    rows = []
    for _, r in th.iterrows():
        f1_sens_ci, f1_spec_ci = _ci_pair(r["f1_max_sens"], r["f1_max_spec"])
        mcc_sens_ci, mcc_spec_ci = _ci_pair(r["mcc_max_sens"], r["mcc_max_spec"])
        # Bootstrap CIs on F1-max and MCC-max at their respective operating points
        (f1_lo, f1_hi), _ = _f1_mcc_boot_at_op(r["f1_max_sens"], r["f1_max_spec"])
        _, (mcc_lo, mcc_hi) = _f1_mcc_boot_at_op(r["mcc_max_sens"], r["mcc_max_spec"])
        # Wilson CIs on closest-to-benchmark operating point
        cp_sens = r.get("closest_to_clinical_grade_sens", float("nan"))
        cp_spec = r.get("closest_to_clinical_grade_spec", float("nan"))
        if pd.notna(cp_sens) and pd.notna(cp_spec):
            cp_sens_ci, cp_spec_ci = _ci_pair(cp_sens, cp_spec)
            cp_str = f"{cp_sens:.3f} / {cp_spec:.3f} (sens CI {cp_sens_ci.split('(')[-1].rstrip(')')}; spec CI {cp_spec_ci.split('(')[-1].rstrip(')')})"
        else:
            cp_str = "—"
        rows.append({
            "Architecture": ARCH_DISPLAY.get(r["architecture"], r["architecture"]),
            "F1-max (95% CI)": fmt_ci(r["f1_max"], f1_lo, f1_hi),
            "Sens at F1-max (95% CI)": f1_sens_ci,
            "Spec at F1-max (95% CI)": f1_spec_ci,
            "MCC-max (95% CI)": fmt_ci(r["mcc_max"], mcc_lo, mcc_hi),
            "Sens at MCC-max (95% CI)": mcc_sens_ci,
            "Spec at MCC-max (95% CI)": mcc_spec_ci,
            "Reached autonomous benchmark?": "Yes" if r["clinical_grade_reachable"] else "No",
            "Sens/Spec at closest point (95% CI)": cp_str,
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

    def _boot_f1_mcc_local(tp, fp, tn, fn, n_iter=500, seed=42):
        """Lightweight parametric bootstrap for F1 and MCC on a 2x2 table."""
        import math
        if any(pd.isna(x) for x in [tp, fp, tn, fn]):
            return ((float("nan"), float("nan")), (float("nan"), float("nan")))
        rng = np.random.RandomState(seed)
        n_pos = int(tp + fn); n_neg = int(tn + fp)
        sens_p = tp / max(n_pos, 1); spec_p = tn / max(n_neg, 1)
        f1s, mccs = [], []
        for _ in range(n_iter):
            tp_b = rng.binomial(n_pos, sens_p)
            tn_b = rng.binomial(n_neg, spec_p)
            fn_b = n_pos - tp_b; fp_b = n_neg - tn_b
            df1 = 2*tp_b + fp_b + fn_b
            f1 = (2*tp_b)/df1 if df1 > 0 else float("nan")
            dm = math.sqrt((tp_b+fp_b)*(tp_b+fn_b)*(tn_b+fp_b)*(tn_b+fn_b))
            mcc = (tp_b*tn_b - fp_b*fn_b)/dm if dm > 0 else float("nan")
            f1s.append(f1); mccs.append(mcc)
        return ((np.nanpercentile(f1s, 2.5), np.nanpercentile(f1s, 97.5)),
                (np.nanpercentile(mccs, 2.5), np.nanpercentile(mccs, 97.5)))

    rows = []
    for _, r in cm.iterrows():
        tp = r.get("tp", float("nan")); fp = r.get("fp", float("nan"))
        tn = r.get("tn", float("nan")); fn = r.get("fn", float("nan"))
        sens_lo, sens_hi = _wilson_ci(tp, tp + fn) if pd.notna(tp) and pd.notna(fn) else (float("nan"), float("nan"))
        spec_lo, spec_hi = _wilson_ci(tn, tn + fp) if pd.notna(tn) and pd.notna(fp) else (float("nan"), float("nan"))
        ppv_lo, ppv_hi = _wilson_ci(tp, tp + fp) if pd.notna(tp) and pd.notna(fp) else (float("nan"), float("nan"))
        (f1_lo, f1_hi), (mcc_lo, mcc_hi) = _boot_f1_mcc_local(tp, fp, tn, fn)
        rows.append({
            "Cascade": fmt_pair(r["stage1"], r["stage2"]),
            "Sens (95% CI)": fmt_ci(r["sensitivity"], sens_lo, sens_hi),
            "Spec (95% CI)": fmt_ci(r["specificity"], spec_lo, spec_hi),
            "PPV (95% CI)": fmt_ci(r["ppv"], ppv_lo, ppv_hi) if pd.notna(r["ppv"]) else "—",
            "F1 (95% CI)": fmt_ci(r["f1"], f1_lo, f1_hi) if pd.notna(r["f1"]) else "—",
            "MCC (95% CI)": fmt_ci(r["mcc"], mcc_lo, mcc_hi) if pd.notna(r["mcc"]) else "—",
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
    print("[render] Table 10 (deployment policies side-by-side)")
    out["table10"] = render_table10_deployment_policies(results_dir, tables_dir)
    print(f"  → {out['table10']}")
    print("[render] Table S1 (physician holdout)")
    out["tableS1"] = render_tableS1_physician_metrics(metrics_df, tables_dir, predictions_dir=predictions_dir)
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
    print("[render] Table S5 (McNemar matrix)")
    out["tableS5"] = render_tableS5_mcnemar_matrix(results_dir, tables_dir)
    print(f"  → {out['tableS5']}")
    print("[render] Table S6 (thresholds)")
    out["tableS6"] = render_tableS6_thresholds(results_dir, tables_dir)
    print(f"  → {out['tableS6']}")
    print("[render] Figure S1 (operating curves)")
    out["figureS1"] = render_figureS1_operating_curves(results_dir, figures_dir)
    print(f"  → {out['figureS1']}")
    print("[render] Figure S2 (calibration)")
    out["figureS2"] = render_figureS2_calibration(results_dir, predictions_dir, figures_dir)
    print(f"  → {out['figureS2']}")
    print("[render] Figure S3 (category sensitivity bar chart)")
    out["figureS3"] = render_figureS3_category_sensitivity(results_dir, figures_dir)
    print(f"  → {out['figureS3']}")
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
