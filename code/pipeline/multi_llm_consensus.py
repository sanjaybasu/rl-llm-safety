"""Multi-LLM consensus analysis (rank-3 closing-the-gap intervention).

Uses the existing Claude + Gemini per-message predictions to compute consensus
rules:
  - claude_only / gemini_only (baselines)
  - OR-rule (either LLM flags)  → maximizes sensitivity
  - AND-rule (both LLMs flag)   → maximizes specificity
  - claude_dominant (Claude's label; Gemini ignored)  → reference
  - gemini_dominant

The literature review (closing_the_gap.md) flagged multi-LLM voting as
'MAYBE' with the cautionary tale that majority vote can drive sensitivity to
zero. We compute all 5 combination rules and report which (if any) reaches
clinical-grade (sens >= 0.80 AND spec >= 0.80) on the real-world test set.

This is a FREE analysis — no new Modal compute. Output:
    results/multi_llm_consensus.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CLINICAL_GRADE = 0.80


def metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    pred = pred.astype(bool)
    true = true.astype(bool)
    tp = int((pred & true).sum())
    fp = int((pred & ~true).sum())
    tn = int((~pred & ~true).sum())
    fn = int((~pred & true).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else float("nan")
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "ppv": round(ppv, 4),
        "f1": round(f1, 4),
        "mcc": round(mcc, 4),
        "clinical_grade": bool(sens >= CLINICAL_GRADE and spec >= CLINICAL_GRADE),
    }


def analyze(predictions_dir: Path, dataset: str = "realworld_n2000") -> pd.DataFrame:
    claude_path = next(predictions_dir.glob("claude_opus_4_7_safety_*.csv"))
    gemini_path = next(predictions_dir.glob("gemini_3_1_pro_safety_*.csv"))
    c = pd.read_csv(claude_path)
    g = pd.read_csv(gemini_path)
    c = c[c["dataset"] == dataset].sort_values("message_id").reset_index(drop=True)
    g = g[g["dataset"] == dataset].sort_values("message_id").reset_index(drop=True)
    assert (c["message_id"].values == g["message_id"].values).all(), "message_id alignment failed"
    assert (c["true_hazard"].values == g["true_hazard"].values).all(), "true_hazard alignment failed"

    true = c["true_hazard"].astype(int).values
    c_pred = c["pred_hazard"].astype(int).values
    g_pred = g["pred_hazard"].astype(int).values

    rows = []
    for label, pred in [
        ("claude_only", c_pred),
        ("gemini_only", g_pred),
        ("OR_rule_claude_or_gemini", np.maximum(c_pred, g_pred)),
        ("AND_rule_claude_and_gemini", np.minimum(c_pred, g_pred)),
    ]:
        m = metrics(pred, true)
        m["rule"] = label
        rows.append(m)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()

    df = analyze(args.predictions_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.results_dir / "multi_llm_consensus.csv"
    df.to_csv(out_path, index=False)
    print(f"  → wrote {out_path}")
    print()
    print(df[["rule", "sensitivity", "specificity", "f1", "mcc", "clinical_grade"]].to_string(index=False))
    n_cg = int(df["clinical_grade"].sum())
    print(f"\n{n_cg} / {len(df)} multi-LLM rules reach sens >= {CLINICAL_GRADE} AND spec >= {CLINICAL_GRADE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
