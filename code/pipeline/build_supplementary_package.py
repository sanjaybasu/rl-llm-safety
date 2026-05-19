"""Build the BMC MIDM Multimedia Appendix 2 — per-message predictions release.

Creates a ZIP archive containing:
- One CSV per architecture × dataset (no patient text)
- README.md documenting the schema and intended use
- A run_manifest.json with run_id, SHA-256 checksums, and timestamp
- The canonical metrics CSVs (results/) for reviewer convenience

This is HIPAA-compliant: no patient text or PHI in the archive.
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path


SUPPLEMENTARY_README = """# Multimedia Appendix 2 — Per-message Predictions

## Contents

This archive contains per-message predictions for every architecture × test record evaluated in the manuscript "AI safety evaluation in an underrepresented Medicaid population: real-world performance of clinical decision support and frontier language models on patient messaging triage" (BMC Medical Informatics and Decision Making, submitted 2026).

## Schema

Each CSV file contains one row per (architecture, message_id) pair with the following columns:

| Column | Type | Description |
|---|---|---|
| `message_id` | string | Opaque CUID; does not link to any patient identifier |
| `dataset` | enum | `realworld_n2000` or `physician_n41` |
| `true_hazard` | 0/1 | Physician-consensus ground truth binary hazard label |
| `true_action` | 1-8 | Physician-consensus ground truth action level (8-point scale) |
| `hazard_category` | string | Physician-consensus ground truth hazard category (23-category taxonomy or `benign`) |
| `pred_proba` | float \| empty | Continuous hazard probability where the architecture exposes calibrated scores (empty for LLMs) |
| `pred_hazard` | 0/1 | Binary hazard prediction at the architecture's calibrated operating threshold |
| `pred_action` | 1-8 | Predicted action level on the 8-point action scale |
| `threshold_used` | float | The calibrated operating threshold used for binarization |
| `architecture` | string | Architecture identifier (e.g., `cql_sens_opt`, `xgboost_sbert`, `claude_opus_4_7_safety`) |
| `model_version` | string | Specific model snapshot (e.g., `claude-opus-4-7`, `XGBoost+SBERT`) |
| `run_id` | UUID | Links every row across architectures to a single canonical pipeline run |
| `inference_time_s` | float | Per-message inference time in seconds |

## Reproducibility

All predictions in this archive derive from a single Modal-orchestrated pipeline run with deterministic seed=42 throughout. The associated code is at https://github.com/sanjaybasu/rl-llm-safety. The pipeline produces these CSVs in one pass; the metric tables and figures in the manuscript main text are rendered from these same files via `code/pipeline/manuscript_renderer.py`.

A reviewer who wishes to independently verify any manuscript value can:

1. Compute sensitivity and specificity at the documented threshold for any architecture × dataset row by counting `pred_hazard` against `true_hazard`.
2. Construct the full receiver operating characteristic curve for any architecture with calibrated `pred_proba` values and verify monotonicity.
3. Compute McNemar discordant-pair statistics between any two architectures on the hazard-positive cases.
4. Compute parametric bootstrap confidence intervals on cross-set (physician → real-world) sensitivity differences using seed=42 and 10,000 iterations.

The `results/` subfolder contains the canonical metrics CSVs computed by the pipeline:

- `metrics_canonical.csv` — Wilson and bootstrap CIs for every architecture × dataset
- `mcnemar_matrix.csv` — Full pairwise McNemar matrix with Hochberg-adjusted significance
- `delta_bootstrap_canonical.csv` — Δ sensitivity (physician → real-world) with bootstrap CIs
- `operating_curves.csv` — Per-architecture full ROC curves
- `thresholds.json` — Per-architecture calibrated thresholds

## Privacy and ethics

The released archive does NOT contain patient message text. Patient messages were generated under a Medicaid managed care entity's data-use agreement and remain governed by that agreement. Bona-fide academic researchers may request access to the message text by contacting the corresponding author (sanjay.basu@ucsf.edu) with a brief description of the proposed reanalysis and confirmation of an applicable IRB or equivalent data-use determination.

The `message_id` values in this archive are opaque CUIDs generated specifically for this evaluation and do not link to any patient identifier or external record.

## Citation

If you use these files in your own research, please cite:

> Basu S. AI safety evaluation in an underrepresented Medicaid population: real-world performance of clinical decision support and frontier language models on patient messaging triage. BMC Medical Informatics and Decision Making, 2026.

## Contact

Corresponding author: Sanjay Basu, MD, PhD (sanjay.basu@ucsf.edu)
Waymark and University of California, San Francisco
"""


def build_archive(
    predictions_dir: Path,
    results_dir: Path,
    output_path: Path,
    run_manifest_path: Path,
) -> Path:
    """Build the Multimedia Appendix 2 ZIP archive."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # README
        zf.writestr("README.md", SUPPLEMENTARY_README)

        # Per-message prediction CSVs
        for csv_path in sorted(Path(predictions_dir).glob("*.csv")):
            arc_name = f"per_message_predictions/{csv_path.name}"
            zf.write(csv_path, arc_name)
            print(f"  added {arc_name}")

        # Canonical metrics CSVs
        for csv_path in sorted(Path(results_dir).glob("*.csv")):
            arc_name = f"results/{csv_path.name}"
            zf.write(csv_path, arc_name)
            print(f"  added {arc_name}")
        for json_path in sorted(Path(results_dir).glob("*.json")):
            arc_name = f"results/{json_path.name}"
            zf.write(json_path, arc_name)
            print(f"  added {arc_name}")

        # Run manifest
        if run_manifest_path.exists():
            zf.write(run_manifest_path, "run_manifest.json")
            print(f"  added run_manifest.json")

    print(f"\nArchive written to {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_archive(args.predictions_dir, args.results_dir, args.output, args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
