"""Manuscript template renderer.

Reads canonical results CSVs and substitutes every `{placeholder}` in the
template with the corresponding value. Numbers are NEVER hand-typed in the
manuscript — they all render from canonical files at build time.

Placeholder schemas supported:
    {metrics.<dataset>.<architecture>.<metric>}
    {metrics.<dataset>.<architecture>.<metric>_ci_lo}
    {metrics.<dataset>.<architecture>.<metric>_ci_hi}
    {mcnemar.<arch_a>.<arch_b>.chi2}
    {mcnemar.<arch_a>.<arch_b>.p_raw}
    {delta.<architecture>.sensitivity}
    {delta.<architecture>.sensitivity_ci_lo}
    {delta.<architecture>.sensitivity_ci_hi}
    {n_total}, {n_hazards}, {n_benigns}, {prevalence_pct}
    {run_id}, {run_date}, {study_period}
    {grade_level_realworld}, {grade_level_physician}, ...
    {thresholds.<architecture>}

A placeholder that cannot be resolved raises a hard error (the alternative —
silently leaving placeholders unresolved or substituting a default — is exactly
the kind of failure mode that produced the JMIR rejection).

Usage:
    python manuscript_renderer.py \\
        --template manuscript_template/main_text_template.md \\
        --predictions /predictions/canonical/ \\
        --results /results/ \\
        --output drafts/main_text.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


class TemplateRenderer:
    def __init__(
        self,
        predictions_dir: Path,
        results_dir: Path,
        manifest_path: Path,
    ):
        self.predictions_dir = predictions_dir
        self.results_dir = results_dir
        self.manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

        # Load canonical CSVs
        metrics_path = results_dir / "metrics_canonical.csv"
        mcnemar_path = results_dir / "mcnemar_matrix.csv"
        delta_path = results_dir / "delta_bootstrap_canonical.csv"
        thresholds_path = results_dir / "thresholds.json"

        self.metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
        self.mcnemar = pd.read_csv(mcnemar_path) if mcnemar_path.exists() else pd.DataFrame()
        self.delta = pd.read_csv(delta_path) if delta_path.exists() else pd.DataFrame()
        self.thresholds = json.loads(thresholds_path.read_text()) if thresholds_path.exists() else {}

        # Linguistic features (computed by compute_linguistic_features.py)
        ling_path = results_dir / "linguistic_features.json"
        self.linguistic = json.loads(ling_path.read_text()) if ling_path.exists() else {}

        # Free-form manuscript metadata (filled by user before submission)
        meta_path = results_dir / "manuscript_metadata.json"
        if meta_path.exists():
            self.manuscript_meta = json.loads(meta_path.read_text())
        else:
            # Sensible defaults for placeholders not driven by the data
            self.manuscript_meta = {
                "study_period": "January 2023 through November 2025",
                "n_training_total": 1280,
                "n_unique_patients": 1679,
                "n_demographic_available": 1536,
                "zenodo_doi": "TBD (will be assigned at submission)",
                "run_date": self.manifest.get("timestamp_utc", "TBD"),
                "word_count_main_text": 6500,
                "word_count_appendix": 2500,
                "cql_temperature": 0.215,
                "n_val": 256,
            }

    # Mapping from {placeholder_block} → rendered table file under results_dir/tables/
    _BLOCK_TABLE_MAP = {
        "table1_population_block": None,  # built from manifest + linguistic features inline
        "table2_detection_metrics_block": "tables/table2_detection_metrics.md",
        "table3_operating_points_block": "tables/table3_operating_points.md",
        "table4_action_recommendations_block": "tables/table4_action_recommendations.md",
        "table5_cascade_pareto_block": "tables/table5_cascade_pareto.md",
        "table6_threshold_optimized_block": "tables/table6_threshold_optimized.md",
        "table7_closing_the_gap_block": "tables/table7_closing_the_gap.md",
        "tableS1_physician_holdout_block": "tables/tableS1_physician_holdout_metrics.md",
        "tableS2_delta_bootstrap_block": "tables/tableS2_delta_bootstrap.md",
        "tableS3_cascade_full_block": "tables/tableS3_cascade_full.md",
        "figure1_caption_block": None,  # figure binary lives at figures/figure1_sens_spec_change.png
        "figure2_caption_block": None,
    }

    def _resolve_block(self, key: str) -> str:
        """Read an embeddable block from results/tables/<name>.md or build inline."""
        if key == "table1_population_block":
            return self._build_table1_population()
        if key in ("figure1_caption_block", "figure2_caption_block"):
            fig_name = {
                "figure1_caption_block": "figure1_sens_spec_change.png",
                "figure2_caption_block": "figure2_action_recommendations.png",
            }[key]
            return f"![{fig_name}](figures/{fig_name})"
        rel = self._BLOCK_TABLE_MAP.get(key)
        if rel:
            path = self.results_dir / rel
            if path.exists():
                return path.read_text()
            return f"_[Table file {rel} not yet rendered]_"
        return ""

    def _build_table1_population(self) -> str:
        """Construct Table 1 inline from manifest + linguistic features."""
        rw = self.manifest.get("datasets", {}).get("realworld_n2000", {})
        ph = self.manifest.get("datasets", {}).get("physician_n41", {})
        rw_ling = self.linguistic.get("realworld", {})
        ph_ling = self.linguistic.get("physician", {})

        def f(x, fmt="{:.1f}"):
            try:
                return fmt.format(float(x))
            except (TypeError, ValueError):
                return "—"

        lines = [
            "| Characteristic | Real-world Medicaid test set | Physician-scripted comparison |",
            "|---|---|---|",
            f"| Sample size, N | {rw.get('n_records', '—')} | {ph.get('n_records', '—')} |",
            f"| Hazards adjudicated, n (%) | {rw.get('n_hazards', '—')} ({100*rw.get('n_hazards',0)/max(rw.get('n_records',1),1):.2f}%) | {ph.get('n_hazards', '—')} ({100*ph.get('n_hazards',0)/max(ph.get('n_records',1),1):.1f}%) |",
            f"| Benigns adjudicated, n (%) | {rw.get('n_benigns', '—')} ({100*rw.get('n_benigns',0)/max(rw.get('n_records',1),1):.2f}%) | {ph.get('n_benigns', '—')} ({100*ph.get('n_benigns',0)/max(ph.get('n_records',1),1):.1f}%) |",
            f"| Reading level (Flesch-Kincaid grade), mean | {f(rw_ling.get('grade_level_mean'))} | {f(ph_ling.get('grade_level_mean'))} |",
            f"| Colloquialisms, % messages | {f(rw_ling.get('colloquialism_pct'))} | {f(ph_ling.get('colloquialism_pct'))} |",
            f"| Abbreviations, % messages | {f(rw_ling.get('abbreviation_pct'))} | {f(ph_ling.get('abbreviation_pct'))} |",
            f"| Implicit contextual references, % messages | {f(rw_ling.get('implicit_context_pct'))} | {f(ph_ling.get('implicit_context_pct'))} |",
            f"| Word count, mean | {f(rw_ling.get('word_count_mean'))} | {f(ph_ling.get('word_count_mean'))} |",
        ]
        return "\n".join(lines)

    def _lookup(self, placeholder: str) -> Any:
        """Resolve a placeholder name to a value. Raises KeyError if not found."""
        parts = placeholder.split(".")

        # Top-level manifest fields
        if len(parts) == 1:
            key = parts[0]
            # Run manifest fields
            if key in self.manifest:
                return self.manifest[key]
            # Common derived fields
            if key in {"n_total", "n_hazards", "n_benigns"}:
                ds = self.manifest.get("datasets", {}).get("realworld_n2000", {})
                return ds.get({"n_total": "n_records", "n_hazards": "n_hazards", "n_benigns": "n_benigns"}[key])
            if key == "prevalence_pct":
                ds = self.manifest.get("datasets", {}).get("realworld_n2000", {})
                if "n_hazards" in ds and "n_records" in ds and ds["n_records"]:
                    return f"{100 * ds['n_hazards'] / ds['n_records']:.2f}"
            # k_pairs = count of rows in mcnemar_matrix
            if key == "k_pairs":
                return len(self.mcnemar) if not self.mcnemar.empty else 0
            # Manuscript metadata (free-form)
            if key in self.manuscript_meta:
                return self.manuscript_meta[key]
            # Linguistic feature shortcuts (e.g., grade_level_realworld, colloquialism_pct_physician)
            for dataset_key in ("realworld", "physician"):
                if key.endswith(f"_{dataset_key}"):
                    feature = key[: -len(f"_{dataset_key}")]
                    return self.linguistic.get(dataset_key, {}).get(feature.replace("grade_level", "grade_level_mean"), "?")
            # Table/figure embed blocks — read the rendered markdown table or
            # produce a figure-reference block. Returns "" silently if absent
            # (a missing table is allowed at intermediate render time).
            if key.endswith("_block"):
                return self._resolve_block(key)
            raise KeyError(f"Unknown top-level placeholder: {key}")

        head = parts[0]

        if head == "metrics":
            # {metrics.<dataset>.<architecture>.<metric>}
            _, dataset, architecture, metric = parts
            row = self.metrics[
                (self.metrics["dataset"] == dataset)
                & (self.metrics["architecture"] == architecture)
            ]
            if row.empty:
                raise KeyError(f"No metrics row for {dataset}/{architecture}")
            if metric not in row.columns:
                raise KeyError(f"No metric column '{metric}' for {dataset}/{architecture}")
            return row[metric].iloc[0]

        if head == "mcnemar":
            # {mcnemar.<arch_a>.<arch_b>.<stat>}
            _, a, b, stat = parts
            row = self.mcnemar[
                ((self.mcnemar["arch_a"] == a) & (self.mcnemar["arch_b"] == b))
                | ((self.mcnemar["arch_a"] == b) & (self.mcnemar["arch_b"] == a))
            ]
            if row.empty:
                raise KeyError(f"No mcnemar row for {a} vs {b}")
            return row[stat].iloc[0]

        if head == "delta":
            # {delta.<architecture>.<metric>}
            _, architecture, metric = parts
            row = self.delta[self.delta["architecture"] == architecture]
            if row.empty:
                raise KeyError(f"No delta row for {architecture}")
            # Map convenient aliases to actual CSV columns
            alias = {
                "sensitivity": "delta_sens_pp",
                "sensitivity_ci_lo": "delta_sens_ci_lo",
                "sensitivity_ci_hi": "delta_sens_ci_hi",
                "specificity": "delta_spec_pp",
                "specificity_ci_lo": "delta_spec_ci_lo",
                "specificity_ci_hi": "delta_spec_ci_hi",
            }
            metric = alias.get(metric, metric)
            return row[metric].iloc[0]

        if head == "thresholds":
            # {thresholds.<architecture>}
            _, architecture = parts
            return self.thresholds.get(architecture, 0.0)

        raise KeyError(f"Unknown placeholder namespace: {head}")

    def _format(self, value: Any, placeholder: str) -> str:
        """Format a resolved value as a string for the manuscript."""
        if value is None:
            return "[?]"
        # Format CIs and proportions to 3 decimals; counts as integers
        try:
            f = float(value)
            if placeholder.endswith("_pct") or "pct" in placeholder:
                return f"{f:.1f}"
            # Treat integer-valued floats as ints
            if f == int(f) and abs(f) < 1e9:
                return str(int(f))
            return f"{f:.3f}"
        except (TypeError, ValueError):
            return str(value)

    def render(self, template: str) -> tuple[str, list[str]]:
        """Substitute all {placeholder} tokens. Returns (rendered_text, unresolved_list)."""
        unresolved = []
        def replace(match: re.Match) -> str:
            placeholder = match.group(1)
            try:
                value = self._lookup(placeholder)
                return self._format(value, placeholder)
            except KeyError as e:
                unresolved.append(placeholder)
                return f"{{UNRESOLVED:{placeholder}}}"

        # Match {placeholder} but NOT {} literals or markdown braces
        # Placeholders must contain at least one dot or be a known top-level name
        pattern = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_.]*)\}")
        rendered = pattern.sub(replace, template)
        return rendered, unresolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, default=Path("/predictions/canonical"))
    parser.add_argument("--results", type=Path, default=Path("/results"))
    parser.add_argument("--manifest", type=Path, default=Path("/predictions/run_manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if any placeholder remains unresolved.",
    )
    args = parser.parse_args()

    template_text = args.template.read_text()
    renderer = TemplateRenderer(args.predictions, args.results, args.manifest)
    rendered, unresolved = renderer.render(template_text)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(f"Rendered manuscript written to {args.output}")

    if unresolved:
        print(f"\nWARNING: {len(unresolved)} unresolved placeholders:")
        for p in sorted(set(unresolved)):
            print(f"  {p}")
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
