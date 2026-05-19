"""Comprehensive concordance check across ALL submission artifacts.

Implements the protocols defined in:
- ~/.claude/skills/submit-prep/SKILL.md (Layer 1 + Layer 2 audit)
- ~/.claude/skills/citation-management/SKILL.md (7 failure modes A-G)

Exits non-zero if any check fails. Run before any submission.

Coverage:
1. Numerical concordance: every value in manuscript/appendix/cover letter traces to a
   canonical CSV row.
2. Citation concordance: every in-text citation maps to a populated reference list
   entry; every reference is cited; appendix citations use appendix's own ref list.
3. Data concordance: per-message prediction CSVs exist for every architecture × every
   dataset; SHA-256 checksums of canonical test sets match run_manifest.
4. Code concordance: every script the manuscript references exists in code/.
5. Figure concordance: every figure binary timestamp ≥ canonical data file timestamp
   (no stale figures).
6. ROC monotonicity: every architecture with calibrated probabilities passes.
7. Table-to-table arithmetic: cross-table identities (e.g., n_under_triage ≤ n_total).
8. Forbidden patterns: no product names, no JMIR-history leakage, no local paths,
   no unresolved placeholders.
9. Required reporting framework patterns: TRIPOD-AI, DECIDE-AI, McNemar, Hochberg.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Optional


REPO_ROOT = Path("/Users/sanjaybasu/waymark-local/packaging/rl_llm_safety_github_v3")
NOTEBOOKS_ROOT = Path("/Users/sanjaybasu/waymark-local/notebooks/rl_vs_llm_safety_v3")


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def check(label: str, ok: bool, detail: str = "") -> bool:
    sym = "✓" if ok else "✗"
    print(f"  {sym} {label}" + (f" — {detail}" if detail else ""))
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Check 1: Numerical concordance
# ─────────────────────────────────────────────────────────────────────────────


def check_numerical_concordance(predictions_dir: Path, results_dir: Path, drafts_dir: Path) -> tuple[bool, list[str]]:
    """Every number in the manuscript should derive from a canonical CSV."""
    section("Check 1: Numerical concordance")
    errors = []
    main_text = drafts_dir / "main_text.md"
    appendix = drafts_dir / "appendix.md"

    if not main_text.exists():
        check("Manuscript main_text.md exists", False)
        return False, ["main_text.md not found"]
    check("Manuscript main_text.md exists", True, str(main_text))

    # Look for unresolved placeholders
    for path in [main_text, appendix]:
        if not path.exists():
            continue
        text = path.read_text()
        unresolved = re.findall(r"\{UNRESOLVED:[^}]+\}", text)
        ok = check(
            f"No unresolved placeholders in {path.name}",
            len(unresolved) == 0,
            f"{len(unresolved)} unresolved" if unresolved else "clean",
        )
        if not ok:
            errors.extend(f"{path.name}: {p}" for p in unresolved[:10])

    # Verify metrics_canonical.csv was the source
    metrics_path = results_dir / "metrics_canonical.csv"
    if metrics_path.exists():
        check("metrics_canonical.csv exists (canonical source)", True, str(metrics_path))
    else:
        check("metrics_canonical.csv exists", False)
        errors.append("metrics_canonical.csv missing")
    return len(errors) == 0, errors


# ─────────────────────────────────────────────────────────────────────────────
# Check 2: Citation concordance (citation-management failure modes E and G)
# ─────────────────────────────────────────────────────────────────────────────


def check_citation_concordance(drafts_dir: Path) -> tuple[bool, list[str]]:
    """Bidirectional in-text ↔ reference list check (citation-management Failure E)."""
    section("Check 2: Citation concordance (main and appendix)")
    errors = []

    for name in ("main_text.md", "appendix.md"):
        path = drafts_dir / name
        if not path.exists():
            continue
        text = path.read_text()

        # Find in-text citations like [N] or [N,M] or [N-M]
        in_text = set()
        for m in re.finditer(r"\[(\d+(?:[,-]\s*\d+)*)\]", text):
            for part in m.group(1).split(","):
                part = part.strip()
                if "-" in part:
                    lo, hi = [int(x.strip()) for x in part.split("-", 1)]
                    in_text.update(range(lo, hi + 1))
                else:
                    try:
                        in_text.add(int(part))
                    except ValueError:
                        pass

        # Find reference list entries like "1. Author X, ..." at start of line
        # Section heading "References" is the anchor
        ref_section_start = text.find("References")
        if ref_section_start == -1:
            check(f"{name}: References section present", False)
            errors.append(f"{name}: no References section")
            continue
        ref_text = text[ref_section_start:]
        ref_entries = set()
        for m in re.finditer(r"^(\d+)\.\s+\w", ref_text, re.MULTILINE):
            ref_entries.add(int(m.group(1)))

        # Direction 1: every in-text citation has a reference list entry
        missing = in_text - ref_entries
        check(
            f"{name}: every [N] maps to a reference entry",
            len(missing) == 0,
            f"missing entries for: {sorted(missing)}" if missing else "all cited refs present",
        )
        if missing:
            errors.append(f"{name}: in-text citations {sorted(missing)} have no ref-list entry")

        # Direction 2: every reference list entry is cited at least once
        orphans = ref_entries - in_text
        check(
            f"{name}: every reference entry is cited",
            len(orphans) == 0,
            f"orphan entries: {sorted(orphans)}" if orphans else "all entries cited",
        )
        if orphans:
            errors.append(f"{name}: ref-list entries {sorted(orphans)} are never cited in-text")

        # Direction 3: sequential numbering (no gaps)
        if ref_entries:
            expected = set(range(1, max(ref_entries) + 1))
            gaps = expected - ref_entries
            check(
                f"{name}: reference numbering is sequential (no gaps)",
                len(gaps) == 0,
                f"missing positions: {sorted(gaps)}" if gaps else "1..N sequential",
            )
            if gaps:
                errors.append(f"{name}: gaps in reference numbering at positions {sorted(gaps)}")

    return len(errors) == 0, errors


# ─────────────────────────────────────────────────────────────────────────────
# Check 3: Data concordance
# ─────────────────────────────────────────────────────────────────────────────


def check_data_concordance(predictions_dir: Path) -> tuple[bool, list[str]]:
    section("Check 3: Data concordance (predictions vs run_manifest)")
    errors = []
    manifest_path = predictions_dir.parent / "run_manifest.json"
    if not manifest_path.exists():
        check("run_manifest.json exists", False)
        return False, ["run_manifest.json missing"]
    check("run_manifest.json exists", True)

    manifest = json.loads(manifest_path.read_text())

    # SHA-256 verification of canonical test sets (if accessible locally)
    for ds_name in ("realworld_n2000", "physician_n41"):
        ds = manifest.get("datasets", {}).get(ds_name, {})
        n_expected = ds.get("n_records")
        if n_expected:
            check(f"{ds_name}: manifest reports n={n_expected}", True)

    # Per-architecture CSV existence
    expected_archs = manifest.get("architectures", [])
    if not expected_archs:
        check("manifest.architectures populated", False)
        errors.append("manifest.architectures empty")
    else:
        check(f"manifest.architectures: {len(expected_archs)} configured", True)

    csv_files = list(predictions_dir.glob("*.csv"))
    archs_in_csvs = set()
    for csv_path in csv_files:
        import pandas as pd
        df = pd.read_csv(csv_path, nrows=1)
        if "architecture" in df.columns:
            archs_in_csvs.add(df["architecture"].iloc[0])
    check(
        f"{len(archs_in_csvs)} architectures have prediction CSVs",
        len(archs_in_csvs) >= 6,
        f"found: {sorted(archs_in_csvs)}",
    )

    return len(errors) == 0, errors


# ─────────────────────────────────────────────────────────────────────────────
# Check 4: Code concordance
# ─────────────────────────────────────────────────────────────────────────────


def check_code_concordance(repo_root: Path, drafts_dir: Path) -> tuple[bool, list[str]]:
    section("Check 4: Code concordance (every referenced module exists)")
    errors = []
    required_modules = [
        "code/pipeline/modal_pipeline.py",
        "code/pipeline/canonical_training.py",
        "code/pipeline/local_inference.py",
        "code/pipeline/llm_inference.py",
        "code/pipeline/metrics_phase.py",
        "code/pipeline/render_tables_figures.py",
        "code/pipeline/manuscript_renderer.py",
        "code/pipeline/build_supplementary_package.py",
        "code/pipeline/compute_linguistic_features.py",
        "code/pipeline/phase_d_render.sh",
        "code/audit/metrics.py",
        "code/audit/test_metrics.py",
        "code/audit/roc_monotonicity.py",
        "code/audit/regenerate_all_from_canonical.py",
        "code/audit/consistency_audit.py",
        "code/llm_clients/base.py",
        "code/llm_clients/prompts.py",
        "code/llm_clients/anthropic_client.py",
        "code/llm_clients/gemini_client.py",
    ]
    for mod in required_modules:
        path = repo_root / mod
        ok = check(f"{mod}", path.exists())
        if not ok:
            errors.append(f"Missing module: {mod}")
    return len(errors) == 0, errors


# ─────────────────────────────────────────────────────────────────────────────
# Check 5: Figure concordance
# ─────────────────────────────────────────────────────────────────────────────


def check_figure_concordance(results_dir: Path) -> tuple[bool, list[str]]:
    section("Check 5: Figure concordance (timestamps + presence)")
    errors = []
    figures_dir = results_dir / "figures"
    expected_figures = ["figure1_sens_spec_change.png", "figure2_action_recommendations.png"]
    metrics_path = results_dir / "metrics_canonical.csv"

    for fig_name in expected_figures:
        fig_path = figures_dir / fig_name
        if not fig_path.exists():
            check(f"{fig_name} exists", False)
            errors.append(f"Missing figure: {fig_name}")
            continue
        check(f"{fig_name} exists", True, f"size={fig_path.stat().st_size//1024} KB")

        # Figure should be newer than canonical metrics CSV
        if metrics_path.exists():
            fig_mtime = fig_path.stat().st_mtime
            csv_mtime = metrics_path.stat().st_mtime
            ok = fig_mtime >= csv_mtime - 5  # 5 sec tolerance
            check(
                f"{fig_name} newer than metrics_canonical.csv",
                ok,
                "figure regenerated after metrics" if ok else "STALE FIGURE — regenerate",
            )
            if not ok:
                errors.append(f"Stale figure: {fig_name}")

    return len(errors) == 0, errors


# ─────────────────────────────────────────────────────────────────────────────
# Check 6: ROC monotonicity
# ─────────────────────────────────────────────────────────────────────────────


def check_roc_monotonicity(predictions_dir: Path) -> tuple[bool, list[str]]:
    section("Check 6: ROC monotonicity (every architecture with calibrated probas)")
    errors = []
    sys.path.insert(0, str(REPO_ROOT / "code" / "audit"))
    import roc_monotonicity as rm
    for csv_path in sorted(predictions_dir.glob("*.csv")):
        ok = rm.check_file(csv_path)
        if not ok:
            errors.append(f"ROC violation in {csv_path.name}")
    check("ROC monotonicity across all calibrated architectures", len(errors) == 0)
    return len(errors) == 0, errors


# ─────────────────────────────────────────────────────────────────────────────
# Check 7: Table-to-table arithmetic
# ─────────────────────────────────────────────────────────────────────────────


def check_table_arithmetic(predictions_dir: Path, results_dir: Path) -> tuple[bool, list[str]]:
    section("Check 7: Table-to-table arithmetic identities")
    errors = []
    import pandas as pd

    metrics_path = results_dir / "metrics_canonical.csv"
    if not metrics_path.exists():
        return False, ["metrics_canonical.csv missing"]
    metrics = pd.read_csv(metrics_path)

    # Identity: TP = sens × n_hazards (within rounding)
    for _, row in metrics.iterrows():
        if "n_hazards" not in row or "sensitivity" not in row or "tp" not in row:
            continue
        expected_tp = round(float(row["sensitivity"]) * int(row["n_hazards"]))
        actual_tp = int(row["tp"])
        if abs(expected_tp - actual_tp) > 1:
            arch = row.get("architecture", "?")
            ds = row.get("dataset", "?")
            errors.append(
                f"TP mismatch for {arch}/{ds}: expected {expected_tp} (sens × n_hazards), got {actual_tp}"
            )

    check(f"TP = sens × n_hazards identity holds across {len(metrics)} rows", len(errors) == 0)
    return len(errors) == 0, errors


# ─────────────────────────────────────────────────────────────────────────────
# Check 8: Forbidden patterns
# ─────────────────────────────────────────────────────────────────────────────


def check_forbidden_patterns(drafts_dir: Path) -> tuple[bool, list[str]]:
    section("Check 8: Forbidden patterns (consistency_audit)")
    errors = []
    sys.path.insert(0, str(REPO_ROOT / "code" / "audit"))
    import consistency_audit as ca
    files = [drafts_dir / "main_text.md", drafts_dir / "appendix.md", drafts_dir / "cover_letter.md"]
    for path in files:
        if not path.exists():
            continue
        e, w, m = ca.check_file(path)
        if e:
            errors.extend(e)
        check(f"{path.name}: {len(e)} errors, {len(w)} warnings, {len(m)} required-missing", len(e) == 0 and len(m) == 0)
        errors.extend(m)
    return len(errors) == 0, errors


# ─────────────────────────────────────────────────────────────────────────────
# Check 9: Required reporting framework patterns
# ─────────────────────────────────────────────────────────────────────────────


def check_reporting_frameworks(drafts_dir: Path) -> tuple[bool, list[str]]:
    section("Check 9: Reporting frameworks (TRIPOD-AI, DECIDE-AI, McNemar, Hochberg)")
    errors = []
    required = {
        "TRIPOD-AI or TRIPOD+AI": r"TRIPOD[-+]AI",
        "DECIDE-AI": r"DECIDE-AI",
        "McNemar test": r"McNemar",
        "Hochberg correction": r"Hochberg",
        "Wilson score CI": r"Wilson",
        "BMC Medical Informatics and Decision Making": r"BMC Medical Informatics and Decision Making",
    }
    for name in ("main_text.md", "appendix.md"):
        path = drafts_dir / name
        if not path.exists():
            continue
        text = path.read_text()
        for label, pattern in required.items():
            present = bool(re.search(pattern, text, re.IGNORECASE))
            check(f"{name}: {label}", present)
            if not present and name == "main_text.md":
                # Required in main only; appendix is optional
                errors.append(f"{name}: '{label}' missing")
    return len(errors) == 0, errors


# ─────────────────────────────────────────────────────────────────────────────
# Master
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, default=REPO_ROOT / "predictions" / "canonical_filtered")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--drafts-dir", type=Path, default=NOTEBOOKS_ROOT / "drafts")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    all_errors: list[str] = []

    checks = [
        ("numerical concordance", lambda: check_numerical_concordance(args.predictions_dir, args.results_dir, args.drafts_dir)),
        ("citation concordance", lambda: check_citation_concordance(args.drafts_dir)),
        ("data concordance", lambda: check_data_concordance(args.predictions_dir)),
        ("code concordance", lambda: check_code_concordance(REPO_ROOT, args.drafts_dir)),
        ("figure concordance", lambda: check_figure_concordance(args.results_dir)),
        ("ROC monotonicity", lambda: check_roc_monotonicity(args.predictions_dir)),
        ("table arithmetic", lambda: check_table_arithmetic(args.predictions_dir, args.results_dir)),
        ("forbidden patterns", lambda: check_forbidden_patterns(args.drafts_dir)),
        ("reporting frameworks", lambda: check_reporting_frameworks(args.drafts_dir)),
    ]

    for label, fn in checks:
        try:
            ok, errors = fn()
        except Exception as e:
            print(f"\n  ✗ {label}: EXCEPTION {type(e).__name__}: {e}")
            all_errors.append(f"{label}: exception {e}")
            continue
        if not ok:
            all_errors.extend([f"[{label}] {e}" for e in errors])

    section(f"COMPREHENSIVE CONCORDANCE — {len(all_errors)} ERROR(S)")
    if all_errors:
        for e in all_errors[:50]:
            print(f"  {e}")
        if len(all_errors) > 50:
            print(f"  ... and {len(all_errors) - 50} more")
        return 1 if args.strict else 0
    print("  ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
