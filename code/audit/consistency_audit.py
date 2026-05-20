"""Consistency-audit linter for the rendered manuscript + appendix + cover letter.

Catches the forbidden patterns that recurred across JMIR ms#94081 revisions.
Exits non-zero on any violation.

This linter does NOT verify numeric correctness — that's
regenerate_all_from_canonical.py's job. This linter checks for stale phrases,
internally contradictory wording, and cardinality/orientation errors that
slip past pure-numeric audits.

Usage:
    python consistency_audit.py --strict \\
        --main /notebooks/rl_vs_llm_safety_v3/drafts/main_text.md \\
        --appendix /notebooks/rl_vs_llm_safety_v3/drafts/appendix.md
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


# Forbidden patterns from prior revision rounds — should NEVER appear in any
# document this round. Each entry: (regex, description, severity).
FORBIDDEN_PATTERNS = [
    # Prior-round residuals that triggered editor letters
    (r"0\.51 \(SD 0\.14\)", "Round 1 RAG cosine-similarity that was wrong", "ERROR"),
    (r"9-tier (probability )?lookup", "Round 3 forbidden phrase", "ERROR"),
    (r"action ?0[: ]", "Action 0 leakage into editor-facing text", "ERROR"),
    (r"34 pp.*GPT-5\.1", "Round 3 stale GPT-5.1 decline", "ERROR"),
    (r"DeepSeek.{1,30}AUROC 0\.5[0-9]+", "DeepSeek balanced-accuracy mislabeled as AUROC", "ERROR"),
    (r"approximately 61%", "Round 1 misreported few-shot stat", "ERROR"),
    (r"\[UNVERIFIED", "Placeholder citation still in document", "ERROR"),
    (r"\bTODO\b", "TODO marker still present", "WARN"),
    (r"\bFIXME\b", "FIXME marker still present", "WARN"),
    (r"\{[a-z_][a-z0-9_.]*\}", "Unresolved manuscript template placeholder", "ERROR"),
    (r"UNRESOLVED:", "Unresolved manuscript template placeholder marker", "ERROR"),
    # Product-name leakage
    (r"\bCOMPASS\b|\bcompass\b(?! point)", "Product name in editor-facing text", "ERROR"),
    (r"Waymark COMPASS|WAYMARK COMPASS", "Product name explicit", "ERROR"),
    # Local-path leakage
    (r"/Users/sanjaybasu", "Local absolute path leaking", "ERROR"),
    (r"waymark-local", "Local workspace name leaking", "ERROR"),
    # JMIR-history leakage (this is a fresh BMC MIDM submission)
    (r"\bms#?94081\b", "Prior JMIR submission ID leaking", "ERROR"),
    (r"\bCoristine\b", "Prior JMIR editor name leaking", "ERROR"),
    (r"Decision E2|Decision C", "JMIR decision letter labels leaking", "ERROR"),
]


# Required patterns — these MUST appear at least once
REQUIRED_PATTERNS = [
    (r"Conservative Q-Learning|Conservative Q-learning", "CQL methodology must be named"),
    (r"constellation", "Constellation architecture must be named"),
    (r"sentence-BERT|sentence-bert|Sentence-BERT", "sentence-BERT must be named"),
    (r"McNemar", "Statistical methods must mention McNemar"),
    (r"Hochberg", "Hochberg correction must be cited"),
    (r"Wilson", "Wilson CI method must be cited"),
    (r"Medicaid", "Population must be named"),
    (r"TRIPOD-AI|TRIPOD\+AI", "Reporting framework must be cited"),
    (r"DECIDE-AI", "DECIDE-AI for CDS evaluation must be cited"),
    (r"bootstrap", "Bootstrap CI methodology must be described"),
]


# Cross-pair consistency: if A appears, B must also appear
CO_OCCURRENCE_RULES = [
    (
        r"single-turn",
        r"multi-turn|Multi-turn",
        "single-turn methodology must explicitly note multi-turn as future work",
    ),
    (
        r"safety-augmented",
        r"system prompt|System prompt",
        "safety-augmented prompt requires methods description",
    ),
]


def check_file(path: Path) -> tuple[list[str], list[str], list[str]]:
    """Check one file. Returns (errors, warnings, required_missing)."""
    text = path.read_text()
    errors: list[str] = []
    warnings: list[str] = []

    # Forbidden patterns
    for pattern, desc, severity in FORBIDDEN_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            line_no = text[:m.start()].count("\n") + 1
            msg = f"  {path.name}:{line_no}: {severity} forbidden pattern '{m.group(0)}' — {desc}"
            if severity == "ERROR":
                errors.append(msg)
            else:
                warnings.append(msg)

    # Required patterns — applied to main manuscript only.
    # Cover letter and title page are short / structured documents where
    # methodological detail and journal-target naming are not required to be
    # repeated; the title page carries the journal target and the cover letter
    # is intentionally concise per submission conventions.
    required_missing: list[str] = []
    if "cover_letter" not in path.name and "title_page" not in path.name:
        for pattern, desc in REQUIRED_PATTERNS:
            if not re.search(pattern, text, re.IGNORECASE):
                required_missing.append(f"  {path.name}: REQUIRED '{pattern}' missing — {desc}")

    # Co-occurrence rules
    for pattern_a, pattern_b, desc in CO_OCCURRENCE_RULES:
        if re.search(pattern_a, text, re.IGNORECASE) and not re.search(
            pattern_b, text, re.IGNORECASE
        ):
            errors.append(f"  {path.name}: CO-OCCURRENCE violated — {desc}")

    return errors, warnings, required_missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("rl-llm-safety v3 — Consistency audit linter")
    print("=" * 70)

    all_errors: list[str] = []
    all_warnings: list[str] = []
    all_required_missing: list[str] = []

    for path in args.paths:
        if not path.exists():
            print(f"  SKIP (not found): {path}")
            continue
        errors, warnings, required_missing = check_file(path)
        all_errors.extend(errors)
        all_warnings.extend(warnings)
        all_required_missing.extend(required_missing)

    if all_errors:
        print(f"\nERRORS ({len(all_errors)}):")
        for e in all_errors:
            print(e)

    if all_warnings:
        print(f"\nWARNINGS ({len(all_warnings)}):")
        for w in all_warnings:
            print(w)

    if all_required_missing:
        print(f"\nREQUIRED PATTERNS MISSING ({len(all_required_missing)}):")
        for r in all_required_missing:
            print(r)

    print("\n" + "=" * 70)
    if all_errors or all_required_missing:
        print(f"FAIL — {len(all_errors)} errors, {len(all_required_missing)} required missing")
        return 1 if args.strict else 0
    elif all_warnings:
        print(f"PASS with warnings — {len(all_warnings)} warnings")
        return 0
    else:
        print("ALL CHECKS PASSED — 0 errors, 0 warnings, all required patterns present")
        return 0


if __name__ == "__main__":
    sys.exit(main())
