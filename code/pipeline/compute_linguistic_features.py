"""Compute linguistic features comparing real-world Medicaid SMS messages to
physician-scripted scenarios.

This is the empirical evidence for the population-gap framing of the manuscript:
real-world messages have lower reading level, more colloquialisms, more
abbreviations, and more implicit context than physician-scripted comparisons.

Output: linguistic_features.json with comparable stats for both datasets.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


# Simple keyword/phrase patterns for colloquialisms, abbreviations
COLLOQUIALISMS = [
    "kinda", "sorta", "gonna", "wanna", "gotta", "ain't", "y'all", "ya know",
    "bout to", "fixin to", "gimme", "lemme", "outta", "bunch", "lots of",
    "stuff", "thing", "guy", "dude", "real bad", "messed up", "freaking",
    "kid", "kids", "got", "lol", "smh", "omg", "wtf",
]

ABBREVIATIONS_PATTERN = re.compile(
    r"\b(b/p|hr|bp|temp|rx|otc|er|ed|abx|dx|tx|sx|nv|n/v|sob|cp|h/a|hx|"
    r"yrs|wks|days?|mins?|hrs?|am|pm|dr|drs|mr|mrs|ms|"
    r"u|ur|r|w/|w/o|b/c|btwn|btw|gf|bf|mom|dad|"
    r"ekg|cbc|cmp|cxr|ct|mri|tia|cva|cad|chf|copd|dm|htn|ckd|"
    r"avg|approx|info|temp|mins|hrs|wks|yrs|secs|"
    r"ok|okay|tho|thru|thx|thnx)\b",
    re.IGNORECASE,
)

IMPLICIT_CONTEXT_TRIGGERS = [
    "again", "as i said", "like before", "the usual", "you know",
    "the same", "everything", "the thing", "this stuff", "doing it",
    "going through", "happening", "what's going on", "feeling weird",
    "not myself", "off", "something wrong", "weird",
]


def flesch_kincaid_grade(text: str) -> float:
    """Approximation of Flesch-Kincaid grade level.

    FK grade = 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59

    Syllable count approximated by vowel groups in each word.
    """
    sentences = max(len(re.findall(r"[.!?]+", text)), 1)
    words = re.findall(r"\b[a-zA-Z]+\b", text)
    n_words = max(len(words), 1)

    def count_syllables(word: str) -> int:
        word = word.lower()
        if len(word) <= 3:
            return 1
        # Count vowel groups
        groups = re.findall(r"[aeiouy]+", word)
        n = len(groups)
        # Adjust for silent 'e'
        if word.endswith("e") and n > 1:
            n -= 1
        return max(n, 1)

    n_syllables = sum(count_syllables(w) for w in words)
    return 0.39 * (n_words / sentences) + 11.8 * (n_syllables / n_words) - 15.59


def has_colloquialism(text: str) -> bool:
    """True if text contains any colloquialism keyword."""
    tl = text.lower()
    return any(c in tl for c in COLLOQUIALISMS)


def count_abbreviations(text: str) -> int:
    """Count abbreviation pattern matches."""
    return len(ABBREVIATIONS_PATTERN.findall(text))


def has_implicit_context(text: str) -> bool:
    """True if text contains an implicit-context trigger."""
    tl = text.lower()
    return any(t in tl for t in IMPLICIT_CONTEXT_TRIGGERS)


def compute_dataset_stats(records: list[dict], message_field: str = "message") -> dict:
    """Compute linguistic stats for a list of records."""
    grade_levels = []
    colloquialism_count = 0
    abbreviation_total = 0
    abbreviation_n_records = 0
    implicit_count = 0
    word_counts = []

    for r in records:
        text = r.get(message_field, "")
        if not isinstance(text, str) or not text.strip():
            continue
        grade_levels.append(flesch_kincaid_grade(text))
        if has_colloquialism(text):
            colloquialism_count += 1
        abbrevs = count_abbreviations(text)
        if abbrevs > 0:
            abbreviation_n_records += 1
        abbreviation_total += abbrevs
        if has_implicit_context(text):
            implicit_count += 1
        word_counts.append(len(re.findall(r"\b[a-zA-Z]+\b", text)))

    n = len(records)
    return {
        "n_records": n,
        "grade_level_mean": round(float(np.mean(grade_levels)), 2) if grade_levels else None,
        "grade_level_median": round(float(np.median(grade_levels)), 2) if grade_levels else None,
        "colloquialism_pct": round(100 * colloquialism_count / max(n, 1), 1),
        "abbreviation_pct": round(100 * abbreviation_n_records / max(n, 1), 1),
        "abbreviation_avg_per_message": round(abbreviation_total / max(n, 1), 2),
        "implicit_context_pct": round(100 * implicit_count / max(n, 1), 1),
        "word_count_median": int(np.median(word_counts)) if word_counts else None,
        "word_count_mean": round(float(np.mean(word_counts)), 1) if word_counts else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--realworld", type=Path, required=True)
    parser.add_argument("--physician", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with open(args.realworld) as f:
        rw = json.load(f)
    with open(args.physician) as f:
        ph = json.load(f)

    rw_field = "message" if "message" in rw[0] else "prompt"
    ph_field = "message" if "message" in ph[0] else "prompt"

    print(f"Computing linguistic features for {len(rw)} real-world records (field={rw_field})...")
    rw_stats = compute_dataset_stats(rw, message_field=rw_field)
    print(f"Computing linguistic features for {len(ph)} physician records (field={ph_field})...")
    ph_stats = compute_dataset_stats(ph, message_field=ph_field)

    output = {
        "realworld": rw_stats,
        "physician": ph_stats,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nReal-world vs physician comparison:")
    print(f"{'Feature':<32}{'Real-world':>12}{'Physician':>12}")
    for key in ("grade_level_mean", "colloquialism_pct", "abbreviation_pct",
                "implicit_context_pct", "word_count_mean"):
        rw_v = rw_stats.get(key, "—")
        ph_v = ph_stats.get(key, "—")
        print(f"  {key:<30}{str(rw_v):>12}{str(ph_v):>12}")

    print(f"\nWritten to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
