"""LLM batch inference — Claude Opus 4.7 + Gemini 3.1 Pro safety-augmented.

For each LLM × each test set, applies the safety-augmented client to every
message and writes a per-architecture per-message canonical CSV.

Uses Anthropic + Google credits (free for this user). No paid OpenAI by default.

Schema matches local_inference.py:
    message_id, dataset, true_hazard, true_action, hazard_category,
    pred_proba, pred_hazard, pred_action, threshold_used,
    architecture, model_version, run_id, inference_time_s
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd


# Allow imports from sibling llm_clients module
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_clients.anthropic_client import ClaudeOpusClient
from llm_clients.gemini_client import GeminiProClient


def load_test_records(path: Path) -> tuple[list[dict], list[str]]:
    with open(path) as f:
        data = json.load(f)
    mids = [r.get("message_id", f"unknown_{i}") for i, r in enumerate(data)]
    return data, mids


def get_test_text(record: dict) -> str:
    for key in ("message", "prompt", "text"):
        if key in record and record[key]:
            return str(record[key])
    return ""


def get_true_hazard(record: dict) -> int:
    for key in ("detection_truth", "ground_truth_detection", "true_hazard"):
        if key in record:
            v = record[key]
            return int(v) if v is not None else 0
    return 0


def get_true_action(record: dict) -> int:
    raw = record.get("action_truth") or record.get("ground_truth_action") or None
    if raw is None or raw == "None":
        return 1
    if isinstance(raw, (int, float)):
        return int(raw)
    action_map = {
        "None": 1, "Self-Care": 1, "Routine Follow-up": 3,
        "Contact Doctor": 4, "Urgent Care": 4, "Same-Day": 5,
        "Call 911/988": 8, "Emergency": 8,
    }
    return action_map.get(str(raw), 1)


def get_hazard_category(record: dict) -> str:
    for key in ("hazard_category", "ground_truth_hazard_category"):
        if key in record:
            return str(record[key] or "benign")
    return "benign"


def run_client_on_dataset(
    client, dataset_name: str, records: list[dict], message_ids: list[str],
    run_id: str, architecture: str,
) -> pd.DataFrame:
    """Run an LLM client on all records in a dataset; return per-message DataFrame."""
    rows = []
    for i, r in enumerate(records):
        text = get_test_text(r)
        try:
            pred = client.predict(message_ids[i], text)
        except Exception as e:
            # Defensive: log and skip; partial results still written below
            print(f"    [{i}/{len(records)}] {architecture}/{dataset_name} ERROR: {e}")
            continue

        rows.append({
            "message_id": message_ids[i],
            "dataset": dataset_name,
            "true_hazard": get_true_hazard(r),
            "true_action": get_true_action(r),
            "hazard_category": get_hazard_category(r),
            "pred_proba": "",  # LLMs don't return calibrated probabilities
            "pred_hazard": int(pred.hazard),
            "pred_action": int(pred.action_num) if pred.action_num else 1,
            "threshold_used": 0.5,  # binary native output
            "architecture": architecture,
            "model_version": pred.model_version,
            "run_id": run_id,
            "inference_time_s": round(pred.inference_time_s, 4),
        })
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(records)} done")
    return pd.DataFrame(rows)


def run_llm_inference(
    test_sets: dict[str, Path],
    predictions_dir: Path,
    run_id: str,
    architectures: Optional[list[str]] = None,
) -> dict:
    """Run all LLMs on all test sets; save per-architecture CSVs.

    architectures: subset to run (default: claude + gemini safety-augmented).
    """
    predictions_dir = Path(predictions_dir)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    if architectures is None:
        architectures = ["claude_opus_4_7_safety", "gemini_3_1_pro_safety"]

    output_paths: dict[str, Path] = {}

    for arch in architectures:
        print(f"\n[LLM inference] {arch}")
        # Construct client based on arch name
        if arch == "claude_opus_4_7_safety":
            client = ClaudeOpusClient(prompt_variant="safety")
        elif arch == "claude_opus_4_7_default":
            client = ClaudeOpusClient(prompt_variant="default")
        elif arch == "gemini_3_1_pro_safety":
            client = GeminiProClient(prompt_variant="safety")
        elif arch == "gemini_3_1_pro_default":
            client = GeminiProClient(prompt_variant="default")
        else:
            print(f"  Unknown LLM architecture: {arch} — skipping")
            continue

        all_dfs = []
        for dataset_name, test_path in test_sets.items():
            records, mids = load_test_records(test_path)
            print(f"  {dataset_name}: n={len(records)} (using {arch})")
            t0 = time.time()
            df = run_client_on_dataset(client, dataset_name, records, mids, run_id, arch)
            print(f"    dataset done in {time.time() - t0:.0f}s, {len(df)} predictions")
            all_dfs.append(df)

        combined = pd.concat(all_dfs, ignore_index=True)
        out_path = predictions_dir / f"{arch}_{run_id}.csv"
        combined.to_csv(out_path, index=False)
        print(f"  → wrote {out_path} ({len(combined)} rows)")
        output_paths[arch] = out_path

    return output_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--realworld", type=Path, required=True)
    parser.add_argument("--physician", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=["claude_opus_4_7_safety", "gemini_3_1_pro_safety"],
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If set, only run on the first N messages of each dataset (smoke test).",
    )
    args = parser.parse_args()
    run_id = args.run_id or str(uuid.uuid4())
    test_sets = {"realworld_n2000": args.realworld, "physician_n41": args.physician}

    # Optional smoke-test limit
    if args.limit:
        # Create a truncated copy in /tmp; pass that
        tmp_dir = Path("/tmp/rl_llm_safety_v3_smoke")
        tmp_dir.mkdir(exist_ok=True)
        truncated = {}
        for name, path in test_sets.items():
            with open(path) as f:
                data = json.load(f)
            truncated_data = data[: args.limit]
            new_path = tmp_dir / f"{name}_limit{args.limit}.json"
            with open(new_path, "w") as f:
                json.dump(truncated_data, f)
            truncated[name] = new_path
        test_sets = truncated
        print(f"SMOKE TEST mode: limited to first {args.limit} messages per dataset")

    run_llm_inference(test_sets, args.predictions_dir, run_id, args.architectures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
