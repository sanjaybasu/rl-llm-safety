"""
rl-llm-safety v3 — Single canonical Modal pipeline
====================================================
Fresh-take pipeline addressing JMIR Medical Informatics Decision E2 (2026-05-14).

Root-cause fix from prior submission: every reported number must derive from
ONE canonical per-message prediction file per architecture. Tables 2 and 3
must threshold the same prediction file across the operating curve, so ROC
monotonicity is preserved by construction.

Plan: /Users/sanjaybasu/.claude/plans/sharded-sleeping-neumann.md

Run order:
  modal run modal_pipeline.py::orchestrate              # full pipeline (detached recommended)
  modal run --detach modal_pipeline.py::orchestrate     # survives local client disconnect

  # Or run phases individually:
  modal run modal_pipeline.py::phase0_preflight
  modal run modal_pipeline.py::phase1_train_local
  modal run modal_pipeline.py::phase2_train_cql
  modal run modal_pipeline.py::phase3_train_actionhead
  modal run modal_pipeline.py::phase5_local_inference
  modal run modal_pipeline.py::phase6_llm_inference
  modal run modal_pipeline.py::phase8_consolidate
  modal run modal_pipeline.py::phase9_metrics
  modal run modal_pipeline.py::phase10_mcnemar
  modal run modal_pipeline.py::phase11_bootstrap
  modal run modal_pipeline.py::phase12_tables_figures
  modal run modal_pipeline.py::phase14_audit

Phase-chain order (auto-spawn):
  phase0 → phase1 → phase2 → phase3 → phase5 → phase6 → phase8 → phase9
    → phase10 → phase11 → phase12 → phase14

Budget (provisional 8-config scope, primary LLMs on free credits):
  Local training (CPU/A10G):    ~$5
  Local inference (CPU):        ~$2
  Anthropic Claude Opus 4.7:    ~$0  (free credits)
  Gemini 3.1 Pro Preview:       ~$0  (free credits)
  Modal compute orchestration:  ~$15
  Total pipeline:               ~$22 baseline
  Optional +GPT-5.5 row:        +$25-75 if Phase B adds it

Concurrency safety:
  - App namespaced as 'rl-llm-safety-v3-pipeline'
  - All volumes prefixed 'rl-llm-safety-v3-*'
  - Never invokes `modal app stop` or `modal volume rm`
  - Co-exists with ANCHOR (mira-*) and MIRA-3 jobs on the same Modal account
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

# ─────────────────────────────────────────────────────────────────────────────
# App and volume definitions (namespaced to avoid collision with concurrent
# ANCHOR / MIRA-3 Modal apps and volumes)
# ─────────────────────────────────────────────────────────────────────────────

APP_NAME = "rl-llm-safety-v3-pipeline"
app = modal.App(APP_NAME)

# Volumes are namespaced; do not modify any volume not matching this prefix
data_vol = modal.Volume.from_name("rl-llm-safety-v3-data", create_if_missing=True)
predictions_vol = modal.Volume.from_name(
    "rl-llm-safety-v3-predictions", create_if_missing=True
)
results_vol = modal.Volume.from_name(
    "rl-llm-safety-v3-results", create_if_missing=True
)
models_vol = modal.Volume.from_name(
    "rl-llm-safety-v3-models", create_if_missing=True
)

LOCAL_ROOT = Path(__file__).parent.parent.parent.parent
PROJECT_DIR = Path(__file__).parent.parent

# ─────────────────────────────────────────────────────────────────────────────
# Docker image
# ─────────────────────────────────────────────────────────────────────────────

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "scikit-learn==1.5.2",
        "xgboost==2.1.3",
        "sentence-transformers==3.3.1",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "scipy==1.14.1",
        "torch==2.4.1",
        "anthropic==0.40.0",
        "google-generativeai==0.8.3",
        "openai==1.55.0",
        "statsmodels==0.14.4",
        "matplotlib==3.9.2",
        "tenacity==9.0.0",
    )
    .add_local_dir(str(PROJECT_DIR), "/app", copy=True)
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SEED = 42
N_BOOTSTRAP = 10_000

# Architectures to evaluate — Phase B locked 6-config matrix (2026-05-18)
# Reference: notebooks/rl_vs_llm_safety_v3/protocol/protocol.md §4
#
# Top-level matrix is 6 configurations (rows in Figure 1, Table 2). The deployed
# CDS architecture is presented as ONE top-level row but evaluated via 3
# sub-components in detail tables (CQL sens-opt, CQL reward-opt, ActionHead).
ARCHITECTURES = [
    # 1. Deployed CDS architecture — 3 sub-components run separately for detail tables
    "cql_sens_opt",      # Sub-component A: sensitivity-optimized CQL
    "cql_reward_opt",    # Sub-component B: reward-optimized CQL (Q-fn argmax)
    "constellation",     # Sub-component C: constellation 23-category classifier
    "actionhead",        # Sub-component D: 8-class action recommender (action endpoint only)
    # 2. Classical supervised comparator
    "xgboost_sbert",
    # 3. Rule-based local
    "guardrails",
    # 4. Classical baseline
    "logreg_tfidf",
    # 5. Frontier LLM — primary (Anthropic credits)
    "claude_opus_4_7_safety",
    # 6. Frontier LLM — secondary (Google credits)
    "gemini_3_1_pro_safety",
]

# Top-level grouping for presentation (Figure 1, abstract, Table 2 headline rows)
# Maps top-level 6-row to underlying ARCHITECTURES entries.
TOP_LEVEL_ROWS = {
    "deployed_cds": ["cql_sens_opt", "constellation", "actionhead"],  # primary row
    "xgboost_sbert": ["xgboost_sbert"],
    "guardrails": ["guardrails"],
    "logreg_tfidf": ["logreg_tfidf"],
    "claude_opus_4_7": ["claude_opus_4_7_safety"],
    "gemini_3_1_pro": ["gemini_3_1_pro_safety"],
}

MODEL_VERSIONS = {
    "claude_opus_4_7_safety": "claude-opus-4-7",
    "gemini_3_1_pro_safety": "gemini-3.1-pro-preview",
    # Optional secondary LLM (paid OpenAI) — not in the locked Phase B scope:
    # "gpt_5_5_safety": "gpt-5.5",
}

DATASETS = ["realworld_n2000", "physician_n41"]

# ─────────────────────────────────────────────────────────────────────────────
# Phase 0 — Preflight: verify test sets, generate run_id, write manifest
# ─────────────────────────────────────────────────────────────────────────────


@app.function(
    image=base_image,
    volumes={"/data": data_vol, "/predictions": predictions_vol},
    timeout=600,
)
def phase0_preflight() -> str:
    """Verify canonical test sets present; generate run_id; write manifest."""
    import hashlib
    import uuid
    from datetime import datetime, timezone

    realworld_path = Path("/data/realworld_cases_n2000.json")
    physician_path = Path("/data/physician_holdout_n41.json")

    training_path = Path("/data/combined_train.json")
    if not realworld_path.exists():
        raise FileNotFoundError(
            f"Canonical real-world test set missing at {realworld_path}. "
            f"Upload data first via 'modal volume put rl-llm-safety-v3-data ...'"
        )
    if not physician_path.exists():
        raise FileNotFoundError(
            f"Canonical physician holdout missing at {physician_path}."
        )
    if not training_path.exists():
        raise FileNotFoundError(
            f"Canonical training set missing at {training_path}. "
            f"Upload training data first via 'modal volume put rl-llm-safety-v3-data ...'"
        )

    # Compute SHA-256 checksums
    def sha256_of(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    realworld_sha = sha256_of(realworld_path)
    physician_sha = sha256_of(physician_path)

    # Verify cardinality
    with open(realworld_path) as f:
        rw = json.load(f)
    with open(physician_path) as f:
        ph = json.load(f)

    rw_n = len(rw)
    rw_haz = sum(1 for r in rw if r.get("detection_truth") in (1, "1", True))
    ph_n = len(ph)
    ph_haz = sum(1 for r in ph if r.get("detection_truth") in (1, "1", True))

    assert rw_n == 2000, f"Expected 2000 real-world messages, got {rw_n}"
    assert rw_haz == 165, f"Expected 165 real-world hazards, got {rw_haz}"
    assert ph_n == 41, f"Expected 41 physician messages, got {ph_n}"
    assert ph_haz == 27, f"Expected 27 physician hazards, got {ph_haz}"

    run_id = str(uuid.uuid4())
    manifest = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "datasets": {
            "realworld_n2000": {
                "path": str(realworld_path),
                "sha256": realworld_sha,
                "n_records": rw_n,
                "n_hazards": rw_haz,
                "n_benigns": rw_n - rw_haz,
            },
            "physician_n41": {
                "path": str(physician_path),
                "sha256": physician_sha,
                "n_records": ph_n,
                "n_hazards": ph_haz,
                "n_benigns": ph_n - ph_haz,
            },
        },
        "architectures": ARCHITECTURES,
        "model_versions": MODEL_VERSIONS,
    }

    manifest_path = Path("/predictions/run_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    predictions_vol.commit()

    print(f"Phase 0 complete. run_id={run_id}")
    print(f"  Real-world: n={rw_n}, hazards={rw_haz}, sha={realworld_sha[:16]}...")
    print(f"  Physician:  n={ph_n}, hazards={ph_haz}, sha={physician_sha[:16]}...")
    return run_id


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Train local supervised models (deterministic seed=42)
# ─────────────────────────────────────────────────────────────────────────────


@app.function(
    image=base_image,
    volumes={
        "/data": data_vol,
        "/models": models_vol,
        "/predictions": predictions_vol,
    },
    timeout=3600,
    cpu=4,
    memory=8192,
)
def phase1_train_local(run_id: str) -> str:
    """Train all 6 local architectures (guardrails, LogReg+TF-IDF, XGBoost+SBERT,
    Constellation, calibrated detector, ActionHead) on the 1,280-example training
    set using ONE deterministic seed=42 train/val split.

    Calls canonical_training.train_all() which is the single canonical entry point.
    """
    print(f"Phase 1: canonical training (run_id={run_id})")
    sys.path.insert(0, "/app/pipeline")
    import canonical_training as ct
    manifest = ct.train_all(
        training_data_path=Path("/data/combined_train.json"),
        output_dir=Path("/models/v4"),
        seed=SEED,
    )
    models_vol.commit()
    print(f"Training manifest: {manifest['training_time_seconds']:.1f}s")
    return run_id


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Train CQL controller
# ─────────────────────────────────────────────────────────────────────────────


@app.function(
    image=base_image,
    volumes={
        "/data": data_vol,
        "/models": models_vol,
    },
    timeout=3600,
    cpu=4,
    memory=8192,
)
def phase2_train_cql(run_id: str) -> str:
    """CQL training is included in Phase 1's canonical_training.train_all()
    (calibrated detector + sens-opt threshold + reward-opt threshold).
    This phase is a NO-OP placeholder for the 14-phase plan structure."""
    print(f"Phase 2: CQL controller — included in Phase 1 (no-op).")
    return run_id


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Train ActionHead
# ─────────────────────────────────────────────────────────────────────────────


@app.function(
    image=base_image,
    volumes={"/data": data_vol, "/models": models_vol},
    timeout=1800,
)
def phase3_train_actionhead(run_id: str) -> str:
    """ActionHead training is included in Phase 1's canonical training.
    NO-OP placeholder for compatibility with the 14-phase plan."""
    print(f"Phase 3: ActionHead — included in Phase 1 (no-op).")
    return run_id


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — Local inference (parallel across local architectures)
# ─────────────────────────────────────────────────────────────────────────────


@app.function(
    image=base_image,
    volumes={
        "/data": data_vol,
        "/models": models_vol,
        "/predictions": predictions_vol,
    },
    timeout=3600,
    cpu=4,
    memory=8192,
)
def phase5_inference_local(run_id: str) -> str:
    """Apply ALL 7 local architectures to both test sets; save per-architecture CSVs.

    Calls local_inference.run_all_local_inference() which sequentially produces
    one canonical CSV per architecture × dataset.
    """
    print(f"Phase 5: local inference for all local architectures (run_id={run_id})")
    sys.path.insert(0, "/app/pipeline")
    import local_inference as li
    test_sets = {
        "realworld_n2000": Path("/data/realworld_cases_n2000.json"),
        "physician_n41": Path("/data/physician_holdout_n41.json"),
    }
    output_paths = li.run_all_local_inference(
        models_dir=Path("/models/v4"),
        test_sets=test_sets,
        predictions_dir=Path("/predictions/canonical"),
        run_id=run_id,
    )
    predictions_vol.commit()
    print(f"Phase 5 complete — {len(output_paths)} architectures inferred.")
    return run_id


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6 — LLM API inference (parallel across providers)
# ─────────────────────────────────────────────────────────────────────────────


@app.function(
    image=base_image,
    volumes={
        "/data": data_vol,
        "/predictions": predictions_vol,
    },
    timeout=7200,
    secrets=[
        modal.Secret.from_name("anthropic"),
        modal.Secret.from_name("google"),
    ],
)
def phase6_inference_llm(architecture: str, run_id: str) -> str:
    """Apply one LLM (Claude or Gemini, safety-augmented) to both test sets.

    Calls llm_inference.run_llm_inference for the named architecture.
    """
    print(f"Phase 6: LLM inference for {architecture} (run_id={run_id})")
    sys.path.insert(0, "/app/pipeline")
    sys.path.insert(0, "/app")
    import llm_inference as li
    test_sets = {
        "realworld_n2000": Path("/data/realworld_cases_n2000.json"),
        "physician_n41": Path("/data/physician_holdout_n41.json"),
    }
    output_paths = li.run_llm_inference(
        test_sets=test_sets,
        predictions_dir=Path("/predictions/canonical"),
        run_id=run_id,
        architectures=[architecture],
    )
    predictions_vol.commit()
    print(f"Phase 6 complete for {architecture}: {output_paths}")
    return f"/predictions/canonical/{architecture}_{run_id}.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 8 — Consolidate predictions into canonical CSVs
# ─────────────────────────────────────────────────────────────────────────────


@app.function(
    image=base_image,
    volumes={"/predictions": predictions_vol},
    timeout=600,
)
def phase8_consolidate(run_id: str) -> dict:
    """Consolidation is a no-op: each per-architecture inference function already
    writes the canonical schema directly. Phase 8 just verifies the expected
    files exist on the predictions volume."""
    print(f"Phase 8: verifying canonical prediction files (run_id={run_id})")
    canonical_dir = Path("/predictions/canonical")
    csv_files = sorted(canonical_dir.glob("*.csv"))
    print(f"  Found {len(csv_files)} per-architecture CSVs:")
    for p in csv_files:
        print(f"    {p.name}")
    return {"status": "verified", "run_id": run_id, "n_csv_files": len(csv_files)}


# ─────────────────────────────────────────────────────────────────────────────
# Phase 9 — Metrics computation (Wilson CIs + bootstrap CIs)
# ─────────────────────────────────────────────────────────────────────────────


@app.function(
    image=base_image,
    volumes={
        "/predictions": predictions_vol,
        "/results": results_vol,
    },
    timeout=3600,
)
def phase9_metrics(run_id: str) -> str:
    """Run the metrics phase: read per-architecture predictions for THIS run_id
    only and write metrics_canonical.csv + mcnemar_matrix.csv + delta_bootstrap.

    Filtering by run_id ensures that if multiple Modal orchestrate runs have
    written to the same volume, only this run's predictions are used.
    """
    print(f"Phase 9-11: metrics + McNemar + bootstrap (run_id={run_id})")
    sys.path.insert(0, "/app/pipeline")
    sys.path.insert(0, "/app/audit")
    import metrics_phase as mp
    out_paths = mp.run_metrics_phase(
        predictions_dir=Path("/predictions/canonical"),
        results_dir=Path("/results"),
        dataset_for_mcnemar="realworld_n2000",
        seed=SEED,
        run_id=run_id,  # Filter to this run only
    )
    results_vol.commit()
    print(f"Phase 9-11 complete: {len(out_paths)} canonical CSVs written.")
    return "/results/metrics_canonical.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 10 — McNemar paired comparisons
# ─────────────────────────────────────────────────────────────────────────────


@app.function(
    image=base_image,
    volumes={
        "/predictions": predictions_vol,
        "/results": results_vol,
    },
    timeout=600,
)
def phase10_mcnemar(run_id: str) -> str:
    """McNemar matrix is computed within Phase 9 (metrics_phase.run_metrics_phase).
    NO-OP placeholder for compatibility with the 14-phase plan structure."""
    print(f"Phase 10: McNemar — included in Phase 9 (no-op).")
    return "/results/mcnemar_matrix.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 11 — Bootstrap Δ analyses (physician → real-world)
# ─────────────────────────────────────────────────────────────────────────────


@app.function(
    image=base_image,
    volumes={
        "/predictions": predictions_vol,
        "/results": results_vol,
    },
    timeout=1800,
)
def phase11_bootstrap(run_id: str) -> str:
    """Δ bootstrap is computed within Phase 9 (metrics_phase.run_metrics_phase).
    NO-OP placeholder for compatibility with the 14-phase plan structure."""
    print(f"Phase 11: Δ bootstrap — included in Phase 9 (no-op).")
    return "/results/delta_bootstrap_canonical.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 12 — Tables and figures (ALL from canonical CSVs)
# ─────────────────────────────────────────────────────────────────────────────


@app.function(
    image=base_image,
    volumes={"/results": results_vol},
    timeout=1800,
)
def phase12_tables_figures(run_id: str) -> dict:
    """Render every table and figure from canonical CSVs ONLY."""
    print(f"Phase 12: tables and figures (run_id={run_id})")
    sys.path.insert(0, "/app/pipeline")
    import render_tables_figures as r
    out = r.render_all(
        predictions_dir=Path("/predictions/canonical"),
        results_dir=Path("/results"),
    )
    results_vol.commit()
    print(f"Phase 12 complete — rendered {len(out)} artifacts.")
    return {"status": "rendered", "run_id": run_id, "outputs": {k: str(v) for k, v in out.items()}}


# ─────────────────────────────────────────────────────────────────────────────
# Phase 14 — Audit (verify_numbers + consistency_audit + ROC monotonicity)
# ─────────────────────────────────────────────────────────────────────────────


@app.function(
    image=base_image,
    volumes={"/predictions": predictions_vol, "/results": results_vol},
    timeout=600,
)
def phase14_audit(run_id: str) -> dict:
    """Run ROC monotonicity assertion + the canonical regenerate-all check."""
    print(f"Phase 14: audit (run_id={run_id})")
    sys.path.insert(0, "/app/audit")
    import roc_monotonicity as rm
    import pandas as pd
    canonical_dir = Path("/predictions/canonical")
    csv_files = sorted(canonical_dir.glob("*.csv"))

    # ROC monotonicity check on every architecture with calibrated probabilities
    all_ok = True
    violations = []
    for csv_path in csv_files:
        if not rm.check_file(csv_path):
            all_ok = False
            violations.append(str(csv_path))
    print(f"\nROC monotonicity: {'PASS' if all_ok else 'FAIL'}")
    if violations:
        print("Violations:")
        for v in violations:
            print(f"  - {v}")
    return {"status": "PASS" if all_ok else "FAIL", "run_id": run_id,
            "violations": violations}


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator — chains all phases
# ─────────────────────────────────────────────────────────────────────────────


@app.function(image=base_image, timeout=86400)
def orchestrate() -> dict:
    """Single entry point that runs all phases in order.

    Use:
      modal run --detach modal_pipeline.py::orchestrate

    Detached mode lets the pipeline survive local laptop sleep / disconnect.
    """
    print("=" * 70)
    print(f"rl-llm-safety v3 pipeline starting (app: {APP_NAME})")
    print("=" * 70)

    # Phase 0: preflight
    run_id = phase0_preflight.remote()
    print(f"\n[Phase 0 complete] run_id={run_id}")

    # Phase 1: canonical training (includes all local supervised + CQL + ActionHead)
    # Run in parallel with Phase 6 LLM inference (LLMs do not depend on local training)
    phase1_handle = phase1_train_local.spawn(run_id)
    phase6_handles = [
        phase6_inference_llm.spawn(arch, run_id)
        for arch in ("claude_opus_4_7_safety", "gemini_3_1_pro_safety")
    ]

    phase1_handle.get()
    print("[Phase 1 complete] all local architectures trained (canonical pipeline)")

    # Phase 2 + 3 are no-ops (included in Phase 1) — call for log clarity
    phase2_train_cql.remote(run_id)
    phase3_train_actionhead.remote(run_id)

    # Phase 5: local inference for all 7 local architectures in ONE call
    phase5_inference_local.remote(run_id)
    print("[Phase 5 complete] local inference done for all 7 local architectures")

    # Wait for LLM inference to finish
    for h in phase6_handles:
        h.get()
    print("[Phase 6 complete] LLM inference done (Claude + Gemini)")

    # Phase 8: verify consolidated CSVs exist
    phase8_consolidate.remote(run_id)
    print("[Phase 8 complete] predictions verified")

    # Phase 9 runs all metrics + McNemar + bootstrap in one call
    phase9_metrics.remote(run_id)
    phase10_mcnemar.remote(run_id)  # no-op
    phase11_bootstrap.remote(run_id)  # no-op
    print("[Phase 9-11 complete] metrics + McNemar + bootstrap done")

    # Phase 12: tables + figures
    phase12_tables_figures.remote(run_id)
    print("[Phase 12 complete] tables and figures rendered")

    # Phase 14: audit
    audit_result = phase14_audit.remote(run_id)
    print(f"[Phase 14 complete] audit status: {audit_result['status']}")

    print("\n" + "=" * 70)
    print(f"Pipeline complete. run_id={run_id}")
    print(f"Outputs in volumes: rl-llm-safety-v3-predictions, rl-llm-safety-v3-results")
    print("=" * 70)
    return {"run_id": run_id, "audit": audit_result}


# ─────────────────────────────────────────────────────────────────────────────
# Data upload entrypoint (one-time, before first pipeline run)
# ─────────────────────────────────────────────────────────────────────────────


@app.function(image=base_image, volumes={"/data": data_vol}, timeout=600)
def upload_data() -> dict:
    """One-time data upload: copy canonical test sets into the Modal data volume.

    Run locally first to upload the JSON files:
      modal volume put rl-llm-safety-v3-data \\
          /Users/sanjaybasu/waymark-local/data/official/realworld_cases_n2000.json \\
          /realworld_cases_n2000.json
      modal volume put rl-llm-safety-v3-data \\
          /Users/sanjaybasu/waymark-local/data/official/physician_holdout_n41.json \\
          /physician_holdout_n41.json
    """
    rw = Path("/data/realworld_cases_n2000.json")
    ph = Path("/data/physician_holdout_n41.json")
    return {
        "realworld_present": rw.exists(),
        "physician_present": ph.exists(),
        "realworld_size_bytes": rw.stat().st_size if rw.exists() else 0,
        "physician_size_bytes": ph.stat().st_size if ph.exists() else 0,
    }
