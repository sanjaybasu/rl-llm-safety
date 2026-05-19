# rl-llm-safety v3 — Fresh-Take Pipeline

Single canonical end-to-end pipeline for the clinical AI safety evaluation paper. Built to resolve the JMIR Medical Informatics Decision E2 (2026-05-14) structural diagnosis: every reported number in the manuscript must derive from ONE canonical per-message prediction file per architecture, with every table and figure thresholding the same prediction file across the operating curve.

## Architecture

```
notebooks/rl_vs_llm_safety_v3/        ← manuscript artifacts (drafts, lit review, figures)
packaging/rl_llm_safety_github_v3/    ← code + data + supplementary release
├── code/
│   ├── pipeline/
│   │   ├── modal_pipeline.py         ← single Modal orchestrator (14 phases)
│   │   ├── local_training.py         ← canonical training module
│   │   └── manuscript_renderer.py    ← template → manuscript with placeholder substitution
│   ├── llm_clients/
│   │   ├── anthropic_client.py       ← Claude Opus 4.7 (primary frontier comparator)
│   │   ├── gemini_client.py          ← Gemini 3.1 Pro Preview (secondary)
│   │   ├── openai_client.py          ← GPT-5.5 (optional, paid)
│   │   ├── deepseek_client.py        ← DeepSeek-R1 (optional, local Ollama)
│   │   ├── prompts.py                ← canonical safety + default system prompts
│   │   └── base.py                   ← shared client interface
│   └── audit/
│       ├── metrics.py                ← canonical metrics (Wilson, bootstrap, AUROC, McNemar)
│       ├── roc_monotonicity.py       ← ROC monotonicity assertion
│       ├── regenerate_all_from_canonical.py  ← single-pass audit gate
│       ├── consistency_audit.py      ← forbidden-pattern linter
│       └── test_metrics.py           ← metrics self-test (9/9 PASS)
├── manuscript_template/
│   └── main_text_template.md         ← parameterized template (every number is {placeholder})
├── predictions/canonical/            ← per-architecture per-message CSVs (release as supp.)
└── results/                          ← canonical metrics CSVs (single source of truth)
```

## Run order

```bash
# Phase A: literature review (one-time, ~30-60 min)
# Launched via Claude Code agent; output: notebooks/rl_vs_llm_safety_v3/lit_review/evidence_map.md

# Phase B: design lock (one-time, ~1 day after Phase A)
# Output: notebooks/rl_vs_llm_safety_v3/protocol/protocol.md (pre-registration)

# Phase C: pipeline execution (~3 hours Modal wall-clock; detached recommended)
modal volume put rl-llm-safety-v3-data \\
    /Users/sanjaybasu/waymark-local/data/official/realworld_cases_n2000.json \\
    /realworld_cases_n2000.json
modal volume put rl-llm-safety-v3-data \\
    /Users/sanjaybasu/waymark-local/data/official/physician_holdout_n41.json \\
    /physician_holdout_n41.json
modal run --detach code/pipeline/modal_pipeline.py::orchestrate

# Phase D: manuscript render + audit (~3-5 days)
python code/pipeline/manuscript_renderer.py \\
    --template manuscript_template/main_text_template.md \\
    --predictions predictions/canonical/ \\
    --results results/ \\
    --output ../notebooks/rl_vs_llm_safety_v3/drafts/main_text.md \\
    --strict

# Audit gates (all must pass before submission)
python code/audit/test_metrics.py                          # 9/9 PASS expected
python code/audit/roc_monotonicity.py predictions/canonical/ --strict
python code/audit/regenerate_all_from_canonical.py --strict
python code/audit/consistency_audit.py \\
    ../notebooks/rl_vs_llm_safety_v3/drafts/main_text.md \\
    ../notebooks/rl_vs_llm_safety_v3/drafts/appendix.md \\
    --strict
```

## Modal namespacing (concurrency safety)

This pipeline shares a Modal account with concurrent ANCHOR and MIRA-3 jobs run by other Claude Code sessions. Strict isolation rules:

- **App name**: `rl-llm-safety-v3-pipeline` (never reuse another app's name)
- **Volume names**: `rl-llm-safety-v3-data`, `rl-llm-safety-v3-predictions`, `rl-llm-safety-v3-results`, `rl-llm-safety-v3-models` (prefix `rl-llm-safety-v3-*`)
- **Forbidden**: `modal app stop` on any non-owned app; `modal volume rm` on any volume; modifying volumes not prefixed `rl-llm-safety-v3-*`

## Target

- **Journal**: BMC Medical Informatics and Decision Making
- **Framing primary**: AI safety evaluation in an underrepresented Medicaid population (low-literacy, multilingual, SMS-substituting-for-PCP)
- **Framing secondary**: real-world validation of a deployed CDS architecture (product name not used)
- **Reporting frameworks**: TRIPOD-AI + DECIDE-AI
- **Per-message supplementary**: predictions by `message_id` (no patient text) released alongside the manuscript

## Cost

| Phase | Item | Cost |
|---|---|---|
| A | Literature review (Anthropic API for agent) | ~$200 |
| C | Modal compute (local training + orchestration) | ~$20 |
| C | Claude Opus 4.7 inference (Anthropic credits) | $0 |
| C | Gemini 3.1 Pro inference (Google credits) | $0 |
| **Total** | | **~$220 baseline** |

If GPT-5.5 row is added (Phase B may decide based on lit review): +$25–75 paid OpenAI.

## What's verified

- All 9 metrics tests pass: Wilson CIs, AUROC vs sklearn, F1/MCC vs sklearn, McNemar vs statsmodels, Hochberg step-up, bootstrap reproducibility (seed=42), ROC monotonicity on synthetic data.
- Test set verified: 2,000 real-world records, 165 hazards / 1,835 benigns; 41 physician records, 27 hazards / 14 benigns; SHA-256 checksums in run manifest.
- Data NOT in any public GitHub repo (HIPAA compliance verified).
- Modal scaffolding follows the validated ANCHOR pattern (`packaging/anchor/modal_train.py`).

## Reference plan

`/Users/sanjaybasu/.claude/plans/sharded-sleeping-neumann.md`
