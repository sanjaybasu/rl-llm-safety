# Project Roadmap — rl-llm-safety v3

**Last updated:** 2026-05-19
**Working directory:** `/Users/sanjaybasu/waymark-local/packaging/rl_llm_safety_github_v3/`
**Drafts output directory:** `/Users/sanjaybasu/waymark-local/notebooks/rl_vs_llm_safety_v3/`

---

## North Star

Resubmit the JMIR-rejected manuscript (decision E2, 2026-05-14) to **BMC Medical Informatics and Decision Making**, reframed as the first real-world AI safety evaluation in a Medicaid patient-messaging population. Every reported number must derive from one single canonical Modal pipeline run with per-message predictions released as supplementary data so any reviewer can independently re-derive every table.

---

## Status dashboard

| Phase | Status | Output |
|---|---|---|
| **A. Literature review** | ✅ Complete | `notebooks/rl_vs_llm_safety_v3/lit_review/evidence_map.md` (16 verified citations + gap claim) |
| **B. Design lock** | ✅ Complete | `notebooks/rl_vs_llm_safety_v3/protocol/protocol.md` (6-config matrix, TRIPOD+AI + DECIDE-AI) |
| **C. Pipeline execution** | ✅ Complete | Modal volume `rl-llm-safety-v3-predictions:canonical/*_3158b720*.csv` (9 architectures × 2 datasets = 18,369 predictions) |
| **C.5 Manuscript rendering** | ✅ Complete | `results_modal_pull/drafts/{main_text,cover_letter,appendix}.md` + `supplementary/multimedia_appendix_2.zip` |
| **D. Audit + concordance** | ✅ Complete | All 9 strict concordance checks pass with 0 errors, 0 warnings |
| **E. Submission** | ⏳ Pending | Cold-read → fill author byline → upload to BMC MIDM portal |

---

## What is actively running on Modal right now

**Nothing.** All ephemeral pipeline apps have stopped successfully:

| App ID | Function | Final state |
|---|---|---|
| `ap-KZ1pcEAcDVl7CQpz9zPKHk` | `resume_pipeline` | ✅ Stopped successfully — produced all 9 architecture × 2 dataset CSVs + metrics |
| `ap-glFWuCAiRVhuS9l1rHEYOz` | `wait_and_render` | ✅ Stopped after auto-triggering Phase 13 |
| `ap-UhitbamvtyEvsJwQ9VFGQn` | `phase13_render_manuscript` (manual re-run after template patches) | ✅ Stopped successfully |
| `ap-lIO7GUwSOBiXAj03UpdtCY` | Deployed app (idle) | Function definitions; not consuming resources |

Local monitor `b5a1tbj4a` will keep watching but its job is done.

---

## Decisions locked (do not revisit without explicit reason)

- **Target journal**: BMC Medical Informatics and Decision Making (no JMIR re-attempt)
- **Framing primary**: AI safety evaluation in an underrepresented Medicaid population
- **Framing secondary**: Real-world validation of a deployed clinical decision support architecture (product name NOT used)
- **Turn structure**: Single-turn screening-at-arrival (multi-turn out of scope; documented as Future Work)
- **Architecture matrix (6 top-level rows)**:
  1. Conservative Q-Learning controller (sens-opt + reward-opt + ActionHead detail)
  2. Constellation architecture
  3. Rule-based guardrails
  4. Logistic regression + TF-IDF
  5. XGBoost + sentence-BERT
  6. Frontier LLMs: Claude Opus 4.7 (safety) + Gemini 3.1 Pro Preview (safety)
- **No paid OpenAI** (GPT-5.5 omitted from default scope; on free Anthropic + Google credits only)
- **No independent statistician** (BMC MIDM does not require)
- **Per-message predictions released** as supplementary (HIPAA-compliant: opaque CUIDs, no message text)
- **Canonical run_id**: `3158b720-8638-4a1d-8faf-ebf391012ed9` (locked; used for all downstream filtering)
- **Modal namespacing**: app `rl-llm-safety-v3-pipeline`, volumes `rl-llm-safety-v3-{data,predictions,results}`

---

## Key artifacts and where they live

| Artifact | Path | Source of truth |
|---|---|---|
| Pipeline code (Phase 0-14) | `code/pipeline/modal_pipeline.py` | This repo |
| LLM clients | `code/llm_clients/*.py` | This repo |
| Audit suite | `code/audit/*.py` | This repo |
| Manuscript templates | `manuscript_template/{main_text,cover_letter,appendix}_template.md` | This repo |
| Pre-flight resilience doc | `RESUME_AFTER_OUTAGE.md` | This repo |
| Failure post-mortem | `LESSONS_LEARNED.md` | This repo |
| Pre-submission checklist | `PRE_SUBMISSION_CHECKLIST.md` | This repo |
| Lit review evidence map | `../../notebooks/rl_vs_llm_safety_v3/lit_review/evidence_map.md` | Notebooks |
| Pre-registration protocol | `../../notebooks/rl_vs_llm_safety_v3/protocol/protocol.md` | Notebooks |
| Per-message predictions | Modal volume `rl-llm-safety-v3-predictions:canonical/*_3158b720*.csv` | Modal |
| Canonical metrics | Modal volume `rl-llm-safety-v3-results:{metrics_canonical.csv, mcnemar_matrix.csv, ...}` | Modal |
| Rendered drafts (after Phase 13) | Modal volume `rl-llm-safety-v3-results:drafts/{main_text,cover_letter,appendix}.md` | Modal |
| Supplementary ZIP (after Phase 13) | Modal volume `rl-llm-safety-v3-results:supplementary/multimedia_appendix_2.zip` | Modal |
| Canonical test data | `../../data/official/{realworld_cases_n2000,physician_holdout_n41}.json` | Waymark local |

---

## Next steps (ordered)

### When the pipeline finishes (~30 min from snapshot)

1. **Verify both Modal apps stopped:**
   ```bash
   modal app list | grep rl-llm-safety-v3
   ```
   Both `ap-KZ1pcE...` and `ap-glFWuC...` should show `stopped`.

2. **Pull final artifacts to local:**
   ```bash
   cd /Users/sanjaybasu/waymark-local/packaging/rl_llm_safety_github_v3
   rm -rf results_modal_pull
   mkdir results_modal_pull
   modal volume get rl-llm-safety-v3-results / results_modal_pull/ --force
   ```

3. **Sanity-check Claude this time:**
   ```bash
   python3 -c "
   import pandas as pd
   df = pd.read_csv('results_modal_pull/predictions_filtered/claude_opus_4_7_safety_3158b720-8638-4a1d-8faf-ebf391012ed9.csv')
   ph = df[df.dataset=='physician_n41']
   print('Claude physician hazard pred distribution:')
   print(pd.crosstab(ph['true_hazard'], ph['pred_hazard'], margins=True))
   print('Claude pred_action diversity:', df['pred_action'].nunique(), 'unique values')
   "
   ```
   Expect: pred_hazard has both 0s and 1s; pred_action has at least 3+ unique values; physician hazards 1/27 predicted captures > 0.

4. **Cold-read the rendered manuscript:**
   - `results_modal_pull/drafts/main_text.md`
   - `results_modal_pull/drafts/cover_letter.md`
   - `results_modal_pull/drafts/appendix.md`

5. **Run strict comprehensive concordance check:**
   ```bash
   cp results_modal_pull/drafts/*.md ../../notebooks/rl_vs_llm_safety_v3/drafts/
   cp -r results_modal_pull/* results/ 2>/dev/null
   python3 code/audit/comprehensive_concordance_check.py \
       --drafts-dir ../../notebooks/rl_vs_llm_safety_v3/drafts \
       --strict
   ```

6. **Fill remaining submission-time placeholders:**
   - Authors byline (`main_text.md` line 3): "TBD"
   - Authors' contributions (Declarations): "TBD at submission"
   - Zenodo DOI (`{zenodo_doi}` placeholder): obtain at archive upload
   - 4 suggested reviewer email addresses (cover letter): up to author

7. **Push to GitHub:** deferred due to `gh auth status` mismatch (`sanjaybasu-waymark` is authenticated; repo target is `github.com/sanjaybasu/rl-llm-safety` personal). User to resolve.

8. **Submit to BMC MIDM portal:** https://bmcmedinformdecismak.biomedcentral.com

---

## Open risks and known issues

| Risk | Mitigation | Status |
|---|---|---|
| Modal Phase 6 worker timeout | Checkpoint every 100 msgs + `resume_pipeline` entrypoint | ✅ Active |
| Silent LLM API failures | Smoke test pre-flight + `error` field in CSV + fail-fast >50% errors in first 50 | ✅ Active in deployed code (next run benefits) |
| Local laptop sleep / restart | All work on Modal volumes; recovery via `RESUME_AFTER_OUTAGE.md` | ✅ Active |
| Stale local kaches confusing monitors | Monitor uses `modal volume get --force` to overwrite | ✅ Active |
| Phase 13 rendering nonsense numbers if metrics are broken | **OPEN** — see `LESSONS_LEARNED.md` "Defenses still open" section #3 | ⚠️ Manual cold-read required |
| `gh auth status` mismatch | Manual push by user with correct auth | ⏳ User action |

---

## Episode log (irreversible facts)

| Date | Event |
|---|---|
| 2026-05-14 | JMIR Medical Informatics decision E2 (final rejection of #94081 after 5 revisions) |
| 2026-05-18 | Phase A literature review complete; Phase B design lock; pipeline scaffolding deployed; first Phase 6 launched |
| 2026-05-19 (early) | First Phase 6 hit 7200s timeout with 0 LLM CSVs (no checkpointing in old code) |
| 2026-05-19 | Resilient Phase 6 (checkpoint+resume) added; timeout raised 7200→43200s; resume_pipeline launched |
| 2026-05-19 | Phase 13 (manuscript rendering) added to Modal; wait_and_render orchestrator launched |
| 2026-05-19 | **CLAUDE BUG SURFACED**: temperature deprecated for `claude-opus-4-7`; all 2,041 predictions were silent errors |
| 2026-05-19 | Claude bug fixed (commit `1ef4cfb`); broken CSV cleaned from volume; re-run launched |
| 2026-05-19 | Smoke test pre-flight + LESSONS_LEARNED.md added (commit `6f03e30`) for future protection |
| 2026-05-19 | Phase 6 re-run completed cleanly with fixed Claude client (Claude 2041/2041 + Gemini skip-resumed); Phase 9-12-14 + auto Phase 13 chain produced full draft set |
| 2026-05-19 | Strict concordance check found 4 final errors (orphan ref [9], cover-letter missing Wilson/Hochberg/multi-turn) — all patched in templates (`2d6f6e4`), Phase 13 re-run, **all 9 strict checks now pass with 0 errors 0 warnings** |
| 2026-05-19 | Cold-read feedback: single-architecture matrix is sterile. Added cascade analysis (Stage 1 high-recall + Stage 2 high-precision) as the scientific contribution. New `cascade_analysis.py` computes 72-pair AND-rule matrix; Table 5 + Table S3 added. Implicit Compass framing: asymmetric-reward CQL controller is the natural Stage 1 screen. (`bb6814a`) |
| 2026-05-19 | Author list updated to JMIR MI #94081 lineage with three cuts + Bernardo Arevalo added: Basu, Patel, Sheth, Arevalo, Morgan, Batniji. Tables/figures embedded at end of main text + appendix per /submit-prep convention. All 9 strict concordance checks still pass. (`bb6814a`) |

---

## Sub-agent / next-session context

If a new Claude session or sub-agent is reading this:
1. Read `LESSONS_LEARNED.md` first — it explains why we have certain defenses in code.
2. Read `RESUME_AFTER_OUTAGE.md` to recover from any disruption.
3. Read this ROADMAP for the current state of play.
4. Do NOT relaunch pipelines without verifying the in-flight ones are actually done (`modal app list`).
5. Canonical run_id is `3158b720-8638-4a1d-8faf-ebf391012ed9`. Always filter to this run_id.
6. Modal volume namespacing: only touch volumes prefixed `rl-llm-safety-v3-*`.
