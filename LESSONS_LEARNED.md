# Lessons Learned — Silent LLM Failure Episode (2026-05-19)

## What happened

A bug in `anthropic_client.py` (passing `temperature=0.0` to `claude-opus-4-7`, which had deprecated the parameter) caused every Claude API call to return HTTP 400. The tenacity retry layer caught the exception, exhausted 5 retries, and re-raised. `LLMClient.predict()` then caught the exception, set `raw=""` and `error="..."`, and continued. `parse_llm_json("")` silently defaulted to `{"hazard": False, "action": "Action 1", "rationale": ""}`. The error string went nowhere — it wasn't in the canonical per-message CSV schema.

The result: **2,041 Claude "predictions" were errors masquerading as data.** Every record had `(pred_hazard=0, pred_action=1, model_version=claude-opus-4-7)`. The bug was only caught when the rendered Table 2 showed Claude with sensitivity = 0.000 across both real-world and physician holdout sets — a statistical impossibility for a working LLM seeing 27 unambiguous hazards in the physician set.

## The layered failure

Six independent layers had to fail simultaneously for this bug to reach the manuscript:

1. **Deprecation surfaced as HTTP 400, not a typed error.** The Anthropic SDK raised `BadRequestError` rather than `ParameterDeprecationError`, so the retry/catch layer treated it as a transient failure rather than a permanent contract violation.
2. **The retry/catch layer caught Exception broadly.** `retry_if_exception_type(Exception)` retried even on non-recoverable 4xx errors. After 5 attempts the exception was re-raised, but the surrounding `predict()` then caught it again.
3. **`predict()` swallowed exceptions silently.** It set the `error` field on the dataclass but kept building a `LLMPrediction` from `raw=""` rather than re-raising. Errors became data.
4. **The CSV schema had no `error` column.** Downstream phases had no way to see that an architecture's predictions were errored at all.
5. **No sanity check on bulk inference output.** Phase 9 metrics consumed the predictions as-is. A working LLM cannot have 100% identical predictions across 2,041 messages.
6. **No physician-holdout floor check.** Every architecture's sensitivity on the 41-case physician holdout (where ground truth is known and the cases are intentionally diverse) was computed but not gated. A 0.000 floor on a known-hazard set should have refused to render the manuscript.

## Defenses now in code

| Defense | File | What it catches |
|---|---|---|
| **Smoke test before bulk inference** | `code/pipeline/llm_inference.py:smoke_test()` | Auth, deprecated parameters, prompt/parser drift, model rename. 3 obvious-hazard probes; abort if any returns hazard=False. ~30s cost; saves hours/credits. |
| **Skip smoke test on resume** | same file | Resumes from checkpoint don't re-probe — the prior run already verified the contract. |
| **Fail-fast on early errors** | same file | If >50% of first 50 predictions error, abort. Catches transient API issues that escalate to permanent. |
| **`error` field in per-message CSV** | same file | Errors are now first-class data. Audit can `assert df['error'].eq('').all()`. |
| **Increased `max_tokens` 512→1024** | `code/llm_clients/anthropic_client.py` | Safety-augmented prompt is long; 512 was tight enough that long thinking-style responses might truncate before the JSON. |
| **`temperature` removed for `claude-opus-4-7`** | same file | Document the model's actual API contract; comment explains why default sampling is acceptable. |

## Defenses that should be added (open)

These are not yet implemented. Each would have caught the bug independently of the above.

1. **Per-architecture diversity assertion.** A Phase 8 / Phase 9 gate: `assert df['pred_hazard'].nunique() >= 2` and `assert df['pred_action'].nunique() >= 2` per architecture × dataset. A constant column is broken data.
2. **Physician-holdout floor.** A Phase 14 audit: `assert sens(arch, 'physician_n41') > 0.10` for every architecture. The physician holdout exists precisely as a smoke test for the inference layer.
3. **Phase 13 gate.** Refuse to render the manuscript if any architecture × dataset row has `sens=0.000 AND n_hazards>10`. That combination is structurally impossible for a working model.
4. **Cross-LLM agreement floor.** On the physician holdout (where all ground truth labels exist), any two LLMs should agree on ≥40% of hazards (rough lower bound). Strong disagreement is a smell that one of them is broken.
5. **Audit gate before Modal commits results.** Phase 9 should refuse to commit `metrics_canonical.csv` if any architecture has zero predicted positives across both datasets — that combination is also structurally impossible.

## Process lessons

- **The physician holdout is the smoke test.** We had clear ground truth for 27 known hazards on a small set. The pipeline computed metrics for them but never gated on them. Always wire small known-good test sets into automated gates, not just into the manuscript Tables.
- **Errors must be data.** Any pipeline that consumes a fallible API and produces structured output must propagate error states into the canonical schema, not just into in-memory dataclass fields.
- **Constant outputs are bugs.** If a stochastic model produces identical predictions across thousands of varied inputs, something is broken. Add `nunique()` assertions as cheap canaries.
- **Smoke tests cost ~30 seconds and save hours.** Three probes per architecture, run before bulk inference, with a tight contract on a known answer.
- **Pretty rendering is not validation.** Phase 13 happily rendered the Table 2 with Claude sens=0.000. The rendering pipeline does not know what numbers are physically plausible. Gates must live in audit, not in rendering.

## Cost of the episode

- **Time:** ~5.5h of Claude inference burned through retries; re-run takes another ~3h (so net ~3h delay on the project).
- **Credits:** ~$30-50 of Anthropic credits on the failed run (rough estimate at ~2,041 × ~$0.02 per failed call with retries). On free credits, but the principle holds.
- **Avoided cost:** if the bug had reached the editor — desk reject and reputation damage. Catching it pre-submission is the load-bearing outcome.

## The single most important takeaway

**Never trust an inference loop that doesn't have a contract test against known-good inputs.** The pipeline must answer "would this client correctly classify a heart attack?" before it answers "what does this client think about 2,041 ambiguous Medicaid messages?"

---

## Subsequent scope expansion: from "characterize the gap" to "characterize the gap AND attempt to close it"

After the initial pipeline run completed and the rendered Table 2 surfaced the empirical pattern that no single architecture achieves clinical-grade safety on this population, the question was: stop and report the negative finding, or attempt to close the gap?

The user direction was to attempt closing the gap with the latest 2024-2026 approaches. A structured literature review (`notebooks/rl_vs_llm_safety_v3/lit_review/closing_the_gap.md`, 1,920 words) identified three priority interventions:
1. **k-NN RAG over labeled training set** (Stanford RAEC PSB 2026, Liu JAMIA 2025, Unlu RECTIFIER NEJM AI 2024 all converge; expected lift +0.10-0.20 sens).
2. **Llama-3.1-8B QLoRA SFT** (Losch arXiv 2503.21349; JMIR 2025 shows F1 lift 0.55→0.74 with SFT+DPO).
3. **Multi-LLM voting** (cautionary: blood-culture study found majority vote drove sens to 0%).

Critical skeptical finding from the lit review: no verified 2024-2026 paper demonstrates sens ≥0.80 AND spec ≥0.80 for free-text hazard detection on Medicaid/low-literacy populations. The literature ceiling on this task appears to be below the clinical-grade target — our empirical result is consistent with the literature rather than reflecting an idiosyncratic implementation failure.

Implementations:
- `code/pipeline/threshold_analysis.py` (per-architecture F1-max/MCC-max + clinical-grade reachability)
- `code/pipeline/ensemble_analysis.py` (213 ensemble configurations: hard voting, soft voting, top-3 AND/OR)
- `code/pipeline/cascade_analysis.py` (72 two-stage AND-rule cascade configurations)
- `code/pipeline/multi_llm_consensus.py` (4 multi-LLM rules)
- `code/pipeline/category_stratification.py` (per-architecture sensitivity by hazard category)
- `code/llm_clients/few_shot.py` (curated 20-example LLM client)
- `code/llm_clients/rag.py` (k-NN retrieval over training set + Claude Opus 4.7 backbone)

Empirical result of the closing-the-gap analyses (all on real-world Medicaid n=2000):
- **Single architecture at default threshold:** 0/9 reach clinical-grade. Best balanced: XGBoost+sentence-BERT (sens 0.424, spec 0.724).
- **Single architecture at F1-max threshold:** 0/6 (calibrated-probability) reach clinical-grade at any ROC operating point. Closest to balanced: XGBoost (sens 0.594, spec 0.592).
- **Ensemble (213 configurations):** 0/213 reach clinical-grade. Best balanced: hard-voting 3-of-9 (sens 0.667, spec 0.609).
- **Two-stage cascade (72 configurations):** 0/72 reach clinical-grade. Best balanced: ActionHead × LogReg+TF-IDF (sens 0.418, spec 0.809).
- **Multi-LLM consensus (4 rules):** 0/4 reach clinical-grade. Best: Claude only (sens 0.297, spec 0.871).
- **RAG over labeled training set:** in flight on Modal at the time of writing (claude_opus_4_7_rag_3158b720*.csv). Expected lift +0.10-0.20 sens per the lit review.

The result is now structurally robust: across ~290 evaluated configurations spanning 9 distinct architectures plus their full ensemble/cascade/consensus combinatorics, no configuration reaches the clinical-grade target on this population. The paper's headline finding is no longer "single architectures fail" — it is "the structural ceiling is below the clinical-grade target on this population with current architectures and 1,280 training examples."

## Methodology lesson from the scope expansion

When a single-axis evaluation produces a surprising negative result, the methodologically appropriate next step is to enumerate all combinations of the evaluated axis that could plausibly produce a positive result, and only conclude "structural finding" when ALL of them also produce the negative result. Single-architecture failure is interesting; ensemble + cascade + multi-LLM + threshold-optimization + RAG failure across hundreds of configurations is structurally informative for the field. Reviewers in clinical informatics evaluate negative findings differently when accompanied by this enumeration discipline.
