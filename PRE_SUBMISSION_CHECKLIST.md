# Pre-Submission Checklist — rl-llm-safety v3 → BMC MIDM

Run this checklist before clicking submit on https://bmcmedinformdecismak.biomedcentral.com.

## Automated gates (all must exit 0)

```bash
cd /Users/sanjaybasu/waymark-local/packaging/rl_llm_safety_github_v3
python code/audit/test_metrics.py                          # 9/9 PASS
python code/audit/roc_monotonicity.py predictions/canonical/ --strict
python code/audit/regenerate_all_from_canonical.py --strict --predictions predictions/canonical/ --results results/
python code/audit/consistency_audit.py \\
    ../../notebooks/rl_vs_llm_safety_v3/drafts/main_text.md \\
    ../../notebooks/rl_vs_llm_safety_v3/drafts/appendix.md \\
    --strict
```

- [ ] Metrics self-test: 9/9 PASS
- [ ] ROC monotonicity: PASS for every architecture with calibrated probabilities
- [ ] Regenerate-all-from-canonical: PASS (every metric in results CSV re-derives from per-message predictions)
- [ ] Consistency audit: 0 errors, 0 required-pattern misses

## Manual cold-read (corresponding author)

- [ ] Verbatim editor-quotation NOT applicable (fresh submission to BMC MIDM, no prior decision letter)
- [ ] Cross-document numerical reconciliation: every abstract value found in corresponding table cell AND Discussion sentence
- [ ] Mathematical derivability: F1, MCC, PPV, NPV re-derive from sens/spec/prevalence for every architecture
- [ ] Figure binary content: every PNG visually consistent with caption claims and underlying CSV
- [ ] No orphan companion files in figures directory
- [ ] No `[UNVERIFIED]` markers anywhere
- [ ] No product names ("COMPASS", "Waymark COMPASS") in editor-facing text
- [ ] No JMIR-history leakage ("ms#94081", "Coristine", "Decision C/E2")
- [ ] No local path leakage (`/Users/sanjaybasu`, `waymark-local`)

## Implementation cardinality reconciliation

- [ ] If any released code uses `_9class` column naming, the manuscript Methods (or appendix Methods section) explicitly reconciles the 9-class implementation labeling with the 8-action documented scale. Cite the evidence: which label indices appear in ground truth, which in predictions.

## Reference list audit

- [ ] Every in-text citation `[N]` maps to a populated reference list entry at sequential position N
- [ ] Every reference list entry has at least one in-text citation
- [ ] Every journal citation web-search-verified via ≥3 independent sources (PubMed by title, PubMed by author+year+journal, CrossRef by DOI)
- [ ] Multi-author citations have author order verified against CrossRef metadata
- [ ] Volume/issue/pages verified for every journal citation
- [ ] Zero preprints where a peer-reviewed published version exists
- [ ] Appendix citations map to appendix's own reference list (if separate from main manuscript's)

## Per-message supplementary release (HIPAA verified)

- [ ] Supplementary CSV contains only: message_id, dataset, true_hazard, true_action, hazard_category, pred_proba, pred_hazard, pred_action, threshold_used, architecture, model_version, run_id, inference_time_s
- [ ] Supplementary CSV does NOT contain message text or any patient identifier
- [ ] Methods section describes the release model and the available-on-request path for message text
- [ ] Zenodo DOI obtained for archival; cited in Methods and Cover Letter

## BMC MIDM specifics

- [ ] Word count within journal limit (verify against author instructions; no hard cap but reviewer expectation ~5,000-7,000 for original research)
- [ ] Reporting framework explicit: TRIPOD-AI + DECIDE-AI; checklists in appendix
- [ ] Open-access APC ~$2,400 — funding source confirmed
- [ ] Cover letter (one page) describes pipeline architecture briefly; does NOT mention JMIR history
- [ ] Author byline matches submission portal metadata
- [ ] Conflict of interest statement complete
- [ ] Data availability statement complete (per-message CSV released; message text under DUA)

## Modal cleanup (post-submission)

- [ ] Keep `rl-llm-safety-v3-predictions` volume (canonical source for reviewer requests)
- [ ] Keep `rl-llm-safety-v3-results` volume (canonical source for any errata)
- [ ] DO NOT delete any non-owned Modal volumes or apps (ANCHOR, MIRA-3, etc.)
- [ ] Tag the pipeline state with git tag `rl-llm-safety-v3-bmcmidm-submission-YYYY-MM-DD` for future revert

## Sleep on it

After all of the above passes, do NOT submit immediately. Re-read the manuscript from a cold start the next day. If anything reads differently when fresh, investigate before submitting.
