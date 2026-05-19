# [TEMPLATE] AI safety evaluation in an underrepresented population: real-world performance of clinical decision support and frontier language models on Medicaid patient messaging triage

**Authors**: TBD
**Target journal**: BMC Medical Informatics and Decision Making
**Reporting framework**: TRIPOD-AI (Collins 2024) + DECIDE-AI (Vasey 2022)

---

<!--
TEMPLATE NOTES (renderer skips HTML comments — these will not appear in output):

Every $-style or brace placeholder in this template renders from
results/metrics_canonical.csv, results/mcnemar_matrix.csv, or
results/delta_bootstrap_canonical.csv at build time via
code/manuscript_renderer.py. No hand-typed numbers anywhere.

Placeholder schema examples (illustrated using $-prefix here so the renderer
does not try to substitute them in the notes):
  $metrics.realworld_n2000.cql_sens_opt.sensitivity      → 0.NNN
  $metrics.realworld_n2000.cql_sens_opt.sensitivity_ci_lo → 0.NNN
  $mcnemar.cql_sens_opt.claude_opus_4_7_safety.chi2      → N.NN
  $delta.cql_sens_opt.sensitivity                         → ±NN pp
  $n_total, $n_hazards, $n_benigns, $prevalence_pct       → from run_manifest
-->


---

## Abstract (target: 350 words structured)

### Background

Clinical artificial intelligence safety evaluations have predominantly focused on academic medical center patient populations, scripted patient-actor scenarios, or standardized clinical knowledge benchmarks. The populations who most depend on artificial intelligence-assisted triage — those whose access to in-person primary care is constrained — have been systematically underrepresented in prior evaluations.

### Objective

This study evaluates the real-world safety of clinical artificial intelligence architectures, including a deployed decision support system and current-generation frontier large language models, on patient-initiated text messages from a Medicaid managed care population.

### Methods

A retrospective evaluation was conducted on {n_total} text messages submitted by Medicaid patients to a managed care entity during {study_period}. Each message was independently adjudicated by three physicians for hazard presence (binary), hazard category, and appropriate triage action (8-point ordinal scale). Architectures evaluated included a deployed decision support system (Conservative Q-Learning controller, constellation classifier, ActionHead recommender), classical supervised baselines (XGBoost with sentence-BERT embeddings, logistic regression with TF-IDF features, rule-based guardrails), and frontier large language models (Claude Opus 4.7 and Gemini 3.1 Pro with safety-augmented system prompts). The primary endpoint was sensitivity for hazard detection at the architecture's calibrated operating threshold, with full operating-curve analyses, action-recommendation appropriateness, and equity-stratified analyses as secondary endpoints. All results derive from a single canonical pipeline run (run identifier {run_id}) with per-message predictions for every architecture released as supplementary data.

### Results

The patient population differed substantially from prior clinical artificial intelligence evaluation cohorts. Messages averaged grade {grade_level_realworld} reading level (versus grade {grade_level_physician} in physician-scripted comparison scenarios), contained colloquialisms in {colloquialism_pct_realworld}% of messages (versus {colloquialism_pct_physician}%), and used abbreviations in {abbreviation_pct_realworld}% (versus {abbreviation_pct_physician}%). Hazard prevalence was {prevalence_pct}% ({n_hazards} hazards across {n_total} messages).

[Results placeholder — paragraphs render from canonical CSVs]

### Conclusions

[Conclusions placeholder]

---

## Introduction

[Population gap paragraph — citing the AMIE 2026 contrast and the underrepresented-population literature gap from Phase A]

[Deployed CDS validation context paragraph — generic architectural description, no product name]

[Study objectives paragraph]

---

## Methods

### Study design and population

This was a retrospective evaluation on text messages from {n_unique_patients} Medicaid managed care patients submitted between {study_period}. {n_total} messages comprised the held-out test set; {n_training_total} messages constituted the development set for model training and calibration.

Patient eligibility criteria: enrolled in the participating Medicaid managed care entity; aged 18 years or older; submitted at least one text message to the care coordination service.

### Outcomes

The primary endpoint was sensitivity for hazard detection at each architecture's calibrated operating threshold, with the 95% confidence interval computed by the Wilson score method.

Secondary endpoints: specificity (Wilson 95% confidence interval), positive predictive value, negative predictive value, F1 (bootstrap 95% confidence interval, 10,000 iterations, seed=42), Matthews correlation coefficient (bootstrap), area under the receiver operating characteristic curve (Hanley-McNeil 95% confidence interval where calibrated probabilities were available), false negatives per 1,000 messages, action-recommendation appropriateness (3-category: appropriate, under-triage, over-triage on the 8-point action scale), equity-stratified sensitivity by sex and self-reported race/ethnicity.

### Architectures evaluated

[Each architecture described generically; no product names]

1. **Rule-based guardrails**: 147 hand-crafted rules encoding symptom combinations, temporal keywords, and severity indicators. Each rule fires when a sentence-BERT [Devlin 2019; Reimers 2019] cosine similarity to its exemplar embeddings exceeds 0.75. Aggregated rule outputs are thresholded at a calibrated operating-point probability ({thresholds.guardrails}) selected on the training/validation split.

2. **Logistic regression with TF-IDF features**: Logistic regression with TF-IDF features (1-3 gram, max 10,000 features, sublinear TF scaling). Threshold {thresholds.logreg} calibrated on training/validation split.

3. **XGBoost with sentence-BERT embeddings**: XGBoost classifier on 768-dimensional sentence-BERT embeddings. Hyperparameters tuned via 5-fold cross-validation on the training set with random_state=42; final threshold {thresholds.xgboost}.

4. **Constellation architecture**: Multinomial logistic regression over 23 hazard categories on sentence-BERT embeddings, with the per-category probabilities aggregated into a single binary hazard probability via the maximum-activation rule. Threshold {thresholds.constellation}.

5. **Conservative Q-Learning controller (sensitivity-optimized)**: Conservative Q-Learning [Kumar 2020] on calibrated probability vectors from an upstream sentence-BERT classifier. The 23-dimensional state space encodes hazard-category probabilities. The action space comprises 8 discrete escalation actions (Action 1 = self-care; Action 8 = emergency services). Asymmetric reward: +10 for correct action, -50 for missed hazard, -2 for false alarm. The sensitivity-optimized variant applies a calibrated probability threshold ({thresholds.cql_sens_opt}) chosen to maximize sensitivity subject to specificity ≥ 0.70 on the validation set.

6. **Conservative Q-Learning controller (reward-optimized)**: Same trained Q-function as variant 5; predictions taken as the Q-function argmax over the 8 actions.

7. **ActionHead**: Independent sentence-BERT + multinomial logistic regression action classifier trained for 8-class action prediction on the 570-example training set.

8. **Claude Opus 4.7 (safety-augmented)**: Anthropic's Claude Opus 4.7 large language model. System prompt explicitly prioritizes patient safety and emphasizes erring toward escalation. Temperature 0; deterministic.

9. **Gemini 3.1 Pro Preview (safety-augmented)**: Google's Gemini 3.1 Pro Preview large language model. Same safety-augmented prompt structure as Claude; temperature 0.

### Single-turn screening evaluation

All architectures were evaluated in single-turn mode: each architecture received one patient message as input and produced one hazard prediction and one action recommendation. This corresponds to the "screening at message arrival" clinical use case — the decision the system makes upon receipt of an incoming message before any clarifying conversation occurs. Multi-turn dialogue extensions are out of scope for this analysis (see Limitations) and identified as Future Work.

### Statistical analysis

All paired comparisons used McNemar's test [McNemar 1947] restricted to hazard-positive cases, with Hochberg step-up correction for family-wise error rate control across the {k_pairs} pairwise comparisons.

Cross-set sensitivity comparisons used parametric bootstrap (10,000 iterations, seed=42) with sensitivities resampled independently from binomial distributions parameterized by observed proportions and sample sizes.

Stratified equity analyses used the same Wilson confidence intervals within each subgroup, with the equalized odds difference (maximum minus minimum subgroup sensitivity) reported.

### Reporting framework and reproducibility

The study follows TRIPOD-AI [Collins 2024] and DECIDE-AI [Vasey 2022] reporting guidelines. The complete TRIPOD-AI checklist appears in Appendix A.

All results derive from a single canonical pipeline run (run identifier {run_id}, conducted on {run_date}). Per-message predictions for every architecture × every test record are released as Supplementary File 2 (one CSV per architecture, containing the columns `message_id, dataset, true_hazard, true_action, hazard_category, pred_proba, pred_hazard, pred_action, threshold_used, architecture, model_version, run_id, inference_time_s`). Patient message text remains under HIPAA data-use agreement and is available to bona-fide researchers on request to the corresponding author. The released supplementary file is sufficient for any reviewer to independently verify receiver operating characteristic monotonicity, McNemar discordant-pair counts, paired bootstrap confidence intervals, and table-to-table arithmetic.

The code repository (https://github.com/sanjaybasu/rl-llm-safety) contains the complete pipeline as deployed: a single Modal-orchestrated entry point (`modal_pipeline.py`) that, in one detached run, produces every per-message prediction file and every reported metric.

### Ethics and oversight

This study was conducted under the [insert IRB/Privacy & Compliance determination]. Patient consent was waived under [insert basis]. No identifiable patient data are reproduced in the manuscript or released supplementary files.

---

## Results

### Population characteristics (Table 1)

[Table 1: demographics + linguistic characteristics — every cell renders from canonical CSV]

### Hazard detection performance (Table 2)

[Table 2: per-architecture sensitivity, specificity, PPV, NPV, F1, MCC, AUROC, FN/1000 with 95% CIs]

### Operating-point analysis (Table 3)

[Table 3: sensitivity at matched specificity, with achieved specificity displayed alongside target — derived from the SAME prediction files as Table 2 by thresholding the operating curve at different points]

### Paired comparisons (Figure 1 + Multimedia Appendix Table S5)

[Figure 1: sensitivity comparison between deployed CDS and frontier LLMs, with McNemar paired-test annotations]

### Action recommendation appropriateness (Table 4)

[Table 4: per-architecture appropriate / under-triage / over-triage rates, grouped by mapping mechanism]

### Equity-stratified analysis

[Sex-stratified, race-stratified sensitivity with Wilson CIs and equalized odds differences]

---

## Discussion

### Principal findings

[Population gap is the lead]

### Comparison with prior work

[Differentiation from Saab/AMIE 2026 — different population, different evaluation paradigm]

### Implications for deployment

[CDS deployment context; FDA Clinical Decision Support guidance]

### Strengths and limitations

[Single-turn screening, message-level split, single-program retrospective]

### Future work

[Multi-turn dialogue extensions; temporal validation; external validation in non-Medicaid populations]

---

## Conclusions

[1-paragraph conclusion]

---

## Declarations

### Ethics approval and consent to participate

[Insert]

### Consent for publication

Not applicable.

### Availability of data and materials

The per-message predictions for every architecture × every test record are released as Supplementary File 2 (also archived at Zenodo, DOI {zenodo_doi}). The code repository is at https://github.com/sanjaybasu/rl-llm-safety. Patient message text is governed by a HIPAA data-use agreement and is available to bona-fide researchers on request to the corresponding author.

### Competing interests

[Insert]

### Funding

[Insert]

### Authors' contributions

[Insert]

### Acknowledgements

[Insert]

---

## References

[References render via citation-management skill with anti-fabrication protocol]
