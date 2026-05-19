# Multimedia Appendix 1 — Supplementary Methods, Tables, and Figures

This appendix accompanies the main manuscript titled "AI safety evaluation in an underrepresented Medicaid population: real-world performance of clinical decision support and frontier language models on patient messaging triage."

Every numerical value in this appendix renders from canonical CSVs at build time via `code/pipeline/manuscript_renderer.py`. No hand-typed numbers.

---

## A. TRIPOD+AI checklist

Following Collins et al. 2024 (BMJ 2024;385:e078378). The TRIPOD+AI items below map each reporting requirement to the corresponding location in the main manuscript.

| Item | Topic | Reporting location |
|---|---|---|
| 1a, 1b | Title — identify as AI prediction model study; specify development/validation | Main text, Title |
| 2 | Structured abstract — relevant information; key results | Main text, Abstract |
| 3a, 3b | Background — clinical context, rationale; objectives | Main text, Introduction |
| 4 | Source of data — type (retrospective); key dates | Main text, Methods §"Study design and population" |
| 5a, 5b, 5c | Participants — setting/eligibility; outcome definition; predictor definition | Main text, Methods §"Study design and population", §"Outcomes" |
| 6a, 6b | Outcome — definition and assessment; censoring/missing | Main text, Methods §"Outcomes" |
| 7a, 7b | Predictors — definition; coding | Main text, Methods §"Architectures evaluated" |
| 8 | Sample size — calculation and rationale | Main text, Methods §"Study design and population" |
| 9 | Missing data — handling | Main text, Methods §"Outcomes"; Appendix §C.4 |
| 10a, 10b, 10c, 10d | Model development — algorithm, hyperparameter selection, model specification, training/calibration | Main text, Methods §"Architectures evaluated"; Appendix §C |
| 10e | Software and computing environment | Code repository; Methods §"Reporting framework and reproducibility" |
| 11 | Risk groups — handling, transparency | Main text, Results §"Equity-stratified analysis" |
| 12 | Model evaluation — performance measures; calibration; uncertainty | Main text, Methods §"Statistical analysis"; Results §"Hazard detection performance", §"Operating-point analysis" |
| 13a, 13b, 13c | Participants — descriptive statistics; outcome distribution; missing | Main text, Results §"Population characteristics"; Table 1 |
| 14a, 14b | Model development — final model; performance | Main text, Results §"Hazard detection performance"; Tables 2-3 |
| 15 | Model performance — measures with uncertainty | Main text, Tables 2-3; Appendix Tables S1-S2 |
| 16 | Model updating | Not applicable to this evaluation |
| 17 | Interpretation — performance in context | Main text, Discussion §"Principal findings", §"Comparison with prior work" |
| 18 | Limitations | Main text, Discussion §"Strengths and limitations" |
| 19 | Generalizability — applicability to other settings | Main text, Discussion §"Strengths and limitations", §"Future work" |
| 20 | Supplementary information — protocol, data, code | Code repository; Supplementary File 2 (per-message predictions); Methods §"Reporting framework and reproducibility" |
| 21 | Funding | Main text, Declarations §"Funding" |
| AI 1-7 | AI-specific items — model architecture, fairness, transparency, reproducibility, interpretability | Methods §"Architectures evaluated"; Appendix §C; Results §"Equity-stratified analysis"; code repository |

## B. DECIDE-AI checklist

Following Vasey et al. 2022 (BMJ 2022;377:e070904). The DECIDE-AI items below map each reporting requirement to the corresponding location in the main manuscript.

| Item | Topic | Reporting location |
|---|---|---|
| 1 | Title — identify as early-stage clinical evaluation of AI-driven decision support | Main text, Title |
| 2 | Structured abstract | Main text, Abstract |
| 3 | Rationale — describe scientific background and clinical problem | Main text, Introduction |
| 4 | Objectives — specify the objectives of the early-stage clinical evaluation | Main text, Introduction final paragraph |
| 5 | Trial registration or protocol availability | Not applicable (retrospective evaluation) |
| 6 | Study design — describe the study design with reference to design type | Main text, Methods §"Study design and population" |
| 7a, 7b | Eligibility criteria, settings, and locations | Main text, Methods §"Study design and population" |
| 8a, 8b | Intervention — describe the AI system and decision support intervention | Main text, Methods §"Architectures evaluated"; Appendix §C |
| 9 | Standards — describe the development and validation standards used | Main text, Methods §"Reporting framework and reproducibility" (TRIPOD+AI, DECIDE-AI) |
| 10a, 10b, 10c | Outcomes — primary, secondary, and safety outcomes | Main text, Methods §"Outcomes" |
| 11 | Sample size determination | Main text, Methods §"Study design and population" |
| 12 | Statistical methods | Main text, Methods §"Statistical analysis" |
| 13 | Data quality assurance — describe quality assurance procedures for inputs/outputs | Main text, Methods §"Ethics and oversight"; Appendix §C.5 |
| 14 | Implementation — describe the level of integration with the clinical workflow | Not directly evaluated — retrospective design without prospective workflow integration; addressed in Discussion §"Implications for deployment" |
| 15 | Human-AI interaction — describe interaction modality | Main text, Methods §"Architectures evaluated"; Discussion §"Strengths and limitations" |
| 16 | Modifications during the study — describe any modifications to AI or protocol | None; single canonical pipeline run as pre-registered |
| 17 | Errors and adverse events — describe any errors and how they were handled | Main text, Methods §"Statistical analysis"; Results §"Hazard detection performance" (false-negative reporting) |
| 18 | Participant flow | Main text, Methods §"Study design and population"; Results §"Population characteristics" |
| 19 | Baseline characteristics | Main text, Results §"Population characteristics"; Table 1 |
| 20 | Outcomes and estimation | Main text, Results §"Hazard detection performance", §"Operating-point analysis", §"Action recommendation appropriateness" |
| 21 | Subgroup analyses | Main text, Results §"Equity-stratified analysis"; Appendix Table S4 |
| 22 | Adverse events — report and discuss | Main text, Results §"Hazard detection performance" (false negatives per 1,000 messages); Discussion §"Principal findings" |
| 23a, 23b | Limitations and generalisability | Main text, Discussion §"Strengths and limitations" |
| 24 | Interpretation — in context of available evidence | Main text, Discussion §"Comparison with prior work" |
| 25 | Implications — clinical, research, and policy | Main text, Discussion §"Implications for deployment", §"Future work" |
| 26 | Future research priorities | Main text, Discussion §"Future work" |
| 27 | Funding, conflicts, and registration | Main text, Declarations

## C. Architecture detail

### C.1 Action scale

The action recommendation endpoint uses an 8-class ordinal scale, with Action 1 representing the least urgent action and Action 8 the most urgent:

1. Self-care guidance / no action needed (least urgent)
2. Routine appointment within 2 weeks
3. Routine prescription refill or follow-up scheduling
4. Urgent care within 24-48 hours
5. Same-day appointment scheduling
6. Urgent prescription refill same day
7. Urgent specialist or physician callback within 1 hour
8. Emergency services (911/ED immediately, most urgent)

The implementation encodes labels in 9 classes (indices 0 through 8) for compatibility with the trained classifier output dimensions. Label index 0 does not appear in any ground-truth label or any model prediction in the released test set. Released supplementary prediction CSVs use the 8-action scale (Action 1 through Action 8); the codebase's `pred_action_9class` column naming reflects the implementation cardinality while the documented scale is 8 actions.

### C.2 Conservative Q-Learning controller

The Conservative Q-Learning (Kumar A et al., NeurIPS 2020) controller maps the 23-dimensional calibrated hazard probability vector (output of the upstream sentence-BERT + multinomial logistic detector with temperature scaling, T={cql_temperature}) to one of the 8 escalation actions. The reward function is asymmetric: +10 for a correct action, -50 for a missed hazard, -2 for a false alarm. The expected reward calculation at inference time selects the action that maximizes expected reward.

The sensitivity-optimized variant applies a calibrated probability threshold ({thresholds.cql_sens_opt}) chosen on the training/validation split to maximize sensitivity subject to specificity at least 0.70. The reward-optimized variant applies the threshold derived analytically from the reward weights: P(hazard) ≥ 2/62 ≈ 0.032.

### C.3 Constellation architecture

A multinomial logistic regression on sentence-BERT embeddings classifies messages into one of 20 observed hazard categories. The binary hazard probability is derived as 1 − P(benign). Threshold for binary hazard prediction: {thresholds.constellation}.

### C.4 Rule-based guardrails (simplified implementation)

The deployed production system uses 147 hand-crafted rules. The implementation released here uses a 14-rule subset for reproducibility tractability; the simplified version is documented in Methods §3.4 of the main text with explicit acknowledgement of the simplification. The 14 rule families cover chest pain, stroke signs, anaphylaxis, suicidality, overdose, obstetric emergency, severe pain, breathing distress, neurological emergency, self-harm, substance overdose, falls, medication errors, and behavioral emergency triggers.

### C.5 Calibration and temperature scaling

The upstream calibrated detector applies temperature scaling on the validation set after multinomial logistic regression training. The optimal temperature minimizes negative log-likelihood on the calibration set ({n_val} held-out examples).

---

## D. Supplementary tables

### Table S1: Physician-holdout metrics with TP/FN/TN/FP per architecture

[Rendered from `results/tables/tableS1_physician_holdout_metrics.csv`]

### Table S2: Δ sensitivity (physician → real-world) with parametric bootstrap CIs

[Rendered from `results/tables/tableS2_delta_bootstrap.csv`]

### Table S3: Hazard category stratification (real-world test set)

[Per-category sensitivity for the deployed CDS architecture, with Wilson CIs. Only categories with n_hazard ≥ 3 reported.]

### Table S4: Equity-stratified sensitivity by self-reported demographics

[Sex-stratified and race-stratified sensitivity with Wilson CIs and equalized odds differences. Demographic data available for {n_demographic_available} of 2,000 test messages.]

### Table S5: Full pairwise McNemar matrix with Hochberg correction

[Rendered from `results/mcnemar_matrix.csv`. All k = 15 pairwise comparisons among the 6 top-level architectures, chi-squared statistics, raw p-values, Hochberg significance.]

### Table S6: Calibrated operating-point thresholds per architecture

[Rendered from `results/thresholds.json`. Per-architecture calibrated decision threshold selected on training/validation split.]

---

## E. Supplementary figures

### Figure S1: Operating-curve overlays for all calibrated-probability architectures

[Receiver-operating-characteristic curves for every architecture with calibrated probability outputs. Curves derived from operating_curves.csv in the canonical pipeline output.]

### Figure S2: Calibration curves and reliability diagrams

[Per-architecture calibration plots showing predicted versus observed hazard frequencies in deciles.]

### Figure S3: Hazard-category sensitivity bar chart

[Per-category sensitivity for the deployed CDS architecture, sorted by category prevalence.]

---

## F. Per-message prediction release (Supplementary File 2)

A separate ZIP archive (also archived at Zenodo, DOI {zenodo_doi}) contains one CSV per architecture × dataset, with the columns:

```
message_id, dataset, true_hazard, true_action, hazard_category,
pred_proba, pred_hazard, pred_action, threshold_used,
architecture, model_version, run_id, inference_time_s
```

These files contain no patient text and no patient identifier beyond opaque message_id values. Patient message text remains under HIPAA data-use agreement and is available to bona-fide researchers on request to the corresponding author.

---

## G. References (Multimedia Appendix 1)

[All citations in the appendix; numbered independently from the main manuscript reference list. Both lists use the verified citations from Phase A literature review.]

1. Kumar A, Zhou A, Tucker G, Levine S. Conservative Q-learning for offline reinforcement learning. Advances in NeurIPS. 2020;33:1179-1191.
2. Collins GS, Moons KGM, Dhiman P, Riley RD, Beam AL, Van Calster B, Ghassemi M, Liu X, Reitsma JB, van Smeden M, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ 2024;385:e078378. doi:10.1136/bmj-2023-078378
3. Vasey B, Nagendran M, Campbell B, Clifton DA, Collins GS, Denaxas S, Denniston AK, Faes L, Geerts B, Ibrahim M, et al. Reporting guideline for the early stage clinical evaluation of decision support systems driven by artificial intelligence: DECIDE-AI. BMJ 2022;377:e070904. doi:10.1136/bmj-2022-070904
