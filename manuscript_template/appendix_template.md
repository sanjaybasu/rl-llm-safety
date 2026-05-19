# [TEMPLATE] Multimedia Appendix 1 — Supplementary Methods, Tables, and Figures

This appendix accompanies the main manuscript titled "AI safety evaluation in an underrepresented Medicaid population: real-world performance of clinical decision support and frontier language models on patient messaging triage."

Every numerical value in this appendix renders from canonical CSVs at build time via `code/pipeline/manuscript_renderer.py`. No hand-typed numbers.

---

## A. TRIPOD+AI checklist

[To be completed item-by-item; TRIPOD+AI item map at https://www.tripod-statement.org/]

## B. DECIDE-AI checklist

[To be completed item-by-item; DECIDE-AI item map at https://decide-ai.org/]

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
