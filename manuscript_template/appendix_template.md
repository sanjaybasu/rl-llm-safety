# Multimedia Appendix 1 — Supplementary Methods, Tables, and Figures

This appendix accompanies the main manuscript titled "AI safety evaluation in an underrepresented Medicaid population: real-world performance of clinical decision support and frontier language models on patient messaging triage," submitted to BMC Medical Informatics and Decision Making (BMC MIDM).

Every numerical value in this appendix derives from the canonical per-message prediction files retained by the corresponding author and available on request for verification.

---

## A. TRIPOD+AI checklist

Following Collins et al. [2]. The TRIPOD+AI items below map each reporting requirement to the corresponding location in the main manuscript.

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

Following Vasey et al. [3]. The DECIDE-AI items below map each reporting requirement to the corresponding location in the main manuscript.

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

The Conservative Q-Learning [1] controller maps the 23-dimensional calibrated hazard probability vector (output of the upstream sentence-BERT + multinomial logistic detector with temperature scaling, T={cql_temperature}) to one of the 8 escalation actions. The reward function is asymmetric: +10 for a correct action, -50 for a missed hazard, -2 for a false alarm. The expected reward calculation at inference time selects the action that maximizes expected reward.

The sensitivity-optimized variant applies a calibrated probability threshold ({thresholds.cql_sens_opt}) chosen on the training/validation split to maximize sensitivity subject to specificity at least 0.70. The reward-optimized variant applies the threshold derived analytically from the reward weights: P(hazard) ≥ 2/62 ≈ 0.032.

### C.3 Constellation architecture

A multinomial logistic regression on sentence-BERT embeddings classifies messages into one of 20 observed hazard categories. The binary hazard probability is derived as 1 − P(benign). Threshold for binary hazard prediction: {thresholds.constellation}.

### C.4 Rule-based guardrails (simplified implementation)

The deployed production system uses 147 hand-crafted rules. The implementation released here uses a 14-rule subset for reproducibility tractability; the simplified version is documented in Methods §3.4 of the main text with explicit acknowledgement of the simplification. The 14 rule families cover chest pain, stroke signs, anaphylaxis, suicidality, overdose, obstetric emergency, severe pain, breathing distress, neurological emergency, self-harm, substance overdose, falls, medication errors, and behavioral emergency triggers.

### C.4b Safety-augmented system prompt (frontier large language model methodology)

Frontier large language models (Claude Opus 4.7 and Gemini 3.1 Pro Preview) were evaluated with a safety-augmented system prompt that (a) describes the medical-triage-assistant role for a Medicaid managed care patient-messaging service, (b) explicitly directs the model to prioritize patient safety and err toward escalation in cases of ambiguity, (c) characterizes the patient population's typical communication patterns (low literacy, colloquialisms, abbreviations, implicit context), (d) enumerates representative hazard categories spanning cardiac, neurological, behavioral-health, substance-use, obstetric, and pediatric emergencies, and (e) specifies the 8-point action recommendation scale (Action 1 = self-care through Action 8 = emergency services). The output format is a structured JSON object containing a binary hazard flag, an action label, and a brief rationale. The full safety-augmented prompt and the default-prompt comparator are released in the code repository.

### C.4c Closing-the-gap intervention methodology

Following the principal-findings result that no single-architecture configuration in the evaluated matrix attained clinical-grade safety performance (sensitivity ≥ 0.80 AND specificity ≥ 0.80) on the real-world test set, the following architectural-combination and language-model-augmentation interventions were evaluated:

Ensemble configurations. All nine architectures were combined via three families of rules: (a) hard-voting k-of-9 rules for k from 1 (any flag) to 9 (unanimous), (b) soft-voting unweighted and F1-weighted means of the calibrated probability outputs at 101 thresholds from 0.0 to 1.0, and (c) top-three AND-rule and OR-rule of the three architectures with the highest single-architecture F1. A total of 213 ensemble configurations were enumerated and evaluated.

Two-stage cascade configurations. All 72 ordered (Stage 1, Stage 2) AND-rule cascade configurations of the nine architectures were enumerated and evaluated. The cascade decision is positive if and only if both stages flag the message; the cascade is mathematically symmetric (the AND operation is commutative), so the 36 unordered pairs each appear in both stage orderings in the full matrix.

Multi-large-language-model consensus. The existing per-message predictions of the safety-augmented Claude Opus 4.7 and Gemini 3.1 Pro Preview configurations were combined under four rules: Claude only, Gemini only, OR-rule (either flagging), and AND-rule (both flagging). No new language-model inference calls were required for this analysis.

Retrieval-augmented generation. A k-nearest-neighbor retrieval index was constructed over the 1,280 labeled training examples using sentence-BERT cosine similarity (all-mpnet-base-v2, 768-dimensional embeddings). At inference time, for each test message, the top-eight stratified-by-class nearest-neighbor training examples (at least two hazards and at least two benigns when available) were retrieved and embedded in the safety-augmented Claude Opus 4.7 system prompt as in-context evidence with their physician-adjudicated reference labels. The Claude Opus 4.7 backbone otherwise matches the safety-augmented configuration in the single-architecture matrix.

Threshold-optimization analysis. For each architecture with calibrated probability output, the receiver operating characteristic curve was swept across all observed threshold values; the F1-maximizing and Matthews-correlation-coefficient-maximizing operating points were extracted, along with the closest-to-clinical-grade operating point and a binary indicator of clinical-grade reachability (whether any point on the curve satisfied sensitivity ≥ 0.80 AND specificity ≥ 0.80).

### C.5 Calibration and temperature scaling

The upstream calibrated detector applies temperature scaling on the validation set after multinomial logistic regression training. The optimal temperature minimizes negative log-likelihood on the calibration set ({n_val} held-out examples).

---

## D. Supplementary tables

### Table S1. Physician-holdout hazard detection metrics per architecture.

{tableS1_physician_holdout_block}

Caption: Per-architecture sensitivity, specificity, PPV, NPV, F1, MCC, AUROC (where calibrated probabilities are available), and false-negative counts on the 41-case physician-scripted comparison set (27 hazards / 14 benigns). 95% confidence intervals: Wilson score method for proportions; parametric bootstrap (10,000 iterations) for F1 and MCC.

### Table S2. Sensitivity change from physician-scripted to real-world test sets with parametric bootstrap confidence intervals.

{tableS2_delta_bootstrap_block}

Caption: For each architecture, sensitivity attained on the physician-scripted 41-case comparison set is reported alongside sensitivity on the real-world {n_total}-message test set. The point estimate of the sensitivity change is reported in percentage points, with 95% confidence intervals from parametric bootstrap (10,000 iterations, seed 42) sampling sensitivities independently from binomial distributions parameterized by the observed proportions and the corresponding test-set sample sizes.

### Table S3. Full 72-pair cascade matrix on the real-world test set.

{tableS3_cascade_full_block}

Caption: All 72 (Stage 1, Stage 2) two-stage AND-rule cascade configurations evaluated on the {n_total}-message real-world test set. The Pareto frontier subset is highlighted in Table 5 of the main text. Sensitivity and specificity are reported as point estimates; full confidence intervals are available in the released supplementary CSV.

### Table S4. Per-architecture sensitivity by hazard category on the real-world test set.

{tableS4_category_stratification_block}

Caption: Sensitivity stratified by physician-adjudicated hazard category, restricted to categories with at least three adjudicated hazards in the real-world test set. Categories are ordered by row index; architecture columns are ordered by mean sensitivity across categories. The 'other_hazard' category includes hazards not falling into the named clinical categories and accounts for the majority of test-set hazards by count, reflecting the natural case mix of a Medicaid managed care messaging service. Behavioral suicidality is the highest-sensitivity category across architectures; the 'other_hazard' residual is the lowest-sensitivity category, indicating that the architectures' performance is best on the most prototypically presented hazard categories and worst on the heterogeneous remainder.

### Table S5. Full pairwise McNemar matrix with Hochberg step-up correction.

{tableS5_mcnemar_matrix_block}

Caption: Full pairwise McNemar discordant-pair matrix among all evaluated architectures on the real-world test set, with chi-square statistics, raw two-sided p-values, and Hochberg step-up significance at α = 0.05.

### Table S6. Calibrated operating-point thresholds per architecture.

{tableS6_thresholds_block}

Caption: Per-architecture calibrated decision threshold selected on the training/validation split.

---

## E. Supplementary figures

### Figure S1. Operating-curve overlays for all calibrated-probability architectures.

{figureS1_caption_block}

Caption: Receiver operating characteristic curves overlaid for every architecture with calibrated probability output (logistic regression with TF-IDF features, XGBoost with sentence-BERT embeddings, constellation, rule-based guardrails, Conservative Q-Learning controller sensitivity-optimized, Conservative Q-Learning controller reward-optimized). The frontier large language models and the action recommender are excluded because they do not return calibrated probability output.

### Figure S2. Calibration curves and reliability diagrams per architecture.

{figureS2_caption_block}

Caption: Per-architecture calibration plots showing predicted versus observed hazard frequencies in deciles, restricted to the architectures with calibrated probability outputs.

### Figure S3. Hazard-category sensitivity by architecture.

{figureS3_caption_block}

Caption: Per-hazard-category sensitivity by architecture on the real-world test set. The horizontal dashed line at sensitivity 0.80 indicates the conventional clinical-grade sensitivity floor in clinical computer-aided detection benchmarks.

---

## F. References

This appendix uses its own sequentially-numbered reference list. The same primary literature is cross-referenced in the main manuscript reference list under different numbers.

1. Kumar A, Zhou A, Tucker G, Levine S. Conservative Q-learning for offline reinforcement learning. Adv Neural Inf Process Syst. 2020;33:1179-1191.

2. Collins GS, Moons KGM, Dhiman P, Riley RD, Beam AL, Van Calster B, Ghassemi M, Liu X, Reitsma JB, van Smeden M, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378. doi:10.1136/bmj-2023-078378. PMID: 38626948.

3. Vasey B, Nagendran M, Campbell B, Clifton DA, Collins GS, Denaxas S, Denniston AK, Faes L, Geerts B, Ibrahim M, et al. Reporting guideline for the early stage clinical evaluation of decision support systems driven by artificial intelligence: DECIDE-AI. BMJ. 2022;377:e070904. doi:10.1136/bmj-2022-070904. PMID: 35584845.

