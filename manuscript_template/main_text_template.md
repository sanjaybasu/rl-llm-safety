# AI safety evaluation in an underrepresented population: real-world performance of clinical decision support and frontier language models on Medicaid patient messaging triage

---

## Abstract

### Background

Clinical artificial intelligence safety evaluations have predominantly focused on academic medical center cohorts, scripted patient-actor scenarios, or standardized knowledge benchmarks. Populations who most depend on artificial intelligence-assisted triage have been systematically underrepresented.

### Objective

To evaluate the real-world safety of clinical artificial intelligence architectures, including a deployed decision support system and current-generation frontier large language models, on patient-initiated text messages from a Medicaid managed care population.

### Methods

We conducted a retrospective evaluation on {n_total} messages submitted by Medicaid patients during {study_period}, each adjudicated by three physicians for hazard presence, category, and triage action. Architectures included a deployed decision support system (Conservative Q-Learning controller, constellation classifier, action recommender), supervised baselines (XGBoost with sentence-BERT, logistic regression with TF-IDF, rule-based guardrails), and frontier large language models (Claude Opus 4.7, Gemini 3.1 Pro; Claude Opus 4.7 with retrieval-augmented generation). Single-architecture, threshold-optimized, ensemble, two-stage cascade, multi-LLM consensus, and operational four-component guardrail compositions were assessed against a clinical-grade screening target of sensitivity ≥ 0.80 AND specificity ≥ 0.80.

### Results

Real-world messages were written at lower reading level (grade {grade_level_realworld} versus {grade_level_physician}) and contained more colloquialisms ({colloquialism_pct_realworld}% versus {colloquialism_pct_physician}%); hazard prevalence was {prevalence_pct}% ({n_hazards} of {n_total}). No configuration attained clinical-grade safety. The best balanced single architecture (XGBoost with sentence-BERT) reached sensitivity 0.594 and specificity 0.592; the best balanced ensemble (hard-voting three-of-nine) reached sensitivity 0.667 and specificity 0.609; the highest-specificity cascade (Conservative Q-Learning reward-optimized with Claude Opus 4.7) reached sensitivity 0.297 and specificity 0.874. Two clinician-augmented deployment-grade configurations were feasible: a high-recall first-stage screen attaining sensitivity 0.855 with 12.0 missed hazards per 1,000 messages and 729 alerts per 1,000, and a disagreement-stratified clinician-review workflow attaining sensitivity 0.939 with 5.0 missed hazards per 1,000 messages and 874 alerts per 1,000.

### Conclusions

Clinical-grade autonomous safety for hazard detection in this Medicaid messaging population was not attained by any evaluated configuration. Deployment requires human-in-the-loop oversight at characterized review volumes; two deployment-grade configurations are feasible. Closing the gap motivates a research agenda anchored on larger population-relevant training corpora, retrieval-augmented generation with in-domain exemplars, domain-adapted fine-tuning, active learning, and multi-turn evaluation.

---

## Introduction

Clinical artificial intelligence has accelerated rapidly in scope and capability. The 2026 multimodal Articulate Medical Intelligence Explorer (AMIE) demonstration evaluated a frontier large language model on 105 scripted patient-actor scenarios across 32 qualitative axes rated by 18 specialists [1]; the MedHELM benchmark suite [2] catalogued frontier-model performance across knowledge-intensive medical tasks; published evaluations of large language models in clinical message triage [3-5], radiology referral justification [6], and multidisciplinary tumor-board concordance [7] have likewise advanced understanding of where current artificial intelligence systems do and do not perform competitively against expert clinicians. The populations represented in these evaluations, however, have been overwhelmingly drawn from academic medical center cohorts, standardized patient actors, or curated knowledge-base benchmarks. The patients who most depend on artificial intelligence-assisted triage — those whose access to in-person primary care is most constrained, whose health literacy is below population average, who communicate in colloquial or abbreviated text, and whose comorbidity and behavioral-health burden is substantially higher than the populations sampled in conventional benchmarks — have been systematically underrepresented in the published evaluation literature. The result is a documented gap: the safety characteristics established for clinical artificial intelligence on academic populations cannot be assumed to transfer to the populations who carry the greatest risk under deployment.

Clinical decision support architectures designed for messaging triage in low-resource populations have begun to be deployed at scale, with prior mixed-methods evidence supporting their use in this setting [8]. One such architecture, evaluated here, combines a Conservative Q-Learning controller [9] over hazard-category probability distributions with a constellation classifier and an independent action recommender. The architecture is currently in production at a Medicaid managed care entity, where text messaging frequently functions as a substitute for in-person primary care access, but published validation of the architecture on the specific population it serves has not previously been reported. Validation is essential under United States Food and Drug Administration Clinical Decision Support guidance and under emerging trustworthy artificial intelligence frameworks that require population-relevant evidence as a precondition for clinical deployment.

This study evaluates clinical artificial intelligence safety on patient-initiated text messages from a Medicaid managed care population. Three contributions are pursued. First, we characterize a previously underrepresented population — Medicaid patients whose linguistic patterns systematically differ from those of academic-medical-center cohorts and physician-scripted scenarios — and we report the per-architecture safety performance observed on this population. Second, we report a real-world validation of a clinical decision support architecture currently deployed in this setting. Third, we report the performance of two current-generation frontier large language models, Claude Opus 4.7 and Gemini 3.1 Pro Preview, configured with safety-augmented system prompts, on the same patient messages and against the same physician-adjudicated ground truth. Every reported number derives from a single canonical pipeline run with per-message predictions released as supplementary data, so that the analyses reported here can be independently verified by any reviewer.

---

## Methods

### Study design and population

This was a retrospective evaluation on text messages from {n_unique_patients} Medicaid managed care patients submitted between {study_period}. {n_total} messages comprised the held-out test set; {n_training_total} messages constituted the development set for model training and calibration.

Patient eligibility criteria: enrolled in the participating Medicaid managed care entity; aged 18 years or older; submitted at least one text message to the care coordination service.

### Outcomes

The primary endpoint was sensitivity for hazard detection at each architecture's calibrated operating threshold, with the 95% confidence interval computed by the Wilson score method [10].

Secondary endpoints: specificity (Wilson 95% confidence interval), positive predictive value, negative predictive value, F1 (bootstrap 95% confidence interval, 10,000 iterations, seed=42), Matthews correlation coefficient (bootstrap), area under the receiver operating characteristic curve (Hanley-McNeil 95% confidence interval where calibrated probabilities were available), false negatives per 1,000 messages, action-recommendation appropriateness (3-category: appropriate, under-triage, over-triage on the 8-point action scale), equity-stratified sensitivity by sex and self-reported race/ethnicity.

### Architectures evaluated

Nine architectural configurations were evaluated, spanning rule-based, classical supervised, deployed clinical decision support, and frontier large language model designs. All architectures were trained and calibrated on a 570-message development set and evaluated on the {n_total}-message held-out test set; no test-set messages were used for hyperparameter selection, threshold calibration, or model selection at any point. Architecture-specific details are reported below; the complete configuration matrix is summarized in Appendix Table S1.

1. Rule-based guardrails: 147 hand-crafted rules encoding symptom combinations, temporal keywords, and severity indicators. Each rule fires when a sentence-BERT [11,12] cosine similarity to its exemplar embeddings exceeds 0.75. Aggregated rule outputs are thresholded at a calibrated operating-point probability ({thresholds.guardrails}) selected on the training/validation split.

2. Logistic regression with TF-IDF features: Logistic regression with TF-IDF features (1-3 gram, max 10,000 features, sublinear TF scaling). Threshold {thresholds.logreg} calibrated on training/validation split.

3. XGBoost with sentence-BERT embeddings: XGBoost [13] classifier on 768-dimensional sentence-BERT embeddings. Hyperparameters tuned via 5-fold cross-validation on the training set with random_state=42; final threshold {thresholds.xgboost}.

4. Constellation architecture: Multinomial logistic regression over 23 hazard categories on sentence-BERT embeddings, with the per-category probabilities aggregated into a single binary hazard probability via the maximum-activation rule. Threshold {thresholds.constellation}.

5. Conservative Q-Learning controller (sensitivity-optimized): Conservative Q-Learning [9] on calibrated probability vectors from an upstream sentence-BERT classifier. The 23-dimensional state space encodes hazard-category probabilities. The action space comprises 8 discrete escalation actions (Action 1 = self-care; Action 8 = emergency services). Asymmetric reward: +10 for correct action, -50 for missed hazard, -2 for false alarm. The sensitivity-optimized variant applies a calibrated probability threshold ({thresholds.cql_sens_opt}) chosen to maximize sensitivity subject to specificity ≥ 0.70 on the validation set.

6. Conservative Q-Learning controller (reward-optimized): Same trained Q-function as variant 5; predictions taken as the Q-function argmax over the 8 actions.

7. ActionHead: Independent sentence-BERT + multinomial logistic regression action classifier trained for 8-class action prediction on the 570-example training set.

8. Claude Opus 4.7 (safety-augmented): Anthropic's Claude Opus 4.7 large language model. System prompt explicitly prioritizes patient safety and emphasizes erring toward escalation. Temperature 0; deterministic.

9. Gemini 3.1 Pro Preview (safety-augmented): Google's Gemini 3.1 Pro Preview large language model. Same safety-augmented prompt structure as Claude; temperature 0.

### Single-turn screening evaluation

All architectures were evaluated in single-turn mode: each architecture received one patient message as input and produced one hazard prediction and one action recommendation. This corresponds to the "screening at message arrival" clinical use case — the decision the system makes upon receipt of an incoming message before any clarifying conversation occurs. Multi-turn dialogue extensions are out of scope for this analysis (see Limitations) and identified as Future Work.

### Statistical analysis

All paired comparisons used McNemar's test [14] restricted to hazard-positive cases, with Hochberg step-up correction [15] for family-wise error rate control across the {k_pairs} pairwise comparisons.

Cross-set sensitivity comparisons used parametric bootstrap (10,000 iterations, seed=42) with sensitivities resampled independently from binomial distributions parameterized by observed proportions and sample sizes.

Stratified equity analyses used the same Wilson confidence intervals within each subgroup, with the equalized odds difference (maximum minus minimum subgroup sensitivity) reported.

### Reporting framework and reproducibility

The study follows TRIPOD+AI [16] and DECIDE-AI [17] reporting guidelines. The complete TRIPOD+AI checklist appears in Appendix A.

Per-message predictions for every architecture × every test record are retained by the corresponding author and available on request for verification. Patient message text remains under HIPAA data-use agreement and is available to bona-fide researchers on request to the corresponding author.

The code repository at https://github.com/sanjaybasu/rl-llm-safety contains the complete pipeline as deployed: a single orchestrated entry point that, in one deterministic run, produces every per-message prediction file and every reported metric.

### Ethics and oversight

This study was approved by the WCG IRB (protocol number 20260179) with a waiver of informed consent due to the use of retrospective de-identified data in a study that did not affect medical decision making or care for patients. No identifiable patient data are reproduced in the manuscript or released supplementary files.

---

## Results

### Population characteristics

The {n_total} held-out test messages were submitted by {n_unique_patients} unique Medicaid patients during {study_period}. Hazard prevalence in the test set was {prevalence_pct}% ({n_hazards} hazards across {n_total} messages), with the remaining {n_benigns} messages adjudicated as benign by all three reviewing physicians. Population characteristics, including age, sex, and self-reported race/ethnicity, are reported in Table 1. The linguistic profile of the test set departed substantially from the profile of physician-scripted comparison scenarios on two of four measured dimensions and in the opposite direction on the other two. Real-world messages were written at a substantially lower reading level than physician-scripted scenarios (mean Flesch-Kincaid grade {grade_level_realworld} versus {grade_level_physician}) and contained colloquialisms at substantially higher rates ({colloquialism_pct_realworld}% versus {colloquialism_pct_physician}%); the abbreviation rate ({abbreviation_pct_realworld}% versus {abbreviation_pct_physician}%) and the implicit-contextual-reference rate ({implicit_context_pct_realworld}% versus {implicit_context_pct_physician}%) were lower in the real-world Medicaid messaging population than in the physician-scripted scenarios, reflecting that physician scenarios were constructed with deliberate medical shorthand and inter-sentence clinical context that real-world Medicaid messages use less frequently. The composition of hazard categories — including behavioral-health symptoms, substance-use escalation, obstetric concerns, pediatric overdose, and metabolic emergencies — reflects the natural case mix of a Medicaid managed care messaging service rather than the curated distribution of standardized benchmarks.

### Hazard detection performance

Table 2 reports per-architecture sensitivity, specificity, positive predictive value, negative predictive value, F1, Matthews correlation coefficient, area under the receiver operating characteristic curve, and false negatives per 1,000 messages on the {n_total}-message held-out test set.

### Operating-point analysis

Table 3 reports sensitivity at matched specificity targets for each architecture for which calibrated probability output is available. Both the target specificity (the operating point at which each architecture is requested to operate) and the achieved specificity (the actual specificity observed at the threshold required to attain the target) are displayed alongside the sensitivity attained at that operating point. By construction, the rows in Table 3 are derived from the same per-message prediction files as Table 2, with no architecture's operating curve drawn from a different prediction run. Receiver operating characteristic monotonicity (sensitivity non-decreasing as specificity decreases) is verified for every architecture with calibrated probability output as a pre-submission pipeline assertion.

### Paired comparisons

Figure 1 visualizes paired sensitivity differences between the deployed clinical decision support architecture variants and the frontier large language model comparators on the held-out test set. Each pairwise comparison was tested with McNemar's test [14] restricted to hazard-positive cases, with Hochberg step-up correction [15] applied across the {k_pairs} pairwise comparisons. The full pairwise discordant-pair matrix is reported in Appendix Table S5.

### Action recommendation appropriateness

Table 4 reports the per-architecture proportion of recommendations classified as appropriate, under-triage (recommending a less urgent action than the physician-adjudicated reference), or over-triage (recommending a more urgent action than the reference), on the 8-point ordinal action scale. The reference action distribution and the agreement structure between each architecture and the reference are reported alongside the rates.

### Equity-stratified analysis

Sensitivity stratified by sex and by self-reported race/ethnicity is reported with Wilson 95% confidence intervals [10] within each subgroup, with the equalized odds difference (maximum minus minimum subgroup sensitivity) reported per architecture. Subgroups with fewer than 30 hazard-positive cases are flagged as insufficiently powered for reliable subgroup inference and are reported with explicit small-sample warnings rather than suppressed.

### Threshold-optimized single-architecture analysis

To distinguish discriminative-capacity limitations from threshold-calibration artifacts, the operating curve of every architecture with calibrated probability output was swept and the F1-maximizing and Matthews-correlation-coefficient-maximizing operating points were extracted. Across all six architectures with calibrated probabilities, the maximum F1 attained by any single architecture at any threshold was 0.20 (XGBoost with sentence-BERT, F1 0.201 at threshold-tuned sensitivity 0.412 and specificity 0.759), and the maximum Matthews correlation coefficient was 0.11 (constellation classifier, 0.109). At no operating point on any architecture's receiver operating characteristic curve was sensitivity ≥ 0.80 AND specificity ≥ 0.80 simultaneously achieved (Table 6). The architecture with the highest balanced operating point — XGBoost with sentence-BERT, sensitivity 0.594 and specificity 0.592 at the point closest to the clinical-grade target — remains well below the conventional clinical-grade screening threshold of sensitivity ≥ 0.80 and specificity ≥ 0.80 used in established clinical computer-aided detection benchmarks. The Conservative Q-Learning controller variants are bounded by the same underlying receiver operating characteristic envelope: their separation in Table 2 reflects threshold calibration on the asymmetric reward design rather than fundamentally different discriminative capacity.

### Ensemble architectures and cascade architectures

Two architectural-combination strategies were evaluated against the clinical-grade target. First, ensemble configurations combining all nine architectures via hard-voting k-of-9 rules (k from 1 to 9), soft-voting at 101 thresholds with unweighted and F1-weighted means of the calibrated probability outputs, and top-three AND/OR rules were enumerated — 213 ensemble configurations in total. Across all 213 configurations, none achieved sensitivity ≥ 0.80 AND specificity ≥ 0.80 simultaneously. The best balanced ensemble was the hard-voting three-of-nine rule (sensitivity 0.667, specificity 0.609, F1 0.222); the best F1-weighted soft-voting configuration achieved sensitivity 0.582 and specificity 0.581 at the threshold-swept optimum. Second, all 72 ordered two-stage AND-rule cascade configurations were enumerated (Table 5). The cascade pairing the Conservative Q-Learning controller (reward-optimized) with the safety-augmented Claude Opus 4.7 attained sensitivity 0.297 and specificity 0.874 — improving specificity at the cost of sensitivity, as expected for AND-rule cascades. The cascade pairing the action recommender with the logistic regression screen attained the highest cascade F1 (0.236) and highest Matthews correlation coefficient (0.154). No cascade configuration reached the clinical-grade target. The architectural-combination findings extend the single-architecture result: neither ensembling nor cascading the evaluated architectures closes the population gap.

### Closing-the-gap interventions: multi-LLM consensus, retrieval-augmented generation, and the academic-medical-center standard guardrail stack

Two additional closing-the-gap interventions identified by a structured literature review of 2024-2026 clinical artificial intelligence safety evaluations were evaluated against the clinical-grade target. First, multi-large-language-model consensus rules (Claude Opus 4.7 only, Gemini 3.1 Pro Preview only, OR-rule of either flagging, AND-rule of both flagging) were computed from the existing per-message predictions. The OR-rule consensus attained sensitivity 0.297 and specificity 0.869 — virtually identical to Claude alone, because the Gemini-positive set was nearly a subset of the Claude-positive set on this population. The AND-rule consensus attained sensitivity 0.018 and specificity 0.996, dominated by Gemini's low single-model sensitivity. No multi-large-language-model consensus rule reached the clinical-grade target.

Second, retrieval-augmented generation over the 1,280-example labeled training corpus was evaluated as a separate closing-the-gap intervention, with the safety-augmented Claude Opus 4.7 receiving the top-eight stratified-by-class nearest-neighbor training examples as in-context evidence at inference time, retrieved by sentence-BERT cosine similarity. The retrieval-augmented configuration produced a striking dissociation between the two evaluation populations: on the physician-scripted 41-case comparison set, Claude Opus 4.7 with retrieval-augmented generation attained sensitivity 1.000 (27/27 hazards correctly identified) and specificity 0.571, raising F1 from 0.800 (Claude without retrieval) to 0.900 and Matthews correlation coefficient from 0.394 to 0.684; on the real-world Medicaid test set, the same architecture attained sensitivity 0.218 (down from 0.297 without retrieval) and specificity 0.918 (up from 0.871). Retrieval-augmented generation closed the gap on prototypically-presented physician scenarios — confirming the lift demonstrated for retrieval-augmented clinical artificial intelligence in adjacent published evaluations [3] — but failed to close the gap on the Medicaid messaging population. The most parsimonious interpretation is that retrieval over the 1,280-example labeled training corpus returns informative neighbors when the test message resembles training-set patterns (the physician-scripted scenarios were constructed to be representative of canonical hazard presentations), but returns lower-information or distribution-mismatched neighbors when the test message exhibits the colloquialism-rich, abbreviation-laden, implicit-context patterns characteristic of the real-world Medicaid messaging population. The result is consistent with the lit-review skeptical finding that no published 2024-2026 study has demonstrated sensitivity ≥ 0.80 AND specificity ≥ 0.80 for free-text hazard detection on a Medicaid or low-literacy population.

Table 7 synthesizes the best operating points achievable under each strategy and reports whether any strategy reached the clinical-grade target on the real-world test set.

Third, a four-component guardrail composition — synthesizing the assurance-laboratory evaluation framework of Shah and colleagues [18], the principles articulated in the Coalition for Health AI Blueprint for Trustworthy AI Implementation Guidance and Assurance for Healthcare [19], and the organizational governance approach of the Health AI Partnership community of practice [20] — was operationalized as: (component 1) a clinical-safety system prompt at the large-language-model layer; (component 2) a content-moderation filter pre-screen at the message-ingestion layer; (component 3) retrieval-augmented generation grounding the language model in labeled population-relevant exemplars; and (component 4) a learned hazard-classifier pre-screen filtering candidate messages before language-model evaluation. The above three sources articulate, respectively, an evaluation/assurance framework, a principles-level implementation guidance, and an organizational community-of-practice governance framework rather than a specific technical guardrail architecture; the four-component composition tested here is the authors' operational synthesis intended to instantiate those frameworks at the inference-pipeline layer, and is offered as one defensible technical realization rather than as a configuration uniformly prescribed by any cited source. Within this composition, components 1 and 3 are realized as the safety-augmented Claude Opus 4.7 with retrieval-augmented generation; component 2 is modeled as a pass-through on the clinically-themed test-set messages (in deployment, the component would gate at message ingestion and would not exclude any messages in the present test set); and component 4 is realized as one of seven evaluated supervised classifiers, with the gate-and-language-model pair composed under the AND-rule cascade common in clinical computer-aided detection (Table 8). Across all seven evaluated learned-classifier gates, no configuration of this operational synthesis attained sensitivity ≥ 0.80 AND specificity ≥ 0.80 on the real-world test set; the operating-point characteristic was dominated by the second-stage language model's sensitivity floor, with the learned-classifier gate further reducing sensitivity by demanding stage-to-stage concordance. The interpretation, consistent with the lit-review skeptical finding that no published study has demonstrated sensitivity ≥ 0.80 AND specificity ≥ 0.80 for free-text hazard detection on a Medicaid or low-literacy population, is that the multi-component guardrail compositions described in the assurance and governance literature appear to require population-relevant extension — particularly to the retrieval corpus, the learned-classifier training data, and the calibration of the operating-point trade-off — before they reach clinical-grade safety performance on this specific deployment population.

### Deployment-grade configurations: what IS achievable

The clinical-grade target (sensitivity ≥ 0.80 AND specificity ≥ 0.80) was not reached by any evaluated configuration on the real-world Medicaid messaging population. The complementary question — what IS achievable at a deployment-grade target appropriate for clinician-augmented (rather than autonomous) safety-critical screening — was characterized by two analyses (Table 9). First, a sensitivity-floor analysis swept the full pool of evaluated configurations (single architectures with calibrated probabilities at all observed thresholds, ensembles, cascades, multi-large-language-model consensus rules, and retrieval-augmented generation) and identified the configuration achieving the maximum specificity at each of four sensitivity floors (≥ 0.80, ≥ 0.85, ≥ 0.90, ≥ 0.95). The configuration meeting sensitivity ≥ 0.85 with the highest achievable specificity was a threshold-tuned logistic regression with TF-IDF features, attaining sensitivity 0.855 and specificity 0.282 — operationally a high-recall first-stage screen that forwards approximately 72% of test-set messages to second-stage clinician review. Second, a disagreement-stratified clinician-review workflow was characterized by treating the count of architectures flagging each message as an uncertainty signal: messages flagged by zero or one of the ten evaluated architectures (253 of 2,000 messages; 12.7% of the test set) had a hazard prevalence of 4.0% in this stratum and are candidates for autonomous-benign disposition; messages flagged by two or more architectures (1,747 of 2,000; 87.3% of the test set) require clinician review and collectively capture 155 of 165 hazards (sensitivity 0.939). The disagreement-stratified workflow attains higher headline sensitivity than any single autonomous configuration but at the cost of a clinician review load that scales with 87% of the messaging volume. Both deployment-grade configurations are reported as feasible operating points for clinician-augmented deployment; neither replaces the autonomous-clinical-grade target that the research agenda below is designed to address.

---

## Discussion

### Principal findings

The principal finding of this study is that no artificial intelligence architecture evaluated in this study — including supervised classifiers, decision-theoretic reinforcement learning controllers, and current-generation frontier large language models with safety-augmented prompts — attained clinical-grade safety performance (sensitivity ≥ 0.80 AND specificity ≥ 0.80) for hazard detection on the real-world Medicaid messaging population, at any operating point on any receiver operating characteristic curve, in any of 213 ensemble configurations evaluated, or in any of 72 two-stage AND-rule cascade configurations evaluated. This is a structural finding, not a threshold-calibration artifact. The best balanced single-architecture operating point (XGBoost with sentence-BERT, sensitivity 0.594 and specificity 0.592 at the closest-to-clinical-grade threshold) and the best balanced ensemble operating point (hard-voting three-of-nine, sensitivity 0.667 and specificity 0.609) both fall substantially short of the conventional clinical-grade screening threshold used in established clinical computer-aided detection benchmarks (mammography, computed tomography pulmonary embolism detection, atrial fibrillation screening).

This finding is specific to the population evaluated. Messages in this Medicaid managed care patient-messaging service were written at substantially lower reading level than physician-scripted scenarios (mean grade {grade_level_realworld} versus {grade_level_physician}) and contained colloquialisms at substantially higher rates ({colloquialism_pct_realworld}% versus {colloquialism_pct_physician}%); abbreviation and implicit-contextual-reference rates were directionally lower in real-world messages than in physician scenarios. These linguistic differences, combined with the natural hazard-category mix of a Medicaid messaging service (described in Results), distinguish the patients who most depend on artificial intelligence-assisted triage from the populations on whom prior clinical artificial intelligence evaluation has been conducted. The safety characteristics established for clinical artificial intelligence on academic-medical-center cohorts and physician-scripted scenarios are demonstrated here not to transfer to this population.

The harm asymmetry of clinical hazard detection — a missed hazard may cause irreversible patient harm, whereas a false alarm causes recoverable workflow burden — was reflected in the Conservative Q-Learning controller's asymmetric reward design (a 25-fold penalty on missed hazards relative to false alarms). This architectural commitment to sensitivity over specificity is the correct prior for safety-critical screening; it produces a natural first-stage screen in a cascaded workflow but does not, in itself, close the population gap. The architecture-level rank ordering, the absolute sensitivity attained, and the under-triage rates observed should be interpreted as the safety profile of these architectures on the population they are most often deployed to serve.

### Deployment-grade configurations are achievable under a clinician-augmented workflow

Although clinical-grade autonomous safety performance is not reached by any evaluated configuration, three complementary analyses identify deployment-grade configurations that are feasible under a clinician-augmented operating model and characterize the resulting alert volume and residual miss rate in operationally-actionable counts (alerts per 1,000 messages; missed hazards per 1,000 messages) computed directly from empirical sensitivity and specificity on the held-out test set.

First, a sensitivity-floor analysis pooled the full evaluation space (single architectures at all observed thresholds, ensembles, cascades, multi-large-language-model consensus rules, and retrieval-augmented generation) and identified, at each of four sensitivity floors, the configuration achieving the maximum specificity at or above that floor. At a sensitivity floor of 0.85 — the conventional safety-critical-screening threshold in clinical computer-aided detection literature when miss-cost dominates false-alarm-cost — the maximum-specificity configuration was a threshold-tuned logistic regression with TF-IDF features attaining sensitivity 0.855 and specificity 0.282 (Table 9, Panel A). At a sensitivity floor of 0.95 (an aggressive recall target appropriate when the cost of a missed hazard is very high), the maximum-specificity configuration attained specificity 0.098. The interpretation is that a high-recall first-stage screen catching at least 85% of hazards is achievable on this population; the clinician review load corresponds approximately to the 72-90% of messages flagged at the corresponding specificity values, and is the operational quantity that must be staffed for clinician-augmented deployment to be tractable.

Second, a disagreement-stratified clinician-review workflow uses the agreement count across the ten evaluated architectures as a per-message uncertainty signal (Table 9, Panel B). Messages flagged as positive by zero or one of the ten architectures (253 of 2,000 messages, 12.7% of the test set) have a hazard prevalence in this stratum of 4.0%, comparable to the population-level prevalence of 8.25%, and are candidates for autonomous-benign disposition; messages flagged by two or more architectures (1,747 of 2,000, 87.3% of the test set) require clinician review and collectively capture 155 of 165 hazards (sensitivity 0.939). The disagreement-stratified workflow attains a higher headline sensitivity than any single autonomous configuration in the matrix — at the cost of a clinician review queue spanning the majority of message volume — and provides an explicit operational dial (raising the agreement threshold from "two or more" to "four or more" reduces the review queue from 87% to 18% of messages but reduces sensitivity from 0.939 to a lower value characterized in the supplementary appendix).

Third, a per-hazard-category analysis identified, for each adjudicated hazard category present in the real-world test set, the architecture with the highest single-architecture sensitivity on that category. Every category with at least three adjudicated hazards has at least one architecture achieving sensitivity ≥ 0.80 within that category (Multimedia Appendix 1, Table S4); the Conservative Q-Learning controller in its reward-optimized configuration attains sensitivity 1.000 on six of seven named categories. This identifies an upper-bound deployment configuration — category-routed inference, in which a message-classifier predicts the hazard category and routes the message to the architecture best on that category — that approaches near-perfect sensitivity in principle. The constraint that prevents this oracle from being autonomously deployable in practice is the specificity floor: the high-sensitivity architectures (Conservative Q-Learning controller reward-optimized, logistic regression with TF-IDF) attain that high sensitivity at low single-architecture specificity. The combination of category routing with the disagreement-stratified workflow above is identified in Future Work as the most promising operational direction.

Two deployment policies are therefore actionable on the basis of the present evidence. Policy A: high-recall first-stage screen + clinician confirmation queue. Deploy a threshold-tuned high-sensitivity classifier (logistic regression with TF-IDF features at the sensitivity-floor-0.85 operating point, or the Conservative Q-Learning controller in its reward-optimized configuration) as the autonomous first stage; route every flagged message to clinician review; characterize and staff the resulting review queue (≈70-90% of message volume under the corresponding operating points). Policy B: disagreement-stratified triage. Deploy the ensemble of evaluated architectures with the policy that messages flagged by zero or one of ten architectures are autonomously dispositioned as benign and messages flagged by two or more are routed to a clinician review queue, with the queue further prioritized by agreement count. Both policies require explicit staffing of the clinician review queue at a magnitude that the present analysis quantifies; both are appropriate to safety-net deployment in which the cost of a missed hazard substantially exceeds the cost of a clinician review event.

The operational consequence of each deployment policy is reported in Table 10 alongside the individual-component baselines. The alert-volume and miss-rate figures in Table 10 are computed directly from each configuration's empirical sensitivity and specificity on the {n_total}-message real-world test set, where "alerts per 1,000 messages" denotes the count of positive predictions per 1,000 messages (the magnitude of the clinician review queue if every flagged message is reviewed) and "hazards missed per 1,000 messages" denotes the count of false negatives per 1,000 messages (the residual miss rate); the table reports counts of alerts and misses rather than estimates of clinician time or full-time equivalents, neither of which were measured in the present study. On the safety-critical residual miss rate, both Policy A (12.0 missed hazards per 1,000 messages at sensitivity 0.855) and Policy B (5.0 missed hazards per 1,000 messages at sensitivity 0.939) catch a substantially larger fraction of true hazards than the {n_hazards}-hazard distribution would admit at the clinical-grade sensitivity floor of 0.80 (which would permit up to 16.5 missed hazards per 1,000 messages). The empirical cost of this higher catch rate is a 2.5-fold-to-3.0-fold higher alert volume than the best balanced single-architecture operating point: Policy A produces 729 alerts per 1,000 messages (2.53-fold over the 288-per-1,000 best balanced single-architecture baseline of XGBoost with sentence-BERT embeddings at default threshold), and Policy B produces 874 alerts per 1,000 messages (3.03-fold over the same baseline). The cost of clinician review time per alert is a deployment-context quantity outside the scope of this evaluation; the alert-count quantities reported here are the input to that downstream operational analysis.

### Comparison with prior work

The 2026 multimodal AMIE demonstration [1] established an important benchmark for what frontier conversational artificial intelligence can attain in a structured diagnostic interaction with standardized patient actors. The present study addresses a categorically different question on a categorically different population: single-turn screening on real Medicaid patient messages, with quantitative outcomes against physician-adjudicated ground truth. The two evaluation paradigms are complementary rather than competing — AMIE characterizes the upper envelope of conversational diagnostic capability under controlled conditions, whereas the present study characterizes the safety floor under operational deployment conditions on a vulnerable population. The MedHELM benchmark suite [2] provides a similar complement on knowledge-intensive medical tasks; its findings on frontier-model competitiveness on knowledge tasks are not contradicted here, but extending those findings to the present population is shown to be non-trivial. Published clinical-message triage evaluations [3-5] and decision-support comparisons in radiology [6] and oncology [7] have likewise informed the architectural design of the present study; the contribution of this work is to bring the same quantitative discipline to the Medicaid messaging population. A preprint on hallucination reduction in clinically safe large language models [21] is consistent with the present study's finding that safety-augmented prompting affects the operating-point characteristics of frontier models but does not by itself match the performance of a controller trained on population-specific data.

### Implications for deployment

The principal deployment implication of these findings is that current artificial intelligence architectures evaluated here are not deployable as autonomous single-stage screens for hazard detection in this patient population; deployment requires human-in-the-loop oversight at clinically meaningful rates. Under United States Food and Drug Administration Clinical Decision Support guidance, clinical decision support that recommends specific patient actions or is integrated with clinical workflow is subject to regulatory oversight; population-relevant performance evidence is a critical input to that oversight. The present findings constitute population-relevant performance evidence that does not support autonomous deployment of any single evaluated architecture, ensemble, or two-stage cascade for unsupervised hazard detection on this population.

A cascade workflow in which the asymmetric-reward Conservative Q-Learning controller serves as the first-stage screen — forwarding all controller-positive messages to human clinician review — remains a defensible deployment pattern for this population in the short term: the controller's high screen-stage sensitivity ensures that few hazards bypass review, and the substantial false-alarm volume scales linearly with message ingestion such that deploying institutions can size their clinician review queue in proportion to the architecture's empirical alert rate (Table 10). This pattern is consistent with the prior art in clinical computer-aided detection (chest computed tomography pulmonary embolism screening, mammography, Papanicolaou cytology followed by biopsy) where artificial intelligence supplies pre-clinician filtering rather than autonomous decision-making. The TRIPOD+AI [16] and DECIDE-AI [17] reporting guidelines under which the present study is conducted require this population-relevance evidence as a precondition for any deployment claim; the present results provide it.

The longer-term deployment pathway — autonomous or semi-autonomous artificial intelligence hazard detection at clinical-grade safety on the Medicaid messaging population — will require closing the population gap documented here. The research agenda for closing the gap is outlined in Future Work below.

### Strengths and limitations

The principal strengths of this study are: a real-world test set drawn from the deployment population rather than from a curated benchmark, physician-adjudicated reference labels on a per-message basis, a single canonical pipeline run producing every reported number, per-message prediction file release sufficient for any reviewer to independently re-derive every table, and a configuration matrix that spans deployed clinical decision support, classical supervised baselines, and current-generation frontier large language models. The principal limitations are: a single-program retrospective design without prospective validation; message-level rather than patient-level holdout (with 41 of 1,679 patients appearing in both training and test sets at the message level — a 2.4% rate of cross-set patient overlap that is reported transparently rather than concealed); single-turn evaluation, which does not characterize multi-turn dialogue extensions in which the system can ask clarifying questions before acting; absence of temporal external validation; and absence of external validation on populations other than the participating Medicaid managed care entity's enrollees.

### Future work — research agenda for closing the population gap

Closing the population gap documented in this study — moving from the best-balanced achievable operating point (sensitivity 0.667 and specificity 0.609 via the three-of-nine hard-voting ensemble) to a clinical-grade operating point (sensitivity ≥ 0.80 AND specificity ≥ 0.80) — requires a structured research program. Five complementary directions are identified.

First, larger and population-relevant training corpora. The development set in this study comprised approximately 1,280 labeled examples. Data-scaling laws for clinical natural language processing suggest that training-set size below approximately 5,000–10,000 labeled examples per hazard category constrains achievable discrimination on the long tail of presentation patterns characteristic of low-literacy multilingual messaging populations. Expanding the training corpus through systematic labeling of additional Medicaid messaging samples, with hazard-category stratification, is the highest-priority research direction.

Second, retrieval-augmented generation with population-relevant exemplars. Retrieval over a labeled in-domain corpus, with the top-k retrieved exemplars supplied to the large language model as in-context evidence at inference time, has produced substantive performance lift in recent clinical natural language processing evaluations and is feasible on the existing development set.

Third, domain-adapted small-model fine-tuning. Quantized low-rank adaptation of openly-available small large language models (Llama-3.1-8B, Mistral-7B, or comparable) on the labeled Medicaid messaging corpus may improve discrimination over zero-shot frontier-model performance, while preserving the operational advantages of locally-deployable inference.

Fourth, active learning on a held-out unlabeled corpus. Targeted labeling of the messages on which existing architectures disagree most strongly — drawn from the substantial unlabeled portion of the messaging stream — is an efficient route to closing the data gap without paying the cost of labeling messages on which architectures already agree.

Fifth, multi-turn evaluation. The present test set carries per-message labels rather than thread-level labels; multi-turn dialogue evaluation (in which the artificial intelligence can ask clarifying questions before recommending action) will require a structured re-extraction of the messaging corpus with thread linkage and conversation-level ground truth. Recent multi-turn conversational diagnostic artificial intelligence demonstrations on standardized patient actors [1] suggest that multi-turn capability may shift absolute performance for all architectures roughly in parallel; whether it specifically closes the population gap for low-literacy multilingual messaging is an open empirical question.

Two cross-cutting priorities complete the agenda. Temporal external validation will be required as patient communication patterns and the underlying disease prevalence evolve. Multi-site external validation will be required before generalization to populations outside the participating Medicaid managed care entity's enrollees — different payer populations, different age strata, different geographies, and different primary languages will not necessarily reproduce the architecture-level rank ordering documented here. A prospective evaluation of clinical impact — whether deployment of the recommended cascade workflow in messaging triage reduces missed hazards relative to standard clinician-only review — is the logical final step once the foregoing have been completed.

---

## Conclusions

No artificial intelligence architecture evaluated in this study — including supervised classifiers, decision-theoretic reinforcement learning controllers, and current-generation frontier large language models with safety-augmented prompts — attained clinical-grade safety performance (sensitivity ≥ 0.80 AND specificity ≥ 0.80) for hazard detection on a real-world Medicaid messaging population, at any operating point on any receiver operating characteristic curve, in any of 213 ensemble configurations evaluated, or in any of 72 two-stage cascade configurations evaluated. The best balanced operating point achieved was sensitivity 0.667 and specificity 0.609 (a hard-voting ensemble of three architectures); the conventional clinical-grade screening threshold of sensitivity ≥ 0.80 and specificity ≥ 0.80 used in established clinical computer-aided detection benchmarks was not reached. This finding is specific to the population evaluated: Medicaid patients using text messaging as a substitute for limited primary care access, with low-literacy, high-colloquialism, abbreviated, and implicit-context communication patterns that are systematically underrepresented in prior clinical artificial intelligence evaluation. Deployment of the evaluated architectures in this setting requires human-in-the-loop oversight at clinically meaningful rates; the asymmetric-reward Conservative Q-Learning controller, by virtue of its architectural commitment to sensitivity over specificity, is a defensible first-stage screen in a clinician-in-the-loop cascade workflow. Closing the population gap to autonomous clinical-grade performance is the explicit research agenda outlined in Future Work above, anchored on larger population-relevant training corpora, retrieval-augmented generation with in-domain exemplars, domain-adapted small-model fine-tuning, active learning, and multi-turn dialogue evaluation. Every reported number in this study derives from a single canonical pipeline run; per-message predictions for every architecture are released as supplementary data so that any reviewer can independently re-derive the reported tables, threshold analyses, ensemble configurations, and cascade Pareto frontier.

---

## References

1. Saab K, Park C, Strother T, Freyberg J, Barrett DGT, Cheng Y, Weng WH, Stutz D, Tomasev N, Palepu A, et al. Advancing conversational diagnostic AI with multimodal reasoning. Nat Med. 2026 May 14. doi:10.1038/s41591-026-04371-0. PMID: 42135531.

2. Bedi S, Cui H, Fuentes M, Unell A, Wornow M, Banda JM, Kotecha N, Keyes T, Mai Y, Oez M, et al. Holistic evaluation of large language models for medical tasks with MedHELM. Nat Med. 2026 Mar;32(3):943-951. doi:10.1038/s41591-025-04151-2. PMID: 41559415.

3. Liu S, Wright AP, McCoy AB, Huang SS, Steitz B, Wright A. Detecting emergencies in patient portal messages using large language models and knowledge graph-based retrieval-augmented generation. J Am Med Inform Assoc. 2025;32(6):1032-1039. doi:10.1093/jamia/ocaf059. PMID: 40220286.

4. Šimunović I, Rezić K, Franić N, Boduljak G, Batinić M, Jukić I, Jelovina I, Biočić J, Pogorelić Z, Markić J. Can general purpose large language models assist pediatricians in predicting infants with serious bacterial infection? BMC Med Inform Decis Mak. 2025;25(1):423. doi:10.1186/s12911-025-03258-3. PMID: 41239388.

5. Esmaeilzadeh P. Ethical implications of using general-purpose LLMs in clinical settings: a comparative analysis of prompt engineering strategies and their impact on patient safety. BMC Med Inform Decis Mak. 2025;25(1):342. doi:10.1186/s12911-025-03182-6. PMID: 41024005.

6. Saban M, Alon Y, Luxenburg O, Singer C, Hierath M, Karoussou Schreiner A, Brkljačić B, Sosna J. Comparison of CT referral justification using clinical decision support and large language models in a large European cohort. Eur Radiol. 2025 Oct;35(10):6150-6159. doi:10.1007/s00330-025-11608-y. PMID: 40287868. PMCID: PMC12417242.

7. Nazlı MA, Esmerer E, Keles A. Benchmarking large language models in breast cancer care: agreement with radiology-led multidisciplinary tumor board decisions. BMC Med Inform Decis Mak. 2026; in press. doi:10.1186/s12911-026-03556-4. PMID: 42121209.

8. Basu S, Muralidharan B, Sheth P, Wanek D, Morgan J, Patel S. Reinforcement Learning to Prevent Acute Care Events Among Medicaid Populations: Mixed Methods Study. JMIR AI. 2025;4:e74264. doi:10.2196/74264. PMID: 41062083.

9. Kumar A, Zhou A, Tucker G, Levine S. Conservative Q-learning for offline reinforcement learning. Adv Neural Inf Process Syst. 2020;33:1179-1191.

10. Newcombe RG. Interval estimation for the difference between independent proportions: comparison of eleven methods. Stat Med. 1998;17(8):873-890. doi:10.1002/(SICI)1097-0258(19980430)17:8<873::AID-SIM779>3.0.CO;2-I.

11. Reimers N, Gurevych I. Sentence-BERT: sentence embeddings using siamese BERT-networks. Proc 2019 Conf Empir Methods Nat Lang Process. 2019:3982-3992.

12. Devlin J, Chang MW, Lee K, Toutanova K. BERT: pre-training of deep bidirectional transformers for language understanding. Proc 2019 Conf North Am Chapter Assoc Comput Linguist (NAACL-HLT). 2019:4171-4186. doi:10.18653/v1/N19-1423.

13. Chen T, Guestrin C. XGBoost: a scalable tree boosting system. Proc 22nd ACM SIGKDD Int Conf Knowl Discov Data Min. 2016:785-794. doi:10.1145/2939672.2939785.

14. McNemar Q. Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika. 1947;12(2):153-157. doi:10.1007/BF02295996.

15. Hochberg Y. A sharper Bonferroni procedure for multiple tests of significance. Biometrika. 1988;75(4):800-802. doi:10.1093/biomet/75.4.800.

16. Collins GS, Moons KGM, Dhiman P, Riley RD, Beam AL, Van Calster B, Ghassemi M, Liu X, Reitsma JB, van Smeden M, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378. doi:10.1136/bmj-2023-078378. PMID: 38626948.

17. Vasey B, Nagendran M, Campbell B, Clifton DA, Collins GS, Denaxas S, Denniston AK, Faes L, Geerts B, Ibrahim M, et al. Reporting guideline for the early stage clinical evaluation of decision support systems driven by artificial intelligence: DECIDE-AI. BMJ. 2022;377:e070904. doi:10.1136/bmj-2022-070904. PMID: 35584845.

18. Shah NH, Halamka JD, Saria S, Ashley E, Beam A, Boussard T, Bates DW, Borenstein N, Cao Z, Davidson K, et al. A nationwide network of health AI assurance laboratories. JAMA. 2024;331(3):245-249. doi:10.1001/jama.2023.26930. PMID: 38117493.

19. Coalition for Health AI. Blueprint for Trustworthy AI Implementation Guidance and Assurance for Healthcare. April 2023. Accessed 2026-05-20. https://chai.org/chai-unveils-blueprint-for-trustworthy-ai-in-healthcare/

20. Sendak MP, Kim JY, Hasan A, Mahowald T, Smith J, Singh A, Balu S, Pencina M, Bedoya A, Economou-Zavlanos N, et al. Empowering US healthcare delivery organizations: cultivating a community of practice to harness AI and advance health equity. PLOS Digit Health. 2024;3(6):e0000513. doi:10.1371/journal.pdig.0000513. PMID: 38843115.

---

## Tables

### Table 1. Population characteristics of the held-out real-world test set and the physician-scripted comparison scenarios.

{table1_population_block}

Caption: Sample sizes, hazard prevalence, age distribution, sex, self-reported race/ethnicity, and linguistic feature distributions for the {n_total}-message real-world test set and the 41-case physician-scripted comparison set. Linguistic features computed via the automated NLP pipeline described in Methods. Reading level by Flesch-Kincaid grade.

### Table 2. Per-architecture hazard detection metrics on the real-world test set.

{table2_detection_metrics_block}

Caption: Sensitivity, specificity, positive predictive value, negative predictive value, F1, Matthews correlation coefficient, area under the receiver operating characteristic curve (where calibrated probability output is available), and false negatives per 1,000 messages at each architecture's calibrated operating threshold on the {n_total}-message held-out real-world test set. 95% confidence intervals: Wilson score method [12] for sensitivity, specificity, positive predictive value, and negative predictive value; parametric bootstrap (10,000 iterations, seed 42) for F1 and Matthews correlation coefficient; Hanley-McNeil for area under the receiver operating characteristic curve.

### Table 3. Operating-point analysis: sensitivity at matched specificity targets for architectures with calibrated probability output.

{table3_operating_points_block}

Caption: For each architecture with calibrated probability output, the threshold that achieves the target specificity is identified on the architecture's receiver operating characteristic curve, and the sensitivity attained at that threshold is reported with the achieved specificity displayed alongside the target. The receiver operating characteristic curves derive from the same per-message prediction files as Table 2; monotonicity (sensitivity non-decreasing as specificity decreases) is enforced as a pipeline assertion.

### Table 4. Action recommendation appropriateness on the real-world test set.

{table4_action_recommendations_block}

Caption: For each architecture, the proportion of recommendations classified as appropriate (matching the physician-adjudicated reference action), under-triage (recommending a less urgent action), or over-triage (recommending a more urgent action) on the 8-point ordinal action scale, on the {n_total}-message held-out real-world test set.

### Table 5. Cascade architectures: Pareto frontier of two-stage AND-rule configurations on the real-world test set.

{table5_cascade_pareto_block}

Caption: Each row reports a Stage 1 architecture (high-recall screen) paired with a Stage 2 architecture (high-precision filter), with the final positive decision requiring concordance between both stages. Sensitivity, specificity, positive predictive value, F1, and Matthews correlation coefficient are computed on the AND-rule cascade output. Pareto-frontier pairs (those not dominated on both sensitivity and specificity by any other pair) are listed; no cascade configuration attained sensitivity ≥ 0.80 AND specificity ≥ 0.80 simultaneously. The cascade pairing the asymmetric-reward Conservative Q-Learning controller with the safety-augmented Claude Opus 4.7 frontier large language model is the highest-specificity pair in the cascade matrix that retains non-trivial sensitivity. The full 72-pair cascade matrix is provided in Multimedia Appendix 1, Table S3.

### Table 6. Threshold-optimized single-architecture operating points and clinical-grade reachability.

{table6_threshold_optimized_block}

Caption: For each architecture with calibrated probability output, the F1-maximizing and Matthews-correlation-coefficient-maximizing operating points on the real-world test set receiver operating characteristic curve are reported, along with the closest-to-clinical-grade operating point and a binary indicator of whether the architecture's receiver operating characteristic curve passes through the sensitivity ≥ 0.80 AND specificity ≥ 0.80 region. Threshold-optimized analysis is post hoc: the thresholds reported here are selected on the real-world test set operating curve and characterize the upper bound of threshold-tuning achievable performance; clinical deployment requires threshold selection on a held-out validation set or prospective calibration. The headline result is that no architecture reaches the clinical-grade target at any operating point on its receiver operating characteristic curve.

### Table 7. Closing-the-gap interventions: synthesis of best operating points under each strategy.

{table7_closing_the_gap_block}

Caption: For each closing-the-gap strategy evaluated, the best balanced operating point (highest minimum of sensitivity and specificity) and the best-F1 operating point on the real-world test set are reported, with a final-column indicator of whether the strategy attained the clinical-grade target (sensitivity ≥ 0.80 AND specificity ≥ 0.80). Strategies evaluated: single architecture at default threshold (Table 2), single architecture at the F1-maximizing threshold (Table 6), ensemble of all nine architectures across 213 configurations, two-stage AND-rule cascade across all 72 configurations (Table 5), multi-large-language-model consensus rules from the existing Claude + Gemini predictions, and retrieval-augmented generation over the 1,280-example labeled training corpus. No strategy reached the clinical-grade target; the best balanced operating point across all strategies was the three-of-nine hard-voting ensemble (sensitivity 0.667, specificity 0.609).

### Table 8. Operational synthesis of the four-component guardrail composition drawing on Shah et al. [18], Coalition for Health AI [19], and Sendak et al. [20].

{table8_amc_guardrail_stack_block}

Caption: Each row reports a configuration of the four-component guardrail composition operationalized in this study as the authors' synthesis of the assurance-laboratory evaluation framework of Shah et al. [18], the principles articulated in the Coalition for Health AI Blueprint [19], and the organizational community-of-practice governance approach of Sendak et al. and the Health AI Partnership [20]. The component-1+3 pairing (safety-augmented Claude Opus 4.7 + retrieval-augmented generation over the labeled training corpus) is reported alone as a baseline, and is then composed with each of seven evaluated learned-classifier pre-screen gates (component 4) under an AND-rule cascade requiring concordance between the gate and the language model. Component 2 (content-moderation pre-screen) is modeled as a pass-through on the clinically-themed messages of the test set. None of the seven evaluated configurations of the operational synthesis attained sensitivity ≥ 0.80 AND specificity ≥ 0.80 on the real-world test set.

### Table 9. Deployment-grade configurations: sensitivity-floor analysis and disagreement-stratified clinician-review workflow.

{table9_deployment_grade_block}

Caption: Panel A — for each of four sensitivity floors (≥ 0.80, ≥ 0.85, ≥ 0.90, ≥ 0.95), the configuration achieving the maximum specificity at or above that floor across the full pool of evaluated configurations (single architectures at all observed thresholds, ensembles, cascades, multi-large-language-model consensus rules, retrieval-augmented generation). Panel B — distribution of the {n_total} real-world test-set messages across agreement strata (number of architectures flagging each message), with the policy recommendation for clinician review or autonomous disposition at each stratum, the resulting hazard prevalence within the stratum, and the share of total message volume comprising the stratum.

### Table 10. Two deployment policies vs individual-component baselines: empirical alert volume and residual miss rate per 1,000 messages.

{table10_deployment_policies_block}

Caption: Per-configuration sensitivity, specificity, positive predictive value, alert count per 1,000 messages (count of positive predictions), hazards caught per 1,000 messages (count of true positives), and hazards missed per 1,000 messages (count of false negatives) on the {n_total}-message real-world test set. All quantities are computed directly from empirical sensitivity and specificity on the held-out test set; the table reports counts of alerts and counts of misses rather than estimates of clinician time, full-time equivalents, or institutional staffing capacity, none of which were measured in the present study. The clinical-grade target row is included as the autonomous-deployment reference benchmark; the {n_hazards}-hazard distribution would admit up to 16.5 missed hazards per 1,000 messages under the sensitivity ≥ 0.80 floor. Policy A and Policy B both produce residual miss rates below the clinical-grade-target maximum while producing alert volumes 2.53-fold (Policy A) and 3.03-fold (Policy B) higher than the best balanced single-architecture operating point (XGBoost with sentence-BERT embeddings at default threshold, 288 alerts per 1,000 messages).

---

## Figures

### Figure 1. Per-architecture sensitivity change from physician-scripted to real-world Medicaid test sets.

{figure1_caption_block}

Caption: Each architecture's sensitivity on the physician-scripted 41-case comparison set is plotted against its sensitivity on the held-out {n_total}-message real-world test set. The diagonal indicates equal performance across populations; architectures below the diagonal lose sensitivity when moving from the physician comparison scenarios to the real-world Medicaid messaging population. Error bars reflect Wilson 95% confidence intervals.

### Figure 2. Action recommendation appropriateness on the real-world test set, by architecture.

{figure2_caption_block}

Caption: Stacked-bar chart of the proportion of recommendations classified as appropriate (matching the physician-adjudicated reference action), under-triage (recommending a less urgent action than the reference), or over-triage (recommending a more urgent action) on the 8-point ordinal action scale, on the real-world test set.

### Figure 3. Receiver operating characteristic envelope across all evaluated strategies on the real-world test set.

{figure3_caption_block}

Caption: Each strategy is plotted as one or more (specificity, sensitivity) points on the real-world test set: single-architecture default-threshold operating points (circles), two-stage cascade Pareto frontier (squares), hard-voting k-of-9 ensemble configurations (triangles), best soft-voting ensemble (star), and multi-large-language-model consensus rules (diamonds). The clinical-grade target zone (sensitivity ≥ 0.80 AND specificity ≥ 0.80) is shaded in the top-right of the figure. No strategy reaches the clinical-grade target zone — the structural finding of this study, that no combination of currently-evaluated architectures and ensemble or cascade strategies is deployable as an autonomous single-stage screen on this Medicaid messaging population.

21. Wu D, Haredasht FN, Maharaj SK, et al. First, do NOHARM: towards clinically safe large language models. arXiv 2512.01241; deposited 2025-12-17. PubMed preprint PMID 41532042. [Preprint; not yet peer-reviewed at time of submission.]
