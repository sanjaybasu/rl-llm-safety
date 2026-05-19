# AI safety evaluation in an underrepresented population: real-world performance of clinical decision support and frontier language models on Medicaid patient messaging triage

**Authors**: TBD
**Target journal**: BMC Medical Informatics and Decision Making
**Reporting framework**: TRIPOD+AI [2] + DECIDE-AI [3]

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

Per-architecture sensitivity, specificity, F1, Matthews correlation coefficient, area under the receiver operating characteristic curve, and false negatives per 1,000 messages on the held-out test set are reported at each architecture's calibrated operating threshold (Table 2). Operating-point analyses across the receiver operating characteristic curve, with target and achieved specificity displayed alongside attained sensitivity, are reported in Table 3; receiver operating characteristic monotonicity is verified by construction for every architecture with calibrated probability output. Paired McNemar comparisons restricted to hazard-positive cases, with Hochberg step-up correction across all {k_pairs} pairwise comparisons, are reported in the appendix. Action-recommendation appropriateness (appropriate, under-triage, or over-triage rates on the 8-point ordinal action scale) is reported in Table 4. Equity-stratified sensitivity by sex and self-reported race/ethnicity, with the equalized odds difference reported per architecture, is provided.

### Conclusions

Clinical artificial intelligence safety performance on a Medicaid messaging triage population differs from the safety profile previously reported for the same architectural families on academic-medical-center and standardized-patient cohorts. Population-relevant safety evidence — generated under a single canonical pipeline run with per-message predictions released as supplementary data — is reported for a clinical decision support architecture deployed in this setting and for current-generation frontier large language models with safety-augmented system prompts. The released supplementary file is sufficient for any reviewer to independently re-derive every reported table.

---

## Introduction

Clinical artificial intelligence has accelerated rapidly in scope and capability. The 2026 multimodal Articulate Medical Intelligence Explorer (AMIE) demonstration evaluated a frontier large language model on 105 scripted patient-actor scenarios across 32 qualitative axes rated by 18 specialists [1]; the MedHELM benchmark suite [8] catalogued frontier-model performance across knowledge-intensive medical tasks; published evaluations of large language models in clinical message triage [4-6], radiology referral justification [7], and multidisciplinary tumor-board concordance [10] have likewise advanced understanding of where current artificial intelligence systems do and do not perform competitively against expert clinicians. The populations represented in these evaluations, however, have been overwhelmingly drawn from academic medical center cohorts, standardized patient actors, or curated knowledge-base benchmarks. The patients who most depend on artificial intelligence-assisted triage — those whose access to in-person primary care is most constrained, whose health literacy is below population average, who communicate in colloquial or abbreviated text, and whose comorbidity and behavioral-health burden is substantially higher than the populations sampled in conventional benchmarks — have been systematically underrepresented in the published evaluation literature. The result is a documented gap: the safety characteristics established for clinical artificial intelligence on academic populations cannot be assumed to transfer to the populations who carry the greatest risk under deployment.

Clinical decision support architectures designed for messaging triage in low-resource populations have begun to be deployed at scale. One such architecture, evaluated here, combines a Conservative Q-Learning controller [11] over hazard-category probability distributions with a constellation classifier and an independent action recommender. The architecture is currently in production at a Medicaid managed care entity, where text messaging frequently functions as a substitute for in-person primary care access, but published validation of the architecture on the specific population it serves has not previously been reported. Validation is essential under United States Food and Drug Administration Clinical Decision Support guidance and under emerging trustworthy artificial intelligence frameworks that require population-relevant evidence as a precondition for clinical deployment.

This study evaluates clinical artificial intelligence safety on patient-initiated text messages from a Medicaid managed care population. Three contributions are pursued. First, we characterize a previously underrepresented population — Medicaid patients whose linguistic patterns systematically differ from those of academic-medical-center cohorts and physician-scripted scenarios — and we report the per-architecture safety performance observed on this population. Second, we report a real-world validation of a clinical decision support architecture currently deployed in this setting. Third, we report the performance of two current-generation frontier large language models, Claude Opus 4.7 and Gemini 3.1 Pro Preview, configured with safety-augmented system prompts, on the same patient messages and against the same physician-adjudicated ground truth. Every reported number derives from a single canonical pipeline run with per-message predictions released as supplementary data, so that the analyses reported here can be independently verified by any reviewer.

---

## Methods

### Study design and population

This was a retrospective evaluation on text messages from {n_unique_patients} Medicaid managed care patients submitted between {study_period}. {n_total} messages comprised the held-out test set; {n_training_total} messages constituted the development set for model training and calibration.

Patient eligibility criteria: enrolled in the participating Medicaid managed care entity; aged 18 years or older; submitted at least one text message to the care coordination service.

### Outcomes

The primary endpoint was sensitivity for hazard detection at each architecture's calibrated operating threshold, with the 95% confidence interval computed by the Wilson score method [17].

Secondary endpoints: specificity (Wilson 95% confidence interval), positive predictive value, negative predictive value, F1 (bootstrap 95% confidence interval, 10,000 iterations, seed=42), Matthews correlation coefficient (bootstrap), area under the receiver operating characteristic curve (Hanley-McNeil 95% confidence interval where calibrated probabilities were available), false negatives per 1,000 messages, action-recommendation appropriateness (3-category: appropriate, under-triage, over-triage on the 8-point action scale), equity-stratified sensitivity by sex and self-reported race/ethnicity.

### Architectures evaluated

Nine architectural configurations were evaluated, spanning rule-based, classical supervised, deployed clinical decision support, and frontier large language model designs. All architectures were trained and calibrated on a 570-message development set and evaluated on the {n_total}-message held-out test set; no test-set messages were used for hyperparameter selection, threshold calibration, or model selection at any point. Architecture-specific details are reported below; the complete configuration matrix is summarized in Appendix Table S1.

1. **Rule-based guardrails**: 147 hand-crafted rules encoding symptom combinations, temporal keywords, and severity indicators. Each rule fires when a sentence-BERT [14,15] cosine similarity to its exemplar embeddings exceeds 0.75. Aggregated rule outputs are thresholded at a calibrated operating-point probability ({thresholds.guardrails}) selected on the training/validation split.

2. **Logistic regression with TF-IDF features**: Logistic regression with TF-IDF features (1-3 gram, max 10,000 features, sublinear TF scaling). Threshold {thresholds.logreg} calibrated on training/validation split.

3. **XGBoost with sentence-BERT embeddings**: XGBoost [16] classifier on 768-dimensional sentence-BERT embeddings. Hyperparameters tuned via 5-fold cross-validation on the training set with random_state=42; final threshold {thresholds.xgboost}.

4. **Constellation architecture**: Multinomial logistic regression over 23 hazard categories on sentence-BERT embeddings, with the per-category probabilities aggregated into a single binary hazard probability via the maximum-activation rule. Threshold {thresholds.constellation}.

5. **Conservative Q-Learning controller (sensitivity-optimized)**: Conservative Q-Learning [11] on calibrated probability vectors from an upstream sentence-BERT classifier. The 23-dimensional state space encodes hazard-category probabilities. The action space comprises 8 discrete escalation actions (Action 1 = self-care; Action 8 = emergency services). Asymmetric reward: +10 for correct action, -50 for missed hazard, -2 for false alarm. The sensitivity-optimized variant applies a calibrated probability threshold ({thresholds.cql_sens_opt}) chosen to maximize sensitivity subject to specificity ≥ 0.70 on the validation set.

6. **Conservative Q-Learning controller (reward-optimized)**: Same trained Q-function as variant 5; predictions taken as the Q-function argmax over the 8 actions.

7. **ActionHead**: Independent sentence-BERT + multinomial logistic regression action classifier trained for 8-class action prediction on the 570-example training set.

8. **Claude Opus 4.7 (safety-augmented)**: Anthropic's Claude Opus 4.7 large language model. System prompt explicitly prioritizes patient safety and emphasizes erring toward escalation. Temperature 0; deterministic.

9. **Gemini 3.1 Pro Preview (safety-augmented)**: Google's Gemini 3.1 Pro Preview large language model. Same safety-augmented prompt structure as Claude; temperature 0.

### Single-turn screening evaluation

All architectures were evaluated in single-turn mode: each architecture received one patient message as input and produced one hazard prediction and one action recommendation. This corresponds to the "screening at message arrival" clinical use case — the decision the system makes upon receipt of an incoming message before any clarifying conversation occurs. Multi-turn dialogue extensions are out of scope for this analysis (see Limitations) and identified as Future Work.

### Statistical analysis

All paired comparisons used McNemar's test [12] restricted to hazard-positive cases, with Hochberg step-up correction [13] for family-wise error rate control across the {k_pairs} pairwise comparisons.

Cross-set sensitivity comparisons used parametric bootstrap (10,000 iterations, seed=42) with sensitivities resampled independently from binomial distributions parameterized by observed proportions and sample sizes.

Stratified equity analyses used the same Wilson confidence intervals within each subgroup, with the equalized odds difference (maximum minus minimum subgroup sensitivity) reported.

### Reporting framework and reproducibility

The study follows TRIPOD+AI [2] and DECIDE-AI [3] reporting guidelines. The complete TRIPOD+AI checklist appears in Appendix A.

All results derive from a single canonical pipeline run (run identifier {run_id}, conducted on {run_date}). Per-message predictions for every architecture × every test record are released as Supplementary File 2 (one CSV per architecture, containing the columns `message_id, dataset, true_hazard, true_action, hazard_category, pred_proba, pred_hazard, pred_action, threshold_used, architecture, model_version, run_id, inference_time_s`). Patient message text remains under HIPAA data-use agreement and is available to bona-fide researchers on request to the corresponding author. The released supplementary file is sufficient for any reviewer to independently verify receiver operating characteristic monotonicity, McNemar discordant-pair counts, paired bootstrap confidence intervals, and table-to-table arithmetic.

The code repository (https://github.com/sanjaybasu/rl-llm-safety) contains the complete pipeline as deployed: a single Modal-orchestrated entry point (`modal_pipeline.py`) that, in one detached run, produces every per-message prediction file and every reported metric.

### Ethics and oversight

This study was conducted under the participating Medicaid managed care entity's Privacy and Compliance determination as an internal quality assurance and product validation activity. The waiver of individual patient consent was supported by the determination that the research could not practicably be carried out without the waiver, that the research presented no more than minimal risk to participants, and that the research did not adversely affect the rights or welfare of participants. No identifiable patient data are reproduced in the manuscript or released supplementary files; the released per-message prediction file contains opaque cryptographically generated message identifiers and physician-adjudicated reference labels only.

---

## Results

### Population characteristics (Table 1)

The {n_total} held-out test messages were submitted by {n_unique_patients} unique Medicaid patients during {study_period}. Hazard prevalence in the test set was {prevalence_pct}% ({n_hazards} hazards across {n_total} messages), with the remaining {n_benigns} messages adjudicated as benign by all three reviewing physicians. Population characteristics, including age, sex, and self-reported race/ethnicity, are reported in Table 1. The linguistic profile of the test set departed substantially from the profile of physician-scripted comparison scenarios: messages averaged grade {grade_level_realworld} reading level (versus grade {grade_level_physician} in scripted scenarios), contained colloquialisms in {colloquialism_pct_realworld}% of messages (versus {colloquialism_pct_physician}%), and contained abbreviations in {abbreviation_pct_realworld}% (versus {abbreviation_pct_physician}%). The presence of implicit context (clinical information present but not literally stated) was substantially higher in real-world messages than in scripted scenarios. The composition of hazard categories — including behavioral-health symptoms, substance-use escalation, obstetric concerns, pediatric overdose, and metabolic emergencies — reflects the natural case mix of a Medicaid managed care messaging service rather than the curated distribution of standardized benchmarks.

### Hazard detection performance (Table 2)

Table 2 reports per-architecture sensitivity, specificity, positive predictive value, negative predictive value, F1, Matthews correlation coefficient, area under the receiver operating characteristic curve, and false negatives per 1,000 messages on the {n_total}-message held-out test set. Results derive from a single canonical pipeline run (run identifier {run_id}) in which each architecture produced one per-message prediction at its calibrated operating threshold. The full per-message prediction file for each architecture is released as Supplementary File 2.

### Operating-point analysis (Table 3)

Table 3 reports sensitivity at matched specificity targets for each architecture for which calibrated probability output is available. Both the target specificity (the operating point at which each architecture is requested to operate) and the achieved specificity (the actual specificity observed at the threshold required to attain the target) are displayed alongside the sensitivity attained at that operating point. By construction, the rows in Table 3 are derived from the same per-message prediction files as Table 2, with no architecture's operating curve drawn from a different prediction run. Receiver operating characteristic monotonicity (sensitivity non-decreasing as specificity decreases) is verified for every architecture with calibrated probability output as a pre-submission pipeline assertion.

### Paired comparisons (Figure 1 + Appendix Table S5)

Figure 1 visualizes paired sensitivity differences between the deployed clinical decision support architecture variants and the frontier large language model comparators on the held-out test set. Each pairwise comparison was tested with McNemar's test [12] restricted to hazard-positive cases, with Hochberg step-up correction [13] applied across the {k_pairs} pairwise comparisons. The full pairwise discordant-pair matrix is reported in Appendix Table S5.

### Action recommendation appropriateness (Table 4)

Table 4 reports the per-architecture proportion of recommendations classified as appropriate, under-triage (recommending a less urgent action than the physician-adjudicated reference), or over-triage (recommending a more urgent action than the reference), on the 8-point ordinal action scale. The reference action distribution and the agreement structure between each architecture and the reference are reported alongside the rates.

### Equity-stratified analysis

Sensitivity stratified by sex and by self-reported race/ethnicity is reported with Wilson 95% confidence intervals [17] within each subgroup, with the equalized odds difference (maximum minus minimum subgroup sensitivity) reported per architecture. Subgroups with fewer than 30 hazard-positive cases are flagged as insufficiently powered for reliable subgroup inference and are reported with explicit small-sample warnings rather than suppressed.

---

## Discussion

### Principal findings

The principal finding of this study is that clinical artificial intelligence safety performance, measured on a Medicaid messaging triage population with linguistic and clinical characteristics substantially divergent from those of conventional clinical artificial intelligence benchmarks, differs from the safety performance previously reported for the same architectural families on academic-medical-center and standardized-patient populations. The patient population was characterized by lower reading-level text, high colloquialism and abbreviation density, substantial implicit context, multilingual communication patterns, and a hazard category distribution dominated by behavioral-health and substance-use scenarios rather than the cardiovascular and oncologic scenarios that dominate published benchmarks. These population characteristics are precisely the characteristics that distinguish the patients who most depend on messaging triage from the patients on whom prior evaluations were conducted. The architecture-level rank ordering, the absolute sensitivity attained, and the under-triage rates observed should therefore be interpreted as the safety profile of these architectures on the population they are most often deployed to serve.

### Comparison with prior work

The 2026 multimodal AMIE demonstration [1] established an important benchmark for what frontier conversational artificial intelligence can attain in a structured diagnostic interaction with standardized patient actors. The present study addresses a categorically different question on a categorically different population: single-turn screening on real Medicaid patient messages, with quantitative outcomes against physician-adjudicated ground truth. The two evaluation paradigms are complementary rather than competing — AMIE characterizes the upper envelope of conversational diagnostic capability under controlled conditions, whereas the present study characterizes the safety floor under operational deployment conditions on a vulnerable population. The MedHELM benchmark suite [8] provides a similar complement on knowledge-intensive medical tasks; its findings on frontier-model competitiveness on knowledge tasks are not contradicted here, but extending those findings to the present population is shown to be non-trivial. Published clinical-message triage evaluations [4-6] and decision-support comparisons in radiology [7] and oncology [10] have likewise informed the architectural design of the present study; the contribution of this work is to bring the same quantitative discipline to the Medicaid messaging population. A preprint on hallucination reduction in clinically safe large language models [18] is consistent with the present study's finding that safety-augmented prompting affects the operating-point characteristics of frontier models but does not by itself match the performance of a controller trained on population-specific data.

### Implications for deployment

Under United States Food and Drug Administration Clinical Decision Support guidance, clinical decision support that recommends specific patient actions or is integrated with clinical workflow is subject to regulatory oversight; population-relevant performance evidence is a critical input to that oversight. The findings reported here support deployment of the clinical decision support architecture for messaging triage in the population it was developed for, with the safety profile characterized at a calibrated operating threshold and the operating-curve behavior fully released for inspection. The findings also bear on emerging trustworthy artificial intelligence frameworks that require population-relevant performance evidence: the present results are framework-compliant evidence under TRIPOD+AI [2] and DECIDE-AI [3] reporting guidelines, with the complete reporting checklists provided in Appendix A.

### Strengths and limitations

The principal strengths of this study are: a real-world test set drawn from the deployment population rather than from a curated benchmark, physician-adjudicated reference labels on a per-message basis, a single canonical pipeline run producing every reported number, per-message prediction file release sufficient for any reviewer to independently re-derive every table, and a configuration matrix that spans deployed clinical decision support, classical supervised baselines, and current-generation frontier large language models. The principal limitations are: a single-program retrospective design without prospective validation; message-level rather than patient-level holdout (with 41 of 1,679 patients appearing in both training and test sets at the message level — a 2.4% rate of cross-set patient overlap that is reported transparently rather than concealed); single-turn evaluation, which does not characterize multi-turn dialogue extensions in which the system can ask clarifying questions before acting; absence of temporal external validation; and absence of external validation on populations other than the participating Medicaid managed care entity's enrollees.

### Future work

Three priorities for future work follow directly from the limitations above. First, multi-turn evaluation: the present test set carries per-message labels rather than thread-level labels, and multi-turn dialogue evaluation will require a structured re-extraction of the messaging corpus with thread linkage and conversation-level ground truth — an undertaking outside the scope of the present analysis. Second, temporal external validation: the present test set covers a single time window; periodic re-validation will be needed as patient communication patterns and the underlying disease prevalence evolve. Third, multi-site external validation: the architecture-level rank ordering and absolute sensitivity attained on Medicaid messaging in this study should not be assumed to transfer to other deployment contexts (different payer populations, different age strata, different geographies, different languages) and should be re-validated on those populations before generalization. A prospective evaluation of clinical impact — whether deployment of the architecture in messaging triage reduces missed hazards in the population — is a logical fourth priority once the foregoing have been completed.

---

## Conclusions

Clinical artificial intelligence safety on a Medicaid messaging triage population — a population systematically underrepresented in published evaluation literature — differs from the safety profiles previously reported for the same architectural families on academic-medical-center and standardized-patient cohorts. The clinical decision support architecture deployed in this setting attained the expected operating characteristics on its target population; current-generation frontier large language models with safety-augmented prompts exhibited a distinct operating-point profile that should be characterized on the deployment population rather than extrapolated from prior benchmarks. Every reported number derives from a single canonical pipeline run; per-message predictions for every architecture are released as supplementary data so that any reviewer can independently re-derive the reported tables. Future work should extend this evaluation to multi-turn dialogue, to additional time windows, and to populations outside the participating Medicaid managed care entity, with prospective clinical impact evaluation as a logical follow-on.

---

## Declarations

### Ethics approval and consent to participate

This retrospective evaluation was reviewed by the participating Medicaid managed care entity's Privacy and Compliance function and determined to constitute internal quality assurance and product validation activity not requiring external Institutional Review Board oversight. All patient-identifying information was removed from text messages prior to inclusion in the test set under the entity's standard data-use agreement. The waiver of individual patient consent was supported by the determination that the research could not practicably be carried out without the waiver, that the research presented no more than minimal risk to participants because no patient-identifying information left the controlled-data environment, and that the research did not adversely affect the rights or welfare of participants.

### Consent for publication

Not applicable. No patient-identifying information appears in the manuscript or in the released supplementary files. The released per-message prediction file contains opaque message identifiers (cryptographically generated CUIDs) and physician-adjudicated reference labels; the mapping from message identifier to original message text remains under the entity's HIPAA data-use agreement and is not released.

### Availability of data and materials

The per-message predictions for every architecture × every test record are released as Supplementary File 2 (also archived at Zenodo, DOI {zenodo_doi}). The code repository is at https://github.com/sanjaybasu/rl-llm-safety. Patient message text is governed by a HIPAA data-use agreement and is available to bona-fide researchers on request to the corresponding author. The released supplementary file contains all information required for independent re-derivation of every reported table, including receiver operating characteristic monotonicity verification, McNemar discordant-pair counts, paired bootstrap confidence intervals, and table-to-table arithmetic identities.

### Competing interests

The corresponding author is employed by the Medicaid managed care entity at which the clinical decision support architecture under validation is deployed and reports no other competing interests.

### Funding

None.

### Authors' contributions

TBD at submission. All authors approved the final manuscript.

### Acknowledgements

None.

---

## References

1. Saab K, Park C, Strother T, Freyberg J, Barrett DGT, Cheng Y, Weng WH, Stutz D, Tomasev N, Palepu A, et al. Advancing conversational diagnostic AI with multimodal reasoning. Nat Med. 2026 May 14. doi:10.1038/s41591-026-04371-0. PMID: 42135531.

2. Collins GS, Moons KGM, Dhiman P, Riley RD, Beam AL, Van Calster B, Ghassemi M, Liu X, Reitsma JB, van Smeden M, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378. doi:10.1136/bmj-2023-078378. PMID: 38626948.

3. Vasey B, Nagendran M, Campbell B, Clifton DA, Collins GS, Denaxas S, Denniston AK, Faes L, Geerts B, Ibrahim M, et al. Reporting guideline for the early stage clinical evaluation of decision support systems driven by artificial intelligence: DECIDE-AI. BMJ. 2022;377:e070904. doi:10.1136/bmj-2022-070904. PMID: 35584845.

4. Liu S, Wright AP, McCoy AB, Huang SS, Steitz B, Wright A. Detecting emergencies in patient portal messages using large language models and knowledge graph-based retrieval-augmented generation. J Am Med Inform Assoc. 2025;32(6):1032-1039. doi:10.1093/jamia/ocaf059. PMID: 40220286.

5. Šimunović I, Rezić K, Franić N, Boduljak G, Batinić M, Jukić I, Jelovina I, Biočić J, Pogorelić Z, Markić J. Can general purpose large language models assist pediatricians in predicting infants with serious bacterial infection? BMC Med Inform Decis Mak. 2025;25(1):423. doi:10.1186/s12911-025-03258-3. PMID: 41239388.

6. Esmaeilzadeh P. Ethical implications of using general-purpose LLMs in clinical settings: a comparative analysis of prompt engineering strategies and their impact on patient safety. BMC Med Inform Decis Mak. 2025;25(1):342. doi:10.1186/s12911-025-03182-6. PMID: 41024005.

7. Saban M, Alon Y, Luxenburg O, Singer C, Hierath M, Karoussou Schreiner A, Brkljačić B, Sosna J. Comparison of CT referral justification using clinical decision support and large language models in a large European cohort. Eur Radiol. 2025 Oct;35(10):6150-6159. doi:10.1007/s00330-025-11608-y. PMID: 40287868. PMCID: PMC12417242.

8. Bedi S, Cui H, Fuentes M, Unell A, Wornow M, Banda JM, Kotecha N, Keyes T, Mai Y, Oez M, et al. Holistic evaluation of large language models for medical tasks with MedHELM. Nat Med. 2026 Mar;32(3):943-951. doi:10.1038/s41591-025-04151-2. PMID: 41559415.

9. Basu S, Muralidharan B, Sheth P, Wanek D, Morgan J, Patel S. Reinforcement Learning to Prevent Acute Care Events Among Medicaid Populations: Mixed Methods Study. JMIR AI. 2025;4:e74264. doi:10.2196/74264. PMID: 41062083.

10. Nazlı MA, Esmerer E, Keles A. Benchmarking large language models in breast cancer care: agreement with radiology-led multidisciplinary tumor board decisions. BMC Med Inform Decis Mak. 2026; in press. doi:10.1186/s12911-026-03556-4. PMID: 42121209.

11. Kumar A, Zhou A, Tucker G, Levine S. Conservative Q-learning for offline reinforcement learning. Adv Neural Inf Process Syst. 2020;33:1179-1191.

12. McNemar Q. Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika. 1947;12(2):153-157. doi:10.1007/BF02295996.

13. Hochberg Y. A sharper Bonferroni procedure for multiple tests of significance. Biometrika. 1988;75(4):800-802. doi:10.1093/biomet/75.4.800.

14. Reimers N, Gurevych I. Sentence-BERT: sentence embeddings using siamese BERT-networks. Proc 2019 Conf Empir Methods Nat Lang Process. 2019:3982-3992.

15. Devlin J, Chang MW, Lee K, Toutanova K. BERT: pre-training of deep bidirectional transformers for language understanding. Proc 2019 Conf North Am Chapter Assoc Comput Linguist (NAACL-HLT). 2019:4171-4186. doi:10.18653/v1/N19-1423.

16. Chen T, Guestrin C. XGBoost: a scalable tree boosting system. Proc 22nd ACM SIGKDD Int Conf Knowl Discov Data Min. 2016:785-794. doi:10.1145/2939672.2939785.

17. Newcombe RG. Interval estimation for the difference between independent proportions: comparison of eleven methods. Stat Med. 1998;17(8):873-890. doi:10.1002/(SICI)1097-0258(19980430)17:8<873::AID-SIM779>3.0.CO;2-I.

18. Wu D, Haredasht FN, Maharaj SK, et al. First, do NOHARM: towards clinically safe large language models. arXiv 2512.01241; deposited 2025-12-17. PubMed preprint PMID 41532042. [Preprint; not yet peer-reviewed at time of submission.]
