2026-05-19

Editor-in-Chief
BMC Medical Informatics and Decision Making

Dear Editor,

We are pleased to submit the enclosed manuscript, "AI safety evaluation in an underrepresented Medicaid population: real-world performance of clinical decision support and frontier language models on patient messaging triage," for consideration by BMC Medical Informatics and Decision Making.

This study addresses a documented gap in the clinical artificial intelligence safety literature. Prior evaluations of clinical decision support and frontier large language models have been conducted predominantly on academic medical center cohorts, scripted patient-actor scenarios, or standardized clinical knowledge benchmarks. The populations who most depend on artificial intelligence-assisted triage — those whose access to in-person primary care is constrained — have been systematically underrepresented. Across a structured search of 11 PubMed query axes spanning Medicaid clinical artificial intelligence, low-literacy patient triage, multilingual natural language processing, and SMS-as-primary-care-substitute, seven returned zero hits. The most relevant published analogue (Liu et al., JAMIA 2025) operated on academic medical center patient portal messages at standard reading levels and did not address the Medicaid, SMS, or low-literacy axes.

We evaluated six artificial intelligence configurations on {n_total} text messages submitted by Medicaid managed care patients during {study_period}. The configurations comprised a deployed local supervised clinical decision support system based on a Conservative Q-Learning controller with constellation classifier and action recommender, a classical supervised baseline (XGBoost with sentence-BERT embeddings), a rule-based guardrail system, a logistic regression with TF-IDF features, and two current-generation frontier large language models with safety-augmented system prompts (Claude Opus 4.7 and Gemini 3.1 Pro Preview). All configurations were evaluated single-turn — corresponding to "screening at message arrival" — and reported under TRIPOD+AI and DECIDE-AI dual reporting frameworks.

Three features distinguish this contribution. First, the patient population is materially different from prior clinical artificial intelligence evaluation cohorts. Messages averaged grade {grade_level_realworld} reading level versus grade {grade_level_physician} in physician-scripted comparison scenarios, contained colloquialisms in {colloquialism_pct_realworld}% of messages versus {colloquialism_pct_physician}% in physician scenarios, and used abbreviations in {abbreviation_pct_realworld}% of messages versus {abbreviation_pct_physician}%. Second, all results derive from a single canonical pipeline run (run identifier {run_id}). Per-message predictions for every architecture × every test record are released as Supplementary File 2 (also archived at Zenodo under {zenodo_doi}). The released file contains no patient identifiers beyond opaque message identifiers, and patient message text remains under data-use agreement; a reviewer with the released supplementary file can independently verify receiver operating characteristic monotonicity, McNemar discordant-pair counts, paired bootstrap confidence intervals, and table-to-table arithmetic. Third, the comparator set includes current-generation frontier large language models (Claude Opus 4.7 and Gemini 3.1 Pro Preview) on real-world patient communications — a comparison that, to our knowledge, has not previously been published in the peer-reviewed literature for a Medicaid managed care population at the linguistic profile observed here.

The manuscript follows TRIPOD+AI (Collins GS et al., BMJ 2024;385:e078378) for the supervised components and DECIDE-AI (Vasey B et al., BMJ 2022;377:e070904) for the deployed-system early-clinical-evaluation framing. The completed reporting checklists appear in Appendices A and B. The manuscript main text is approximately {word_count_main_text} words; Multimedia Appendix 1 is approximately {word_count_appendix} words. Four figures and four tables are included in the main text; supplementary figures and tables appear in Multimedia Appendix 1.

The code repository (https://github.com/sanjaybasu/rl-llm-safety) contains the complete pipeline. The single Modal-orchestrated entry point produces every per-message prediction file and every reported metric in one deterministic run, with each architecture's calibrated threshold selected on a held-out validation set and never on the test set.

This study has not been published elsewhere and is not under consideration by another journal. All authors have approved the final version. The authors declare no competing interests beyond the ones listed in the declarations.

We propose the following potential reviewers, none of whom are former collaborators of the authors. We do not request the exclusion of any specific reviewer.

[Suggested reviewers — to be filled by author]
1.
2.
3.
4.

Thank you for considering this work.

Sincerely,

Sanjay Basu, MD, PhD
Waymark and University of California, San Francisco
sanjay.basu@ucsf.edu
