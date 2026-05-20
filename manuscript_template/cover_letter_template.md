2026-05-20

Editor-in-Chief
BMC Medical Informatics and Decision Making

Dear Editor,

We submit the enclosed manuscript, AI safety evaluation in an underrepresented population: real-world performance of clinical decision support and frontier language models on Medicaid patient messaging triage, for consideration by BMC Medical Informatics and Decision Making.

This study addresses a documented gap in the clinical artificial intelligence safety literature. Prior evaluations of clinical decision support and frontier large language models have been conducted predominantly on academic medical center cohorts, scripted patient-actor scenarios, or standardized clinical knowledge benchmarks. The populations who most depend on artificial intelligence-assisted triage — Medicaid patients communicating by text message in lieu of in-person primary care access — have been systematically underrepresented. We evaluated approximately 290 architecture-and-combination configurations (nine architectures, 213 ensembles, 72 cascades, four multi-large-language-model consensus rules, six threshold-optimized single architectures, one retrieval-augmented generation configuration, and seven configurations of an operational synthesis of the four-component guardrail composition discussed in current health AI assurance literature) on {n_total} real-world Medicaid messaging hazard-detection cases. No configuration attained clinical-grade autonomous safety performance (sensitivity ≥ 0.80 AND specificity ≥ 0.80). Two deployment-grade configurations under clinician-augmented operation were feasible, with empirical alert volumes and residual miss rates characterized per 1,000 messages.

All results derive from a single deterministic pipeline. Per-message predictions for every architecture × every test record are released as Multimedia Appendix 2 alongside this manuscript. The released file contains no patient identifiers beyond opaque message identifiers and is sufficient for any reviewer to independently verify every reported value. The code repository is at https://github.com/sanjaybasu/rl-llm-safety.

The manuscript follows TRIPOD+AI (Collins GS et al., BMJ 2024;385:e078378) for the supervised components and DECIDE-AI (Vasey B et al., BMJ 2022;377:e070904) for the deployed-system early-clinical-evaluation framing. The completed reporting checklists appear in Multimedia Appendix 1, Appendices A and B. The main text contains one figure (a sensitivity-change plot) and ten tables, with three additional figures (operating-curve overlays, calibration diagrams, and a per-category sensitivity bar chart) and four supplementary tables in Multimedia Appendix 1. This study has not been published elsewhere and is not under consideration by another journal.

We propose the following potential reviewers, drawn from four different institutions across the United States. None are former collaborators of the authors, and none are employed by firms that compete with the participating Medicaid managed care entity:

1. Monica Agrawal, PhD — Assistant Professor of Biostatistics and Bioinformatics, Computer Science, and Biomedical Engineering, Division of Translational Biomedical Informatics, Duke University. Co-author on a 2025 J Am Med Inform Assoc paper applying a unified health large-language-model evaluation framework to in-basket message replies, with explicit attention to failure-mode analysis in real patient-messaging workflows. Email: monica.agrawal@duke.edu

2. R. Andrew Taylor, MD, MHS — Associate Professor of Biomedical Informatics and Data Science and Director of AI and Data Science, Department of Emergency Medicine, Yale School of Medicine. Principal investigator of National Institutes of Health–funded work on improving factual correctness of large language models in healthcare, with senior authorship on multiple recent papers evaluating large-language-model clinical decision support and large-language-model-as-judge frameworks. Email: richard.taylor@yale.edu

3. Adam Rodman, MD, MPH, FACP — Assistant Professor of Medicine, Harvard Medical School; Director of AI Programs, Carl J. Shapiro Institute, Beth Israel Deaconess Medical Center; Associate Editor, NEJM AI. Mid-career voice on real-world (versus benchmark) evaluation of clinical large language models. Email: arodman@bidmc.harvard.edu

4. Siru Liu, PhD — Assistant Professor, Department of Biomedical Informatics, Vanderbilt University Medical Center. First author on the most directly comparable patient-portal messaging triage evaluation (Liu et al., J Am Med Inform Assoc 2025;32:1032-1039); familiar with the operating-point characteristics this manuscript extends to a Medicaid messaging population. Email: siru.liu.1@vanderbilt.edu

We do not request the exclusion of any specific reviewer.

Thank you for considering this work.

Sincerely,

Sanjay Basu, MD, PhD
Waymark and University of California, San Francisco
sanjay.basu@waymarkcare.com
