2026-05-20

Editor-in-Chief
BMC Medical Informatics and Decision Making

Dear Editor,

We submit the enclosed manuscript, AI safety evaluation in an underrepresented population: real-world performance of clinical decision support and frontier language models on Medicaid patient messaging triage, for consideration by BMC Medical Informatics and Decision Making.

This study addresses a documented gap in the clinical artificial intelligence safety literature. Prior evaluations of clinical decision support and frontier large language models have been conducted predominantly on academic medical center cohorts, scripted patient-actor scenarios, or standardized clinical knowledge benchmarks. The populations who most depend on artificial intelligence-assisted triage — Medicaid patients communicating by text message in lieu of in-person primary care access — have been underrepresented. We evaluated approximately 290 architecture-and-combination configurations (nine architectures, 213 ensembles, 72 cascades, four multi-large-language-model consensus rules, six threshold-optimized single architectures, one retrieval-augmented generation configuration, and seven configurations of an operational synthesis of the four-component guardrail composition discussed in current health AI assurance literature) on {n_total} real-world Medicaid messaging hazard-detection cases drawn from a multi-state clinically integrated network across three U.S. states (Virginia, Washington, Ohio). No configuration attained a stringent symmetric autonomous benchmark of sensitivity ≥ 0.80 AND specificity ≥ 0.80 (anchored to the IDx-DR autonomous-diagnostic-AI pivotal trial). Two clinician-augmented deployment-grade configurations were feasible, instantiating the classical clinical-screening cascade of a high-sensitivity artificial intelligence screen followed by clinician confirmatory review, with empirical alert volumes and residual miss rates characterized per 1,000 messages.

All results derive from a single deterministic pipeline. Per-message predictions for every architecture × every test record are retained by the corresponding author and available on request for verification purposes; patient message text remains under HIPAA data-use agreement. The complete code repository is publicly available at https://github.com/sanjaybasu/rl-llm-safety.

The manuscript follows TRIPOD+AI (Collins GS et al., BMJ 2024;385:e078378) for the supervised components and DECIDE-AI (Vasey B et al., BMJ 2022;377:e070904) for the deployed-system early-clinical-evaluation framing. The completed reporting checklists appear in Multimedia Appendix 1, Appendices A and B. The main text contains one figure (a sensitivity-change plot) and ten tables, with three additional figures (operating-curve overlays, calibration diagrams, and a per-category sensitivity bar chart) and four supplementary tables in Multimedia Appendix 1. This study has not been published elsewhere and is not under consideration by another journal.

We propose the following potential reviewers:

1. Siru Liu, PhD — Assistant Professor, Department of Biomedical Informatics, Vanderbilt University Medical Center. First author on the most directly comparable patient-portal messaging triage evaluation (Liu et al., J Am Med Inform Assoc 2025;32:1032-1039). Email: siru.liu.1@vanderbilt.edu

2. R. Andrew Taylor, MD, MHS — Associate Professor of Biomedical Informatics and Data Science and Director of AI and Data Science, Department of Emergency Medicine, Yale School of Medicine. Principal investigator of National Institutes of Health–funded work on improving factual correctness of large language models in healthcare, with senior authorship on multiple recent papers evaluating large-language-model clinical decision support and large-language-model-as-judge frameworks. Email: richard.taylor@yale.edu

3. Marika M. Cusick, PhD — Assistant Professor, Department of Health Policy and Management, Johns Hopkins Bloomberg School of Public Health. First author on Cusick et al., "A Novel Decision-Modeling Framework for Health Policy Analyses When Outcomes Are Influenced by Social and Disease Processes," Medical Decision Making 2026; methodological expertise in decision-modeling under social processes for health policy analyses. Email: marikacusick@jhu.edu

4. Deepshikha C. Ashana, MD, MBA, MS — Assistant Professor, Department of Population Health Sciences and Division of Pulmonary, Allergy, and Critical Care Medicine, Duke University School of Medicine. Co-author on the American Thoracic Society Research Statement on mitigating racial and ethnic disparities in U.S. critical care medicine (Hauschildt et al., AJRCCM 2025); critical-care perspective on equity in clinical artificial intelligence deployment. Email: deepshikha.ashana@duke.edu

We do not request the exclusion of any specific reviewer.

Thank you for considering this work.

Sincerely,

Sanjay Basu, MD, PhD
Waymark and University of California, San Francisco
sanjay.basu@waymarkcare.com
