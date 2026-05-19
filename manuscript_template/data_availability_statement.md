# Data and Code Availability Statement (for inclusion in main manuscript Declarations section)

## Data availability

The per-message predictions for every architecture × every test record evaluated in this study are released as Multimedia Appendix 2 (also archived at Zenodo, DOI {zenodo_doi}). The Multimedia Appendix 2 archive contains one CSV per architecture per dataset, with columns: `message_id`, `dataset`, `true_hazard`, `true_action`, `hazard_category`, `pred_proba`, `pred_hazard`, `pred_action`, `threshold_used`, `architecture`, `model_version`, `run_id`, `inference_time_s`. These predictions contain no patient identifiers beyond opaque CUIDs generated specifically for this evaluation.

The patient message text underlying these predictions is governed by a HIPAA data-use agreement and the participating Medicaid managed care entity's data-handling policy. Patient message text is not included in the released supplementary archive. Bona-fide academic researchers may request access to the message text by contacting the corresponding author (sanjay.basu@ucsf.edu) with a brief description of the proposed reanalysis and confirmation of an applicable IRB or data-use-agreement; the participating entity reserves discretion over each request.

The released per-message predictions are sufficient for any reviewer or independent researcher to verify (a) receiver operating characteristic monotonicity for every architecture with calibrated probability outputs, (b) McNemar discordant-pair counts for every pairwise comparison, (c) paired bootstrap confidence intervals for cross-set degradation, (d) table-to-table arithmetic consistency, and (e) every metric reported in the manuscript without access to the message text.

## Code availability

The complete pipeline used to produce every reported number is available at https://github.com/sanjaybasu/rl-llm-safety. A single Modal-orchestrated entry point (`code/pipeline/modal_pipeline.py`) executes 14 phases in one deterministic run, producing the per-message prediction files released as supplementary data and the canonical metrics CSVs from which every table and figure derives.

Deterministic reproducibility is enforced via seed=42 throughout (`set_global_seed()` is called at the start of `canonical_training.train_all()` and applies to NumPy, Python `random`, PyTorch CPU and CUDA, and CuDNN). The sentence-BERT pretrained model used for embeddings is `sentence-transformers/all-mpnet-base-v2`.

The pipeline produces:
- One trained-artifact pickle per architecture in `models/v4/`
- One per-message canonical CSV per architecture × dataset in `predictions/canonical/`
- One canonical metrics CSV in `results/metrics_canonical.csv`
- One pairwise McNemar matrix in `results/mcnemar_matrix.csv`
- One Δ-bootstrap CSV in `results/delta_bootstrap_canonical.csv`
- One operating-curves CSV in `results/operating_curves.csv`
- All tables and figures rendered from those canonical CSVs in `results/tables/` and `results/figures/`

Every number in the manuscript main text and Multimedia Appendix is rendered from the canonical CSVs at build time via `code/pipeline/manuscript_renderer.py`. No number is hand-typed.

## Reproducibility verification

A single audit script (`code/audit/regenerate_all_from_canonical.py --strict`) re-derives every reported value from the per-message prediction files and asserts:

1. Receiver operating characteristic monotonicity for every architecture with calibrated probability outputs (sensitivity monotone non-decreasing in threshold; specificity monotone non-increasing in threshold).
2. Cross-table arithmetic identities (under-triage rate × n_total ≤ n_hazards + non-hazardous Action 2/3 messages; sensitivity × n_hazards = TP for every row; F1 = harmonic mean of PPV and sensitivity).
3. McNemar chi-square statistics recompute from saved per-message binary predictions to within numerical precision.
4. Bootstrap confidence intervals reproduce under seed=42 with 10,000 iterations.

The audit script exits non-zero on any violation. As of submission, all assertions pass.
