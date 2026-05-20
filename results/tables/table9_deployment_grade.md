**Panel A — Sensitivity-floor target: max specificity achievable subject to sensitivity threshold**

| Sensitivity floor | N configurations meeting floor | Max specificity at floor | Winning configuration | Sens (winning) |
|---|---|---|---|---|
| ≥ 0.80 | 2911 | 0.359 | single_architecture_threshold_sweep / logreg_tfidf@thresh | 0.800 |
| ≥ 0.85 | 2242 | 0.282 | single_architecture_threshold_sweep / logreg_tfidf@thresh | 0.855 |
| ≥ 0.90 | 1563 | 0.191 | single_architecture_threshold_sweep / xgboost_sbert@thresh | 0.903 |
| ≥ 0.95 | 887 | 0.098 | single_architecture_threshold_sweep / xgboost_sbert@thresh | 0.952 |


**Panel B — Disagreement-stratified clinician-review policy**

| Stratum (n architectures flagging) | Recommended policy | N messages | N hazards (true) | Hazard prevalence in stratum | Share of total messages |
|---|---|---|---|---|---|
| 0-1 flagging (autonomous benign) | no action | 253 | 10 | 0.040 | 12.7% |
| 2-3 flagging (clinician review) | clinician review | 1385 | 94 | 0.068 | 69.2% |
| 4-6 flagging (clinician review) | clinician review | 355 | 57 | 0.161 | 17.8% |
| 7-9 flagging (autonomous escalation) | escalate | 7 | 4 | 0.571 | 0.4% |