| Configuration | Sensitivity | Specificity | PPV | Alerts per 1,000 messages (clinician review load) | Hazards caught per 1,000 messages | Hazards missed per 1,000 messages | Note |
|---|---|---|---|---|---|---|---|
| **Clinical-grade target (autonomous deployment)** | ≥ 0.800 | ≥ 0.800 | — | — | — | ≤ 16.5 | Clinical CAD/CDS benchmark (mammography, CT PE) |
| Best balanced single architecture (XGBoost + sentence-BERT) | 0.424 | 0.724 | 0.121 | 288 | 35.0 | 47.5 | default threshold; baseline reference |
| Highest-sensitivity single architecture (Logistic regression + TF-IDF) | 0.970 | 0.093 | 0.088 | 913 | 80.0 | 2.5 | default threshold; reference for sens-floor analysis |
| Best balanced ensemble (hard_3_of_9) | 0.667 | 0.609 | 0.133 | 413 | 55.0 | 27.5 | hard- or soft-voting; reference for ensemble closing-the-gap finding |
| Best balanced cascade (ActionHead (action recommender) × Logistic regression + TF-IDF) | 0.418 | 0.809 | 0.165 | 209 | 34.5 | 48.0 | two-stage AND-rule; reference for cascade closing-the-gap finding |
| **Policy A** — high-recall screen (logreg_tfidf@thresh) | 0.855 | 0.282 | 0.097 | 729 | 70.5 | 12.0 | sens >= 0.85 floor; ALL flagged messages → clinician confirmation queue |
| **Policy B** — disagreement-stratified triage (≥2 of 10 architectures flag → clinician review) | 0.939 | 0.132 | 0.089 | 873 | 77.5 | 5.0 | 0-1 flagging → autonomous benign; 2+ → clinician review; 7+ → autonomous escalation |
