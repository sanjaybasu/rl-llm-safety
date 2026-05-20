| Architecture | F1-max | Sens at F1-max | Spec at F1-max | MCC-max | Sens at MCC-max | Spec at MCC-max | Clinical-grade reachable? | Sens/Spec at closest point |
|---|---|---|---|---|---|---|---|---|
| Constellation architecture | 0.197 | 0.376 | 0.781 | 0.109 | 0.218 | 0.904 | No | 0.564 / 0.563 |
| CQL controller (reward-optimized) | 0.175 | 0.412 | 0.704 | 0.096 | 0.067 | 0.983 | No | 0.539 / 0.540 |
| CQL controller (sensitivity-optimized) | 0.175 | 0.412 | 0.704 | 0.096 | 0.067 | 0.983 | No | 0.539 / 0.540 |
| Rule-based guardrails | 0.153 | 1.000 | 0.002 | 0.083 | 0.061 | 0.983 | No | 0.061 / 0.983 |
| Logistic regression + TF-IDF | 0.183 | 0.412 | 0.723 | 0.095 | 0.873 | 0.281 | No | 0.551 / 0.564 |
| XGBoost + sentence-BERT | 0.201 | 0.412 | 0.759 | 0.108 | 0.412 | 0.759 | No | 0.594 / 0.592 |
