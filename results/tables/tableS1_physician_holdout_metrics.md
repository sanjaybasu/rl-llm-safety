| Architecture | TP/FN/TN/FP | Sensitivity (95% CI) | Specificity (95% CI) | F1 | MCC | AUROC |
|---|---|---|---|---|---|---|
| ActionHead (action recommender) | 11/16/10/4 | 0.407 (0.245–0.593) | 0.714 (0.454–0.883) | 0.524 | 0.120 | N/A |
| Constellation architecture | 23/4/8/6 | 0.852 (0.675–0.941) | 0.571 (0.326–0.786) | 0.821 | 0.441 | 0.772 |
| CQL controller (reward-optimized) | 27/0/5/9 | 1.000 (0.875–1.000) | 0.357 (0.163–0.612) | 0.857 | 0.518 | 0.794 |
| CQL controller (sensitivity-optimized) | 20/7/9/5 | 0.741 (0.553–0.868) | 0.643 (0.388–0.837) | 0.769 | 0.373 | 0.794 |
| Rule-based guardrails | 2/25/13/1 | 0.074 (0.021–0.234) | 0.929 (0.685–0.987) | 0.133 | 0.005 | 0.503 |
| Logistic regression + TF-IDF | 27/0/2/12 | 1.000 (0.875–1.000) | 0.143 (0.040–0.399) | 0.818 | 0.314 | 1.000 |
| XGBoost + sentence-BERT | 26/1/11/3 | 0.963 (0.817–0.993) | 0.786 (0.524–0.924) | 0.929 | 0.780 | 0.979 |
