| Strategy | Best balanced (sens, spec) | Best F1 | Sens at best F1 | Spec at best F1 | Clinical-grade reached? |
|---|---|---|---|---|---|
| **Clinical-grade target** | ≥0.80, ≥0.80 | — | — | — | Target |
| Single architecture (default threshold) | 0.424, 0.724 (XGBoost + sentence-BERT) | 0.234 | 0.424 | 0.802 | No |
| Single architecture (threshold-optimized) | 0.594, 0.592 (XGBoost + sentence-BERT) | 0.201 | 0.412 | 0.759 | No |
| Ensemble of 9 architectures (213 configurations evaluated) | 0.667, 0.609 (hard_3_of_9) | 0.232 | 0.370 | 0.836 | No |
| Two-stage cascade (72 configurations evaluated) | 0.418, 0.809 (ActionHead (action recommender) × Logistic regression + TF-IDF) | 0.236 | 0.418 | 0.809 | No |
| Multi-LLM consensus (Claude + Gemini) | 0.297, 0.871 (claude_only) | 0.217 | 0.297 | 0.871 | No |
