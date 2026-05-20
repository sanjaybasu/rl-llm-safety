| Cascade | Sens | Spec | PPV | F1 | MCC |
|---|---|---|---|---|---|
| CQL reward-opt × LogReg+TF-IDF | 0.915 | 0.167 | 0.090 | 0.164 | 0.062 |
| ActionHead × LogReg+TF-IDF | 0.418 | 0.809 | 0.165 | 0.236 | 0.154 |
| CQL reward-opt × XGBoost+SBERT | 0.418 | 0.736 | 0.125 | 0.192 | 0.095 |
| LogReg+TF-IDF × XGBoost+SBERT | 0.418 | 0.736 | 0.124 | 0.192 | 0.095 |
| ActionHead × CQL reward-opt | 0.412 | 0.809 | 0.162 | 0.233 | 0.149 |
| Claude Opus 4.7 × CQL reward-opt | 0.297 | 0.874 | 0.174 | 0.220 | 0.135 |
| Claude Opus 4.7 × LogReg+TF-IDF | 0.291 | 0.875 | 0.173 | 0.217 | 0.132 |
| ActionHead × XGBoost+SBERT | 0.255 | 0.888 | 0.169 | 0.203 | 0.119 |
| ActionHead × Claude Opus 4.7 | 0.176 | 0.945 | 0.223 | 0.197 | 0.135 |
| Claude Opus 4.7 × XGBoost+SBERT | 0.158 | 0.944 | 0.202 | 0.177 | 0.114 |
| Constellation × XGBoost+SBERT | 0.067 | 0.982 | 0.244 | 0.105 | 0.089 |
| Constellation × CQL reward-opt | 0.067 | 0.972 | 0.177 | 0.097 | 0.062 |
| Constellation × LogReg+TF-IDF | 0.067 | 0.972 | 0.177 | 0.097 | 0.062 |
| CQL reward-opt × Guardrails | 0.061 | 0.983 | 0.244 | 0.097 | 0.085 |
| Guardrails × LogReg+TF-IDF | 0.061 | 0.983 | 0.238 | 0.097 | 0.083 |
| ActionHead × Constellation | 0.049 | 0.981 | 0.186 | 0.077 | 0.056 |
| Guardrails × XGBoost+SBERT | 0.042 | 0.992 | 0.318 | 0.075 | 0.090 |
| Claude Opus 4.7 × Constellation | 0.042 | 0.986 | 0.219 | 0.071 | 0.063 |
| Claude Opus 4.7 × Guardrails | 0.030 | 0.989 | 0.200 | 0.053 | 0.048 |
| CQL sens-opt × XGBoost+SBERT | 0.024 | 0.998 | 0.500 | 0.046 | 0.096 |
| ActionHead × CQL sens-opt | 0.024 | 0.997 | 0.400 | 0.046 | 0.082 |
| Constellation × CQL sens-opt | 0.024 | 0.997 | 0.400 | 0.046 | 0.082 |
| CQL reward-opt × CQL sens-opt | 0.024 | 0.997 | 0.400 | 0.046 | 0.082 |
| CQL sens-opt × LogReg+TF-IDF | 0.024 | 0.997 | 0.400 | 0.046 | 0.082 |
| ActionHead × Guardrails | 0.024 | 0.994 | 0.250 | 0.044 | 0.055 |
| Gemini 3.1 Pro × XGBoost+SBERT | 0.018 | 0.999 | 0.600 | 0.035 | 0.094 |
| Claude Opus 4.7 × CQL sens-opt | 0.018 | 0.998 | 0.500 | 0.035 | 0.083 |
| Claude Opus 4.7 × Gemini 3.1 Pro | 0.018 | 0.996 | 0.300 | 0.034 | 0.056 |
| CQL reward-opt × Gemini 3.1 Pro | 0.018 | 0.995 | 0.231 | 0.034 | 0.044 |
| Gemini 3.1 Pro × LogReg+TF-IDF | 0.018 | 0.995 | 0.231 | 0.034 | 0.044 |
| Constellation × Gemini 3.1 Pro | 0.012 | 1.000 | 0.667 | 0.024 | 0.082 |
| Constellation × Guardrails | 0.012 | 1.000 | 0.667 | 0.024 | 0.082 |
| ActionHead × Gemini 3.1 Pro | 0.012 | 0.998 | 0.333 | 0.023 | 0.050 |
| CQL sens-opt × Gemini 3.1 Pro | 0.006 | 1.000 | 1.000 | 0.012 | 0.075 |
| CQL sens-opt × Guardrails | 0.006 | 1.000 | 1.000 | 0.012 | 0.075 |
| Gemini 3.1 Pro × Guardrails | 0.000 | 1.000 | 0.000 | 0.000 | -0.007 |
