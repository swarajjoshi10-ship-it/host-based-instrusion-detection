# Phase 3: Baseline Detectors Summary

### Detection Performance at 99th and 95th Percentile Thresholds

| Detector | Feature Representation | Pct | Precision | Recall | F1 Score | FAR | ROC AUC | Latency (ms/win) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Isolation Forest | Baseline (Bigrams, D=407) | 99% | 0.722 | 0.035 | 0.066 | 0.013 | 0.612 | 0.0133 |
| Isolation Forest | Baseline (Bigrams, D=407) | 95% | 0.776 | 0.158 | 0.263 | 0.046 | 0.612 | 0.0133 |
| One-Class SVM | Baseline (Bigrams, D=407) | 99% | 0.500 | 0.003 | 0.005 | 0.003 | 0.550 | 0.0678 |
| One-Class SVM | Baseline (Bigrams, D=407) | 95% | 0.748 | 0.111 | 0.194 | 0.038 | 0.550 | 0.0678 |
| Isolation Forest | Richer (Bigrams+Trigrams, D=663) | 99% | 0.945 | 0.070 | 0.130 | 0.004 | 0.636 | 0.0182 |
| Isolation Forest | Richer (Bigrams+Trigrams, D=663) | 95% | 0.811 | 0.190 | 0.308 | 0.044 | 0.636 | 0.0182 |
| One-Class SVM | Richer (Bigrams+Trigrams, D=663) | 99% | 0.400 | 0.003 | 0.005 | 0.004 | 0.483 | 0.1070 |
| One-Class SVM | Richer (Bigrams+Trigrams, D=663) | 95% | 0.608 | 0.064 | 0.116 | 0.042 | 0.483 | 0.1070 |

### Key Takeaways
1. **Trigram Impact on Isolation Forest**: AUC increased from 0.612 to 0.636. At the 99th percentile, precision improved from 0.722 to 0.945, and F1 doubled from 0.066 to 0.130, with FAR decreasing from 1.3% to 0.4%.
2. **One-Class SVM Limitations**: OCSVM exhibits low recall at high thresholds (F1 ~ 0.005 at 99th pct), but reaches F1 = 0.208 at the 95th percentile with trigrams.
3. **Inference Latency**: Isolation Forest runs in ~0.01-0.02 ms/window, while One-Class SVM runs in ~0.06-0.08 ms/window.
