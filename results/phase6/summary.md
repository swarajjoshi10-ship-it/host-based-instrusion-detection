# Phase 6: Comprehensive Evaluation & Resource Measurement Summary

mode: full

Evaluated across **5 random seed(s)** on ADFA-LD benchmark split.
Total rows in all_runs.csv: **40** (4 models × 5 seeds × 2 percentiles = 40).

### Measurement Protocol
- **Model size**: `len(pickle.dumps(model))` for sklearn/AE; `torch.save(state_dict, BytesIO)` for LSTM.
- **Threading**: `IsolationForest(n_jobs=1)`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `torch.set_num_threads(1)`.
- **inference_mem_mb**: process RSS after scoring minus RSS before scoring (capped at 0).
- **cpu_time_per_1000_windows_s**: process CPU time during scoring / n_windows × 1000.

### Multi-Seed Performance Comparison (Mean +/- Std)

| Detector | Threshold | Precision | Recall | F1 Score | FAR | ROC AUC | ms/win | Size (KB) | inf_mem MB | cpu/1k win s |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Isolation Forest | 99% | 0.888 +/- 0.048 | 0.070 +/- 0.036 | 0.128 +/- 0.060 | 0.008 +/- 0.004 | 0.668 +/- 0.034 | 0.0121 | 956.8 KB | 0.10 | 0.011954 |
| Isolation Forest | 95% | 0.804 +/- 0.021 | 0.207 +/- 0.018 | 0.329 +/- 0.024 | 0.050 +/- 0.004 | 0.668 +/- 0.034 | 0.0121 | 956.8 KB | 0.10 | 0.011954 |
| One-Class SVM | 99% | 0.314 +/- 0.058 | 0.003 +/- 0.001 | 0.007 +/- 0.001 | 0.008 +/- 0.003 | 0.479 +/- 0.007 | 0.1193 | 1370.6 KB | 0.00 | 0.114774 |
| One-Class SVM | 95% | 0.635 +/- 0.045 | 0.082 +/- 0.006 | 0.146 +/- 0.010 | 0.048 +/- 0.009 | 0.479 +/- 0.007 | 0.1193 | 1370.6 KB | 0.00 | 0.114774 |
| Autoencoder (Proposed) | 99% | 0.363 +/- 0.134 | 0.004 +/- 0.002 | 0.008 +/- 0.004 | 0.008 +/- 0.003 | 0.752 +/- 0.020 | 0.0073 | 1036.3 KB | 4.39 | 0.019043 |
| Autoencoder (Proposed) | 95% | 0.584 +/- 0.025 | 0.070 +/- 0.003 | 0.125 +/- 0.006 | 0.050 +/- 0.005 | 0.752 +/- 0.020 | 0.0073 | 1036.3 KB | 4.39 | 0.019043 |
| LSTM Next-Call | 99% | 0.938 +/- 0.014 | 0.129 +/- 0.005 | 0.228 +/- 0.007 | 0.009 +/- 0.002 | 0.828 +/- 0.009 | 0.0588 | 158.0 KB | 8.04 | 0.057716 |
| LSTM Next-Call | 95% | 0.865 +/- 0.029 | 0.276 +/- 0.035 | 0.417 +/- 0.042 | 0.042 +/- 0.006 | 0.828 +/- 0.009 | 0.0588 | 158.0 KB | 8.04 | 0.057716 |

### Generated Figures
- **ROC Curves**: saved to `roc_curves.png`
- **Confusion Matrices**: saved to `confusion_matrices.png`
