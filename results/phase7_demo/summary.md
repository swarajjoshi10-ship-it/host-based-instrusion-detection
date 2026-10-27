mode: demo

# Phase 7 - LID-DS 2021 (test half, mean +/- std over model seeds)

## Bruteforce_CWE-307

- Scoring rule: mean (chosen on the tuning half)
- Protocol: train 210 | calibration 60 | tuning half 453 (69 attacks) | test half 454 (70 attacks)
- Windows: vocabulary 38, train windows 20000 (cap 20000)
- Trivial baselines (AUC): length 0.5988, threads 0.8532, freq_if 0.9876

| Threshold | Precision | Recall | F1 | False alarm rate | AUC |
|---|---|---|---|---|---|
| 99th pct (normal data) | 0.707+/-0.0 | 1.0+/-0.0 | 0.828+/-0.0 | 0.075+/-0.0 | 0.999+/-0.0 |
| 95th pct (normal data) | 0.648+/-0.0 | 1.0+/-0.0 | 0.786+/-0.0 | 0.099+/-0.0 | 0.999+/-0.0 |
| Tuned (uses attacks) | 1.0+/-0.0 | 0.957+/-0.0 | 0.978+/-0.0 | 0.0+/-0.0 | 0.999+/-0.0 |

Model: 28968 parameters, 116.0 KB, 0.0322 ms/window (1 thread)

- Note: This scenario is easy: a frequency-only Isolation Forest already reaches AUC 0.988, so a high LSTM score here is not strong evidence on its own.
- Note: Thresholds learned from validation recordings gave about 7.5% false alarms at the 99th percentile, so validation and test normals differ somewhat.
- Note: The tuned threshold uses labelled attack recordings from the tuning half, so it is not a normal-data-only setting.
