# Host-Based Intrusion Detection Using System Call Analysis

Operating Systems project. An anomaly-based host intrusion detector that learns normal Linux system call behaviour and flags deviations. Compared detectors: Isolation Forest, One-Class SVM, a lightweight autoencoder and an LSTM next-call model, on ADFA-LD and LID-DS 2021.

## Protocol
- Training uses normal data only. The vocabulary is built from training data only.
- ADFA-LD: the 746 attack traces and an equal number of normal validation traces are split into a tuning half and a test half. All settings and the scoring rule were chosen on the tuning half. The test half was used for the final result.
- LID-DS: train on normal training recordings, threshold from normal validation recordings, test recordings split into a tuning half and a test half.
- Thresholds: 99th and 95th percentile of normal calibration scores, and one tuned for best F1 on the tuning half (this one uses labelled attacks).

## How to run
```
python run_phase.py 1            # data loading and verification
python run_phase.py 2            # preprocessing and features
python run_phase.py 3            # Isolation Forest and One-Class SVM
python run_phase.py 4            # lightweight autoencoder
python run_phase.py 5            # LSTM next-call baseline
python run_phase.py 6            # 5-seed evaluation and resource measurement
python phase6b_improve.py --data . --exp A
python phase6b_improve.py --data . --exp B --hidden 64 --layers 1
python phase6b_improve.py --data . --exp C --hidden 64 --layers 1 --L 20
python phase6b_improve.py --data . --exp E --hidden 64 --layers 1 --L 20 --extra_train 2000
python phase6b_improve.py --data . --exp FINAL --hidden 64 --layers 1 --L 20 --extra_train 2000 --ens 1 --rule mean --seeds 5
python phase6c_rules.py --data . --extra_train 3000
python run_phase.py 7            # LID-DS 2021, every scenario under .\LID-DS (use --demo for a quick test)
python run_phase.py 8            # builds this folder
```

## Results
Full tables: final_tables.md. Plots: ablation_auc.png, detector_comparison.png.

### ADFA-LD final (test half)

| Threshold | Precision | Recall | F1 | False alarm rate | AUC |
|---|---|---|---|---|---|
| 99th pct (normal data) | 0.923 +/- 0.015 | 0.097 +/- 0.016 | 0.175 +/- 0.026 | 0.008 +/- 0.002 | 0.854 +/- 0.001 |
| 95th pct (normal data) | 0.875 +/- 0.006 | 0.377 +/- 0.003 | 0.527 +/- 0.004 | 0.054 +/- 0.003 | 0.854 +/- 0.001 |
| Tuned (uses attacks) | 0.751 +/- 0.009 | 0.923 +/- 0.004 | 0.828 +/- 0.004 | 0.306 +/- 0.016 | 0.854 +/- 0.001 |

### LID-DS 2021 (test half)

**Bruteforce_CWE-307** (scoring rule mean, 3 model seeds, 454 test recordings, 70 attacks)

| Threshold | Precision | Recall | F1 | False alarm rate | AUC |
|---|---|---|---|---|---|
| 99th pct (normal data) | 0.684 +/- 0.008 | 1.0 +/- 0.0 | 0.812 +/- 0.006 | 0.084 +/- 0.003 | 1.0 +/- 0.0 |
| 95th pct (normal data) | 0.631 +/- 0.017 | 1.0 +/- 0.0 | 0.774 +/- 0.013 | 0.107 +/- 0.008 | 1.0 +/- 0.0 |
| Tuned (uses attacks) | 1.0 +/- 0.0 | 1.0 +/- 0.0 | 1.0 +/- 0.0 | 0.0 +/- 0.0 | 1.0 +/- 0.0 |

Model: 28968 parameters, 116.0 KB, 0.03 ms/window (1 thread)

## Honest notes
- ADFA-LD (test half): F1 at the 95th-percentile threshold is 0.527 with a false alarm rate of 0.054. The F1 target of 0.90 was NOT met.
- ADFA-LD: a threshold tuned for the best F1 gives F1 0.828 but a false alarm rate of 0.306, which is too high for practical use, so it is not the headline result.
- Bruteforce_CWE-307 (test half): F1 0.812 at the 99th-percentile threshold (false alarm rate 0.084) and 0.774 at the 95th (false alarm rate 0.107), using thresholds learned from normal data only.
- Phase 6c reused the same ADFA-LD test half as a second look; its best AUC differs from the 5-seed final by 0.012, so the earlier 5-seed result stays the headline.
- CPU overhead on a live host was not measured. Resource numbers are single-threaded replay measurements of inference time, memory during scoring, and serialized model size.
- The Isolation Forest, One-Class SVM and autoencoder use frequency and hashed n-gram features, so they do not see full call order. The LSTM does.
