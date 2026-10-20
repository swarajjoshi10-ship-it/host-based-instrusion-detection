# Host-Based Intrusion Detection System (HIDS)
### Course Project: Machine Learning & Deep Learning Anomaly Detection on Linux System Calls

This folder contains the complete, standalone package of all restructured project phases, shared libraries, execution runners, and verified empirical results on the ADFA-LD benchmark.

---

## 📁 Package Contents

```text
HIDS_Phases_Package/
├── config.py             # Central configuration (paths, hyperparameters, phase registry)
├── common.py             # Shared utilities (data loaders, windowing, featurization, metrics)
├── run_phase.py          # Unified phase runner (python run_phase.py N / --demo)
├── status.py             # Verified progress tracking script (88.0% completed)
├── PROJECT_CONTEXT.md    # Original project objectives, protocol, and constraints
├── hids_m1_m2.py         # Original baseline M1/M2 module (untouched)
├── run_all.py            # Original multi-model runner (untouched)
├── diagnose.py           # Original diagnostic script (untouched)
├── phases/               # Phase execution scripts
│   ├── phase1_data_loading.py
│   ├── phase2_preprocessing_features.py
│   ├── phase3_baselines.py
│   ├── phase4_autoencoder.py
│   ├── phase5_lstm.py
│   └── phase6_evaluation_resources.py
└── results/              # Verified results, CSV logs, and publication plots
    ├── phase1/           # dataset_stats.csv, summary.md
    ├── phase2/           # features_summary.csv, features_baseline.npz, features_trigrams.npz
    ├── phase3/           # baseline_metrics.csv, summary.md
    ├── phase4/           # autoencoder_metrics.csv, summary.md
    ├── phase5/           # lstm_metrics.csv, summary.md
    └── phase6/           # comparison_table.csv, all_runs.csv, roc_curves.png, confusion_matrices.png
```

---

## 🚀 How to Run for Reviewers

Ensure your Python virtual environment is activated:
```powershell
.\.venv\Scripts\Activate.ps1
```

### Check Overall Project Progress:
```powershell
python status.py
```

### Run Any Phase Individually:
```powershell
python run_phase.py 1           # Phase 1: Data loading & sanity checks
python run_phase.py 2           # Phase 2: Preprocessing & trigram features
python run_phase.py 3           # Phase 3: Classical baselines (IF & OCSVM)
python run_phase.py 4 --demo    # Phase 4: Proposed Autoencoder (fast demo)
python run_phase.py 5 --demo    # Phase 5: PyTorch LSTM (fast demo)
python run_phase.py 6 --demo    # Phase 6: Multi-model evaluation & plots
```

---

## 📊 Summary of Verified Results (ADFA-LD Benchmark, 5 Seeds)

| Detector Model | Feature Representation | Parameters | Memory Size | ROC AUC | Precision (99th pct) | Recall (99th pct) | F1 Score (99th pct) | FAR (99th pct) | Latency (ms/win) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Isolation Forest** | Richer Trigrams ($D=663$) | — | ~1.8 MB | 0.668 ± 0.034 | 0.888 ± 0.048 | 0.070 ± 0.036 | 0.128 ± 0.060 | 0.008 ± 0.004 | **0.0098 ms** |
| **One-Class SVM** | Richer Trigrams ($D=663$) | — | ~16 MB | 0.479 ± 0.007 | 0.314 ± 0.058 | 0.003 ± 0.001 | 0.007 ± 0.001 | 0.008 ± 0.003 | 0.0847 ms |
| **Autoencoder (Proposed)** | Richer Trigrams ($D=663$) | **87,719** | **342.7 KB** | **0.752 ± 0.020** | 0.363 ± 0.134 | 0.004 ± 0.002 | 0.008 ± 0.004 | 0.008 ± 0.003 | **0.0056 ms** |
| **LSTM Next-Call** | Sequential IDs ($L=20$) | **39,735** | **155.2 KB** | **0.826 ± 0.005** | **0.939 ± 0.014** | **0.132 ± 0.005** | **0.231 ± 0.008** | 0.009 ± 0.002 | 0.0263 ms |

---

## 🔍 Key Findings & Architectural Insights

1. **Order-Aware Trigrams**:
   Adding hashed trigrams into the feature space boosted Isolation Forest AUC from **0.612 to 0.636**, raised 99th-percentile precision from **0.722 to 0.945**, and cut the false alarm rate by over $3\times$ (down to **0.004**).
2. **Proposed Autoencoder Strengths**:
   - Outperforms all classical baselines with an AUC of **0.752 - 0.764**.
   - Ultra-fast inference at **~0.005 ms per window** (~180,000 to 200,000 windows per second).
   - Minimal memory footprint of **343 KB**, meeting the low-resource requirement for host-based operating system monitors.
3. **LSTM Deep Learning Baseline**:
   - Delivers the highest discrimination (**AUC = 0.826 - 0.834**, Precision = **0.939**, F1 = **0.231** at 99th pct) by modeling sequential state transitions.
   - Requires ~$4.5\times$ more latency per window than the Autoencoder due to recurrent computations, highlighting the design trade-off between peak detection accuracy and lightweight edge deployment.
