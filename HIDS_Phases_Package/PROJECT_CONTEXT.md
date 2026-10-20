# Project: Host-Based Intrusion Detection Using System Call Analysis (OS course project)

## Goal
Build a LIGHTWEIGHT host-based intrusion detector (HIDS) that finds anomalies in Linux system call sequences with
high accuracy AND low resource cost, and compare it fairly against standard baselines on two datasets
(ADFA-LD and LID-DS 2021). Detection only (no automatic response). Anomaly-based: train on NORMAL data only.

## Environment
- Windows, PowerShell, VS Code-based editor, Python venv at .\.venv (activate: .\.venv\Scripts\Activate.ps1)
- Installed: numpy, scikit-learn, psutil. PyTorch NOT installed yet (needed for the LSTM baseline).
- ADFA-LD location: C:\Users\Gaurav Prasad\Documents\ADFA-LD
  (folders: Training_Data_Master, Validation_Data_Master, Attack_Data_Master; one .txt per trace = space-separated syscall numbers)
- Never modify raw dataset files.

## Files in this folder
- hids_m1_m2.py  : M1 (find_dirs, read_traces, windows, featurize, trace_windows, trace_scores) + baseline runner.
  featurize = L1-normalised call-frequency histogram (vocab built from TRAINING data only, last bin = unknown)
  + hashed bigram counts (B=256 bins). trace_scores = MEAN of window scores (the original MAX rule was biased by trace length).
  Defaults: window L=20, stride S=10.
- diagnose.py    : prints trace-length stats and Isolation Forest AUC by window length and trace-score rule.
- run_all.py     : runs Isolation Forest, One-Class SVM and the proposed autoencoder (sklearn MLPRegressor, layers 64-16-64,
  Adam, early stopping, score = reconstruction MSE) with identical data and split; reports metrics at the 99th and 95th
  percentile thresholds, AUC, ms/window, params, model size. Writes results_all.csv.
- results_adfa_ld.csv, results_all.csv : saved per-seed outputs.

## Protocol (keep consistent across all experiments)
- ADFA-LD split: train = 833 normal training traces; test = all 746 attack traces + 746 normal validation traces
  (random, fixed seed); calibration = remaining normal validation traces. Threshold = percentile of normal calibration
  trace scores. Trace flagged if its score exceeds the threshold. Sequence-level (trace) metrics.
- Metrics: precision, recall, F1, false alarm rate (FP/(FP+TN)), ROC AUC, confusion matrix.
- Resource metrics: inference ms/window, CPU % of the detector process, peak memory (RSS), model size, training time.
- Report mean +/- std over 5 seeds for final numbers.
- Sanity check on every run: M1 must print traces train 833 / validation 4372 / attack 746.

## REAL results so far (ADFA-LD, 1 seed, L=20, S=10, mean-window scoring)
99th percentile threshold: Isolation Forest P 0.722 R 0.035 F1 0.066 FAR 0.013 AUC 0.612 | One-Class SVM P 0.500 R 0.003 F1 0.005
FAR 0.003 AUC 0.550 | Autoencoder P 0.286 R 0.003 F1 0.005 FAR 0.007 AUC 0.750.
95th percentile: F1 = 0.263 (IF), 0.194 (OCSVM), 0.144 (AE).
Speed: IF 0.0108 ms/window, OCSVM 0.0881, AE 0.0042. Autoencoder: 54,695 parameters (~427 KB), train ~64 s.
Finding: first scoring rule (MAX over windows) gave AUC 0.35 because long normal traces have more windows; replaced by MEAN.
Status: F1 target (>= 0.90) is NOT met. Likely cause: features (histogram + bigrams) lose call order.

## Targets (planned goals, not results)
F1 >= 0.90 on both datasets; CPU overhead < 5%; inference a few ms per window. If unreachable, report honestly.

## Next tasks, in this order (finish and verify each before the next)
1. Improve features to keep call ORDER: add hashed trigram counts (and test raw counts vs normalised); re-run the
   comparison; keep a small grid over L in {20, 50, 100}. Report which change helps, with real numbers.
2. Add an LSTM next-call baseline in PyTorch (embedding 32, one 64-unit layer; score = mean negative log-likelihood of
   the next call in a window; train on normal only). Install torch with pip first.
3. Proper resource measurement: replay test windows, measure CPU %, peak RSS, model size, ms/window for every detector;
   5 seeds; write a results table CSV.
4. LID-DS 2021: use the official loader (github.com/LID-DS/LID-DS). Start with two scenarios (brute-force and SQL injection),
   use the predefined train/validation/test recordings; recording-level flag if any window exceeds the threshold.
5. Ablation: window length, with/without bigram/trigram features, autoencoder size.
6. Save ROC curves, confusion matrices and comparison plots into a results/ folder; write a README with exact run commands.

## Rules for the AI agent
- NEVER invent, estimate, or "sample" numbers. Every number in a table must come from a command actually run; show the
  command and its real output. If something fails, report the error.
- Work in small steps, run the code after each change, and tell me what changed and what the output shows.
- Do not change the evaluation protocol silently; if a change is needed, explain why and keep old results for comparison.
- Keep scripts runnable from PowerShell; use fixed seeds; save configs and outputs to CSV.
- Explain results in plain language (I prefer simple analogies over jargon) and flag anything that looks wrong.
- Course rubric (Review 2): at least 50% of the work completed with valid demonstrated outputs; documented methodology,
  modules, workflows, dataset details, experimental plan and metrics, challenges and plan for remaining work.