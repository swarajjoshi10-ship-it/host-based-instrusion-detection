"""
Phase 6: Comprehensive Evaluation and Resource Measurement
Evaluates all 4 detectors (Isolation Forest, One-Class SVM, Proposed Autoencoder, PyTorch LSTM)
across 5 random seeds to report rigorous mean +/- std metrics.

Measurement fixes applied:
  - Model size: ALL detectors measured consistently:
      sklearn models -> len(pickle.dumps(model)) / 1024  (NO hardcoded values)
      LSTM           -> torch.save(state_dict, BytesIO)  (NO hardcoded values)
  - Threading   : IsolationForest n_jobs=1; torch.set_num_threads(1); env pins in common.py.
  - Memory      : inference_mem_mb = process RSS delta during scoring only.
                  rss_before_mb = process RSS before scoring (separate column).
  - CPU cost    : cpu_time_per_1000_windows_s = process CPU time during scoring / n_windows * 1000.
  - Mode tag    : every summary and CSV row contains mode="full" or mode="demo".
"""
import sys, os, csv, time, random
from pathlib import Path
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_curve, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import common  # pins OMP/MKL env vars before numpy/sklearn

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
torch.set_num_threads(1)   # single-threaded PyTorch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── LSTM model (identical to Phase 5) ─────────────────────────────────────────
class SyscallLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        return self.fc(out)


def train_lstm_model(Xtr_seq, vocab_size, epochs=5, seed=0):
    torch.manual_seed(seed)
    model = SyscallLSTM(vocab_size, embed_dim=32, hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(Xtr_seq[:, :-1], Xtr_seq[:, 1:]), batch_size=256, shuffle=True)
    model.train()
    for ep in range(epochs):
        ep_loss = 0.0; nb = 0
        for in_b, tgt_b in loader:
            optimizer.zero_grad()
            logits = model(in_b)
            B, T, C = logits.shape
            loss = criterion(logits.reshape(B * T, C), tgt_b.reshape(B * T))
            loss.backward(); optimizer.step()
            ep_loss += loss.item(); nb += 1
        print(f"    LSTM epoch {ep+1}/{epochs} | loss {ep_loss/nb:.4f}")
    return model


def score_lstm(model, seq_tensor):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    scores = []
    loader = DataLoader(TensorDataset(seq_tensor), batch_size=512, shuffle=False)
    with torch.no_grad():
        for (batch,) in loader:
            inp = batch[:, :-1]; tgt = batch[:, 1:]
            logits = model(inp)
            B, T, C = logits.shape
            loss = criterion(logits.reshape(B * T, C), tgt.reshape(B * T))
            scores.extend(loss.reshape(B, T).mean(dim=1).cpu().numpy())
    return np.array(scores, dtype=np.float32)


# ── Main phase runner ──────────────────────────────────────────────────────────
def run_phase6(demo=False):
    mode_label = "demo" if demo else "full"

    common.print_banner(
        phase_num=6,
        title="Comprehensive Evaluation and Resource Measurement",
        analogy="The ultimate head-to-head trials: putting all four detection architectures through rigorous multi-seed testing under identical conditions, tracking both security accuracy and hardware resource costs.",
        inputs="ADFA-LD traces across 5 seeds, sliding windows L=20, S=10, all 4 detectors",
        outputs="results/phase6/comparison_table.csv, roc_curves.png, confusion_matrices.png, summary.md"
    )
    print(f"[Mode] {mode_label.upper()}")

    out_dir = common.ensure_dir(config.RESULTS_DIR / "phase6")

    # Sanity Checks
    print("[Sanity Check] Verifying previous phase outputs...")
    p1_stats = config.RESULTS_DIR / "phase1" / "dataset_stats.csv"
    if not p1_stats.is_file():
        print("[FATAL ERROR] Phase 1 output missing. Run Phase 1 first.")
        sys.exit(1)
    print("  [PASSED] Dataset and environment verified.")

    # Load traces
    d = common.find_dirs(config.DATA_DIR)
    train_traces  = common.read_traces(d["train"])
    val_traces    = common.read_traces(d["val"])
    attack_traces = common.read_traces(d["attack"])

    vocab, V = common.build_vocabulary(train_traces)
    vocab_size = V + 1

    L, S = config.DEFAULT_L, config.DEFAULT_S
    B_bi, B_tri = config.DEFAULT_B_BI, config.DEFAULT_B_TRI

    seeds = config.DEMO_SEEDS if demo else config.DEFAULT_SEEDS
    print(f"Running multi-seed evaluation on seeds: {seeds} {'[DEMO MODE]' if demo else '[FULL MODE]'}\n")

    # Pre-build training windows (fixed normal data, same for all seeds)
    tw, _ = common.trace_windows(train_traces, L, S)
    Xtr = common.featurize(tw, vocab, V, B=B_bi, B_tri=B_tri, use_trigrams=True, normalize=True)

    arr_tr = np.zeros((len(tw), L), dtype=np.int64)
    for r, w in enumerate(tw):
        arr_tr[r] = [vocab.get(c, V) for c in w]
    Xtr_seq = torch.tensor(arr_tr, dtype=torch.long)

    all_seed_results = []
    roc_plot_data = {}

    for seed in seeds:
        t_seed = time.perf_counter()
        print(f"\n{'='*60}")
        print(f"--- Seed {seed} ---")
        npr = np.random.RandomState(seed)
        calib_traces, test_traces, y_test = common.split_adfa_ld(val_traces, attack_traces, seed=seed)
        n_calib = len(calib_traces)
        n_test  = len(test_traces)

        cw, c_own = common.trace_windows(calib_traces, L, S)
        Xc = common.featurize(cw, vocab, V, B=B_bi, B_tri=B_tri, use_trigrams=True, normalize=True)

        ew, e_own = common.trace_windows(test_traces, L, S)
        Xe = common.featurize(ew, vocab, V, B=B_bi, B_tri=B_tri, use_trigrams=True, normalize=True)

        arr_c = np.zeros((len(cw), L), dtype=np.int64)
        for r, w in enumerate(cw):
            arr_c[r] = [vocab.get(c, V) for c in w]
        Xc_seq = torch.tensor(arr_c, dtype=torch.long)

        arr_e = np.zeros((len(ew), L), dtype=np.int64)
        for r, w in enumerate(ew):
            arr_e[r] = [vocab.get(c, V) for c in w]
        Xe_seq = torch.tensor(arr_e, dtype=torch.long)

        # ── Model 1: Isolation Forest ──────────────────────────────────────────
        print(f"  [IF] Training IsolationForest (n_jobs=1)...")
        m_if = IsolationForest(n_estimators=100, random_state=seed, n_jobs=1)  # SINGLE-THREADED
        t0 = time.perf_counter()
        m_if.fit(Xtr)
        train_s_if = time.perf_counter() - t0
        print(f"       Train time: {train_s_if:.1f} s")

        # Measured serialized size (pickle)
        size_kb_if = common.sklearn_serialized_kb(m_if)
        n_params_if = 0  # ensemble of trees; no single parameter count

        cs_if = common.trace_scores(-m_if.decision_function(Xc), c_own, n_calib, method="mean")
        prof_if = common.profile_inference(
            score_fn=lambda: -m_if.decision_function(Xe),
            n_windows=len(Xe)
        )
        es_if_win = prof_if["scores"]
        es_if = common.trace_scores(es_if_win, e_own, n_test, method="mean")
        met_if = common.compute_metrics(y_test, es_if, cs_if, config.PERCENTILES)

        if seed == seeds[0]:
            roc_plot_data["Isolation Forest"] = (y_test, es_if, met_if["auc"])

        for pct in config.PERCENTILES:
            all_seed_results.append({
                "mode": mode_label,
                "model": "Isolation Forest", "seed": seed, "pct": pct,
                "precision": met_if[pct]["precision"],
                "recall": met_if[pct]["recall"],
                "f1": met_if[pct]["f1"],
                "far": met_if[pct]["far"],
                "auc": met_if["auc"],
                "ms_per_win": 1000 * prof_if["wall_s"] / len(Xe),
                "inference_mem_mb": prof_if["inference_mem_mb"],
                "rss_before_mb": prof_if["rss_before_mb"],
                "cpu_time_per_1000_windows_s": prof_if["cpu_time_per_1000_windows_s"],
                "train_s": train_s_if,
                "params": n_params_if,
                "size_kb": size_kb_if,
                "tp": met_if[pct]["tp"], "fp": met_if[pct]["fp"],
                "tn": met_if[pct]["tn"], "fn": met_if[pct]["fn"]
            })

        # ── Model 2: One-Class SVM ─────────────────────────────────────────────
        print(f"  [SVM] Training One-Class SVM...")
        fit_X = Xtr
        if len(Xtr) > config.SVM_MAX_SAMPLES:
            fit_X = Xtr[npr.choice(len(Xtr), config.SVM_MAX_SAMPLES, replace=False)]
        m_svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
        t0 = time.perf_counter()
        m_svm.fit(fit_X)
        train_s_svm = time.perf_counter() - t0
        print(f"       Train time: {train_s_svm:.1f} s")

        size_kb_svm = common.sklearn_serialized_kb(m_svm)  # measured via pickle

        cs_svm = common.trace_scores(-m_svm.decision_function(Xc), c_own, n_calib, method="mean")
        prof_svm = common.profile_inference(
            score_fn=lambda: -m_svm.decision_function(Xe),
            n_windows=len(Xe)
        )
        es_svm_win = prof_svm["scores"]
        es_svm = common.trace_scores(es_svm_win, e_own, n_test, method="mean")
        met_svm = common.compute_metrics(y_test, es_svm, cs_svm, config.PERCENTILES)

        if seed == seeds[0]:
            roc_plot_data["One-Class SVM"] = (y_test, es_svm, met_svm["auc"])

        for pct in config.PERCENTILES:
            all_seed_results.append({
                "mode": mode_label,
                "model": "One-Class SVM", "seed": seed, "pct": pct,
                "precision": met_svm[pct]["precision"],
                "recall": met_svm[pct]["recall"],
                "f1": met_svm[pct]["f1"],
                "far": met_svm[pct]["far"],
                "auc": met_svm["auc"],
                "ms_per_win": 1000 * prof_svm["wall_s"] / len(Xe),
                "inference_mem_mb": prof_svm["inference_mem_mb"],
                "rss_before_mb": prof_svm["rss_before_mb"],
                "cpu_time_per_1000_windows_s": prof_svm["cpu_time_per_1000_windows_s"],
                "train_s": train_s_svm,
                "params": 0,
                "size_kb": size_kb_svm,
                "tp": met_svm[pct]["tp"], "fp": met_svm[pct]["fp"],
                "tn": met_svm[pct]["tn"], "fn": met_svm[pct]["fn"]
            })

        # ── Model 3: Autoencoder (Proposed) ────────────────────────────────────
        print(f"  [AE] Training Autoencoder...")
        ae_epochs = config.DEMO_AE_EPOCHS if demo else config.AE_EPOCHS
        m_ae = MLPRegressor(
            hidden_layer_sizes=(64, 16, 64), activation="relu", solver="adam",
            learning_rate_init=1e-3, batch_size=128, max_iter=ae_epochs,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=5,
            random_state=seed
        )
        t0 = time.perf_counter()
        m_ae.fit(Xtr, Xtr)
        train_s_ae = time.perf_counter() - t0
        print(f"       Train time: {train_s_ae:.1f} s ({m_ae.n_iter_} iterations)")

        ae_params = sum(c.size for c in m_ae.coefs_) + sum(i.size for i in m_ae.intercepts_)
        size_kb_ae = common.sklearn_serialized_kb(m_ae)  # measured via pickle

        cs_ae_win = ((Xc - m_ae.predict(Xc)) ** 2).mean(axis=1)
        cs_ae = common.trace_scores(cs_ae_win, c_own, n_calib, method="mean")

        prof_ae = common.profile_inference(
            score_fn=lambda: ((Xe - m_ae.predict(Xe)) ** 2).mean(axis=1),
            n_windows=len(Xe)
        )
        es_ae_win = prof_ae["scores"]
        es_ae = common.trace_scores(es_ae_win, e_own, n_test, method="mean")
        met_ae = common.compute_metrics(y_test, es_ae, cs_ae, config.PERCENTILES)

        if seed == seeds[0]:
            roc_plot_data["Autoencoder (Proposed)"] = (y_test, es_ae, met_ae["auc"])

        for pct in config.PERCENTILES:
            all_seed_results.append({
                "mode": mode_label,
                "model": "Autoencoder (Proposed)", "seed": seed, "pct": pct,
                "precision": met_ae[pct]["precision"],
                "recall": met_ae[pct]["recall"],
                "f1": met_ae[pct]["f1"],
                "far": met_ae[pct]["far"],
                "auc": met_ae["auc"],
                "ms_per_win": 1000 * prof_ae["wall_s"] / len(Xe),
                "inference_mem_mb": prof_ae["inference_mem_mb"],
                "rss_before_mb": prof_ae["rss_before_mb"],
                "cpu_time_per_1000_windows_s": prof_ae["cpu_time_per_1000_windows_s"],
                "train_s": train_s_ae,
                "params": ae_params,
                "size_kb": size_kb_ae,
                "tp": met_ae[pct]["tp"], "fp": met_ae[pct]["fp"],
                "tn": met_ae[pct]["tn"], "fn": met_ae[pct]["fn"]
            })

        # ── Model 4: PyTorch LSTM ──────────────────────────────────────────────
        print(f"  [LSTM] Training LSTM...")
        lstm_eps = 3 if demo else 8
        t0 = time.perf_counter()
        m_lstm = train_lstm_model(Xtr_seq, vocab_size, epochs=lstm_eps, seed=seed)
        train_s_lstm = time.perf_counter() - t0
        print(f"       Train time: {train_s_lstm:.1f} s")

        lstm_params = common.torch_param_count(m_lstm)
        size_kb_lstm = common.torch_serialized_kb(m_lstm)  # measured via BytesIO

        cs_lstm_win = score_lstm(m_lstm, Xc_seq)
        cs_lstm = common.trace_scores(cs_lstm_win, c_own, n_calib, method="mean")

        prof_lstm = common.profile_inference(
            score_fn=lambda: score_lstm(m_lstm, Xe_seq),
            n_windows=len(Xe_seq)
        )
        es_lstm_win = prof_lstm["scores"]
        es_lstm = common.trace_scores(es_lstm_win, e_own, n_test, method="mean")
        met_lstm = common.compute_metrics(y_test, es_lstm, cs_lstm, config.PERCENTILES)

        if seed == seeds[0]:
            roc_plot_data["LSTM Next-Call"] = (y_test, es_lstm, met_lstm["auc"])

        for pct in config.PERCENTILES:
            all_seed_results.append({
                "mode": mode_label,
                "model": "LSTM Next-Call", "seed": seed, "pct": pct,
                "precision": met_lstm[pct]["precision"],
                "recall": met_lstm[pct]["recall"],
                "f1": met_lstm[pct]["f1"],
                "far": met_lstm[pct]["far"],
                "auc": met_lstm["auc"],
                "ms_per_win": 1000 * prof_lstm["wall_s"] / len(Xe_seq),
                "inference_mem_mb": prof_lstm["inference_mem_mb"],
                "rss_before_mb": prof_lstm["rss_before_mb"],
                "cpu_time_per_1000_windows_s": prof_lstm["cpu_time_per_1000_windows_s"],
                "train_s": train_s_lstm,
                "params": lstm_params,
                "size_kb": size_kb_lstm,
                "tp": met_lstm[pct]["tp"], "fp": met_lstm[pct]["fp"],
                "tn": met_lstm[pct]["tn"], "fn": met_lstm[pct]["fn"]
            })

        elapsed_seed = time.perf_counter() - t_seed
        print(f"  Seed {seed} complete in {elapsed_seed:.1f} s.")

    # ── Aggregate Mean +/- Std Over Seeds ─────────────────────────────────────
    models = ["Isolation Forest", "One-Class SVM", "Autoencoder (Proposed)", "LSTM Next-Call"]
    summary_table_rows = []

    print("\n" + "=" * 135)
    print(f"{'Detector':24s} | {'Pct':>4s} | {'Precision':>14s} | {'Recall':>14s} | "
          f"{'F1':>14s} | {'FAR':>14s} | {'AUC':>14s} | {'ms/win':>8s} | {'Size KB':>9s} | {'inf_mem MB':>10s}")
    print("=" * 135)

    for m_name in models:
        for pct in config.PERCENTILES:
            subset = [r for r in all_seed_results if r["model"] == m_name and r["pct"] == pct]
            prec_m, prec_s = np.mean([r["precision"] for r in subset]), np.std([r["precision"] for r in subset])
            rec_m,  rec_s  = np.mean([r["recall"]    for r in subset]), np.std([r["recall"]    for r in subset])
            f1_m,   f1_s   = np.mean([r["f1"]        for r in subset]), np.std([r["f1"]        for r in subset])
            far_m,  far_s  = np.mean([r["far"]       for r in subset]), np.std([r["far"]       for r in subset])
            auc_m,  auc_s  = np.mean([r["auc"]       for r in subset]), np.std([r["auc"]       for r in subset])
            lat_m          = np.mean([r["ms_per_win"] for r in subset])
            inf_mem_m      = np.mean([r["inference_mem_mb"]            for r in subset])
            cpu1k_m        = np.mean([r["cpu_time_per_1000_windows_s"] for r in subset])
            size_kb_m      = np.mean([r["size_kb"]   for r in subset])
            params         = subset[0]["params"]

            summary_table_rows.append({
                "mode": mode_label,
                "model": m_name,
                "pct": pct,
                "precision": f"{prec_m:.3f} +/- {prec_s:.3f}",
                "recall":    f"{rec_m:.3f} +/- {rec_s:.3f}",
                "f1":        f"{f1_m:.3f} +/- {f1_s:.3f}",
                "far":       f"{far_m:.3f} +/- {far_s:.3f}",
                "auc":       f"{auc_m:.3f} +/- {auc_s:.3f}",
                "ms_per_window": f"{lat_m:.4f}",
                "inference_mem_mb": f"{inf_mem_m:.2f}",
                "cpu_time_per_1000_windows_s": f"{cpu1k_m:.6f}",
                "params": params,
                "size_kb": f"{size_kb_m:.1f} KB",
            })

            print(f"{m_name:24s} | {pct:4.0f} | {prec_m:.3f}+/-{prec_s:.3f} | {rec_m:.3f}+/-{rec_s:.3f} | "
                  f"{f1_m:.3f}+/-{f1_s:.3f} | {far_m:.3f}+/-{far_s:.3f} | {auc_m:.3f}+/-{auc_s:.3f} | "
                  f"{lat_m:8.4f} | {size_kb_m:7.1f} KB | {inf_mem_m:10.2f}")

    # ── Save CSVs ──────────────────────────────────────────────────────────────
    comp_csv = out_dir / "comparison_table.csv"
    with open(comp_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_table_rows)

    all_csv = out_dir / "all_runs.csv"
    with open(all_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_seed_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_seed_results)

    # ── Plot 1: ROC Curves ─────────────────────────────────────────────────────
    print("\n[Plots] Generating ROC curve comparisons...")
    plt.figure(figsize=(8, 6), dpi=300)
    colors = {
        "Isolation Forest": "#1f77b4", "One-Class SVM": "#ff7f0e",
        "Autoencoder (Proposed)": "#2ca02c", "LSTM Next-Call": "#d62728"
    }
    for m_name, (y_true, scores, auc_val) in roc_plot_data.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        plt.plot(fpr, tpr, label=f"{m_name} (AUC = {auc_val:.3f})",
                 color=colors.get(m_name, "gray"), linewidth=2.0)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.500)")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FAR)", fontsize=12, labelpad=8)
    plt.ylabel("True Positive Rate (Recall)", fontsize=12, labelpad=8)
    plt.title("ADFA-LD ROC Curves: Detection Performance Comparison", fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=10, frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    roc_png = out_dir / "roc_curves.png"
    plt.savefig(roc_png); plt.close()

    # ── Plot 2: Confusion Matrices ─────────────────────────────────────────────
    print("[Plots] Generating Confusion Matrix grid (99th percentile, first seed)...")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=300)
    first_seed = seeds[0]
    for ax, m_name in zip(axes, models):
        match = [r for r in all_seed_results
                 if r["model"] == m_name and r["seed"] == first_seed and r["pct"] == 99][0]
        cm = np.array([[match["tn"], match["fp"]], [match["fn"], match["tp"]]])
        ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(f"{m_name}\n(99th pct)", fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Attack"], fontsize=9)
        ax.set_yticklabels(["Normal", "Attack"], fontsize=9)
        ax.set_xlabel("Predicted", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("True Label", fontsize=10)
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                text_color = "white" if val > cm.max() / 2 else "black"
                ax.text(j, i, f"{val:,}", ha="center", va="center",
                        color=text_color, fontsize=11, fontweight="bold")
    plt.suptitle("Confusion Matrices at 99th Percentile Calibration Threshold",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    cm_png = out_dir / "confusion_matrices.png"
    plt.savefig(cm_png); plt.close()

    # ── Summary Markdown ───────────────────────────────────────────────────────
    summary_md = out_dir / "summary.md"
    n_rows_all = len(all_seed_results)
    with open(summary_md, "w") as fh:
        fh.write(f"# Phase 6: Comprehensive Evaluation & Resource Measurement Summary\n\n")
        fh.write(f"mode: {mode_label}\n\n")
        fh.write(f"Evaluated across **{len(seeds)} random seed(s)** on ADFA-LD benchmark split.\n")
        fh.write(f"Total rows in all_runs.csv: **{n_rows_all}** "
                 f"(4 models × {len(seeds)} seeds × 2 percentiles = {4*len(seeds)*2}).\n\n")
        fh.write("### Measurement Protocol\n")
        fh.write("- **Model size**: `len(pickle.dumps(model))` for sklearn/AE; `torch.save(state_dict, BytesIO)` for LSTM.\n")
        fh.write("- **Threading**: `IsolationForest(n_jobs=1)`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `torch.set_num_threads(1)`.\n")
        fh.write("- **inference_mem_mb**: process RSS after scoring minus RSS before scoring (capped at 0).\n")
        fh.write("- **cpu_time_per_1000_windows_s**: process CPU time during scoring / n_windows × 1000.\n\n")
        fh.write("### Multi-Seed Performance Comparison (Mean +/- Std)\n\n")
        fh.write("| Detector | Threshold | Precision | Recall | F1 Score | FAR | ROC AUC | ms/win | Size (KB) | inf_mem MB | cpu/1k win s |\n")
        fh.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in summary_table_rows:
            fh.write(f"| {r['model']} | {r['pct']:.0f}% | {r['precision']} | {r['recall']} | "
                     f"{r['f1']} | {r['far']} | {r['auc']} | {r['ms_per_window']} | "
                     f"{r['size_kb']} | {r['inference_mem_mb']} | {r['cpu_time_per_1000_windows_s']} |\n")
        fh.write("\n### Generated Figures\n")
        fh.write(f"- **ROC Curves**: saved to `roc_curves.png`\n")
        fh.write(f"- **Confusion Matrices**: saved to `confusion_matrices.png`\n")

    written_files = [str(comp_csv), str(all_csv), str(roc_png), str(cm_png), str(summary_md)]
    print("\n" + "=" * 80)
    print(f"PHASE 6 COMPLETE  [mode: {mode_label}]")
    print(f"all_runs.csv rows: {n_rows_all}")
    print("Files written:")
    for wf in written_files:
        print(f"  - {wf}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    run_phase6(demo=args.demo)
