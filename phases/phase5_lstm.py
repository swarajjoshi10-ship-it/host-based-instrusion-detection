"""
Phase 5: LSTM Next-Call Baseline in PyTorch
Trains an autoregressive LSTM next-call prediction network (embedding 32, 1-layer LSTM 64)
on normal system call sequences. Scores window anomalies via mean Negative Log-Likelihood (NLL).
Evaluates trace-level metrics at 99th and 95th percentile thresholds.

Measurement fixes applied:
  - Model size: torch.save(state_dict) to BytesIO -> measured KB.
  - Threading : torch.set_num_threads(1); env pins in common.py.
  - Memory    : inference_mem_mb = RSS delta during scoring only.
  - CPU cost  : cpu_time_per_1000_windows_s via process CPU time.
"""
import sys, os, csv, time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import common  # pins OMP/MKL env vars before numpy

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    torch.set_num_threads(1)   # single-threaded PyTorch inference/training
    TORCH_AVAILABLE = True
except ImportError:
    # Define stubs so module-level class bodies don't crash at parse time.
    # The sanity check in run_phase5() will exit cleanly if torch is absent.
    import types
    torch = None  # type: ignore
    nn = types.SimpleNamespace(Module=object)  # type: ignore
    optim = None  # type: ignore
    TensorDataset = DataLoader = None  # type: ignore
    TORCH_AVAILABLE = False


class SyscallLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        logits = self.fc(out)
        return logits


def windows_to_id_tensor(wins, vocab, V):
    """Convert raw syscall window lists to integer index tensor."""
    arr = np.zeros((len(wins), len(wins[0])), dtype=np.int64)
    for r, w in enumerate(wins):
        arr[r] = [vocab.get(c, V) for c in w]
    return torch.tensor(arr, dtype=torch.long)


def compute_nll_scores(model, tensor_data, batch_size=512):
    """Compute mean Negative Log-Likelihood (NLL) for each window."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    scores = []
    loader = DataLoader(TensorDataset(tensor_data), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (batch,) in loader:
            inp = batch[:, :-1]
            tgt = batch[:, 1:]
            logits = model(inp)
            B, T, C = logits.shape
            loss = criterion(logits.reshape(B * T, C), tgt.reshape(B * T))
            loss = loss.reshape(B, T).mean(dim=1)
            scores.extend(loss.cpu().numpy())
    return np.array(scores, dtype=np.float32)


def run_phase5(demo=False):
    mode_label = "demo" if demo else "full"

    common.print_banner(
        phase_num=5,
        title="LSTM Next-Call Baseline in PyTorch",
        analogy="Predictive autocomplete for system calls: the LSTM continuously guesses the next system call in a sequence. If an attacker triggers unexpected syscall transitions, prediction error (negative log-likelihood) spikes.",
        inputs="ADFA-LD raw traces with window length L=20, stride S=10, vocabulary from Phase 2",
        outputs="results/phase5/lstm_metrics.csv, results/phase5/summary.md"
    )
    print(f"[Mode] {mode_label.upper()}")

    out_dir = common.ensure_dir(config.RESULTS_DIR / "phase5")

    # -------------------------------------------------------------
    # Sanity Check 1: PyTorch availability
    # -------------------------------------------------------------
    print("[Sanity Check 1/2] Checking PyTorch availability...")
    if not TORCH_AVAILABLE:
        print("[FATAL ERROR] PyTorch is not installed. Please install torch first.")
        sys.exit(1)
    print(f"  [PASSED] PyTorch {torch.__version__} detected. Threads set to 1.")

    # -------------------------------------------------------------
    # Sanity Check 2: Phase 1 & 2 outputs
    # -------------------------------------------------------------
    print("\n[Sanity Check 2/2] Verifying previous phase outputs...")
    p2_base = config.RESULTS_DIR / "phase2" / "features_baseline.npz"
    if not p2_base.is_file():
        print(f"[FATAL ERROR] Phase 2 output missing at {p2_base}. Run Phase 2 first.")
        sys.exit(1)
    print("  [PASSED] Previous phase outputs verified.")

    # -------------------------------------------------------------
    # Step 1: Load traces and build window sequences
    # -------------------------------------------------------------
    print("\n[Step 1/4] Loading traces and preparing sequence tensors...")
    d = common.find_dirs(config.DATA_DIR)
    train_traces = common.read_traces(d["train"])
    val_traces   = common.read_traces(d["val"])
    attack_traces = common.read_traces(d["attack"])

    seed = config.DEMO_SEEDS[0] if demo else config.DEFAULT_SEEDS[0]
    torch.manual_seed(seed)
    np.random.seed(seed)

    calib_traces, test_traces, y_test = common.split_adfa_ld(val_traces, attack_traces, seed=seed)
    vocab, V = common.build_vocabulary(train_traces)
    vocab_size = V + 1  # 150 known + 1 unknown token

    L = config.DEFAULT_L
    S = config.DEFAULT_S
    train_wins, train_owner = common.trace_windows(train_traces, L, S)
    calib_wins, calib_owner = common.trace_windows(calib_traces, L, S)
    test_wins, test_owner   = common.trace_windows(test_traces, L, S)

    Xtr_seq = windows_to_id_tensor(train_wins, vocab, V)
    Xc_seq  = windows_to_id_tensor(calib_wins, vocab, V)
    Xe_seq  = windows_to_id_tensor(test_wins, vocab, V)

    print(f"  Vocabulary size    : {vocab_size} tokens (including unknown token {V})")
    print(f"  Train sequences    : {Xtr_seq.shape}")
    print(f"  Calib sequences    : {Xc_seq.shape}")
    print(f"  Test sequences     : {Xe_seq.shape}")

    # -------------------------------------------------------------
    # Step 2: Initialize LSTM model
    # -------------------------------------------------------------
    embed_dim  = 32
    hidden_dim = 64
    model = SyscallLSTM(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim)

    n_params = common.torch_param_count(model)
    # Measured serialized size via torch.save(state_dict) -> BytesIO
    size_kb = common.torch_serialized_kb(model)   # measured, not estimated

    print(f"\n[Step 2/4] Model Architecture:")
    print(f"  Embedding : {vocab_size} -> {embed_dim}")
    print(f"  LSTM      : 1 layer, {hidden_dim} hidden units")
    print(f"  Output    : {hidden_dim} -> {vocab_size}")
    print(f"  Parameters: {n_params:,}  |  Serialized size: {size_kb:.1f} KB")

    # -------------------------------------------------------------
    # Step 3: Train LSTM next-call predictor
    # -------------------------------------------------------------
    epochs = 3 if demo else 8
    batch_size = 256
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(Xtr_seq[:, :-1], Xtr_seq[:, 1:]),
        batch_size=batch_size,
        shuffle=True
    )

    print(f"\n[Step 3/4] Training LSTM on normal training sequences ({epochs} epochs)...")
    t0 = time.perf_counter()
    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        n_batches = 0
        for in_batch, tgt_batch in train_loader:
            optimizer.zero_grad()
            logits = model(in_batch)
            B, T, C = logits.shape
            loss = criterion(logits.reshape(B * T, C), tgt_batch.reshape(B * T))
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            n_batches += 1
        avg_loss = ep_loss / n_batches
        elapsed = time.perf_counter() - t0
        print(f"  Epoch {ep+1:2d}/{epochs:2d} | CE Loss: {avg_loss:.4f} | elapsed: {elapsed:.1f} s")
    train_time = time.perf_counter() - t0
    print(f"  Training finished in {train_time:.1f} s ({train_time/epochs:.2f} s/epoch)")

    # Refresh serialized size after training (weights changed)
    size_kb = common.torch_serialized_kb(model)
    print(f"  Serialized size (post-training): {size_kb:.1f} KB")

    # -------------------------------------------------------------
    # Step 4: Scoring & Evaluation
    # -------------------------------------------------------------
    print("\n[Step 4/4] Evaluating NLL anomaly scores on calibration and test sets...")

    # Calibration inference (scores only, not profiled)
    calib_win_scores = compute_nll_scores(model, Xc_seq)
    calib_trace_scores = common.trace_scores(calib_win_scores, calib_owner, len(calib_traces), method="mean")

    # Test inference — profiled
    prof = common.profile_inference(
        score_fn=lambda: compute_nll_scores(model, Xe_seq),
        n_windows=len(Xe_seq)
    )
    test_win_scores = prof["scores"]
    ms_per_window   = (prof["wall_s"] / len(Xe_seq)) * 1000.0
    inference_mem_mb = prof["inference_mem_mb"]
    cpu_per_1000    = prof["cpu_time_per_1000_windows_s"]
    rss_before_mb   = prof["rss_before_mb"]

    # Aggregate test trace scores & metrics
    test_trace_scores = common.trace_scores(test_win_scores, test_owner, len(test_traces), method="mean")
    metrics = common.compute_metrics(y_test, test_trace_scores, calib_trace_scores, percentiles=config.PERCENTILES)
    auc = metrics["auc"]

    rows = []
    print("\n" + "=" * 110)
    print(f"{'Model':24s} | {'Pct':>4s} | {'Prec':>6s} | {'Recall':>6s} | {'F1':>6s} | "
          f"{'FAR':>6s} | {'AUC':>6s} | {'ms/win':>8s} | {'inf_mem MB':>10s} | {'cpu/1k win s':>12s}")
    print("=" * 110)

    for pct in config.PERCENTILES:
        m = metrics[pct]
        row = {
            "mode": mode_label,
            "model": "LSTM (next-call NLL)",
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "epochs": epochs,
            "params": n_params,
            "size_kb": round(size_kb, 2),        # measured via BytesIO
            "train_s": round(train_time, 3),
            "pct": pct,
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "far": m["far"],
            "auc": auc,
            "ms_per_window": ms_per_window,
            "inference_mem_mb": inference_mem_mb,  # delta RSS during scoring
            "rss_before_mb": rss_before_mb,
            "cpu_time_per_1000_windows_s": cpu_per_1000,
            "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"]
        }
        rows.append(row)
        print(f"  {'LSTM (next-call NLL)':24s} | {pct:4.0f} | {m['precision']:6.3f} | {m['recall']:6.3f} | "
              f"{m['f1']:6.3f} | {m['far']:6.3f} | {auc:6.3f} | {ms_per_window:8.4f} | "
              f"{inference_mem_mb:10.2f} | {cpu_per_1000:12.6f}")

    # -------------------------------------------------------------
    # Save CSV and Summary
    # -------------------------------------------------------------
    csv_file = out_dir / "lstm_metrics.csv"
    with open(csv_file, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_md = out_dir / "summary.md"
    with open(summary_md, "w") as fh:
        fh.write(f"# Phase 5: LSTM Next-Call Baseline Summary\n\n")
        fh.write(f"mode: {mode_label}\n\n")
        fh.write("### Model Architecture\n")
        fh.write(f"- **Embedding**: Size {embed_dim} for {vocab_size} vocabulary tokens\n")
        fh.write(f"- **Recurrent Layer**: 1-layer LSTM with {hidden_dim} units\n")
        fh.write(f"- **Classifier Head**: Linear projection from {hidden_dim} to {vocab_size} logits\n")
        fh.write(f"- **Total Parameters**: {n_params:,} | Serialized size: {size_kb:.1f} KB (BytesIO)\n")
        fh.write(f"- **Training Time**: {train_time:.1f} s ({epochs} epochs)\n")
        fh.write(f"- **Threading**: torch.set_num_threads(1) + OMP/MKL=1.\n\n")
        fh.write("### Anomaly Scoring Mechanism\n")
        fh.write("For each sliding window $w = (c_0, c_1, \\dots, c_{L-1})$, the model predicts $P(c_t \\mid c_{<t})$.\n")
        fh.write("The window anomaly score is the mean Negative Log-Likelihood (NLL):\n")
        fh.write("$$\\text{Score}(w) = -\\frac{1}{L-1} \\sum_{t=1}^{L-1} \\log P(c_t \\mid c_{<t})$$\n")
        fh.write("Trace scores are the mean of window scores.\n\n")
        fh.write("### Detection Performance\n\n")
        fh.write("| Model | Percentile | Precision | Recall | F1 Score | FAR | ROC AUC | ms/win | inf_mem MB | cpu/1k win s |\n")
        fh.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in rows:
            fh.write(f"| {r['model']} | {r['pct']:.0f}% | {r['precision']:.3f} | {r['recall']:.3f} | "
                     f"{r['f1']:.3f} | {r['far']:.3f} | {r['auc']:.3f} | {r['ms_per_window']:.4f} | "
                     f"{r['inference_mem_mb']:.2f} | {r['cpu_time_per_1000_windows_s']:.6f} |\n")

    written_files = [str(csv_file), str(summary_md)]
    print("\n" + "=" * 80)
    print(f"PHASE 5 COMPLETE  [mode: {mode_label}]")
    print("Files written:")
    for wf in written_files:
        print(f"  - {wf}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    run_phase5(demo=args.demo)
