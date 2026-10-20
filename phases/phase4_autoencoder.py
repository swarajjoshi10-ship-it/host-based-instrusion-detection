"""
Phase 4: Proposed Lightweight Autoencoder
Trains and evaluates a lightweight neural reconstruction autoencoder (MLPRegressor 64-16-64)
on normal system call behavior. Flags anomalies based on reconstruction Mean Squared Error (MSE).
Reports parameter counts, MEASURED serialized model size (pickle), training time,
inference ms/window, inference_mem_mb, cpu_time_per_1000_windows_s, and
detection metrics at 99th and 95th percentile calibration thresholds.

Threading is pinned to 1 via common.py env vars (OMP/MKL/NUMEXPR) before any numpy import.
"""
import sys, os, csv, time
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import common  # pins OMP/MKL before numpy

def build_autoencoder(seed, max_iter):
    return MLPRegressor(
        hidden_layer_sizes=(64, 16, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        batch_size=128,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=seed
    )

def compute_mse_scores(model, X):
    """Compute per-window reconstruction Mean Squared Error (MSE)."""
    preds = model.predict(X)
    return ((X - preds) ** 2).mean(axis=1)

def run_phase4(demo=False):
    mode_label = "demo" if demo else "full"

    common.print_banner(
        phase_num=4,
        title="Proposed Lightweight Autoencoder",
        analogy="Teaching a neural network to summarize and reconstruct normal system call patterns through an hourglass bottleneck; when an attack arrives, the network fails to reconstruct it, producing a high reconstruction error spike.",
        inputs="Preprocessed feature arrays from Phase 2 (features_baseline.npz, features_trigrams.npz)",
        outputs="results/phase4/autoencoder_metrics.csv, results/phase4/summary.md"
    )
    print(f"[Mode] {mode_label.upper()}")

    out_dir = common.ensure_dir(config.RESULTS_DIR / "phase4")

    # -------------------------------------------------------------
    # Sanity Check: Verify Phase 2 arrays exist
    # -------------------------------------------------------------
    print("[Sanity Check] Verifying Phase 2 preprocessed arrays exist...")
    p2_base = config.RESULTS_DIR / "phase2" / "features_baseline.npz"
    p2_tri = config.RESULTS_DIR / "phase2" / "features_trigrams.npz"
    if not p2_base.is_file() or not p2_tri.is_file():
        print(f"[FATAL ERROR] Phase 2 feature arrays missing. Run Phase 2 first.")
        sys.exit(1)
    print("  [PASSED] Preprocessed feature files verified.")

    data_base = np.load(p2_base)
    data_tri = np.load(p2_tri)

    seed = config.DEMO_SEEDS[0] if demo else config.DEFAULT_SEEDS[0]
    max_epochs = config.DEMO_AE_EPOCHS if demo else config.AE_EPOCHS

    feature_configs = [
        ("Baseline (Bigrams, D=407)", data_base),
        ("Richer (Bigrams+Trigrams, D=663)", data_tri)
    ]

    all_rows = []

    hdr = f"{'Features':34s} | {'Params':>7s} | {'Size(KB)':>8s} | {'Train(s)':>8s} | {'Pct':>4s} | {'Prec':>6s} | {'Recall':>6s} | {'F1':>6s} | {'FAR':>6s} | {'AUC':>6s} | {'ms/win':>8s} | {'inf_mem MB':>10s} | {'cpu/1k win s':>12s}"
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("=" * len(hdr))

    for feat_name, d in feature_configs:
        Xtr = d["Xtr"]
        Xc  = d["Xc"]
        Xe  = d["Xe"]
        y   = d["y"]
        c_own = d["calib_owner"]
        e_own = d["test_owner"]
        n_calib = len(np.unique(c_own))
        n_test  = len(y)

        # Build and train Autoencoder
        model = build_autoencoder(seed, max_epochs)
        print(f"  Training Autoencoder on {feat_name} ({len(Xtr)} windows, max_iter={max_epochs})...")
        t0 = time.perf_counter()
        model.fit(Xtr, Xtr)
        train_time = time.perf_counter() - t0
        print(f"    -> Trained {model.n_iter_} iteration(s) in {train_time:.1f} s")

        # Parameter count
        n_params = sum(c.size for c in model.coefs_) + sum(i.size for i in model.intercepts_)

        # Serialized size via pickle (measured, not estimated)
        size_kb = common.sklearn_serialized_kb(model)

        # Calibration inference (not profiled for resource; just scores needed)
        calib_win_scores = compute_mse_scores(model, Xc)
        calib_trace_scores = common.trace_scores(calib_win_scores, c_own, n_calib, method="mean")

        # Test inference — profiled via common.profile_inference
        prof = common.profile_inference(
            score_fn=lambda: compute_mse_scores(model, Xe),
            n_windows=len(Xe)
        )
        test_win_scores = prof["scores"]
        ms_per_window = (prof["wall_s"] / len(Xe)) * 1000.0
        inference_mem_mb = prof["inference_mem_mb"]
        cpu_per_1000 = prof["cpu_time_per_1000_windows_s"]
        rss_before_mb = prof["rss_before_mb"]

        # Test trace aggregation & metrics
        test_trace_scores = common.trace_scores(test_win_scores, e_own, n_test, method="mean")
        metrics = common.compute_metrics(y, test_trace_scores, calib_trace_scores, percentiles=config.PERCENTILES)
        auc = metrics["auc"]

        for pct in config.PERCENTILES:
            m = metrics[pct]
            row = {
                "mode": mode_label,
                "model": "Autoencoder (MLP 64-16-64)",
                "features": feat_name,
                "epochs_trained": model.n_iter_,
                "params": n_params,
                "size_kb": round(size_kb, 2),          # measured via pickle
                "train_s": round(train_time, 3),
                "pct": pct,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "far": m["far"],
                "auc": auc,
                "ms_per_window": ms_per_window,
                "inference_mem_mb": inference_mem_mb,   # delta RSS during scoring
                "rss_before_mb": rss_before_mb,         # process RSS before scoring
                "cpu_time_per_1000_windows_s": cpu_per_1000,
                "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"]
            }
            all_rows.append(row)
            print(f"  {feat_name:34s} | {n_params:7d} | {size_kb:8.1f} | {train_time:8.1f} | "
                  f"{pct:4.0f} | {m['precision']:6.3f} | {m['recall']:6.3f} | {m['f1']:6.3f} | "
                  f"{m['far']:6.3f} | {auc:6.3f} | {ms_per_window:8.4f} | "
                  f"{inference_mem_mb:10.2f} | {cpu_per_1000:12.6f}")

    # -------------------------------------------------------------
    # Save CSV and Summary
    # -------------------------------------------------------------
    csv_file = out_dir / "autoencoder_metrics.csv"
    with open(csv_file, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary_md = out_dir / "summary.md"
    with open(summary_md, "w") as fh:
        fh.write(f"# Phase 4: Proposed Lightweight Autoencoder Summary\n\n")
        fh.write(f"mode: {mode_label}\n\n")
        fh.write("### Model Architecture\n")
        fh.write("- **Structure**: Fully connected bottleneck autoencoder ($D \\to 64 \\to 16 \\to 64 \\to D$)\n")
        fh.write("- **Activation**: ReLU hidden units, Linear output\n")
        fh.write("- **Optimization**: Adam (lr=1e-3, batch 128, early stopping)\n")
        fh.write("- **Size**: Measured via `len(pickle.dumps(model))` — no hardcoded values.\n")
        fh.write("- **Threading**: Single-threaded (OMP/MKL=1).\n\n")
        fh.write("### Detection Metrics & Resource Footprint\n\n")
        fh.write("| Feature Set | Parameters | Size (KB, pickle) | Train Time | Pct | Precision | Recall | F1 | FAR | ROC AUC | ms/win | inf_mem MB | cpu/1k win s |\n")
        fh.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in all_rows:
            fh.write(f"| {r['features']} | {r['params']:,} | {r['size_kb']:.1f} | {r['train_s']:.1f} s | "
                     f"{r['pct']:.0f}% | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
                     f"{r['far']:.3f} | {r['auc']:.3f} | {r['ms_per_window']:.4f} | "
                     f"{r['inference_mem_mb']:.2f} | {r['cpu_time_per_1000_windows_s']:.6f} |\n")

    written_files = [str(csv_file), str(summary_md)]
    print("\n" + "=" * 80)
    print(f"PHASE 4 COMPLETE  [mode: {mode_label}]")
    print("Files written:")
    for wf in written_files:
        print(f"  - {wf}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    run_phase4(demo=args.demo)
