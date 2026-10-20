"""
Phase 2: Preprocessing and Feature Extraction
Extracts sliding windows, constructs vocabulary exclusively from normal training data,
computes unigram histograms, hashed bigrams, and hashed trigrams, and saves preprocessed
feature matrices and statistics for downstream detector phases.
"""
import sys, os, csv, json, time
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import common

def run_phase2(demo=False):
    common.print_banner(
        phase_num=2,
        title="Preprocessing and Feature Extraction",
        analogy="Translating a long stream of raw phone calls into short sentences and counting common word pairs and triplets so an anomaly detector can spot weird phrasing.",
        inputs="ADFA-LD raw traces (verified in Phase 1) with L=20, S=10",
        outputs="results/phase2/features_summary.csv, features_baseline.npz, features_trigrams.npz, summary.md"
    )

    out_dir = common.ensure_dir(config.RESULTS_DIR / "phase2")
    
    # -------------------------------------------------------------
    # Sanity Check 1: Verify Phase 1 completion
    # -------------------------------------------------------------
    print("[Sanity Check 1/2] Verifying Phase 1 outputs exist...")
    phase1_key = config.PROJECT_ROOT / config.PHASE_REGISTRY[1]["key_file"]
    if not phase1_key.is_file():
        print(f"[FATAL ERROR] Phase 1 output not found at {phase1_key}. Run Phase 1 first.")
        sys.exit(1)
    print("  [PASSED] Phase 1 dataset statistics verified.")

    # -------------------------------------------------------------
    # Step 1: Load traces and apply standard protocol split
    # -------------------------------------------------------------
    print("\n[Step 1/5] Loading traces and applying protocol split (Seed 0)...")
    d = common.find_dirs(config.DATA_DIR)
    train_traces = common.read_traces(d["train"])
    val_traces = common.read_traces(d["val"])
    attack_traces = common.read_traces(d["attack"])

    seed = config.DEMO_SEEDS[0] if demo else config.DEFAULT_SEEDS[0]
    calib_traces, test_traces, y_test = common.split_adfa_ld(val_traces, attack_traces, seed=seed)
    
    print(f"  Training traces    : {len(train_traces)} (Normal only)")
    print(f"  Calibration traces : {len(calib_traces)} (Normal validation remainder)")
    print(f"  Test traces        : {len(test_traces)} (746 attack + {len(test_traces)-len(attack_traces)} normal validation)")

    # -------------------------------------------------------------
    # Step 2: Vocabulary construction (TRAINING DATA ONLY)
    # -------------------------------------------------------------
    print("\n[Step 2/5] Building vocabulary strictly from normal training traces...")
    vocab, V = common.build_vocabulary(train_traces)
    print(f"  Vocabulary size V  : {V} unique system call IDs")
    print(f"  Unknown call index : {V} (reserved for unseen syscalls during calibration/test)")
    
    # Sanity Check 2: Vocabulary validation
    print("\n[Sanity Check 2/2] Validating vocabulary boundaries...")
    if V <= 0:
        print("[FATAL ERROR] Vocabulary is empty!")
        sys.exit(1)
    if V != 150:
        print(f"  [NOTE] Observed vocabulary size V={V} (published ADFA-LD benchmark is 150)")
    else:
        print(f"  [PASSED] Exact ADFA-LD vocabulary size V={V} confirmed.")

    # -------------------------------------------------------------
    # Step 3: Window generation (L=20, S=10)
    # -------------------------------------------------------------
    L = config.DEFAULT_L
    S = config.DEFAULT_S
    print(f"\n[Step 3/5] Slicing traces into sliding windows (L={L}, S={S})...")
    
    t0 = time.perf_counter()
    train_wins, train_owner = common.trace_windows(train_traces, L, S)
    calib_wins, calib_owner = common.trace_windows(calib_traces, L, S)
    test_wins, test_owner = common.trace_windows(test_traces, L, S)
    win_time = time.perf_counter() - t0

    print(f"  Training windows   : {len(train_wins):6d} (from {len(train_traces)} traces)")
    print(f"  Calibration windows: {len(calib_wins):6d} (from {len(calib_traces)} traces)")
    print(f"  Test windows       : {len(test_wins):6d} (from {len(test_traces)} traces)")
    print(f"  Windowing elapsed  : {win_time:.3f} s")

    # Sample windows display
    print("\n  Sample extracted windows (first 3 training windows):")
    for i in range(min(3, len(train_wins))):
        print(f"    Window {i} (len {len(train_wins[i])}): {train_wins[i]}")

    # -------------------------------------------------------------
    # Step 4: Featurization - Baseline (Unigram + Hashed Bigram)
    # -------------------------------------------------------------
    B_bi = config.DEFAULT_B_BI
    B_tri = config.DEFAULT_B_TRI
    
    print(f"\n[Step 4/5] Extracting Baseline Features (Histogram + Hashed Bigrams, B={B_bi})...")
    t0 = time.perf_counter()
    Xtr_base = common.featurize(train_wins, vocab, V, B=B_bi, use_trigrams=False, normalize=True)
    Xc_base = common.featurize(calib_wins, vocab, V, B=B_bi, use_trigrams=False, normalize=True)
    Xe_base = common.featurize(test_wins, vocab, V, B=B_bi, use_trigrams=False, normalize=True)
    base_feat_time = time.perf_counter() - t0
    
    base_D = Xtr_base.shape[1]
    base_sparsity = float(np.mean(Xtr_base == 0.0) * 100.0)
    print(f"  Baseline dimension : {base_D} (V+1={V+1} unigrams + {B_bi} bigram bins)")
    print(f"  Training shape     : {Xtr_base.shape}")
    print(f"  Zero sparsity      : {base_sparsity:.2f}% zero entries")
    print(f"  Elapsed time       : {base_feat_time:.3f} s")

    # -------------------------------------------------------------
    # Step 5: Featurization - Richer Features (+ Hashed Trigrams)
    # -------------------------------------------------------------
    print(f"\n[Step 5/5] Extracting Order-Aware Features (+ Hashed Trigrams, B_tri={B_tri})...")
    t0 = time.perf_counter()
    Xtr_tri = common.featurize(train_wins, vocab, V, B=B_bi, B_tri=B_tri, use_trigrams=True, normalize=True)
    Xc_tri = common.featurize(calib_wins, vocab, V, B=B_bi, B_tri=B_tri, use_trigrams=True, normalize=True)
    Xe_tri = common.featurize(test_wins, vocab, V, B=B_bi, B_tri=B_tri, use_trigrams=True, normalize=True)
    tri_feat_time = time.perf_counter() - t0
    
    tri_D = Xtr_tri.shape[1]
    tri_sparsity = float(np.mean(Xtr_tri == 0.0) * 100.0)
    print(f"  Trigram dimension  : {tri_D} (V+1={V+1} + {B_bi} bigrams + {B_tri} trigrams)")
    print(f"  Training shape     : {Xtr_tri.shape}")
    print(f"  Zero sparsity      : {tri_sparsity:.2f}% zero entries")
    print(f"  Elapsed time       : {tri_feat_time:.3f} s")

    # -------------------------------------------------------------
    # Save Preprocessed Matrices and Summary
    # -------------------------------------------------------------
    print("\n[Saving] Writing preprocessed feature arrays (.npz) and summary...")
    
    # Save baseline features
    base_npz = out_dir / "features_baseline.npz"
    np.savez_compressed(
        base_npz,
        Xtr=Xtr_base, Xc=Xc_base, Xe=Xe_base,
        y=y_test,
        calib_owner=calib_owner, test_owner=test_owner,
        V=V, D=base_D, L=L, S=S
    )
    
    # Save trigram features
    tri_npz = out_dir / "features_trigrams.npz"
    np.savez_compressed(
        tri_npz,
        Xtr=Xtr_tri, Xc=Xc_tri, Xe=Xe_tri,
        y=y_test,
        calib_owner=calib_owner, test_owner=test_owner,
        V=V, D=tri_D, L=L, S=S
    )
    
    # Save features summary CSV
    summary_csv = out_dir / "features_summary.csv"
    summary_rows = [
        {
            "feature_set": "Baseline (Histogram + Bigrams)",
            "dimension": base_D,
            "train_windows": len(Xtr_base),
            "calib_windows": len(Xc_base),
            "test_windows": len(Xe_base),
            "zero_sparsity_pct": f"{base_sparsity:.2f}",
            "extract_time_s": f"{base_feat_time:.3f}"
        },
        {
            "feature_set": "Richer (Histogram + Bigrams + Trigrams)",
            "dimension": tri_D,
            "train_windows": len(Xtr_tri),
            "calib_windows": len(Xc_tri),
            "test_windows": len(Xe_tri),
            "zero_sparsity_pct": f"{tri_sparsity:.2f}",
            "extract_time_s": f"{tri_feat_time:.3f}"
        }
    ]
    with open(summary_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # Save Markdown Summary
    summary_md = out_dir / "summary.md"
    with open(summary_md, "w") as fh:
        fh.write("# Phase 2: Preprocessing and Features Summary\n\n")
        fh.write("### Windowing & Vocabulary Configuration\n")
        fh.write(f"- **Window Length (L)**: {L}\n")
        fh.write(f"- **Window Stride (S)**: {S}\n")
        fh.write(f"- **Vocabulary Size (V)**: {V} unique system calls (derived strictly from normal training data)\n")
        fh.write(f"- **Unknown Call Index**: {V} (index 150)\n\n")
        fh.write("### Dataset Window Counts\n")
        fh.write(f"- **Training Windows**: {len(train_wins):,}\n")
        fh.write(f"- **Calibration Windows**: {len(calib_wins):,}\n")
        fh.write(f"- **Test Windows**: {len(test_wins):,} ({len(attack_traces)} attack traces + {len(test_traces)-len(attack_traces)} normal validation traces)\n\n")
        fh.write("### Feature Representation Comparison\n\n")
        fh.write("| Feature Set | Dimension $D$ | Train Windows | Zero Sparsity | Extraction Time |\n")
        fh.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for r in summary_rows:
            fh.write(f"| {r['feature_set']} | {r['dimension']} | {r['train_windows']} | {r['zero_sparsity_pct']}% | {r['extract_time_s']} s |\n")
        fh.write("\n### Output Artifacts for Downstream Detectors\n")
        fh.write(f"- `features_baseline.npz`: Baseline features ($D={base_D}$) ready for Phase 3 and Phase 4.\n")
        fh.write(f"- `features_trigrams.npz`: Richer order-aware features ($D={tri_D}$) ready for comparative evaluation.\n")

    written_files = [str(summary_csv), str(base_npz), str(tri_npz), str(summary_md)]
    print("\n" + "=" * 80)
    print("PHASE 2 COMPLETE")
    print("Files written:")
    for wf in written_files:
        print(f"  - {wf}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    run_phase2(demo=args.demo)
