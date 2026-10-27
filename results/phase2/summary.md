# Phase 2: Preprocessing and Features Summary

### Windowing & Vocabulary Configuration
- **Window Length (L)**: 20
- **Window Stride (S)**: 10
- **Vocabulary Size (V)**: 150 unique system calls (derived strictly from normal training data)
- **Unknown Call Index**: 150 (index 150)

### Dataset Window Counts
- **Training Windows**: 29,605
- **Calibration Windows**: 172,010
- **Test Windows**: 64,665 (746 attack traces + 746 normal validation traces)

### Feature Representation Comparison

| Feature Set | Dimension $D$ | Train Windows | Zero Sparsity | Extraction Time |
| :--- | :---: | :---: | :---: | :---: |
| Baseline (Histogram + Bigrams) | 407 | 29605 | 96.67% | 5.628 s |
| Richer (Histogram + Bigrams + Trigrams) | 663 | 29605 | 96.49% | 10.058 s |

### Output Artifacts for Downstream Detectors
- `features_baseline.npz`: Baseline features ($D=407$) ready for Phase 3 and Phase 4.
- `features_trigrams.npz`: Richer order-aware features ($D=663$) ready for comparative evaluation.
