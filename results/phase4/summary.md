# Phase 4: Proposed Lightweight Autoencoder Summary

mode: full

### Model Architecture
- **Structure**: Fully connected bottleneck autoencoder ($D \to 64 \to 16 \to 64 \to D$)
- **Activation**: ReLU hidden units, Linear output
- **Optimization**: Adam (lr=1e-3, batch 128, early stopping)
- **Size**: Measured via `len(pickle.dumps(model))` — no hardcoded values.
- **Threading**: Single-threaded (OMP/MKL=1).

### Detection Metrics & Resource Footprint

| Feature Set | Parameters | Size (KB, pickle) | Train Time | Pct | Precision | Recall | F1 | FAR | ROC AUC | ms/win | inf_mem MB | cpu/1k win s |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (Bigrams, D=407) | 54,695 | 649.4 | 50.2 s | 99% | 0.286 | 0.003 | 0.005 | 0.007 | 0.750 | 0.0052 | 0.05 | 0.012323 |
| Baseline (Bigrams, D=407) | 54,695 | 649.4 | 50.2 s | 95% | 0.667 | 0.080 | 0.144 | 0.040 | 0.750 | 0.0052 | 0.05 | 0.012323 |
| Richer (Bigrams+Trigrams, D=663) | 87,719 | 1036.5 | 73.8 s | 99% | 0.500 | 0.004 | 0.008 | 0.004 | 0.764 | 0.0057 | 2.99 | 0.016189 |
| Richer (Bigrams+Trigrams, D=663) | 87,719 | 1036.5 | 73.8 s | 95% | 0.615 | 0.075 | 0.134 | 0.047 | 0.764 | 0.0057 | 2.99 | 0.016189 |
