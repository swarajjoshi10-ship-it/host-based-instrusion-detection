# Phase 5: LSTM Next-Call Baseline Summary

mode: full

### Model Architecture
- **Embedding**: Size 32 for 151 vocabulary tokens
- **Recurrent Layer**: 1-layer LSTM with 64 units
- **Classifier Head**: Linear projection from 64 to 151 logits
- **Total Parameters**: 39,735 | Serialized size: 158.0 KB (BytesIO)
- **Training Time**: 26.1 s (8 epochs)
- **Threading**: torch.set_num_threads(1) + OMP/MKL=1.

### Anomaly Scoring Mechanism
For each sliding window $w = (c_0, c_1, \dots, c_{L-1})$, the model predicts $P(c_t \mid c_{<t})$.
The window anomaly score is the mean Negative Log-Likelihood (NLL):
$$\text{Score}(w) = -\frac{1}{L-1} \sum_{t=1}^{L-1} \log P(c_t \mid c_{<t})$$
Trace scores are the mean of window scores.

### Detection Performance

| Model | Percentile | Precision | Recall | F1 Score | FAR | ROC AUC | ms/win | inf_mem MB | cpu/1k win s |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| LSTM (next-call NLL) | 99% | 0.960 | 0.127 | 0.225 | 0.005 | 0.835 | 0.0331 | 1.41 | 0.033103 |
| LSTM (next-call NLL) | 95% | 0.887 | 0.275 | 0.420 | 0.035 | 0.835 | 0.0331 | 1.41 | 0.033103 |
