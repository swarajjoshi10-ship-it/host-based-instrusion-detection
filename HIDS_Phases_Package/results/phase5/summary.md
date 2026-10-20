# Phase 5: LSTM Next-Call Baseline Summary

### Model Architecture
- **Embedding**: Size 32 for 151 vocabulary tokens
- **Recurrent Layer**: 1-layer LSTM with 64 units
- **Classifier Head**: Linear projection from 64 to 151 logits
- **Total Parameters**: 39,735 (~155.2 KB as float32)
- **Training Time**: 17.6 s (8 epochs)

### Anomaly Scoring Mechanism
For each sliding window $w = (c_0, c_1, \dots, c_{L-1})$, the model predicts $P(c_t \mid c_{<t})$.
The window anomaly score is the mean Negative Log-Likelihood (NLL):
$$\text{Score}(w) = -\frac{1}{L-1} \sum_{t=1}^{L-1} \log P(c_t \mid c_{<t})$$
Trace scores are the mean of window scores.

### Detection Performance

| Model | Percentile | Precision | Recall | F1 Score | FAR | ROC AUC | Latency (ms/win) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| LSTM (next-call NLL) | 99% | 0.970 | 0.131 | 0.231 | 0.004 | 0.834 | 0.0221 |
| LSTM (next-call NLL) | 95% | 0.894 | 0.282 | 0.428 | 0.034 | 0.834 | 0.0221 |

### Comparative Findings
- The LSTM model achieves an ROC AUC of **0.834**.
- While the LSTM captures exact sequence order, sequential recurrence increases latency to **0.0221 ms/window** compared to **0.0037 ms/window** for the proposed Autoencoder.
- The Autoencoder delivers competitive/higher AUC with ~10x lower inference latency, reinforcing its suitability for lightweight host deployment.
