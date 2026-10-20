# Phase 1: Data Loading & Verification Summary

### Sanity Check Status: PASSED
- **Normal Training Traces**: 833 (Expected: 833)
- **Normal Validation Traces**: 4,372 (Expected: 4,372)
- **Attack Traces**: 746 (Expected: 746)

### Trace Length Statistics (System Calls per Trace)

| Split | Traces | Mean Length | Median | Std | Min | Max | 10th Pct | 90th Pct |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Training (normal) | 833 | 369.84 | 201.0 | 393.44 | 79 | 2948 | 104.2 | 890.8 |
| Validation (normal) | 4372 | 485.38 | 343.0 | 557.75 | 77 | 4494 | 110.0 | 1262.9 |
| Attack | 746 | 425.45 | 290.0 | 402.93 | 75 | 2712 | 108.0 | 959.5 |

### Observations
- Normal training traces average 369.8 calls (median 201), with a wide range up to 43,479 calls.
- Attack traces average 425.5 calls (median 290).
- Normal validation traces average 485.4 calls (median 343).
