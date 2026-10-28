"""
Recompute a saved model's thresholds from normal calibration runs, without retraining (the weights are unchanged).

  python3 live/recalibrate.py --model live/models/host_ebpf.npz --calib live/recordings_ebpf/normal_calib
"""
import argparse
from model import Detector, calibrate, read_runs, save

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--calib", nargs="+", required=True)
a = ap.parse_args()
det = Detector.load(a.model)
old = dict(det.thresholds)
calibrate(det, list(read_runs(a.calib).values()))
for k in sorted(det.thresholds):
    print(f"  {k:10s} {old.get(k, float('nan')):.4f} -> {det.thresholds[k]:.4f}")
save(a.model, det.arrays, det.meta)
print(f"[saved] {a.model}")
