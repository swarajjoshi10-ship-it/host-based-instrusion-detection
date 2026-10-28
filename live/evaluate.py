"""
Score recorded live runs with a saved detector and report run-level metrics.

  python3 live/evaluate.py --model live/models/host.npz --normal live/recordings/normal_test --attack live/recordings/attack

A run (one monitored execution, all its processes) is flagged when any of its processes would trigger the live alert
rule (running mean score above the threshold after min_windows windows, or at exit), exactly as monitor.py decides.
Recorded traces are in x86_64 numbering and are translated when the model uses another one (e.g. ADFA-LD's i386).
Writes per-run scores to <model>[.tag].scores.csv and its summary rows to live/results/live_eval.csv
(replacing earlier rows for the same model and tag).
"""
import argparse, csv, os
import numpy as np
from sklearn.metrics import roc_auc_score
from model import Detector, ROOT, read_runs, live_stat
from syscalls import convert

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--normal", nargs="+", required=True)
ap.add_argument("--attack", nargs="+", required=True); ap.add_argument("--tag", default="")
a = ap.parse_args()

det = Detector.load(a.model)
rows = []                                                     # run, label, score, worst-trace unknown rate, processes, calls
for label, folders in ((0, a.normal), (1, a.attack)):
    for run, traces in read_runs(folders).items():
        traces = [convert(t, det.numbering) for t in traces]
        sc = [live_stat(det.trace_scores(t), det.min_windows) for t in traces]; k = int(np.argmax(sc))
        rows.append((run, label, sc[k], float(np.mean([det.oov_rate(t) for t in traces])), len(traces), sum(map(len, traces))))
y = np.array([r[1] for r in rows]); sc = np.array([r[2] for r in rows])

print(f"model {os.path.basename(a.model)} (numbering {det.numbering}, trained on {det.meta['source']})")
print(f"runs: {int((y == 0).sum())} normal, {int(y.sum())} attack; mean processes per run {np.mean([r[4] for r in rows]):.1f}")
print(f"unknown-call rate: normal {np.mean([r[3] for r in rows if r[1] == 0]):.3f}, attack {np.mean([r[3] for r in rows if r[1] == 1]):.3f}")
auc = roc_auc_score(y, sc) if 0 < y.sum() < len(y) else float("nan")
print(f"AUC {auc:.3f}")
summary = []
for name in ("live_p95", "live_p99"):
    thr = det.thresholds[name]; pred = sc > thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1); f1 = 2 * p * r / max(p + r, 1e-12); far = fp / max(fp + tn, 1)
    print(f"{name:8s} thr {thr:.3f}: precision {p:.3f} recall {r:.3f} F1 {f1:.3f} false alarm rate {far:.3f}  (TP {tp} FP {fp} FN {fn} TN {tn})")
    summary.append(dict(model=os.path.basename(a.model), tag=a.tag, threshold=name, n_normal=int((y == 0).sum()), n_attack=int(y.sum()),
                        auc=round(auc, 4), precision=round(p, 4), recall=round(r, 4), f1=round(f1, 4), far=round(far, 4)))

base = os.path.splitext(a.model)[0] + (f".{a.tag}" if a.tag else "")
with open(base + ".scores.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["run", "label", "score", "unknown_rate", "processes", "calls"])
    for r in rows: w.writerow([os.path.relpath(r[0]), r[1], round(r[2], 4), round(r[3], 4), r[4], r[5]])
out = os.path.join(ROOT, "live", "results", "live_eval.csv"); os.makedirs(os.path.dirname(out), exist_ok=True)
old = list(csv.DictReader(open(out))) if os.path.exists(out) else []
old = [r for r in old if (r["model"], r["tag"]) != (summary[0]["model"], summary[0]["tag"])]   # a rerun replaces its rows
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(summary[0])); w.writeheader(); w.writerows(old + summary)
print(f"per-run scores -> {os.path.relpath(base)}.scores.csv ; summary written to {os.path.relpath(out)}")
