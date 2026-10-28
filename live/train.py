"""
Train the live detector on folders of NORMAL traces (one .txt per process, space-separated syscall numbers).

  Host model (traces recorded on this machine by monitor.py --record, x86_64 numbers):
      python3 live/train.py --traces live/recordings/normal_train --calib live/recordings/normal_calib \
              --numbering x86_64 --ignore rt_sigreturn --out live/models/host.npz
  ADFA-LD model (the offline dataset, i386 numbers):
      python3 live/train.py --traces Training_Data_Master --calib Validation_Data_Master --numbering i386 --out live/models/adfa.npz

Without --calib, a random 20% of the runs is held out to set the thresholds (95th / 99th percentile of normal scores).
The saved .npz holds the weights, vocabulary and thresholds; detection needs only numpy.
--ignore rt_sigreturn drops the return from a signal handler: signals arrive asynchronously, so where that call lands
in a trace is timing noise (used for the host models; decided after seeing results, see README).
"""
import argparse, os, random
from model import fit, read_runs, save
from syscalls import X64, I386

ap = argparse.ArgumentParser()
ap.add_argument("--traces", nargs="+", required=True); ap.add_argument("--calib", nargs="*", default=[])
ap.add_argument("--numbering", required=True, choices=["x86_64", "i386"])
ap.add_argument("--out", required=True); ap.add_argument("--L", type=int, default=20)
ap.add_argument("--hidden", type=int, default=64); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--min_windows", type=int, default=5, help="windows a running task needs before it can be flagged")
ap.add_argument("--ignore", default="", help="comma-separated system call names dropped from every trace, e.g. rt_sigreturn")
a = ap.parse_args()

train = list(read_runs(a.traces).values())
if a.calib: calib = list(read_runs(a.calib).values())
else:
    rng = random.Random(a.seed); rng.shuffle(train); k = max(len(train) // 5, 1)
    calib, train = train[:k], train[k:]
table = X64 if a.numbering == "x86_64" else I386
ignore = [table[n] for n in a.ignore.split(",") if n]
if ignore: print(f"[train] ignoring {a.ignore} (numbers {ignore}) in every trace")
arrays, meta = fit(train, calib, a.numbering, L=a.L, hidden=a.hidden, seed=a.seed, min_windows=a.min_windows,
                   source=" ".join(a.traces), ignore=ignore)
save(a.out, arrays, meta)
print(f"[saved] {a.out}  ({meta['params']} parameters, {os.path.getsize(a.out) / 1024:.1f} KB, {meta['epochs']} epochs, {meta['train_s']} s)")
