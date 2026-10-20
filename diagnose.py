import argparse, random
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
import hids_m1_m2 as H

def agg(scores, owner, n, how):
    cuts = np.flatnonzero(np.diff(owner)) + 1
    groups = np.split(np.asarray(scores), cuts)
    out = []
    for g in groups:
        if how == "max": out.append(g.max())
        elif how == "mean": out.append(g.mean())
        else: out.append(np.sort(g)[-3:].mean())
    return np.array(out)

ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); a = ap.parse_args()
d = H.find_dirs(a.data)
train_t, val_t, att_t = H.read_traces(d["train"]), H.read_traces(d["val"]), H.read_traces(d["attack"])
def stats(name, T):
    L = np.array([len(t) for t in T])
    print(f"{name:18s} n={len(T):5d}  length mean={L.mean():8.1f}  median={np.median(L):7.1f}  p10={np.percentile(L,10):7.1f}  p90={np.percentile(L,90):8.1f}")
print("[trace lengths in system calls]")
stats("train (normal)", train_t); stats("validation (normal)", val_t); stats("attack", att_t)
random.seed(0); idx = list(range(len(val_t))); random.shuffle(idx)
test_norm = [val_t[i] for i in idx[:len(att_t)]]; test = test_norm + att_t
y = np.array([0]*len(test_norm) + [1]*len(att_t))
vocab = {c: k for k, c in enumerate(sorted({c for t in train_t for c in t}))}; V = len(vocab)
print("\n[Isolation Forest AUC by window length L and trace-score rule]  (0.5 = coin flip)")
print(f"{'L':>5s} {'max':>8s} {'mean':>8s} {'top3-mean':>10s}")
for L in (20, 50, 100):
    S = max(L // 2, 1)
    tw, _ = H.trace_windows(train_t, L, S); Xtr = H.featurize(tw, vocab, V, 256)
    m = IsolationForest(n_estimators=100, random_state=0, n_jobs=-1).fit(Xtr)
    ew, own = H.trace_windows(test, L, S); Xe = H.featurize(ew, vocab, V, 256)
    s = -m.decision_function(Xe)
    print(f"{L:5d} " + " ".join(f"{roc_auc_score(y, agg(s, own, len(test), h)):8.3f}" for h in ("max", "mean")) + f" {roc_auc_score(y, agg(s, own, len(test), 'top3')):10.3f}")