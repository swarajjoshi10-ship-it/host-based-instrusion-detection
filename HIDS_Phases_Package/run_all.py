import argparse, random, time, csv
import numpy as np, psutil
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import hids_m1_m2 as H

def build(name, seed, epochs):
    if name == "Isolation Forest": return IsolationForest(n_estimators=100, random_state=seed, n_jobs=-1)
    if name == "One-Class SVM": return OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    return MLPRegressor(hidden_layer_sizes=(64, 16, 64), activation="relu", solver="adam", learning_rate_init=1e-3,
                        batch_size=128, max_iter=epochs, early_stopping=True, validation_fraction=0.1,
                        n_iter_no_change=5, random_state=seed)
def win_scores(name, model, X):
    if name.startswith("Autoencoder"): return ((X - model.predict(X)) ** 2).mean(axis=1)
    return -model.decision_function(X)
NAMES = ["Isolation Forest", "One-Class SVM", "Autoencoder (proposed)"]

def run_seed(train_t, val_t, att_t, a, seed):
    rng = random.Random(seed); npr = np.random.RandomState(seed)
    idx = list(range(len(val_t))); rng.shuffle(idx)
    n_test = min(len(att_t), len(val_t) // 2)
    test_norm = [val_t[i] for i in idx[:n_test]]; calib = [val_t[i] for i in idx[n_test:]]
    vocab = {c: k for k, c in enumerate(sorted({c for t in train_t for c in t}))}; V = len(vocab)
    tw, _ = H.trace_windows(train_t, a.L, a.S); Xtr = H.featurize(tw, vocab, V, a.B)
    cw, c_own = H.trace_windows(calib, a.L, a.S); Xc = H.featurize(cw, vocab, V, a.B)
    test = test_norm + att_t; y = np.array([0] * len(test_norm) + [1] * len(att_t))
    ew, e_own = H.trace_windows(test, a.L, a.S); Xe = H.featurize(ew, vocab, V, a.B)
    rows = []
    for name in NAMES:
        m = build(name, seed, a.epochs); fit_X = Xtr
        if name == "One-Class SVM" and len(Xtr) > a.svm_max: fit_X = Xtr[npr.choice(len(Xtr), a.svm_max, replace=False)]
        t0 = time.perf_counter()
        (m.fit(fit_X, fit_X) if name.startswith("Autoencoder") else m.fit(fit_X))
        train_s = time.perf_counter() - t0
        cs = H.trace_scores(win_scores(name, m, Xc), c_own, len(calib))
        proc = psutil.Process(); proc.cpu_percent(None)
        t0 = time.perf_counter(); es = win_scores(name, m, Xe); inf_s = time.perf_counter() - t0
        cpu = proc.cpu_percent(None); rss = proc.memory_info().rss / 1e6
        ts = H.trace_scores(es, e_own, len(test)); auc = roc_auc_score(y, ts)
        params = (sum(c.size for c in m.coefs_) + sum(i.size for i in m.intercepts_)) if name.startswith("Autoencoder") else 0
        for pct in (99, 95):
            thr = np.percentile(cs, pct); pred = (ts > thr).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            rows.append(dict(model=name, seed=seed, pct=pct, precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred),
                f1=f1_score(y, pred), far=fp / max(fp + tn, 1), auc=auc, ms_per_window=1000 * inf_s / len(Xe), train_s=train_s,
                cpu_pct=cpu, rss_mb=rss, params=params, model_kb=params * 8 / 1024, tp=tp, fp=fp, tn=tn, fn=fn))
    return rows

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--L", type=int, default=20); ap.add_argument("--S", type=int, default=10)
ap.add_argument("--B", type=int, default=256); ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--svm_max", type=int, default=5000)
ap.add_argument("--epochs", type=int, default=50); ap.add_argument("--out", default="results_all.csv")
a = ap.parse_args()
d = H.find_dirs(a.data)
train_t, val_t, att_t = H.read_traces(d["train"]), H.read_traces(d["val"]), H.read_traces(d["attack"])
print(f"[M1] traces: train {len(train_t)}, validation {len(val_t)}, attack {len(att_t)} | L={a.L}, S={a.S}")
allrows = []
for s in range(a.seeds):
    allrows += run_seed(train_t, val_t, att_t, a, s); print(f"  seed {s} done")
with open(a.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)
for pct in (99, 95):
    print(f"\n[threshold = {pct}th percentile of normal calibration scores -> about {100 - pct}% false alarms allowed]  mean +/- std over seeds")
    print(f"{'Detector':24s} {'Prec':>13s} {'Recall':>13s} {'F1':>13s} {'FAR':>13s} {'AUC':>13s} {'ms/win':>8s}")
    for n in NAMES:
        R = [r for r in allrows if r['model'] == n and r['pct'] == pct]
        f = lambda k: f"{np.mean([r[k] for r in R]):.3f}+/-{np.std([r[k] for r in R]):.3f}"
        print(f"{n:24s} {f('precision'):>13s} {f('recall'):>13s} {f('f1'):>13s} {f('far'):>13s} {f('auc'):>13s} {np.mean([r['ms_per_window'] for r in R]):8.4f}")
ae = [r for r in allrows if r['model'].startswith("Autoencoder") and r['pct'] == 99]
print(f"\nAutoencoder size: {ae[0]['params']} parameters (~{ae[0]['model_kb']:.0f} KB as float64) | train time {np.mean([r['train_s'] for r in ae]):.1f}s")
print(f"Saved per-seed details to {a.out}")