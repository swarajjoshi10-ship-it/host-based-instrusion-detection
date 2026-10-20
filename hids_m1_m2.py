"""
Host-Based Intrusion Detection Using System Call Analysis
Modules M1 (data loading + preprocessing) and M2 (baselines: Isolation Forest, One-Class SVM)
plus the metric part of M4, for ADFA-LD.

Run (PowerShell):
    pip install numpy scikit-learn psutil
    python hids_m1_m2.py --data "C:\\path\\to\\ADFA-LD" --seeds 5

--data must contain the three folders: Training_Data_Master, Validation_Data_Master, Attack_Data_Master
(folder names are matched loosely: anything containing 'train', 'valid', 'attack').
"""
import argparse, os, time, csv, random
import numpy as np
import psutil
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# ---------------- M1: loading ----------------
def find_dirs(root):
    found = {}
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if not os.path.isdir(p): continue
        low = name.lower()
        if "train" in low: found["train"] = p
        elif "valid" in low: found["val"] = p
        elif "attack" in low: found["attack"] = p
    missing = {"train", "val", "attack"} - set(found)
    if missing: raise SystemExit(f"Could not find folders for: {missing} inside {root}")
    return found

def read_traces(folder):
    traces = []
    for dp, _, files in os.walk(folder):
        for f in sorted(files):
            if not f.lower().endswith(".txt"): continue
            with open(os.path.join(dp, f), "r", errors="ignore") as fh:
                toks = fh.read().split()
            seq = [int(t) for t in toks if t.isdigit()]
            if seq: traces.append(seq)
    return traces

def windows(seq, L, S):
    if len(seq) <= L: return [seq]
    return [seq[i:i + L] for i in range(0, len(seq) - L + 1, S)]

def featurize(wins, vocab, V, B):
    """call-frequency histogram (V+1, last = unknown) + hashed bigram counts (B bins), each L1-normalised."""
    X = np.zeros((len(wins), V + 1 + B), dtype=np.float32)
    for r, w in enumerate(wins):
        ids = [vocab.get(c, V) for c in w]
        for c in ids: X[r, c] += 1
        for a, b in zip(ids[:-1], ids[1:]): X[r, V + 1 + ((a * 31 + b) % B)] += 1
        s1 = X[r, :V + 1].sum(); s2 = X[r, V + 1:].sum()
        if s1: X[r, :V + 1] /= s1
        if s2: X[r, V + 1:] /= s2
    return X

def trace_windows(traces, L, S):
    wins, owner = [], []
    for i, t in enumerate(traces):
        w = windows(t, L, S); wins += w; owner += [i] * len(w)
    return wins, np.array(owner)

def trace_scores(win_scores, owner, n):
    sums = np.zeros(n); cnt = np.zeros(n)
    for s, o in zip(win_scores, owner):
        sums[o] += s; cnt[o] += 1
    return sums / np.maximum(cnt, 1)

# ---------------- M2: baselines ----------------
def make_models(seed):
    return {
        "Isolation Forest": IsolationForest(n_estimators=100, random_state=seed, n_jobs=-1),
        "One-Class SVM": OneClassSVM(kernel="rbf", gamma="scale", nu=0.05),
    }

def run_once(train_t, val_t, att_t, args, seed):
    rng = random.Random(seed); np_rng = np.random.RandomState(seed)
    # split: test = all attack + equal number of normal validation traces; calibration = the rest
    idx = list(range(len(val_t))); rng.shuffle(idx)
    n_test_norm = min(len(att_t), len(val_t) // 2)
    test_norm = [val_t[i] for i in idx[:n_test_norm]]
    calib = [val_t[i] for i in idx[n_test_norm:]]
    # vocabulary from training data only
    vocab = {c: k for k, c in enumerate(sorted({c for t in train_t for c in t}))}; V = len(vocab)
    tw, _ = trace_windows(train_t, args.L, args.S)
    Xtr = featurize(tw, vocab, V, args.B)
    cw, c_own = trace_windows(calib, args.L, args.S); Xc = featurize(cw, vocab, V, args.B)
    test_traces = test_norm + att_t; y = np.array([0] * len(test_norm) + [1] * len(att_t))
    ew, e_own = trace_windows(test_traces, args.L, args.S); Xe = featurize(ew, vocab, V, args.B)
    rows = []
    for name, model in make_models(seed).items():
        fit_X = Xtr
        if name == "One-Class SVM" and len(Xtr) > args.svm_max:
            fit_X = Xtr[np_rng.choice(len(Xtr), args.svm_max, replace=False)]
        t0 = time.perf_counter(); model.fit(fit_X); train_s = time.perf_counter() - t0
        cs = trace_scores(-model.decision_function(Xc), c_own, len(calib))
        thr = np.percentile(cs, args.pct)
        proc = psutil.Process(); proc.cpu_percent(None)
        t0 = time.perf_counter(); es = -model.decision_function(Xe); inf_s = time.perf_counter() - t0
        cpu = proc.cpu_percent(None); rss = proc.memory_info().rss / 1e6
        ts = trace_scores(es, e_own, len(test_traces)); pred = (ts > thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append(dict(model=name, seed=seed, precision=precision_score(y, pred, zero_division=0),
            recall=recall_score(y, pred), f1=f1_score(y, pred), far=fp / max(fp + tn, 1), auc=roc_auc_score(y, ts),
            ms_per_window=1000 * inf_s / len(Xe), train_s=train_s, cpu_pct=cpu, rss_mb=rss, tp=tp, fp=fp, tn=tn, fn=fn))
    return rows, dict(V=V, train_windows=len(Xtr), calib_traces=len(calib), test_normal=len(test_norm), test_attack=len(att_t), D=Xtr.shape[1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--L", type=int, default=20); ap.add_argument("--S", type=int, default=10)
    ap.add_argument("--B", type=int, default=256); ap.add_argument("--pct", type=float, default=99.0)
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--svm_max", type=int, default=5000)
    ap.add_argument("--out", default="results_adfa_ld.csv")
    args = ap.parse_args()
    d = find_dirs(args.data)
    train_t, val_t, att_t = read_traces(d["train"]), read_traces(d["val"]), read_traces(d["attack"])
    print(f"[M1] traces loaded -> train(normal): {len(train_t)}, validation(normal): {len(val_t)}, attack: {len(att_t)}")
    print("     (published ADFA-LD counts: 833 / 4373 / 746)")
    allrows, info = [], None
    for s in range(args.seeds):
        rows, info = run_once(train_t, val_t, att_t, args, s); allrows += rows; print(f"  seed {s} done")
    print(f"[M1] vocabulary V={info['V']}, feature size D={info['D']}, training windows={info['train_windows']}, calibration traces={info['calib_traces']}, test normal/attack={info['test_normal']}/{info['test_attack']}")
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)
    print("\n[M2 + M4 metrics] mean +/- std over seeds (trace-level, ADFA-LD)")
    print(f"{'Detector':18s} {'Prec':>13s} {'Recall':>13s} {'F1':>13s} {'FAR':>13s} {'AUC':>13s} {'ms/window':>10s}")
    for name in sorted({r['model'] for r in allrows}):
        R = [r for r in allrows if r['model'] == name]
        f = lambda k: f"{np.mean([r[k] for r in R]):.3f}+/-{np.std([r[k] for r in R]):.3f}"
        print(f"{name:18s} {f('precision'):>13s} {f('recall'):>13s} {f('f1'):>13s} {f('far'):>13s} {f('auc'):>13s} {np.mean([r['ms_per_window'] for r in R]):10.4f}")
    print(f"\nPer-seed details saved to {args.out}")
if __name__ == "__main__":
    main()
