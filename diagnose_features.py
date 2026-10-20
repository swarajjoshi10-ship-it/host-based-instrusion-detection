import argparse, random, time
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import hids_m1_m2 as H

def eval_config(train_t, val_t, att_t, L, S, use_tri, norm, seed=0):
    rng = random.Random(seed)
    idx = list(range(len(val_t)))
    rng.shuffle(idx)
    n_test_norm = min(len(att_t), len(val_t) // 2)
    test_norm = [val_t[i] for i in idx[:n_test_norm]]
    calib = [val_t[i] for i in idx[n_test_norm:]]

    vocab = {c: k for k, c in enumerate(sorted({c for t in train_t for c in t}))}
    V = len(vocab)
    
    tw, _ = H.trace_windows(train_t, L, S)
    Xtr = H.featurize(tw, vocab, V, B=256, B_tri=256, use_trigrams=use_tri, normalize=norm)
    
    cw, c_own = H.trace_windows(calib, L, S)
    Xc = H.featurize(cw, vocab, V, B=256, B_tri=256, use_trigrams=use_tri, normalize=norm)
    
    test_traces = test_norm + att_t
    y = np.array([0] * len(test_norm) + [1] * len(att_t))
    ew, e_own = H.trace_windows(test_traces, L, S)
    Xe = H.featurize(ew, vocab, V, B=256, B_tri=256, use_trigrams=use_tri, normalize=norm)
    
    t0 = time.perf_counter()
    m = IsolationForest(n_estimators=100, random_state=seed, n_jobs=-1).fit(Xtr)
    fit_s = time.perf_counter() - t0
    
    cs = H.trace_scores(-m.decision_function(Xc), c_own, len(calib))
    es = -m.decision_function(Xe)
    ts = H.trace_scores(es, e_own, len(test_traces))
    
    auc = roc_auc_score(y, ts)
    
    # 99th pct threshold
    thr99 = np.percentile(cs, 99.0)
    pred99 = (ts > thr99).astype(int)
    f1_99 = f1_score(y, pred99)
    p_99 = precision_score(y, pred99, zero_division=0)
    r_99 = recall_score(y, pred99)
    tn, fp, fn, tp = confusion_matrix(y, pred99, labels=[0, 1]).ravel()
    far_99 = fp / max(fp + tn, 1)

    # 95th pct threshold
    thr95 = np.percentile(cs, 95.0)
    pred95 = (ts > thr95).astype(int)
    f1_95 = f1_score(y, pred95)
    p_95 = precision_score(y, pred95, zero_division=0)
    r_95 = recall_score(y, pred95)
    tn95, fp95, fn95, tp95 = confusion_matrix(y, pred95, labels=[0, 1]).ravel()
    far_95 = fp95 / max(fp95 + tn95, 1)

    return {
        "D": Xtr.shape[1],
        "auc": auc,
        "f1_99": f1_99, "p_99": p_99, "r_99": r_99, "far_99": far_99,
        "f1_95": f1_95, "p_95": p_95, "r_95": r_95, "far_95": far_95,
        "fit_s": fit_s,
        "n_windows": len(Xtr)
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    
    d = H.find_dirs(args.data)
    train_t, val_t, att_t = H.read_traces(d["train"]), H.read_traces(d["val"]), H.read_traces(d["attack"])
    print(f"Loaded traces: train={len(train_t)}, val={len(val_t)}, att={len(att_t)}")
    
    configs = [
        ("Bigrams (L1-norm) [Baseline]", False, True),
        ("Bigrams (Raw counts)", False, False),
        ("Bigrams+Trigrams (L1-norm)", True, True),
        ("Bigrams+Trigrams (Raw counts)", True, False),
    ]
    
    print("\n" + "="*95)
    print(f"{'Config':32s} | {'L':>3s} | {'D':>4s} | {'AUC':>6s} | {'P@99':>6s} | {'R@99':>6s} | {'F1@99':>6s} | {'FAR@99':>6s} | {'F1@95':>6s}")
    print("="*95)
    
    for label, use_tri, norm in configs:
        for L in (20, 50, 100):
            S = max(L // 2, 1)
            res = eval_config(train_t, val_t, att_t, L, S, use_tri, norm, seed=0)
            print(f"{label:32s} | {L:3d} | {res['D']:4d} | {res['auc']:6.3f} | {res['p_99']:6.3f} | {res['r_99']:6.3f} | {res['f1_99']:6.3f} | {res['far_99']:6.3f} | {res['f1_95']:6.3f}")

if __name__ == "__main__":
    main()
