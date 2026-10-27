"""
Phase 6c - better trace-scoring rules and an optional small combiner for the final LSTM (ADFA-LD).

PROTOCOL
- One LSTM is trained on normal traces only (hidden 64, 1 layer, window 20, plus --extra_train normal traces).
- Every choice (which rule, the combiner) is made on the TUNING half only.
- The TEST half is then evaluated once for the chosen method. NOTE: the test half was already used once in
  Phase 6b FINAL, so this is a second look. Report both results and say so.
- The combiner uses labelled attacks from the tuning half, so it is no longer purely normal-only. Say so in the report.
- Thresholds always come from normal CALIBRATION traces (99th / 95th / 90th percentile of normal scores).

Run (after phase6b_improve.py is in the folder):  python phase6c_rules.py --data . --extra_train 3000
"""
import argparse, csv, os, random, re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
import phase6b_improve as P
import hids_m1_m2 as H

OUT = os.path.join("results", "phase6c")
FEATS = ["mean", "max", "top10", "p90", "top5", "frac_over", "loglen"]
RULES = FEATS[:6]

def attack_types(folder):
    types = []
    for dp, _, files in os.walk(folder):
        for f in sorted(files):
            if not f.lower().endswith(".txt"):
                continue
            with open(os.path.join(dp, f), "r", errors="ignore") as fh:
                toks = fh.read().split()
            if any(t.isdigit() for t in toks):
                types.append(re.sub(r"_\d+$", "", os.path.basename(dp)))
    return types

def trace_features(ws, owner, n, q):
    idx = np.flatnonzero(np.diff(owner)) + 1
    starts = np.r_[0, idx]
    F = np.zeros((n, len(FEATS)))
    for g, o in zip(np.split(np.asarray(ws), idx), owner[starts]):
        s = np.sort(g)
        k10 = max(int(np.ceil(0.10 * len(g))), 1)
        k5 = max(int(np.ceil(0.05 * len(g))), 1)
        F[o] = [g.mean(), g.max(), s[-k10:].mean(), np.percentile(g, 90), s[-k5:].mean(), (g > q).mean(), np.log(len(g))]
    return F

def report(label, y, score, calib_scores, rows, extra, pred_store=None):
    auc = roc_auc_score(y, score)
    out = []
    for pct in (99, 95, 90):
        thr = np.percentile(calib_scores, pct)
        m = P.metrics(y, score, thr)
        rows.append(dict(extra_train=extra, method=label, pct=pct, auc=round(auc, 4), **{k: round(v, 4) for k, v in m.items()}))
        out.append((pct, m))
        if pred_store is not None and pct == 95:
            pred_store[label] = (score > thr).astype(int)
    print("  " + label.ljust(30) + "AUC " + str(round(auc, 3)) + " | " + " | ".join(
        "pct" + str(p) + ": P " + str(round(m["precision"], 3)) + " R " + str(round(m["recall"], 3)) + " F1 " + str(round(m["f1"], 3)) + " FAR " + str(round(m["far"], 3)) for p, m in out))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--extra_train", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    d = H.find_dirs(a.data)
    train_t, val_t, att_t = H.read_traces(d["train"]), H.read_traces(d["val"]), H.read_traces(d["attack"])
    types = attack_types(d["attack"])
    print("[data] train " + str(len(train_t)) + ", validation " + str(len(val_t)) + ", attack " + str(len(att_t)) + "  (expect 833 / 4372 / 746)")
    tr, calib, tune, test = P.make_split(train_t, val_t, att_t, a.split_seed, a.extra_train)
    print("[split] training traces " + str(len(tr)) + " | calibration (normal) " + str(len(calib)) + " | tuning half " + str(len(tune[0])) + "+" + str(len(tune[1]))
          + " | test half " + str(len(test[0])) + "+" + str(len(test[1])))
    cfg = dict(hidden=64, layers=1, L=20, S=10, ens=1)
    res = P.run_config(tr, calib, tune, test, cfg, seed=a.seed, want_test=True)
    q = float(np.percentile(res["cal"], 99))
    Xt, ot, y_tune = res["enc"]["tune"]
    Xf, of, y_test = res["enc"]["test"]
    Fc = trace_features(res["cal"], res["own_cal"], len(calib), q)
    Ft = trace_features(res["win"]["tune"], ot, len(y_tune), q)
    Ff = trace_features(res["win"]["test"], of, len(y_test), q)

    print("\n[1] Trace-scoring rules ranked on the TUNING half (AUC)")
    tune_auc = {r: roc_auc_score(y_tune, Ft[:, FEATS.index(r)]) for r in RULES}
    for r in sorted(tune_auc, key=tune_auc.get, reverse=True):
        print("  " + r.ljust(10) + str(round(tune_auc[r], 3)))
    best_rule = max(tune_auc, key=tune_auc.get)
    j = FEATS.index(best_rule)
    print("  -> best rule on the tuning half: " + best_rule)

    rows = []
    preds = {}
    print("\n[2] TEST half (thresholds from normal calibration traces; pct99 = about 1% false alarms, pct90 = about 10%)")
    report("rule: mean (earlier final)", y_test, Ff[:, FEATS.index("mean")], Fc[:, FEATS.index("mean")], rows, a.extra_train, preds)
    report("rule: " + best_rule + " (chosen)", y_test, Ff[:, j], Fc[:, j], rows, a.extra_train, preds)

    print("\n[3] Combiner: logistic regression on all trace features, trained on the TUNING half (uses labelled attacks)")
    sc = StandardScaler().fit(Ft)
    lr = LogisticRegression(max_iter=2000, C=1.0)
    cv = cross_val_predict(LogisticRegression(max_iter=2000, C=1.0), sc.transform(Ft), y_tune, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="decision_function")
    print("  5-fold cross-validated AUC inside the tuning half: " + str(round(roc_auc_score(y_tune, cv), 3)))
    lr.fit(sc.transform(Ft), y_tune)
    report("combiner (LR on features)", y_test, lr.decision_function(sc.transform(Ff)), lr.decision_function(sc.transform(Fc)), rows, a.extra_train, preds)
    print("  feature weights: " + ", ".join(FEATS[i] + " " + str(round(float(lr.coef_[0][i]), 2)) for i in range(len(FEATS))))

    half = len(att_t) // 2
    ai = list(range(len(att_t)))
    random.Random(a.split_seed).shuffle(ai)
    test_att_idx = ai[half:2 * half]
    if len(types) == len(att_t):
        print("\n[4] Recall per attack type on the TEST half (threshold at the 95th percentile of normal scores)")
        n_norm = len(test[0])
        for label, pred in preds.items():
            print("  " + label)
            for t in sorted(set(types)):
                pos = [n_norm + k for k, i in enumerate(test_att_idx) if types[i] == t]
                if pos:
                    print("     " + t.ljust(18) + "recall " + str(round(float(pred[pos].mean()), 3)) + "  (" + str(len(pos)) + " traces)")
    else:
        print("\n[4] skipped per-type recall: found " + str(len(types)) + " attack types for " + str(len(att_t)) + " traces")

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "rules.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)
    print("\nSaved " + path + ". Reminder: the test half was used before, so report this as a second look.")

if __name__ == "__main__":
    main()
