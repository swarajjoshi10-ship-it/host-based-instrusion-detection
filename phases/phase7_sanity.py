"""
Sanity check for the LID-DS result: can trivial features separate attacks from normal recordings?
Run after --step prepare:   python phase7_sanity.py Bruteforce_CWE-307
"""
import os, pickle, sys
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

name = sys.argv[1]
with open(os.path.join("results", "phase7", "lidds_" + name + ".pkl"), "rb") as fh:
    blob = pickle.load(fh)
data, vid = blob["data"], blob["vocab_ids"]
V = len(vid)
id2name = {v: k for k, v in vid.items()}

def feats(recs):
    n_calls, n_thr = [], []
    H = np.zeros((len(recs), V))
    for i, r in enumerate(recs):
        c = np.zeros(V)
        for t in r["threads"]:
            c += np.bincount(t, minlength=V)
        n_calls.append(c.sum())
        n_thr.append(len(r["threads"]))
        H[i] = c / max(c.sum(), 1)
    return np.array(n_calls), np.array(n_thr), H

train = [r for r in data["training"] if not r["exploit"]]
valid = [r for r in data["validation"] if not r["exploit"]]
test = data["test"]
y = np.array([1 if r["exploit"] else 0 for r in test])
ct, tt, Ht = feats(train)
cv, tv, Hv = feats(valid)
ce, te, He = feats(test)

print("[recording sizes]  mean calls per recording / mean threads per recording")
print("  training normal   : " + str(round(ct.mean())) + " / " + str(round(tt.mean(), 1)))
print("  validation normal : " + str(round(cv.mean())) + " / " + str(round(tv.mean(), 1)))
print("  test normal       : " + str(round(ce[y == 0].mean())) + " / " + str(round(te[y == 0].mean(), 1)))
print("  test attack       : " + str(round(ce[y == 1].mean())) + " / " + str(round(te[y == 1].mean(), 1)))

def sep(a):
    return max(a, 1 - a)
print("\n[can trivial features alone separate attacks? 0.5 = no, 1.0 = perfectly]")
print("  recording length (number of calls) : " + str(round(sep(roc_auc_score(y, ce)), 3)))
print("  number of threads                  : " + str(round(sep(roc_auc_score(y, te)), 3)))
m = IsolationForest(n_estimators=200, random_state=0).fit(Ht)
print("  Isolation Forest on call-frequency histogram (no order information): " + str(round(roc_auc_score(y, -m.decision_function(He)), 3)))

print("\n[which system calls differ most: share of all calls, normal vs attack test recordings]")
fn, fa = He[y == 0].mean(0), He[y == 1].mean(0)
for i in np.argsort(-np.abs(fa - fn))[:8]:
    print("  " + id2name[i].ljust(14) + "normal " + str(round(100 * fn[i], 2)) + "%   attack " + str(round(100 * fa[i], 2)) + "%")
print("\nHow to read it: if length or thread count already gives about 1.0, the scenario is easy for a trivial reason.")
print("If only the call-share differences separate them, the attack really changes behaviour, which is a legitimate signal.")