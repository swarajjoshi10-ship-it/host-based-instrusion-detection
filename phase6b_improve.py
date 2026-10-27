"""
Phase 6b - Improvement experiments for the LSTM next-call detector on ADFA-LD.

PROTOCOL (never broken by this script)
- The 746 attack traces are split into a TUNING half and a TEST half (fixed split seed).
  Normal validation traces are split the same way (equal sizes), plus a CALIBRATION set (the rest).
- Experiments A-E are scored on the TUNING half only. The TEST half is used only by --exp FINAL.
- Training uses normal traces only. The vocabulary is built from training traces only.

Run from the folder that contains hids_m1_m2.py (it reuses find_dirs / read_traces / trace_windows from it):
    python phase6b_improve.py --data "C:\\path\\to\\ADFA-LD" --exp A
    python phase6b_improve.py --data ... --exp B --hidden 128 --layers 1
    python phase6b_improve.py --data ... --exp C --hidden 128 --layers 1 --L 50
    python phase6b_improve.py --data ... --exp E --hidden 128 --layers 1 --L 50 --extra_train 2000
    python phase6b_improve.py --data ... --exp FINAL --hidden 128 --layers 1 --L 50 --extra_train 2000 --ens 3 --rule top10
Every run appends its real numbers to results/phase6b/experiments.csv (or final.csv).
"""
import argparse, csv, os, random, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import hids_m1_m2 as H

OUT = os.path.join("results", "phase6b")

# ------------------------------------------------------------------ data
def make_split(train_t, val_t, att_t, split_seed=0, extra_train=0):
    rng = random.Random(split_seed)
    ai = list(range(len(att_t))); rng.shuffle(ai); half = len(att_t) // 2
    att_tune = [att_t[i] for i in ai[:half]]; att_test = [att_t[i] for i in ai[half:2 * half]]
    vi = list(range(len(val_t))); rng.shuffle(vi)
    norm_tune = [val_t[i] for i in vi[:half]]; norm_test = [val_t[i] for i in vi[half:2 * half]]
    rest = [val_t[i] for i in vi[2 * half:]]
    extra, calib = rest[:extra_train], rest[extra_train:]
    return train_t + extra, calib, (norm_tune, att_tune), (norm_test, att_test)

def encode(traces, vocab, V, L, S):
    wins, owner = H.trace_windows(traces, L, S)
    PAD = V + 1
    arr = np.full((len(wins), L), PAD, dtype=np.int64)
    for r, w in enumerate(wins):
        arr[r, :len(w)] = [vocab.get(c, V) for c in w]
    return torch.from_numpy(arr), np.asarray(owner)

# ------------------------------------------------------------------ model
class LSTMNext(nn.Module):
    def __init__(self, n_tokens, emb=32, hid=64, layers=1):
        super().__init__()
        self.emb = nn.Embedding(n_tokens, emb, padding_idx=n_tokens - 1)
        self.lstm = nn.LSTM(emb, hid, layers, batch_first=True)
        self.out = nn.Linear(hid, n_tokens)
    def forward(self, x):
        h, _ = self.lstm(self.emb(x)); return self.out(h)

def window_nll(model, X, PAD, bs=2048):
    """mean negative log-likelihood of the next call inside each window (higher = more anomalous)"""
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = X[i:i + bs]; logits = model(xb)[:, :-1]; tgt = xb[:, 1:]
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none").view(tgt.shape)
            mask = (tgt != PAD).float()
            out.append(((ce * mask).sum(1) / mask.sum(1).clamp(min=1)).numpy())
    return np.concatenate(out)

def train_lstm(Xtr_all, V, hid, layers, seed, max_epochs=30, patience=3, bs=256):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    PAD = V + 1
    perm = torch.randperm(len(Xtr_all), generator=g); nval = max(int(0.1 * len(perm)), 1)
    Xval, Xtr = Xtr_all[perm[:nval]], Xtr_all[perm[nval:]]
    model = LSTMNext(V + 2, hid=hid, layers=layers); opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    best, best_state, bad, ep_done = 1e9, None, 0, 0
    for ep in range(max_epochs):
        model.train(); idx = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(idx), bs):
            xb = Xtr[idx[i:i + bs]]; logits = model(xb)[:, :-1]; tgt = xb[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=PAD)
            opt.zero_grad(); loss.backward(); opt.step()
        vloss = float(window_nll(model, Xval, PAD).mean()); ep_done = ep + 1
        print(f"      epoch {ep_done:2d}  held-out normal NLL {vloss:.4f}")
        if vloss < best - 1e-4: best, bad, best_state = vloss, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state); return model, ep_done

# ------------------------------------------------------------------ scoring + metrics
def agg(scores, owner, rule):
    groups = np.split(np.asarray(scores), np.flatnonzero(np.diff(owner)) + 1); out = []
    for g in groups:
        if rule == "mean": out.append(g.mean())
        elif rule == "max": out.append(g.max())
        else: out.append(np.sort(g)[-max(int(np.ceil(0.1 * len(g))), 1):].mean())   # top10: mean of worst 10% windows
    return np.array(out)

def metrics(y, ts, thr):
    pred = (ts > thr).astype(int); tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1); f1 = 2 * p * r / max(p + r, 1e-12)
    return dict(precision=p, recall=r, f1=f1, far=fp / max(fp + tn, 1))

def best_f1_threshold(y, ts):
    best, thr = -1, None
    for t in np.unique(ts):
        f = metrics(y, ts, t)["f1"]
        if f > best: best, thr = f, t
    return thr

# ------------------------------------------------------------------ one configuration
def run_config(train_t, calib, tune, test, cfg, seed, want_test=False):
    t0 = time.perf_counter()
    vocab = {c: k for k, c in enumerate(sorted({c for t in train_t for c in t}))}; V = len(vocab); PAD = V + 1
    Xtr, _ = encode(train_t, vocab, V, cfg["L"], cfg["S"])
    Xcal, own_cal = encode(calib, vocab, V, cfg["L"], cfg["S"])
    sets = {"tune": tune}
    if want_test: sets["test"] = test
    enc = {}
    for name, (norm, att) in sets.items():
        X, own = encode(norm + att, vocab, V, cfg["L"], cfg["S"]); enc[name] = (X, own, np.array([0] * len(norm) + [1] * len(att)))
    cal_s = np.zeros(len(Xcal)); win = {k: np.zeros(len(v[0])) for k, v in enc.items()}; epochs = []
    for m in range(cfg["ens"]):
        print(f"    model {m + 1}/{cfg['ens']} (seed {seed + m})")
        model, ep = train_lstm(Xtr, V, cfg["hidden"], cfg["layers"], seed + m); epochs.append(ep)
        cal_s += window_nll(model, Xcal, PAD) / cfg["ens"]
        for k, (X, own, y) in enc.items(): win[k] += window_nll(model, X, PAD) / cfg["ens"]
    return dict(cal=cal_s, own_cal=own_cal, enc=enc, win=win, epochs=epochs, train_s=time.perf_counter() - t0,
                n_train_windows=len(Xtr))

def evaluate(res, rule, pct, split):
    X, own, y = res["enc"][split]
    ts = agg(res["win"][split], own, rule)
    cs = agg(res["cal"], res["own_cal"], rule)
    thr = np.percentile(cs, pct)
    m = metrics(y, ts, thr); m["auc"] = roc_auc_score(y, ts); return m, ts, y, cs

# ------------------------------------------------------------------ main
def save_rows(path, rows):
    os.makedirs(OUT, exist_ok=True); full = os.path.join(OUT, path); new = not os.path.exists(full)
    with open(full, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if new: w.writeheader()
        w.writerows(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--exp", required=True, choices=["A", "B", "C", "E", "FINAL"])
    ap.add_argument("--hidden", type=int, default=64); ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--L", type=int, default=20); ap.add_argument("--extra_train", type=int, default=0)
    ap.add_argument("--ens", type=int, default=1); ap.add_argument("--rule", default="mean", choices=["mean", "max", "top10"])
    ap.add_argument("--split_seed", type=int, default=0); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max_traces", type=int, default=0, help="debug only: limit traces per folder")
    a = ap.parse_args()
    d = H.find_dirs(a.data)
    train_t, val_t, att_t = H.read_traces(d["train"]), H.read_traces(d["val"]), H.read_traces(d["attack"])
    if a.max_traces: train_t, val_t, att_t = train_t[:a.max_traces], val_t[:a.max_traces * 3], att_t[:a.max_traces]
    print(f"[data] train {len(train_t)}, validation {len(val_t)}, attack {len(att_t)}  (expect 833 / 4372 / 746)")

    base = dict(hidden=a.hidden, layers=a.layers, L=a.L, S=max(a.L // 2, 1), extra=a.extra_train, ens=a.ens)
    if a.exp == "A":   grid = [dict(base, hidden=h, layers=l, extra=0, ens=1) for h in (64, 128) for l in (1, 2)]
    elif a.exp == "B": grid = [dict(base, L=L, S=max(L // 2, 1)) for L in (20, 50, 100)]
    elif a.exp == "C": grid = [dict(base, extra=e) for e in (0, 2000)]
    elif a.exp == "E": grid = [dict(base, ens=n) for n in (1, 3)]
    else:              grid = [base]

    if a.exp != "FINAL":
        rows = []
        for cfg in grid:
            print(f"\n=== exp {a.exp}: hidden={cfg['hidden']} layers={cfg['layers']} L={cfg['L']} extra_train={cfg['extra']} ens={cfg['ens']}")
            tr, calib, tune, test = make_split(train_t, val_t, att_t, a.split_seed, cfg["extra"])
            cfg2 = dict(cfg, **{"L": cfg["L"], "S": cfg["S"]})
            res = run_config(tr, calib, tune, test, cfg2, seed=0)
            print(f"  training windows {res['n_train_windows']}, epochs {res['epochs']}, time {res['train_s']:.0f}s  (scored on TUNING half only)")
            for rule in ("mean", "max", "top10"):
                for pct in (99, 95):
                    m, *_ = evaluate(res, rule, pct, "tune")
                    rows.append(dict(exp=a.exp, hidden=cfg["hidden"], layers=cfg["layers"], L=cfg["L"], extra_train=cfg["extra"], ens=cfg["ens"],
                                     rule=rule, pct=pct, split="tune", epochs=max(res["epochs"]), train_s=round(res["train_s"], 1), **{k: round(v, 4) for k, v in m.items()}))
            print(f"  {'rule':6s} {'AUC':>6s} | pct=99: {'P':>6s} {'R':>6s} {'F1':>6s} {'FAR':>6s} | pct=95: {'P':>6s} {'R':>6s} {'F1':>6s} {'FAR':>6s}")
            for rule in ("mean", "max", "top10"):
                m99, *_ = evaluate(res, rule, 99, "tune"); m95, *_ = evaluate(res, rule, 95, "tune")
                print(f"  {rule:6s} {m99['auc']:6.3f} | {'':8s} {m99['precision']:6.3f} {m99['recall']:6.3f} {m99['f1']:6.3f} {m99['far']:6.3f} | {'':8s} {m95['precision']:6.3f} {m95['recall']:6.3f} {m95['f1']:6.3f} {m95['far']:6.3f}")
        save_rows("experiments.csv", rows); print(f"\nSaved to {OUT}\\experiments.csv. Pick the best setting on these TUNING numbers, then run the next experiment.")
        return

    # ---------------- FINAL: fixed config chosen on tuning data, evaluated ONCE on the TEST half
    cfg = grid[0]; print(f"\n=== FINAL: {cfg}  rule={a.rule}  split_seed={a.split_seed}  (model seeds 0..{a.seeds - 1})")
    tr, calib, tune, test = make_split(train_t, val_t, att_t, a.split_seed, cfg["extra"])
    rows = []
    for s in range(a.seeds):
        print(f"\n--- model seed {s}")
        res = run_config(tr, calib, tune, test, cfg, seed=100 * s, want_test=True)
        _, ts_tune, y_tune, cs = evaluate(res, a.rule, 99, "tune"); thr_tuned = best_f1_threshold(y_tune, ts_tune)
        _, ts_test, y_test, _ = evaluate(res, a.rule, 99, "test"); auc = roc_auc_score(y_test, ts_test)
        for label, thr in (("calib99", np.percentile(cs, 99)), ("calib95", np.percentile(cs, 95)), ("tuned_on_tuning_half", thr_tuned)):
            m = metrics(y_test, ts_test, thr)
            rows.append(dict(seed=s, threshold=label, rule=a.rule, hidden=cfg["hidden"], layers=cfg["layers"], L=cfg["L"], extra_train=cfg["extra"],
                             ens=cfg["ens"], auc=round(auc, 4), epochs=max(res["epochs"]), train_s=round(res["train_s"], 1), **{k: round(v, 4) for k, v in m.items()}))
    save_rows("final.csv", rows)
    print(f"\n[FINAL on TEST half] mean +/- std over {a.seeds} model seeds (split fixed)")
    print(f"{'threshold':22s} {'Prec':>13s} {'Recall':>13s} {'F1':>13s} {'FAR':>13s} {'AUC':>13s}")
    for label in ("calib99", "calib95", "tuned_on_tuning_half"):
        R = [r for r in rows if r["threshold"] == label]; f = lambda k: f"{np.mean([r[k] for r in R]):.3f}+/-{np.std([r[k] for r in R]):.3f}"
        print(f"{label:22s} {f('precision'):>13s} {f('recall'):>13s} {f('f1'):>13s} {f('far'):>13s} {f('auc'):>13s}")
    print(f"\nSaved to {OUT}\\final.csv")

if __name__ == "__main__":
    main()
    